"""Face-shaped paste-back mask from a face-parsing model.

The default paste-back blends the swapped face through a fixed feathered
ellipse, which cuts into the forehead and jaw of a long face and shows past
the cheeks of a round one. This module parses the *real* face in the aligned
crop (BiSeNet, CelebAMask-HQ labels) and returns a soft mask of skin, brows,
eyes, nose, mouth and lips, so the swap follows the actual face outline and
leaves hair, ears, hands and hats in front of the face untouched.

The parser costs ~45 ms per face on Apple Silicon (GPU) and a few ms on an
NVIDIA GPU, so in live use it runs on a background thread (FaceMaskWorker):
each frame pastes with the newest mask, which lives in aligned-face space and
therefore already follows head movement; only expression changes lag a few
frames.
"""
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

import _vendor.modules.globals as g
from _vendor.modules.status import update_status

NAME = "DLC.FACE-PARSER"
MODEL = "bisenet_resnet_18.onnx"
INPUT_SIZE = 512
# CelebAMask-HQ labels kept as "face": skin, brows, eyes, nose, mouth, lips.
# Left out: background, glasses (HyperSwap's own mask handles occluders),
# ears, earrings, neck, necklace, clothes, hair, hat.
FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)
# Face probability -> mask: the parser is less confident on plain skin than on
# features, so stretch [LOW, HIGH] to [0, 1].
LOW, HIGH = 0.2, 0.6
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_mask_model(src: str, dst: str) -> str:
    """Write a single-output version of the parser: the auxiliary heads are
    pruned and a softmax + 1x1 convolution sums the face classes, so a run
    returns one (1, 1, 512, 512) probability map instead of three 19-class
    tensors. The input shape is fixed for CoreML."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(src)
    graph = m.graph
    logits = graph.output[0].name
    del graph.output[:]
    weights = np.zeros((1, 19, 1, 1), np.float32)
    weights[0, list(FACE_CLASSES)] = 1.0
    graph.initializer.append(numpy_helper.from_array(weights, "face_sum_w"))
    graph.node.extend([
        helper.make_node("Softmax", [logits], ["probs"], axis=1),
        helper.make_node("Conv", ["probs", "face_sum_w"], ["face_mask"], kernel_shape=[1, 1],
                         pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1], group=1),
    ])
    graph.output.append(helper.make_tensor_value_info("face_mask", TensorProto.FLOAT, [1, 1, INPUT_SIZE, INPUT_SIZE]))
    dims = graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_value, dims[2].dim_value, dims[3].dim_value = 1, INPUT_SIZE, INPUT_SIZE
    # Drop the nodes and weights only the removed heads used.
    needed, keep = {"face_mask"}, []
    for node in reversed(list(graph.node)):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    del graph.node[:]
    graph.node.extend(reversed(keep))
    used = {i for node in graph.node for i in node.input}
    inits = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)
    del graph.value_info[:]
    onnx.checker.check_model(m)
    tmp = dst + ".tmp"
    onnx.save(m, tmp)
    os.replace(tmp, dst)
    return dst


def _providers(execution_providers) -> list:
    out = []
    for p in execution_providers:
        name = p[0] if isinstance(p, tuple) else p
        if name == "CoreMLExecutionProvider":
            # GPU only: the Neural Engine is busy with the swap model.
            out.append(("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU",
                "SpecializationStrategy": "FastPrediction", "AllowLowPrecisionAccumulationOnGPU": 1,
            }))
        else:
            out.append(name)
    return out or ["CPUExecutionProvider"]


class FaceParser:
    """Soft face mask (uint8, 0..255) for an aligned face crop."""

    def __init__(self, execution_providers):
        import onnxruntime
        from _vendor.modules.model_downloader import ensure_model
        from _vendor.modules.onnx_optimize import IS_APPLE_SILICON, optimize_for_coreml

        src = ensure_model(MODEL)
        if src is None:
            raise RuntimeError(f"could not obtain {MODEL}")
        path = os.path.join(os.path.dirname(src), "bisenet_resnet_18_facemask.onnx")
        if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(src):
            build_mask_model(src, path)
        if IS_APPLE_SILICON:
            path = optimize_for_coreml(path, input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE))
        self.session = onnxruntime.InferenceSession(path, providers=_providers(execution_providers))

    def __call__(self, aimg: np.ndarray) -> np.ndarray:
        size = aimg.shape[0]
        x = cv2.resize(aimg, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR) if size != INPUT_SIZE else aimg
        x = (x[:, :, ::-1].astype(np.float32) * (1.0 / 255.0) - _MEAN) / _STD
        prob = self.session.run(None, {"input": np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0, 0]
        mask = np.clip((prob - LOW) * (1.0 / (HIGH - LOW)), 0.0, 1.0)
        if size != INPUT_SIZE:
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA if size < INPUT_SIZE else cv2.INTER_LINEAR)
        return (mask * 255.0 + 0.5).astype(np.uint8)


class FaceMaskWorker:
    """Runs the parser on a background thread, always on the newest crop.

    ``mask(aimg, M)`` hands ``aimg`` (aligned with the 2x3 matrix ``M``) to
    the worker and returns the newest finished mask at ``aimg``'s size. The
    crop is parsed synchronously instead when no usable mask exists: on the
    first call (e.g. a still image), when the mask is older than STALE_S
    (the face was gone), or when the alignment jumped since the mask's crop
    (another face, or a cut) - a mask made for a different face would cut
    into this one.
    """

    STALE_S = 0.5
    MAX_SHIFT = 0.15   # crop centre movement, as a fraction of the face size
    MAX_SCALE = 1.25   # face size ratio

    def __init__(self, parser: FaceParser):
        self._parser = parser
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._mask_time = 0.0
        self._mask_m: Optional[np.ndarray] = None    # alignment of the crop the mask came from
        self._pending_m: Optional[np.ndarray] = None
        self._thread = threading.Thread(target=self._run, name="DLC.FACE-PARSER", daemon=True)
        self._thread.start()

    def mask(self, aimg: np.ndarray, M: Optional[np.ndarray] = None) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            mask, age, mask_m = self._mask, now - self._mask_time, self._mask_m
            self._pending, self._pending_m = aimg, M
        if mask is None or age > self.STALE_S or self._jumped(mask_m, M, aimg.shape[0]):
            mask = self._parser(aimg)
            with self._lock:
                self._mask, self._mask_time, self._mask_m = mask, now, M
        else:
            self._wake.set()
        size = aimg.shape[0]
        if mask.shape[0] != size:
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
        return mask

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                aimg, M, self._pending = self._pending, self._pending_m, None
            if aimg is None:
                continue
            try:
                mask = self._parser(aimg)
            except Exception as exc:  # keep the last mask; report once per failure burst
                update_status(f"Face parsing failed: {exc}", NAME)
                time.sleep(1.0)
                continue
            with self._lock:
                self._mask, self._mask_time, self._mask_m = mask, time.monotonic(), M

    @classmethod
    def _jumped(cls, old: Optional[np.ndarray], new: Optional[np.ndarray], size: int) -> bool:
        """Whether alignment ``new`` is too far from ``old`` to reuse its mask."""
        if old is None or new is None:
            return False
        a, b = cv2.invertAffineTransform(old), cv2.invertAffineTransform(new)
        centre = np.array([size / 2.0, size / 2.0, 1.0])
        scale_a = float(np.sqrt(abs(np.linalg.det(a[:, :2]))))
        scale_b = float(np.sqrt(abs(np.linalg.det(b[:, :2]))))
        shift = float(np.linalg.norm(a @ centre - b @ centre)) / (max(scale_a, 1e-6) * size)
        ratio = max(scale_a, scale_b) / max(min(scale_a, scale_b), 1e-6)
        return shift > cls.MAX_SHIFT or ratio > cls.MAX_SCALE


_WORKER: Optional[FaceMaskWorker] = None
_WORKER_KEY = None
_WORKER_LOCK = threading.Lock()


def get_worker() -> Optional[FaceMaskWorker]:
    """The shared mask worker for the current execution providers, or None
    if the parser cannot be loaded (the ellipse mask is used then)."""
    global _WORKER, _WORKER_KEY
    key = tuple(p[0] if isinstance(p, tuple) else p for p in g.execution_providers)
    with _WORKER_LOCK:
        if _WORKER_KEY != key:  # first use, or providers changed (a failure is remembered per key)
            try:
                _WORKER = FaceMaskWorker(FaceParser(g.execution_providers))
                _WORKER_KEY = key
                update_status("Face-shaped mask ready.", NAME)
            except Exception as exc:
                update_status(f"Face-shaped mask unavailable ({exc}); using the ellipse.", NAME)
                _WORKER, _WORKER_KEY = None, key
        return _WORKER
