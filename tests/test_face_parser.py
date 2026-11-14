"""Face-shaped paste-back mask: model trimming, background worker, paste-back."""
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

import _vendor.modules.globals as g
import _vendor.modules.processors.frame.face_swapper as fs
from _vendor.modules import live
from _vendor.modules.processors.frame import face_parser as fp


# ─── model trimming ───────────────────────────────────────────────────────


def _toy_parser(path):
    """A stand-in BiSeNet: (N,3,512,512) -> three (N,19,512,512) outputs."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((19, 3, 1, 1)).astype(np.float32), "w")
    w_aux = numpy_helper.from_array(rng.standard_normal((19, 3, 1, 1)).astype(np.float32), "w_aux")
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["output"], kernel_shape=[1, 1]),
        helper.make_node("Conv", ["input", "w_aux"], ["aux0"], kernel_shape=[1, 1]),
        helper.make_node("Relu", ["aux0"], ["aux1"]),
    ]
    dyn = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch_size", 19, 512, 512])
    graph = helper.make_graph(
        nodes, "toy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", 3, 512, 512])],
        [dyn("output"), dyn("aux0"), dyn("aux1")], [w, w_aux])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.ir_version = 8  # what the real BiSeNet export uses; newer IR versions are not readable by ONNX Runtime
    onnx.save(model, path)
    return rng.standard_normal  # unused; keeps the signature obvious


def test_build_mask_model_sums_face_class_probabilities(tmp_path):
    import onnx
    import onnxruntime as ort

    src, dst = str(tmp_path / "toy.onnx"), str(tmp_path / "toy_facemask.onnx")
    _toy_parser(src)
    fp.build_mask_model(src, dst)

    model = onnx.load(dst)
    assert [o.name for o in model.graph.output] == ["face_mask"]
    assert {i.name for i in model.graph.initializer} >= {"w", "face_sum_w"}
    assert "w_aux" not in {i.name for i in model.graph.initializer}      # aux head pruned

    x = np.random.default_rng(1).standard_normal((1, 3, 512, 512)).astype(np.float32)
    got = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    logits = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    want = (e / e.sum(1, keepdims=True))[:, list(fp.FACE_CLASSES)].sum(1, keepdims=True)
    assert got.shape == (1, 1, 512, 512)
    np.testing.assert_allclose(got, want, atol=1e-5)


# ─── background worker ────────────────────────────────────────────────────


class _FakeParser:
    def __init__(self, delay=0.0):
        self.calls, self.threads, self.delay = 0, [], delay

    def __call__(self, aimg):
        self.calls += 1
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return np.full(aimg.shape[:2], self.calls, dtype=np.uint8)


def _crop(size=256):
    return np.zeros((size, size, 3), dtype=np.uint8)


def _m(tx=0.0, scale=1.0):
    return np.array([[scale, 0.0, tx], [0.0, scale, 0.0]], dtype=np.float64)


def _wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.01)
    return cond()


def test_first_mask_is_parsed_synchronously_then_in_the_background():
    parser = _FakeParser()
    worker = fp.FaceMaskWorker(parser)
    first = worker.mask(_crop(), _m())
    assert parser.calls == 1 and first[0, 0] == 1
    assert parser.threads[0] != "DLC.FACE-PARSER"               # caller's thread
    second = worker.mask(_crop(), _m())
    assert second[0, 0] == 1                                    # newest finished mask, no wait
    assert _wait_for(lambda: parser.calls == 2)
    assert parser.threads[1] == "DLC.FACE-PARSER"               # worker thread
    assert _wait_for(lambda: worker.mask(_crop(), _m())[0, 0] >= 2)


def test_mask_is_resized_to_the_crop():
    worker = fp.FaceMaskWorker(_FakeParser())
    worker.mask(_crop(256), _m())
    assert worker.mask(_crop(512), _m()).shape == (512, 512)    # pixel boost crop


def test_stale_mask_is_reparsed_synchronously():
    parser = _FakeParser()
    worker = fp.FaceMaskWorker(parser)
    worker.mask(_crop(), _m())
    with patch.object(fp.FaceMaskWorker, "STALE_S", 0.0):
        time.sleep(0.01)
        out = worker.mask(_crop(), _m())
    assert out[0, 0] == 2 and parser.threads[1] != "DLC.FACE-PARSER"


@pytest.mark.parametrize("m2, jumped", [
    (_m(tx=10.0), False),          # small move: reuse
    (_m(tx=120.0), True),          # centre moved by about half a face
    (_m(scale=1.6), True),         # face size changed a lot
])
def test_alignment_jump_forces_a_fresh_mask(m2, jumped):
    parser = _FakeParser()
    worker = fp.FaceMaskWorker(parser)
    worker.mask(_crop(), _m())
    out = worker.mask(_crop(), m2)
    fresh = bool(out[0, 0] == 2) and parser.threads[1] != "DLC.FACE-PARSER"
    assert fresh is jumped


# ─── paste-back ───────────────────────────────────────────────────────────


def test_border_template_hides_the_crop_edges_only():
    border = fs._get_soft_alpha(256, "border")
    assert border[128, 128] == 255 and border[0, 128] == 0 and border[128, 255] == 0
    assert border[0, 0] == 0 and (border[30:-30, 30:-30] == 255).all()
    ellipse = fs._get_soft_alpha(256)
    assert ellipse[0, 0] == 0 and ellipse[128, 128] == 255 and ellipse[128, 8] < 128   # oval cuts the sides
    assert fs._get_soft_alpha(256, "border") is border          # cached


class _ConstWorker:
    def __init__(self, value):
        self.value, self.calls = value, []

    def mask(self, aimg, M=None):
        self.calls.append((aimg.shape, M))
        return np.full(aimg.shape[:2], self.value, dtype=np.uint8)


@pytest.fixture
def paste_globals():
    keys = ("face_parsing_mask", "opacity", "mouth_mask", "poisson_blend")
    saved = {k: getattr(g, k) for k in keys}
    g.opacity, g.mouth_mask, g.poisson_blend = 1.0, False, False
    yield
    for k, v in saved.items():
        setattr(g, k, v)


def _paste(value):
    frame = np.full((300, 300, 3), 50, dtype=np.uint8)
    fake = np.full((256, 256, 4), 200, dtype=np.uint8)           # swap output...
    fake[:, :, 3] = 255                                          # ...and a full HyperSwap mask (alpha)
    M = np.array([[1.0, 0.0, -20.0], [0.0, 1.0, -20.0]])       # face crop at (20, 20)
    worker = _ConstWorker(value)
    with patch.object(fp, "get_worker", lambda: worker):
        out = fs.paste_swapped_face(frame, fake, M, target_face=None)
    return out, worker


def test_face_mask_limits_the_paste(paste_globals):
    g.face_parsing_mask = True
    out, worker = _paste(0)                                    # parser: no face anywhere
    assert worker.calls and worker.calls[0][0] == (256, 256, 3)
    assert out.max() == 50                                     # nothing pasted
    out, _ = _paste(255)                                       # parser: all face
    assert out[148, 148, 0] == 200                             # pasted in the middle
    assert out[148, 32, 0] == 200                              # ...and where the oval has faded out


def test_face_mask_off_uses_the_oval(paste_globals):
    g.face_parsing_mask = False
    out, worker = _paste(0)
    assert not worker.calls
    assert out[148, 148, 0] == 200 and out[148, 32, 0] < 150   # oval fades towards the sides


def test_configure_maps_face_mask(paste_globals):
    with patch.object(live, "get_frame_processors_modules", lambda *_: []):
        live.configure(live.LiveConfig(face_mask=False, execution_providers=["cpu"]))
        assert g.face_parsing_mask is False
        live.configure(live.LiveConfig(execution_providers=["cpu"]))
        assert g.face_parsing_mask is True                     # on by default
