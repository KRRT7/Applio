"""Headless live face swap: configuration, per-frame pipeline and live session.

This is the embeddable core of Deep-Live-Cam's live mode. It has no UI
dependency, so another application (CLI, Gradio, ...) can drive it:

    from _vendor.modules import live

    live.configure(live.LiveConfig(source_path="face.jpg", enhancer="gfpgan"))
    live.ensure_models()
    session = live.LiveSession(camera_index=1, sinks=[VirtualCameraSink()])
    session.start()
    ...  # session.latest_frame / session.fps
    session.stop()

Settings live in ``modules.globals`` (the processors read them from there), so
there is one active configuration per process. The pipeline re-reads them on
every frame, so calling configure() while a session runs applies immediately.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from _vendor import modules  # noqa: F401

import _vendor.modules.globals as _modules_globals  # noqa: F401  (loads the submodule; vendoring-safe form of `import modules.globals`)
from _vendor.modules import imread_unicode
from _vendor.modules.execution import (
    decode_execution_providers,
    suggest_default_execution_provider,
    suggest_execution_threads,
)
from _vendor.modules.face_analyser import (
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    reset_face_analyser,
)
from _vendor.modules.gpu_processing import gpu_flip
from _vendor.modules.processors.frame.core import get_frame_processors_modules
from _vendor.modules.status import update_status
from _vendor.modules.video_capture import VideoCapturer

Frame = np.ndarray
FrameSink = Callable[[Frame], None]

# How long the face may go undetected before the last async swap result is
# dropped. A short detector miss keeps it, so the swap resumes at once instead
# of showing the real face for a whole inference.
ASYNC_SWAP_FACE_LOSS_GRACE_S = 0.5

# Kept in step with modules.processors.frame.face_swapper.SWAP_MODELS (not
# imported here so configure() does not load the swapper module).
SWAP_MODELS = ("inswapper_128", "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256")

ENHANCERS = {
    "gfpgan": "face_enhancer",
    "gpen256": "face_enhancer_gpen256",
    "gpen512": "face_enhancer_gpen512",
}


# Source photos already reported as unreadable (cleared when the source set
# changes), so a bad path is reported once rather than once per pipeline.
_REPORTED_BAD_SOURCES: set = set()


# ─── configuration ───────────────────────────────────────────────────────


@dataclass
class LiveConfig:
    """Everything the live pipeline reads. Maps onto ``modules.globals``."""

    source_path: Optional[str] = None
    # More photos of the same person. Their face embeddings are averaged with
    # source_path's, which gives a steadier identity (in our tests 4 photos
    # raised identity similarity from 0.785 to 0.810 on held-out frames).
    source_paths: Sequence[str] = ()
    swap_model: str = "hyperswap_1c_256"    # see SWAP_MODELS; fastest + best on Apple Silicon in our benchmark
    enhancer: Optional[str] = None          # None | "gfpgan" | "gpen256" | "gpen512"
    execution_providers: Sequence[str] = field(
        default_factory=lambda: [suggest_default_execution_provider()]
    )                                       # short names: "coreml", "cuda", "cpu", ...
    det_size: int = modules.globals.DEFAULT_DET_SIZE  # 160 | 320 | 640
    many_faces: bool = False
    mouth_mask: float = 0.0                 # 0 = off, 100 = original mouth down to chin
    color_correction: bool = False
    poisson_blend: bool = False
    opacity: float = 1.0
    sharpness: float = 0.0
    mirror: bool = False
    async_swap: bool = False
    show_fps: bool = False


def configure(config: LiveConfig) -> None:
    """Apply a LiveConfig to the process-wide settings the processors read."""
    if config.enhancer is not None and config.enhancer not in ENHANCERS:
        raise ValueError(f"unknown enhancer {config.enhancer!r}; use one of {sorted(ENHANCERS)}")
    if config.swap_model not in SWAP_MODELS:
        raise ValueError(f"unknown swap model {config.swap_model!r}; use one of {SWAP_MODELS}")
    if config.det_size not in (160, 320, 640):
        raise ValueError("det_size must be 160, 320 or 640")

    g = modules.globals
    providers = decode_execution_providers(list(config.execution_providers))
    if not providers:
        providers = decode_execution_providers(["cpu"])
    if providers != g.execution_providers:
        g.execution_providers = providers
        g.execution_threads = suggest_execution_threads(providers)
        reset_face_analyser()
    elif g.execution_threads is None:
        g.execution_threads = suggest_execution_threads(providers)

    if config.det_size != g.det_size:
        g.det_size = config.det_size
        reset_face_analyser()

    paths = [p for p in [config.source_path, *config.source_paths] if p]
    new_paths = tuple(dict.fromkeys(paths))  # de-duplicated, order kept
    if new_paths != tuple(g.source_paths):
        _REPORTED_BAD_SOURCES.clear()
    g.source_paths = new_paths
    g.source_path = g.source_paths[0] if g.source_paths else None
    g.swap_model = config.swap_model
    g.target_path = None
    g.map_faces = False
    g.many_faces = config.many_faces
    g.mouth_mask_size = float(config.mouth_mask)
    g.mouth_mask = config.mouth_mask > 0
    g.show_mouth_mask_box = False
    g.color_correction = config.color_correction
    g.poisson_blend = config.poisson_blend
    g.opacity = config.opacity
    g.sharpness = config.sharpness
    g.live_mirror = config.mirror
    g.live_async_swap = config.async_swap
    g.show_fps = config.show_fps

    if "face_swapper" not in g.frame_processors:
        g.frame_processors.insert(0, "face_swapper")
    for name, module in ENHANCERS.items():
        g.fp_ui[module] = config.enhancer == name
    # Load/unload enhancer modules to match fp_ui. A running FramePipeline
    # holds this same module list, so an enhancer switched on mid-session
    # takes effect on its next frame.
    get_frame_processors_modules(g.frame_processors)


def ensure_models() -> bool:
    """Download any missing models for the configured processors."""
    names = ["face_swapper"] + [m for m, on in modules.globals.fp_ui.items() if on]
    try:
        processors = get_frame_processors_modules(names)
    except SystemExit:
        return False
    return all(p.pre_check() for p in processors)


def preload_models() -> None:
    """Load the face analyser and every configured model now.

    Models otherwise load on the first frame, which on CoreML means several
    seconds of compilation while the live preview crawls.
    """
    from _vendor.modules.face_analyser import get_face_analyser

    get_face_analyser()
    for processor in get_frame_processors_modules(modules.globals.frame_processors):
        loader = (getattr(processor, "get_face_swapper", None)
                  or getattr(processor, "get_face_enhancer", None)
                  or getattr(processor, "get_enhancer", None))
        if loader is not None:
            loader()


def swap_image(target: Frame) -> Frame:
    """Swap the configured source face into a single BGR image."""
    pipeline = FramePipeline(camera_fps=1.0)  # detect on every frame
    try:
        return pipeline.process(target.copy())
    finally:
        pipeline.close()


# ─── per-frame pipeline ──────────────────────────────────────────────────


class FramePipeline:
    """Runs detect -> swap -> enhance on successive frames of one stream.

    Holds the per-stream state: the loaded source face, cached detections
    (detection runs every few frames), the async swapper and an FPS counter.
    Not thread-safe; use one per stream.
    """

    def __init__(self, camera_fps: float = 30.0):
        self._frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        self._async_swapper = None
        self._face_lost_at = None
        self._source_image = None
        self._last_source_path = None
        self._reported_source_error = None
        self._last_source_paths = None
        self._prev_time = time.time()
        self._frame_count = 0
        self.fps = 0.0
        self._det_count = 0
        self._cached_target_face = None
        self._cached_many_faces = None
        self._det_interval = max(1, round(camera_fps * 0.08))

    def process(self, frame: Frame) -> Frame:
        temp_frame = frame
        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        if not modules.globals.map_faces:
            temp_frame = self._process_single_or_many(temp_frame)
        else:
            temp_frame = self._process_mapped(temp_frame)

        current_time = time.time()
        self._frame_count += 1
        if current_time - self._prev_time >= 0.5:
            self.fps = self._frame_count / (current_time - self._prev_time)
            self._frame_count = 0
            self._prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame, f"FPS: {self.fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
        return temp_frame

    def close(self) -> None:
        if self._async_swapper is not None:
            self._async_swapper.close()
            self._async_swapper = None

    # The two branches below are the former Qt live-preview worker loop,
    # unchanged apart from living on an object.

    def _load_source(self, paths: tuple):
        """Source face for ``paths``: one photo as-is, several averaged.

        Returns None when no photo yields a face. Unreadable photos are
        reported once per path and skipped.
        """
        faces = []
        for path in paths:
            frame = imread_unicode(path)
            if frame is None:
                if path not in _REPORTED_BAD_SOURCES:
                    _REPORTED_BAD_SOURCES.add(path)
                    update_status(f"Could not read source image: {path}")
                continue
            face = get_one_face(frame)
            if face is not None:
                faces.append(face)
        if len(paths) == 1 or len(faces) <= 1:
            return faces[0] if faces else None
        from insightface.app.common import Face

        mean = np.mean([f.normed_embedding for f in faces], axis=0)
        averaged = Face(dict(faces[0]))
        averaged.embedding = (mean / np.linalg.norm(mean)).astype(np.float32)
        return averaged

    def _process_single_or_many(self, temp_frame: Frame) -> Frame:
        paths = tuple(modules.globals.source_paths) or (
            (modules.globals.source_path,) if modules.globals.source_path else ()
        )
        if len(paths) > 1:
            self._last_source_path = None
            if paths != self._last_source_paths:
                self._source_image = self._load_source(paths)
                # Keep retrying while nothing loads (a photo may still be
                # being written); otherwise cache until the set changes.
                self._last_source_paths = paths if self._source_image is not None else None
        else:
            self._last_source_paths = None
            source_path = paths[0] if paths else None
            if not source_path:
                # Clearing the selection also invalidates cached and in-flight
                # swaps before processing another frame.
                self._source_image = None
                self._last_source_path = None
                self._reported_source_error = None
                if self._async_swapper is not None:
                    self._async_swapper.clear()
            elif source_path != self._last_source_path:
                source_frame = imread_unicode(source_path)
                if source_frame is None:
                    # Clear the cached path along with the image so a
                    # transiently unreadable source is retried and
                    # A -> bad B -> A reloads A. Reported once per path.
                    self._source_image = None
                    self._last_source_path = None
                    if self._reported_source_error != source_path:
                        self._reported_source_error = source_path
                        update_status(f"Could not read source image: {source_path}")
                else:
                    self._last_source_path = source_path
                    self._reported_source_error = None
                    self._source_image = get_one_face(source_frame)

        self._det_count += 1
        if self._det_count % self._det_interval == 0:
            if modules.globals.many_faces:
                self._cached_target_face = None
                self._cached_many_faces = detect_many_faces_fast(temp_frame)
            else:
                self._cached_target_face = detect_one_face_fast(temp_frame)
                self._cached_many_faces = None

        cached_faces = None
        if self._cached_many_faces:
            cached_faces = self._cached_many_faces
        elif self._cached_target_face is not None:
            cached_faces = [self._cached_target_face]

        # Fast detection skips the 2d106 landmark model, but the mouth mask
        # needs it. Attach landmarks on demand.
        if modules.globals.mouth_mask and cached_faces:
            ensure_landmarks(temp_frame, cached_faces)

        source_image = self._source_image
        for fp in self._frame_processors:
            if fp.NAME == "DLC.FACE-ENHANCER":
                if modules.globals.fp_ui["face_enhancer"]:
                    temp_frame = fp.process_frame(None, temp_frame, detected_faces=cached_faces)
            elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                    temp_frame = fp.process_frame(None, temp_frame, detected_faces=cached_faces)
            elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                    temp_frame = fp.process_frame(None, temp_frame, detected_faces=cached_faces)
            elif fp.NAME == "DLC.FACE-SWAPPER":
                swapped_bboxes = []
                if modules.globals.many_faces and self._cached_many_faces:
                    result = temp_frame.copy()
                    for t_face in self._cached_many_faces:
                        result = fp.swap_face(source_image, t_face, result)
                        if hasattr(t_face, "bbox") and t_face.bbox is not None:
                            swapped_bboxes.append(t_face.bbox.astype(int))
                    temp_frame = result
                elif self._cached_target_face is not None:
                    if modules.globals.live_async_swap:
                        if self._async_swapper is None:
                            self._async_swapper = fp.AsyncLiveSwapper()
                        temp_frame = self._async_swapper.swap(
                            source_image, self._cached_target_face, temp_frame
                        )
                    else:
                        temp_frame = fp.swap_face(source_image, self._cached_target_face, temp_frame)
                    if hasattr(self._cached_target_face, "bbox") and self._cached_target_face.bbox is not None:
                        swapped_bboxes.append(self._cached_target_face.bbox.astype(int))
                temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
            else:
                temp_frame = fp.process_frame(source_image, temp_frame)

        # Once the face has been gone for a while, or async swap is switched
        # off, the last async result is stale.
        if self._cached_target_face is not None:
            self._face_lost_at = None
        elif self._face_lost_at is None:
            self._face_lost_at = time.monotonic()
        if self._async_swapper is not None and (
            not modules.globals.live_async_swap
            or (
                self._face_lost_at is not None
                and time.monotonic() - self._face_lost_at > ASYNC_SWAP_FACE_LOSS_GRACE_S
            )
        ):
            self._async_swapper.clear()
        return temp_frame

    def _process_mapped(self, temp_frame: Frame) -> Frame:
        modules.globals.target_path = None
        # Single-face state goes stale while map-faces runs: resume from a
        # fresh detection and a fresh async result.
        self._cached_target_face = None
        self._cached_many_faces = None
        if self._async_swapper is not None:
            self._async_swapper.clear()
        for fp in self._frame_processors:
            if fp.NAME == "DLC.FACE-ENHANCER":
                if modules.globals.fp_ui["face_enhancer"]:
                    temp_frame = fp.process_frame_v2(temp_frame)
            elif fp.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                fp_key = fp.NAME.split(".")[-1].lower().replace("-", "_")
                if modules.globals.fp_ui.get(fp_key, False):
                    temp_frame = fp.process_frame_v2(temp_frame)
            else:
                temp_frame = fp.process_frame_v2(temp_frame)
        return temp_frame


# ─── live session ────────────────────────────────────────────────────────


def _put_latest(q: queue.Queue, item) -> None:
    """Put without blocking, dropping the oldest item when full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class LiveSession:
    """Camera -> FramePipeline -> sinks, on two background threads.

    ``sinks`` are callables taking each processed BGR frame (for example a
    VirtualCameraSink). A sink that raises is dropped for the rest of the
    session. The newest processed frame is always available as
    ``latest_frame`` for previews.
    """

    def __init__(
        self,
        camera_index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        sinks: Sequence[FrameSink] = (),
        capture_factory: Callable[[int], VideoCapturer] = VideoCapturer,
    ):
        self.camera_index = camera_index
        self.requested = (width, height, fps)
        self.sinks: List[FrameSink] = list(sinks)
        self._capture_factory = capture_factory
        self._cap = None
        self._pipeline: Optional[FramePipeline] = None
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._capture_q: queue.Queue = queue.Queue(maxsize=2)
        self._latest_lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self.camera_size = (0, 0)
        self.camera_fps = 0.0
        self.error: Optional[str] = None

    # public API

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @property
    def fps(self) -> float:
        return self._pipeline.fps if self._pipeline is not None else 0.0

    @property
    def latest_frame(self) -> Optional[Frame]:
        with self._latest_lock:
            return self._latest

    def start(self) -> None:
        if self.running:
            raise RuntimeError("session already running")
        width, height, fps = self.requested
        self._stop.clear()
        self.error = None
        self._cap = self._capture_factory(self.camera_index)
        if not self._cap.start(width, height, fps):
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        self.camera_size = (self._cap.actual_width, self._cap.actual_height)
        self.camera_fps = self._cap.actual_fps or 30.0
        update_status(
            f"Camera {self.camera_index} running at "
            f"{self.camera_size[0]}x{self.camera_size[1]}@{self.camera_fps:.0f}fps",
            "DLC.LIVE",
        )
        self._pipeline = FramePipeline(self.camera_fps)
        self._threads = [
            threading.Thread(target=self._capture_loop, name="dlc-capture", daemon=True),
            threading.Thread(target=self._process_loop, name="dlc-process", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop both threads, wait for them, then release camera and sinks."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        if self.running:
            return  # timed out; caller may stop() again
        self._threads = []
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._pipeline is not None:
            self._pipeline.close()
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    def __enter__(self) -> "LiveSession":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # threads

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self.error = "camera stopped delivering frames"
                self._stop.set()
                break
            _put_latest(self._capture_q, frame)

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._capture_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                out = self._pipeline.process(frame)
            except Exception as exc:
                self.error = f"processing failed: {exc}"
                update_status(self.error, "DLC.LIVE")
                self._stop.set()
                break
            with self._latest_lock:
                self._latest = out
            for sink in list(self.sinks):
                try:
                    sink(out)
                except Exception as exc:
                    update_status(f"Dropping frame sink {sink!r}: {exc}", "DLC.LIVE")
                    self.sinks.remove(sink)


# ─── CLI entry point (``run.py --live``) ─────────────────────────────────


def run_cli(camera: Optional[int], width: int, height: int, virtual_camera: bool) -> int:
    """Run a live session until Ctrl+C, printing FPS once a second."""
    from _vendor.modules.cameras import list_cameras
    from _vendor.modules.virtual_camera import VirtualCameraSink

    if camera is None:
        cams = list_cameras()
        if not cams:
            update_status("No cameras found")
            return 1
        camera = cams[0][0]
        update_status(f"Using camera {camera}: {cams[0][1]}", "DLC.LIVE")
    if not ensure_models():
        return 1
    preload_models()
    sinks = [VirtualCameraSink()] if virtual_camera else []
    session = LiveSession(camera, width, height, sinks=sinks)
    try:
        session.start()
    except RuntimeError as exc:
        update_status(str(exc), "DLC.LIVE")
        return 1
    try:
        while session.running:
            time.sleep(1.0)
            print(f"\r[DLC.LIVE] {session.fps:5.1f} fps", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        session.stop()
    if session.error:
        update_status(session.error, "DLC.LIVE")
        return 1
    return 0


__all__ = [
    "ASYNC_SWAP_FACE_LOSS_GRACE_S", "ENHANCERS", "SWAP_MODELS", "FramePipeline", "LiveConfig",
    "LiveSession", "configure", "ensure_models", "preload_models", "run_cli", "swap_image",
]
