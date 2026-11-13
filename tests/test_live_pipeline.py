"""Tests for the vendored Deep-Live-Cam live pipeline (_vendor.modules.live).

No model is loaded: face detection and the frame-processor list are stubbed,
so these exercise control flow only. The source-reload, async-swap, session
and configure cases are ported from the original Deep-Live-Cam test suite.
"""
import itertools
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

import _vendor.modules.globals as g
from _vendor.modules import cameras, live
from _vendor.modules.virtual_camera import VirtualCameraSink


def _frame():
    return np.zeros((16, 16, 3), dtype=np.uint8)


@pytest.fixture
def no_faces():
    """A pipeline with no processors and no detections."""
    with patch.object(live, "get_frame_processors_modules", lambda *_: []), \
         patch.object(live, "detect_one_face_fast", lambda *_: None), \
         patch.object(live, "detect_many_faces_fast", lambda *_: None), \
         patch.object(live, "update_status", lambda *_a, **_k: None):
        yield


@pytest.fixture
def clean_globals():
    """Restore the live settings each test may change."""
    keys = ("source_path", "source_paths", "map_faces", "many_faces", "mouth_mask", "live_async_swap",
            "live_mirror", "show_fps", "live_stabilization", "live_detect_every_frame")
    saved = {k: getattr(g, k) for k in keys}
    g.source_paths = ()
    g.live_stabilization = 0.0
    g.live_detect_every_frame = False
    yield
    for k, v in saved.items():
        setattr(g, k, v)


class _Swapper:
    NAME = "DLC.FACE-SWAPPER"

    @staticmethod
    def apply_post_processing(frame, _bboxes):
        return frame

    @staticmethod
    def process_frame_v2(frame):
        return frame


# ─── source handling ──────────────────────────────────────────────────────


@pytest.mark.parametrize("async_enabled", [False, True])
def test_clearing_source_stops_swaps_and_same_path_reloads(clean_globals, async_enabled):
    sources, loaded, calls = [], [], []

    def load_source(_frame):
        loaded.append(object())
        return loaded[-1]

    def record_swap(source, _target, frame):
        sources.append(source)
        return frame

    class _Async:
        swap = staticmethod(record_swap)

        def clear(self):
            calls.append("clear")

        def close(self):
            calls.append("close")

    class Swapper(_Swapper):
        AsyncLiveSwapper = _Async
        swap_face = staticmethod(record_swap)

    g.map_faces = g.many_faces = g.mouth_mask = g.live_mirror = g.show_fps = False
    g.source_path, g.live_async_swap = "A", async_enabled
    with patch.object(live, "imread_unicode", lambda *_a, **_k: _frame()), \
         patch.object(live, "get_one_face", load_source), \
         patch.object(live, "get_frame_processors_modules", lambda *_: [Swapper]), \
         patch.object(live, "detect_one_face_fast", lambda *_: object()):
        pipeline = live.FramePipeline(10.0)
        pipeline.process(_frame())
        assert sources[-1] is loaded[0]
        g.source_path = None
        pipeline.process(_frame())
        pipeline.process(_frame())
        assert sources[-2:] == [None, None]
        if async_enabled:
            assert "clear" in calls
        g.source_path = "A"
        pipeline.process(_frame())
        assert len(loaded) == 2 and sources[-1] is loaded[1]
        pipeline.close()


def test_switching_back_to_a_working_source_reloads_it(clean_globals, no_faces):
    reads = []

    def fake_imread(path, *_a, **_k):
        reads.append(path)
        return None if path == "B" else _frame()

    g.map_faces = g.live_mirror = g.show_fps = False
    g.source_path = "A"
    with patch.object(live, "imread_unicode", fake_imread), patch.object(live, "get_one_face", lambda *_: object()):
        pipeline = live.FramePipeline(30.0)
        pipeline.process(_frame())
        g.source_path = "B"
        pipeline.process(_frame())
        g.source_path = "A"
        pipeline.process(_frame())
    assert reads.count("A") == 2, "A was not reloaded after an unreadable source"


def test_several_source_photos_are_averaged(clean_globals, no_faces):
    from insightface.app.common import Face

    def face_with(vec):
        f = Face(bbox=np.zeros(4), kps=np.zeros((5, 2)), det_score=1.0)
        f.embedding = np.asarray(vec, np.float32)
        return f

    faces = {"a.jpg": face_with([1, 0]), "b.jpg": face_with([0, 1])}
    with patch.object(live, "imread_unicode", lambda p, *_a, **_k: p), \
         patch.object(live, "get_one_face", lambda img: faces[img]):
        pipeline = live.FramePipeline(30.0)
        avg = pipeline._load_source(("a.jpg", "b.jpg"))
        single = pipeline._load_source(("a.jpg",))
    assert np.allclose(avg.normed_embedding, [2 ** -0.5, 2 ** -0.5])
    assert single is faces["a.jpg"]


# ─── async swap ───────────────────────────────────────────────────────────


def test_pipeline_drives_the_async_swapper(clean_globals):
    calls = []

    class _Async:
        def __init__(self):
            calls.append("init")

        def swap(self, _s, _t, frame):
            calls.append("swap")
            return frame

        def clear(self):
            calls.append("clear")

        def close(self):
            calls.append("close")

    class Swapper(_Swapper):
        AsyncLiveSwapper = _Async

        @staticmethod
        def swap_face(_s, _t, frame):
            calls.append("sync")
            return frame

    face = object()
    detections = iter([face, None, face] + [None] * 4 + [face])
    g.map_faces = g.many_faces = g.mouth_mask = g.live_mirror = g.show_fps = False
    g.source_path, g.live_async_swap = "A", True
    with patch.object(live, "imread_unicode", lambda *_a, **_k: _frame()), \
         patch.object(live, "get_one_face", lambda *_: object()), \
         patch.object(live, "get_frame_processors_modules", lambda *_: [Swapper]), \
         patch.object(live, "detect_one_face_fast", lambda *_: next(detections)), \
         patch.object(live, "update_status", lambda *_: None), \
         patch.object(live, "ASYNC_SWAP_FACE_LOSS_GRACE_S", 0.2):
        pipeline = live.FramePipeline(10.0)
        for _ in range(3):
            pipeline.process(_frame())
        for _ in range(4):
            pipeline.process(_frame())
            time.sleep(0.1)
        pipeline.process(_frame())
        g.map_faces = True
        pipeline.process(_frame())
        pipeline.close()
    steps = [s for s, _ in itertools.groupby(calls)]
    assert steps == ["init", "swap", "clear", "swap", "clear", "close"]
    assert calls[:3] == ["init", "swap", "swap"]


# ─── stabilization ────────────────────────────────────────────────────────


def test_landmark_smoother_damps_jitter_and_resets_on_jumps():
    sm = live.LandmarkSmoother()
    base = np.array([[100, 100], [140, 100], [120, 120], [105, 140], [135, 140]], float)
    rng = np.random.default_rng(0)
    raw = [base + rng.normal(0, 2, base.shape) for _ in range(60)]
    out = [sm(k, (i + 1) / 30, 0.5) for i, k in enumerate(raw)]
    jitter = lambda seq: np.mean([np.abs(np.diff(np.array(seq), n=2, axis=0)).mean()])
    assert jitter(out) < 0.6 * jitter(raw)
    jumped = base + 500  # a different face far away: no smoothing across it
    assert np.allclose(sm(jumped, 61 / 30, 0.5), jumped)


def test_stabilization_and_detection_interval_reach_the_pipeline(clean_globals):
    seen = []
    g.map_faces = g.live_mirror = g.show_fps = False
    g.source_path = None
    with patch.object(live, "get_frame_processors_modules", lambda *_: []), \
         patch.object(live, "detect_one_face_fast", lambda *_: seen.append(1)), \
         patch.object(live, "update_status", lambda *_: None):
        pipeline = live.FramePipeline(30.0)  # detects every 2nd frame by default
        for _ in range(6):
            pipeline.process(_frame())
        assert len(seen) == 3
        g.live_detect_every_frame = True
        for _ in range(6):
            pipeline.process(_frame())
        assert len(seen) == 9


# ─── live session ─────────────────────────────────────────────────────────


class _StubCapture:
    read_delay = 0.3

    def __init__(self, device_index=0):
        self.actual_width = self.actual_height = 16
        self.actual_fps = 30.0
        self.released = False

    def start(self, width=0, height=0, fps=0):
        return True

    def read(self):
        time.sleep(self.read_delay)
        return True, _frame()

    def release(self):
        self.released = True


def test_stop_joins_threads_and_releases_everything(clean_globals, no_faces):
    g.source_path, g.map_faces = None, False
    received, closed, caps = [], [], []

    class Sink:
        def __call__(self, frame):
            received.append(frame)

        def close(self):
            closed.append(True)

    session = live.LiveSession(0, sinks=[Sink()], capture_factory=lambda i: caps.append(_StubCapture(i)) or caps[-1])
    session.start()
    deadline = time.time() + 5
    while not received and time.time() < deadline:
        time.sleep(0.02)
    session.stop()
    assert received and session.latest_frame is not None and not session.running
    assert not any(t.name.startswith("dlc-") for t in threading.enumerate())
    assert caps[0].released and closed == [True]


def test_failed_camera_start_raises_and_releases(clean_globals, no_faces):
    caps = []

    class Dead(_StubCapture):
        def start(self, *_a):
            return False

    session = live.LiveSession(3, capture_factory=lambda i: caps.append(Dead(i)) or caps[-1])
    with pytest.raises(RuntimeError):
        session.start()
    session.stop()
    assert caps[0].released and not session.running


def test_failing_sink_is_dropped_without_stopping_the_session(clean_globals, no_faces):
    g.source_path, g.map_faces = None, False
    good = []

    class Fast(_StubCapture):
        read_delay = 0.01

    def bad(_frame):
        raise OSError("device gone")

    session = live.LiveSession(0, sinks=[bad, good.append], capture_factory=Fast)
    session.start()
    deadline = time.time() + 5
    while len(good) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert session.running
    session.stop()
    assert len(good) >= 3 and session.sinks == [good.append]


def test_session_detects_and_renders_on_separate_threads_in_order(clean_globals, no_faces):
    g.source_path, g.map_faces, g.live_detect_every_frame = None, False, True
    detect_threads, render_threads, out = set(), set(), []

    class Counting(_StubCapture):
        read_delay = 0.005
        n = 0

        def read(self):
            time.sleep(self.read_delay)
            Counting.n += 1
            return True, np.full((16, 16, 3), Counting.n % 256, dtype=np.uint8)

    def detect(_frame):
        detect_threads.add(threading.current_thread().name)
        time.sleep(0.01)
        return None

    real_render = live.FramePipeline.render

    def render(self, det):
        render_threads.add(threading.current_thread().name)
        time.sleep(0.02)
        return real_render(self, det)

    with patch.object(live, "detect_one_face_fast", detect), patch.object(live.FramePipeline, "render", render):
        session = live.LiveSession(0, sinks=[lambda f: out.append(int(f[0, 0, 0]))], capture_factory=Counting)
        session.start()
        deadline = time.time() + 5
        while len(out) < 10 and time.time() < deadline:
            time.sleep(0.02)
        session.stop()
    assert detect_threads == {"dlc-detect"} and render_threads == {"dlc-process"}
    assert len(out) >= 10 and all(a < b for a, b in zip(out, out[1:]))  # newest frames, never reordered


def test_detection_failure_stops_the_session_with_an_error(clean_globals, no_faces):
    g.source_path, g.map_faces, g.live_detect_every_frame = None, False, True

    class Fast(_StubCapture):
        read_delay = 0.01

    def boom(_frame):
        raise RuntimeError("detector died")

    with patch.object(live, "detect_one_face_fast", boom):
        session = live.LiveSession(0, capture_factory=Fast)
        session.start()
        deadline = time.time() + 5
        while session.running and time.time() < deadline:
            time.sleep(0.02)
        session.stop()
    assert not session.running and "detector died" in session.error


# ─── configure ────────────────────────────────────────────────────────────


@pytest.fixture
def saved_config_globals():
    keys = ("fp_ui", "frame_processors", "execution_providers", "execution_threads", "det_size", "source_path",
            "source_paths", "mouth_mask", "mouth_mask_size", "many_faces", "live_mirror", "live_async_swap",
            "show_fps", "color_correction", "swap_model", "live_stabilization", "live_detect_every_frame")
    saved = {k: getattr(g, k) for k in keys}
    saved["fp_ui"], saved["frame_processors"] = dict(saved["fp_ui"]), list(saved["frame_processors"])
    yield
    for k, v in saved.items():
        setattr(g, k, v)


def test_configure_maps_config_onto_globals(saved_config_globals):
    with patch.object(live, "reset_face_analyser", lambda: None):
        live.configure(live.LiveConfig(
            source_path="face.jpg", source_paths=["b.jpg", "face.jpg"], enhancer="gpen512",
            execution_providers=["cpu"], swap_model="hyperswap_1b_256", det_size=320, mouth_mask=30,
            many_faces=True, mirror=True, async_swap=True, stabilization=0.5, detect_every_frame=True,
        ))
    assert g.source_path == "face.jpg" and g.source_paths == ("face.jpg", "b.jpg")  # de-duplicated
    assert g.swap_model == "hyperswap_1b_256" and g.det_size == 320
    assert g.execution_providers == ["CPUExecutionProvider"]
    assert g.fp_ui == {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": True}
    assert g.mouth_mask and g.mouth_mask_size == 30.0
    assert g.many_faces and g.live_mirror and g.live_async_swap
    assert g.live_stabilization == 0.5 and g.live_detect_every_frame


def test_default_swap_model_and_validation(saved_config_globals):
    assert live.LiveConfig().swap_model == "hyperswap_1c_256"
    for bad in (dict(swap_model="simswap_512"), dict(enhancer="codeformer"), dict(det_size=300),
                dict(stabilization=2.0)):
        with pytest.raises(ValueError):
            live.configure(live.LiveConfig(execution_providers=["cpu"], **bad))


def test_swap_model_list_matches_the_swapper():
    from _vendor.modules.processors.frame import face_swapper
    assert live.SWAP_MODELS == face_swapper.SWAP_MODELS


# ─── sinks and cameras ────────────────────────────────────────────────────


def test_delayed_sink_delays_skips_ahead_and_passes_through():
    now, got = [0.0], []
    sink = live.DelayedSink(lambda f: got.append((round(now[0] * 1000), f)), 260, clock=lambda: now[0])
    for i in range(20):
        now[0] = i * 0.033
        sink(i)
    assert {t - f * 33 for t, f in got} == {264}  # constant lag, quantised to frames
    got.clear()
    sink.delay_ms = 100
    for i in range(20, 26):
        now[0] = i * 0.033
        sink(i)
    assert {t - f * 33 for t, f in got} == {132}  # no burst of the backlog
    got.clear()
    sink.delay_ms = 0
    now[0] = 1.0
    sink(99)
    assert got == [(1000, 99)]


def test_virtual_camera_failure_disables_instead_of_raising():
    class Broken:
        class PixelFormat:
            BGR = "bgr"

        @staticmethod
        def Camera(**_kw):
            raise RuntimeError("no virtual camera installed")

    with patch.dict(sys.modules, {"pyvirtualcam": Broken}), \
         patch("_vendor.modules.virtual_camera.update_status", lambda *_a: None):
        sink = VirtualCameraSink()
        sink(_frame())
        sink(_frame())
    assert not sink.enabled and "no virtual camera" in sink.error


def test_virtual_camera_reopens_when_frame_size_changes():
    opened = []

    class Cam:
        def __init__(self, width, height, **_kw):
            self.width, self.height, self.device = width, height, "Fake"
            opened.append((width, height))

        def send(self, _f):
            pass

        def close(self):
            pass

    class Mod:
        class PixelFormat:
            BGR = "bgr"
        Camera = Cam

    with patch.dict(sys.modules, {"pyvirtualcam": Mod}), \
         patch("_vendor.modules.virtual_camera.update_status", lambda *_a: None):
        sink = VirtualCameraSink()
        sink(np.zeros((480, 640, 3), np.uint8))
        sink(np.zeros((480, 640, 3), np.uint8))
        sink(np.zeros((720, 1280, 3), np.uint8))
    assert opened == [(640, 480), (1280, 720)]


def test_virtual_cameras_sort_last_on_macos():
    fake = [type("C", (), {"index": i, "name": n})() for i, n in
            [(0, "OBS Virtual Camera"), (1, "FaceTime HD Camera"), (2, "iPhone Camera")]]
    mod = type("M", (), {"enumerate_cameras": staticmethod(lambda *_: fake)})
    with patch.object(cameras.platform, "system", lambda: "Darwin"), \
         patch.dict(sys.modules, {"cv2_enumerate_cameras": mod}):
        result = cameras.list_cameras()
    assert [n for _, n in result] == ["FaceTime HD Camera", "iPhone Camera", "OBS Virtual Camera"]
