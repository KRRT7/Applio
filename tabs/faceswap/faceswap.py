"""Face Swap tab: live face swap from a camera via the vendored Deep-Live-Cam.

The camera is read and processed locally (deep_live_cam.LiveSession); this
tab only drives it and shows a preview. Output can also go to a virtual
camera so video-call apps can use it.
"""

import atexit
import json
import os
import sys
import threading
import time
from collections import deque

import cv2
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.app_shell import step_card
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")
# Keep Deep-Live-Cam's models with Applio's other models (git-ignored), not
# inside _vendor/. Must be set before the vendored package is imported.
os.environ.setdefault("DLC_MODELS_DIR", os.path.join(now_dir, "rvc", "models", "faceswap"))

SWAP_MODELS = [
    ("HyperSwap 1c (recommended)", "hyperswap_1c_256"),
    ("HyperSwap 1b", "hyperswap_1b_256"),
    ("HyperSwap 1a", "hyperswap_1a_256"),
    ("InSwapper 128", "inswapper_128"),
]
ENHANCERS = [
    ("None (recommended)", "none"),
    ("GPEN-256 (sharper, ~2x slower)", "gpen256"),
    ("GFPGAN (sharpest, ~3x slower)", "gfpgan"),
    ("GPEN-512 (slowest)", "gpen512"),
]
RESOLUTIONS = ["640x480", "960x540", "1280x720", "1920x1080"]
DET_SIZES = [160, 320, 640]
PREVIEW_WIDTH = 640

DEFAULTS = {
    "source_paths": [],
    "camera": "",
    "resolution": "1280x720",
    "virtual_camera": True,
    "video_delay": 0,
    "swap_model": "hyperswap_1c_256",
    "enhancer": "none",
    "det_size": 320,
    "mouth_mask": 0,
    "sharpness": 0.0,
    "stabilization": 0.5,
    "detect_every_frame": True,
    "color_correction": False,
    "many_faces": False,
    "mirror": False,
    "async_swap": False,
    "show_fps": False,
}

_LOCK = threading.Lock()
_SESSION = None
_VCAM = None
_MESSAGES = deque(maxlen=1)


# ─── library access ──────────────────────────────────────────────────────


def _dlc():
    """Import the vendored deep_live_cam lazily so Applio starts even if its
    dependencies (the 'faceswap' dependency group) are missing."""
    try:
        from _vendor import deep_live_cam
    except ImportError as exc:
        raise gr.Error(
            f"Face swap dependencies are missing ({exc}). Run `uv sync`; they "
            "are in the 'faceswap' dependency group."
        ) from exc
    if not getattr(_dlc, "_listening", False):
        deep_live_cam.add_status_listener(_on_status)
        _dlc._listening = True
    return deep_live_cam


# Routine messages already reflected in the status line (camera size,
# virtual camera device) are not repeated.
_QUIET_SCOPES = {"DLC.LIVE", "DLC.VCAM"}


def _on_status(message, scope):
    if scope not in _QUIET_SCOPES or "unavailable" in message or "fail" in message.lower():
        _MESSAGES.append((time.monotonic(), message))


def camera_choices():
    try:
        return [(name, f"{index}:{name}") for index, name in _dlc().list_cameras()]
    except gr.Error:
        return []


# ─── settings ────────────────────────────────────────────────────────────


def load_settings():
    settings = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            settings.update(json.load(f).get("faceswap", {}))
    except (OSError, ValueError):
        pass
    if not settings["source_paths"] and settings.get("source_path"):
        settings["source_paths"] = [settings["source_path"]]  # pre-multi-photo settings
    settings.pop("source_path", None)
    return settings


def save_settings(values):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        config = {}
    config["faceswap"] = values
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Error saving face swap settings: {exc}")


# ─── session control ─────────────────────────────────────────────────────


def _source_paths(gallery):
    """File paths from the source gallery value (items may be (path, caption))."""
    paths = []
    for item in gallery or []:
        path = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(path, dict):
            path = path.get("path") or path.get("name")
        if path and os.path.isfile(path):
            paths.append(path)
    return paths


def _live_config(sources, swap_model, enhancer, det_size, mouth_mask, sharpness,
                 stabilization, detect_every_frame, color_correction, many_faces,
                 mirror, async_swap, show_fps):
    dlc = _dlc()
    paths = _source_paths(sources)
    return dlc.LiveConfig(
        source_path=paths[0] if paths else None,
        source_paths=paths[1:],
        swap_model=swap_model,
        enhancer=None if enhancer == "none" else enhancer,
        execution_providers=[dlc.suggest_default_execution_provider()],
        det_size=int(det_size),
        mouth_mask=float(mouth_mask),
        sharpness=float(sharpness),
        stabilization=float(stabilization),
        detect_every_frame=bool(detect_every_frame),
        color_correction=bool(color_correction),
        many_faces=bool(many_faces),
        mirror=bool(mirror),
        async_swap=bool(async_swap),
        show_fps=bool(show_fps),
    )


# The video swap path lags the camera by roughly this much (one camera frame
# plus processing), so it is subtracted from the voice delay when matching.
_VIDEO_OWN_LATENCY_MS = 65
# Per-block processing plus audio device buffering, on top of chunk+crossfade.
_VOICE_OVERHEAD_MS = 25


def estimate_voice_delay(chunk_size_ms, cross_fade_s):
    """Video delay (ms) that lines swapped lips up with realtime voice.

    Voice output trails the mic by about chunk + crossfade + overhead; the
    video already trails the camera by _VIDEO_OWN_LATENCY_MS. An estimate:
    fine-tune by clapping on camera and watching a recording.
    """
    voice = float(chunk_size_ms or 0) + float(cross_fade_s or 0) * 1000 + _VOICE_OVERHEAD_MS
    return int(round(max(0.0, voice - _VIDEO_OWN_LATENCY_MS) / 10) * 10)


def _stop_session():
    global _SESSION, _VCAM
    with _LOCK:
        session, _SESSION, _VCAM = _SESSION, None, None
    if session is not None:
        session.stop()


def start_live(source, camera, resolution, virtual_camera, video_delay, *quality):
    """Generator: (status, timer) updates while models load and the camera starts."""
    global _SESSION, _VCAM
    if not _source_paths(source):
        raise gr.Error(i18n("Add at least one source face photo first."))
    if not camera:
        raise gr.Error(i18n("No camera selected."))
    _stop_session()
    dlc = _dlc()

    dlc.configure(_live_config(source, *quality))
    yield i18n("Checking models (first run downloads them)..."), gr.Timer(active=False)
    if not dlc.ensure_models():
        raise gr.Error(i18n("Could not download the face swap models. Check your connection."))

    index = int(str(camera).split(":", 1)[0])
    width, height = (int(v) for v in resolution.split("x"))
    vcam = dlc.DelayedSink(dlc.VirtualCameraSink(), float(video_delay or 0)) if virtual_camera else None
    session = dlc.LiveSession(index, width, height, sinks=[vcam] if vcam else [])
    yield i18n("Loading models (the first start compiles them, ~10-20 s)..."), gr.Timer(active=False)
    dlc.preload_models()
    yield i18n("Starting camera..."), gr.Timer(active=False)
    try:
        session.start()
    except RuntimeError as exc:
        raise gr.Error(str(exc))
    with _LOCK:
        _SESSION, _VCAM = session, vcam

    save_settings(dict(zip(
        ["source_paths", "camera", "resolution", "virtual_camera", "video_delay", "swap_model", "enhancer",
         "det_size", "mouth_mask", "sharpness", "stabilization", "detect_every_frame",
         "color_correction", "many_faces", "mirror", "async_swap", "show_fps"],
        [_source_paths(source), camera, resolution, bool(virtual_camera), float(video_delay or 0), *quality],
    )))
    yield _status_text(), gr.Timer(active=True)


def start_live_with_voice(chunk_size_ms, cross_fade_s, source, camera, resolution, virtual_camera, video_delay, *quality):
    """start_live for "Start face + voice": a 0 video delay is set to the
    estimate that matches the voice, and the slider is updated to show it."""
    delay = float(video_delay or 0) or estimate_voice_delay(chunk_size_ms, cross_fade_s)
    first = True
    for status, timer in start_live(source, camera, resolution, virtual_camera, delay, *quality):
        yield status, timer, (gr.update(value=delay) if first else gr.skip())
        first = False


def set_video_delay(delay_ms):
    """Change the delay of the running virtual-camera output."""
    with _LOCK:
        vcam = _VCAM
    if vcam is not None and hasattr(vcam, "delay_ms"):
        vcam.delay_ms = float(delay_ms or 0)


def stop_live():
    _stop_session()
    return i18n("Stopped."), gr.Timer(active=False), None


def apply_live(source, *quality):
    """Push changed settings to a running session (applies on the next frame)."""
    with _LOCK:
        running = _SESSION is not None
    if not running:
        return
    dlc = _dlc()
    config = _live_config(source, *quality)
    dlc.configure(config)
    dlc.ensure_models()  # downloads a newly selected model/enhancer if needed


def _realtime():
    """The Realtime tab module, if its server-side controls are available."""
    try:
        from tabs.realtime import realtime
    except Exception:
        return None
    return realtime if realtime.SESSION_CONTROLS else None


def _voice_status():
    rt = _realtime()
    return rt.voice_status() if rt is not None else None


def _status_text():
    with _LOCK:
        session, vcam = _SESSION, _VCAM
    voice = _voice_status()
    if session is None:
        return f"{i18n('Face stopped')} · {voice}" if voice else i18n("Stopped.")
    if session.error:
        return f"{i18n('Error')}: {session.error}"
    w, h = session.camera_size
    parts = [
        f"{i18n('Live')} · {session.fps:.1f} fps",
        f"{i18n('camera')} {w}x{h}@{session.camera_fps:.0f}",
    ]
    if vcam is not None:
        if vcam.error:
            parts.append(f"{i18n('virtual camera unavailable')}: {vcam.error}")
        elif vcam.device:
            parts.append(f"{i18n('virtual camera')}: {vcam.device}")
    if voice:
        parts.append(voice)
    if _MESSAGES and time.monotonic() - _MESSAGES[-1][0] < 5.0:
        parts.append(_MESSAGES[-1][1])  # latest library message, for 5 s
    return " · ".join(parts)


def poll_preview():
    with _LOCK:
        session = _SESSION
    if session is None:
        # Keep the shared status line updating while voice runs on its own.
        voice = _voice_status()
        return gr.skip(), _status_text(), gr.Timer(active=voice is not None)
    frame = session.latest_frame
    if not session.running:
        _stop_session()
        return gr.skip(), _status_text() if session.error is None else f"{i18n('Error')}: {session.error}", gr.Timer(active=False)
    if frame is None:
        return gr.skip(), _status_text(), gr.Timer()
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH:
        frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), _status_text(), gr.Timer()


atexit.register(_stop_session)


# ─── UI ──────────────────────────────────────────────────────────────────


def faceswap_tab():
    saved = load_settings()
    cameras = camera_choices()
    camera_values = [value for _label, value in cameras]
    saved_camera = saved["camera"] if saved["camera"] in camera_values else (camera_values[0] if camera_values else None)
    saved_sources = [p for p in saved["source_paths"] if os.path.isfile(p)] or None

    with gr.Blocks():
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                with step_card(1, "Source Face", i18n, description="Photos of the face to wear, all of the same person. 3-5 sharp, front-facing shots (eyes open, mouth closed) give a steadier likeness than one."):
                    source = gr.Gallery(label=i18n("Source face photos"), type="filepath", value=saved_sources,
                                        interactive=True, columns=2, height=300, object_fit="cover",
                                        allow_preview=False, preview=False)

                with step_card(2, "Camera & Output", i18n, description="Pick your camera. Virtual camera output lets Zoom, Meet, WhatsApp, etc. use the result."):
                    with gr.Row():
                        camera = gr.Dropdown(label=i18n("Camera"), choices=cameras, value=saved_camera, scale=3)
                        refresh = gr.Button(i18n("Refresh"), scale=1, size="sm")
                    resolution = gr.Dropdown(label=i18n("Resolution"), choices=RESOLUTIONS, value=saved["resolution"])
                    virtual_camera = gr.Checkbox(
                        label=i18n("Send to virtual camera (OBS Virtual Camera)"), value=saved["virtual_camera"],
                        info=i18n("On macOS: install OBS and start its virtual camera once; OBS does not need to run."),
                    )
                    video_delay = gr.Slider(label=i18n("Video delay (ms)"), minimum=0, maximum=1000, step=10, value=saved["video_delay"],
                                            info=i18n("Delays the virtual-camera video so lips match converted voice. Fine-tune by clapping on camera."))
                    match_voice = gr.Button(i18n("Match voice delay"), size="sm", visible=_realtime() is not None)

                with step_card(3, "Quality", i18n, description="Changes apply immediately while live."):
                    swap_model = gr.Dropdown(label=i18n("Swap model"), choices=SWAP_MODELS, value=saved["swap_model"])
                    enhancer = gr.Dropdown(label=i18n("Face enhancer"), choices=ENHANCERS, value=saved["enhancer"])
                    det_size = gr.Dropdown(label=i18n("Face detection size"), choices=DET_SIZES, value=saved["det_size"],
                                           info=i18n("Lower is faster; 320 is plenty at webcam distance."))
                    mouth_mask = gr.Slider(label=i18n("Mouth mask"), minimum=0, maximum=100, step=1, value=saved["mouth_mask"],
                                           info=i18n("Keep your real mouth for natural speech. 0 = off."))
                    sharpness = gr.Slider(label=i18n("Sharpness"), minimum=0, maximum=1, step=0.05, value=saved["sharpness"],
                                          info=i18n("Sharpen the swapped face. Nearly free; try 0.2-0.4."))
                    stabilization = gr.Slider(label=i18n("Stabilization"), minimum=0, maximum=1, step=0.05, value=saved["stabilization"],
                                              info=i18n("Steadies the face between frames (less edge shimmer). Higher = steadier but laggier; 0.5 is a good balance."))
                    detect_every_frame = gr.Checkbox(label=i18n("Detect every frame"), value=saved["detect_every_frame"],
                                                     info=i18n("Smoother tracking when you move (~2 ms per frame). Turn off on slower machines."))
                    with gr.Row():
                        color_correction = gr.Checkbox(label=i18n("Color correction"), value=saved["color_correction"])
                        many_faces = gr.Checkbox(label=i18n("All faces"), value=saved["many_faces"])
                    with gr.Row():
                        mirror = gr.Checkbox(label=i18n("Mirror"), value=saved["mirror"])
                        show_fps = gr.Checkbox(label=i18n("Show FPS"), value=saved["show_fps"])
                    async_swap = gr.Checkbox(label=i18n("Async swap"), value=saved["async_swap"],
                                             info=i18n("Smoother motion on slow hardware; expressions lag slightly."))

            with gr.Column(scale=2, min_width=360):
                preview = gr.Image(label=i18n("Live preview"), interactive=False, format="jpeg", height=480)
                with gr.Row():
                    start = gr.Button(i18n("Start"), variant="primary")
                    stop = gr.Button(i18n("Stop"))
                rt = _realtime()
                if rt is not None:
                    with gr.Row():
                        start_both = gr.Button(i18n("Start face + voice"), variant="primary")
                        stop_all = gr.Button(i18n("Stop all"))
                    gr.Markdown(i18n("Voice uses the model, devices and settings chosen in the Realtime tab."),
                                elem_classes="applio-card-desc")
                status = gr.Textbox(label=i18n("Status"), value=i18n("Stopped."), interactive=False)

        timer = gr.Timer(0.2, active=False)
        quality = [swap_model, enhancer, det_size, mouth_mask, sharpness, stabilization, detect_every_frame,
                   color_correction, many_faces, mirror, async_swap, show_fps]

        refresh.click(lambda: gr.Dropdown(choices=camera_choices()), outputs=camera)
        start.click(start_live, inputs=[source, camera, resolution, virtual_camera, video_delay, *quality], outputs=[status, timer])
        video_delay.change(set_video_delay, inputs=video_delay, show_progress="hidden")
        stop.click(stop_live, outputs=[status, timer, preview])
        if rt is not None:
            controls = rt.SESSION_CONTROLS
            face_inputs = [source, camera, resolution, virtual_camera, video_delay, *quality]
            # Two listeners on one click run concurrently: the face session
            # here, and the Realtime tab's own start handler with its own
            # controls (so that tab's buttons and latency readout stay in sync).
            start_both.click(start_live_with_voice, inputs=[*controls["timing_inputs"], *face_inputs],
                             outputs=[status, timer, video_delay])
            match_voice.click(estimate_voice_delay, inputs=controls["timing_inputs"], outputs=video_delay)
            start_both.click(controls["start_fn"], inputs=controls["start_inputs"], outputs=controls["outputs"])
            stop_all.click(stop_live, outputs=[status, timer, preview])
            stop_all.click(controls["stop_fn"], outputs=controls["outputs"])
        timer.tick(poll_preview, outputs=[preview, status, timer], show_progress="hidden")
        for control in [source, *quality]:
            control.change(apply_live, inputs=[source, *quality], show_progress="hidden")
