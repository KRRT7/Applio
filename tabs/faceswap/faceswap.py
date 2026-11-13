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
    "source_path": "",
    "camera": "",
    "resolution": "1280x720",
    "virtual_camera": True,
    "swap_model": "hyperswap_1c_256",
    "enhancer": "none",
    "det_size": 320,
    "mouth_mask": 0,
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


def _live_config(source, swap_model, enhancer, det_size, mouth_mask,
                 color_correction, many_faces, mirror, async_swap, show_fps):
    dlc = _dlc()
    return dlc.LiveConfig(
        source_path=source or None,
        swap_model=swap_model,
        enhancer=None if enhancer == "none" else enhancer,
        execution_providers=[dlc.suggest_default_execution_provider()],
        det_size=int(det_size),
        mouth_mask=float(mouth_mask),
        color_correction=bool(color_correction),
        many_faces=bool(many_faces),
        mirror=bool(mirror),
        async_swap=bool(async_swap),
        show_fps=bool(show_fps),
    )


def _stop_session():
    global _SESSION, _VCAM
    with _LOCK:
        session, _SESSION, _VCAM = _SESSION, None, None
    if session is not None:
        session.stop()


def start_live(source, camera, resolution, virtual_camera, *quality):
    """Generator: (status, timer) updates while models load and the camera starts."""
    global _SESSION, _VCAM
    if not source or not os.path.isfile(source):
        raise gr.Error(i18n("Choose a source face image first."))
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
    vcam = dlc.VirtualCameraSink() if virtual_camera else None
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
        ["source_path", "camera", "resolution", "virtual_camera", "swap_model", "enhancer",
         "det_size", "mouth_mask", "color_correction", "many_faces", "mirror",
         "async_swap", "show_fps"],
        [source, camera, resolution, bool(virtual_camera), *quality],
    )))
    yield _status_text(), gr.Timer(active=True)


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


def _status_text():
    with _LOCK:
        session, vcam = _SESSION, _VCAM
    if session is None:
        return i18n("Stopped.")
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
    if _MESSAGES and time.monotonic() - _MESSAGES[-1][0] < 5.0:
        parts.append(_MESSAGES[-1][1])  # latest library message, for 5 s
    return " · ".join(parts)


def poll_preview():
    with _LOCK:
        session = _SESSION
    if session is None:
        return gr.skip(), i18n("Stopped."), gr.Timer(active=False)
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
    saved_source = saved["source_path"] if saved["source_path"] and os.path.isfile(saved["source_path"]) else None

    with gr.Blocks():
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                with step_card(1, "Source Face", i18n, description="The face to wear. A sharp, front-facing photo with eyes open and mouth closed works best."):
                    source = gr.Image(label=i18n("Source face"), type="filepath", value=saved_source, height=260, sources=["upload"])

                with step_card(2, "Camera & Output", i18n, description="Pick your camera. Virtual camera output lets Zoom, Meet, WhatsApp, etc. use the result."):
                    with gr.Row():
                        camera = gr.Dropdown(label=i18n("Camera"), choices=cameras, value=saved_camera, scale=3)
                        refresh = gr.Button(i18n("Refresh"), scale=1, size="sm")
                    resolution = gr.Dropdown(label=i18n("Resolution"), choices=RESOLUTIONS, value=saved["resolution"])
                    virtual_camera = gr.Checkbox(
                        label=i18n("Send to virtual camera (OBS Virtual Camera)"), value=saved["virtual_camera"],
                        info=i18n("On macOS: install OBS and start its virtual camera once; OBS does not need to run."),
                    )

                with step_card(3, "Quality", i18n, description="Changes apply immediately while live."):
                    swap_model = gr.Dropdown(label=i18n("Swap model"), choices=SWAP_MODELS, value=saved["swap_model"])
                    enhancer = gr.Dropdown(label=i18n("Face enhancer"), choices=ENHANCERS, value=saved["enhancer"])
                    det_size = gr.Dropdown(label=i18n("Face detection size"), choices=DET_SIZES, value=saved["det_size"],
                                           info=i18n("Lower is faster; 320 is plenty at webcam distance."))
                    mouth_mask = gr.Slider(label=i18n("Mouth mask"), minimum=0, maximum=100, step=1, value=saved["mouth_mask"],
                                           info=i18n("Keep your real mouth for natural speech. 0 = off."))
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
                status = gr.Textbox(label=i18n("Status"), value=i18n("Stopped."), interactive=False)

        timer = gr.Timer(0.2, active=False)
        quality = [swap_model, enhancer, det_size, mouth_mask, color_correction, many_faces, mirror, async_swap, show_fps]

        refresh.click(lambda: gr.Dropdown(choices=camera_choices()), outputs=camera)
        start.click(start_live, inputs=[source, camera, resolution, virtual_camera, *quality], outputs=[status, timer])
        stop.click(stop_live, outputs=[status, timer, preview])
        timer.tick(poll_preview, outputs=[preview, status, timer], show_progress="hidden")
        for control in [source, *quality]:
            control.change(apply_live, inputs=[source, *quality], show_progress="hidden")
