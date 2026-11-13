"""Face Swap tab: live face swap from a camera via the vendored Deep-Live-Cam.

The camera is read and processed locally (deep_live_cam.LiveSession); this
tab only drives it and shows a preview. Output can also go to a virtual
camera so video-call apps can use it.
"""

import atexit
import json
import os
import re
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
    "profile": "",
    "face_enabled": True,
    "voice_enabled": True,
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
# Timestamped activity log shown under the status summary (newest last).
_LOG = deque(maxlen=300)
_LOG_LOCK = threading.Lock()
# Keep refreshing Status for a while after a start, even if the face side
# failed, so the voice side's progress still shows up.
_POLL_UNTIL = 0.0


def _log(message):
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    with _LOG_LOCK:
        _LOG.append(line)
    print(f"[FaceSwap] {message}")


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


def _on_status(message, scope):
    """Every Deep-Live-Cam status message goes into the activity log."""
    _log(f"Face ({scope.replace('DLC.', '').lower()}): {message}")


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


_QUALITY_NAMES = ["swap model", "enhancer", "detection size", "mouth mask", "sharpness",
                  "stabilization", "detect every frame", "color correction", "all faces",
                  "mirror", "async swap", "show fps"]


def _keep_polling():
    return _voice_status() is not None or time.monotonic() < _POLL_UNTIL


def _face_fail(message):
    """Log a face-start failure, show a toast, and return the status update."""
    _log(f"Face ERROR: {message}")
    gr.Warning(message)
    return _status_text(), gr.Timer(active=_keep_polling())


def start_live(source, camera, resolution, virtual_camera, video_delay, *quality):
    """Generator: (status, timer) updates while models load and the camera starts.

    Every step and failure is written to the activity log shown in Status.
    """
    global _SESSION, _VCAM
    paths = _source_paths(source)
    settings = ", ".join(f"{n} {v}" for n, v in zip(_QUALITY_NAMES, quality))
    _log(f"Face: starting | {len(paths)} source photo(s) | camera {camera} | {resolution} | "
         f"virtual camera {'on' if virtual_camera else 'off'} | video delay {float(video_delay or 0):.0f} ms")
    _log(f"Face: settings | {settings}")
    if not paths:
        yield _face_fail(i18n("Add at least one source face photo first."))
        return
    if not camera:
        yield _face_fail(i18n("No camera selected."))
        return
    try:
        _stop_session()
        dlc = _dlc()
        config = _live_config(source, *quality)
        dlc.configure(config)
        _log(f"Face: using {config.execution_providers[0]} | sources: " + ", ".join(os.path.basename(p) for p in paths))
        _log("Face: checking models (first run downloads them)...")
        yield _status_text(), gr.Timer(active=False)
        if not dlc.ensure_models():
            yield _face_fail(i18n("Could not download the face swap models. Check your connection."))
            return

        index = int(str(camera).split(":", 1)[0])
        width, height = (int(v) for v in resolution.split("x"))
        vcam = dlc.DelayedSink(dlc.VirtualCameraSink(), float(video_delay or 0)) if virtual_camera else None
        session = dlc.LiveSession(index, width, height, sinks=[vcam] if vcam else [])
        _log("Face: loading models (the first start compiles them, ~10-20 s)...")
        yield _status_text(), gr.Timer(active=False)
        t0 = time.monotonic()
        dlc.preload_models()
        _log(f"Face: models loaded in {time.monotonic() - t0:.1f} s; starting camera {index}...")
        yield _status_text(), gr.Timer(active=False)
        try:
            session.start()
        except RuntimeError as exc:
            yield _face_fail(str(exc))
            return
        with _LOCK:
            _SESSION, _VCAM = session, vcam
        w, h = session.camera_size
        _log(f"Face: live | camera running at {w}x{h}@{session.camera_fps:.0f}fps")
    except gr.Error as exc:
        yield _face_fail(str(exc))
        return
    except Exception as exc:  # anything unexpected: keep it visible in Status
        yield _face_fail(f"{type(exc).__name__}: {exc}")
        return

    save_settings({**load_settings(), **dict(zip(
        ["source_paths", "camera", "resolution", "virtual_camera", "video_delay", "swap_model", "enhancer",
         "det_size", "mouth_mask", "sharpness", "stabilization", "detect_every_frame",
         "color_correction", "many_faces", "mirror", "async_swap", "show_fps"],
        [paths, camera, resolution, bool(virtual_camera), float(video_delay or 0), *quality],
    ))})
    yield _status_text(), gr.Timer(active=True)


def start_live_with_voice(chunk_size_ms, cross_fade_s, source, camera, resolution, virtual_camera, video_delay, *quality):
    """start_live for "Start face + voice": a 0 video delay is set to the
    estimate that matches the voice, and the slider is updated to show it."""
    global _POLL_UNTIL
    _POLL_UNTIL = time.monotonic() + 30.0
    _log("Start face + voice pressed")
    delay = float(video_delay or 0) or estimate_voice_delay(chunk_size_ms, cross_fade_s)
    if not float(video_delay or 0):
        _log(f"Face + voice: video delay auto-set to {delay:.0f} ms "
             f"(voice chunk {float(chunk_size_ms or 0):.0f} ms + crossfade {float(cross_fade_s or 0) * 1000:.0f} ms)")
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
    with _LOCK:
        was_running = _SESSION is not None
    _stop_session()
    _log("Face: stopped" if was_running else "Face: stop pressed (was not running)")
    return _status_text(), gr.Timer(active=_keep_polling()), None


def voice_start_logged(*args):
    """The Realtime tab's start handler, with its settings and every status
    change written to the activity log. Yields the handler's updates as-is."""
    rt = _realtime()
    controls = rt.SESSION_CONTROLS
    pairs = []
    for component, value in zip(controls["start_inputs"], args):
        label = getattr(component, "label", None)
        if label and value not in (None, ""):
            pairs.append(f"{label}: {value}")
    _log("Voice: starting | " + " | ".join(pairs))
    # Progress counters ("Loading model... (3s)", warm-up bars) are logged
    # once; the live latency readout is in the summary line instead.
    last_kind, said_running, t_start = None, False, time.monotonic()
    try:
        for update in controls["start_fn"](*args):
            text = update[0] if isinstance(update, tuple) else update
            text = text.get("value", "") if isinstance(text, dict) else str(text)
            kind = re.sub(r"\s*\(.*?\).*$", "", text).strip()  # drop "(3s)", "(12 blocks) ███"
            if text.startswith("Latency"):
                if not said_running:
                    _log(f"Voice: running after {time.monotonic() - t_start:.0f} s ({text})")
                    said_running = True
            elif text and kind != last_kind:
                _log(f"Voice: {kind or text}")
            last_kind = kind
            yield update
    except Exception as exc:
        _log(f"Voice ERROR: {type(exc).__name__}: {exc}")
        raise
    _log("Voice: stopped")


def voice_stop_logged():
    result = _realtime().SESSION_CONTROLS["stop_fn"]()
    text = result[0] if isinstance(result, tuple) else result
    _log(f"Voice: stop pressed ({text})")
    return result


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


# Thresholds for the voice diagnostics written to the activity log.
_SILENT_DB = -60.0        # below this the mic (or output) counts as silent
_SOUND_DB = -50.0         # above this the mic counts as having sound
_SPEECH_DB = -35.0        # roughly speaking level; VAD rightly mutes quieter room noise
_VOICE_WATCH = {}


def _watch_voice():
    """Log notable live-audio events (silence, dropouts, overruns, errors)."""
    rt = _realtime()
    st = rt.voice_stats() if rt is not None and hasattr(rt, "voice_stats") else None
    w = _VOICE_WATCH
    if not st or not st.get("blocks"):
        if not st:
            w.clear()
        return
    now = time.monotonic()
    if not w:
        w.update(silent_since=None, warned_silent=False, had_sound=False, gated_since=None,
                 warned_gated=False, last_counts={}, last_report=float("-inf"), errors=0)
    mic, out = st["input_db"], st["output_db"]

    if mic < _SILENT_DB:
        if w["silent_since"] is None:
            w["silent_since"] = now
        if not w["warned_silent"] and now - w["silent_since"] >= 5.0:
            _log(f"Voice WARNING: no sound from the mic for 5 s (level {mic:.0f} dB). Check the Input "
                 "device and its volume, and that Windows allows microphone access.")
            w["warned_silent"] = True
    elif mic > _SOUND_DB:
        if not w["had_sound"] or w["warned_silent"]:
            _log(f"Voice: mic sound detected ({mic:.0f} dB)")
        w.update(silent_since=None, warned_silent=False, had_sound=True)

    if mic > _SPEECH_DB and out < _SILENT_DB:
        if w["gated_since"] is None:
            w["gated_since"] = now
        if not w["warned_gated"] and now - w["gated_since"] >= 3.0:
            _log(f"Voice WARNING: you seem to be speaking ({mic:.0f} dB) but the converted output is "
                 "silent. Check Enable VAD / Silence Threshold in the Realtime tab.")
            w["warned_gated"] = True
    else:
        w.update(gated_since=None, warned_gated=False)

    counts = {k: st[k] for k in ("overruns", "input_overflows", "output_underflows")}
    new = {k: v - w["last_counts"].get(k, 0) for k, v in counts.items() if v > w["last_counts"].get(k, 0)}
    if new and now - w["last_report"] >= 5.0:
        parts = []
        if "overruns" in new:
            parts.append(f"{new['overruns']} block(s) took longer than the block time "
                         f"({st['latency_ms']:.0f} ms > {st['block_ms']:.0f} ms)")
        if "output_underflows" in new:
            parts.append(f"{new['output_underflows']} output dropout(s)")
        if "input_overflows" in new:
            parts.append(f"{new['input_overflows']} lost input block(s)")
        _log("Voice WARNING: " + "; ".join(parts) + ". Audio may stutter: try Async swap or a lighter "
             "face setting, or a larger Chunk Size.")
        w["last_counts"], w["last_report"] = counts, now

    if st["callback_errors"] > w["errors"]:
        _log(f"Voice ERROR in the audio stream ({st['callback_errors']} so far): {st['last_error']}")
        w["errors"] = st["callback_errors"]


def _summary():
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
    return " · ".join(parts)


def _status_text():
    """Live summary line, then the activity log (newest last)."""
    with _LOG_LOCK:
        lines = list(_LOG)
    return _summary() + ("\n" + "─" * 40 + "\n" + "\n".join(lines) if lines else "")


def poll_preview():
    _watch_voice()
    with _LOCK:
        session = _SESSION
    if session is None:
        # Keep the status updating while voice runs on its own, or shortly
        # after a start (so a failed face start still shows voice progress).
        return gr.skip(), _status_text(), gr.Timer(active=_keep_polling())
    frame = session.latest_frame
    if not session.running:
        _stop_session()
        _log(f"Face ERROR: {session.error}" if session.error else "Face: session ended")
        return gr.skip(), _status_text(), gr.Timer(active=_keep_polling())
    if frame is None:
        return gr.skip(), _status_text(), gr.Timer()
    h, w = frame.shape[:2]
    if w > PREVIEW_WIDTH:
        frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), _status_text(), gr.Timer()


atexit.register(_stop_session)


# ─── UI ──────────────────────────────────────────────────────────────────


# ─── Live page: combined face + voice ───────────────────────────────────


def _save_toggles(face_enabled, voice_enabled):
    save_settings({**load_settings(), "face_enabled": bool(face_enabled), "voice_enabled": bool(voice_enabled)})


def start_face_if_enabled(face_enabled, voice_enabled, chunk_size_ms, cross_fade_s, *face_inputs):
    """Live page Start, face side: (status, timer, video delay) updates."""
    global _POLL_UNTIL
    _POLL_UNTIL = time.monotonic() + 30.0
    _save_toggles(face_enabled, voice_enabled)
    _log(f"Start pressed | face swap {'on' if face_enabled else 'off'} | voice {'on' if voice_enabled else 'off'}")
    if not face_enabled:
        if _SESSION is not None:
            _stop_session()
            _log("Face: stopped (face swap is switched off)")
        yield _status_text(), gr.Timer(active=True), gr.skip()
        return
    if voice_enabled:
        yield from start_live_with_voice(chunk_size_ms, cross_fade_s, *face_inputs)
    else:
        for status, timer in start_live(*face_inputs):
            yield status, timer, gr.skip()


def start_voice_if_enabled(voice_enabled, *voice_inputs):
    """Live page Start, voice side. The Realtime controls it would normally
    update are hidden on the Live page, so its updates are discarded."""
    skip = (gr.skip(), gr.skip(), gr.skip())
    if not voice_enabled:
        yield skip
        return
    for _update in voice_start_logged(*voice_inputs):
        yield skip


def stop_voice():
    rt = _realtime()
    # rt.running is set as soon as voice starts loading, before voice_status()
    # reports anything, so a Stop during model loading still stops it.
    if rt is not None and (rt.running or rt.voice_status() is not None):
        voice_stop_logged()
    return gr.skip(), gr.skip(), gr.skip()


# ─── profiles (per-person photos + voice + look) ─────────────────────────


def _profile_dropdown(value=None):
    from tabs.faceswap import profiles
    return gr.Dropdown(choices=profiles.list_profiles(), value=value)


def _make_profile_handlers(has_voice):
    """Load/save/delete handlers; voice outputs only when voice controls exist."""
    from tabs.faceswap import profiles

    n_voice = len(profiles.VOICE_KEYS) if has_voice else 0

    def load(name):
        skips = [gr.skip()] * (1 + len(profiles.FACE_KEYS) + n_voice)
        name = (name or "").strip()
        if not name or name not in profiles.list_profiles():
            return (*skips, _status_text())
        try:
            prof = profiles.load_profile(name)
        except (OSError, ValueError) as exc:
            _log(f"Profile ERROR: could not load '{name}': {exc}")
            return (*skips, _status_text())
        face_updates = [gr.update(value=prof["face"][k]) if k in prof["face"] else gr.skip()
                        for k in profiles.FACE_KEYS]
        voice_updates = []
        voice = prof["voice"]
        if has_voice:
            controls = _realtime().SESSION_CONTROLS
            choices = {"model_file": controls["list_models"](), "index_file": controls["list_indexes"]()}
            for k in profiles.VOICE_KEYS:
                if k not in voice:
                    voice_updates.append(gr.skip())
                elif k in choices:
                    voice_updates.append(gr.update(value=profiles.match_choice(voice[k], choices[k])))
                else:
                    voice_updates.append(gr.update(value=voice[k]))
        save_settings({**load_settings(), "profile": name})
        details = f"{len(prof['faces'])} photo(s)"
        if voice.get("model_file"):
            details += (f" | voice {os.path.basename(voice['model_file'])}, pitch {voice.get('pitch')}, "
                        f"index {voice.get('index_rate')}, protect {voice.get('protect')}, {voice.get('f0_method')}")
        _log(f"Profile '{name}' loaded: {details}")
        rt = _realtime()
        if rt is not None and rt.voice_status() is not None and voice.get("model_file"):
            _log("Profile: a different voice model takes effect on the next Start")
        return (gr.update(value=prof["faces"] or None), *face_updates, *voice_updates, _status_text())

    def save(name, source, *values):
        name = (name or "").strip()
        if not name:
            gr.Warning(i18n("Type a profile name first."))
            return gr.skip(), gr.skip(), _status_text()
        face = dict(zip(profiles.FACE_KEYS, values[:len(profiles.FACE_KEYS)]))
        voice = dict(zip(profiles.VOICE_KEYS, values[len(profiles.FACE_KEYS):])) if has_voice else {}
        existed = name in profiles.list_profiles()
        try:
            prof = profiles.save_profile(name, _source_paths(source), face, voice)
        except (OSError, ValueError) as exc:
            _log(f"Profile ERROR: could not save '{name}': {exc}")
            return gr.skip(), gr.skip(), _status_text()
        save_settings({**load_settings(), "profile": name})
        _log(f"Profile '{name}' {'updated' if existed else 'created'}: {len(prof['faces'])} photo(s)"
             + (f", voice {os.path.basename(voice.get('model_file') or '')}" if voice else ""))
        return _profile_dropdown(name), gr.update(value=prof["faces"] or None), _status_text()

    def delete(name):
        name = (name or "").strip()
        if not name or name not in profiles.list_profiles():
            return gr.skip(), _status_text()
        try:
            profiles.delete_profile(name)
        except (OSError, ValueError) as exc:
            _log(f"Profile ERROR: could not delete '{name}': {exc}")
            return gr.skip(), _status_text()
        save_settings({**load_settings(), "profile": ""})
        _log(f"Profile '{name}' deleted")
        return _profile_dropdown(None), _status_text()

    return load, save, delete


def live_tab():
    """One page for live face swap and live voice: settings in Video / Voice
    sub-tabs, with preview, Start/Stop and the activity log alongside."""
    from tabs.faceswap import profiles
    from tabs.realtime import realtime as realtime_ui

    saved = load_settings()
    cameras = camera_choices()
    camera_values = [value for _label, value in cameras]
    saved_camera = saved["camera"] if saved["camera"] in camera_values else (camera_values[0] if camera_values else None)
    saved_sources = [p for p in saved["source_paths"] if os.path.isfile(p)] or None

    with gr.Blocks():
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                with gr.Tabs():
                    with gr.Tab(i18n("Video")):
                        with step_card(1, "Source Face", i18n, description="Photos of the face to wear, all of the same person. 3-5 sharp, front-facing shots (eyes open, mouth closed) give a steadier likeness than one."):
                            source = gr.Gallery(label=i18n("Source face photos"), type="filepath", value=saved_sources,
                                                interactive=True, columns=4, height=200, object_fit="cover",
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
                            match_voice = gr.Button(i18n("Match voice delay"), size="sm")

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

                    with gr.Tab(i18n("Voice")):
                        realtime_ui.realtime_tab(embedded=True)

            with gr.Column(scale=2, min_width=360):
                with gr.Row(equal_height=True):
                    known = profiles.list_profiles()
                    profile = gr.Dropdown(label=i18n("Profile"), choices=known, allow_custom_value=True, scale=3,
                                          value=saved["profile"] if saved["profile"] in known else None,
                                          info=i18n("Pick a person to load their photos, voice and look. Type a new name and Save to create one."))
                    with gr.Column(scale=1, min_width=120):
                        save_profile_btn = gr.Button(i18n("Save profile"), size="sm")
                        delete_profile_btn = gr.Button(i18n("Delete"), size="sm", variant="stop")
                preview = gr.Image(label=i18n("Live preview"), interactive=False, format="jpeg", height=420)
                rt = _realtime()  # None in client mode: voice then starts from the Voice tab
                with gr.Row():
                    face_on = gr.Checkbox(label=i18n("Face swap"), value=saved["face_enabled"])
                    voice_on = gr.Checkbox(label=i18n("Voice"), value=saved["voice_enabled"] and rt is not None,
                                           interactive=rt is not None)
                with gr.Row():
                    start = gr.Button(i18n("Start"), variant="primary")
                    stop = gr.Button(i18n("Stop"))
                gr.Markdown(i18n("Start runs whatever is switched on. Voice uses the settings in the Voice tab."),
                            elem_classes="applio-card-desc")
                status = gr.Textbox(label=i18n("Status"), value=i18n("Stopped."), interactive=False,
                                    lines=12, max_lines=22, autoscroll=True, buttons=["copy"])

        timer = gr.Timer(0.2, active=False)
        quality = [swap_model, enhancer, det_size, mouth_mask, sharpness, stabilization, detect_every_frame,
                   color_correction, many_faces, mirror, async_swap, show_fps]
        face_inputs = [source, camera, resolution, virtual_camera, video_delay, *quality]

        refresh.click(lambda: gr.Dropdown(choices=camera_choices()), outputs=camera)
        video_delay.change(set_video_delay, inputs=video_delay, show_progress="hidden")
        stop.click(stop_live, outputs=[status, timer, preview])
        if rt is not None:
            controls = rt.SESSION_CONTROLS
            # One click, two listeners: the face side here and the voice side
            # through the Realtime handler (its own controls are hidden).
            start.click(start_face_if_enabled, inputs=[face_on, voice_on, *controls["timing_inputs"], *face_inputs],
                        outputs=[status, timer, video_delay])
            start.click(start_voice_if_enabled, inputs=[voice_on, *controls["start_inputs"]],
                        outputs=controls["outputs"])
            stop.click(stop_voice, outputs=controls["outputs"])
            match_voice.click(estimate_voice_delay, inputs=controls["timing_inputs"], outputs=video_delay)
        else:
            zero = gr.Number(value=0, visible=False)
            start.click(start_face_if_enabled, inputs=[face_on, voice_on, zero, zero, *face_inputs],
                        outputs=[status, timer, video_delay])
            match_voice.visible = False
        # Profiles: face photos + look settings, plus the Voice tab's voice settings.
        face_profile = [swap_model, enhancer, mouth_mask, sharpness, color_correction]
        voice_profile = ([rt.SESSION_CONTROLS["profile_inputs"][k] for k in profiles.VOICE_KEYS]
                         if rt is not None else [])
        load_profile, save_profile, delete_profile = _make_profile_handlers(rt is not None)
        profile.select(load_profile, inputs=profile, outputs=[source, *face_profile, *voice_profile, status])
        save_profile_btn.click(save_profile, inputs=[profile, source, *face_profile, *voice_profile],
                               outputs=[profile, source, status])
        delete_profile_btn.click(delete_profile, inputs=profile, outputs=[profile, status],
                                 js="(name) => (name && confirm('Delete profile \"' + name + '\"? Its saved photos are removed.')) ? name : null")

        timer.tick(poll_preview, outputs=[preview, status, timer], show_progress="hidden")
        for control in [source, *quality]:
            control.change(apply_live, inputs=[source, *quality], show_progress="hidden")
