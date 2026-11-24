"""The Live page's "on air" indicator and its panic hold.

What the person on the call receives is not what the preview shows at a
glance: the swap may be frozen on a lost face, the virtual camera may be off,
the voice may be loading or have stopped. ``on_air_html`` renders that from a
snapshot of the live state. The hold (a button, plus a global hotkey on
Windows) freezes the video sent to the call on its last frame and mutes the
converted voice until pressed again; ``HoldGate`` does the video side.
"""

from __future__ import annotations

import html
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

DEFAULT_HOTKEY = "ctrl+alt+h"


# ─── video hold ───────────────────────────────────────────────────────────


class HoldGate:
    """Frame sink wrapper: while ``is_held()``, keep sending the last frame
    it passed on (black if none yet) instead of new ones.

    It sits right in front of the virtual camera, after the lip-sync delay,
    so a hold takes effect on the next frame the call receives, and the frame
    it repeats is one the call already saw (never a raw camera frame: the
    pipeline guards those upstream).
    """

    def __init__(self, sink: Callable[[np.ndarray], None], is_held: Callable[[], bool]):
        self.sink = sink
        self._is_held = is_held
        self._last: Optional[np.ndarray] = None

    def __call__(self, frame: np.ndarray) -> None:
        if self._is_held():
            if self._last is None or self._last.shape != frame.shape:
                self._last = np.zeros_like(frame)
            self.sink(self._last)
            return
        self._last = frame
        self.sink(frame)

    @property
    def error(self):
        return getattr(self.sink, "error", None)

    @property
    def device(self):
        return getattr(self.sink, "device", None)

    def close(self) -> None:
        self._last = None
        close = getattr(self.sink, "close", None)
        if close is not None:
            close()


# ─── hotkey ───────────────────────────────────────────────────────────────

_MODIFIERS = {
    "ctrl": 0x0002,
    "control": 0x0002,
    "alt": 0x0001,
    "shift": 0x0004,
    "win": 0x0008,
}
_MOD_ORDER = ("ctrl", "alt", "shift", "win")
_NAMED_KEYS = {
    "space": 0x20,
    "pause": 0x13,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
}


@dataclass(frozen=True)
class Hotkey:
    modifiers: frozenset  # subset of _MOD_ORDER
    key: str  # "h", "f9", "space", ...
    vk: int  # Windows virtual-key code

    @property
    def label(self) -> str:
        mods = [m.capitalize() for m in _MOD_ORDER if m in self.modifiers]
        return "+".join(
            mods + [self.key.upper() if len(self.key) == 1 else self.key.capitalize()]
        )

    @property
    def win_modifiers(self) -> int:
        flags = 0
        for m in self.modifiers:
            flags |= _MODIFIERS[m]
        return flags

    def js_condition(self, event: str = "e") -> str:
        """A JavaScript expression true for a keydown ``event`` of this hotkey."""
        js_names = {
            "ctrl": "ctrlKey",
            "alt": "altKey",
            "shift": "shiftKey",
            "win": "metaKey",
        }
        parts = [
            f"{'' if mod in self.modifiers else '!'}{event}.{prop}"
            for mod, prop in js_names.items()
        ]
        key = " " if self.key == "space" else self.key
        parts.append(f"{event}.key.toLowerCase() === {key.lower()!r}")
        return " && ".join(parts)


def parse_hotkey(text: str) -> Hotkey:
    """``"ctrl+alt+h"`` -> Hotkey. Needs at least one modifier (a bare key
    would fire while typing in any app)."""
    parts = [p.strip().lower() for p in str(text or "").split("+") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"hotkey {text!r} needs a modifier and a key, e.g. ctrl+alt+h")
    *mods, key = parts
    unknown = [m for m in mods if m not in _MODIFIERS]
    if unknown:
        raise ValueError(f"unknown modifier(s) {unknown} in hotkey {text!r}")
    mods = {"ctrl" if m == "control" else m for m in mods}
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key in _NAMED_KEYS:
        vk = _NAMED_KEYS[key]
    else:
        raise ValueError(f"unknown key {key!r} in hotkey {text!r}")
    return Hotkey(frozenset(mods), key, vk)


class GlobalHotkey:
    """A system-wide hotkey (works while WhatsApp, not the browser, has focus).

    Windows only, via RegisterHotKey on a thread of its own; elsewhere
    ``start()`` reports that and the page shortcut remains.
    """

    _WM_HOTKEY = 0x0312
    _WM_QUIT = 0x0012
    _MOD_NOREPEAT = 0x4000

    def __init__(self, hotkey: Hotkey, callback: Callable[[], None]):
        self.hotkey = hotkey
        self._callback = callback
        self._thread: Optional[threading.Thread] = None
        self._thread_id = None
        self._ready = threading.Event()
        self.error: Optional[str] = None

    def start(self) -> Optional[str]:
        """Register the hotkey; None on success, else why it is unavailable."""
        if sys.platform != "win32":
            self.error = "global hotkeys are only available on Windows for now"
            return self.error
        self._thread = threading.Thread(
            target=self._run, name="applio-hold-hotkey", daemon=True
        )
        self._thread.start()
        self._ready.wait(5)
        return self.error

    def stop(self) -> None:
        if self._thread is None or self._thread_id is None:
            return
        import ctypes

        ctypes.windll.user32.PostThreadMessageW(self._thread_id, self._WM_QUIT, 0, 0)
        self._thread.join(2)
        self._thread = None

    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(
            None, 1, self.hotkey.win_modifiers | self._MOD_NOREPEAT, self.hotkey.vk
        ):
            code = ctypes.get_last_error()
            self.error = (
                f"{self.hotkey.label} is already used by another program"
                if code == 1409
                else f"RegisterHotKey failed (error {code})"
            )
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == self._WM_HOTKEY:
                    try:
                        self._callback()
                    except (
                        Exception
                    ) as exc:  # a failing callback must not kill the hotkey
                        print(f"Hold hotkey callback failed: {exc}")
        finally:
            user32.UnregisterHotKey(None, 1)


# ─── indicator ────────────────────────────────────────────────────────────

_LEVELS = (
    "off",
    "ok",
    "hold",
    "warn",
    "bad",
)  # in rising precedence for the summary pill


def video_row(face: Optional[dict], held: bool) -> tuple:
    """(level, text) for what the call's video shows. ``face`` is None when
    face swap is off, else {state, error, sending, vcam_error}."""
    if face is None:
        return "off", "Off"
    if face.get("error"):
        return "bad", f"Stopped: {face['error']}"
    if not face.get("sending"):
        why = face.get("vcam_error") or "the virtual camera is off"
        return "warn", f"Not sent to the call ({why}); preview only"
    if held:
        return "hold", "Frozen by you: the call sees the last frame"
    state = face.get("state")
    if state in ("swapping", "holding"):
        return "ok", "Swapped face"
    if state == "frozen":
        return "warn", "Frozen: face not found (the camera is never shown)"
    if state == "waiting":
        return "warn", "Black: no face swapped yet"
    return "bad", "Real camera: no face swap is applied"


def voice_row(voice: Optional[dict], held: bool) -> tuple:
    """(level, text) for what the call hears. ``voice`` is None when voice is
    off, else {status, failed, ready}."""
    if voice is None:
        return "off", "Off"
    if voice.get("failed"):
        return "bad", "Stopped: the call hears silence (press Stop, then Start)"
    if held:
        return "hold", "Muted by you"
    if not voice.get("ready"):
        return "warn", voice.get("status") or "Loading"
    return "ok", voice.get("status") or "Converting"


def on_air_html(
    face: Optional[dict],
    voice: Optional[dict],
    held: bool,
    hotkey_label: Optional[str],
    hotkey_error: Optional[str] = None,
) -> str:
    rows = [("Video", *video_row(face, held)), ("Voice", *voice_row(voice, held))]
    live = any(level != "off" for _, level, _ in rows)
    worst = max((level for _, level, _ in rows), key=_LEVELS.index)
    if held and live:
        pill_level, pill = "hold", "ON HOLD"
    elif live:
        pill_level, pill = worst, "ON AIR"
    else:
        pill_level, pill = "off", "OFF AIR"
    body = "".join(
        f'<div class="applio-onair-row"><span class="lbl">{name}</span>'
        f'<span class="applio-onair-dot {level}" aria-hidden="true"></span>'
        f'<span class="txt {level}">{html.escape(text)}</span></div>'
        for name, level, text in rows
    )
    key = html.escape(hotkey_label or "the Hold button")
    foot = (
        f"<b>{key}</b> holds: freezes the video and mutes the voice; press it again to resume."
        if hotkey_label
        else "The Hold button freezes the video and mutes the voice."
    )
    if hotkey_error:
        foot += f" Global hotkey unavailable ({html.escape(hotkey_error)}): it works only while this page has focus."
    return (
        f'<div class="applio-onair {pill_level}" role="status" aria-live="polite">'
        f'<div class="applio-onair-head"><span class="applio-onair-pill {pill_level}">{pill}</span>'
        f'<span class="applio-onair-title">What the call receives</span></div>'
        f'{body}<div class="applio-onair-foot">{foot}</div></div>'
    )
