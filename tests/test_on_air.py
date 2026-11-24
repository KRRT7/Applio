"""The Live page's on-air indicator and panic hold (tabs/faceswap/on_air.py
and its wiring in tabs/faceswap/faceswap.py)."""

import os
import sys
import threading
import types

import numpy as np
import pytest

from tabs.faceswap import on_air


def _frame(v):
    return np.full((4, 4, 3), v, dtype=np.uint8)


# ─── video hold ───────────────────────────────────────────────────────────


def test_hold_gate_repeats_the_last_frame_the_call_saw():
    sent, held = [], [False]
    gate = on_air.HoldGate(lambda f: sent.append(int(f[0, 0, 0])), lambda: held[0])
    gate(_frame(1))
    gate(_frame(2))
    held[0] = True
    gate(_frame(3))
    gate(_frame(4))
    held[0] = False
    gate(_frame(5))
    assert sent == [1, 2, 2, 2, 5]


def test_hold_gate_sends_black_when_held_before_any_frame():
    sent = []
    gate = on_air.HoldGate(lambda f: sent.append(f), lambda: True)
    gate(_frame(9))
    assert not sent[0].any()


def test_hold_gate_passes_sink_status_and_close_through():
    closed = []
    sink = types.SimpleNamespace(
        error="no camera", device="OBS", close=lambda: closed.append(True)
    )
    sink.__call__ = None
    gate = on_air.HoldGate(sink, lambda: False)
    assert gate.error == "no camera" and gate.device == "OBS"
    gate.close()
    assert closed == [True]


# ─── hotkey ───────────────────────────────────────────────────────────────


def test_parse_hotkey():
    hk = on_air.parse_hotkey("Ctrl+Alt+H")
    assert hk.label == "Ctrl+Alt+H" and hk.vk == ord("H") and hk.win_modifiers == 0x3
    assert on_air.parse_hotkey("shift+ctrl+f9").label == "Ctrl+Shift+F9"
    assert on_air.parse_hotkey("ctrl+f9").vk == 0x78
    assert on_air.parse_hotkey("alt+space").vk == 0x20
    for bad in ("h", "ctrl", "hyper+h", "ctrl+f25", "ctrl+??"):
        with pytest.raises(ValueError):
            on_air.parse_hotkey(bad)


def test_hotkey_js_condition_requires_exactly_its_modifiers():
    cond = on_air.parse_hotkey("ctrl+alt+h").js_condition("e")
    assert cond == (
        "e.ctrlKey && e.altKey && !e.shiftKey && !e.metaKey && "
        "e.key.toLowerCase() === 'h'"
    )
    assert "e.metaKey" in on_air.parse_hotkey("win+f9").js_condition("e").split(" && ")


def test_global_hotkey_reports_unsupported_platforms(monkeypatch):
    monkeypatch.setattr(on_air.sys, "platform", "darwin")
    listener = on_air.GlobalHotkey(on_air.parse_hotkey("ctrl+alt+h"), lambda: None)
    assert "only available on Windows" in listener.start()


@pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("APPLIO_INPUT_TESTS"),
    reason="presses keys on this desktop: Windows, APPLIO_INPUT_TESTS=1",
)
def test_global_hotkey_fires_from_any_window():
    import ctypes

    hit = threading.Event()
    listener = on_air.GlobalHotkey(on_air.parse_hotkey("ctrl+alt+shift+f24"), hit.set)
    assert listener.start() is None
    try:
        user32 = ctypes.windll.user32
        keys = (0x11, 0x12, 0x10, 0x87)  # ctrl, alt, shift, F24
        for vk in keys:
            user32.keybd_event(vk, 0, 0, 0)
        for vk in reversed(keys):
            user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP
        assert hit.wait(3)
    finally:
        listener.stop()


# ─── indicator ────────────────────────────────────────────────────────────

FACE_OK = {"state": "swapping", "error": None, "sending": True, "vcam_error": None}
VOICE_OK = {"status": "voice on · 12 ms", "failed": False, "ready": True}


def test_off_air_when_nothing_runs():
    page = on_air.on_air_html(None, None, False, "Ctrl+Alt+H")
    assert "OFF AIR" in page and 'class="applio-onair off"' in page


def test_on_air_green_when_swapping_and_converting():
    page = on_air.on_air_html(FACE_OK, VOICE_OK, False, "Ctrl+Alt+H")
    assert (
        "ON AIR" in page
        and 'class="applio-onair ok"' in page
        and "Swapped face" in page
    )


def test_on_hold_shows_frozen_video_and_muted_voice():
    page = on_air.on_air_html(FACE_OK, VOICE_OK, True, "Ctrl+Alt+H")
    assert "ON HOLD" in page and "Frozen by you" in page and "Muted by you" in page


@pytest.mark.parametrize(
    "face, level, text",
    [
        ({**FACE_OK, "state": "holding"}, "ok", "Swapped face"),
        ({**FACE_OK, "state": "frozen"}, "warn", "face not found"),
        ({**FACE_OK, "state": "waiting"}, "warn", "Black"),
        ({**FACE_OK, "state": "off"}, "bad", "Real camera"),
        ({**FACE_OK, "sending": False}, "warn", "virtual camera is off"),
        (
            {**FACE_OK, "sending": False, "vcam_error": "OBS missing"},
            "warn",
            "OBS missing",
        ),
        ({**FACE_OK, "error": "camera unplugged"}, "bad", "camera unplugged"),
    ],
)
def test_video_states(face, level, text):
    got_level, got_text = on_air.video_row(face, held=False)
    assert got_level == level and text in got_text


def test_a_failed_voice_wins_over_a_hold_and_turns_the_pill_red():
    assert on_air.voice_row({**VOICE_OK, "failed": True}, held=True)[0] == "bad"
    page = on_air.on_air_html(FACE_OK, {**VOICE_OK, "failed": True}, False, None)
    assert 'class="applio-onair bad"' in page


def test_loading_voice_is_amber_and_errors_are_escaped():
    assert on_air.voice_row(
        {"status": "loading voice model", "failed": False, "ready": False}, False
    ) == ("warn", "loading voice model")
    page = on_air.on_air_html(
        {**FACE_OK, "error": "<b>boom</b>"}, None, False, None, "taken & busy"
    )
    assert (
        "<b>boom</b>" not in page
        and "&lt;b&gt;boom" in page
        and "taken &amp; busy" in page
    )


# ─── wiring in the Live page ──────────────────────────────────────────────


@pytest.fixture
def live_page(monkeypatch):
    from tabs.faceswap import faceswap as fs

    callbacks = types.SimpleNamespace(muted=False)
    rt = types.SimpleNamespace(
        callbacks=callbacks,
        SESSION_CONTROLS={"x": 1},
        voice_status=lambda: None,
        voice_stats=lambda: None,
    )
    monkeypatch.setattr(fs, "_realtime", lambda: rt)
    monkeypatch.setitem(fs._HOLD, "on", False)
    fs._LOG.clear()
    yield fs, callbacks
    fs._HOLD["on"] = False
    fs._LOG.clear()


def test_hold_mutes_the_voice_and_freezes_the_gate(live_page):
    fs, callbacks = live_page
    fs.toggle_hold("test")
    assert fs._is_held() and callbacks.muted
    assert any("HOLD on (test)" in line for line in fs._LOG)
    fs.toggle_hold("test")
    assert not fs._is_held() and not callbacks.muted
    assert any("HOLD off (test)" in line for line in fs._LOG)


def test_a_voice_started_during_a_hold_is_muted_on_the_next_tick(live_page):
    fs, callbacks = live_page
    fs.set_hold(True, "test")
    callbacks.muted = False  # a fresh AudioCallbacks
    fs._apply_voice_hold()
    assert callbacks.muted


def test_stop_ends_the_hold(live_page):
    fs, callbacks = live_page
    fs.set_hold(True, "test")
    fs.stop_live()
    assert not fs._is_held() and not callbacks.muted
