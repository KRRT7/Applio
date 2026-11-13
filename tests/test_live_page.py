"""Tests for the Live page logic (tabs/faceswap/faceswap.py).

The realtime voice module is replaced by small fakes, so no audio device,
camera or model is used. Includes regressions for bugs found by hand.
"""
import types

import pytest

from tabs.faceswap import faceswap as fs


@pytest.fixture
def clean_log():
    fs._LOG.clear()
    fs._VOICE_WATCH.clear()
    yield fs._LOG
    fs._LOG.clear()
    fs._VOICE_WATCH.clear()


def _fake_realtime(running=False, status=None, stats=None, stopped=None):
    rt = types.SimpleNamespace(running=running, SESSION_CONTROLS={})
    rt.voice_status = lambda: status
    rt.voice_stats = lambda: stats

    def stop_fn():
        if stopped is not None:
            stopped.append(True)
        return ("Realtime stopped.", None, None)

    rt.SESSION_CONTROLS = {"stop_fn": stop_fn}
    return rt


def test_estimate_voice_delay():
    assert fs.estimate_voice_delay(250, 0.05) == 260   # 250 + 50 + 25 - 65
    assert fs.estimate_voice_delay(0, 0) == 0           # never negative
    assert fs.estimate_voice_delay(None, None) == 0


def test_source_paths_accepts_gallery_item_shapes(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    gallery = [(str(a), None), [str(b), "caption"], {"path": str(a)}, (str(tmp_path / "missing.jpg"), None)]
    assert fs._source_paths(gallery) == [str(a), str(b), str(a)]
    assert fs._source_paths(None) == []


def test_stop_stops_voice_while_its_model_is_still_loading(monkeypatch, clean_log):
    """Regression: voice_status() is None early in loading, but Stop must still stop it."""
    stopped = []
    monkeypatch.setattr(fs, "_realtime", lambda: _fake_realtime(running=True, status=None, stopped=stopped))
    fs.stop_voice()
    assert stopped == [True]


def test_stop_does_nothing_when_voice_is_off(monkeypatch, clean_log):
    stopped = []
    monkeypatch.setattr(fs, "_realtime", lambda: _fake_realtime(running=False, status=None, stopped=stopped))
    fs.stop_voice()
    assert stopped == []


def _stats(mic, out, **kw):
    base = dict(input_db=mic, output_db=out, latency_ms=3.0, block_ms=250.0, blocks=10, overruns=0,
                input_overflows=0, output_underflows=0, callback_errors=0, last_error=None)
    base.update(kw)
    return base


def test_silent_mic_warning_after_five_seconds_then_recovery(monkeypatch, clean_log):
    now = [100.0]
    monkeypatch.setattr(fs.time, "monotonic", lambda: now[0])
    stats = {"v": _stats(-80, -90)}
    monkeypatch.setattr(fs, "_realtime", lambda: _fake_realtime(running=True, status="voice on", stats=stats["v"]))
    fs._watch_voice()
    now[0] += 4.9
    fs._watch_voice()
    assert not any("no sound from the mic" in line for line in clean_log)
    now[0] += 0.2
    fs._watch_voice()
    assert sum("no sound from the mic" in line for line in clean_log) == 1
    stats["v"] = _stats(-30, -28)
    fs._watch_voice()
    assert "mic sound detected" in clean_log[-1]


def test_speaking_with_silent_output_points_at_vad(monkeypatch, clean_log):
    now = [0.0]
    monkeypatch.setattr(fs.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(fs, "_realtime", lambda: _fake_realtime(running=True, status="voice on", stats=_stats(-25, -90)))
    for _ in range(4):
        fs._watch_voice()
        now[0] += 1.1
    assert sum("Enable VAD" in line for line in clean_log) == 1


def test_room_noise_with_silent_output_is_not_a_warning(monkeypatch, clean_log):
    """VAD correctly mutes quiet room noise (~-40 dB); that must not warn."""
    now = [0.0]
    monkeypatch.setattr(fs.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(fs, "_realtime", lambda: _fake_realtime(running=True, status="voice on", stats=_stats(-40, -90)))
    for _ in range(6):
        fs._watch_voice()
        now[0] += 1.0
    assert not any("Enable VAD" in line for line in clean_log)


def test_dropouts_and_overruns_are_reported_once_per_batch(monkeypatch, clean_log):
    now = [0.0]
    monkeypatch.setattr(fs.time, "monotonic", lambda: now[0])
    stats = {"v": _stats(-30, -28)}
    monkeypatch.setattr(fs, "_realtime", lambda: _fake_realtime(running=True, status="voice on", stats=stats["v"]))
    fs._watch_voice()
    stats["v"] = _stats(-30, -28, overruns=2, output_underflows=1, latency_ms=300.0)
    fs._watch_voice()
    fs._watch_voice()
    warnings = [line for line in clean_log if "Audio may stutter" in line]
    assert len(warnings) == 1 and "2 block(s)" in warnings[0] and "1 output dropout" in warnings[0]


def test_voice_progress_counters_are_logged_once(monkeypatch, clean_log):
    updates = ["Starting Realtime...", "Loading model...", "Loading model... (1s)", "Loading model... (2s)",
               "Realtime is starting!", "Warming up... (3 blocks) ███", "Warming up... (2 blocks) ███",
               "Latency: 2.00 ms | Volume: 0 dB", "Latency: 3.00 ms | Volume: 0 dB"]

    class Comp:
        def __init__(self, label):
            self.label = label

    rt = _fake_realtime(running=True)
    rt.SESSION_CONTROLS.update(start_inputs=[Comp("Pitch")], start_fn=lambda *a: iter((u, None, None) for u in updates))
    monkeypatch.setattr(fs, "_realtime", lambda: rt)
    list(fs.voice_start_logged(10))
    voice_lines = [line.split("  ", 1)[1] for line in clean_log if "Voice:" in line]
    assert voice_lines[0] == "Voice: starting | Pitch: 10"
    assert sum("Loading model" in line for line in voice_lines) == 1
    assert sum("Warming up" in line for line in voice_lines) == 1
    assert sum("running" in line for line in voice_lines) == 1
    assert voice_lines[-1] == "Voice: stopped"


def test_face_off_start_does_not_open_a_camera(monkeypatch, isolated_config, clean_log):
    monkeypatch.setattr(fs, "_realtime", lambda: None)
    monkeypatch.setattr(fs, "start_live", lambda *a: pytest.fail("face started although switched off"))
    out = list(fs.start_face_if_enabled(False, True, 250, 0.05))
    assert len(out) == 1 and any("face swap off" in line for line in clean_log)
    assert fs.load_settings()["face_enabled"] is False  # toggles are remembered
