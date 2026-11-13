"""Tests for the realtime audio diagnostics (rvc/realtime/audio.py) and the
voice status line built from them (tabs/realtime/realtime.py)."""
import types

import numpy as np
import pytest

from rvc.realtime.audio import Audio


@pytest.fixture
def audio():
    """An Audio manager without devices or a model: process_data is faked."""
    a = Audio.__new__(Audio)
    a.input_audio_gain = 1.0
    a.running = True
    a._reset_stats()
    a.block_ms = 250.0
    a.process_data = lambda indata: (indata[:, 0] * 0.5, 0.0, 3.0)  # (out, vol, ms)
    return a


def test_levels_blocks_and_overruns(audio):
    tone = (0.1 * np.sin(np.linspace(0, 200, 12000))).astype(np.float32)[:, None]
    audio.process_data_with_time(tone)
    st = audio.stats()
    assert st["blocks"] == 1 and st["overruns"] == 0
    assert -24 < st["input_db"] < -22          # 0.1 amplitude sine: ~-23 dB RMS
    assert st["output_db"] == pytest.approx(st["input_db"] - 6.02, abs=0.1)  # fake halves it
    audio.process_data = lambda indata: (indata[:, 0], 0.0, 300.0)  # slower than the 250 ms block
    audio.process_data_with_time(tone)
    assert audio.stats()["overruns"] == 1


def test_silence_reads_as_minus_infinity(audio):
    audio.process_data_with_time(np.zeros((1000, 1), np.float32))
    assert audio.stats()["input_db"] == float("-inf")


def test_driver_flags_and_callback_errors_are_counted(audio):
    flags = type("Flags", (), {"input_overflow": True, "output_underflow": True, "__bool__": lambda self: True})()
    audio._note_status(flags)
    audio._note_status(None)
    audio.out_queue = None
    audio.use_monitor = False

    def boom(_indata):
        raise RuntimeError("model crashed")

    audio.process_data = boom
    audio.audio_stream_no_output_callback(np.zeros((10, 1), np.float32), 10, None, None)
    st = audio.stats()
    assert st["input_overflows"] == 1 and st["output_underflows"] == 1
    assert st["callback_errors"] == 1 and "model crashed" in st["last_error"]


def test_voice_status_line_formatting(monkeypatch):
    from tabs.realtime import realtime as rt

    stats = dict(input_db=-32.0, output_db=float("-inf"), latency_ms=4.0, block_ms=250.0, blocks=5,
                 overruns=1, input_overflows=0, output_underflows=2, callback_errors=0, last_error=None)
    am = types.SimpleNamespace(stats=lambda: stats, latency=4.0)
    cb = types.SimpleNamespace(vc=types.SimpleNamespace(ready=True, vc_model=types.SimpleNamespace(warmup_blocks=0)))
    monkeypatch.setattr(rt, "running", True)
    monkeypatch.setattr(rt, "audio_manager", am)
    monkeypatch.setattr(rt, "callbacks", cb)
    line = rt.voice_status()
    assert line == "voice on · mic -32 dB ▮▮▮▯▯▯ · out silent ▯▯▯▯▯▯ · 4/250 ms per block · overruns 1 · dropouts 2"
    monkeypatch.setattr(rt, "running", False)
    assert rt.voice_status() is None and rt.voice_stats() is None
