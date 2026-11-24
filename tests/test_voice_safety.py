"""What the listener hears when realtime voice falls behind or fails.

The converted voice must never stutter (replay a block), click, loop or go
quiet without the operator being told. The worker process is faked here;
no model or audio device is used.
"""

import numpy as np
import pytest

from rvc.realtime import callbacks as cb_mod
from rvc.realtime.audio import Audio
from rvc.realtime.callbacks import AudioCallbacks
from rvc.realtime.core import gate_with_hangover, speech_hangover_blocks

N = 480  # 10 ms at 48 kHz
FADE = int(48000 * cb_mod.EDGE_FADE_S)


class _FakeProcess:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


class _FakeWorker:
    """Hands out queued results; None means "not ready in time"."""

    def __init__(self):
        self.results, self.submitted, self.last_skipped = [], 0, 0
        self._process = _FakeProcess()

    def submit(self, audio, params):
        self.submitted += 1

    def retrieve(self):
        if not self.results:
            return None
        item = self.results.pop(0)
        if item is None:
            return None
        audio, self.last_skipped = item
        return audio, 0.1, 5.0, 0


def _callbacks():
    cb = AudioCallbacks.__new__(AudioCallbacks)
    cb.pass_through = False
    cb._last_output = None
    cb._last_vol = 0
    cb._fade = FADE
    cb._in_gap = False
    cb.late_blocks = 0
    cb.voice_failed = False
    cb.vc = _FakeWorker()
    return cb


def _block(value):
    return np.full(N, value, dtype=np.float32)


def _play(cb, *results):
    cb.vc.results = list(results)
    return [cb.change_voice(np.zeros(N, dtype=np.float32))[0] for _ in results]


def test_a_late_block_fades_to_silence_instead_of_replaying():
    cb = _callbacks()
    first, late = _play(cb, (_block(0.5), 0), None)
    assert np.all(first == 0.5)
    assert (
        late[0] == pytest.approx(0.5) and late[FADE - 1] < 0.01
    )  # ramps down from the last sample
    assert np.all(late[FADE:] == 0)  # then silence, not the old block
    assert cb.late_blocks == 1


def test_audio_resumes_with_a_fade_in_after_a_gap():
    cb = _callbacks()
    _, _, resumed, steady = _play(
        cb, (_block(0.5), 0), None, (_block(0.5), 0), (_block(0.5), 0)
    )
    assert resumed[0] == 0 and resumed[FADE // 2] == pytest.approx(0.25, abs=0.01)
    assert np.all(resumed[FADE:] == 0.5)
    assert np.all(steady == 0.5)  # no fade once audio is flowing


def test_a_dropped_result_is_smoothed_over():
    cb = _callbacks()
    _, after_skip = _play(cb, (_block(0.5), 0), (_block(-0.5), 1))
    assert after_skip[0] == 0 and np.all(after_skip[FADE:] == -0.5)


def test_muting_fades_out_keeps_converting_and_resumes_with_a_fade_in():
    cb = _callbacks()
    _play(cb, (_block(0.5), 0))
    cb.muted = True
    submitted = cb.vc.submitted
    first, second = _play(cb, (_block(0.5), 0), (_block(0.5), 0))
    assert (
        first[0] == pytest.approx(0.5)
        and np.all(first[FADE:] == 0)
        and np.all(second == 0)
    )
    assert (
        cb.vc.submitted == submitted + 2 and cb.late_blocks == 0
    )  # the worker keeps its context
    cb.muted = False
    (resumed,) = _play(cb, (_block(0.5), 0))
    assert resumed[0] == 0 and np.all(resumed[FADE:] == 0.5)


def test_a_dead_worker_means_silence_and_a_flag_never_a_loop():
    cb = _callbacks()
    _play(cb, (_block(0.5), 0))
    cb.vc._process.alive = False
    submitted = cb.vc.submitted
    outs = _play(cb, (_block(0.9), 0), (_block(0.9), 0))
    assert cb.voice_failed
    assert cb.vc.submitted == submitted  # nothing sent to the dead process
    assert np.all(outs[0][FADE:] == 0) and np.all(outs[1] == 0)


def test_nothing_ready_before_the_first_block_is_plain_silence():
    cb = _callbacks()
    (out,) = _play(cb, None)
    assert (
        np.all(out == 0) and cb.late_blocks == 0
    )  # still loading: not counted as late


# ─── silence gate hangover ────────────────────────────────────────────────


def test_hangover_is_about_300_ms_of_blocks():
    assert speech_hangover_blocks(4000) == 1  # 250 ms blocks
    assert speech_hangover_blocks(1600) == 3  # 100 ms blocks
    assert speech_hangover_blocks(16000) == 1  # never zero


def test_gate_stays_open_through_a_short_dip_and_closes_after_the_hangover():
    verdicts = [
        False,
        True,
        False,
        True,
        True,
        True,
        False,
    ]  # True = detector says silence
    hold, out = 0, []
    for silent in verdicts:
        is_silence, hold = gate_with_hangover(silent, hold, hangover_blocks=2)
        out.append(is_silence)
    assert out == [False, False, False, False, False, True, False]


# ─── audio stream callbacks ───────────────────────────────────────────────


def test_a_failing_audio_callback_outputs_silence_not_the_stale_buffer():
    a = Audio.__new__(Audio)
    a.running = True
    a._reset_stats()
    a.block_ms = 10.0
    a.input_audio_gain = a.output_audio_gain = 1.0
    a.use_monitor = False

    def boom(_indata):
        raise RuntimeError("model fell over")

    a.process_data = boom
    outdata = np.full((N, 2), 0.7, dtype=np.float32)
    a.audio_stream_callback(np.zeros((N, 1), dtype=np.float32), outdata, N, None, None)
    assert np.all(outdata == 0) and a.stats()["callback_errors"] == 1


def test_stats_report_late_blocks_and_voice_failure():
    a = Audio.__new__(Audio)
    a._reset_stats()
    a.callbacks = _callbacks()
    a.callbacks.late_blocks, a.callbacks.voice_failed = 3, True
    st = a.stats()
    assert st["late_blocks"] == 3 and st["voice_failed"] is True
