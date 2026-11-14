import sys
import threading

import psutil
import pytest

from rvc.lib import win_rt

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows scheduling APIs")


def _in_thread(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join()
    return out[0]


@windows_only
def test_boost_thread_registers_with_mmcss_once_per_thread():
    def run():
        first = win_rt.boost_thread(win_rt.PRO_AUDIO)
        return first, win_rt._STATE.mmcss, win_rt.boost_thread(win_rt.PRO_AUDIO)

    assert _in_thread(run) == (True, True, True)
    # Each live task name exists on a stock Windows install.
    for task in (win_rt.CAPTURE, win_rt.GAMES):
        assert _in_thread(lambda: win_rt.boost_thread(task))


@windows_only
def test_unknown_task_fails_softly_and_is_cached():
    def run():
        return win_rt.boost_thread("No Such Task"), win_rt._STATE.mmcss

    assert _in_thread(run) == (False, False)


@windows_only
def test_boost_and_restore_process_priority():
    proc = psutil.Process()
    before = proc.nice()
    previous = win_rt.boost_process()
    try:
        assert previous is not None
        assert proc.nice() == psutil.ABOVE_NORMAL_PRIORITY_CLASS
    finally:
        win_rt.restore_process(previous)
    assert proc.nice() == before


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(win_rt, "IS_WINDOWS", False)
    assert _in_thread(lambda: win_rt.boost_thread()) is False
    assert win_rt.boost_process() is None
    win_rt.restore_process(0x8000)  # must not raise
