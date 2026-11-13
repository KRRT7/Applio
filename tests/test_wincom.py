"""Tests for the per-thread COM helper used by the Windows camera code."""
import sys
import threading
import types
from unittest.mock import patch

from _vendor.modules import wincom


def _fresh_state(monkeypatch):
    monkeypatch.setattr(wincom, "_STATE", threading.local())


def test_noop_off_windows(monkeypatch):
    _fresh_state(monkeypatch)
    calls = []
    fake = types.SimpleNamespace(COINIT_APARTMENTTHREADED=2, CoInitializeEx=lambda m: calls.append(m))
    with patch.object(wincom.platform, "system", lambda: "Darwin"), patch.dict(sys.modules, {"comtypes": fake}):
        wincom.ensure_com()
    assert calls == []


def test_initialises_once_per_thread_on_windows(monkeypatch):
    _fresh_state(monkeypatch)
    calls = []
    fake = types.SimpleNamespace(COINIT_APARTMENTTHREADED=2, CoInitializeEx=lambda m: calls.append(m))
    with patch.object(wincom.platform, "system", lambda: "Windows"), patch.dict(sys.modules, {"comtypes": fake}):
        wincom.ensure_com()
        wincom.ensure_com()  # same thread: no second call
        t = threading.Thread(target=wincom.ensure_com)
        t.start()
        t.join()  # another thread: initialises again
    assert calls == [2, 2]


def test_already_initialised_differently_is_fine(monkeypatch):
    _fresh_state(monkeypatch)

    def changed_mode(_m):
        raise OSError("RPC_E_CHANGED_MODE")

    fake = types.SimpleNamespace(COINIT_APARTMENTTHREADED=2, CoInitializeEx=changed_mode)
    with patch.object(wincom.platform, "system", lambda: "Windows"), patch.dict(sys.modules, {"comtypes": fake}):
        wincom.ensure_com()  # must not raise
    assert wincom._STATE.ready
