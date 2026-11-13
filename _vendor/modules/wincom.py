"""Per-thread COM initialisation for Windows camera code.

pygrabber (DirectShow, used to list and open cameras on Windows) talks to
COM, which must be initialised on every thread that uses it. Embedding
apps call the camera code from worker threads (e.g. Gradio handlers),
where nothing has done that, and pygrabber fails with
``[WinError -2147221008] CoInitialize has not been called``.
"""

import platform
import threading

_STATE = threading.local()


def ensure_com() -> None:
    """Initialise COM for the calling thread once (no-op off Windows)."""
    if platform.system() != "Windows" or getattr(_STATE, "ready", False):
        return
    try:
        import comtypes

        comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
    except OSError:
        # Already initialised on this thread with another threading model
        # (RPC_E_CHANGED_MODE): COM is usable, which is all we need.
        pass
    _STATE.ready = True
