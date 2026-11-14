"""Windows real-time scheduling hints for the Live audio and video loops.

By default Windows schedules Applio like any background app: the voice worker
and face-swap threads run at normal priority, compete with the browser, OBS or
a game, and on Windows 11 can be moved to efficiency cores (EcoQoS) when the
window loses focus. That shows up as audio dropouts and frame stalls.

Three hints fix that, the same ones DAWs and capture tools use:

* ``boost_thread(task)`` registers the calling thread with the Multimedia
  Class Scheduler Service (MMCSS), which raises its priority while it runs and
  still reserves CPU for the rest of the system (``SystemResponsiveness``).
* ``boost_process()`` raises the process to ABOVE_NORMAL priority and opts it
  out of power throttling (EcoQoS).
* ``restore_process(state)`` undoes ``boost_process``.

Every function is a no-op off Windows and never raises: a failed hint must not
stop a live session.
"""

import ctypes
import sys
import threading

IS_WINDOWS = sys.platform == "win32"

# MMCSS task names (HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\
# Multimedia\SystemProfile\Tasks). These ship with every Windows install.
PRO_AUDIO = "Pro Audio"
CAPTURE = "Capture"
GAMES = "Games"

_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_POWER_THROTTLING_CURRENT_VERSION = 1
_POWER_THROTTLING_EXECUTION_SPEED = 0x1

_STATE = threading.local()


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def _kernel32():
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    k32.GetPriorityClass.restype = wintypes.DWORD
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    if hasattr(k32, "SetProcessInformation"):  # Windows 8+
        k32.SetProcessInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        k32.SetProcessInformation.restype = wintypes.BOOL
    return k32


def boost_thread(task: str = PRO_AUDIO) -> bool:
    """Register the calling thread with MMCSS under ``task``, once per thread.

    Cheap to call repeatedly (e.g. at the top of an audio callback): after the
    first call it is a thread-local attribute check. Returns True if the
    thread is registered.
    """
    if not IS_WINDOWS:
        return False
    done = getattr(_STATE, "mmcss", None)
    if done is not None:
        return done
    ok = False
    try:
        from ctypes import wintypes

        avrt = ctypes.WinDLL("avrt", use_last_error=True)
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        index = wintypes.DWORD(0)
        # The handle is released by Windows when the thread exits.
        ok = bool(avrt.AvSetMmThreadCharacteristicsW(task, ctypes.byref(index)))
    except (OSError, AttributeError, ctypes.ArgumentError):
        ok = False
    _STATE.mmcss = ok
    return ok


def boost_process():
    """Raise this process to ABOVE_NORMAL and opt it out of EcoQoS.

    Returns the previous priority class to pass to ``restore_process``, or
    None if nothing was changed.
    """
    if not IS_WINDOWS:
        return None
    try:
        k32 = _kernel32()
        proc = k32.GetCurrentProcess()
        previous = k32.GetPriorityClass(proc) or None
        if previous is not None and previous != _ABOVE_NORMAL_PRIORITY_CLASS:
            k32.SetPriorityClass(proc, _ABOVE_NORMAL_PRIORITY_CLASS)
        _set_power_throttling(k32, proc, throttle=False)
        return previous
    except (OSError, AttributeError, ctypes.ArgumentError):
        return None


def restore_process(previous) -> None:
    """Undo ``boost_process``: restore the priority class and EcoQoS default."""
    if not IS_WINDOWS or previous is None:
        return
    try:
        k32 = _kernel32()
        proc = k32.GetCurrentProcess()
        k32.SetPriorityClass(proc, previous)
        _set_power_throttling(k32, proc, throttle=None)
    except (OSError, AttributeError, ctypes.ArgumentError):
        pass


def _set_power_throttling(k32, proc, throttle) -> None:
    """throttle=False opts out of EcoQoS; None hands the decision back to Windows."""
    state = _PowerThrottlingState(
        _POWER_THROTTLING_CURRENT_VERSION,
        0 if throttle is None else _POWER_THROTTLING_EXECUTION_SPEED,
        _POWER_THROTTLING_EXECUTION_SPEED if throttle else 0,
    )
    if not hasattr(k32, "SetProcessInformation"):
        return  # pre-Windows 8: no EcoQoS to opt out of
    k32.SetProcessInformation(
        proc,
        _PROCESS_POWER_THROTTLING,
        ctypes.byref(state),
        ctypes.sizeof(state),
    )
