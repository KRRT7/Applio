"""Status reporting with no UI dependency.

Processors call update_status(); whoever embeds Deep-Live-Cam (the CLI, a
Gradio app, ...) can subscribe with add_status_listener() to show messages.
"""

from __future__ import annotations

import threading
from typing import Callable, List

StatusListener = Callable[[str, str], None]

_LISTENERS: List[StatusListener] = []
_LOCK = threading.Lock()


def update_status(message: str, scope: str = "DLC.CORE") -> None:
    print(f"[{scope}] {message}")
    with _LOCK:
        listeners = list(_LISTENERS)
    for listener in listeners:
        try:
            listener(message, scope)
        except Exception:
            pass


def add_status_listener(listener: StatusListener) -> None:
    with _LOCK:
        if listener not in _LISTENERS:
            _LISTENERS.append(listener)


def remove_status_listener(listener: StatusListener) -> None:
    with _LOCK:
        if listener in _LISTENERS:
            _LISTENERS.remove(listener)
