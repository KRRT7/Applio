"""Enumerate capture devices as (index, name) pairs usable with VideoCapturer."""

from __future__ import annotations

import platform
from typing import List, Tuple

import cv2

Camera = Tuple[int, str]


def list_cameras() -> List[Camera]:
    """Return available cameras, real devices first.

    Virtual cameras (e.g. "OBS Virtual Camera") are sorted last: AVFoundation
    often lists them first, and using one as the input while also sending to
    it loops the output back in.
    """
    system = platform.system()
    if system == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph

            from _vendor.modules.wincom import ensure_com

            ensure_com()  # pygrabber uses COM; may run on a worker thread
            cams = list(enumerate(FilterGraph().get_input_devices()))
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            cams = []
    elif system == "Darwin":
        try:
            from cv2_enumerate_cameras import enumerate_cameras

            cams = [(c.index, c.name) for c in enumerate_cameras(cv2.CAP_AVFOUNDATION)]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            cams = [(0, "Camera 0"), (1, "Camera 1")]
    else:
        cams = []
        for i in range(10):
            cap = cv2.VideoCapture(f"/dev/video{i}")
            if cap.isOpened():
                cams.append((i, f"Camera {i}"))
                cap.release()
    cams.sort(key=lambda c: is_virtual_camera(c[1]))
    return cams


def is_virtual_camera(name: str) -> bool:
    return "virtual" in name.lower()
