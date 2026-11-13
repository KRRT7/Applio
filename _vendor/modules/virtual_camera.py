"""Send processed frames to a system virtual camera via pyvirtualcam.

On macOS this targets the OBS Virtual Camera (install OBS and start its
virtual camera once so macOS approves the extension; OBS itself does not need
to be running).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from _vendor.modules.status import update_status


class VirtualCameraSink:
    """Frame sink: call it with BGR frames.

    The device is opened lazily on the first frame (so its size matches what
    the pipeline produces) and reopened if the size changes. Any failure
    disables the sink and records the reason in ``error`` instead of raising,
    so a missing driver never breaks the live loop.
    """

    def __init__(self, fps: float = 30.0, device: Optional[str] = None):
        self.fps = fps
        self.device_name = device
        self.error: Optional[str] = None
        self._cam = None

    @property
    def enabled(self) -> bool:
        return self.error is None

    @property
    def device(self) -> Optional[str]:
        return self._cam.device if self._cam is not None else None

    def __call__(self, bgr_frame: np.ndarray) -> None:
        if self.error is not None:
            return
        h, w = bgr_frame.shape[:2]
        try:
            import pyvirtualcam

            if self._cam is not None and (self._cam.width, self._cam.height) != (w, h):
                self.close()
            if self._cam is None:
                self._cam = pyvirtualcam.Camera(
                    width=w,
                    height=h,
                    fps=max(1, round(self.fps)),
                    fmt=pyvirtualcam.PixelFormat.BGR,
                    device=self.device_name,
                )
                update_status(f"Virtual camera: {self._cam.device} {w}x{h}", "DLC.VCAM")
            self._cam.send(np.ascontiguousarray(bgr_frame))
        except Exception as exc:  # missing package, no device, device busy
            self.error = str(exc) or exc.__class__.__name__
            self.close()
            update_status(f"Virtual camera unavailable: {self.error}", "DLC.VCAM")

    def close(self) -> None:
        cam, self._cam = self._cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
