"""Public API for embedding Deep-Live-Cam in another application.

    from _vendor import deep_live_cam as dlc

    dlc.configure(dlc.LiveConfig(source_path="face.jpg", swap_model="hyperswap_1a_256"))
    dlc.ensure_models()                       # downloads missing models
    dlc.preload_models()                      # optional: load/compile now, not on frame 1
    cams = dlc.list_cameras()                 # [(index, name), ...] real cameras first
    session = dlc.LiveSession(cams[0][0], sinks=[dlc.VirtualCameraSink()])
    session.start()
    frame = session.latest_frame              # newest swapped BGR frame, for previews
    session.stop()

    out = dlc.swap_image(bgr_image)           # one-off swap of a single image

Settings are process-wide (one configuration and one live session per
process); configure() can be called again while a session runs and takes
effect on the next frame. Subscribe to progress/errors with
add_status_listener(fn(message, scope)).
"""

from _vendor.modules.cameras import is_virtual_camera, list_cameras
from _vendor.modules.execution import suggest_default_execution_provider, suggest_execution_providers
from _vendor.modules.live import (
    ENHANCERS,
    SWAP_MODELS,
    FramePipeline,
    LiveConfig,
    LiveSession,
    configure,
    ensure_models,
    preload_models,
    swap_image,
)
from _vendor.modules.metadata import version as __version__
from _vendor.modules.paths import MODELS_DIR
from _vendor.modules.status import add_status_listener, remove_status_listener
from _vendor.modules.virtual_camera import VirtualCameraSink

__all__ = [
    "ENHANCERS",
    "SWAP_MODELS",
    "FramePipeline",
    "LiveConfig",
    "LiveSession",
    "MODELS_DIR",
    "VirtualCameraSink",
    "__version__",
    "add_status_listener",
    "configure",
    "ensure_models",
    "is_virtual_camera",
    "list_cameras",
    "preload_models",
    "remove_status_listener",
    "suggest_default_execution_provider",
    "suggest_execution_providers",
    "swap_image",
]
