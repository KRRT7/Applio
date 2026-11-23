import os
import sys

import cv2
import numpy as np

# insightface imports insightface.app.mask_renderer (unused here), which imports
# albumentations, which imports its optional albumentations.pytorch, i.e. torch,
# whenever torch is installed. The UI process runs face swap on ONNX Runtime
# alone and keeps torch out (~0.3 GB of RAM); albumentations skips the
# submodule on ImportError, which a None entry raises. Left alone if torch's
# side was already imported.
sys.modules.setdefault("albumentations.pytorch", None)
# albumentations also queries PyPI for a newer version on every import.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


# Utility function to support unicode characters in file paths for reading.
# OpenCV's cv2.imread() encodes the path with the locale ANSI code page on
# Windows, so it silently returns None for paths containing non-ASCII
# characters (Chinese, Japanese, Cyrillic, accents, ...). Reading the bytes
# through NumPy (which uses Python's unicode-aware file I/O) and decoding them
# in memory sidesteps that limitation. Returns None on failure, matching
# cv2.imread() so it stays a drop-in replacement.
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


# Utility function to support unicode characters in file paths for writing.
# cv2.imwrite() has the same ANSI-path limitation, so we encode the image in
# memory and write the bytes out with NumPy's unicode-aware file I/O. Returns
# True/False like cv2.imwrite() so it stays a drop-in replacement.
def imwrite_unicode(path, img, params=None):
    try:
        root, ext = os.path.splitext(path)
        if not ext:
            ext = ".png"
        result, encoded_img = cv2.imencode(ext, img, params if params is not None else [])
        if not result:
            return False
        encoded_img.tofile(path)
        return True
    except Exception:
        return False


# ONNX Runtime's CUDA provider needs the CUDA / cuDNN DLLs loaded first.
# Importing torch used to do that as a side effect; the UI process no longer
# imports torch (its CUDA runtime cost ~0.3 GB of RAM there), so load them
# explicitly. preload_dlls() finds torch's bundled DLLs (torch/lib) from its
# package metadata without importing it, then falls back to nvidia-* wheels
# and the default search path. Nothing to do when torch is already loaded.
def _preload_cuda_dlls():
    import sys

    if "torch" in sys.modules:
        return
    try:
        import onnxruntime

        if (
            hasattr(onnxruntime, "preload_dlls")
            and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        ):
            onnxruntime.preload_dlls()
    except Exception as error:  # the CPU / other providers still work
        print(f"[DLC] Could not preload the CUDA DLLs: {error}")


_preload_cuda_dlls()
