"""Shared path constants for the Deep-Live-Cam project."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DLC_MODELS_DIR lets an embedding application keep models elsewhere.
MODELS_DIR = os.environ.get("DLC_MODELS_DIR") or os.path.join(ROOT_DIR, "models")
