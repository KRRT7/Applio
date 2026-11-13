"""Shared test setup.

Tests run from the repository root with Applio's modules importable, and
never touch the user's real settings or profiles: the Live page's config
file and the profiles folder are redirected to a temporary directory.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point the Live page's settings and the profiles folder at tmp_path."""
    from tabs.faceswap import faceswap, profiles

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(faceswap, "CONFIG_PATH", str(config))
    monkeypatch.setattr(profiles, "PROFILES_DIR", str(tmp_path / "profiles"))
    return tmp_path
