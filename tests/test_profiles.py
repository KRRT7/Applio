"""Tests for Live profiles storage (tabs/faceswap/profiles.py)."""
import os

import pytest

from tabs.faceswap import profiles as P


@pytest.fixture
def photos(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PROFILES_DIR", str(tmp_path / "profiles"))
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(3):
        p = src / f"photo{i + 1}.jpg"
        p.write_bytes(f"image {i + 1}".encode())
        paths.append(str(p))
    return paths


def test_save_copies_photos_in_order_and_stores_settings(photos):
    prof = P.save_profile("Ana María", photos, {"swap_model": "hyperswap_1c_256", "sharpness": 0.2, "ignored": 1},
                          {"model_file": "logs\\ana\\ana.pth", "pitch": 10})
    assert P.list_profiles() == ["Ana María"]
    assert [os.path.basename(f) for f in prof["faces"]] == ["01_photo1.jpg", "02_photo2.jpg", "03_photo3.jpg"]
    assert all(os.path.dirname(f).endswith(os.path.join("ana_maría", "faces")) for f in prof["faces"])
    assert prof["face"] == {"swap_model": "hyperswap_1c_256", "sharpness": 0.2}  # unknown keys dropped
    assert prof["voice"] == {"model_file": "logs/ana/ana.pth", "pitch": 10}       # portable separators


def test_resave_from_its_own_photos_reorders_and_drops(photos):
    prof = P.save_profile("Ana", photos, {}, {})
    prof = P.save_profile("Ana", [prof["faces"][2], prof["faces"][0]], {}, {})
    assert [os.path.basename(f) for f in prof["faces"]] == ["01_photo3.jpg", "02_photo1.jpg"]
    assert open(prof["faces"][0], "rb").read() == b"image 3"


def test_delete_removes_only_the_profile(photos):
    P.save_profile("Ana", photos, {}, {})
    P.delete_profile("Ana")
    assert P.list_profiles() == []
    assert all(os.path.isfile(p) for p in photos), "original photos must survive"


def test_rejects_empty_names_and_paths_outside_the_profiles_folder(photos):
    with pytest.raises(ValueError):
        P.save_profile("   ", photos, {}, {})
    assert P.slug("../../etc") == "etc"  # separators are stripped, so a name cannot escape


def test_match_choice_ignores_separator_differences():
    choices = [os.path.join("logs", "martu", "martu.index"), os.path.join("logs", "paloma", "paloma.index")]
    assert P.match_choice("logs/paloma/paloma.index", choices) == choices[1]
    assert P.match_choice("logs/missing.index", choices) == "logs/missing.index"
    assert P.match_choice("", choices) is None
