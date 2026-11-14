"""Tests for contrast-adaptive sharpening and HyperSwap pixel boost."""

import cv2
import numpy as np
import pytest

import _vendor.modules.globals as g
from _vendor.modules.gpu_processing import cas_sharpen, gpu_sharpen
from _vendor.modules.processors.frame.face_swapper import HyperSwapper


def _edge_image():
    img = np.full((64, 64, 3), 60, np.uint8)
    img[:, 32:] = 180
    return cv2.GaussianBlur(img, (0, 0), 2)  # a soft edge to sharpen


def test_cas_zero_is_identity_and_flat_stays_flat():
    img = _edge_image()
    assert cas_sharpen(img, 0) is img
    flat = np.full((32, 32, 3), 128, np.uint8)
    assert np.array_equal(cas_sharpen(flat, 1.0), flat)


def test_cas_sharpens_edges_with_less_overshoot_than_unsharp():
    img = _edge_image()
    lap = lambda im: cv2.Laplacian(
        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F
    ).var()
    cas = cas_sharpen(img, 0.6)
    assert lap(cas) > 1.3 * lap(img)
    # overshoot beyond the local 5x5 range of the original (bright/dark fringes)
    k = np.ones((5, 5), np.uint8)
    lo, hi = cv2.erode(img, k).astype(int), cv2.dilate(img, k).astype(int)
    over = lambda s: (
        np.maximum(0, s.astype(int) - hi) + np.maximum(0, lo - s.astype(int))
    ).sum()
    unsharp = gpu_sharpen(img, strength=0.4, sigma=2)  # roughly matched detail
    assert lap(unsharp) >= 0.9 * lap(cas)
    assert over(cas) == 0 < over(unsharp)  # CAS clamps to the local range: no halos


class _IdentitySession:
    """Stands in for the HyperSwap session: returns the target unchanged and a full mask."""

    def __init__(self):
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        target = feeds["target"]
        return [target, np.ones((1, 1, 256, 256), np.float32)]


@pytest.fixture
def swapper():
    s = HyperSwapper.__new__(HyperSwapper)
    s.session = _IdentitySession()
    s.model_file = "fake.onnx"
    face = type(
        "F", (), {"normed_embedding": np.ones(512, np.float32) / np.sqrt(512)}
    )()
    saved = g.swap_pixel_boost
    yield s, face
    g.swap_pixel_boost = saved


def test_input_size_follows_the_setting_live(swapper):
    s, _ = swapper
    g.swap_pixel_boost = False
    assert s.input_size == (256, 256)
    g.swap_pixel_boost = True
    assert s.input_size == (512, 512)


def test_pixel_boost_split_and_interleave_round_trips(swapper):
    s, face = swapper
    rng = np.random.default_rng(0)
    aimg = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    out = s.swap_aligned(face, aimg)
    assert s.session.calls == 4 and out.shape == (512, 512, 4)
    assert (
        np.abs(out[:, :, :3].astype(int) - aimg.astype(int)).max() <= 1
    )  # normalisation rounding only
    assert (out[:, :, 3] == 255).all()


def test_normal_size_runs_once(swapper):
    s, face = swapper
    out = s.swap_aligned(face, np.zeros((256, 256, 3), np.uint8))
    assert s.session.calls == 1 and out.shape == (256, 256, 4)
