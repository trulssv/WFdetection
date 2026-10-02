"""Checks of the walnut geometry against the dataset's own system matrix.
Skipped when the data have not been downloaded (see data/README.md)."""

import math

import numpy as np
import pytest
import torch

from data.walnut import ROOT, load_walnut, walnut_geometry
from geometry.tomo import image_space

pytestmark = pytest.mark.skipif(not (ROOT / "Data82.mat").exists(), reason="walnut data missing")


def test_odl_geometry_matches_system_matrix():
    import scipy.io as sio
    from operators.radon import RayTransform
    A = sio.loadmat(ROOT / "Data82.mat")["A"].tocsr()
    n = 82
    rng = np.random.default_rng(0)
    c = -1 + (np.arange(n) + 0.5) * 2 / n
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    X = sum(np.exp(-((X1 - a)**2 + (X2 - b)**2) / (2 * 0.05**2))
            for a, b in rng.uniform(-0.6, 0.6, (6, 2)))       # a MATLAB image X
    ref = (A @ X.reshape(-1, order="F")).reshape(n, 120, order="F").T   # [projection, det]
    R = RayTransform(image_space(n), walnut_geometry(120, n))
    y = R(torch.as_tensor(X.T.copy(), dtype=torch.float32)[None])[0].numpy()  # ODL image = X^T
    assert np.corrcoef(y.ravel(), ref.ravel())[0, 1] > 0.999
    # A is in mm, ODL in units of L
    assert np.sum(ref * y) / np.sum(y * y) == pytest.approx(82 * 0.513 / 2, rel=0.01)


def test_load_walnut():
    d = load_walnut(120, det_binning=8)
    y = d["sinogram"]
    assert y.shape == (120, 287) and np.isfinite(y).all()
    assert np.median(y[:, :20]) < 0.05 and y.max() > 0.5     # air ~ 0, walnut attenuates
