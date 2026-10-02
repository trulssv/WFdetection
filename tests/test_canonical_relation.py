import math

import numpy as np
import pytest
import torch

from geometry.lifted import (LiftedGrid, image_to_sinogram_coords,
                             sinogram_to_image_coords)
from operators.canonical_relation import CanonicalRelation


@pytest.fixture
def grid():
    return LiftedGrid(n=64, n_theta=16)


def gaussian_field(grid, centers, sigma=0.08):
    """Field with a spatial Gaussian bump in every orientation slice; centers[k] is
    the bump centre of slice k."""
    c = grid.coords()
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    u = np.stack([np.exp(-((X1 - p[0])**2 + (X2 - p[1])**2) / (2 * sigma**2)) for p in centers])
    return torch.as_tensor(u[None, None], dtype=torch.float32)


def test_pointwise_coords_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.7, 0.7, (100, 2))
    th = rng.uniform(0, math.pi, 100)
    s, phi, t = image_to_sinogram_coords(x, th)
    x2, _ = sinogram_to_image_coords(s, phi, t)
    np.testing.assert_allclose(x2, x, atol=1e-12)
    # (s, t) is a rotation of x
    np.testing.assert_allclose(s**2 + t**2, np.sum(x**2, -1), atol=1e-12)


def test_adjoint_dot_product(grid):
    C = CanonicalRelation(grid)
    torch.manual_seed(0)
    u = torch.randn(2, 3, *grid.shape)
    v = torch.randn(2, 3, *grid.shape)
    lhs = (C(u) * v).sum()
    rhs = (u * C.adjoint(v)).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_bump_lands_at_predicted_point(grid):
    C = CanonicalRelation(grid)
    x0 = np.array([0.3, -0.2])
    u = gaussian_field(grid, [x0] * grid.n_theta)
    v = C(u)[0, 0].numpy()
    c = grid.coords()
    for k, th in enumerate(grid.thetas()):
        s0, _, t0 = image_to_sinogram_coords(x0, th)
        i, j = np.unravel_index(np.argmax(v[k]), v[k].shape)
        assert abs(c[i] - s0) <= grid.h and abs(c[j] - t0) <= grid.h


def test_inverse_and_unitarity_inside_disk(grid):
    C = CanonicalRelation(grid)
    rng = np.random.default_rng(1)
    centers = rng.uniform(-0.4, 0.4, (grid.n_theta, 2))
    u = gaussian_field(grid, centers, sigma=0.1)
    v = C(u)
    rel = lambda a, b: float((a - b).norm() / b.norm())
    # two bilinear interpolations smooth by O(h^2 / sigma^2), about 2% here
    assert rel(C.inverse(v), u) < 0.04
    # the exact transpose of bilinear resampling scatters with non-uniform
    # (moire) weights, so it is a worse approximation of the inverse
    assert rel(C.adjoint(v), u) < 0.08
    assert abs(float(v.norm() / u.norm()) - 1) < 0.01
