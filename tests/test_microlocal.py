"""Integration test of the theory: M(R f) peaks where C(WF(f)) predicts."""

import numpy as np
import pytest
import torch

from data.phantoms import CurveSamples, Ellipse, Phantom, Polygon, lifted_membership
from evaluation.metrics import label_masks, roc_auc, t_localization
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from loss.data_discrepancy import MixtureLikelihood
from operators.canonical_relation import CanonicalRelation
from operators.microlocal import SinogramAnalyzer
from operators.radon import RayTransform


@pytest.fixture(scope="module")
def setup():
    n = 96
    grid = LiftedGrid(n=n, n_theta=48)
    phantom = Phantom([Ellipse((-0.25, 0.1), (0.35, 0.2), 0.5, 1.0),
                       Ellipse((0.35, -0.25), (0.15, 0.25), -0.3, -0.6),
                       Polygon([[0.1, 0.35], [0.5, 0.3], [0.3, 0.6]], 0.8)])
    geom = parallel_geometry(n_angles=300, n_det=2 * n)
    R = RayTransform(image_space(n), geom)
    f = torch.as_tensor(phantom.render(n), dtype=torch.float32)
    y = R(f[None, None])
    phis, s = sinogram_axes(geom)
    M = SinogramAnalyzer(grid, phis, s)
    c = M(y)
    return grid, phantom, M, c


def test_coefficients_peak_at_predicted_t(setup):
    """Checks the sign and offset of t.  Only smooth edges are used: all points
    of a straight edge map to one sinogram point, whose WF contains a whole
    interval of t, so a per-point argmax is ill-posed there."""
    grid, phantom, _, c = setup
    cs = CurveSamples.concat([sh.samples(grid.h / 2) for sh in phantom.shapes
                              if isinstance(sh, Ellipse)])
    res = t_localization(c[0, 0].numpy(), grid, cs.x, cs.theta, window=8)
    err = res["local_error"]
    assert res["n_samples"] > 100
    assert abs(np.mean(err)) <= 0.5          # unbiased
    assert np.median(np.abs(err)) <= 1.0
    assert res["global_hit_rate"] > 0.65


def test_coefficients_separate_wavefront_set(setup):
    grid, phantom, _, c = setup
    C = CanonicalRelation(grid)
    m = torch.as_tensor(lifted_membership(grid, phantom.samples(grid.h / 2, "full")),
                        dtype=torch.float32)
    v = C(m[None, None])[0, 0].numpy()
    pos, neg = label_masks(v)
    assert roc_auc(c[0, 0].numpy(), pos, neg) > 0.9
    # the same on the image side, after pulling the coefficients back
    u_hat = C.inverse(c)[0, 0].numpy()
    pos, neg = label_masks(m.numpy())
    assert roc_auc(u_hat, pos, neg) > 0.9


def test_noise_std_and_posterior(setup):
    _, _, M, c = setup
    sigma = M.coefficient_noise_std(0.01)
    assert sigma.shape == (M.grid.n,) and torch.all(sigma > 0)
    lik = MixtureLikelihood(sigma, edge_scale=float(c.max()) / 4)
    with torch.no_grad():
        p = lik.posterior(c)
    assert 0 <= float(p.min()) and float(p.max()) <= 1
