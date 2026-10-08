import math

import numpy as np
import pytest
import torch

from data.phantoms import Ellipse, Phantom, lifted_membership
from evaluation.metrics import label_masks, roc_auc
from geometry.lifted import LiftedGrid
from loss.data_discrepancy import OrientedEdgeForwardModel
from operators.canonical_relation import CanonicalRelation
from operators.image_analyzer import ImageDomainAnalyzer


def straight_step(grid, theta, supersample=8):
    """Indicator of {(x - x0) . omega(theta) > 0} through the centre x0 of cell
    (n/2, n/2), area-averaged over each cell."""
    n, ss = grid.n, supersample
    c = -1 + (np.arange(n * ss) + 0.5) * grid.h / ss - grid.h / 2
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    f = (X1 * math.cos(theta) + X2 * math.sin(theta) > 0).astype(np.float64)
    return f.reshape(n, ss, n, ss).mean(axis=(1, 3))


def test_unit_step_response_and_angular_response():
    grid = LiftedGrid(64, 16)
    A = ImageDomainAnalyzer(grid, 1.5, 2.0)
    r = A.angular_response()
    torch.testing.assert_close(torch.diag(r), torch.ones(16), rtol=0, atol=1e-6)
    assert r[0, 8].abs() < 1e-6                     # orthogonal step: no response
    assert (r[0, 1:8].abs().diff() <= 1e-6).all()   # monotone decay towards 90 degrees
    for k in (0, 3, 5):
        th = grid.thetas()[k]
        f = torch.as_tensor(straight_step(grid, th), dtype=torch.float32)[None, None]
        b = A(f)[0, 0]
        centre = b[:, 32, 32]
        assert abs(float(centre[k]) - 1) < 0.02
        torch.testing.assert_close(centre, r[:, k].abs(), rtol=0, atol=0.03)


def test_detects_ellipse_wavefront_set():
    grid = LiftedGrid(64, 16)
    e = Ellipse((0.1, -0.05), (0.45, 0.25), 0.7, 1.0)
    f = torch.as_tensor(Phantom([e]).render(64), dtype=torch.float32)[None, None]
    b = ImageDomainAnalyzer(grid)(f)[0, 0].numpy()
    pos, neg = label_masks(lifted_membership(grid, e.samples(grid.h / 2)))
    assert roc_auc(b, pos, neg) > 0.95


@pytest.mark.parametrize("theta_k", [2, 7])
def test_forward_model_unit_height(theta_k):
    """q = 1 on a straight curve resolved over 2 cells predicts b = r (peak 1),
    and the analyzer's response to the matching sharp edge is the same."""
    grid = LiftedGrid(64, 16)
    A = ImageDomainAnalyzer(grid)
    fwd = OrientedEdgeForwardModel(grid, A.angular_response(), A.sigma_n)
    th = grid.thetas()[theta_k]
    x = torch.as_tensor(grid.coords(), dtype=torch.float32) / grid.h
    X1, X2 = torch.meshgrid(x, x, indexing="ij")
    d = X1 * math.cos(th) + X2 * math.sin(th)
    dk = (torch.arange(16) - theta_k + 8) % 16 - 8
    q = torch.exp(-dk.float()[:, None, None]**2 / 8 - d**2 / 8)[None, None]
    peak = fwd(q)[0, 0, theta_k, 31:33, 31:33].mean()
    assert abs(float(peak) - 1) < 0.05


def test_forward_model_rejects_tangent_streaks():
    """Unlike the sinogram model (tests/test_lpd.py), the image-side model
    tells the true amplitude field from the same field smeared along its
    tangent lines: a 31-cell smear (about the width of the sinogram fibre
    response) fits much worse."""
    n = 64
    grid = LiftedGrid(n, 32)
    e = Ellipse((0.05, -0.05), (0.45, 0.3), 0.4, 1.0)
    f = torch.as_tensor(Phantom([e]).render(n), dtype=torch.float32)[None, None]
    A = ImageDomainAnalyzer(grid, 1.5, 2.0)
    b = A(f)
    fwd = OrientedEdgeForwardModel(grid, A.angular_response(), A.sigma_n)
    q = torch.as_tensor(lifted_membership(grid, e.samples(grid.h / 2), 2 * grid.h, 2 * grid.dtheta),
                        dtype=torch.float32)[None, None]
    C = CanonicalRelation(grid)

    def residual(qq):
        bh = fwd(qq)
        alpha = float((bh * b).sum() / (bh * bh).sum())
        return float((b - alpha * bh).abs().mean() / b.mean())

    def smear(w):
        v = torch.nn.functional.avg_pool3d(C(q), (1, 1, w), 1, (0, 0, w // 2))
        return C.inverse(v)

    r_true = residual(q)
    assert r_true < 0.35                          # measured 0.28 (zero model: 1)
    assert residual(smear(31)) > r_true + 0.3     # measured 0.84


def test_detrending_annihilates_linear_intensity():
    grid = LiftedGrid(64, 16)
    x = torch.as_tensor(grid.coords(), dtype=torch.float32)
    X1, X2 = torch.meshgrid(x, x, indexing="ij")
    ramp = (0.7 * X1 - 0.4 * X2)[None, None]
    inner = slice(30, 34)   # away from the zero padding
    plain = ImageDomainAnalyzer(grid, 1.5, 2.0)(ramp)[0, 0, :, inner, inner]
    detrended = ImageDomainAnalyzer(grid, 1.5, 2.0, detrend=3.0)(ramp)[0, 0, :, inner, inner]
    assert plain.max() > 0.01
    assert detrended.abs().max() < 1e-5


def test_ridges_recover_thin_curve():
    """The ridges of a blurred lifted curve are the curve itself (up to one cell)."""
    from evaluation.ridges import best_f, rasterize
    grid = LiftedGrid(64, 16)
    e = Ellipse((0.1, 0.0), (0.5, 0.3), 0.3, 1.0)
    thin = rasterize(grid, e.samples(grid.h / 4))
    blurred = torch.as_tensor(lifted_membership(grid, e.samples(grid.h / 2), 2 * grid.h, 2 * grid.dtheta),
                              dtype=torch.float32)
    rec = best_f([blurred], [thin], grid)
    assert rec["F"] > 0.9 and rec["F_x"] > 0.95
    assert best_f([torch.as_tensor(thin, dtype=torch.float32)], [thin], grid)["F"] == 1.0
