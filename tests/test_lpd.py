import numpy as np
import pytest
import torch

from data.phantoms import Ellipse, Phantom, lifted_membership
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from loss.data_discrepancy import MicrolocalForwardModel
from models.layers import LiftedBlock, s1_to_rp1_even, s1_to_rp1_odd, sinogram_rp1_to_s1
from models.lpd import LiftedLPD
from operators.canonical_relation import CanonicalRelation
from operators.microlocal import SinogramAnalyzer
from operators.radon import RayTransform


def test_outputs_and_constraints():
    grid = LiftedGrid(32, 8)
    model = LiftedLPD(grid, n_iter=2, hidden=8).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, *grid.shape))
    for k in "abcJm":
        assert out[k].shape == (2, 1, *grid.shape)
    assert (out["a"] >= 0).all() and (out["m"] >= 0).all() and (out["m"] < 1).all()
    assert (out["J"] > 0).all() and (out["J"] <= model.J_max).all()


def test_antipodal_projections():
    x = torch.randn(1, 1, 8, 4, 4)
    even, odd = s1_to_rp1_even(x), s1_to_rp1_odd(x)
    # symmetric extension back to S^1 reproduces the projected parts
    torch.testing.assert_close(even + odd, 0.5 * (x[:, :, :4] + x[:, :, 4:]) + 0.5 * (x[:, :, :4] - x[:, :, 4:]))
    v = torch.randn(1, 1, 4, 6, 6)
    w = sinogram_rp1_to_s1(v)
    torch.testing.assert_close(w[:, :, 4:], v.flip(-1, -2))


def test_rotation_is_phi_shift_on_double_cover():
    """Proposition 4.6: rotating the image by pi/2 shifts C u by K/2 slices in phi."""
    grid = LiftedGrid(32, 8)                     # 2K = 16 slices, pi/2 = 4 slices
    C = CanonicalRelation(grid, double_cover=True)
    torch.manual_seed(0)
    u = torch.nn.functional.avg_pool3d(torch.randn(1, 1, 16, 32, 32), 3, 1, 1)
    yy, xx = torch.meshgrid(grid.coords_torch(), grid.coords_torch(), indexing="ij")
    u = u * ((xx**2 + yy**2) < 0.8).float()      # support in the disk
    # rotation by +pi/2: u'(x, theta) = u(R_{-pi/2} x, theta - pi/2)
    # R_{-pi/2}(x1, x2) = (x2, -x1)  ->  u'[i1, i2] = u[i2, n-1-i1]
    u_rot = torch.roll(torch.rot90(u, k=1, dims=(-2, -1)), shifts=4, dims=2)
    lhs = C(u_rot)
    rhs = torch.roll(C(u), shifts=4, dims=2)
    assert float((lhs - rhs).norm() / rhs.norm()) < 1e-4


def test_periodic_block_commutes_with_fibre_shift():
    torch.manual_seed(0)
    blk = LiftedBlock(2, 1, 4)
    x, e = torch.randn(1, 2, 8, 6, 6), torch.randn(1, 1, 8, 6, 6)
    torch.testing.assert_close(blk(torch.roll(x, 3, 2), torch.roll(e, 3, 2)),
                               torch.roll(blk(x, e), 3, 2), rtol=1e-5, atol=1e-5)


def test_forward_model_explains_coefficients():
    """R_t C q with the true amplitude field reproduces |My| far better than a
    zero model, and better than q smeared along the tangent lines over more
    than the width of the fibre response.  Smears narrower than that width
    (~30 cells) fit equally well: the deconvolution along t is ill-conditioned
    and must be resolved by the prior (research/background.md, Sec. 5.3)."""
    n = 64
    grid = LiftedGrid(n, 32)
    e = Ellipse((0.05, -0.05), (0.45, 0.3), 0.4, 1.0)
    geom = parallel_geometry(3 * n, 2 * n)
    y = RayTransform(image_space(n), geom)(torch.as_tensor(Phantom([e]).render(n),
                                                          dtype=torch.float32)[None, None])
    phis, s = sinogram_axes(geom)
    M = SinogramAnalyzer(grid, phis, s)
    c = M(y).abs()
    C = CanonicalRelation(grid)
    fwd = MicrolocalForwardModel(C, M.fibre_response())
    # amplitude at the network's resolution (2-cell blur, cf. models.lpd.reference_line_density)
    q = torch.as_tensor(lifted_membership(grid, e.samples(grid.h / 2), 2 * grid.h, 2 * grid.dtheta),
                        dtype=torch.float32)[None, None]

    def residual(qq):
        ch = fwd(qq)
        alpha = float((ch * c).sum() / (ch * ch).sum())      # best global amplitude
        return float((c - alpha * ch).abs().mean() / c.mean())

    def smear(w):
        v = torch.nn.functional.avg_pool3d(C(q), (1, 1, w), 1, (0, 0, w // 2))
        return C.inverse(v)

    r_true = residual(q)
    assert r_true < 0.65                              # measured 0.56 (zero model: 1)
    assert abs(residual(smear(15)) - r_true) < 0.03   # narrow smear: indistinguishable
    assert residual(smear(63)) > r_true + 0.03        # wide smear: worse
