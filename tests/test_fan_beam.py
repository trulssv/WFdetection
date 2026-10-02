import math

import numpy as np
import pytest
import torch

from data.phantoms import Ellipse
from geometry.lifted import LiftedGrid, omega, omega_perp
from geometry.tomo import fan_geometry, image_space, parallel_geometry, sinogram_axes
from operators.fan_beam import (FanBeamCanonicalRelation, FanToParallel, fan_to_line,
                                image_to_fan_wavefront, line_to_fan, reduce_line)
from operators.microlocal import half_ramp
from operators.radon import RayTransform

RS, RD, W2 = 5.0, 9.0, 2.6        # short source distance: strong fan effects


def test_line_map_matches_odl_rays():
    g = fan_geometry(16, 32, RS, RD, W2)
    rng = np.random.default_rng(0)
    for beta, u in zip(rng.uniform(0, 2 * math.pi, 20), rng.uniform(-W2, W2, 20)):
        src, det = g.src_position(beta), g.det_point_position(beta, u)
        s, phi = fan_to_line(beta, u, RS, RD)
        d = (det - src) / np.linalg.norm(det - src)
        np.testing.assert_allclose(d, omega_perp(phi), atol=1e-10)   # ray direction
        assert abs(src @ omega(phi) - s) < 1e-10 and abs(det @ omega(phi) - s) < 1e-10


def test_line_to_fan_inverts_both_measurements():
    rng = np.random.default_rng(1)
    beta, u = rng.uniform(0, 2 * math.pi, 50), rng.uniform(-W2, W2, 50)
    s, phi = fan_to_line(beta, u, RS, RD)
    B, U = line_to_fan(s, phi, RS, RD)
    np.testing.assert_allclose(B[0], beta, atol=1e-10)
    np.testing.assert_allclose(U[0], u, atol=1e-10)
    s2, phi2 = fan_to_line(B[1], U[1], RS, RD)          # the second ray is the same line
    a, p, _ = reduce_line(s, phi)
    a2, p2, _ = reduce_line(s2, phi2)
    np.testing.assert_allclose(a2, a, atol=1e-10)
    np.testing.assert_allclose(np.mod(p2 - p + 1e-9, math.pi), 1e-9, atol=1e-8)


def blob(n, x0, sigma):
    c = -1 + (np.arange(n) + 0.5) * 2 / n
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    return np.exp(-((X1 - x0[0])**2 + (X2 - x0[1])**2) / (2 * sigma**2))


def test_rebinning_reproduces_parallel_data():
    n = 64
    sp = image_space(n)
    fan = fan_geometry(720, 256, RS, RD, W2)
    par = parallel_geometry(180, 128, det_radius=0.95)
    f = torch.as_tensor(blob(n, (0.2, -0.3), 0.1) + blob(n, (-0.3, 0.1), 0.15), dtype=torch.float32)
    g = RayTransform(sp, fan)(f[None, None])
    y = RayTransform(sp, par)(f[None, None])
    betas, us = sinogram_axes(fan)
    phis, s = sinogram_axes(par)
    y_hat = FanToParallel(betas, us, RS, RD, phis, s)(g)
    assert float((y_hat - y).norm() / y.norm()) < 0.02


def test_lifted_fan_relation_moves_bump_to_predicted_point():
    grid = LiftedGrid(n=64, n_theta=32)
    x0, k0 = np.array([0.3, -0.25]), 10
    u = torch.zeros(1, 1, *grid.shape)
    u[0, 0, k0] = torch.as_tensor(blob(grid.n, x0, 0.04), dtype=torch.float32)
    betas = (np.arange(360) + 0.5) * 2 * math.pi / 360
    us = np.linspace(-W2, W2, 128)
    v = FanBeamCanonicalRelation(grid, betas, us, RS, RD)(u)[0, 0].numpy()
    wf = image_to_fan_wavefront(x0, grid.thetas()[k0], RS, RD)
    ts = grid.coords()
    for m in range(2):           # both measurements of the tangent line
        ib = np.argmin(np.abs(np.angle(np.exp(1j * (betas - wf["beta"][m])))))
        iu, it = np.unravel_index(np.argmax(v[ib]), v[ib].shape)
        assert abs(us[iu] - wf["u"][m]) < 2 * (us[1] - us[0])
        assert abs(ts[it] - wf["t"][m]) < 2 * grid.h


def test_fan_covector_matches_data():
    """The predicted covector sigma is normal to the singular curves of the fan
    sinogram of an ellipse (checked with the gradient of the smoothed data)."""
    n = 128
    e = Ellipse((0.1, 0.05), (0.45, 0.3), 0.3, 1.0)
    from data.phantoms import Phantom
    f = torch.as_tensor(Phantom([e]).render(n), dtype=torch.float32)
    fan = fan_geometry(1440, 512, RS, RD, W2)
    g = RayTransform(image_space(n), fan)(f[None, None])
    betas, us = sinogram_axes(fan)
    db, du = betas[1] - betas[0], us[1] - us[0]
    g = half_ramp(g, du)
    # Gaussian smoothing and gradient in physical (beta, u) units
    k = torch.arange(-12, 13, dtype=torch.float32)
    gb = torch.exp(-(k * db)**2 / (2 * 0.02**2)); gb /= gb.sum()
    gu = torch.exp(-(k * du)**2 / (2 * 0.02**2)); gu /= gu.sum()
    gs = torch.nn.functional.conv2d(g, gb.view(1, 1, -1, 1), padding=(12, 0))
    gs = torch.nn.functional.conv2d(gs, gu.view(1, 1, 1, -1), padding=(0, 12))[0, 0].numpy()
    d_beta = np.gradient(gs, db, axis=0)
    d_u = np.gradient(gs, du, axis=1)
    cs = e.samples(0.02)
    wf = image_to_fan_wavefront(cs.x, cs.theta, RS, RD)
    errs = []
    for m in range(2):
        ib = np.round((wf["beta"][m] - betas[0]) / db).astype(int) % len(betas)
        iu = np.round((wf["u"][m] - us[0]) / du).astype(int)
        ok = (iu > 15) & (iu < len(us) - 15)
        grad = np.stack([d_beta[ib[ok], iu[ok]], d_u[ib[ok], iu[ok]]], -1)
        grad /= np.linalg.norm(grad, axis=-1, keepdims=True)
        errs.append(np.degrees(np.arccos(np.clip(np.abs(np.sum(grad * wf["sigma"][m][ok], -1)), 0, 1))))
    assert np.median(np.concatenate(errs)) < 5.0
