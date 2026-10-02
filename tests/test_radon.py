import math

import numpy as np
import pytest
import torch
from odl.applications import tomo

from geometry.lifted import omega
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from operators.radon import RayTransform


def blob(n, x0, sigma=0.03):
    c = -1 + (np.arange(n) + 0.5) * 2 / n
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    return np.exp(-((X1 - x0[0])**2 + (X2 - x0[1])**2) / (2 * sigma**2))


def test_odl_convention_s_equals_x_dot_omega():
    """The sinogram of a blob at x0 peaks at s = x0 . omega(phi) (index order [phi, s])."""
    n = 64
    geom = parallel_geometry(n_angles=24, n_det=128)
    R = RayTransform(image_space(n), geom)
    x0 = np.array([0.35, -0.2])
    y = R(torch.as_tensor(blob(n, x0), dtype=torch.float32)[None]).numpy()[0]
    phis, s = sinogram_axes(geom)
    for k, phi in enumerate(phis):
        assert abs(s[np.argmax(y[k])] - x0 @ omega(phi)) <= 1.5 * (s[1] - s[0])


def test_antipodal_symmetry():
    """R f(s, phi + pi) = R f(-s, phi), used to pad sinograms periodically."""
    n = 64
    geom = parallel_geometry(n_angles=32, n_det=96, angle_range=(0, 2 * math.pi))
    R = RayTransform(image_space(n), geom)
    y = R(torch.as_tensor(blob(n, [0.3, 0.1], 0.1), dtype=torch.float32)[None])[0]
    torch.testing.assert_close(y[16:], y[:16].flip(-1), rtol=1e-3, atol=1e-3 * float(y.max()))


def test_autograd_gradient_is_transpose():
    """Autograd must use the matrix transpose A^T = (w_X / w_Y) A^*, not ODL's
    weighted adjoint A^*.  Exact on the matched CPU projector pair."""
    n = 32
    R = RayTransform(image_space(n), parallel_geometry(16, 48), impl="astra_cpu")
    torch.manual_seed(0)
    f = torch.randn(2, n, n, requires_grad=True)
    g = torch.randn(2, 16, 48)
    lhs = (R(f) * g).sum()
    lhs.backward()
    rhs = (f * f.grad).sum()               # = sum f . A^T g for a linear map
    assert abs(float(lhs) - float(rhs)) <= 1e-4 * abs(float(lhs))
    w = R.op.domain.cell_volume / R.op.range.cell_volume
    torch.testing.assert_close(f.grad, w * R.adjoint(g), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not tomo.ASTRA_CUDA_AVAILABLE, reason="no ASTRA CUDA")
def test_gpu_projector_pair_nearly_matched():
    """ASTRA's GPU forward / back-projectors are not an exactly matched pair;
    the mismatch is ~1e-3 of <Rf, g> for positive data."""
    n = 64
    R = RayTransform(image_space(n), parallel_geometry(90, 128), impl="astra_cuda")
    torch.manual_seed(0)
    f, g = torch.rand(1, n, n, requires_grad=True), torch.rand(1, 90, 128)
    lhs = (R(f) * g).sum()
    lhs.backward()
    assert abs(float(lhs) - float((f * f.grad).sum())) <= 2e-3 * abs(float(lhs))
