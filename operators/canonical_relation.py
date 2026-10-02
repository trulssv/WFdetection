"""Canonical relation of the 2D parallel-beam ray transform on lifted fields.

For R f(s, phi) = int f(s omega(phi) + t omega_perp(phi)) dt the canonical
relation maps the image wavefront point (x, theta) to the sinogram wavefront
point (s, phi, t) = (x . omega(theta), theta, x . omega_perp(theta)), where t
parametrizes the sinogram covector direction (1, -t) in (s, phi) coordinates.
For a fixed orientation slice this is just the rotation (s, t) = R_{-theta} x,
so on lifted fields

    (C u)(s, phi, t) = u(s omega(phi) + t omega_perp(phi), phi).

C is a volume-preserving strict contactomorphism, C^*(ds - t dphi) = alpha
(``research/background.md``, Proposition 4.4), and unitary on fields
supported in the unit disk.  Points that rotate out of
[-1, 1]^2 are sampled as zero, so objects should lie inside the unit disk.
"""

import torch
import torch.nn.functional as F

from geometry.lifted import LiftedGrid


def _slice_rotation_grid(grid: LiftedGrid, inverse: bool, double_cover: bool) -> torch.Tensor:
    """Sampling grids for ``grid_sample``, shape (n_theta, n, n, 2) (or 2 n_theta
    slices covering [0, 2 pi) on the double cover).

    The spatial domain is [-1, 1]^2 and ``align_corners=False``, so physical
    coordinates equal grid_sample's normalized coordinates.  grid_sample reads
    the last tensor axis from ``grid[..., 0]`` and the second-last from
    ``grid[..., 1]``.
    """
    c = grid.coords_torch(dtype=torch.float64)
    th = torch.arange(grid.n_theta * (2 if double_cover else 1), dtype=torch.float64)
    th = (th * grid.dtheta)[:, None, None]
    cos, sin = torch.cos(th), torch.sin(th)
    a, b = torch.meshgrid(c, c, indexing="ij")  # output axes (row, column)
    if not inverse:
        # output (s, t) -> input x = s omega + t omega_perp, input axes (x1, x2)
        x1 = a * cos - b * sin
        x2 = a * sin + b * cos
        return torch.stack([x2, x1], dim=-1).float()
    # output (x1, x2) -> input (s, t) = (x . omega, x . omega_perp)
    s = a * cos + b * sin
    t = -a * sin + b * cos
    return torch.stack([t, s], dim=-1).float()


def _resample_slices(u: torch.Tensor, sample_grid: torch.Tensor) -> torch.Tensor:
    """Resample every orientation slice of ``u`` (B, C, K, n, n) with its own grid."""
    B, C, K, n1, n2 = u.shape
    x = u.permute(0, 2, 1, 3, 4).reshape(B * K, C, n1, n2)
    g = sample_grid.to(device=u.device, dtype=u.dtype).repeat(B, 1, 1, 1)
    y = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return y.reshape(B, K, C, n1, n2).permute(0, 2, 1, 3, 4)


class CanonicalRelation(torch.nn.Module):
    """Lifted image field (B, C, K, n, n) on (theta, x1, x2)  ->
    lifted sinogram field (B, C, K, n, n) on (phi, s, t).

    With ``double_cover=True`` both sides use 2K slices covering [0, 2 pi):
    the image side is then Omega x S^1 and the sinogram side the space of
    oriented lines with a point on them; the formula is the same.  This is the
    setting of the network (``models/lpd.py``), where convolutions must not see
    the identification (s, phi + pi, t) ~ (-s, phi, -t).

    ``forward`` is the discrete pushforward C, ``adjoint`` its exact discrete
    adjoint (computed with autograd, so the dot-product test holds to rounding
    error) and ``inverse`` the inverse rotation.  In the continuum C^* = C^{-1};
    discretely, the transpose of bilinear resampling scatters with non-uniform
    (moire) weights and is a few percent off the inverse.  Use ``inverse`` to
    pull fields back in architectures and ``adjoint`` where exact adjointness
    matters (gradients, dot-product identities).
    """

    def __init__(self, grid: LiftedGrid, double_cover: bool = False):
        super().__init__()
        self.grid, self.double_cover = grid, double_cover
        self.shape = ((2 if double_cover else 1) * grid.n_theta, grid.n, grid.n)
        self.register_buffer("_fwd_grid", _slice_rotation_grid(grid, False, double_cover),
                             persistent=False)
        self.register_buffer("_inv_grid", _slice_rotation_grid(grid, True, double_cover),
                             persistent=False)

    def _check(self, u):
        if u.dim() != 5 or tuple(u.shape[2:]) != self.shape:
            raise ValueError(f"expected shape (B, C, {self.shape}), got {tuple(u.shape)}")

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        self._check(u)
        return _resample_slices(u, self._fwd_grid)

    def inverse(self, v: torch.Tensor) -> torch.Tensor:
        self._check(v)
        return _resample_slices(v, self._inv_grid)

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        self._check(v)
        # C is linear, so its adjoint is the vector-Jacobian product at any point.
        keep_graph = torch.is_grad_enabled() and v.requires_grad
        with torch.enable_grad():
            u0 = torch.zeros_like(v, requires_grad=True)
            out = self.forward(u0)
            (u,) = torch.autograd.grad(out, u0, grad_outputs=v, create_graph=keep_graph)
        return u
