"""Full-aperture microlocal analyzer: sinogram -> lifted image field.

The sinogram analyzer M (:mod:`operators.microlocal`) is local in (s, phi):
its kernels see an angular window of about +-3 sigma_t, so the position t
along the ray is resolved only to ~1 / aspect (the wide fibre response,
``research/background.md`` Prop. 5.3).  Pulled back by C^{-1}, every edge
point then smears along its tangent line.

The position along the ray is determined by *all* angles together, which is
what filtered back-projection does.  This analyzer therefore applies an
oriented derivative-of-Gaussian bank to the FBP image,

    b[..., k, i1, i2] = |(W_k * FBP y)(x_i)|,

with W_k differentiating along the normal omega(theta_k) = (cos, sin)(theta_k)
and smoothing along the tangent.  b is still a fixed linear map of y followed
by a modulus (W_k FBP = R^* applied to sinogram filters spanning all angles),
so the method stays indirect, but its response to an edge is local in x: a
straight edge with normal theta_e and contrast J at distance d gives

    b(x, theta_k) = J |r(theta_k, theta_e)| exp(-d^2 / (2 sigma_n^2)),

with the angular response r of :meth:`ImageDomainAnalyzer.angular_response`.
Interference only happens between edges closer than ~sigma_n, instead of
between all singularities on a ray.  This is the model of
:class:`loss.data_discrepancy.OrientedEdgeForwardModel`.

The analyzer needs a good FBP, i.e. dense angles.  For sparse or limited
angles, FBP has streak artefacts and the angular sampling would have to enter
the forward model.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from geometry.lifted import LiftedGrid


class ImageDomainAnalyzer(torch.nn.Module):
    """Image (B, 1, n, n) on (x1, x2)  ->  coefficients (B, 1, K, n, n) on (theta, x1, x2).

    Parameters
    ----------
    grid : LiftedGrid
        Target lifted grid; the image must have grid.n cells per axis.
    sigma_cells : float
        Width of the kernels across the edge, in cells.
    aspect : float
        Ratio between tangential and normal kernel width.  It sets the
        orientation selectivity (about 1 / aspect radians); larger values lose
        response at corners and on strongly curved edges.
    detrend : float
        Ratio L of the coarse (isotropic) scale subtracted from each kernel;
        0 disables detrending.
    """

    def __init__(self, grid: LiftedGrid, sigma_cells: float = 1.5, aspect: float = 2.0,
                 detrend: float = 0.0):
        super().__init__()
        self.grid, self.sigma_n, self.aspect, self.detrend = grid, sigma_cells, aspect, detrend
        self.sigma_t = aspect * sigma_cells
        self.sigma_coarse = detrend * sigma_cells if detrend else None
        self.radius = math.ceil(3.0 * max(self.sigma_t, self.sigma_coarse or 0))
        self.register_buffer("kernels", self._kernel_bank(), persistent=False)

    def _offsets(self):
        o = np.arange(-self.radius, self.radius + 1, dtype=np.float64)
        return np.meshgrid(o, o, indexing="ij")    # (Delta x1, Delta x2) in cells

    def _kernel_bank(self) -> torch.Tensor:
        D1, D2 = self._offsets()
        bank = []
        for th in self.grid.thetas():
            a = D1 * math.cos(th) + D2 * math.sin(th)     # along the normal
            b = -D1 * math.sin(th) + D2 * math.cos(th)    # along the tangent
            k = a * np.exp(-a**2 / (2 * self.sigma_n**2) - b**2 / (2 * self.sigma_t**2))
            if self.detrend:
                # both parts with unit response to a unit ramp along the normal,
                # so their difference annihilates locally linear intensity
                kc = a * np.exp(-(a**2 + b**2) / (2 * self.sigma_coarse**2))
                k = k / np.sum(k * a) - kc / np.sum(kc * a)
            # unit response to a unit step through the centre (cell-averaged,
            # the same step as in angular_response)
            k /= np.sum(k * np.clip(0.5 + a, 0, 1))
            bank.append(k)
        # conv2d is a correlation: out(x) = sum_D k(D) f(x + D), which is the
        # derivative along +omega for this kernel.
        return torch.as_tensor(np.stack(bank)[:, None], dtype=torch.float32)

    def filter_signed(self, f: torch.Tensor) -> torch.Tensor:
        """Signed coefficients (B, K, n, n)."""
        n = self.grid.n
        if f.dim() != 4 or f.shape[1] != 1 or tuple(f.shape[2:]) != (n, n):
            raise ValueError(f"expected (B, 1, {n}, {n}), got {tuple(f.shape)}")
        return F.conv2d(f, self.kernels.to(f.dtype), padding=self.radius)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.filter_signed(f).abs().unsqueeze(1)

    @torch.no_grad()
    def angular_response(self) -> torch.Tensor:
        """Matrix r[k, k'] = response of kernel k to a unit step across a straight
        line with normal theta_k' through the kernel centre (r[k, k] = 1).

        It depends only on k - k' (mod K, up to sign), so the forward model
        applies it as a circular convolution in theta.
        """
        D1, D2 = self._offsets()
        th = self.grid.thetas()
        # antialiased step (pixel-area approximation): a sharp 0/1 step on the
        # pixel grid makes r jagged and nonzero at 90 degrees
        steps = np.stack([np.clip(0.5 + D1 * math.cos(t) + D2 * math.sin(t), 0, 1) for t in th])
        k = self.kernels[:, 0].double().cpu().numpy()
        return torch.as_tensor(np.einsum("kab,lab->kl", k, steps), dtype=torch.float32)

