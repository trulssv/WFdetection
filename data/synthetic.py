"""On-the-fly synthetic tomography data for unsupervised training.

Each sample is a random cartoon phantom (``data.phantoms.random_phantom``),
optionally plus a smooth background (which has no wavefront set and tests
that the analyzer and the network ignore smooth structure), projected with
the ODL parallel-beam ray transform, corrupted by Gaussian noise and analyzed
by M.  Training uses only the coefficients; the lifted wavefront-set
membership is returned for validation only.
"""

import math

import numpy as np
import torch

from data.phantoms import lifted_membership, random_phantom
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from operators.microlocal import SinogramAnalyzer
from operators.radon import RayTransform


def normalize_coefficients(c: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    """Divide each sample by a high quantile of |c| (contrast normalization)."""
    flat = c.abs().flatten(1)
    k = max(int(q * flat.shape[1]), 1)
    scale = flat.kthvalue(k, dim=1).values.clamp_min(1e-12)
    return c / scale.view(-1, *([1] * (c.dim() - 1)))


def smooth_background(rng, n, n_bumps=4, amplitude=0.3):
    c = -1 + (np.arange(n) + 0.5) * 2 / n
    X1, X2 = np.meshgrid(c, c, indexing="ij")
    f = np.zeros((n, n))
    for _ in range(n_bumps):
        x0 = rng.uniform(-0.6, 0.6, 2)
        s = rng.uniform(0.25, 0.5)
        f += rng.uniform(-1, 1) * np.exp(-((X1 - x0[0])**2 + (X2 - x0[1])**2) / (2 * s**2))
    # smooth taper to zero outside the disk: a hard cut-off would be an
    # (unlabelled) edge
    r = np.sqrt(X1**2 + X2**2)
    taper = np.clip((0.95 - r) / 0.25, 0, 1)
    taper = taper**2 * (3 - 2 * taper)          # C^1 smoothstep
    return amplitude * f * taper


class SyntheticTomography:
    """Generator of (coefficients, sinogram, image, membership) batches.

    Parameters
    ----------
    grid : LiftedGrid
    n_angles, n_det : sinogram size (dense angles over [0, pi))
    noise : sinogram noise std relative to max |y| of each sample
    background : amplitude of the smooth background (0 to disable)
    """

    def __init__(self, grid: LiftedGrid, n_angles=None, n_det=None, noise=0.01,
                 background=0.3, device="cpu"):
        n = grid.n
        self.grid, self.noise, self.background, self.device = grid, noise, background, device
        self.geometry = parallel_geometry(n_angles or 3 * n, n_det or 2 * n)
        self.R = RayTransform(image_space(n), self.geometry)
        phis, s = sinogram_axes(self.geometry)
        self.M = SinogramAnalyzer(grid, phis, s).to(device)

    def sample(self, rng: np.random.Generator, batch: int, labels: bool = False) -> dict:
        n = self.grid.n
        phantoms = [random_phantom(rng) for _ in range(batch)]
        f = np.stack([p.render(n) + (smooth_background(rng, n, amplitude=self.background)
                                     if self.background else 0) for p in phantoms])
        f = torch.as_tensor(f, dtype=torch.float32)[:, None]
        y = self.R(f).to(self.device)
        sig = self.noise * y.abs().flatten(1).max(1).values.view(-1, 1, 1, 1)
        y = y + sig * torch.randn(y.shape, generator=None, device=self.device)
        with torch.no_grad():
            c = normalize_coefficients(self.M(y))
        out = {"c": c, "y": y, "f": f}
        if labels:
            m = np.stack([lifted_membership(self.grid, p.samples(self.grid.h / 2, "full"))
                          for p in phantoms])
            out["membership"] = torch.as_tensor(m, dtype=torch.float32)[:, None]
        return out
