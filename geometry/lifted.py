"""Discrete grids on the lifted spaces Omega x RP^1.

Both the image wavefront set and the sinogram wavefront set are stored as
fields on a grid of the same shape:

* image side:     u[..., k, i1, i2]  at  (theta_k, x1_i1, x2_i2)
* sinogram side:  v[..., k, is, it]  at  (phi_k,   s_is,  t_it)

where the spatial / (s, t) coordinates are cell centres of a uniform
partition of [-1, 1] and theta_k = phi_k = pi * k / n_theta.  Tensors use the
layout (B, C, n_theta, n, n), so the orientation axis is the "depth" axis of a
3D convolution.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class LiftedGrid:
    """Uniform grid on [-1, 1]^2 x RP^1 (orientations mod pi).

    Parameters
    ----------
    n : int
        Number of cells per spatial axis.
    n_theta : int
        Number of orientation samples on [0, pi).
    """

    n: int
    n_theta: int

    @property
    def h(self) -> float:
        """Spatial cell size."""
        return 2.0 / self.n

    @property
    def dtheta(self) -> float:
        """Orientation cell size."""
        return math.pi / self.n_theta

    @property
    def cell_volume(self) -> float:
        """Volume element h^2 * dtheta of one voxel in Omega x RP^1."""
        return self.h**2 * self.dtheta

    @property
    def shape(self) -> tuple:
        return (self.n_theta, self.n, self.n)

    def coords(self) -> np.ndarray:
        """Cell centres of the spatial axis, shape (n,)."""
        return -1.0 + (np.arange(self.n) + 0.5) * self.h

    def thetas(self) -> np.ndarray:
        """Orientation samples pi * k / n_theta, shape (n_theta,)."""
        return np.arange(self.n_theta) * self.dtheta

    def coords_torch(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.coords(), device=device, dtype=dtype)

    def thetas_torch(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.thetas(), device=device, dtype=dtype)


def omega(theta):
    """Unit vector (cos theta, sin theta) along the last axis."""
    theta = np.asarray(theta)
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def omega_perp(theta):
    """Unit vector (-sin theta, cos theta) along the last axis."""
    theta = np.asarray(theta)
    return np.stack([-np.sin(theta), np.cos(theta)], axis=-1)


def image_to_sinogram_coords(x, theta):
    """Canonical relation of the parallel-beam ray transform, pointwise.

    The image wavefront point (x, theta) (normal direction theta mod pi) is
    seen by the ray with angle phi = theta and offset s = x . omega(theta);
    the sinogram covector is proportional to (1, -t) in (s, phi) coordinates,
    with t = x . omega_perp(theta) the position along the ray.

    Parameters
    ----------
    x : array_like, shape (..., 2)
    theta : array_like, shape (...)

    Returns
    -------
    s, phi, t : arrays of shape (...)
    """
    x = np.asarray(x)
    s = np.sum(x * omega(theta), axis=-1)
    t = np.sum(x * omega_perp(theta), axis=-1)
    return s, np.asarray(theta), t


def sinogram_to_image_coords(s, phi, t):
    """Inverse of :func:`image_to_sinogram_coords`."""
    s, t = np.asarray(s), np.asarray(t)
    x = s[..., None] * omega(phi) + t[..., None] * omega_perp(phi)
    return x, np.asarray(phi)
