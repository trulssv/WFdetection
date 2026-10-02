"""Fan-beam canonical relation and fan-to-parallel rebinning.

Every fan-beam measurement is a line integral, so fan data are parallel-beam
data in different line coordinates.  For the flat-detector geometry of
:func:`geometry.tomo.fan_geometry` (source R_beta (0, -R_s), detector point
R_beta (u, R_d), D = R_s + R_d) the ray (beta, u) is the line

    phi = beta - gamma,   s = R_s sin(gamma),   gamma = arctan(u / D),

with ray direction omega_perp(phi) (source -> detector).  Hence
g(beta, u) = R f(s(beta, u), phi(beta, u)), and the fan-beam canonical
relation is the parallel one composed with the cotangent lift of the line
reparametrization Phi: (beta, u) -> (s, phi)
(``research/background.md``, Proposition 4.7):

    (x, theta)  ->  (beta, u; sigma) with Phi(beta, u) = (x.omega(phi), phi),
                    phi in {theta, theta + pi},
                    sigma = DPhi^T (1, -t) = (-t, s_u - t phi_u),
                    t = x . omega_perp(phi).

The fibre coordinate t (position of the tangency point along the ray) is
intrinsic to the ray, so on lifted fields

    (C_fan u)(beta, u, t) = u(s omega(phi) + t omega_perp(phi), phi mod pi).

Over a full 2 pi scan every line is measured twice, by (beta, u) and by
(beta + pi - 2 gamma, -u).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from geometry.lifted import LiftedGrid, omega, omega_perp


# ---------------------------------------------------------------------------
# pointwise maps
# ---------------------------------------------------------------------------

def fan_to_line(beta, u, src_radius, det_radius):
    """(beta, u) -> (s, phi) of the measured line; phi is not reduced mod pi."""
    D = src_radius + det_radius
    gamma = np.arctan(np.asarray(u) / D)
    return src_radius * np.sin(gamma), np.asarray(beta) - gamma


def reduce_line(s, phi):
    """Representative with phi in [0, pi), using (s, phi) ~ (-s, phi + pi)."""
    k = np.floor(np.asarray(phi) / math.pi)
    sign = np.where(k % 2 == 0, 1.0, -1.0)
    return sign * s, phi - k * math.pi, sign


def line_to_fan(s, phi, src_radius, det_radius):
    """Both fan coordinates (beta, u) of the line (s, phi), beta in [0, 2 pi).

    Returns arrays of shape (2, ...): the measurement with ray direction
    omega_perp(phi) and the one with the opposite direction.  Lines with
    |s| >= src_radius are not measured (NaN).
    """
    D = src_radius + det_radius
    s, phi = np.asarray(s, float), np.asarray(phi, float)
    with np.errstate(invalid="ignore"):
        gamma = np.arcsin(s / src_radius)
    beta1, u1 = np.mod(phi + gamma, 2 * math.pi), D * np.tan(gamma)
    beta2, u2 = np.mod(phi + math.pi - gamma, 2 * math.pi), -D * np.tan(gamma)
    return np.stack([beta1, beta2]), np.stack([u1, u2])


def fan_line_jacobian(u, src_radius, det_radius):
    """Partial derivatives (s_u, phi_u) of Phi; s_beta = 0 and phi_beta = 1."""
    D = src_radius + det_radius
    u = np.asarray(u, float)
    r2 = u**2 + D**2
    return src_radius * D**2 / r2**1.5, -D / r2


def image_to_fan_wavefront(x, theta, src_radius, det_radius):
    """Fan-data wavefront points of the image wavefront point (x, theta).

    Returns a dict of arrays with a leading axis of length 2 (the two
    measurements of the tangent line): ``beta``, ``u``, ``t`` and the unit
    covector ``sigma`` (..., 2) in (beta, u) coordinates (defined up to sign).
    """
    x, theta = np.asarray(x, float), np.asarray(theta, float)
    out = {k: [] for k in ("beta", "u", "t", "sigma")}
    for phi in (theta, theta + math.pi):
        s = np.sum(x * omega(phi), -1)
        t = np.sum(x * omega_perp(phi), -1)
        D = src_radius + det_radius
        with np.errstate(invalid="ignore"):
            gamma = np.arcsin(s / src_radius)
        u = D * np.tan(gamma)
        beta = np.mod(phi + gamma, 2 * math.pi)
        s_u, phi_u = fan_line_jacobian(u, src_radius, det_radius)
        sig = np.stack([-t, s_u - t * phi_u], -1)
        sig /= np.linalg.norm(sig, axis=-1, keepdims=True)
        for k, v in zip(("beta", "u", "t", "sigma"), (beta, u, t, sig)):
            out[k].append(v)
    return {k: np.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# lifted operator and rebinning
# ---------------------------------------------------------------------------

def _uniform(x, name):
    x = np.asarray(x, float)
    d = np.diff(x)
    if not np.allclose(d, d[0], rtol=1e-4):
        raise ValueError(f"{name} must be uniformly spaced")
    return x, float(d[0])


class FanBeamCanonicalRelation(torch.nn.Module):
    """Lifted image field (B, C, K, n, n) on (theta, x1, x2)  ->
    lifted fan-data field (B, C, n_beta, n_u, n_t) on (beta, u, t).

    ``ts`` defaults to the spatial cell centres of the grid.  Sampling is
    trilinear in (theta, x1, x2) with periodic theta, so the input must be
    pi-periodic in theta (memberships, densities; not the flux component b).
    """

    def __init__(self, grid: LiftedGrid, betas, us, src_radius, det_radius, ts=None):
        super().__init__()
        self.grid = grid
        betas, us = np.asarray(betas, float), np.asarray(us, float)
        ts = grid.coords() if ts is None else np.asarray(ts, float)
        s, phi = fan_to_line(betas[:, None], us[None, :], src_radius, det_radius)
        x = (s[..., None, None] * omega(phi)[:, :, None, :]
             + ts[None, None, :, None] * omega_perp(phi)[:, :, None, :])   # (nb, nu, nt, 2)
        th = np.broadcast_to(np.mod(phi, math.pi)[:, :, None], x.shape[:-1])
        K = grid.n_theta
        th_idx = th / grid.dtheta + 1.0          # +1: one periodic slice of padding
        g = np.stack([x[..., 1], x[..., 0], 2 * (th_idx + 0.5) / (K + 2) - 1], -1)
        self.register_buffer("sample_grid", torch.as_tensor(g[None], dtype=torch.float32),
                             persistent=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        up = torch.cat([u[:, :, -1:], u, u[:, :, :1]], dim=2)
        g = self.sample_grid.to(u.device, u.dtype).expand(u.shape[0], -1, -1, -1, -1)
        return F.grid_sample(up, g, mode="bilinear", padding_mode="zeros", align_corners=False)


class FanToParallel(torch.nn.Module):
    """Resample a full-scan fan sinogram (B, 1, n_beta, n_u) onto a parallel grid
    (B, 1, n_phi, n_s), averaging the two measurements of each line.

    betas must uniformly cover [0, 2 pi); lines outside the detector are 0.
    """

    def __init__(self, betas, us, src_radius, det_radius, phis, s):
        super().__init__()
        betas, db = _uniform(betas, "betas")
        us, du = _uniform(us, "us")
        if not np.isclose(len(betas) * db, 2 * math.pi, rtol=1e-3):
            raise ValueError("betas must cover [0, 2 pi)")
        P, S = np.meshgrid(np.asarray(phis, float), np.asarray(s, float), indexing="ij")
        B, U = line_to_fan(S, P, src_radius, det_radius)            # (2, n_phi, n_s)
        nb, nu = len(betas), len(us)
        b_idx = (np.mod(B - betas[0], 2 * math.pi)) / db + 1.0       # +1: periodic padding
        u_idx = (U - us[0]) / du
        valid = np.isfinite(U) & (u_idx >= -0.5) & (u_idx <= nu - 0.5)
        g = np.stack([2 * (u_idx + 0.5) / nu - 1, 2 * (b_idx + 0.5) / (nb + 2) - 1], -1)
        g = np.where(valid[..., None], g, 2.0)                       # outside -> zero
        self.register_buffer("sample_grid", torch.as_tensor(g, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("weight", torch.as_tensor(valid.sum(0), dtype=torch.float32),
                             persistent=False)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        gp = torch.cat([g[..., -1:, :], g, g[..., :1, :]], dim=-2)
        out = 0
        for k in range(2):
            grid = self.sample_grid[k][None].to(g.device, g.dtype).expand(g.shape[0], -1, -1, -1)
            out = out + F.grid_sample(gp, grid, mode="bilinear", padding_mode="zeros",
                                      align_corners=False)
        return out / self.weight.clamp_min(1).to(g.dtype)
