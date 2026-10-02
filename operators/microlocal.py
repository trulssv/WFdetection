"""Fixed microlocal analyzer M: sinogram  ->  lifted sinogram field.

A singularity of the sinogram along a curve s = h(phi) has slope
dh/dphi = t (the position along the ray where it touches the edge, see
:mod:`operators.canonical_relation`), i.e. its conormal in (s, phi)
coordinates is proportional to (1, -t).  M therefore filters the sinogram
with a bank of anisotropic derivative-of-Gaussian kernels: kernel j
differentiates along the conormal (1, -t_j) and smooths along the tangent
(t_j, 1).  The magnitudes are then resampled from the native (phi, s) grid
onto the lifted grid (phi_k = theta_k, s_i, t_j), giving

    c[..., k, i, j] = |(K_j * Lambda y)(s_i, phi_k)|.

``Lambda`` is an optional half-order ramp filter |sigma_s|^(1/2) along s.  R
is an FIO of order -1/2, so a jump in f becomes a square-root singularity in
R f; Lambda^(1/2) turns it back into a jump-type singularity so that
coefficient magnitudes scale like the image contrast.

M assumes dense angular sampling (the kernels need several angles across
their support).  Sparse-view data needs a different analyzer; see
``operators/README.md``.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from geometry.lifted import LiftedGrid


def half_ramp(y: torch.Tensor, ds: float) -> torch.Tensor:
    """Apply the Fourier multiplier |sigma|^(1/2) along the last (detector) axis."""
    n = y.shape[-1]
    yh = torch.fft.rfft(y, n=2 * n, dim=-1)
    freq = 2 * math.pi * torch.fft.rfftfreq(2 * n, d=ds).to(y.device, y.dtype)
    return torch.fft.irfft(yh * freq.sqrt(), n=2 * n, dim=-1)[..., :n]


def _uniform_spacing(x: np.ndarray, name: str) -> float:
    d = np.diff(x)
    if not np.allclose(d, d[0], rtol=1e-4):
        raise ValueError(f"{name} must be uniformly spaced")
    return float(d[0])


class SinogramAnalyzer(torch.nn.Module):
    """Sinogram (B, 1, n_phi, n_det)  ->  coefficients (B, 1, K, n, n) on (phi, s, t).

    Parameters
    ----------
    grid : LiftedGrid
        Target lifted grid; filter j has slope t_j = grid.coords()[j].
    phis : array, shape (n_phi,)
        Native projection angles, uniformly covering [0, pi).
    s_det : array, shape (n_det,)
        Detector cell centres, uniform and symmetric about 0.
    sigma_n : float, optional
        Width of the kernels across the singular curve, in (s, phi) units.
        Default: 1.5 times the coarser of the two sample spacings.
    aspect : float
        Ratio between tangential and normal kernel width.  It sets the
        orientation selectivity and hence the resolution in t (roughly
        1 / aspect in slope units); larger values lose response on strongly
        curved singular curves.
    use_half_ramp : bool
        Apply Lambda^(1/2) before filtering.
    """

    def __init__(self, grid: LiftedGrid, phis, s_det, sigma_n=None, aspect=6.0,
                 use_half_ramp=True):
        super().__init__()
        phis, s_det = np.asarray(phis, float), np.asarray(s_det, float)
        self.grid = grid
        self.dphi = _uniform_spacing(phis, "phis")
        self.ds = _uniform_spacing(s_det, "s_det")
        if not np.isclose(len(phis) * self.dphi, math.pi, rtol=1e-3):
            raise ValueError("phis must cover [0, pi) (full angular range)")
        if not np.allclose(s_det, -s_det[::-1], atol=1e-6 * self.ds):
            raise ValueError("detector must be symmetric about 0")
        self.phi0, self.s0 = float(phis[0]), float(s_det[0])
        self.n_phi, self.n_det = len(phis), len(s_det)
        self.use_half_ramp = use_half_ramp

        self.sigma_n = sigma_n if sigma_n is not None else 1.5 * max(self.dphi, self.ds)
        self.sigma_t = aspect * self.sigma_n
        radius = 3.0 * self.sigma_t
        kp, ks = math.ceil(radius / self.dphi), math.ceil(radius / self.ds)
        self.pad = kp + 2
        self.register_buffer("kernels", self._kernel_bank(grid.coords(), kp, ks),
                             persistent=False)
        self.register_buffer("sample_grid", self._sample_grid(), persistent=False)

    def _kernel_bank(self, ts, kp, ks) -> torch.Tensor:
        P, S = np.meshgrid(np.arange(-kp, kp + 1) * self.dphi,
                           np.arange(-ks, ks + 1) * self.ds, indexing="ij")
        bank = []
        for t in ts:
            norm = math.hypot(1.0, t)
            a = (S - t * P) / norm   # coordinate along the conormal (1, -t)
            b = (t * S + P) / norm   # coordinate along the tangent (t, 1)
            k = a * np.exp(-a**2 / (2 * self.sigma_n**2) - b**2 / (2 * self.sigma_t**2))
            k /= np.sum(k[a > 0])    # unit response to a unit step across the curve
            bank.append(k)
        return torch.as_tensor(np.stack(bank)[:, None], dtype=torch.float32)

    def _sample_grid(self) -> torch.Tensor:
        """Normalized grid_sample coordinates of (theta_k, s_i) in the padded sinogram."""
        h_pad = self.n_phi + 2 * self.pad
        phi_idx = (self.grid.thetas() - (self.phi0 - self.pad * self.dphi)) / self.dphi
        s_idx = (self.grid.coords() - self.s0) / self.ds
        P, S = np.meshgrid(phi_idx, s_idx, indexing="ij")
        g = np.stack([2 * (S + 0.5) / self.n_det - 1, 2 * (P + 0.5) / h_pad - 1], axis=-1)
        return torch.as_tensor(g[None], dtype=torch.float32)

    def pad_phi(self, y: torch.Tensor) -> torch.Tensor:
        """Extend in phi using R f(s, phi + pi) = R f(-s, phi)."""
        p = self.pad
        return torch.cat([y[..., -p:, :].flip(-1), y, y[..., :p, :].flip(-1)], dim=-2)

    def filter_native(self, y: torch.Tensor) -> torch.Tensor:
        """Signed coefficients on the padded native grid, (B, n_t, n_phi + 2 pad, n_det)."""
        if y.dim() != 4 or y.shape[1] != 1 or tuple(y.shape[2:]) != (self.n_phi, self.n_det):
            raise ValueError(f"expected (B, 1, {self.n_phi}, {self.n_det}), got {tuple(y.shape)}")
        if self.use_half_ramp:
            y = half_ramp(y, self.ds)
        k = self.kernels.to(y.dtype)
        return F.conv2d(self.pad_phi(y), k, padding=(k.shape[-2] // 2, k.shape[-1] // 2))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        c = self.filter_native(y).abs()
        g = self.sample_grid.to(c.device, c.dtype).expand(c.shape[0], -1, -1, -1)
        c = F.grid_sample(c, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        # (B, n_t, K, n_s) -> (B, 1, K, n_s, n_t)
        return c.permute(0, 2, 3, 1).unsqueeze(1)

    @torch.no_grad()
    def fibre_response(self) -> torch.Tensor:
        """Matrix r[j, j'] = response of kernel j to a unit step across a straight
        singular line with conormal (1, -t_j') through the kernel centre.

        Column j' is the predicted coefficient profile along t of a jump-type
        singularity at t_j' (``research/background.md``, Proposition 5.3); the
        diagonal is 1 by the kernel normalization.
        """
        kp, ks = self.kernels.shape[-2] // 2, self.kernels.shape[-1] // 2
        P, S = np.meshgrid(np.arange(-kp, kp + 1) * self.dphi,
                           np.arange(-ks, ks + 1) * self.ds, indexing="ij")
        ts = self.grid.coords()
        steps = np.stack([(S - t * P > 0).astype(np.float64) for t in ts])   # (n_t, kp, ks)
        k = self.kernels[:, 0].double().cpu().numpy()
        return torch.as_tensor(np.einsum("jab,lab->jl", k, steps), dtype=torch.float32)

    @torch.no_grad()
    def coefficient_noise_std(self, sigma: float, n_trials: int = 8, device=None) -> torch.Tensor:
        """Std of the signed coefficients under white sinogram noise of std ``sigma``,
        per filter j (shape (n_t,)), estimated by Monte Carlo."""
        noise = sigma * torch.randn(n_trials, 1, self.n_phi, self.n_det, device=device)
        c = self.filter_native(noise)[:, :, self.pad:-self.pad]
        return c.std(dim=(0, 2, 3))
