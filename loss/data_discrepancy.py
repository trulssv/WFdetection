"""Unsupervised data terms on the microlocal sinogram coefficients c = M y.

Two models are implemented (``research/background.md``, Section 5).

``ForwardModelDiscrepancy`` (default; Definition 5.4).  The magnitude of the
coefficients is predicted from a nonnegative amplitude field q on the lifted
image (q = contrast x membership):

    |c|(phi, s, t)  =  sum_t' r(t, t') (C q)(phi, s, t')  +  noise,

where C is the canonical relation and r the fibre response of the filter bank
(``SinogramAnalyzer.fibre_response``, Proposition 5.3): a single jump at t0
produces the known profile r(., t0) along the fibre, which is many cells
wide.  The noise is modelled as Laplace (heavy tails absorb the model error
at crossings and curved singular curves), so

    D(q; c) = mean | |c| - R_t C q | / scale        (smoothed absolute value).

Explaining a streak along t with one point is then the cheapest option; the
tangent-line ambiguity becomes a (badly conditioned) deconvolution along the
fibre, which the curve prior regularizes.

``OrientedEdgeForwardModel`` (full aperture).  The same Laplace term on the
lifted *image* coefficients b = |W FBP y| (``operators/image_analyzer.py``),
predicted by a local model

    b(x, theta)  =  G_x * sum_theta' |r(theta, theta')| q(x, theta')  +  noise,

with r the angular response of the oriented filter bank and G_x a small
spatial blur.  The response of an edge is local in x, so a streak along a
tangent line does not explain b: the data term itself fixes t, which the
fibre response leaves to the prior.

``MixtureLikelihood`` (baseline).  Treats every coefficient independently as
noise or edge,

    p(c | v) = v p1(c) + (1 - v) p0(c),   v = C u,

with p0 the known noise distribution.  Every voxel of a streak is then strong
evidence for an edge, so its posterior covers all tangent lines of every
edge (63% of the image in milestone 1).  Kept for comparison.
"""

import math

import torch
import torch.nn.functional as F


class MicrolocalForwardModel(torch.nn.Module):
    """c_hat = R_t (C q) / w: predicted coefficient magnitudes, shape (B, 1, K, n, n).

    The sum over t' is divided by the width w = sqrt(2 pi) * blur_cells of a
    curve resolved over ``blur_cells`` cells (the resolution of the network's
    outputs), so that q = 1 on such a curve predicts a streak of unit height:
    the amplitude q then has the units of |c|.

    Parameters
    ----------
    C : CanonicalRelation
    fibre_response : tensor (n_t, n_t)
        r[j, j'] from ``SinogramAnalyzer.fibre_response``; magnitudes are used.
    blur_cells : float
    """

    def __init__(self, C, fibre_response, blur_cells: float = 2.0):
        super().__init__()
        self.C = C
        w = math.sqrt(2 * math.pi) * blur_cells
        self.register_buffer("r", fibre_response.abs().float() / w)

    def forward(self, q):
        v = self.C(q)
        return torch.einsum("...k,jk->...j", v, self.r.to(v.dtype))


class OrientedEdgeForwardModel(torch.nn.Module):
    """b_hat = G_x * (r (*)_theta q) / w: predicted image-side coefficient
    magnitudes, shape (B, 1, K, n, n) on (theta, x1, x2).

    A sharp edge of contrast J gives b = J |r(theta, theta_e)| exp(-d^2 / 2 sigma^2),
    where sigma (``sigma_cells``) is the width of an edge in b: the kernel width
    combined with the blur of the FBP (about 2.6 cells for the default kernel
    sigma = 1.5 and a Hann-filtered FBP; with the kernel width alone, the
    network has to widen q to fit the edges).  The network's amplitude q is resolved over
    ``blur_cells`` cells in x and theta, so G_x is a Gaussian of std
    sqrt(max(sigma^2 - blur^2, 0)) cells, and w is fixed numerically such that
    q = 1 on a straight curve resolved over ``blur_cells`` predicts unit height:
    q then has the units of b.

    Parameters
    ----------
    grid : LiftedGrid
    angular_response : tensor (K, K)
        r[k, k'] from ``ImageDomainAnalyzer.angular_response``; magnitudes are used.
    sigma_cells : float
        Width of an edge in b, in cells (see above).
    blur_cells : float
    sigma_coarse : float, optional
        Coarse scale of a detrended analyzer (``ImageDomainAnalyzer.sigma_coarse``).
        The signed normal profile of an edge is then
        N(d; sigma) - N(d; sigma_coarse): the prediction is
        |v - G_k * v| with v the theta-blurred q and G_k a normalized Gaussian of
        std sqrt(sigma_coarse^2 - blur^2) along the normal omega(theta_k).
    """

    def __init__(self, grid, angular_response, sigma_cells: float, blur_cells: float = 2.0,
                 sigma_coarse: float = None):
        super().__init__()
        self.register_buffer("r", angular_response.abs().float())
        if not sigma_coarse:
            self.coarse = None
        else:
            sc = math.sqrt(max(sigma_coarse**2 - blur_cells**2, 1.0))
            rad = math.ceil(3 * sc)
            o = torch.arange(-rad, rad + 1, dtype=torch.float64)
            D1, D2 = torch.meshgrid(o, o, indexing="ij")
            th = torch.as_tensor(grid.thetas(), dtype=torch.float64)[:, None, None]
            a = D1 * torch.cos(th) + D2 * torch.sin(th)
            b = -D1 * torch.sin(th) + D2 * torch.cos(th)
            g = torch.exp(-a**2 / (2 * sc**2) - b**2 / (2 * 0.6**2))
            g = g / g.sum(dim=(-2, -1), keepdim=True)
            self.register_buffer("coarse", g[:, None].float())   # (K, 1, kh, kw)
        s = math.sqrt(max(sigma_cells**2 - blur_cells**2, 0.0))
        if s > 0.3:
            rad = math.ceil(3 * s)
            g = torch.exp(-torch.arange(-rad, rad + 1, dtype=torch.float32)**2 / (2 * s**2))
            self.register_buffer("g", g / g.sum())
        else:
            self.g = None
        self.register_buffer("w", torch.ones(()))
        self.w = self._unit_height(grid, blur_cells)

    @torch.no_grad()
    def _unit_height(self, grid, blur):
        K, n = grid.n_theta, grid.n
        k0 = K // 4
        th0 = grid.thetas()[k0]
        x = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
        X1, X2 = torch.meshgrid(x, x, indexing="ij")
        d = X1 * math.cos(th0) + X2 * math.sin(th0)
        dk = (torch.arange(K) - k0 + K // 2) % K - K // 2
        q = torch.exp(-dk.float()[:, None, None]**2 / (2 * blur**2) - d**2 / (2 * blur**2))
        return self._blur(q[None, None])[0, 0, k0, n // 2, n // 2]

    def _blur(self, q):
        v = torch.einsum("bckij,lk->bclij", q, self.r.to(q.dtype))
        if self.g is not None:
            B, C, K, n1, n2 = v.shape
            g = self.g.to(v.dtype)
            p = len(g) // 2
            x = v.reshape(B * C * K, 1, n1, n2)
            x = F.conv2d(x, g.view(1, 1, -1, 1), padding=(p, 0))
            x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, p))
            v = x.reshape(B, C, K, n1, n2)
        if self.coarse is not None:
            B, C, K, n1, n2 = v.shape
            g = self.coarse.to(v.dtype)
            x = v.reshape(B * C, K, n1, n2)
            x = x - F.conv2d(x, g, padding=g.shape[-1] // 2, groups=K)
            v = x.reshape(B, C, K, n1, n2).abs()
        return v

    def forward(self, q):
        return self._blur(q) / self.w.to(q.dtype)


class ForwardModelDiscrepancy(torch.nn.Module):
    """Laplace negative log-likelihood mean(rho(|c| - c_hat)) / scale, with the
    smoothed absolute value rho(r) = sqrt(r^2 + delta^2) - delta."""

    def __init__(self, scale=1.0, delta=1e-2):
        super().__init__()
        self.scale, self.delta = scale, delta

    def forward(self, c_hat, c, mask=None):
        r = c.abs() - c_hat
        rho = torch.sqrt(r**2 + self.delta**2) - self.delta
        if mask is None:
            return rho.mean() / self.scale
        mask = mask.to(rho.dtype).expand_as(rho)
        return (rho * mask).sum() / mask.sum().clamp_min(1) / self.scale


class MixtureLikelihood(torch.nn.Module):
    """D(v; c) = -mean log(v p1(|c|) + (1 - v) p0(|c|)).

    Parameters
    ----------
    sigma0 : float or tensor
        Noise std of the signed coefficients; broadcastable against c (for
        example shape (n_t,) for a per-filter std, which broadcasts along the
        last axis).
    edge_scale : float
        Initial scale of the exponential edge distribution.
    learn_edge_scale : bool
        Make the edge scale a trainable parameter.
    """

    def __init__(self, sigma0, edge_scale=1.0, learn_edge_scale=True, eps=1e-6):
        super().__init__()
        self.register_buffer("sigma0", torch.as_tensor(sigma0, dtype=torch.float32))
        log_scale = torch.tensor(math.log(edge_scale))
        if learn_edge_scale:
            self.log_edge_scale = torch.nn.Parameter(log_scale)
        else:
            self.register_buffer("log_edge_scale", log_scale)
        self.eps = eps

    def log_p0(self, c):
        s = self.sigma0.clamp_min(self.eps)
        return math.log(2.0) - torch.log(s) - 0.5 * math.log(2 * math.pi) - c**2 / (2 * s**2)

    def log_p1(self, c):
        return -self.log_edge_scale - c / self.log_edge_scale.exp()

    def log_likelihood(self, v, c):
        """Elementwise log p(c | v)."""
        c = c.abs()
        v = v.clamp(self.eps, 1 - self.eps)
        return torch.logaddexp(torch.log(v) + self.log_p1(c),
                               torch.log1p(-v) + self.log_p0(c))

    def posterior(self, c, prior=0.5):
        """P(on wavefront set | c) for a constant prior probability."""
        c = c.abs()
        l1 = math.log(prior) + self.log_p1(c)
        l0 = math.log(1 - prior) + self.log_p0(c)
        return torch.sigmoid(l1 - l0)

    def forward(self, v, c, mask=None):
        ll = self.log_likelihood(v, c)
        if mask is None:
            return -ll.mean()
        mask = mask.to(ll.dtype).expand_as(ll)
        return -(ll * mask).sum() / mask.sum().clamp_min(1)
