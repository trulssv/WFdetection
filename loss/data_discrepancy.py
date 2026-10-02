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

``MixtureLikelihood`` (baseline).  Treats every coefficient independently as
noise or edge,

    p(c | v) = v p1(c) + (1 - v) p0(c),   v = C u,

with p0 the known noise distribution.  Every voxel of a streak is then strong
evidence for an edge, so its posterior covers all tangent lines of every
edge (63% of the image in milestone 1).  Kept for comparison.
"""

import math

import torch


class MicrolocalForwardModel(torch.nn.Module):
    """c_hat = R_t (C q): predicted coefficient magnitudes, shape (B, 1, K, n, n).

    Parameters
    ----------
    C : CanonicalRelation
    fibre_response : tensor (n_t, n_t)
        r[j, j'] from ``SinogramAnalyzer.fibre_response``; magnitudes are used.
    """

    def __init__(self, C, fibre_response):
        super().__init__()
        self.C = C
        self.register_buffer("r", fibre_response.abs().float())

    def forward(self, q):
        v = self.C(q)
        return torch.einsum("...k,jk->...j", v, self.r.to(v.dtype))


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
