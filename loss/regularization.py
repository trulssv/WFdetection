"""Curve prior on lifted wavefront-set fields: the energy of a Legendrian current.

Theory and proofs: ``research/background.md``, Sections 2, 3 and 6.

The wavefront set of a cartoon image is a finite union of Legendrian curves
in the cosphere bundle X = Omega x S^1 (Corollary 2.9).  It is represented as
a 1-current with density

    tau = a X1 + c X2 + b X3,
    X1 = (-sin th, cos th, 0),  X2 = (cos th, sin th, 0),  X3 = d/dth,

where th is the normal angle, X1 the canonical tangent J n (so that a >= 0
for a positively Legendrian current, Definition 3.3), b = a * kappa the
rotation rate and c the normal component, which vanishes for sharp curves
but not for blurred ones (Proposition 6.3).  The prior is the Gibbs measure
of the energy (Definition 3.5)

    E(T) = M_g(T) + nu M(dT),
    M_g(T) = int_X sqrt(a^2 + zeta^2 c^2 + xi^2 b^2) dX      (mass in the metric g),
    M(dT)  = int_X |X1 a + X2 c + d_th b| dX                 (boundary mass),

using that X1, X2, X3 are divergence free.  For a union of disjoint
Legendrian curves E = sum_k (L_xi(gamma_k) + nu #endpoints_k)
(Proposition 3.6), which gives the desired properties (Theorem 3.8):
independence, smoothness, stability (+2 nu per break), closure, and sparsity
(every closed component costs >= 2 pi xi when a >= 0, Theorem 3.9).

``length='elastica'`` replaces the mass by alpha sqrt(a^2 + zeta^2 c^2) +
beta b^2 / a, the perspective form of Euler's elastica (corners forbidden).

Discretization (Section 6): theta in [0, pi) with a, c pi-periodic and b
pi-antiperiodic (Lemma 6.1); fourth-order central differences; fields must be
resolved over >= 2 cells for the boundary mass to be accurate.  Integrals are
over Omega x RP^1, half the value on the double cover.
"""

import torch

from geometry.lifted import LiftedGrid


def _d_space(u: torch.Tensor, dim: int, h: float) -> torch.Tensor:
    """Fourth-order central difference along ``dim`` (negative) with zero extension."""
    spec = [0, 0] * (-dim - 1) + [2, 2]   # F.pad lists the last axis first
    up = torch.nn.functional.pad(u, spec)
    n = u.shape[dim]
    f = lambda o: up.narrow(dim, 2 + o, n)
    return (-f(2) + 8 * f(1) - 8 * f(-1) + f(-2)) / (12 * h)


class CurvePrior(torch.nn.Module):
    """R(a, b, c) for fields of shape (..., n_theta, n, n) on (theta, x1, x2);
    c defaults to zero (an exactly horizontal field).

    Parameters
    ----------
    grid : LiftedGrid
    xi : float
        Length scale converting angle into length ('subriemannian').
    zeta : float
        Weight of the normal component c (horizontality), zeta >= 1.
    nu : float
        Cost per curve endpoint.
    length : {'subriemannian', 'elastica'}
    alpha, beta : float
        Elastica weights.
    eps : float
        Smoothing of |.| and sqrt and regularization of b^2 / a, for gradients.
    """

    def __init__(self, grid: LiftedGrid, xi=0.1, nu=1.0, zeta=2.0, length="subriemannian",
                 alpha=1.0, beta=0.01, eps=1e-6):
        super().__init__()
        if length not in ("subriemannian", "elastica"):
            raise ValueError(length)
        self.grid, self.xi, self.nu, self.zeta, self.length = grid, xi, nu, zeta, length
        self.alpha, self.beta, self.eps = alpha, beta, eps
        th = grid.thetas_torch(dtype=torch.float64)
        self.register_buffer("_sin", torch.sin(th)[:, None, None].float(), persistent=False)
        self.register_buffer("_cos", torch.cos(th)[:, None, None].float(), persistent=False)

    def x1_derivative(self, a: torch.Tensor) -> torch.Tensor:
        """X1 a = -sin(theta) d_x1 a + cos(theta) d_x2 a."""
        h = self.grid.h
        return -self._sin * _d_space(a, -2, h) + self._cos * _d_space(a, -1, h)

    def x2_derivative(self, c: torch.Tensor) -> torch.Tensor:
        """X2 c = cos(theta) d_x1 c + sin(theta) d_x2 c."""
        h = self.grid.h
        return self._cos * _d_space(c, -2, h) + self._sin * _d_space(c, -1, h)

    def theta_derivative(self, b: torch.Tensor) -> torch.Tensor:
        """Fourth-order central difference in theta with b(theta + pi) = -b(theta)."""
        bp = torch.cat([-b[..., -2:, :, :], b, -b[..., :2, :, :]], dim=-3)
        n = b.shape[-3]
        f = lambda o: bp[..., 2 + o:2 + o + n, :, :]
        return (-f(2) + 8 * f(1) - 8 * f(-1) + f(-2)) / (12 * self.grid.dtheta)

    def divergence(self, a, b, c=None):
        d = self.x1_derivative(a) + self.theta_derivative(b)
        return d if c is None else d + self.x2_derivative(c)

    def length_density(self, a, b, c=None):
        spatial2 = a**2 if c is None else a**2 + (self.zeta * c)**2
        if self.length == "subriemannian":
            return torch.sqrt(spatial2 + (self.xi * b)**2 + self.eps**2) - self.eps
        spatial = torch.sqrt(spatial2 + self.eps**2) - self.eps
        return self.alpha * spatial + self.beta * b**2 / (a.clamp_min(0) + self.eps)

    def _integrate(self, f):
        return f.sum(dim=(-3, -2, -1)) * self.grid.cell_volume

    def mass(self, a, b, c=None):
        """M_g(T), the mass of the current in the metric g_{xi, zeta}."""
        return self._integrate(self.length_density(a, b, c))

    def boundary_mass(self, a, b, c=None):
        """M(dT) = int |div tau|; counts endpoints (Proposition 3.6)."""
        d = self.divergence(a, b, c)
        return self._integrate(torch.sqrt(d**2 + self.eps**2) - self.eps)

    def forward(self, a, b, c=None):
        """E(T) = M_g(T) + nu M(dT) = -log prior + const, reduced over the last three axes."""
        return self.mass(a, b, c) + self.nu * self.boundary_mass(a, b, c)
