"""Learned primal-dual network on the lifted spaces.

Unrolls a primal-dual scheme for the MAP problem (``research/background.md``,
Section 7)

    min_T  D(R_t C q(T); c) + lambda E(T),

with the primal variable on the lifted image Omega x S^1 and the dual variable
on the lifted sinogram of oriented lines (phi in [0, 2 pi), s, t).  As in
Adler & Oktem's LPD, the proximal steps are replaced by small CNNs and the
operator by C (forward) and C^{-1} (back), all on the double covers:

    d_{i+1} = d_i + Gamma_i([d_i, C p_i, c])
    p_{i+1} = p_i + Lambda_i([p_i, C^{-1} d_{i+1}])

Both CNNs are periodic in the fibre.  Working on the double covers avoids
the reflection in the identification (s, phi + pi, t) ~ (-s, phi, -t)
(``models/layers.py``) and lets the antiperiodic flux component b be
represented.  The output head projects onto antipodally symmetric currents
(a, c even, b odd), so the outputs are fields on RP^1 of the right parity
(Lemma 6.1).  Since C turns rotations into phi-shifts (Proposition 4.6), the
dual CNNs are exactly equivariant under rotations by multiples of pi / K.

Outputs (fields of shape (B, 1, K, n, n) on (theta, x1, x2)):
    a, b, c : the current tau = a X1 + c X2 + b X3, with a >= 0
    J       : contrast in (0, J_max]
    m       : membership 1 - exp(-a / a_ref)
where a_ref is the peak line density of a curve blurred over 2 cells, so a
well-resolved curve has m ~ 0.6 on its centre line.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from geometry.lifted import LiftedGrid
from operators.canonical_relation import CanonicalRelation

from .layers import LiftedBlock, LiftedConv, s1_to_rp1_even, s1_to_rp1_odd, sinogram_rp1_to_s1


def reference_line_density(grid: LiftedGrid, cells: float = 2.0) -> float:
    """Peak density 1 / (2 pi sigma_x sigma_theta) of a curve blurred over ``cells``."""
    return 1.0 / (2 * math.pi * cells * grid.h * cells * grid.dtheta)


class LiftedLPD(nn.Module):
    def __init__(self, grid: LiftedGrid, n_iter=5, n_primal=6, n_dual=6, hidden=24,
                 J_max=2.0, use_checkpoint=True):
        super().__init__()
        self.grid, self.n_iter = grid, n_iter
        self.n_primal, self.n_dual = n_primal, n_dual
        self.J_max, self.use_checkpoint = J_max, use_checkpoint
        self.a_ref = reference_line_density(grid)
        self.C = CanonicalRelation(grid, double_cover=True)
        self.dual = nn.ModuleList(LiftedBlock(n_dual, n_primal + 1, hidden)
                                  for _ in range(n_iter))
        self.primal = nn.ModuleList(LiftedBlock(n_primal, n_dual, hidden)
                                    for _ in range(n_iter))
        self.head = nn.Sequential(LiftedConv(n_primal, hidden), nn.PReLU(hidden),
                                  LiftedConv(hidden, 4))

    def _iteration(self, i, p, d, c):
        d = self.dual[i](d, torch.cat([self.C(p), c], dim=1))
        p = self.primal[i](p, self.C.inverse(d))
        return p, d

    def forward(self, c: torch.Tensor) -> dict:
        """c: normalized coefficient magnitudes (B, 1, K, n, n) on (phi, s, t), phi in [0, pi)."""
        B, _, K, n, _ = c.shape
        c = sinogram_rp1_to_s1(c)
        p = c.new_zeros(B, self.n_primal, 2 * K, n, n)
        d = c.new_zeros(B, self.n_dual, 2 * K, n, n)
        for i in range(self.n_iter):
            if self.use_checkpoint and self.training:
                p, d = checkpoint(self._iteration, i, p, d, c, use_reentrant=False)
            else:
                p, d = self._iteration(i, p, d, c)
        out = self.head(p)
        a = self.a_ref * F.softplus(s1_to_rp1_even(out[:, 0:1]))
        b = self.a_ref * s1_to_rp1_odd(out[:, 1:2])
        cn = self.a_ref * s1_to_rp1_even(out[:, 2:3])
        J = self.J_max * torch.sigmoid(s1_to_rp1_even(out[:, 3:4]))
        return {"a": a, "b": b, "c": cn, "J": J, "m": 1 - torch.exp(-a / self.a_ref)}
