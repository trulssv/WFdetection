"""Convolutions on lifted grids with the correct identifications in the fibre.

The network works on the double covers (``research/background.md``,
Section 7.3): the image side Omega x S^1 and the sinogram side of oriented
lines, phi in [0, 2 pi), both with 2K slices and plain periodic padding.  On
the RP^1 quotient of the sinogram side the identification
(s, phi + pi, t) ~ (-s, phi, -t) contains a point reflection of the (s, t)
plane, and a convolution is well defined there only if its kernel is
reflection symmetric; "twisted" padding would create a seam.  On the double
cover no such constraint arises.

Spatial axes are zero padded.  Tensors have layout (B, C, fibre, n, n).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_fibre(x: torch.Tensor, p: int) -> torch.Tensor:
    """Periodic padding of the fibre axis (dim 2) by p slices on each side."""
    if p == 0:
        return x
    return torch.cat([x[:, :, -p:], x, x[:, :, :p]], dim=2)


class LiftedConv(nn.Module):
    """3D convolution, periodic in the fibre and zero padded in space."""

    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv3d(c_in, c_out, k, padding=(0, self.p, self.p))

    def forward(self, x):
        return self.conv(pad_fibre(x, self.p))


class LiftedBlock(nn.Module):
    """Residual update  x + f([x, extra])  with a small CNN f."""

    def __init__(self, c_state, c_extra, hidden, n_layers=3):
        super().__init__()
        layers, c = [], c_state + c_extra
        for _ in range(n_layers - 1):
            layers += [LiftedConv(c, hidden), nn.PReLU(hidden)]
            c = hidden
        layers.append(LiftedConv(c, c_state))
        self.net = nn.Sequential(*layers)

    def forward(self, x, extra):
        return x + self.net(torch.cat([x, extra], dim=1))


def s1_to_rp1_even(x: torch.Tensor) -> torch.Tensor:
    """Project a field on S^1 (2K slices) to an antipodally even field on RP^1."""
    K = x.shape[2] // 2
    return 0.5 * (x[:, :, :K] + x[:, :, K:])


def s1_to_rp1_odd(x: torch.Tensor) -> torch.Tensor:
    """Project to an antipodally odd field (b(theta + pi) = -b(theta)), stored on RP^1."""
    K = x.shape[2] // 2
    return 0.5 * (x[:, :, :K] - x[:, :, K:])


def rp1_to_s1_even(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x], dim=2)


def sinogram_rp1_to_s1(v: torch.Tensor) -> torch.Tensor:
    """Extend a lifted sinogram field from phi in [0, pi) to [0, 2 pi) using
    v(s, phi + pi, t) = v(-s, phi, -t)."""
    return torch.cat([v, v.flip(-1, -2)], dim=2)
