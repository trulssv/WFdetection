"""ODL ray transform as a differentiable torch module.

The ODL operator is evaluated on numpy arrays (ASTRA does the work, on the GPU
when available) and wrapped in an autograd function whose backward pass is the
ODL adjoint.  Batches are looped over, which is fine for data generation and
baselines; training will mostly happen on lifted fields, where the cheap
operator is :class:`operators.canonical_relation.CanonicalRelation`.
"""

import numpy as np
import torch
from odl.applications import tomo


def _apply(op, x: torch.Tensor) -> torch.Tensor:
    """Apply an ODL operator to every (B, C)-slice of x."""
    lead = x.shape[:-2]
    xs = x.detach().reshape(-1, *x.shape[-2:]).cpu().numpy()
    ys = np.stack([np.asarray(op(op.domain.element(xi)).asarray()) for xi in xs])
    y = torch.from_numpy(ys.astype(np.float32)).to(x.device)
    return y.reshape(*lead, *y.shape[-2:])


class _OdlLinearFunction(torch.autograd.Function):
    """Autograd wrapper of a linear ODL operator A.

    ODL's ``A.adjoint`` is the adjoint for the *weighted* inner products
    <x, x'>_X = w_X sum x x' (w = cell volume), whereas autograd needs the
    matrix transpose: A^T = (w_X / w_Y) A^*.
    """

    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op
        return _apply(op, x)

    @staticmethod
    def backward(ctx, grad):
        op = ctx.op
        scale = op.domain.cell_volume / op.range.cell_volume
        return scale * _apply(op.adjoint, grad), None


class RayTransform(torch.nn.Module):
    """y = R f for f of shape (..., n, n); y has shape (..., n_angles, n_det)."""

    def __init__(self, space, geometry, impl=None):
        super().__init__()
        if impl is None:
            impl = "astra_cuda" if tomo.ASTRA_CUDA_AVAILABLE else "astra_cpu"
        self.space, self.geometry = space, geometry
        self.op = tomo.RayTransform(space, geometry, impl=impl)
        self._fbp = {}

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return _OdlLinearFunction.apply(f, self.op)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """L2 adjoint R^* (ODL's weighted adjoint, i.e. the continuous back-projection)."""
        return _OdlLinearFunction.apply(y, self.op.adjoint)

    def fbp(self, y: torch.Tensor, filter_type="Hann", frequency_scaling=1.0) -> torch.Tensor:
        """Filtered back-projection (not differentiable; used for baselines)."""
        key = (filter_type, frequency_scaling)
        if key not in self._fbp:
            self._fbp[key] = tomo.fbp_op(self.op, filter_type=filter_type,
                                         frequency_scaling=frequency_scaling)
        return _apply(self._fbp[key], y)
