"""Thin wavefront-set estimates and tolerance-based scores.

A wavefront set is a union of curves in Omega x RP^1, so it has no width.
Soft lifted fields (the analyzer's b, the network's membership m) are blurred
over a few cells; a thin estimate is their *ridge* set, obtained by non-maximum
suppression as in Canny's edge detector, but in the lifted space:

    (x, theta) is a ridge point  iff  u(x, theta) >= u(x +- omega(theta) h, theta)
                                  and u(x, theta) >= u(x, theta +- dtheta),

i.e. u is a local maximum across the curve (along the normal omega(theta))
and in orientation.  Along the curve u is not suppressed.

Thin estimates are compared with the thin (rasterized) analytic wavefront
set by precision and recall up to a tolerance, as in edge-detection
benchmarks: a predicted point is correct if a true point lies within
``tol_x`` cells in space and ``tol_theta`` cells in orientation, and
conversely for recall.  Scores are reported both in the lifted space and for
the spatial projection (singular support).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from geometry.lifted import LiftedGrid


def ridges(u: torch.Tensor, grid: LiftedGrid) -> torch.Tensor:
    """Boolean ridge mask of a lifted field u, shape (..., K, n, n)."""
    lead, (K, n, _) = u.shape[:-3], u.shape[-3:]
    x = u.reshape(-1, K, n, n).float()
    B = x.shape[0]
    th = torch.as_tensor(grid.thetas(), dtype=x.dtype, device=x.device)
    c = grid.coords_torch(device=x.device, dtype=x.dtype)
    a, b = torch.meshgrid(c, c, indexing="ij")                     # (x1, x2)
    keep = torch.ones_like(x, dtype=torch.bool)
    slices = x.permute(1, 0, 2, 3).reshape(K * B, 1, n, n)          # slice k, batch b
    for sgn in (1.0, -1.0):
        x1 = a[None] + sgn * grid.h * torch.cos(th)[:, None, None]
        x2 = b[None] + sgn * grid.h * torch.sin(th)[:, None, None]
        g = torch.stack([x2, x1], dim=-1).repeat_interleave(B, dim=0)   # (K B, n, n, 2)
        nb = F.grid_sample(slices, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        keep &= (x >= nb.reshape(K, B, n, n).permute(1, 0, 2, 3))
    keep &= (x >= torch.roll(x, 1, dims=1)) & (x >= torch.roll(x, -1, dims=1))
    return keep.reshape(*lead, K, n, n)


def thin_estimate(u: torch.Tensor, grid: LiftedGrid, threshold: float) -> torch.Tensor:
    """u on its ridges above ``threshold`` (relative to max u), zero elsewhere."""
    r = ridges(u, grid) & (u >= threshold * u.amax(dim=(-3, -2, -1), keepdim=True))
    return u * r


def rasterize(grid: LiftedGrid, cs) -> np.ndarray:
    """Thin analytic wavefront set: the voxels of the curve samples
    (``data.phantoms.CurveSamples``, sampled at <= h / 2), boolean (K, n, n)."""
    out = np.zeros(grid.shape, dtype=bool)
    i = np.floor((cs.x + 1.0) / grid.h).astype(int)
    k = np.mod(np.round(np.mod(cs.theta, math.pi) / grid.dtheta).astype(int), grid.n_theta)
    ok = (i >= 0).all(1) & (i < grid.n).all(1)
    out[k[ok], i[ok, 0], i[ok, 1]] = True
    return out


def _dilate(mask: np.ndarray, tol_x: float, tol_theta: int) -> np.ndarray:
    """Dilate a (K, n, n) mask by a disk of radius tol_x cells and +-tol_theta
    orientation cells (periodic in theta)."""
    r = int(math.floor(tol_x))
    o = np.arange(-r, r + 1)
    disk = (o[:, None]**2 + o[None, :]**2) <= tol_x**2
    m = ndimage.binary_dilation(mask, structure=disk[None]) if r > 0 else mask.copy()
    out = m.copy()
    for s in range(1, tol_theta + 1):
        out |= np.roll(m, s, axis=0) | np.roll(m, -s, axis=0)
    return out


def precision_recall(pred: np.ndarray, true: np.ndarray, tol_x: float = 1.5, tol_theta: int = 1):
    """Counts (true positives of pred, |pred|, recalled true points, |true|) for
    boolean (K, n, n) masks."""
    tp = int((pred & _dilate(true, tol_x, tol_theta)).sum())
    rec = int((true & _dilate(pred, tol_x, tol_theta)).sum())
    return tp, int(pred.sum()), rec, int(true.sum())


def f_score(counts) -> dict:
    """Precision, recall and F from summed counts."""
    tp, n_pred, rec, n_true = (sum(c[i] for c in counts) for i in range(4))
    p, r = tp / max(n_pred, 1), rec / max(n_true, 1)
    return {"P": p, "R": r, "F": 2 * p * r / max(p + r, 1e-12)}


def best_f(fields, trues, grid: LiftedGrid, thresholds=np.linspace(0.05, 0.9, 18), **tol):
    """Dataset-optimal threshold (ODS): F of the thin estimates of ``fields``
    (list of (K, n, n) tensors) against ``trues`` (list of boolean masks), in the
    lifted space and for the spatial projection.  Returns a dict with F, P, R,
    the threshold, and F_x (spatial)."""
    rid = [ridges(u, grid).cpu().numpy() for u in fields]
    us = [u.cpu().numpy() for u in fields]
    best = None
    for t in thresholds:
        preds = [r & (u >= t * u.max()) for r, u in zip(rid, us)]
        lifted = f_score([precision_recall(p, g, **tol) for p, g in zip(preds, trues)])
        spatial = f_score([precision_recall(p.any(0)[None], g.any(0)[None], tol.get("tol_x", 1.5), 0)
                           for p, g in zip(preds, trues)])
        rec = {**lifted, "threshold": float(t), "F_x": spatial["F"]}
        if best is None or rec["F"] > best["F"]:
            best = rec
    return best
