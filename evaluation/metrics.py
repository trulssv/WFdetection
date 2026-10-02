"""Metrics comparing lifted fields with analytic wavefront sets."""

import numpy as np

from geometry.lifted import LiftedGrid, image_to_sinogram_coords


def roc_auc(score: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for the two boolean masks.

    Equals P(score at a random positive > score at a random negative), computed
    from ranks (Mann-Whitney U).
    """
    pos, neg = np.asarray(score)[positive], np.asarray(score)[negative]
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def label_masks(membership: np.ndarray, pos_thresh=0.5, neg_thresh=0.01):
    """Positive / negative voxels of a soft label, with a gap between them."""
    return membership >= pos_thresh, membership < neg_thresh


def t_localization(c: np.ndarray, grid: LiftedGrid, x: np.ndarray, theta: np.ndarray,
                   window: int = 8):
    """Where along t do the coefficients peak, at the predicted sinogram points?

    For every wavefront sample (x, theta) the canonical relation predicts the
    lifted sinogram point (s, phi = theta mod pi, t).  On the coefficient line
    c[k, i_s, :] at the nearest (phi_k, s_i) we locate the maximum

    * within +-``window`` cells of the predicted t ("local"), and
    * over the whole line ("global"; may pick another edge on the same ray).

    Returns a dict with the signed local errors (in cells) and the fraction of
    samples whose global maximum is within 2 cells of the prediction.
    """
    th = np.mod(theta, np.pi)
    s, _, t = image_to_sinogram_coords(x, th)
    k = np.mod(np.round(th / grid.dtheta).astype(int), grid.n_theta)
    fs = (s + 1) / grid.h - 0.5
    ft = (t + 1) / grid.h - 0.5
    i = np.round(fs).astype(int)
    ok = (i >= 0) & (i < grid.n) & (ft >= 0) & (ft <= grid.n - 1)
    k, i, ft = k[ok], i[ok], ft[ok]
    lines = c[k, i, :]                                   # (m, n_t)
    j = np.arange(grid.n)[None, :]
    near = np.abs(j - ft[:, None]) <= window
    local = np.argmax(np.where(near, lines, -np.inf), axis=1)
    glob = np.argmax(lines, axis=1)
    return {
        "local_error": local - ft,
        "global_hit_rate": float(np.mean(np.abs(glob - ft) <= 2)),
        "n_samples": int(ok.sum()),
    }
