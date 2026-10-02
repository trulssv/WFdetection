"""Plotting helpers for lifted fields."""

import matplotlib
import numpy as np


def orientation_rgb(field: np.ndarray, vmax: float = None) -> np.ndarray:
    """Lifted field (K, n, n) -> RGB image (n, n, 3) for ``imshow(rgb.transpose(1, 0, 2),
    origin='lower')``: brightness = max over theta, hue = argmax theta (mod pi)."""
    K = field.shape[0]
    mx = field.max(axis=0)
    vmax = vmax if vmax is not None else (mx.max() + 1e-12)
    hsv = np.stack([field.argmax(axis=0) / K, np.ones_like(mx), np.clip(mx / vmax, 0, 1)], -1)
    return matplotlib.colors.hsv_to_rgb(hsv)
