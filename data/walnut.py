"""FIPS 2D walnut data (Hämäläinen et al. 2015, arXiv:1502.04064).

Download (Zenodo record 1254206) into ``data/raw/walnut/``:
FullSizeSinograms.mat, GroundTruthReconstruction.mat and, for the geometry
check in ``tests/test_walnut.py``, Data82.mat.

Geometry (documentation, Fig. 2): flat-detector fan beam, focus-to-object
110 mm, focus-to-detector 300 mm, 2296 detector pixels of 0.05 mm
(W = 114.8 mm), full 2 pi scans with 120 (3 degree) or 1200 (0.3 degree)
projections.  Lengths are normalized by the half-width L of the field of view
used by the dataset's reconstructions (82 pixels of 0.513 mm), so the object
lives in Omega = [-1, 1]^2.

Calibration (against the system matrix A of Data82.mat, correlation 0.9999,
see ``tests/test_walnut.py``): with ODL's counter-clockwise source angle
beta_k = 2 pi k / n_proj (k = acquisition index), the detector order is
unchanged, and an ODL image array f[i1, i2] equals the transpose of the
dataset's MATLAB image X (i.e. x1 runs along MATLAB columns and x2 along
MATLAB rows, downwards).
"""

import math
from pathlib import Path

import numpy as np
import scipy.io as sio

from geometry.tomo import fan_geometry

ROOT = Path(__file__).resolve().parent / "raw" / "walnut"

SRC_TO_ORIGIN_MM = 110.0
SRC_TO_DET_MM = 300.0
DET_PIXEL_MM = 0.05
N_DET = 2296
HALF_FOV_MM = 82 * 0.513 / 2


def walnut_geometry(n_proj: int, n_det: int):
    """Normalized ODL fan geometry for ``n_proj`` projections and ``n_det``
    (binned) detector pixels."""
    L = HALF_FOV_MM
    half_step = math.pi / n_proj      # cell centres at beta_k = 2 pi k / n_proj
    return fan_geometry(n_proj, n_det, SRC_TO_ORIGIN_MM / L, (SRC_TO_DET_MM - SRC_TO_ORIGIN_MM) / L,
                        N_DET * DET_PIXEL_MM / 2 / L,
                        angle_range=(-half_step, 2 * math.pi - half_step))


def _flat_field(intensity: np.ndarray, edge: int = 40) -> float:
    """Unattenuated intensity I0 from the outermost detector pixels (air)."""
    air = np.concatenate([intensity[:, :edge], intensity[:, -edge:]], axis=1)
    return float(np.percentile(air, 90))


def load_walnut(n_proj: int = 1200, det_binning: int = 4, root=ROOT):
    """Log-attenuation fan sinogram of the walnut.

    Returns
    -------
    dict with
        sinogram : float32 array (n_proj, n_det // det_binning), indexed
                   [projection, detector]; equals R f for the image
                   f = L * mu (mu the attenuation in 1/mm) in normalized units
        geometry : matching ODL fan geometry (normalized units)
        I0       : estimated flat field
    """
    if n_proj not in (120, 1200):
        raise ValueError("n_proj must be 120 or 1200")
    path = Path(root) / "FullSizeSinograms.mat"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; see data/README.md for the download")
    I = sio.loadmat(path)[f"sinogram{n_proj}"].astype(np.float64).T     # (n_proj, 2296)
    if N_DET % det_binning:
        raise ValueError(f"det_binning must divide {N_DET}")
    I = I.reshape(n_proj, N_DET // det_binning, det_binning).sum(-1)
    I0 = _flat_field(I)
    y = -np.log(np.clip(I / I0, 1e-6, None))
    return {"sinogram": y.astype(np.float32),
            "geometry": walnut_geometry(n_proj, N_DET // det_binning),
            "I0": I0}


def load_reference(root=ROOT, n: int = None):
    """High-resolution FBP of 1200 projections (2296 x 2296), as an ODL-oriented
    array (transposed), optionally block-averaged down to n x n.

    Its pixel size (0.05 mm / magnification) and field of view differ from Omega;
    it is meant for visual comparison only.
    """
    X = sio.loadmat(Path(root) / "GroundTruthReconstruction.mat")["FBP1200"].astype(np.float32)
    f = X.T
    if n is not None:
        k = f.shape[0] // n
        f = f[: n * k, : n * k].reshape(n, k, n, k).mean(axis=(1, 3))
    return f
