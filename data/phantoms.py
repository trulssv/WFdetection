"""Piecewise-constant phantoms with analytically known wavefront sets.

A phantom is a sum of shapes (ellipses, polygons) with constant values.  Its
wavefront set (away from accidental cancellations) is the set of boundary
points with their normals, plus the full fibre {p} x RP^1 at polygon corners.

Lifted representations on a :class:`geometry.LiftedGrid` (layout
(n_theta, n, n) on (theta, x1, x2)):

* ``membership`` - soft indicator in [0, 1] of the wavefront set, the target
  for a classifier.
* ``flux`` - the components (a, b, c) of the blurred 1-current
  tau = a X1 + c X2 + b X3 of the lifted boundary curves (see
  ``loss/regularization.py``): a is the line density (its integral is the
  curve length), b = a * kappa the rotation rate, and c the small normal
  component that blurring a horizontal curve necessarily creates.  Normals
  are oriented outward and curves traversed with tangent J n (normal rotated
  by +90 degrees); for theta in [pi, 2pi) the sample is stored at theta - pi
  with b -> -b, which is how the orientation-reversed antipodal copy appears
  on RP^1.

Curve samples are (x, theta, ds, dtheta): position, outward normal angle in
[0, 2 pi), and the arc-length and normal-angle increments of the sample.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from geometry.lifted import LiftedGrid


@dataclass
class CurveSamples:
    """Discretized lifted curve: arrays of equal length."""

    x: np.ndarray        # (m, 2) positions
    theta: np.ndarray    # (m,) outward normal angle in [0, 2 pi)
    ds: np.ndarray       # (m,) arc-length element (>= 0)
    dtheta: np.ndarray   # (m,) signed normal-angle element

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if len(p.theta)]
        if not parts:
            return CurveSamples(np.zeros((0, 2)), *(np.zeros(0) for _ in range(3)))
        return CurveSamples(*(np.concatenate([getattr(p, f) for p in parts])
                              for f in ("x", "theta", "ds", "dtheta")))

    def subriemannian_length(self, xi: float) -> float:
        """sum sqrt(ds^2 + xi^2 dtheta^2): the curve's cost under the length term
        of :class:`loss.regularization.CurvePrior`."""
        return float(np.sum(np.hypot(self.ds, xi * self.dtheta)))


@dataclass
class Ellipse:
    center: tuple
    axes: tuple          # semi-axes (r1, r2)
    angle: float = 0.0   # rotation of the first axis, radians
    value: float = 1.0

    def _rot(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    def indicator(self, x1, x2):
        q = np.stack([x1 - self.center[0], x2 - self.center[1]], -1) @ self._rot()
        return ((q[..., 0] / self.axes[0])**2 + (q[..., 1] / self.axes[1])**2 <= 1).astype(float)

    def samples(self, step: float, p_range=(0.0, 2 * math.pi)) -> CurveSamples:
        """Boundary samples, counter-clockwise; ``p_range`` selects an arc."""
        r1, r2 = self.axes
        perim = math.pi * (3 * (r1 + r2) - math.sqrt((3 * r1 + r2) * (r1 + 3 * r2)))
        span = (p_range[1] - p_range[0]) / (2 * math.pi)
        m = max(int(math.ceil(span * perim / step)), 8)
        dp = (p_range[1] - p_range[0]) / m
        p = p_range[0] + (np.arange(m) + 0.5) * dp
        R = self._rot()
        x = np.asarray(self.center) + np.stack([r1 * np.cos(p), r2 * np.sin(p)], -1) @ R.T
        normal = np.stack([np.cos(p) / r1, np.sin(p) / r2], -1) @ R.T
        speed = np.hypot(r1 * np.sin(p), r2 * np.cos(p))
        # d(normal angle)/dp for the ellipse, positive (convex, counter-clockwise)
        dth_dp = r1 * r2 / (r1**2 * np.sin(p)**2 + r2**2 * np.cos(p)**2)
        theta = np.mod(np.arctan2(normal[:, 1], normal[:, 0]), 2 * math.pi)
        return CurveSamples(x, theta, speed * dp, dth_dp * dp)


@dataclass
class Polygon:
    vertices: np.ndarray   # (m, 2), counter-clockwise
    value: float = 1.0

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, float)
        v = self.vertices
        area2 = np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
        if area2 < 0:
            self.vertices = v[::-1].copy()

    def indicator(self, x1, x2):
        # even-odd rule
        inside = np.zeros(np.broadcast(x1, x2).shape, bool)
        v = self.vertices
        for (ax, ay), (bx, by) in zip(v, np.roll(v, -1, axis=0)):
            cond = (ay > x2) != (by > x2)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = ax + (x2 - ay) * (bx - ax) / (by - ay)
            inside ^= cond & (x1 < xc)
        return inside.astype(float)

    def samples(self, step: float, corners: str = "arc") -> CurveSamples:
        """Edge samples plus corner samples.

        corners = 'arc'  : the vertical arc of the lifted curve turning by the
                           exterior angle (this closes the lifted curve);
                  'full' : the full fibre {p} x [0, 2 pi) (the actual WF set, with
                           ds = dtheta = 0 outside the arc, i.e. no flux);
                  'none' : edges only.
        """
        v = self.vertices
        parts, normals = [], []
        for a, b in zip(v, np.roll(v, -1, axis=0)):
            d = b - a
            L = float(np.hypot(*d))
            n_out = np.array([d[1], -d[0]]) / L
            normals.append(math.atan2(n_out[1], n_out[0]))
            m = max(int(math.ceil(L / step)), 2)
            u = (np.arange(m) + 0.5) / m
            th = np.full(m, math.atan2(n_out[1], n_out[0]) % (2 * math.pi))
            parts.append(CurveSamples(a + u[:, None] * d, th, np.full(m, L / m), np.zeros(m)))
        if corners != "none":
            for i in range(len(v)):
                th0, th1 = normals[i - 1], normals[i]          # edges into / out of vertex i
                ext = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi   # signed exterior angle
                m = max(int(math.ceil(abs(ext) / step)), 2)
                th = th0 + (np.arange(m) + 0.5) / m * ext
                parts.append(CurveSamples(np.repeat(v[i][None], m, 0), th % (2 * math.pi),
                                          np.zeros(m), np.full(m, ext / m)))
                if corners == "full":
                    mf = max(int(math.ceil(2 * math.pi / step)), 8)
                    thf = (np.arange(mf) + 0.5) / mf * 2 * math.pi
                    parts.append(CurveSamples(np.repeat(v[i][None], mf, 0), thf,
                                              np.zeros(mf), np.zeros(mf)))
        return CurveSamples.concat(parts)


@dataclass
class Phantom:
    shapes: list = field(default_factory=list)

    def render(self, n: int, supersample: int = 4) -> np.ndarray:
        """Image on the n x n cell grid of [-1, 1]^2, indexed [i1, i2], with
        box-filter anti-aliasing."""
        m = n * supersample
        c = -1.0 + (np.arange(m) + 0.5) * (2.0 / m)
        x1, x2 = np.meshgrid(c, c, indexing="ij")
        img = sum(sh.value * sh.indicator(x1, x2) for sh in self.shapes)
        img = np.zeros((m, m)) + img
        return img.reshape(n, supersample, n, supersample).mean(axis=(1, 3))

    def samples(self, step: float, corners: str = "arc") -> CurveSamples:
        parts = []
        for sh in self.shapes:
            parts.append(sh.samples(step, corners) if isinstance(sh, Polygon) else sh.samples(step))
        return CurveSamples.concat(parts)


# ---------------------------------------------------------------------------
# Rasterization onto the lifted grid
# ---------------------------------------------------------------------------

def _splat(grid: LiftedGrid, cs: CurveSamples, sigma_x, sigma_th, mode):
    """Gaussian splatting of samples onto the (theta, x1, x2) grid.

    mode = 'flux': each sample is the vector ds X1(theta_m) + dtheta X3, blurred
    with a normalized Gaussian in *fixed Cartesian components* (so that blurring
    commutes with the divergence) and expressed in the frame (X1, X2, X3) of
    the voxel's theta.  Offset Delta = theta - theta_m gives
        a = ds cos(Delta),  c = ds sin(Delta),  b = dtheta.
    Wrapping across theta = pi maps (a, c, b) -> (a, c, -b).
    Returns (a, b, c).

    mode = 'max': peak-one Gaussians, maximum over samples (soft indicator).
    """
    K, n, h, dth = grid.n_theta, grid.n, grid.h, grid.dtheta
    rx, rt = int(math.ceil(3 * sigma_x / h)), int(math.ceil(3 * sigma_th / dth))
    ox = np.arange(-rx, rx + 1)
    ot = np.arange(-rt, rt + 1)
    # continuous indices; theta in [0, 2 pi) is indexed on the RP^1 grid with wraps
    fi = (cs.x + 1.0) / h - 0.5
    ft = cs.theta / dth
    i0 = np.round(fi).astype(int)
    k0 = np.round(ft).astype(int)
    I1 = i0[:, 0, None, None, None] + ox[None, None, :, None]
    I2 = i0[:, 1, None, None, None] + ox[None, None, None, :]
    KK = k0[:, None, None, None] + ot[None, :, None, None]
    delta = (KK - ft[:, None, None, None]) * dth
    g = np.exp(-((I1 - fi[:, 0, None, None, None])**2 + (I2 - fi[:, 1, None, None, None])**2)
               * h**2 / (2 * sigma_x**2) - delta**2 / (2 * sigma_th**2))
    I1, I2, KK = np.broadcast_arrays(I1, I2, KK)
    delta = np.broadcast_to(delta, g.shape)
    valid = (I1 >= 0) & (I1 < n) & (I2 >= 0) & (I2 < n)
    idx = (np.mod(KK, K)[valid], I1[valid], I2[valid])

    if mode == "max":
        out = np.zeros(grid.shape)
        np.maximum.at(out, idx, g[valid])
        return out

    g = g / (g.sum(axis=(1, 2, 3), keepdims=True) * grid.cell_volume)
    flip = np.where(np.floor_divide(KK, K) % 2 == 1, -1.0, 1.0)
    ds, dt = cs.ds[:, None, None, None], cs.dtheta[:, None, None, None]
    outs = []
    for vals in (ds * np.cos(delta) * g, dt * flip * g, ds * np.sin(delta) * g):
        out = np.zeros(grid.shape)
        np.add.at(out, idx, vals[valid])
        outs.append(out)
    return tuple(outs)


def lifted_flux(grid: LiftedGrid, cs: CurveSamples, sigma_x=None, sigma_th=None):
    """Return (a, b, c), each of shape grid.shape: the blurred current
    tau = a X1 + c X2 + b X3 of the curve samples (see ``loss/regularization.py``).

    The default blur is 2 cells, the resolution the prior's finite differences
    need for an accurate divergence."""
    sigma_x = sigma_x or 2 * grid.h
    sigma_th = sigma_th or 2 * grid.dtheta
    return _splat(grid, cs, sigma_x, sigma_th, "flux")


def lifted_membership(grid: LiftedGrid, cs: CurveSamples, sigma_x=None, sigma_th=None):
    """Soft indicator in [0, 1] of the lifted samples, shape grid.shape."""
    sigma_x = sigma_x or grid.h
    sigma_th = sigma_th or grid.dtheta
    return _splat(grid, cs, sigma_x, sigma_th, "max")


# ---------------------------------------------------------------------------
# Random phantoms
# ---------------------------------------------------------------------------

def random_phantom(rng: np.random.Generator, n_ellipses=(2, 5), n_polygons=(0, 2),
                   radius: float = 0.85) -> Phantom:
    """Random ellipses and convex polygons inside the disk of the given radius.

    Values are bounded away from 0 so that every boundary is a visible edge
    (overlaps can still cancel edges by accident; this is ignored).
    """
    def value():
        return rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 1.0)

    shapes = []
    for _ in range(rng.integers(n_ellipses[0], n_ellipses[1] + 1)):
        r = rng.uniform(0.08, 0.4, size=2)
        rho = rng.uniform(0, max(radius - r.max(), 0.0))
        phi = rng.uniform(0, 2 * math.pi)
        shapes.append(Ellipse((rho * math.cos(phi), rho * math.sin(phi)), tuple(r),
                              rng.uniform(0, math.pi), value()))
    for _ in range(rng.integers(n_polygons[0], n_polygons[1] + 1)):
        m = rng.integers(3, 7)
        size = rng.uniform(0.1, 0.35)
        rho = rng.uniform(0, radius - size)
        phi = rng.uniform(0, 2 * math.pi)
        ang = np.sort(rng.uniform(0, 2 * math.pi, m))
        verts = np.stack([np.cos(ang), np.sin(ang)], -1) * size
        shapes.append(Polygon(verts + [rho * math.cos(phi), rho * math.sin(phi)],
                              value()))
    return Phantom(shapes)
