"""ODL tomography geometries used in the project.

Conventions (checked against ODL 1.0, see ``tests/test_radon.py``): for the
parallel-beam geometry at angle phi the detector axis is
omega(phi) = (cos phi, sin phi) and rays run along
omega_perp(phi) = (-sin phi, cos phi), so

    R f(s, phi) = int f(s omega(phi) + t omega_perp(phi)) dt.

Sinograms are indexed ``y[angle, detector]``.  The image space is always
[-1, 1]^2 with ``f[i1, i2]`` the value at (x1_i1, x2_i2) (ODL's 'ij' order).
"""

import math

import odl
from odl.applications import tomo


def image_space(n: int, dtype: str = "float32"):
    """ODL space of n x n pixels on [-1, 1]^2."""
    return odl.uniform_discr([-1.0, -1.0], [1.0, 1.0], (n, n), dtype=dtype)


def parallel_geometry(n_angles: int, n_det: int, det_radius: float = math.sqrt(2.0),
                      angle_range=(0.0, math.pi)):
    """Parallel-beam geometry with uniform angles and a centred flat detector.

    The detector must be symmetric about 0 so that the identity
    R f(s, phi + pi) = R f(-s, phi) can be used to extend sinograms
    periodically in phi (see :mod:`operators.microlocal`).
    """
    angles = odl.uniform_partition(angle_range[0], angle_range[1], n_angles)
    det = odl.uniform_partition(-det_radius, det_radius, n_det)
    return tomo.Parallel2dGeometry(angles, det)


def fan_geometry(n_angles: int, n_det: int, src_radius: float, det_radius: float,
                 det_half_width: float, angle_range=(0.0, 2 * math.pi)):
    """Flat-detector fan-beam geometry (ODL convention).

    At source angle beta the source is at R_beta (0, -src_radius), the detector
    centre at R_beta (0, det_radius) and the detector axis is R_beta (1, 0);
    beta increases counter-clockwise.  ``det_half_width`` is measured on the
    detector (not magnified).  See :mod:`operators.fan_beam` for the map to
    parallel-beam line coordinates.
    """
    angles = odl.uniform_partition(angle_range[0], angle_range[1], n_angles)
    det = odl.uniform_partition(-det_half_width, det_half_width, n_det)
    return tomo.FanBeamGeometry(angles, det, src_radius=src_radius, det_radius=det_radius)


def sinogram_axes(geometry):
    """Angle and detector cell centres of a 2D geometry as numpy arrays."""
    phis = geometry.motion_partition.meshgrid[0].ravel()
    s = geometry.det_partition.meshgrid[0].ravel()
    return phis, s
