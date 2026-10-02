import math

import numpy as np
import pytest
import torch

from data.phantoms import Ellipse, Polygon, lifted_flux
from geometry.lifted import LiftedGrid
from loss.regularization import CurvePrior

GRID = LiftedGrid(n=96, n_theta=48)
XI = 0.1


def prior_terms(cs, **kw):
    a, b, c = (torch.as_tensor(f, dtype=torch.float64) for f in lifted_flux(GRID, cs))
    R = CurvePrior(GRID, xi=XI, nu=1.0, **kw).double()
    return float(R.mass(a, b, c)), float(R.boundary_mass(a, b, c))


ELLIPSE = Ellipse((0.1, -0.05), (0.5, 0.3), 0.4)


def test_flux_integrals_match_curve():
    cs = ELLIPSE.samples(GRID.h / 4)
    a, b, c = lifted_flux(GRID, cs)
    # a and c are the components of the blurred tangent, |(a, c)| integrates to length
    assert np.sum(np.hypot(a, c)) * GRID.cell_volume == pytest.approx(cs.ds.sum(), rel=0.02)
    # outward normal turns by 2 pi around a convex curve
    assert np.sum(np.abs(b)) * GRID.cell_volume == pytest.approx(2 * math.pi, rel=0.05)


def test_length_term_matches_analytic():
    cs = ELLIPSE.samples(GRID.h / 4)
    # zeta = 1: |(a, c)| is the blurred tangent, so the length is exact up to blur
    length, _ = prior_terms(cs, zeta=1.0)
    assert length == pytest.approx(cs.subriemannian_length(XI), rel=0.03)
    # zeta > 1 penalizes the normal component created by blurring: small bias
    length2, _ = prior_terms(cs, zeta=2.0)
    assert length < length2 < 1.06 * length


def test_endpoints_closed_vs_open():
    closed = ELLIPSE.samples(GRID.h / 4)
    half = ELLIPSE.samples(GRID.h / 4, p_range=(0.3, 0.3 + math.pi))
    two = ELLIPSE.samples(GRID.h / 4, p_range=(0.3, 0.3 + math.pi)), \
        ELLIPSE.samples(GRID.h / 4, p_range=(0.3 + math.pi + 0.6, 0.3 + 2 * math.pi - 0.3))
    from data.phantoms import CurveSamples
    _, e_closed = prior_terms(closed)
    _, e_half = prior_terms(half)
    _, e_two = prior_terms(CurveSamples.concat(list(two)))
    assert e_closed < 0.4
    assert e_half == pytest.approx(2.0, rel=0.2)
    assert e_two == pytest.approx(4.0, rel=0.2)


def test_normal_component_is_needed():
    """Dropping c (forcing a horizontal blurred field) leaves a spurious endpoint
    cost along the whole curve."""
    a, b, c = (torch.as_tensor(f) for f in lifted_flux(GRID, ELLIPSE.samples(GRID.h / 4)))
    R = CurvePrior(GRID).double()
    assert float(R.boundary_mass(a, b)) > 5 * float(R.boundary_mass(a, b, c))


def test_polygon_corner_arcs_close_the_curve():
    square = Polygon([[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]])
    _, e_arc = prior_terms(square.samples(GRID.h / 4, corners="arc"))
    _, e_none = prior_terms(square.samples(GRID.h / 4, corners="none"))
    assert e_arc < 0.5
    assert e_none == pytest.approx(8.0, rel=0.2)   # 4 open edges, 2 endpoints each


def test_elastica_and_gradients():
    cs = ELLIPSE.samples(GRID.h / 4)
    a, b, c = (torch.as_tensor(f, dtype=torch.float32) for f in lifted_flux(GRID, cs))
    a.requires_grad_(True)
    b.requires_grad_(True)
    R = CurvePrior(GRID, length="elastica", alpha=1.0, beta=0.01, eps=1e-3)
    val = R(a[None], b[None], c[None])
    val.sum().backward()
    assert torch.isfinite(val).all() and torch.isfinite(a.grad).all()
