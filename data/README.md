# data/

## `phantoms.py`: phantoms with analytic wavefront sets

Piecewise-constant phantoms (`Ellipse`, `Polygon`, combined in `Phantom`)
whose WF set is known exactly: boundary points with their normals, plus the
full fibre at polygon corners (ignoring accidental cancellations where
overlapping shapes have equal and opposite values). These are the ground
truth for every check of the unsupervised method.

* `Phantom.render(n)` gives the image on the $n\times n$ grid of $[-1,1]^2$,
  indexed `[i1, i2]`, with supersampled anti-aliasing.
* `Phantom.samples(step, corners)` returns `CurveSamples(x, theta, ds, dtheta)`:
  positions, outward normal angle in $[0,2\pi)$, and arc-length / normal-angle
  increments. `corners='arc' | 'full' | 'none'` selects the corner treatment
  (the exterior-angle arc that closes the lifted curve / the full fibre, which
  is the actual WF set / no corners).
* `lifted_membership(grid, samples)` gives a soft indicator in $[0,1]$ on the
  lifted grid (peak-one Gaussians, max over samples). This is the
  classification target.
* `lifted_flux(grid, samples)` gives the blurred current $(a,b,c)$ used by the
  curve prior. The blur is a normalized Gaussian, 2 cells by default (the
  resolution the prior needs), applied in fixed Cartesian components so that it
  commutes with the divergence. Hence the normal component
  $c=a\sin(\theta-\theta_m)$, and $\int|(a,c)|\,dX$ equals the curve
  length. Samples with $\theta\in[\pi,2\pi)$ are stored at $\theta-\pi$
  with $b\mapsto-b$.
* `random_phantom(rng)` draws random ellipses and convex polygons inside the
  disk of radius 0.85, with $|$value$|\ge 0.2$.

Shapes must stay inside the unit disk, where the canonical relation operator
is unitary.

## Walnut data (planned)

FIPS 2D walnut, fan-beam:
<https://fips.fi/open-datasets/x-ray-tomographic-datasets/tomographic-x-ray-data-of-a-walnut/>.
The loader will go in `walnut.py` once the fan-beam relation is implemented
(milestone 4). Raw downloads belong in `data/raw/`, which should not be
committed.
