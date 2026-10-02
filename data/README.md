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

## `synthetic.py`: training data

`SyntheticTomography(grid, noise, background)` generates batches on the fly:
a random phantom plus an optional smooth background (which has no wavefront
set and checks that smooth structure is ignored; it is tapered to zero with
a $C^1$ smoothstep, since a hard cut-off would be an unlabelled edge), the ODL parallel-beam
projection (3n angles, 2n detector cells), relative Gaussian noise, and
$c=My$ normalized per sample by its 99.9% quantile
(`normalize_coefficients`). Training uses only `c`; `labels=True`
additionally returns the lifted membership for validation.

## `walnut.py`: FIPS walnut

FIPS 2D walnut (Hämäläinen et al., arXiv:1502.04064), Zenodo record 1254206.
Download into `data/raw/walnut/` (not committed):

```bash
mkdir -p data/raw/walnut && cd data/raw/walnut
for f in FullSizeSinograms.mat GroundTruthReconstruction.mat Data82.mat Documentation_v1.pdf MD5SUMS; do
  curl -sL -o $f "https://zenodo.org/api/records/1254206/files/$f/content"; done
md5sum -c MD5SUMS 2>/dev/null | grep -v FAILED   # (Data164/328 are not needed)
```

* `load_walnut(n_proj=1200|120, det_binning)`: log-attenuation fan sinogram
  `[projection, detector]`, with $I_0$ estimated from the air columns, plus
  the matching ODL fan geometry in normalized units. Lengths are divided by
  the half-width $L=21.03$ mm of the dataset's field of view, so the
  sinogram equals $Rf$ for $f=L\mu$.
* `walnut_geometry(n_proj, n_det)`: source–origin 110 mm, source–detector
  300 mm, 2296 pixels of 0.05 mm, full $2\pi$ scan with angles
  $\beta_k=2\pi k/n_{\text{proj}}$.
* `load_reference(n)`: the dataset's FBP of 1200 projections, block-averaged,
  in ODL orientation.

**Calibration.** The rotation direction, start angle, detector order and
image orientation were determined by matching the ODL projection of a test
image with the dataset's own system matrix $A$ (Data82.mat). The best match
has correlation 0.9999 and scale 21.07 mm/unit ≈ $L$
(`tests/test_walnut.py`). Result: ODL's counter-clockwise $\beta$ from 0,
detector not flipped, and ODL array = transpose of the MATLAB image. Show
walnut images with `imshow(f.T, origin="upper")` to match the dataset's
figures.
