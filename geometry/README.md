# geometry/

Discrete grids and coordinate conventions that every other module relies on.

## `lifted.py`

`LiftedGrid(n, n_theta)` is a uniform grid on $[-1,1]^2\times\mathbb{RP}^1$:

* spatial cell centres $-1 + (i+\tfrac12)h$ with $h = 2/n$, from `coords()`;
* orientations $\theta_k = \pi k / n_\theta$, from `thetas()`;
* the voxel volume $h^2\,\Delta\theta$ (`cell_volume`), used for all integrals.

The **same** grid describes both lifted spaces:

| side | tensor index `[..., k, i, j]` | coordinates |
|---|---|---|
| image | `u[..., k, i1, i2]` | $(\theta_k, x_{1,i_1}, x_{2,i_2})$ |
| sinogram | `v[..., k, is, it]` | $(\varphi_k=\theta_k, s_{i_s}, t_{i_t})$ |

Tensor layout is `(B, C, n_theta, n, n)`, so orientation is the depth axis of
`Conv3d`. Periodicity in $\theta$ has to be handled explicitly; see
`loss/regularization.py` for the antiperiodic rule that applies to the flux
component $b$.

Pointwise maps `image_to_sinogram_coords` / `sinogram_to_image_coords`
implement the canonical relation

$$(x,\theta)\ \mapsto\ (s,\varphi,t)=(x\cdot\omega(\theta),\ \theta,\ x\cdot\omega^\perp(\theta)),$$

where $t$ parametrizes the sinogram covector direction $(1,-t)$ in $(s,\varphi)$
coordinates. Geometrically, $t$ is the position along the ray at which the ray
touches the edge.

## `tomo.py`

Thin helpers around ODL 1.0:

* `image_space(n)` gives the $n\times n$ ODL space on $[-1,1]^2$;
* `parallel_geometry(n_angles, n_det, det_radius=√2)` gives uniform angles on
  $[0,\pi)$ and a centred detector (it must be symmetric, because the
  analyzer pads sinograms using $Rf(s,\varphi+\pi)=Rf(-s,\varphi)$);
* `sinogram_axes(geometry)` returns the angle and detector cell centres.

ODL's parallel-beam convention at angle $\varphi$ is detector axis
$\omega(\varphi)$ and ray direction $\omega^\perp(\varphi)$. Sinograms are
indexed `[angle, detector]`.
