# operators/

Operators between the image, the sinogram, and their lifted wavefront-set
representations.

## `canonical_relation.py`: $C$

**Derivation.** Write $Rf(s,\varphi)=\int f(s\omega+t\omega^\perp)\,dt$.
Suppose an edge passes through $x$ with normal $\omega(\theta)$. It produces a
singularity only in the ray tangent to it, i.e. $\varphi=\theta$ and
$s=x\cdot\omega(\theta)$. Near that ray the singular set of $Rf$ is the curve
$s=h(\varphi)$, where $h$ is the local support function of the edge. Its slope
is $h'(\varphi)=x\cdot\omega^\perp(\varphi)=:t$, which is the position along the
ray where the ray touches the edge. The conormal of the curve, and hence the
sinogram covector, is therefore $\propto(1,-t)$. Using $t$ as the fibre
coordinate on the sinogram side gives

$$(x,\theta)\mapsto(s,\varphi,t)=(x\cdot\omega(\theta),\theta,x\cdot\omega^\perp(\theta)),$$

which, for each fixed $\theta$, is a **rotation** of $x$ by $-\theta$. On lifted fields

$$(Cu)(s,\varphi,t)=u\big(s\,\omega(\varphi)+t\,\omega^\perp(\varphi),\varphi\big).$$

**Implementation.** One `grid_sample` per orientation slice (bilinear,
zero padding). `adjoint` is the exact discrete adjoint, computed with autograd.
`inverse` samples the inverse rotation. On fields supported in the unit disk,
$C$ is unitary up to interpolation error, so `adjoint` ≈ `inverse`. Since
$|t|\le 1$ inside the disk, the sinogram grid in $(s,\varphi,t)$ has the same
size as the image grid.

**Consequence.** With dense, full-angle data, $C$ is a bijection, so
detecting the WF set reduces to local sinogram analysis followed by
relabelling. The curve prior becomes essential only for sparse-view or
limited-angle data. There a WF point is observed only if its orientation
$\theta$ is a measured or visible angle, and $t$ (position along the ray) is
unobservable from a single view.

**Fan beam** (walnut data) is not implemented yet. Every fan-beam measurement
is a line, so the relation reduces to this one after composing with the
explicit fan-to-parallel reparametrization $(\beta,u)\mapsto(s,\varphi)$ of
line space and its cotangent lift.

## `microlocal.py`: $M$

`SinogramAnalyzer(grid, phis, s_det)` maps a sinogram `(B,1,n_phi,n_det)` to
lifted coefficients `(B,1,K,n,n)` on $(\varphi,s,t)$:

1. optional half-order ramp filter $|\sigma_s|^{1/2}$ along $s$. $R$ has order
   $-\tfrac12$, so image jumps become square-root singularities in the
   sinogram, and this filter turns them back into jump-type singularities;
2. periodic extension in $\varphi$ via $Rf(s,\varphi+\pi)=Rf(-s,\varphi)$;
3. a bank of anisotropic derivative-of-Gaussian kernels, where kernel $j$
   differentiates along the conormal $(1,-t_j)$ and smooths along the tangent
   $(t_j,1)$ (`sigma_n`, `aspect`), normalized to unit response to a unit step;
4. the magnitude, bilinearly resampled to $(\varphi_k=\theta_k, s_i)$.

`coefficient_noise_std(sigma)` estimates the per-filter coefficient std under
white sinogram noise by Monte Carlo. The data term uses it as its noise model.

**Resolution in $t$ is intrinsically poor.** The $t$-resolution is set by the
kernels' orientation selectivity, roughly $1/$`aspect` in slope units, i.e.
many cells. Longer kernels lose response because the singular curves
(sinusoids) are curved. In milestone 1 the estimate of $t$ is unbiased (mean
error 0.03 cells; median |error| 0.55 cells within a ±8-cell window), but at
fixed $(s,\varphi)$ the response is a long streak in $t$. Pulled back by
$C^{-1}$, every edge point spreads along its own tangent line. This is
inherent to local analysis: $t=dh/d\varphi$ is a derivative *along* the
singular curve, so resolving it needs non-local consistency, which is the
prior's job (see `loss/README.md`).

**Limitation:** the analyzer assumes dense angular sampling (several angles
across each kernel). Sparse-view data will need a separate analyzer that
detects jumps in $s$ within each projection and is uninformative in $t$.

## `radon.py`: $R$

`RayTransform(space, geometry)` wraps the ODL ray transform (ASTRA, GPU if
available) in a torch autograd function. Caveat: ODL's `op.adjoint` is the
adjoint for *weighted* inner products ($w$ = cell volume), whereas autograd
needs the matrix transpose $R^T=(w_X/w_Y)R^*$. The backward pass applies that
factor, and `RayTransform.adjoint` returns the (weighted) $L^2$ adjoint $R^*$.
It loops over batch elements through numpy, which is fine for data generation
and baselines. `fbp(y)` gives an FBP reconstruction for the
classical baseline.
