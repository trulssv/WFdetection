# loss/

Training objective $D(\mathcal R_tC(Jm);\,My)+\lambda E(a,b,c)$: an
unsupervised data term and a curve prior on the lifted wavefront set. Both
are negative log-densities, so minimizing their sum is MAP estimation, and
minimizing its expectation over data trains an amortized MAP solver
([research/background.md](../research/background.md), Prop. 7.2). The theory
behind everything below is in §3 (prior), §5 (data term) and §6
(discretization) of that document.

## `regularization.py`: curve prior `CurvePrior`

`CurvePrior(grid, xi, nu, zeta)` computes $E(T)=\mathbf M_g(T)+\nu\,\mathbf M(\partial T)$:
`mass(a, b, c)` is the length in the metric $g_{\xi,\zeta}$, and
`boundary_mass(a, b, c)` $=\int|\operatorname{div}\tau|$ counts endpoints.
`forward` returns $E$, the negative log-prior up to a constant.

The WF set is a 1-current $\tau=aX_1+cX_2+bX_3$ on $X$, with

* $X_1=(-\sin\theta,\cos\theta,0)$: the canonical tangent $Jn$ of an edge with
  normal angle $\theta$;
* $X_2=(\cos\theta,\sin\theta,0)$: the normal;
* $X_3=\partial_\theta$;
* $a\ge0$: line density; $b=a\kappa$: rotation rate; $c$: normal component.

For a sharp curve $\Gamma$, $\tau$ is its unit tangent times arc-length
measure, and $c=0$ (horizontality).

$$\mathcal R(a,b,c)=\int_X L(a,b,c)\,dX+\nu\int_X|X_1a+X_2c+\partial_\theta b|\,dX$$

with

* `'subriemannian'`: $L=\sqrt{a^2+\zeta^2c^2+\xi^2b^2}$. This is the Riemannian
  approximation of the sub-Riemannian metric (exact as $\zeta\to\infty$);
  corners cost $\xi\cdot$angle.
* `'elastica'`: $L=\alpha\sqrt{a^2+\zeta^2c^2}+\beta b^2/a$. Corners are
  forbidden.

$X_1,X_2,X_3$ are divergence-free, so the second integrand is
$|\operatorname{div}\tau|$: the boundary mass of the current, which counts
endpoints.

| desired property | how it is obtained |
|---|---|
| independence | local integrand ⇒ additive over separated components |
| smoothness | $L$ penalizes length and turning |
| stability | splitting a curve adds 2 endpoints, cost $+2\nu$ |
| closure | closed curves have $\operatorname{div}\tau=0$ |
| sparsity | open component ≥ $2\nu$; closed one ≥ $2\pi\xi$ (resp. $4\pi\sqrt{\alpha\beta}$) since it turns by $2\pi$ |

Note: the earlier proposal that the length term alone gives stability is
wrong. Removing a point does not change length, so the endpoint term is needed.

### Why the normal component $c$ is needed

At finite resolution every field is blurred, and **a blurred horizontal
current is not horizontal**. A sample with tangent $X_1(\theta_m)$, smeared
into the slice $\theta$, has components $a\cos(\theta-\theta_m)$ along
$X_1(\theta)$ and $a\sin(\theta-\theta_m)$ along $X_2(\theta)$. If $c$ is
dropped, a closed curve gets a spurious divergence of order
$\sigma_\theta/\sigma_x$ per unit length, which does **not** vanish under
refinement. Measured: endpoint mass ≈ 15 instead of 1.2 for the milestone-1
phantom. With $c$, blurring (in fixed Cartesian components) commutes with the
divergence, and only the discretization error remains. Horizontality is then
a metric weight $\zeta$. The trade-off: for a correctly blurred curve,
$c\approx a\,\Delta\theta$, so $\zeta>1$ inflates its length by roughly
$(\zeta^2-1)\sigma_\theta^2/2$ (≈ 3% at the default $\zeta=2$, 2-cell blur).

### Discretization

* Fourth-order central differences; zero extension in space; in $\theta$, $a$
  and $c$ are $\pi$-periodic and $b$ is $\pi$-*anti*periodic (the antipodal
  copy of a curve is traversed in reverse).
* **Resolution requirement:** the divergence is accurate only for fields
  resolved over ≥ 1.5–2 cells in $x$ and $\theta$. On a closed ellipse, the
  endpoint mass (ideal 0) is 10.6 at 1-cell blur with 2nd-order differences
  and 0.26 at 2-cell blur with 4th-order differences. Network outputs that are
  sharper than this pay a spurious cost proportional to their length.
* Integrals are over $\Omega\times\mathbb{RP}^1$, half the value on the double
  cover. $|\cdot|$ and $\sqrt{\cdot}$ are smoothed by `eps`. The functional is
  convex and 1-homogeneous in $(a,b,c)$.

### Known caveats

* Convex relaxations of curve energies tend to produce diffuse superpositions
  of curves ("ghosts"); watch for this once the prior is used for training.
* At polygon corners the true WF set is the whole fibre $\{p\}\times\mathbb{RP}^1$,
  but the closed lifted curve only uses the arc of the exterior angle. The
  rest of the fibre is a weaker singularity (Fourier decay one order faster),
  and the prior will tend to suppress it.
* The network outputs $(a,b,c)$ directly, with $a\ge0$; the classifier output
  is $m=1-e^{-a/a_{\mathrm{ref}}}$ (`models/README.md`). Positivity of $a$ is
  what makes the per-component lower bound of Theorem 3.9 hold.

Validated in `tests/test_regularization.py` ($96^2\times48$ grid):

* with $\zeta=1$, the length term is within 3% of the analytic
  sub-Riemannian length of an ellipse;
* endpoint mass: closed ellipse < 0.4, half an ellipse ≈ 2, two arcs ≈ 4,
  square without corner arcs ≈ 8;
* dropping $c$ increases the closed-curve endpoint mass more than 5×.

## `data_discrepancy.py`

### `MicrolocalForwardModel` + `ForwardModelDiscrepancy` (default)

Measurement model (Def. 5.4): $|c|=\mathcal R_tCq+\varepsilon$ with Laplace
noise, $q=J\,m\ge0$ the amplitude on the lifted image, and $\mathcal R_t$ the
known fibre response (`SinogramAnalyzer.fibre_response`, Prop. 5.3).
`MicrolocalForwardModel(C, r)(q)` returns $\hat c=\mathcal R_tCq$, normalized
so that $q=1$ on a curve resolved over 2 cells predicts a unit-height streak.
(Without the normalization the sum over the tube in $t$ made $m\approx0.1$
sufficient to explain full-strength edges.) `ForwardModelDiscrepancy()(ĉ, c)`
returns the smoothed mean absolute residual.

A single wavefront point now explains its whole streak along $t$, so mass
spread over whole tangent lines no longer fits (unlike in the mixture model
below). But the deconvolution along the fibre is badly ill-conditioned:
smears narrower than the fibre response (~30 cells) fit as well as the true
point (`tests/test_lpd.py`). Within that width, $t$ is determined by the
prior (the envelope of the tangent lines is the cheapest explanation,
Cor. 4.5).

**Weighting.** $D$ is a mean over all lifted voxels, whereas $E$ is an
integral (≈ curve length). Explaining an edge of unit length lowers $D$ by
only ~$10^{-3}$ (the edge's streaks cover a small fraction of the voxels). So
λ must be of order $10^{-4}$–$10^{-3}$. With λ = 0.01 the network collapsed to
$m\equiv0$ within 250 steps.

### `MixtureLikelihood` (baseline)

$$D(v;c)=-\operatorname{mean}\log\big(v\,p_1(|c|)+(1-v)\,p_0(|c|)\big),\qquad v=Cu,\ c=My.$$

* $p_0$: half-normal with the known coefficient noise std (from
  `SinogramAnalyzer.coefficient_noise_std`; $M$ is linear and the noise is
  Gaussian).
* $p_1$: exponential with a learnable scale, standing in for edge contrast.

This is a pseudo-likelihood: neighbouring coefficients are correlated through
the filter support, and the data weight relative to the prior absorbs that.
`posterior(c, prior)` gives the learning-free per-voxel posterior
$P(\text{WF}\mid c)$, which serves as the classical baseline.

**Finding from milestone 1: the data term alone is ambiguous along $t$.**
Local filtering determines $s$ and $\varphi$ of a sinogram singularity
sharply, but $t$ (position along the ray) only to within many cells. Pulled
back to the image, each edge point therefore spreads along the edge's
*tangent line*. Calibrated to sinogram noise, the posterior exceeds 0.5 on
63% of the image (it is large on every tangent line). Any $u$ that puts mass
anywhere on those tangent lines explains the data almost equally well. The
streaks are straight horizontal segments, one per orientation slice, and
together they sweep a surface in $X$. The true WF set is their envelope: a
single horizontal curve that crosses each streak once, and the cheapest way
under $\mathcal R$ to explain all of them. On the sinogram side this is the
contact condition $t=dh/d\varphi$ along the singular curve $s=h(\varphi)$. So $p_0$ cannot be a pure noise model, and the relative
weight of $D$ and $\mathcal R$ (and the endpoint cost of tangent streaks)
matters.

**Open design questions**: multi-scale coefficients and decay ratios to
separate edges from texture; a Poisson or log-transform noise model for real
data; angle-split self-supervision (predict from angle subset A, evaluate $D$
on subset B). See §9 of the background document.
