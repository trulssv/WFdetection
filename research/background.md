# Wavefront sets as Legendrian currents: background and methods

*Unsupervised indirect wavefront-set detection from tomographic data.*

This document collects the mathematical background of the project and
derives every component of the method from it: the representation of the
wavefront set, the curve prior, the canonical relation used to compare image
and data, the data terms, the discretization and the network. Standard results
are stated with references, and results specific to this project are proved.
The reader is assumed to know basic differential geometry (manifolds, vector
fields, differential forms, cotangent bundles) and some distribution theory.
Microlocal analysis, contact geometry and geometric measure theory are
introduced as needed. Measured numbers and experiment logs are kept in
[`experiments/README.md`](../experiments/README.md).

**Contents**

1. [Introduction](#1-introduction)
2. [Wavefront sets and contact geometry](#2-wavefront-sets-and-contact-geometry)
3. [Wavefront sets as currents; the prior](#3-wavefront-sets-as-currents-the-prior)
4. [Tomography and canonical relations](#4-tomography-and-canonical-relations)
5. [The measurement models](#5-the-measurement-models)
6. [Discretization](#6-discretization)
7. [Inference: MAP estimation and learned primal–dual](#7-inference-map-estimation-and-learned-primaldual)
8. [Relation to prior work](#8-relation-to-prior-work)
9. [Open problems](#9-open-problems)
10. [References](#10-references)

A. [Fan-beam geometry](#a-fan-beam-geometry)

**Notation.**

| symbol | meaning |
|---|---|
| $\Omega\subset\mathbb R^2$, $B$ | image domain (in code $[-1,1]^2$); unit disk $B$ containing the objects |
| $\omega(\theta)=(\cos\theta,\sin\theta)$, $\omega^\perp(\theta)=(-\sin\theta,\cos\theta)$ | unit vectors |
| $X=\Omega\times S^1$, $X'=\Omega\times\mathbb{RP}^1$ | cosphere bundle and projective cotangent bundle |
| $X_1=\omega^\perp(\theta)\cdot\nabla_x$, $X_2=\omega(\theta)\cdot\nabla_x$, $X_3=\partial_\theta$ | moving frame on $X$ |
| $\alpha=\cos\theta\,dx_1+\sin\theta\,dx_2$; $A(x,\theta)=(x,\theta+\pi)$ | contact form on $X$; antipodal map |
| $Rf(s,\varphi)$, $R^\#$ | ray transform, its formal adjoint |
| $\tilde Y=\mathbb R_s\times S^1_\varphi\times\mathbb R_t$, $Y=\tilde Y/\iota$ | lifted sinogram spaces of oriented / unoriented lines |
| $\beta=ds-t\,d\varphi$ | contact form on $\tilde Y$ |
| $C:X\to\tilde Y$ | lifted canonical relation, $(x,\theta)\mapsto(x\cdot\omega,\theta,x\cdot\omega^\perp)$ |
| $T=\vec\tau\,\Vert T\Vert$; $T=\tau\,dX$ | 1-current (polar form; absolutely continuous form) |
| $\tau=aX_1+bX_3+cX_2$ | frame components: tangential $a$, turning $b$, normal $c$ |
| $\ell_\theta,\ \zeta,\ \nu,\ \lambda$ | turning length scale (`xi` in code), horizontality weight, endpoint weight, prior weight |
| $E$, $\mathbf M_g$, $\mathbf M(\partial\cdot)$ | energy, mass in the metric $g$, boundary mass |
| $M$, $y_M=\vert My\vert$ | sinogram analyzer and its coefficient magnitudes on $Y$ |
| $W_\theta$, $y_W=\vert W_\theta*\mathrm{FBP}\,y\vert$ | image-side analyzer and its coefficient magnitudes on $X'$ |
| $r$, $r_\theta$ | fibre response (sinogram), angular response (image side) |
| $\mathcal G_{\mathbf n}$; $u_\perp$, $u_\parallel$ | generic oriented derivative-of-Gaussian kernel with unit normal $\mathbf n$, $K_t=\mathcal G_{\mathbf n(t)}$, $W_\theta=\mathcal G_{\omega(\theta)}$; its coordinates across / along the singular curve (Sections 5.1, 5.4) |
| $*$ | "filter with" (convolution; the code's correlation gives the same modulus, Section 5.1) |
| $\hat y_M=\mathcal R_tCq$, $\hat y_W=F_X(q)$ | predicted coefficients (Definitions 5.3, 5.4) |
| $\vert e\vert_\delta=\sqrt{e^2+\delta^2}-\delta$ | smoothed absolute value of a residual $e$ in the data terms, $\delta=10^{-2}$ |
| $\eta$ | relative threshold for ridge extraction (`evaluation/ridges.py`) |
| $q=a/a_{\mathrm{ref}}$, $m=1-e^{-q}$ | amplitude (edge strength) in the data terms; membership, a display and threshold transform (Section 7.3) |
| $\mathcal D_Y$, $\mathcal D_X$, $b_L$ | sinogram and image-side data terms, Laplace scale |
| $\sigma$; $\sigma_\perp,\sigma_\parallel,\sigma_e,\sigma_x,\sigma_\theta$ | sinogram covector scale; kernel widths across / along, edge and mollifier widths |
| $K$, $n$ | number of $\theta$-slices on $[0,\pi)$, image grid size |

**Spaces and parities.** Fields of the model live on $X'$ ($\theta\in[0,\pi)$)
and on $Y$ ($\varphi\in[0,\pi)$), which is what the code stores. The network
computes on the double covers $X$ and $\tilde Y$ (Section 4.2), each with $2K$
slices and plain periodic convolutions. By Lemma 2.6, $a$, $c$, $m$, $q$
and $y_W$ are even under $A$ (functions on $X'$), and $b$ is odd
($\pi$-antiperiodic, a section of the Möbius bundle). On $\tilde Y$,
$y_M$ is extended by $y_M(s,\varphi+\pi,t)=y_M(-s,\varphi,-t)$.

---

## 1. Introduction

### 1.1 What is an edge?

Image analysis has many working definitions of "edge": large gradients,
zero crossings of the Laplacian, maxima of a filter response. All of them
depend on a scale and a threshold. Distribution theory has a definition
without either: an edge is a place where the image is **singular** (not
$C^\infty$), together with the **direction** in which it is singular. A jump
across a smooth curve is singular only in the direction normal to the curve.
Along the curve the image is as regular as the curve itself. The
*wavefront set* $\mathrm{WF}(f)\subset\Omega\times(\mathbb R^2\setminus0)$
(Definition 2.1) records exactly these position–direction pairs. It is the
mathematically canonical notion of "edge with orientation", and detecting it
is the classification task of this project.

Two features of $\mathrm{WF}(f)$ drive the method.

1. **It has geometric structure.** For a cartoon image (piecewise smooth
   with piecewise smooth boundaries) the wavefront set consists of finitely
   many curves in the 3-manifold $X=\Omega\times S^1$, plus full fibres over
   corners and junctions. The curves are not arbitrary: the direction is the
   normal of the curve's projection, so they are tangent to a fixed plane
   field. They are *Legendrian* curves of the canonical contact structure
   (Corollary 2.10). A prior on wavefront sets should therefore be a prior on
   Legendrian curves. This turns the vague desideratum "edges form smooth
   curves, and the orientation changes continuously along them" into a
   precise geometric constraint.
2. **It propagates predictably through the measurement.** The ray transform
   is a Fourier integral operator. Its *canonical relation* $C$ maps
   $\mathrm{WF}(f)$ bijectively onto $\mathrm{WF}(Rf)$ (Theorem 4.3), so the
   wavefront set of the image is determined by that of the data. $C$ is
   moreover a contactomorphism (Proposition 4.4), so the Legendrian prior has
   an exact counterpart on the data side. Avoiding reconstruction matters for
   limited data, where FBP adds spurious singularities (Theorem 4.7); with full
   data the default data term does use FBP (Section 5.4).

### 1.2 The method in one paragraph

We represent a candidate wavefront set as a 1-current $T$ on $X$, a
vector-valued measure: a union of curves is a sum of currents, length is mass,
and endpoints are the boundary $\partial T$. The prior is the Gibbs measure of
the convex, local energy $E(T)=\mathbf M_g(T)+\nu\,\mathbf M(\partial T)$:
length in a sub-Riemannian-type metric plus $\nu$ times the boundary mass
(for curves of multiplicity one, the number of endpoints); see the table
below. The data enter through fixed linear analyzers followed by a modulus:
sinogram coefficients $y_M=|My|$ on $Y$ and, by default, full-aperture
image-side coefficients $y_W=|W_\theta*\mathrm{FBP}\,y|$ on $X'$. Each is
predicted from the amplitude $q=a/a_{\mathrm{ref}}$ (the normalized line density, which measures edge strength) by a known
blur, along the fibre after pushing $q$ through $C$ (Definition 5.3) or
locally in $X'$ (Definition 5.4). The MAP estimate is computed by an unrolled
learned primal–dual network that alternates CNN updates on $\tilde Y$ and on
$X$, coupled by $C$, and is trained by minimizing the same objective over a
distribution of data. No labels are used. For the diffuse, real-weighted
output of the network, the prior is an $\ell^1$/TV-type sparsity penalty on
total multiplicity, not on the number of components (Section 3.3).

### 1.3 Five desiderata for a prior, made precise

The project started from five informal requirements: a prior $P$ on unions of
curves $x=\sqcup_k x_k$ in $X$ should have:

| desideratum | precise version | hypotheses | where |
|---|---|---|---|
| 1. independence, $P(x)=\prod_kP(x_k)$ | $E$ is additive on currents with disjoint supports | normal currents | Prop 3.7(i) |
| 2. smoothness | $E$ charges total turning, $\int\sqrt{1+\ell_\theta^2\kappa^2}\,ds$; corners cost $\ell_\theta\cdot$angle | Legendrian curves | Prop 3.7(ii) |
| 3. stability (breaking a curve is unlikely) | opening a gap of length $\varepsilon$ costs $2\nu-\varepsilon$ | Legendrian curves | Prop 3.7(iii) |
| 4. closed curves are favoured | closed curves have $\partial T=0$; cutting one open costs $2\nu-\varepsilon$ | Legendrian curves | Prop 3.7(i),(iii) |
| 5. sparsity | every component costs at least $\min(2\nu,2\pi\ell_\theta)$ | integral, Legendrian ($c=0$), $a\ge0$ | Thm 3.8 |

A length term alone *rewards* gaps (it saves $\varepsilon$). Only the boundary
term makes a short gap cost something: gaps shorter than $2\nu$ are
penalized. For real-weighted currents, which is what the convex relaxation and
the network produce, property 5 holds only per unit multiplicity (Section 3.3).

---

## 2. Wavefront sets and contact geometry

### 2.1 The wavefront set

**Definition 2.1 (wavefront set; Hörmander [H1, §8.1]).** Let
$u\in\mathcal D'(\Omega)$. A point $(x_0,\xi_0)\in\Omega\times(\mathbb
R^2\setminus0)$ is *not* in $\mathrm{WF}(u)$ if there are a cutoff
$\chi\in C_c^\infty(\Omega)$ with $\chi(x_0)\ne0$ and an open cone
$\Gamma\ni\xi_0$ such that
$$|\widehat{\chi u}(\xi)|\le C_N(1+|\xi|)^{-N}\qquad\text{for all }\xi\in\Gamma,\ N\in\mathbb N.$$

**Proposition 2.2 (basic properties [H1, Ch. 8]).**
1. $\mathrm{WF}(u)$ is closed and conic in $\xi$; its projection to $\Omega$
   is the singular support of $u$.
2. If $u$ is real-valued, $(x,\xi)\in\mathrm{WF}(u)\iff(x,-\xi)\in\mathrm{WF}(u)$
   (since $\widehat{\chi u}(-\xi)=\overline{\widehat{\chi u}(\xi)}$).
3. $\mathrm{WF}(Pu)\subset\mathrm{WF}(u)$ for every pseudodifferential operator
   $P$, with equality where $P$ is elliptic.

Because $\mathrm{WF}(u)$ is conic, only the direction of $\xi$ matters.

**Definition 2.3 (cosphere bundle).** The *cosphere bundle*
$S^*\Omega=(T^*\Omega\setminus0)/\mathbb R_{>0}$ is identified with
$X=\Omega\times S^1$ via $\xi=|\xi|\,\omega(\theta)$. The *projective
cotangent bundle* $P^*\Omega=(T^*\Omega\setminus0)/\mathbb R^\times$ is
identified with $X'=\Omega\times\mathbb{RP}^1$, $\theta$ mod $\pi$. By
Proposition 2.2(2), for real images $\mathrm{WF}(u)$ is the preimage of a
subset of $X'$.

### 2.2 The contact structure

The cotangent bundle carries the tautological 1-form $\xi\cdot dx$.
Restricted to unit covectors it descends to $X$:

**Definition 2.4 (contact form, frame).** On $X$ let
$$\alpha=\cos\theta\,dx_1+\sin\theta\,dx_2,$$
and define the frame
$$X_1=-\sin\theta\,\partial_{x_1}+\cos\theta\,\partial_{x_2},\qquad
X_2=\cos\theta\,\partial_{x_1}+\sin\theta\,\partial_{x_2},\qquad
X_3=\partial_\theta .$$
Then $\ker\alpha=\operatorname{span}\{X_1,X_3\}=:\mathcal H$ (the *horizontal
distribution*) and $\alpha(X_2)=1$. One computes
$$d\alpha=-\sin\theta\,d\theta\wedge dx_1+\cos\theta\,d\theta\wedge dx_2,\qquad
\alpha\wedge d\alpha=-\,dx_1\wedge dx_2\wedge d\theta\neq0,$$
so $\alpha$ is a **contact form** and $\mathcal H$ is a contact structure
[G]. The brackets
$$[X_1,X_2]=0,\qquad[X_3,X_1]=-X_2,\qquad[X_3,X_2]=X_1$$
are those of the Lie algebra $\mathfrak{se}(2)$; indeed $X$ is an open subset
of $SE(2)\cong\mathbb R^2\times S^1$ and $X_1,X_2,X_3$ are left-invariant
vector fields. Since $X_1,X_3$ and their bracket span $TX$, $\mathcal H$ is
bracket generating (Hörmander's condition). By Chow–Rashevskii, any two
points of $X$ (for connected $\Omega$) are joined by a horizontal curve, and
$(X,\mathcal H)$ with a metric on $\mathcal H$ is a sub-Riemannian manifold.
This is the geometry of the primary visual cortex model of Petitot and
Citti–Sarti [Pe, CS].

All three fields are divergence free with respect to the volume
$dX=dx_1\,dx_2\,d\theta$, because their coefficients do not depend on the
variable they differentiate.

**Definition 2.5 (Legendrian curve).** A Lipschitz curve
$\gamma=(z,\theta):I\to X$ is *Legendrian* (horizontal) if
$\dot\gamma(t)\in\mathcal H$ a.e., i.e. $\alpha(\dot\gamma)=\dot z\cdot\omega(\theta)=0$.
Equivalently $\dot\gamma=vX_1+\dot\theta X_3$, i.e. $\dot z=v\,\omega^\perp(\theta)$,
for a speed function $v$. It is *positively Legendrian* if $v\ge0$: the
projected curve moves in the direction $\omega^\perp(\theta)$, so
$\omega(\theta)$ is its normal and the curve is traversed with the normal on
its right-hand side.

**Lemma 2.6 (antipodal symmetry).** Let $A(x,\theta)=(x,\theta+\pi)$. Then
$A_*X_1=-X_1$, $A_*X_2=-X_2$, $A_*X_3=X_3$ and $A^*\alpha=-\alpha$. If $\gamma$
is positively Legendrian, then $A\circ\gamma$ with reversed orientation is
positively Legendrian, with turning $-\dot\theta$. Consequently a current
invariant under $A$ composed with orientation reversal (Section 3) has frame
components satisfying
$$(a,b,c)(\theta+\pi)=(a,-b,c)(\theta),$$
so on $X'$, $a$ and $c$ are functions and $b$ is $\pi$-antiperiodic.

*Proof.* $X_{1,2}(\theta+\pi)=-X_{1,2}(\theta)$ and $\omega(\theta+\pi)=-\omega(\theta)$.
So $A$ pushes $aX_1+bX_3+cX_2$ to $-aX_1+bX_3-cX_2$ at $\theta+\pi$, and
reversing the orientation gives $aX_1-bX_3+cX_2$. ∎

### 2.3 The wavefront set of a cartoon image

**Theorem 2.7 (conormal singularities; [KQ]).** Let
$D_0\subset\Omega$ be open with $C^\infty$ boundary and $g\in C^\infty(\Omega)$
with $g\neq0$ on $\partial D_0$. Then
$$\mathrm{WF}(g\,\mathbf 1_{D_0})=N^*\partial D_0\setminus0=\{(x,\xi):x\in\partial D_0,\ \xi\in\mathbb R^\times n(x)\},$$
the conormal bundle of the boundary, where $n$ is the unit normal.

*Sketch.* Away from $\partial D_0$ the function is smooth. Near a boundary
point, a diffeomorphism straightens $\partial D_0$ to $\{x_2=0\}$. The
function becomes $\tilde g\,\mathbf 1_{x_2>0}$, whose localized Fourier
transform decays rapidly in every direction except $\pm e_2$, where it decays
only like $|\xi|^{-1}$. Wavefront sets transform as covectors under
diffeomorphisms (cf. the conormal wavefront set of the surface density
$\delta_{\partial D_0}$, [H1, Thm 8.1.5]). ∎

**Proposition 2.8 (Legendrian lift).** Let $z:I\to\Omega$ be a regular
$C^2$ curve with unit normal $\omega(\theta(t))$, oriented so that
$\dot z=|\dot z|\,\omega^\perp(\theta)$. Its *Legendrian lift*
$\gamma=(z,\theta)$ is a positively Legendrian curve with
$$\dot\gamma=|\dot z|\,(X_1+\kappa X_3),$$
where $\kappa=\dot\theta/|\dot z|$ is the signed curvature. Conversely, a
positively Legendrian curve with $v>0$ is the lift of its projection.
(Legendrian curves with $v$ changing sign project to *fronts* with cusps
[A2]. Positivity excludes them.)

*Proof.* $\alpha(\dot\gamma)=|\dot z|\,\omega^\perp\cdot\omega=0$ and
$\dot z=|\dot z|X_1$. Differentiating the unit normal gives
$\dot\theta=\kappa|\dot z|$. Conversely, if $\dot z=v\,\omega^\perp(\theta)$ with
$v>0$, then $\omega(\theta)$ is the normal of $z$. ∎

**Proposition 2.9 (corners).** Let $D_0$ have piecewise $C^\infty$ boundary
and let $p$ be a point where two $C^\infty$ arcs meet at an interior angle
$\neq\pi$. Then $\mathrm{WF}(\mathbf 1_{D_0})$ contains the whole fibre
$\{p\}\times(\mathbb R^2\setminus0)$. For a polygon,
$\widehat{\chi\mathbf 1_{D_0}}$ decays like $|\xi|^{-2}$ in directions outside
the two edge normals, one order faster than along them.

*Sketch.* Near $p$, $\mathbf 1_{D_0}$ is to leading order the indicator of a
sector, homogeneous of degree 0. Its Fourier transform is homogeneous of
degree $-2$ and nonzero in all but finitely many directions (Corollary 2.10);
the edges contribute degree $-1$ terms in their normal directions. ∎

Piecewise $C^2$ would not suffice: where $\partial D_0$ is $C^2$ but not
$C^\infty$ (e.g. a curvature jump), the transform decays only polynomially in
every direction, so the full fibre is in $\mathrm{WF}$ too. We call every
non-$C^\infty$ boundary point a *corner*.

Lifted to $X$, the boundary of a polygon with outward normals $\theta_i$ is
a closed Legendrian curve: the edges at constant $\theta_i$ are connected at
the corners by vertical arcs ($\dot z=0$, $\dot\theta\ne0$), which are
horizontal since $X_3\in\mathcal H$. The arc turns by the exterior angle.
**Modeling choice.** The prior targets this closed Legendrian lift (edges
plus turning arcs; `data/phantoms.py`, `corners='arc'`). The rest of the
corner fibre (a weaker singularity, Proposition 2.9) and the full fibres at
junctions are not modeled. The validation labels do include full corner
fibres (`data/synthetic.py:85-89`, `'full'`), so validation AUC also rewards
corner directions that the prior does not represent.

**Corollary 2.10 (structure of cartoon wavefront sets).** Let
$f=\sum_k J_k\mathbf 1_{D_k}+g$ with $g$ smooth and finitely many domains with
piecewise $C^\infty$ boundaries. Let $S_f$ be the jump set, $V\subset S_f$ the
finite set of points where $S_f$ is not locally a single $C^\infty$ curve
(corners, junctions, crossings), and $\Gamma_\ell\subset X$ the closures of the
lifts, with both normal directions, of the components of $S_f\setminus V$.
Then the image of $\mathrm{WF}(f)$ in $X$ satisfies
$$\mathrm{WF}(f)\subset\bigcup_\ell\Gamma_\ell\ \cup\ \bigcup_{p\in V}\{p\}\times S^1,$$
with equality over the components along which the jump $[f]$ is nonzero.
Each $\Gamma_\ell$ is a positively Legendrian curve (Proposition 2.8) together
with its antipodal image, which is positively Legendrian when given the
reversed orientation (Lemma 2.6). Let $p\in V$ be a point near which $f$ is,
up to a smooth function, homogeneous of degree 0 (straight edges, as in
polygons), and let $J(\vartheta)$ be the jump of $f$ across the ray
$p+\mathbb R_{>0}\omega(\vartheta)$, counter-clockwise. Then
$\{p\}\times S^1\subset\mathrm{WF}(f)$ if and only if
$\vartheta\mapsto J(\vartheta)+J(\vartheta+\pi)$ is not identically zero. This
holds at corners and at T-junctions, but not at transversal crossings, which
carry only the conormals of the two edges.

*Sketch.* Away from $V$, apply Theorem 2.7 locally. At $p\in V$, the degree
$-2$ part of the localized Fourier transform at $\xi=|\xi|\omega(\phi)$ is
$|\xi|^{-2}\sum_jJ(\vartheta_j)\tan(\vartheta_j-\phi)$, summed over the rays.
As $\tan$ is $\pi$-periodic, this vanishes identically iff the even part
vanishes; otherwise it is nonzero for all but finitely many $\phi$, and
$\mathrm{WF}$ is closed. ∎

This is the structural fact behind the whole method: **the prior should live
on (positively) Legendrian curves in $X$.** Two consequences:

* *Crossings separate in $X$.* Two edges crossing in $\Omega$ at a nonzero
  angle have disjoint lifts, and the crossing carries no extra fibre, so
  independence of components (desideratum 1) is natural in $X$. At a
  T-junction $\mathrm{WF}(f)$ contains the full fibre; the model drops it, and
  the stem's lift ends with an endpoint. The phantoms contain crossings but no
  deliberate T-junctions, so this case is untested.
* *Orientation is not free.* A field that puts orientation $\theta$ on an edge
  whose tangent is not $\omega^\perp(\theta)$ is not Legendrian. The contact
  condition is a hard constraint between position and orientation, which a
  prior that treats the $\theta$-channels as independent images ignores.

---

## 3. Wavefront sets as currents; the prior

To put a prior on unions of curves we need a space where "union" is a linear
operation and length, endpoints and closedness are continuous functionals.
Currents provide it [F, M].

### 3.1 Currents

**Definition 3.1 (1-currents, mass, boundary).** A *1-current* on $X$ is a
continuous linear functional on compactly supported smooth 1-forms. The
*boundary* $\partial T$ is the 0-current $\partial T(\phi)=T(d\phi)$. Given a
Riemannian metric $g$, the *mass* is
$\mathbf M_g(T)=\sup\{T(\eta):|\eta|_{g^*}\le1\}$. $T$ is *normal* if
$\mathbf M(T)+\mathbf M(\partial T)<\infty$, and *integral* if $T$ and
$\partial T$ are integer-multiplicity rectifiable [F, 4.1.24]. Fix the
reference metric $g_0$ in which $X_1,X_2,X_3$ are orthonormal. A normal
1-current has a polar decomposition $T=\vec\tau\,\Vert T\Vert$ with
$\Vert T\Vert$ its $g_0$-mass measure and $|\vec\tau|_{g_0}=1$
$\Vert T\Vert$-a.e.; then $\mathbf M_g(T)=\int|\vec\tau|_g\,d\Vert T\Vert$ for
every $g$. When $T$ is absolutely continuous we write $T=\tau\,dX$ with a
vector field $\tau$ (the case used on grids):
$$T(\eta)=\int_X\eta(\tau)\,dX,\qquad\mathbf M_g(T)=\int_X|\tau|_g\,dX,\qquad\partial T=-(\operatorname{div}\tau)\,dX.$$
$T$ is a *cycle* if $\partial T=0$.

**Definition 3.2 (current of a curve).** An oriented Lipschitz curve
$\gamma:[0,1]\to X$ defines $[\gamma](\eta)=\int_0^1\eta_{\gamma(t)}(\dot\gamma(t))\,dt$.
Then $\mathbf M_g([\gamma])\le L_g(\gamma)$, with equality if $\gamma$ is
injective (then $\Vert[\gamma]\Vert$ is $\mathcal H^1$ on the image and
$\vec\tau$ the unit tangent), and
$\partial[\gamma]=\delta_{\gamma(1)}-\delta_{\gamma(0)}$. So
$\mathbf M(\partial[\gamma])=2$ for an arc and $0$ for a closed curve.

**Definition 3.3 (Legendrian currents).** Write
$\vec\tau=aX_1+bX_3+cX_2$ with functions $a,b,c$ defined $\Vert T\Vert$-a.e.
$T$ is *Legendrian* if $c=0$ $\Vert T\Vert$-a.e. ($\alpha(\vec\tau)=0$) and
*positively Legendrian* if additionally $a\ge0$. For $T=\tau\,dX$ the same
conditions apply to the components of $\tau$. By Proposition 2.8, the currents
of lifted edges are positively Legendrian with $\vec\tau\propto X_1+\kappa X_3$;
on the grid, $a$ is a line density and $b\approx a\kappa$.

**Theorem 3.4 (decomposition into curves).** All statements are in
$\mathbb R^d$; they apply on $X$ through the embedding
$(x,\theta)\mapsto(x,\cos\theta,\sin\theta)\in\mathbb R^4$, since the curves lie
in $\operatorname{supp}T$.
1. [F, 4.2.25] An integral 1-current is a countable sum $T=\sum_k[\gamma_k]$ of
   Lipschitz curves that are injective arcs or simple closed curves, with
   $$\mathbf M(T)=\sum_k\mathbf M([\gamma_k]),\qquad\mathbf M(\partial T)=\sum_k\mathbf M(\partial[\gamma_k]),\qquad\mathbf M([\gamma_k])=L(\gamma_k).$$
2. [Sm, PS] A normal 1-current splits as $T=T_{\mathrm{ac}}+Z$ with $Z$ a cycle, $T_{\mathrm{ac}}$
   acyclic and $\mathbf M(T)=\mathbf M(T_{\mathrm{ac}})+\mathbf M(Z)$. Moreover
   $T_{\mathrm{ac}}=\int[\gamma]\,d\mu(\gamma)$ over simple arcs with
   $\Vert T_{\mathrm{ac}}\Vert=\int\Vert[\gamma]\Vert\,d\mu$ as measures and
   $\mathbf M(\partial T)=2\Vert\mu\Vert$, while $Z$ is a superposition of
   elementary solenoids, which need not be closed curves.

Since $\Vert T\Vert\le\sum_k\Vert[\gamma_k]\Vert$ always (likewise in
(ii)), equality of total masses forces equality of the measures, hence
$\vec\tau_{[\gamma_k]}=\vec\tau$ a.e. So the mass identities hold in every
metric $g$, and the pieces of a (positively) Legendrian current are
(positively) Legendrian.

### 3.2 The energy and the prior

**Definition 3.5 (energy).** For $\ell_\theta>0$, $\zeta\ge1$ let
$g=g_{\ell_\theta,\zeta}$ be the left-invariant metric with orthogonal frame
$X_1,X_3,X_2$ and $|X_1|=1$, $|X_3|=\ell_\theta$, $|X_2|=\zeta$. For $\nu>0$,
$$E(T)=\mathbf M_g(T)+\nu\,\mathbf M(\partial T)=\int|\vec\tau|_g\,d\Vert T\Vert+\nu\,\mathbf M(\partial T).$$
For $T=\tau\,dX$,
$$E(T)=\int_X\sqrt{a^2+\ell_\theta^2b^2+\zeta^2c^2}\,dX+\nu\int_X\bigl|X_1a+X_3b+X_2c\bigr|\,dX ,$$
using $\operatorname{div}(aX_1+bX_3+cX_2)=X_1a+X_3b+X_2c$, valid because the
$X_i$ are divergence free. On Legendrian currents ($c=0$) the mass is the
sub-Riemannian length with $|X_3|=\ell_\theta$, the length scale at which
turning and moving cost the same. $\zeta$ penalizes the normal component:
finite $\zeta$ is the Riemannian approximation of the sub-Riemannian metric,
which is recovered as $\zeta\to\infty$; finite resolution forces a moderate
$\zeta$ (Proposition 6.2).

*Variant (elastica).* The convex, 1-homogeneous mass
$\int\alpha_e\sqrt{a^2+\zeta^2c^2}+\beta_e\,b^2/a\,dX$ gives Euler's elastica
$\int(\alpha_e+\beta_e\kappa^2)\,ds$ on lifted curves [Mu]. It makes corners
infinitely expensive, so the sub-Riemannian form is the default.

**Definition 3.6 (prior).** The prior is the Gibbs measure
$P(dT)\propto\exp(-\lambda E(T))$, formal on currents. We use $P$ only through
(i) the MAP functional $-\log P=\lambda E+\mathrm{const}$ and (ii) its
discretization, where $\exp(-\lambda E_h)$ is a proper log-concave density
(Proposition 6.3). Samples of $\exp(-\lambda E_h)$ do not look like curves,
as is familiar for TV-type priors [LS]: the prior is a MAP device.

**Proposition 3.7 (energy of curve configurations).**
1. *Independence.* If $T_1,T_2$ are normal with
   $\operatorname{supp}T_1\cap\operatorname{supp}T_2=\emptyset$, then
   $E(T_1+T_2)=E(T_1)+E(T_2)$. In particular, for injective arcs or simple
   closed Legendrian curves $\gamma_k$ with pairwise disjoint images,
   $$E\Bigl(\sum_k[\gamma_k]\Bigr)=\sum_k\Bigl(L_{\ell_\theta}(\gamma_k)+\nu\,\mathbf M(\partial[\gamma_k])\Bigr),\qquad L_{\ell_\theta}(\gamma)=\int\sqrt{|\dot z|^2+\ell_\theta^2\dot\theta^2}\,dt ,$$
   with $\mathbf M(\partial[\gamma_k])=2$ for arcs and $0$ for closed curves.
2. *Turning.* For the lift of a $C^2$ curve,
   $E=\int\sqrt{1+\ell_\theta^2\kappa^2}\,ds+\nu\,\mathbf M(\partial[\gamma])$. This
   is a total-turning penalty with only a weak (strictly convex) preference
   for spreading curvature; a corner costs a finite $\ell_\theta\cdot$angle,
   which is intended, since cartoons have corners.
3. *Gaps.* Deleting from one $\gamma_k$ in (i), open or closed, a sub-arc of
   $L_{\ell_\theta}$-length $\varepsilon$ around an interior point increases
   $E$ by $2\nu-\varepsilon$.

*Proof.* (i) Both terms are integrals of local densities, and
$\operatorname{supp}\partial T_i\subset\operatorname{supp}T_i$; for curves,
$\mathbf M_g([\gamma_k])=L_g(\gamma_k)=L_{\ell_\theta}(\gamma_k)$ since
$c=0$. (ii) Proposition 2.8. (iii) By (i): the pieces are injective with
disjoint images, their total length is $L_{\ell_\theta}(\gamma_k)-\varepsilon$,
and they have two more endpoints than $\gamma_k$. ∎

*Remark.* As $\varepsilon\to0$ the cut curve converges to $[\gamma_k]$ in
mass, but the energy jumps by $2\nu$: $E$ is lower semicontinuous [F, 4.1.7]
but not continuous. A measure-zero gap is invisible, as it should be, and gaps
of length $\varepsilon<2\nu$ are penalized, while longer ones are favoured:
$2\nu$ is the length below which the prior bridges gaps.

**Theorem 3.8 (every component costs a fixed minimum).** Let $T$ be an
integral, positively Legendrian 1-current and $T=\sum_k[\gamma_k]$ its
decomposition (Theorem 3.4(i)). Then $E(T)=\sum_kE([\gamma_k])$, every arc
has $E([\gamma_k])\ge2\nu$, and every closed $\gamma_k$ satisfies
$$L_{\ell_\theta}(\gamma_k)\ge\ell_\theta\int|\dot\theta|\,dt\ \ge\ 2\pi\ell_\theta .$$
Hence $E(T)\ge N\min(2\nu,2\pi\ell_\theta)$, where $N$ is the number of curves
(a curve of integer multiplicity $k$ counts $k$ times).

*Proof.* Additivity and positivity of the pieces follow from Theorem 3.4. For
arcs, $\mathbf M(\partial[\gamma_k])=2$. Let $\gamma=(z,\theta)$ be closed and
positively Legendrian, $\dot z=v\,\omega^\perp(\theta)$ with $v\ge0$. Suppose
the image of $\theta$ lies in an open half-circle $(\theta_0-\pi,\theta_0)$.
Then $\frac{d}{dt}z\cdot\omega(\theta_0)=v\sin(\theta_0-\theta)\ge0$, with
equality only where $v=0$. Since $z$ is closed, $v=0$ a.e., so $z$ is constant
and $\theta$ is a closed path in an interval. Then $[\gamma]=0$, contradicting
$\mathbf M([\gamma])=L(\gamma)>0$. Hence the image of $\theta$ is not contained
in any open half-circle. Either it is the whole circle, so the total
variation is at least $2\pi$, or it is an arc of length at least $\pi$ that a
closed path must traverse in both directions, giving variation at least
$2\pi$. Finally $\sqrt{|\dot z|^2+\ell_\theta^2\dot\theta^2}\ge\ell_\theta|\dot\theta|$. ∎

This is Fenchel's theorem [Fe] adapted to Legendrian curves with corners.
**Positivity $a\ge0$ is essential:** the thin triangle front with legs along
$\omega^\perp(\theta_0)$, $-\omega^\perp(\theta_0\pm\delta)$ at those
orientations (lengths $\varepsilon$, $\varepsilon/(2\cos\delta)$ twice, joined
by vertical arcs) is a non-retracing closed Legendrian curve with turning
$4\delta$ and length $\approx2\varepsilon$, so its energy tends to 0.

### 3.3 What the prior guarantees for the network output

The network output (Section 7.3) has $a\ge0$ but is neither Legendrian nor
integral, and Theorem 3.8 covers neither case.

* *Non-Legendrian currents.* With $c\ne0$, $\dot z=a\,\omega^\perp(\theta)+c\,\omega(\theta)$,
  and the derivative in the proof gains the term $c\cos(\theta_0-\theta)$,
  which has no sign. A loop of size $\ell$ can move backwards with its normal
  component at orientations $\theta_0\pm\delta$, paying about
  $\zeta\ell/\delta$ in normal mass and $\ell_\theta\delta$ in turning; so
  small closed loops cost $O(\sqrt{\zeta\ell_\theta\ell})\to0$. Since $c\ne0$
  is forced by finite resolution (Proposition 6.2), the minimum cost per
  component is lost at the resolution scale.
* *Non-integral currents.* $E$ is 1-homogeneous: a component of weight $w$
  costs $w\,E([\gamma])$. For $T=\sum_kw_k[\gamma_k]$ with real $w_k>0$ and the
  $\gamma_k$ as in Theorem 3.8, $E(T)\ge\min(2\nu,2\pi\ell_\theta)\sum_kw_k$.
  This bounds the *total multiplicity*, not the number of components: the
  relaxed prior is an $\ell^1$/TV-type sparsity prior, not a cardinality
  prior, and weak spurious components are cheap (open problem 4).

*Remark (relation to total variation).* For $f=\sum_kJ_k\mathbf 1_{D_k}+g$,
$\mathrm{TV}(f)=\int|\nabla g|\,dx+\int_{S_f}|[f]|\,d\mathcal H^1$ (BV structure
theorem). At $\ell_\theta=0$ the lift of $S_f$ weighted by $|[f]|$ has mass
equal to the jump part (twice it with the antipodal copy); this is
$\sum_k|J_k|\,\mathcal H^1(\partial D_k)=\mathrm{TV}(f)$ only if $g=0$ and the
boundaries are pairwise disjoint up to $\mathcal H^1$-null sets (shared edges
carry $|J_i-J_j|$). So $E$ on a contrast-weighted current is a lifted,
curvature- and endpoint-aware TV, related to [CP2] and [BPW]. The wavefront
set is independent of contrast, so one could ask for a prior on the
*unweighted* current. With $q=a/a_{\mathrm{ref}}$ (Section 7.3) the line
density $a$ is proportional to edge strength, so the prior acts on a
contrast-weighted current, a lifted TV, by design, and $\lambda$ sets the
weakest contrast that is still detected (open problem 3).

---

## 4. Tomography and canonical relations

### 4.1 The ray transform as a Fourier integral operator

**Definition 4.1.** For $f\in\mathcal E'(B)$, $B$ the unit disk,
$$Rf(s,\varphi)=\int_{\mathbb R}f\bigl(s\,\omega(\varphi)+t\,\omega^\perp(\varphi)\bigr)\,dt,\qquad(s,\varphi)\in\mathbb R\times[0,\pi).$$
It satisfies $Rf(-s,\varphi+\pi)=Rf(s,\varphi)$, which extends it to
$\mathbb R\times S^1_\varphi$. Its formal adjoint is
$R^\#g(x)=\int_0^\pi g(x\cdot\omega(\varphi),\varphi)\,d\varphi$.

**Theorem 4.2 (Guillemin–Sternberg, Quinto [GS, Q1]).** $R$ is an elliptic
Fourier integral operator [H2, Ch. 25] of order $-\tfrac12$ with canonical
relation
$$\mathcal C=\Bigl\{\bigl(s,\varphi;\ \sigma,\,-\sigma\,x\cdot\omega^\perp(\varphi)\bigr),\ \bigl(x;\ \sigma\,\omega(\varphi)\bigr)\ :\ s=x\cdot\omega(\varphi),\ \sigma\neq0\Bigr\}.$$
$\mathcal C$ is the graph of a diffeomorphism between conic open sets (the
Bolker condition), and $R^\#R=2\pi(-\Delta)^{-1/2}$ is an elliptic
pseudodifferential operator of order $-1$. In particular
$\|Rf\|_{H^{r+1/2}}\simeq\|f\|_{H^r}$ for $f$ supported in $B$.

*Derivation of $\mathcal C$.* Write $Rf(s,\varphi)=(2\pi)^{-1}\iint e^{i\sigma(s-x\cdot\omega(\varphi))}f(x)\,dx\,d\sigma$.
The phase $\Phi=\sigma(s-x\cdot\omega(\varphi))$ has
$\partial_s\Phi=\sigma$, $\partial_\varphi\Phi=-\sigma x\cdot\omega^\perp(\varphi)$,
$-\partial_x\Phi=\sigma\omega(\varphi)$, and the critical set
$\partial_\sigma\Phi=0$ is $s=x\cdot\omega(\varphi)$. ∎

**Theorem 4.3 (microlocal correspondence).** For $f\in\mathcal E'(B)$, with
$Rf$ regarded on $\mathbb R\times S^1_\varphi$,
$$\mathrm{WF}(Rf)=\mathcal C\circ\mathrm{WF}(f).$$

*Proof.* "$\subset$" is Hörmander's bound $\mathrm{WF}(Ku)\subset\mathrm{WF}'(K)\circ\mathrm{WF}(u)$
[H1, Thm 8.2.13] applied to the Schwartz kernel $K$ of $R$. Its wavefront
set is the (twisted) canonical relation $\mathcal C$, and it has no
covectors that vanish on either side, so the extra terms of the theorem
are empty [KQ].
For "$\supset$", $R^\#R$ is elliptic, so by Proposition 2.2(3)
$\mathrm{WF}(f)=\mathrm{WF}(R^\#Rf)\subset\mathcal C^t\circ\mathrm{WF}(Rf)$.
Since $\mathcal C$ is a bijective canonical graph, applying $\mathcal C$ gives
$\mathcal C\circ\mathrm{WF}(f)\subset\mathrm{WF}(Rf)$. ∎

### 4.2 The canonical relation on cosphere bundles

Covectors in $\mathrm{WF}(Rf)$ have $\sigma_s=\sigma\ne0$; write them as
$\sigma(1,-t)$. Then $t=-\sigma_\varphi/\sigma_s$ is the affine chart over
$\sigma_s\ne0$ of the projectivized cotangent bundle of $\mathbb R\times S^1_\varphi$,
and $t=x\cdot\omega^\perp(\varphi)$ is the position along the ray of the
point $x$ where it touches the edge. With
$\tilde Y=\mathbb R_s\times S^1_\varphi\times\mathbb R_t$, the **lifted
canonical relation** is
$$C:X\to\tilde Y,\qquad C(x,\theta)=(s,\varphi,t)=\bigl(x\cdot\omega(\theta),\ \theta,\ x\cdot\omega^\perp(\theta)\bigr),$$
a diffeomorphism from $B\times S^1$ onto $\{s^2+t^2<1\}$, a rotation in each
$\theta$-slice. It intertwines $A$ with $\iota(s,\varphi,t)=(-s,\varphi+\pi,-t)$,
so it descends to $X'\to Y=\tilde Y/\iota$. On fields, $(Cu)(s,\varphi,t)=u(s\omega(\varphi)+t\omega^\perp(\varphi),\varphi)$
(`operators/canonical_relation.py`).

**Proposition 4.4 ($C$ is a strict contactomorphism).** Let
$\beta=ds-t\,d\varphi$ on $\tilde Y$, the tautological form
$\sigma_sds+\sigma_\varphi d\varphi$ on $\sigma_s=1$. Then $C^*\beta=\alpha$ and
$$C_*X_1=\partial_t,\qquad C_*X_2=\partial_s,\qquad C_*X_3=\partial_\varphi+t\,\partial_s-s\,\partial_t .$$
Consequently $C$ preserves volume ($C^*(\beta\wedge d\beta)=\alpha\wedge d\alpha$,
with $\beta\wedge d\beta=ds\wedge d\varphi\wedge dt$,
$\alpha\wedge d\alpha=-dx_1\wedge dx_2\wedge d\theta$); $g_{\ell_\theta,\zeta}$
pushes forward to the metric with orthogonal frame $\partial_t$,
$\partial_\varphi+t\partial_s-s\partial_t$, $\partial_s$ of lengths $1$,
$\ell_\theta$, $\zeta$; and a curve with $\dot\gamma=aX_1+bX_3+cX_2$ maps to
one with $\dot\varphi=b$, $\dot s-t\dot\varphi=c$, $\dot t+s\dot\varphi=a$. So
Legendrian means $\dot s=t\dot\varphi$, and positively Legendrian means
additionally $\dot t+s\dot\varphi\ge0$.

*Proof.* With $s=x\cdot\omega(\theta)$, $t=x\cdot\omega^\perp(\theta)$, $\varphi=\theta$:
$ds=\omega\cdot dx+t\,d\theta$, so $ds-t\,d\varphi=\omega(\theta)\cdot dx=\alpha$.
Differentiating $(s,\varphi,t)$ along $X_1$, $X_2$, $X_3$ gives
$(0,0,1)$, $(1,0,0)$ and $(t,1,-s)$. ∎

**This is the central link between the prior and the data.** On the sinogram
side the Legendrian condition reads $ds=t\,d\varphi$. Where the sinogram
singularities lie on a curve $s=h(\varphi)$ (Section 5.1), the fibre
coordinate of the singularity is $t=h'(\varphi)$, the **slope** of the
singular curve. A singular curve in the sinogram determines its lift, and
hence the image edge, through its derivative:

**Corollary 4.5 (envelope formula).** If the sinogram singular set is locally
the graph $s=h(\varphi)$, the corresponding image edge is the envelope of
the lines $\{x\cdot\omega(\varphi)=h(\varphi)\}$:
$$z(\varphi)=h(\varphi)\,\omega(\varphi)+h'(\varphi)\,\omega^\perp(\varphi).$$
($h$ is the local support function of the edge.) The graph hypothesis fails
for straight segments, which map to a single point $(s_0,\varphi_0)$ with a
vertical fibre ($C_*X_1=\partial_t$), and at inflection points, where the
sinogram curve has a cusp.

Local analysis of the sinogram determines $s$ and $\varphi$ of a singularity
sharply, but its slope $t=h'$ only coarsely (Section 5.2): estimating $t$
requires a window along the singular curve, and its curvature limits the
window (Proposition 5.2). Following the curve is what the Legendrian prior
does.

**Proposition 4.6 (equivariance).** For a rotation
$\rho_\psi(x,\theta)=(R_\psi x,\theta+\psi)$ and a translation
$\mathcal T_y(x,\theta)=(x+y,\theta)$,
$$C\circ\rho_\psi\circ C^{-1}:(s,\varphi,t)\mapsto(s,\varphi+\psi,t),\qquad
C\circ\mathcal T_y\circ C^{-1}:(s,\varphi,t)\mapsto(s+y\cdot\omega(\varphi),\varphi,t+y\cdot\omega^\perp(\varphi)).$$
Rotations about the origin act on the lifted sinogram as pure shifts in
$\varphi$; translations do not act by shifts.

*Proof.* $R_\psi x\cdot\omega(\theta+\psi)=x\cdot\omega(\theta)$ and
likewise for $\omega^\perp$. Translations are immediate. ∎

*Fan beam.* A fan-beam scan measures $g=\Phi^*Rf$ for a local
diffeomorphism $\Phi$ from fan coordinates to $(s,\varphi)$; wavefront sets
pull back covariantly and $t$ is intrinsic to the ray, so $C$ composes with
$\Phi$ unchanged (Appendix A).

### 4.3 Limited data

**Theorem 4.7 (visible singularities; Quinto [Q2], Frikel–Quinto [FQ]).** If
only angles $\varphi\in\mathcal A\subset[0,\pi)$ are measured, the points of
$\mathrm{WF}(f)$ with $\theta\in\operatorname{int}\mathcal A$ are recovered
stably from $R_{\mathcal A}f$, while those with $\theta\notin\bar{\mathcal A}$
are invisible ($R_{\mathcal A}$ is not microlocally elliptic there).
Reconstruction by FBP adds spurious singularities along lines at the boundary
angles of $\mathcal A$.

So with limited or sparse angular data, part of $\mathrm{WF}(f)$ must be
inferred from the rest. That minimal-energy completion across the invisible
directions recovers it is a design hypothesis, not a theorem (open problem 1).
This is the regime in which the method should outperform local detection most
clearly.

---

## 5. The measurement models

### 5.1 Directional multiscale transforms detect wavefront sets

**Theorem 5.1 (Kutyniok–Labate [KL]; Candès–Donoho [CD]).** Let
$\mathcal{SH}f(a_s,\theta,p)$ be the continuous shearlet (or curvelet)
transform at scale $a_s$, orientation $\theta$ and position $p$. Then
$(p,\theta)\notin\mathrm{WF}(f)$ if and only if
$\mathcal{SH}f(a_s,\theta',p')=O(a_s^N)$ as $a_s\to0$, for all $N$, uniformly in
a neighbourhood of $(p,\theta)$. At a jump discontinuity across a smooth
curve, the coefficients in the normal direction decay only like $a_s^{3/4}$.

Our sinogram analyzer $M$ (`operators/microlocal.py`) is a single-scale
version:

1. $\Lambda^{1/2}$ along $s$. Since $R$ has order $-\tfrac12$
   (Theorem 4.2), a jump of $f$ across an edge of *nonzero curvature* becomes
   a square-root singularity $\propto(h(\varphi)-s)_+^{1/2}$ of $Rf$, with
   factor $\propto$ contrast $\times\sqrt{\text{radius of curvature}}$. The
   half-order ramp returns it to an order-0 singularity,
   $\Lambda^{1/2}x_+^{1/2}=c_1H(x)+c_2\log|x|$. Straight edges (the polygons
   among the phantoms) violate the hypothesis (Corollary 4.5).
2. A bank of anisotropic derivative-of-Gaussian kernels. Both analyzers
   (this one and $W_\theta$ of Section 5.4) use one generic kernel: for a unit
   normal $\mathbf n$ in a 2-D domain with coordinates $u$,
   $$\mathcal G_{\mathbf n}(u)\propto(u\cdot\mathbf n)\,
   e^{-(u\cdot\mathbf n)^2/2\sigma_\perp^2-(u\cdot\mathbf n^\perp)^2/2\sigma_\parallel^2},$$
   which differentiates across a curve with normal $\mathbf n$ (width
   $\sigma_\perp$) and smooths along it (width $\sigma_\parallel$); we write
   $u_\perp=u\cdot\mathbf n$, $u_\parallel=u\cdot\mathbf n^\perp$. Here
   $K_{t_j}=\mathcal G_{\mathbf n(t_j)}$ on $u=(\Delta s,\Delta\varphi)$, with
   $\mathbf n(t)=(1,-t)/\sqrt{1+t^2}$ the conormal of a singular curve of slope
   $t$ (tangent $(t,1)$), and $\sigma_\parallel=6\sigma_\perp$. Kernel $j$ is
   normalized to unit response to a unit step. The two instances differ in
   the domain ($(s,\varphi)$ vs. $x$), in how the orientation is parametrized
   (slope $t$ vs. angle $\theta$, $\mathbf n=\omega(\theta)$), in the aspect
   $\sigma_\parallel/\sigma_\perp$ (6 vs. 2) and in the pre-filter
   ($\Lambda^{1/2}$ vs. FBP).
3. $y_M(s,\varphi,t_j)=|(K_{t_j}*\Lambda^{1/2}y)(s,\varphi)|$, resampled
   onto the lifted grid of $Y$. We write $*$ for "filter with". The code's
   `conv2d` is a correlation $\star$, but the kernels are odd, so
   $|K\star u|=|K*u|$ exactly. The modulus $|\cdot|$ is the pointwise
   magnitude of the signed response. It removes the sign of the jump (dark to
   bright vs. bright to dark), which the wavefront set ignores
   ($\theta\sim\theta+\pi$), and makes the coefficients comparable with the
   amplitude $q\ge0$ of Section 5.3; the cost is that edges on one ray add
   before the modulus is taken (Remark (a) of Section 5.3).

### 5.2 The fibre response

**Proposition 5.2 (fibre response).** If $\Lambda^{1/2}y$ is locally a
straight jump of height $J_Y$ across the line through $p=(s_0,\varphi_0)$ with
conormal $(1,-t_0)$, increasing in the conormal direction, then the signed
coefficient is
$$(K_{t_j}*\Lambda^{1/2}y)(s_0,\varphi_0)=J_Y\,r(t_j,t_0),\qquad r(t,t')=\int K_t(\Delta)\,\mathbf 1\{(1,-t')\cdot\Delta>0\}\,d\Delta .$$
For curved singular curves this holds to leading order, with relative error
$O(\kappa_Y\sigma_\parallel^2/\sigma_\perp)$, where $\kappa_Y$ is the curvature of
the singular curve in $(s,\varphi)$. By Section 5.1, $J_Y$ is proportional to
the image contrast times $\sqrt{\text{radius of curvature}}$; it is not the
amplitude $q$ of Section 7.3.

*Proof.* Linearity, and the kernel has zero mean. The curved case follows by
comparing the curve with its tangent line over the kernel support. ∎

The function $r(\cdot,t_0)$ peaks at $t_0$ (normalization $r(t,t)=1$) but is
**wide**: its width is set by the angular resolution of the kernels,
$\sim1/\text{aspect}$ in slope units. Longer kernels cannot help, because the
singular curves are curved (the error term above). Measured profiles match
$r$ closely (numbers in `experiments/README.md`).

### 5.3 The local sinogram model

Pulled back by $C^{-1}$, each edge point spreads along its own tangent line
(fixed $(s,\varphi)$, varying $t$). A per-voxel classifier of the
coefficients (the learning-free two-component mixture on $y_M$ of
`experiments/milestone1_canonical_relation.py`) therefore sees strong
evidence on every tangent line of every edge: the **tangent-line
ambiguity**. Since the fibre blur is known (Proposition 5.2), it can be put
into the forward model:

**Definition 5.3 (local sinogram model).** Let $q\ge0$ be an amplitude field
on $X'$ (the normalized line density $a/a_{\mathrm{ref}}$, Section 7.3), and
$(\mathcal R_tCq)(s,\varphi,t)=w^{-1}\sum_{t'}|r(t,t')|\,(Cq)(s,\varphi,t')$. The
normalization $w=\sqrt{2\pi}\,\sigma_{\mathrm{cells}}$ is the width (in cells)
of a curve resolved over $\sigma_{\mathrm{cells}}=2$ cells, so that $q=1$ on such
a curve predicts a streak of unit height. We model
$$y_M=\mathcal R_t\,C\,q+e,\qquad e\ \text{i.i.d. Laplace}(0,b_L),$$
with prediction $\hat y_M=\mathcal R_tCq$ and negative log-likelihood
$\mathcal D_Y(q;y_M)=b_L^{-1}\operatorname{mean}\bigl|y_M-\hat y_M\bigr|_\delta$.
The mean over voxels only rescales $b_L$, and
$|e|_\delta=\sqrt{e^2+\delta^2}-\delta$, $\delta=10^{-2}$, is a smoothed
$|e|$ (`loss/data_discrepancy.py:172`); both data terms use it.

Remarks. (a) Taking moduli loses the linearity of $M$: two singularities
on the same ray with different $t$ interfere; the heavy-tailed noise absorbs
this and the curvature error. (b) The model removes the gross ambiguity
(mass spread over a whole tangent line no longer explains the data), but it is
a deconvolution along the fibre with a wide, smooth kernel, so it is badly
ill-conditioned in $t$: smears narrower than the kernel fit as well as the
point. **Within the fibre response, $t$ is left to the prior.** That the
Legendrian envelope (Corollary 4.5) then wins is the design intent, not a
theorem; in experiments the MAP of this model keeps the streaks. (c) A smooth
background produces small fine-scale coefficients (Theorem 5.1) not explained
by $q$.

### 5.4 The full-aperture image-side model

The ambiguity of Section 5.3 comes from analysing the sinogram *locally*,
not from the data. With dense angles, $t$ is fixed once the angles in a
neighbourhood of $\varphi$ are combined, which is what back-projection does.
Let $W_\theta=\mathcal G_{\omega(\theta)}$ be the generic kernel of
Section 5.1 on $u=\Delta x$, i.e. $u_\perp=\Delta x\cdot\omega(\theta)$,
$u_\parallel=\Delta x\cdot\omega^\perp(\theta)$,
with $\sigma_\perp=1.5$ cells across and $\sigma_\parallel=2\sigma_\perp$ along the edge
(aspect 2, `operators/image_analyzer.py`). Then
$W_\theta*R^\#g=R^\#\bigl((RW_\theta)*_sg\bigr)$ with 1-D filters
$(RW_\theta)(\cdot,\varphi)$; for an isotropic $W_\theta$ these are
$\cos(\theta-\varphi)\,G'(s)$. So $W_\theta*\mathrm{FBP}\,y$ is a fixed linear
map of $y$ that combines all angles. Its response to a straight edge of
contrast $J$ at distance $d$ is $J\,r_\theta(\theta,\theta_e)\,e^{-d^2/2\sigma_e^2}$,
where $r_\theta$ is the angular response (wide, since the aspect is small) and
$\sigma_e\approx2.6$ cells is the edge width in $y_W$ (kernel plus FBP blur).

**Definition 5.4 (full-aperture image-side model).** With
$y_W(x,\theta)=|(W_\theta*\mathrm{FBP}\,y)(x)|$ and $q$ as in Definition 5.3,
$$y_W=\hat y_W+e,\qquad \hat y_W=F_X(q)=w_X^{-1}\,G*_x\bigl(|r_\theta|\circledast_\theta q\bigr),\qquad e\ \text{i.i.d. Laplace}(0,b_L),$$
where $G$ is a Gaussian of width $\sqrt{\sigma_e^2-\sigma_{\mathrm{cells}}^2}$ in
$x$ (the amplitude is already resolved over $\sigma_{\mathrm{cells}}=2$ cells),
$\circledast_\theta$ is circular convolution in $\theta$, and $w_X$ is fixed
numerically so that $q=1$ on a straight curve resolved over 2 cells predicts
unit height. The data term is
$\mathcal D_X(q;y_W)=b_L^{-1}\operatorname{mean}\bigl|y_W-\hat y_W\bigr|_\delta$, with
$|\cdot|_\delta$ as in Definition 5.3 (`OrientedEdgeForwardModel` in
`loss/data_discrepancy.py`). This is the default
in training (`experiments/train_lpd.py:77`).

Two edges now interfere only within about $\sigma_e$ of each other, and a
tangent-line streak no longer explains the data. FBP is linear in $y$ and, for
full data, elliptic: it adds no spurious singularities, so "avoiding
reconstruction artifacts" is an argument only for limited data (Theorem 4.7).
The sinogram path is kept as network input (Section 7.2). The price is a full
angular range; for sparse or limited views the angular sampling has to enter
the model (open problem 1).

---

## 6. Discretization

### 6.1 Grids

Fields live on the grid $\{\theta_k=k\pi/K\}\times\{\text{cell centres of }[-1,1]^2\}$
(`geometry/lifted.py`). The same grid is used for $Y$ with $(s,t)$ in place
of $x$, which is possible because $|t|\le1$ inside the unit disk. Parities
and double covers: see the paragraph after the notation table.

### 6.2 Blurred currents

A discrete field has finite resolution. We represent a curve by its current
mollified with a Gaussian $\rho$ of widths $\sigma_x,\sigma_\theta$ (2 cells
each; `data/phantoms.py:lifted_flux`).

**Proposition 6.1 (mollification).** Let $T*\rho$ denote componentwise
convolution in Cartesian coordinates of $\mathbb R^2\times\mathbb R/2\pi\mathbb Z$. Then
$\partial(T*\rho)=(\partial T)*\rho$, $\mathbf M_{\zeta=1}(T*\rho)\le\mathbf M_{\zeta=1}(T)$ and
$\mathbf M(\partial(T*\rho))\le\mathbf M(\partial T)$. Equality holds
approximately in the last inequality when positive and negative endpoints are
separated by several $\sigma_x$.

*Proof.* Convolution commutes with constant-coefficient differential
operators. For $\zeta=1$ the norm $\sqrt{|v_x|^2+\ell_\theta^2v_\theta^2}$ is
independent of $\theta$, so Jensen's inequality applies. ∎

**Proposition 6.2 (horizontality defect).** Mollification does not preserve
the Legendrian condition. Smearing a sample with tangent $X_1(\theta_m)$
into the slice $\theta$ gives components $a\cos(\theta-\theta_m)$ along
$X_1(\theta)$ and $a\sin(\theta-\theta_m)$ along $X_2(\theta)$. Hence:

1. If $c$ is set to zero (a horizontal field), the divergence no longer
   commutes with mollification. A closed curve acquires a spurious boundary
   mass of order $\sigma_\theta/\sigma_x$ per unit length, independent of the
   grid size.
2. With $c$ kept, $\mathbf M_g(T*\rho)\approx(1+\tfrac12(\zeta^2-1)\sigma_\theta^2)\,\mathbf M_{\zeta=1}(T*\rho)$
   to second order in $\sigma_\theta$: a multiplicative bias; the formula gives about 6% at
   $\zeta=2$ and $K=32$ ($\sigma_\theta=2\pi/K$), and the measured bias is
   smaller (`experiments/README.md`).

So the representation must include $c$, and $\zeta<\infty$ must stay
moderate: the Riemannian approximation is forced by finite resolution.

**Proposition 6.3 (the discrete prior is proper and log-concave).** Let
$E_h$ be the discretized energy on the finite-dimensional space of grid
fields $\tau=(a,b,c)$ with $a\ge0$. Then $E_h$ is convex, and
$E_h(\tau)\ge\min(1,\ell_\theta)\,h^2\Delta\theta\,\|\tau\|_{2,1}$, where
$\|\tau\|_{2,1}=\sum_{\text{cells}}|\tau_{\text{cell}}|_2$ (using $\zeta\ge1$),
so $\int\exp(-\lambda E_h)<\infty$. The Gibbs prior is a proper log-concave
density.

*Proof.* The mass term is a sum of norms of linear images of $\tau$, hence
convex and coercive; the boundary term is a sum of absolute values of linear
maps. ∎

### 6.3 Finite differences and resolution

Derivatives are fourth-order central differences with zero extension in
space. Only $b$ is differentiated in $\theta$ ($\operatorname{div}\tau=X_1a+X_3b+X_2c$),
antiperiodically on $[0,\pi)$. On a mollified closed curve, fourth-order
differences with a 2-cell blur leave about 40 times less spurious boundary
mass than second-order differences with a 1-cell blur (`experiments/README.md`).
Network outputs must therefore be resolved over about 2 cells in every
direction, or they pay a spurious cost proportional to their length.

### 6.4 Parameters

Values the text relies on (training defaults unless noted).

| quantity | value | source |
|---|---|---|
| grid | $n=96$, $K=32$ ($48$ in some tests) | `experiments/train_lpd.py:58-59` |
| prior | $\ell_\theta=0.3$ (class default $0.1$), $\nu=0.5$ (class default $1.0$), $\zeta=2$ | `train_lpd.py:66-68`, `loss/regularization.py:71` |
| prior weight $\lambda$ | per run | `train_lpd.py:65`, `experiments/README.md` |
| amplitude | $q=a/a_{\mathrm{ref}}$, no contrast output (`--contrast` restores $q=Jm$) | `models/lpd.py`, `train_lpd.py` |
| data term | $\mathcal D_X$ (`--data-term image`), $b_L=1$ | `train_lpd.py:77,227` |
| sinogram | $3n$ angles on $[0,\pi)$, $2n$ detector cells, noise $0.01\max\vert y\vert$ | `data/synthetic.py:66,79` |
| sinogram analyzer | $\sigma_\perp=1.5\times$ coarser sample spacing, aspect $6$ | `operators/microlocal.py:71` |
| image analyzer | $\sigma_\perp=1.5$ cells, aspect 2, edge width $\sigma_e=2.6$ cells | `train_lpd.py:81-84` |
| resolution of currents, $q$ | 2 cells in $x$ and $\theta$ | `data/phantoms.py:223`, `models/lpd.py:50` |
| LPD | 5 iterations, 24 hidden channels | `train_lpd.py:69-70` |
| production run (slides) | $n=256$, $K=32$, no contrast, $\ell_\theta=0.3$, batch 2, $\lambda=10^{-4}$, 1500 steps: 1.8 s/step, 45 min | `outputs/train/n256_lam1e-4_noJ_xi0.3/` |
| compute benchmark (slides) | $n=256$, $K=64$: 3.2 s/step, 17.4 GB peak, 1.4 h | `research/presentation/assets/scaling.json` |

With $\nu=0.5$ the bridging length $2\nu=1$ is the radius of $B$, so the prior
penalizes essentially every gap and $\nu$ acts as an endpoint penalty rather
than as a gap scale. A full turn costs $2\pi\ell_\theta\approx1.9>2\nu$, so
the sparsity bound $\min(2\nu,2\pi\ell_\theta)$ of Theorem 3.8 is $2\nu=1$.

---

## 7. Inference: MAP estimation and learned primal–dual

### 7.1 The objective

The network outputs the frame components $(a,b,c)$; the amplitude in the
data term is the normalized line density $q=a/a_{\mathrm{ref}}$ (Section 7.3).
With $\mathcal D$ one of $\mathcal D_X$ (default), $\mathcal D_Y$, or a
weighted sum, the **MAP problem** is
$$\min_{\tau}\ \mathcal D\bigl(a/a_{\mathrm{ref}}\bigr)+\lambda E(\tau).\tag{7.1}$$
The predictions $\hat y_M=\mathcal R_tCq$ and $\hat y_W=F_X(q)$ are linear
in $q$, and $|\cdot|_\delta$ is convex, so $\mathcal D$ is convex in $a$;
with Proposition 6.3, **(7.1) is convex** in $\tau$ on $\{a\ge0\}$. (With
a detrended analyzer, $F_X$ contains a modulus and $\mathcal D_X$ is no
longer convex.) The training problem over the network weights is of course
still non-convex.

**Proposition 7.1 (amortized MAP).** Let $F_\Theta$ be a parametric map
from data to $\tau$ and
$\mathcal L(\Theta)=\mathbb E_y\bigl[\mathcal D(a_\Theta/a_{\mathrm{ref}};y)+\lambda E(\tau_\Theta)\bigr]$.
If there is $\Theta^*$ with $F_{\Theta^*}(y)\in\arg\min(7.1)$ for a.e.
$y$, then $\Theta^*$ minimizes $\mathcal L$, and every minimizer of
$\mathcal L$ solves (7.1) for a.e. $y$.

*Proof.* $\mathcal L(\Theta)\ge\mathbb E_y[\min(7.1)]=\mathcal L(\Theta^*)$,
with equality iff the integrand is minimal a.e. ∎

So, under realizability and global minimization of $\mathcal L$, **training
the network on the unsupervised objective is learning a MAP solver** for the
model of Sections 3 and 5. No labels are needed. The labels of the synthetic
phantoms are used only for validation (with full corner fibres, Section 2.3).

### 7.2 Learned primal–dual

Learned primal–dual (LPD) [AÖ] unrolls the primal–dual hybrid gradient
method of Chambolle–Pock [CP1] for $\min_uF(Ku)+G(u)$, replaces its proximal
maps by CNNs acting on $[d,Ku,\text{data}]$ and $[u,K^*d]$, and keeps several
channels of memory. Here $K=C$, the primal variable lives on $X$ and the dual
on $\tilde Y$; by default the primal blocks also receive $y_W$
(`models/lpd.py:72-76`; `--no-image-input` disables it):
$$d_{i+1}=d_i+\Gamma_i\bigl([d_i,\ Cp_i,\ y_M]\bigr),\qquad p_{i+1}=p_i+\Lambda_i\bigl([p_i,\ C^{-1}d_{i+1},\ y_W]\bigr).$$
We use $C^{-1}$ rather than the discrete adjoint. In the continuum they agree
($C$ preserves volume, so its pullback is unitary, Proposition 4.4);
discretely, the transpose of bilinear resampling has moiré-like weights.

### 7.3 Architecture constraints from the geometry

* **Double covers.** The network works on $X$ and $\tilde Y$ (see the
  paragraph after the notation table): on $Y$, $\iota$ includes a point
  reflection of the $(s,t)$-plane, so a kernel would descend only if
  $k(\Delta s,\Delta\varphi,\Delta t)=k(-\Delta s,\Delta\varphi,-\Delta t)$, and
  "twisted" padding would create a seam.
* **Parity of the outputs.** The head outputs $(a,b,c)$ on $S^1$ and
  projects them onto antipodally symmetric currents,
  $a,c\mapsto\tfrac12(a(\theta)+a(\theta+\pi))$ and
  $b\mapsto\tfrac12(b(\theta)-b(\theta+\pi))$ (Lemma 2.6).
* **Positivity.** $a=a_{\mathrm{ref}}\,\mathrm{softplus}(\cdot)\ge0$. But
  $c\ne0$ (by design, Proposition 6.2) and the weights are real, so
  Theorem 3.8 does not apply: small closed loops cost
  $O(\sqrt{\zeta\ell_\theta\ell})$, and only the $\zeta$-weighted normal mass
  limits them (Section 3.3).
* **Amplitude and membership.** The amplitude in the data term is
  $q=a/a_{\mathrm{ref}}$, with $a_{\mathrm{ref}}$ the peak line density of a
  curve blurred over two cells, so that $q=1$ on such a curve predicts an
  edge of unit height (Definitions 5.3, 5.4). The membership
  $m=1-e^{-a/a_{\mathrm{ref}}}$ is a monotone transform of $a$ used for
  display and thresholds only; it does not change ROC AUC. Since the data
  are normalized to a unit 99.9% quantile, $a$ measures edge strength:
  **weak edges get a small $m$**, and $\lambda$ sets the weakest contrast
  that is still detected (open problem 3).

  *Why there is no contrast output.* An earlier version used $q=J\,m(a)$
  with a contrast $J=J_{\max}\,\mathrm{sigmoid}(\cdot)\in(0,J_{\max})$
  (still available as `--contrast`). $(J,m)$ is not identifiable from $q$,
  and the prior charges $a$ but not $J$, so $J$ and $a$ trade scale: a
  smaller $a$ with a larger $J$ fits the data equally well at a lower prior
  cost. With $J_{\max}=1$, $J$ saturated ($J\ge0.95$ on 99% of detected
  voxels); with $J_{\max}=4$ the network used $J$ to escape the prior and
  collapsed (AUC 0.49). The saturating $m$ also made (7.1) non-convex.
  Dropping $J$ removes both problems (results in `experiments/README.md`).
* **Turning on a blurred band.** Each detected edge is a band about two
  cells wide in $x$ and spread in $\theta$. A line of the current inside the
  band whose angle differs from the band's leaves it after a short distance,
  an open end that $\nu\,\mathbf M(\partial T)$ charges. The cheapest way to
  close it is to turn back into the band: $\operatorname{div}\tau\approx0$
  forces $X_3b\approx-X_1a$, and since $X_1a$ changes sign across the ridge,
  so does $b$. Then $b$ is a dipole across each edge on top of the true
  $a\kappa$, and pays for blur rather than curvature. The dipole is weak
  when the band is sharp in $\theta$, which removing the contrast output
  achieved; a moderate turning scale $\ell_\theta=0.3$ helps further at
  $K=64$, while larger $\ell_\theta$ or the elastica shrink the true
  curvature (`experiments/README.md`).
* **Equivariance.** By Proposition 4.6, rotations about the origin by
  multiples of $\pi/K$ act on the dual variables as cyclic shifts in
  $\varphi$, so the dual CNNs are equivariant to them; translations act
  non-trivially on $\tilde Y$, so the dual CNNs are not translation
  equivariant. On the primal side, only rotations by multiples of $\pi/2$ are
  exact grid symmetries, and ordinary convolutions do not respect the
  combined action (rotation plus $\theta$-shift) in general; SE(2) group
  convolutions [Bk] would.

---

## 8. Relation to prior work

* **Sub-Riemannian geometry of $SE(2)$ and perceptual completion.** Petitot
  [Pe] and Citti–Sarti [CS] model V1 as the contact manifold $X$. Minimizers
  of $E$ between two points are $g_\zeta$-geodesics (sub-Riemannian as
  $\zeta\to\infty$); with positivity they may contain in-place rotations, as
  for the cuspless geodesics with keypoints of [DBRS], whose projections
  model association fields. Mumford [Mu] derives
  elastica from a stochastic direction process, and Williams–Jacobs [WJ]
  compute stochastic completion fields on $X$. We use the same geometry for
  the wavefront set itself instead of for image completion.
* **Convex curvature regularization by lifting.** Bredies–Pock–Wirth [BPW]
  and Chambolle–Pock (total roto-translational variation, [CP2]) relax
  curvature-dependent energies of *image level lines* using currents in
  $\Omega\times S^1$. Our energy is of the same type but acts on the wavefront
  current directly, includes a boundary-mass term (endpoints), and is used as
  a prior for detection rather than for reconstruction.
* **Wavefront-set extraction with learning.** Andrade-Loarca, Kutyniok,
  Öktem and Petersen [AKÖP1] extract digital wavefront sets from images with
  shearlets and a CNN (supervised). In [AKÖP2] they use the canonical relation
  of the ray transform for limited-angle reconstruction. These are the closest
  works. The differences here are that detection is *indirect* (from the data
  through fixed linear analyzers: local sinogram filters and, by default, an
  oriented filter bank applied to the FBP; no regularized or learned
  reconstruction), *unsupervised* (amortized MAP of an explicit Bayesian
  model), and uses a Legendrian current prior with a lifted primal–dual
  architecture.
* **Microlocal analysis of tomography.** Quinto [Q1, Q2], Frikel–Quinto [FQ],
  Krishnan–Quinto [KQ] cover visible singularities and artifacts.
* **Learned reconstruction and self-supervision.** LPD [AÖ]; equivariant
  imaging [CTD] and Noise2Inverse [HPB] are alternatives for the unsupervised
  part (Section 9).

---

## 9. Open problems

The current state of the experiments is logged in
[`experiments/README.md`](../experiments/README.md).

1. **Sparse and limited angles.** Both analyzers need dense angles: the
   sinogram analyzer to resolve $t$, the image-side one for the FBP. For
   sparse views, $t$ is not observable from a single projection at all, so
   the data term must be uninformative in $t$ and the prior has to do
   everything. This is the most interesting regime (Theorem 4.7) and needs
   its own analyzer.
2. **Multi-scale data term.** Theorem 5.1 characterizes the wavefront set by
   the decay across scales. A single scale cannot separate a weak edge from
   strong smooth curvature or texture. Ratios of coefficients across scales
   would give a contrast-invariant decay exponent.
3. **Membership versus edge strength.** With $q=a/a_{\mathrm{ref}}$
   (Section 7.3) the prior acts on a contrast-weighted current, a lifted
   TV, so $m$ measures edge strength and weak edges are cheap to drop. If
   $m$ should instead mean "an edge is present" whatever its contrast, the
   data can be normalized locally (for example $y_W$ divided by a local
   maximum), so that every edge has unit height; a hierarchical prior on a
   separate contrast is the principled but non-convex alternative.
4. **Non-integrality (ghosts).** The relaxed prior penalizes total
   multiplicity, not the number of components (Section 3.3), so convex
   relaxations of curve energies produce superpositions of weak curves [BPW].
   Whether the network's MAP estimates are sharp should be monitored.
5. **Self-supervision across angles.** Predicting from an angle subset and
   scoring $\mathcal D$ on the complement tests whether the prior generalizes
   across orientations (Theorem 4.7 makes this meaningful).
6. **Exact SE(2) equivariance** on the primal side with group convolutions.

---

## 10. References

- [A2] V. I. Arnold, *Singularities of Caustics and Wave Fronts*, Kluwer, 1990.
- [AÖ] J. Adler, O. Öktem, Learned primal-dual reconstruction, *IEEE Trans. Med. Imaging* 37 (2018).
- [AKÖP1] H. Andrade-Loarca, G. Kutyniok, O. Öktem, P. Petersen, Extraction of digital wavefront sets using applied harmonic analysis and deep neural networks, *SIAM J. Imaging Sci.* 12 (2019).
- [AKÖP2] H. Andrade-Loarca, G. Kutyniok, O. Öktem, P. Petersen, Deep microlocal reconstruction for limited-angle tomography, *Appl. Comput. Harmon. Anal.* 59 (2022).
- [Bk] E. Bekkers et al., Roto-translation covariant convolutional networks for medical image analysis, *MICCAI* 2018.
- [BPW] K. Bredies, T. Pock, B. Wirth, A convex, lower semicontinuous approximation of Euler's elastica energy, *SIAM J. Math. Anal.* 47 (2015).
- [CD] E. Candès, D. Donoho, Continuous curvelet transform I, II, *Appl. Comput. Harmon. Anal.* 19 (2005).
- [CP1] A. Chambolle, T. Pock, A first-order primal-dual algorithm for convex problems with applications to imaging, *J. Math. Imaging Vis.* 40 (2011).
- [CP2] A. Chambolle, T. Pock, Total roto-translational variation, *Numer. Math.* 142 (2019).
- [CS] G. Citti, A. Sarti, A cortical based model of perceptual completion in the roto-translation space, *J. Math. Imaging Vis.* 24 (2006).
- [CTD] D. Chen, J. Tachella, M. Davies, Equivariant imaging: learning beyond the range space, *ICCV* 2021.
- [DBRS] R. Duits, U. Boscain, F. Rossi, Y. Sachkov, Association fields via cuspless sub-Riemannian geodesics in SE(2), *J. Math. Imaging Vis.* 49 (2014).
- [F] H. Federer, *Geometric Measure Theory*, Springer, 1969.
- [Fe] W. Fenchel, Über Krümmung und Windung geschlossener Raumkurven, *Math. Ann.* 101 (1929).
- [FQ] J. Frikel, E. T. Quinto, Characterization and reduction of artifacts in limited angle tomography, *Inverse Problems* 29 (2013).
- [G] H. Geiges, *An Introduction to Contact Topology*, Cambridge University Press, 2008.
- [GS] V. Guillemin, S. Sternberg, *Geometric Asymptotics*, AMS, 1977.
- [H1] L. Hörmander, *The Analysis of Linear Partial Differential Operators I*, Springer, 1983.
- [H2] L. Hörmander, *The Analysis of Linear Partial Differential Operators IV*, Springer, 1985.
- [HPB] A. Hendriksen, D. Pelt, K. J. Batenburg, Noise2Inverse, *IEEE Trans. Comput. Imaging* 6 (2020).
- [KL] G. Kutyniok, D. Labate, Resolution of the wavefront set using continuous shearlets, *Trans. Amer. Math. Soc.* 361 (2009).
- [KQ] V. P. Krishnan, E. T. Quinto, Microlocal analysis in tomography, in *Handbook of Mathematical Methods in Imaging*, Springer, 2015.
- [LS] M. Lassas, S. Siltanen, Can one use total variation prior for edge-preserving Bayesian inversion?, *Inverse Problems* 20 (2004).
- [M] F. Morgan, *Geometric Measure Theory: A Beginner's Guide*, Academic Press, 1988 (5th ed. 2016).
- [Mu] D. Mumford, Elastica and computer vision, in *Algebraic Geometry and its Applications*, Springer, 1994.
- [Pe] J. Petitot, The neurogeometry of pinwheels as a sub-Riemannian contact structure, *J. Physiol. Paris* 97 (2003).
- [PS] E. Paolini, E. Stepanov, Decomposition of acyclic normal currents in a metric space, *J. Funct. Anal.* 263 (2012).
- [Q1] E. T. Quinto, The dependence of the generalized Radon transform on defining measures, *Trans. Amer. Math. Soc.* 257 (1980).
- [Q2] E. T. Quinto, Singularities of the X-ray transform and limited data tomography in $\mathbb R^2$ and $\mathbb R^3$, *SIAM J. Math. Anal.* 24 (1993).
- [Sm] S. K. Smirnov, Decomposition of solenoidal vector charges into elementary solenoids, and the structure of normal one-dimensional flows, *St. Petersburg Math. J.* 5 (1994).
- [WJ] L. Williams, D. Jacobs, Stochastic completion fields, *Neural Comput.* 9 (1997).
- [W] K. Hämäläinen et al., Tomographic X-ray data of a walnut, arXiv:1502.04064, 2015.

---

## A. Fan-beam geometry

For a flat-detector fan beam with source $R_\beta(0,-R_s)$ and detector
point $R_\beta(u,R_d)$, $R_{sd}=R_s+R_d$, the measurement $g(\beta,u)$ is the
integral of $f$ over the line through these two points.

**Proposition A.1 (fan-beam canonical relation).** Let
$\psi_f=\arctan(u/R_{sd})$ and $\Phi(\beta,u)=(R_s\sin\psi_f,\ \beta-\psi_f)$. Then
$g=\Phi^*Rf$, the ray direction (source to detector) is
$\omega^\perp(\beta-\psi_f)$, and
$$\mathrm{WF}(g)=\Phi^*\mathrm{WF}(Rf)=\bigl\{(\beta,u;\ d\Phi^T\sigma(1,-t))\bigr\},\qquad d\Phi^T(1,-t)=\bigl(-t,\ s_u-t\,\varphi_u\bigr),$$
with $s_u=R_sR_{sd}^2/(u^2+R_{sd}^2)^{3/2}$ and $\varphi_u=-R_{sd}/(u^2+R_{sd}^2)$. The fibre
coordinate $t$ is intrinsic to the ray, so on lifted fields
$(C_{\mathrm{fan}}u)(\beta,u,t)=u\bigl(s\,\omega(\varphi)+t\,\omega^\perp(\varphi),\ \varphi\bmod\pi\bigr)$,
$(s,\varphi)=\Phi(\beta,u)$. Over a full $2\pi$ scan every line is measured
twice.

*Proof.* Direct computation of the line through the source and the detector
point. $\Phi$ is a local diffeomorphism, and wavefront sets pull back by
$(y,\eta)\mapsto(y,d\Phi(y)^T\eta)$ [H1, Thm 8.2.4]. ∎

Implementation: `operators/fan_beam.py` (`tests/test_fan_beam.py`). For the
walnut data [W] the geometry is calibrated against the dataset's system
matrix (`tests/test_walnut.py`).
