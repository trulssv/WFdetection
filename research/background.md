# Wavefront sets as Legendrian currents: background and methods

*Unsupervised indirect wavefront-set detection from tomographic data.*

This document collects the mathematical background of the project and
derives every component of the method from it: the representation of the
wavefront set, the curve prior, the canonical relation used to compare image
and data, the data term, the discretization and the network. Standard results
are stated with references, and results specific to this project are proved.
The reader is assumed to know basic differential geometry (manifolds, vector
fields, differential forms, cotangent bundles) and some distribution theory.
Microlocal analysis, contact geometry and geometric measure theory are
introduced as needed.

**Contents**

1. [Introduction](#1-introduction)
2. [Wavefront sets and contact geometry](#2-wavefront-sets-and-contact-geometry)
3. [Wavefront sets as currents; the prior](#3-wavefront-sets-as-currents-the-prior)
4. [Tomography and canonical relations](#4-tomography-and-canonical-relations)
5. [The measurement model](#5-the-measurement-model)
6. [Discretization](#6-discretization)
7. [Inference: MAP estimation and learned primal–dual](#7-inference-map-estimation-and-learned-primaldual)
8. [Relation to prior work](#8-relation-to-prior-work)
9. [Open problems](#9-open-problems)
10. [References](#10-references)

**Notation.**

| symbol | meaning |
|---|---|
| $\Omega\subset\mathbb R^2$ | image domain; in code $[-1,1]^2$, objects supported in the unit disk $D$ |
| $\omega(\theta)=(\cos\theta,\sin\theta)$, $\omega^\perp(\theta)=(-\sin\theta,\cos\theta)$ | unit vectors |
| $X=\Omega\times S^1$, $X'=\Omega\times\mathbb{RP}^1$ | cosphere bundle and its projectivization |
| $X_1=\omega^\perp(\theta)\cdot\nabla_x$, $X_2=\omega(\theta)\cdot\nabla_x$, $X_3=\partial_\theta$ | moving frame on $X$ |
| $\alpha=\cos\theta\,dx_1+\sin\theta\,dx_2$ | contact form on $X$ |
| $Rf(s,\varphi)=\int f(s\omega(\varphi)+t\omega^\perp(\varphi))\,dt$ | ray transform |
| $Y$ | lifted sinogram space with coordinates $(s,\varphi,t)$ |
| $C:X\to Y$ | canonical relation, $(x,\theta)\mapsto(x\cdot\omega,\theta,x\cdot\omega^\perp)$ |
| $M$ | microlocal analyzer (directional filter bank) on sinograms |
| $T=\tau\,dX$, $\tau=aX_1+cX_2+bX_3$ | 1-current representing a wavefront set |
| $\mathbf M_g$, $\mathbf M(\partial\cdot)$ | mass in the metric $g$, boundary mass |

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
   with piecewise smooth boundaries) the wavefront set is a finite union of
   curves in the 3-manifold $X=\Omega\times S^1$. These curves are not
   arbitrary: the direction is the normal of the curve's projection, so the
   curves are tangent to a fixed plane field. They are *Legendrian* curves of
   the canonical contact structure (Corollary 2.9). A prior on wavefront sets
   should therefore be a prior on Legendrian curves. This turns the vague
   desideratum "edges form smooth curves, and the orientation changes
   continuously along them" into a precise geometric constraint.
2. **It propagates predictably through the measurement.** The ray transform
   is a Fourier integral operator. Its *canonical relation* $C$ maps
   $\mathrm{WF}(f)$ bijectively onto $\mathrm{WF}(Rf)$ (Theorem 4.3), and $C$
   respects the contact structures (Proposition 4.4). So the wavefront set of
   the image can be inferred from the data without reconstructing the image.
   This avoids the artifacts of reconstruction, which are themselves spurious
   singularities.

### 1.2 The method in one paragraph

We represent a candidate wavefront set as a 1-current $T$ on $X$. This is a
vector-field-valued measure: a union of curves is a sum of currents, length
is mass, and endpoints are the boundary $\partial T$. The prior is the Gibbs
measure of the energy $E(T)=\mathbf M_g(T)+\nu\,\mathbf M(\partial T)$: the
length of the curves in a sub-Riemannian-type metric plus $\nu$ times the
number of endpoints. This single convex, local functional has all the desired
properties: independence of components, smoothness, stability, preference for
closed curves, and sparsity (Theorem 3.8). The data are summarized by
coefficients $c=My$ of a fixed directional filter bank on the sinogram. They
are predicted from $T$ by pushing a contrast-weighted version of it through
$C$ and blurring it with the known filter response along the fibre
(Definition 5.4). The MAP estimate
$\arg\min_T D(T;c)+\lambda E(T)$ is computed by an unrolled learned
primal–dual network. It alternates CNN updates on the lifted image $X$ and
on the lifted sinogram $Y$, coupled by $C$, and is trained by minimizing the
same objective over a distribution of data. No labels are used.

### 1.3 Five desiderata for a prior, made precise

The project started from five informal requirements: a prior $P$ on unions of
curves $x=\sqcup_k x_k$ in $X$ should have:

| desideratum | precise version | where |
|---|---|---|
| 1. independence, $P(x)=\prod_kP(x_k)$ | $E$ is additive on currents with disjoint supports | Thm 3.8(i) |
| 2. smoothness | $E$ contains the length in a metric that charges turning | Thm 3.8(ii) |
| 3. stability (breaking a curve is unlikely) | removing a point costs $2\nu$ | Thm 3.8(iii) |
| 4. closed curves are favoured | closed curves have $\partial T=0$ | Thm 3.8(iv) |
| 5. sparsity | every component costs at least $\min(2\nu,2\pi\xi)$ | Thm 3.9 |

Property 3 does **not** follow from a length term, contrary to a first
intuition: removing a point does not change length. It needs the boundary term.

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
subset of $X'$. The code stores fields on $X'$, i.e. $\theta\in[0,\pi)$.

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
are those of the Lie algebra $\mathfrak{se}(2)$; indeed $X\cong SE(2)$ and
$X_1,X_2,X_3$ are left-invariant vector fields. Since $X_1,X_3$ and their
bracket span $TX$, $\mathcal H$ is bracket generating (Hörmander's condition).
By Chow–Rashevskii, any two points of $X$ are joined by a horizontal curve,
and $(X,\mathcal H)$ with a metric on $\mathcal H$ is a sub-Riemannian
manifold. This is the geometry of the primary visual cortex model of
Petitot and Citti–Sarti [Pe, CS].

All three fields are divergence free with respect to the volume
$dX=dx_1\,dx_2\,d\theta$, because their coefficients do not depend on the
variable they differentiate.

**Definition 2.5 (Legendrian curve).** A Lipschitz curve
$\gamma=(z,\theta):I\to X$ is *Legendrian* (horizontal) if
$\dot\gamma(t)\in\mathcal H$ a.e., i.e. $\alpha(\dot\gamma)=\dot z\cdot\omega(\theta)=0$.
Equivalently $\dot\gamma=\lambda X_1+\mu X_3$ for functions $\lambda,\mu$. It is
*positively Legendrian* if $\lambda\ge0$: the projected curve moves in the
direction $J\omega(\theta)=\omega^\perp(\theta)$, i.e. $\omega(\theta)$ is its
normal and the curve is traversed with the normal on its right-hand side.

### 2.3 The wavefront set of a cartoon image

**Theorem 2.6 (conormal singularities; e.g. [H1, Thm 8.1.5], [KQ]).** Let
$D_0\subset\Omega$ be open with $C^\infty$ boundary and $g\in C^\infty(\Omega)$
with $g\neq0$ on $\partial D_0$. Then
$$\mathrm{WF}(g\,\mathbf 1_{D_0})=N^*\partial D_0\setminus0=\{(x,\lambda n(x)):x\in\partial D_0,\ \lambda\neq0\},$$
the conormal bundle of the boundary, where $n$ is the unit normal.

*Sketch.* Away from $\partial D_0$ the function is smooth. Near a boundary
point, a diffeomorphism straightens $\partial D_0$ to $\{x_2=0\}$. The
function becomes $\tilde g\,\mathbf 1_{x_2>0}$, whose localized Fourier
transform decays rapidly in every direction except $\pm e_2$, where it decays
only like $|\xi|^{-1}$. Wavefront sets transform as covectors under
diffeomorphisms, giving the conormal directions. ∎

**Proposition 2.7 (Legendrian lift).** Let $z:I\to\Omega$ be a regular
$C^2$ curve with unit normal $\omega(\theta(t))$, oriented so that
$\dot z=|\dot z|\,\omega^\perp(\theta)$. Its *Legendrian lift*
$\gamma=(z,\theta)$ is a positively Legendrian curve with
$$\dot\gamma=|\dot z|\,(X_1+\kappa X_3),$$
where $\kappa=\dot\theta/|\dot z|$ is the signed curvature. Conversely, a
positively Legendrian curve with $\lambda>0$ is the lift of its projection.
(Legendrian curves with $\lambda$ changing sign project to *fronts* with cusps
[A2]. Positivity excludes them.)

*Proof.* $\alpha(\dot\gamma)=|\dot z|\,\omega^\perp\cdot\omega=0$ and
$\dot z=|\dot z|X_1$. Differentiating the unit normal gives
$\dot\theta=\kappa|\dot z|$. Conversely, if $\dot z=\lambda\omega^\perp(\theta)$ with
$\lambda>0$, then $\omega(\theta)$ is the normal of $z$. ∎

**Proposition 2.8 (corners).** Let $D_0$ be a polygon (or piecewise $C^2$
domain) and $p$ a corner with interior angle $\neq\pi$. Then
$\mathrm{WF}(\mathbf 1_{D_0})$ contains the whole fibre $\{p\}\times(\mathbb
R^2\setminus0)$. In directions outside the two edge normals,
$\widehat{\chi\mathbf 1_{D_0}}$ decays like $|\xi|^{-2}$, one order faster
than along the normals.

*Sketch.* Near $p$, $\mathbf 1_{D_0}$ is the indicator of a sector, a
homogeneous function of degree 0. Its Fourier transform is homogeneous of
degree $-2$ and nonzero in every direction, while the edges contribute degree
$-1$ terms in their normal directions. ∎

Lifted to $X$, the boundary of a polygon with outward normals $\theta_i$ is
a closed Legendrian curve: the edges at constant $\theta_i$ are connected at
the corners by vertical arcs ($\dot z=0$, $\dot\theta\ne0$), which are
horizontal since $X_3\in\mathcal H$. The arc turns by the exterior angle. The
rest of the corner fibre is the weaker singularity of Proposition 2.8.

**Corollary 2.9 (structure of cartoon wavefront sets).** Let
$f=\sum_k J_k\mathbf 1_{D_k}+g$ with $g$ smooth, finitely many domains with
piecewise $C^2$ boundaries, and no cancellation of jumps. Then the image of
$\mathrm{WF}(f)$ in $X$ is
$$\bigcup_\ell\Gamma_\ell\ \cup\ \bigcup_{p\text{ corner}}\{p\}\times S^1,$$
where $\Gamma_\ell$ are compact positively Legendrian curves (closed when the
boundary components are closed), together with their antipodal images
$(x,\theta)\mapsto(x,\theta+\pi)$. The images are again Legendrian, but
negatively oriented (Lemma 6.1).

This is the structural fact behind the whole method: **the prior should live
on (positively) Legendrian curves in $X$.** Two consequences:

* *Crossings separate in $X$.* Two edges crossing in $\Omega$ at a nonzero
  angle have disjoint lifts. Independence of components (desideratum 1) is
  therefore much more natural in $X$ than in $\Omega$. At a T-junction the
  stem's lift simply ends: it has an endpoint.
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
$\mathbf M(T)+\mathbf M(\partial T)<\infty$. A normal 1-current is a vector
measure, $T=\vec\tau\,\|T\|$. When it is absolutely continuous we write
$T=\tau\,dX$ with a vector field $\tau$:
$$T(\eta)=\int_X\eta(\tau)\,dX,\qquad\mathbf M_g(T)=\int_X|\tau|_g\,dX,\qquad\partial T=-(\operatorname{div}\tau)\,dX.$$
$T$ is a *cycle* if $\partial T=0$.

**Definition 3.2 (current of a curve).** An oriented Lipschitz curve
$\gamma:[0,1]\to X$ defines $[\gamma](\eta)=\int_0^1\eta_{\gamma(t)}(\dot\gamma(t))\,dt$.
Then $\mathbf M_g([\gamma])\le L_g(\gamma)$, with equality if $\gamma$ is
injective, and $\partial[\gamma]=\delta_{\gamma(1)}-\delta_{\gamma(0)}$. So
$\mathbf M(\partial[\gamma])=2$ for an arc and $0$ for a closed curve.

**Definition 3.3 (positively Legendrian current).** Write the density in the
frame, $\tau=aX_1+cX_2+bX_3$. $T$ is *Legendrian* if $c=0$ a.e.
($\alpha(\tau)=0$) and *positively Legendrian* if additionally $a\ge0$. By
Proposition 2.7, the currents of lifted edges are positively Legendrian, with
$a$ = line density and $b=a\kappa$.

The following theorem says that normal 1-currents are exactly superpositions
of curves, with no loss of mass. This justifies working with currents
instead of explicit curve lists.

**Theorem 3.4 (Smirnov's decomposition [Sm]; see also [PS]).** Every normal
1-current $T$ in $\mathbb R^d$ can be written as $T=\int[\gamma]\,d\mu(\gamma)$
for a finite positive measure $\mu$ on Lipschitz curves, with
$$\mathbf M(T)=\int\mathbf M([\gamma])\,d\mu(\gamma)\qquad\text{(no cancellation).}$$
If $T$ is *acyclic* (it has no nonzero cycle $Z$ with $\mathbf M(T-Z)+\mathbf M(Z)=\mathbf M(T)$),
the curves can be taken to be arcs with
$\mathbf M(\partial T)=\int\mathbf M(\partial[\gamma])\,d\mu=2\mu(\{\gamma\})$.
A cycle decomposes into closed curves (or "elementary solenoids").

(The theorem is local, so it applies on $X$ through charts and on the torus
$\mathbb R^2\times\mathbb R/2\pi\mathbb Z$.) If $\tau$ is parallel to
$\mathcal H$ and $a\ge0$, so are the tangents of $\mu$-a.e. curve. So every
positively Legendrian normal current is a superposition of positively
Legendrian curves.

### 3.2 The energy and the prior

**Definition 3.5 (energy).** For $\xi>0$, $\zeta\ge1$ let $g=g_{\xi,\zeta}$ be the
left-invariant metric with orthogonal frame $X_1,X_2,X_3$ and
$|X_1|=1$, $|X_2|=\zeta$, $|X_3|=\xi$. For $\nu>0$,
$$E(T)=\mathbf M_g(T)+\nu\,\mathbf M(\partial T)
=\int_X\sqrt{a^2+\zeta^2c^2+\xi^2b^2}\,dX+\nu\int_X\bigl|X_1a+X_2c+X_3b\bigr|\,dX .$$
The second form uses $\operatorname{div}(aX_1+cX_2+bX_3)=X_1a+X_2c+X_3b$,
valid because the $X_i$ are divergence free. On Legendrian currents ($c=0$)
the mass is the sub-Riemannian length with $|X_3|=\xi$: a curve that turns
by an angle $\Delta\theta$ while moving a distance $\ell$ costs at least
$\sqrt{\ell^2+\xi^2\Delta\theta^2}$. $\xi$ is the length scale at which
turning and moving cost the same. The parameter $\zeta$ penalizes
non-horizontal (non-Legendrian) components. In the limit $\zeta\to\infty$
this is the standard Riemannian approximation of the sub-Riemannian metric,
and Section 6.2 explains why a finite $\zeta$ is needed in practice.

*Variant (elastica).* Replacing the mass by
$\int\alpha_e\sqrt{a^2+\zeta^2c^2}+\beta_e\,b^2/a\,dX$ gives, on the lift of a
curve, $\int(\alpha_e+\beta_e\kappa^2)\,ds$: Euler's elastica, the
log-density of Mumford's direction process [Mu]. The term $b^2/a$ is the
perspective of $\kappa^2$, so the functional is still convex and
1-homogeneous. It assigns infinite cost to vertical arcs, i.e. it forbids
corners, so the sub-Riemannian form is the default.

**Proposition 3.6 (energy of curves).** Let $\gamma_1,\dots,\gamma_N$ be
injective Legendrian curves with pairwise disjoint images, $n_k\in\{0,2\}$
endpoints each, and $T=\sum_k[\gamma_k]$. Then
$$E(T)=\sum_{k=1}^N\Bigl(L_\xi(\gamma_k)+\nu\,n_k\Bigr),\qquad L_\xi(\gamma)=\int\sqrt{|\dot z|^2+\xi^2\dot\theta^2}\,dt .$$

*Proof.* Mass and boundary mass are additive for currents with disjoint
supports. $\mathbf M_g([\gamma_k])=L_g(\gamma_k)=L_\xi(\gamma_k)$ since $c=0$, and
$\mathbf M(\partial[\gamma_k])=n_k$ by Definition 3.2. ∎

**Definition 3.7 (prior).** The prior is the Gibbs measure
$P(dT)\propto\exp(-\lambda E(T))$. On currents this is formal, since there is
no Lebesgue measure on an infinite-dimensional space. It is made rigorous in
two equivalent ways: as the MAP functional $-\log P=\lambda E+\mathrm{const}$
(all we use for estimation), and on the discretized current space, where
$\exp(-\lambda E_h)$ is a proper log-concave density (Proposition 6.4).

**Theorem 3.8 (the desiderata).** Let $T,T_1,T_2$ be normal 1-currents.
1. *Independence.* If $\operatorname{supp}T_1\cap\operatorname{supp}T_2=\emptyset$,
   then $E(T_1+T_2)=E(T_1)+E(T_2)$, hence $P(T_1+T_2)\propto P(T_1)P(T_2)$.
2. *Smoothness.* For the lift of a $C^2$ curve,
   $E=\int\sqrt{1+\xi^2\kappa^2}\,ds+\nu n$, increasing in $|\kappa|$. Among
   curves joining two points of $X$, $E$ prefers sub-Riemannian geodesics.
   Their projections are the "association fields" of [CS, DBRS].
3. *Stability.* Let $\gamma$ be an injective Legendrian curve and let
   $\gamma'_\varepsilon,\gamma''_\varepsilon$ be obtained by deleting the
   sub-arc of $L_\xi$-length $\varepsilon$ around an interior point. Then
   $E([\gamma'_\varepsilon]+[\gamma''_\varepsilon])=E([\gamma])+2\nu-\varepsilon$.
4. *Closure.* If $\gamma$ is closed and $\gamma'_\varepsilon$ is obtained by
   deleting a sub-arc of length $\varepsilon$, then
   $E([\gamma'_\varepsilon])=E([\gamma])+2\nu-\varepsilon$.
5. *Sparsity.* See Theorem 3.9.

*Proof.* (i) Both terms are integrals of local densities, and the boundary
of $T_i$ is supported in $\operatorname{supp}T_i$. (ii) Propositions 2.7 and
3.6. (iii)–(iv) Proposition 3.6: the pieces are injective with disjoint
images, and their total length is $L_\xi(\gamma)-\varepsilon$. They have two
more endpoints than $\gamma$. ∎

*Remark.* As $\varepsilon\to0$, $[\gamma'_\varepsilon]+[\gamma''_\varepsilon]\to[\gamma]$
in mass, but the energy jumps by $2\nu$: $E$ is lower semicontinuous (mass
and boundary mass are lsc under weak convergence [F, 4.1.7]) but not
continuous. A measure-zero gap is invisible, as it should be, and any gap of
positive length $\varepsilon<2\nu$ is penalized by $2\nu-\varepsilon>0$. This is
the precise form of "a regular curve is more likely than the two pieces
obtained by breaking it". Gaps longer than $2\nu$ are favoured, so $\nu$ is
the length scale below which the prior bridges gaps.

**Theorem 3.9 (every component costs a fixed minimum).** Let $T$ be a
positively Legendrian normal 1-current and $T=\int[\gamma]\,d\mu$ a
decomposition without cancellation (Theorem 3.4). Then $\mu$-a.e. arc
contributes at least $2\nu$ to $E$, and $\mu$-a.e. closed curve $\gamma$
satisfies
$$L_\xi(\gamma)\ge\xi\int|\dot\theta|\,dt\ \ge\ 2\pi\xi .$$
In particular a configuration of $N$ components has
$E\ge N\min(2\nu,2\pi\xi)$: under the prior, each additional component
costs at least a factor $e^{-\lambda\min(2\nu,2\pi\xi)}$ in probability.

*Proof.* For arcs, $\mathbf M(\partial[\gamma])=2$. Let $\gamma=(z,\theta)$ be
closed, positively Legendrian, $\dot z=\lambda\omega^\perp(\theta)$ with
$\lambda\ge0$. Suppose the image of $\theta$ lies in an open half-circle
$(\theta_0-\pi,\theta_0)$. Then
$\frac{d}{dt}z\cdot\omega(\theta_0)=\lambda\sin(\theta_0-\theta)\ge0$, with
equality only where $\lambda=0$. Since $z$ is closed, $\lambda\equiv0$, so
$\gamma$ is vertical: $z$ is constant and $\theta$ is a closed path inside an
interval. Such a path retraces itself, and its current vanishes. This
contradicts the absence of cancellation unless $\gamma$ is constant. Hence
the image of $\theta$ is not contained in any open half-circle. Either it is
the whole circle, so the total variation is at least $2\pi$, or it is an arc
of length at least $\pi$ that a closed path must traverse in both
directions, giving variation at least $2\pi$. Finally
$\sqrt{|\dot z|^2+\xi^2\dot\theta^2}\ge\xi|\dot\theta|$. ∎

This is Fenchel's theorem [Fe] (total absolute curvature $\ge2\pi$), adapted
to Legendrian curves with corners. **Positivity $a\ge0$ is essential.**
Without it, the back-and-forth curve $z(t)=p+\sin t\,\omega^\perp(\theta_0)$
at fixed $\theta_0$ is a closed Legendrian curve with $\int|\dot\theta|=0$,
and arbitrarily small loops would be free. The network enforces $a\ge0$
(Section 7.3).

*Remark (relation to total variation).* If the current is weighted by the
jump, $T_f=\sum_k|J_k|[\Gamma_k]$, then with $\xi=0$ its mass projects to
$\sum_k|J_k|\,\mathcal H^1(\partial D_k)=\mathrm{TV}(f)$ by the coarea
formula. So $E$ is a lifted, curvature- and endpoint-aware total variation,
related to the total roto-translational variation of Chambolle–Pock [CP2] and
the elastica relaxation of Bredies–Pock–Wirth [BPW]. We use the *unweighted*
current in the prior, since the wavefront set is a geometric object
independent of contrast. Contrast enters only the data term (Section 5.3).

---

## 4. Tomography and canonical relations

### 4.1 The ray transform as a Fourier integral operator

**Definition 4.1.** For $f\in\mathcal E'(D)$, $D$ the unit disk,
$$Rf(s,\varphi)=\int_{\mathbb R}f\bigl(s\,\omega(\varphi)+t\,\omega^\perp(\varphi)\bigr)\,dt,\qquad(s,\varphi)\in\mathbb R\times[0,\pi).$$
It satisfies $Rf(-s,\varphi+\pi)=Rf(s,\varphi)$. Its formal adjoint is
$R^\#g(x)=\int_0^\pi g(x\cdot\omega(\varphi),\varphi)\,d\varphi$.

**Theorem 4.2 (Guillemin–Sternberg, Quinto [GS, Q1]).** $R$ is an elliptic
Fourier integral operator of order $-\tfrac12$ with canonical relation
$$\mathcal C=\Bigl\{\bigl(s,\varphi;\ \sigma,\,-\sigma\,x\cdot\omega^\perp(\varphi)\bigr),\ \bigl(x;\ \sigma\,\omega(\varphi)\bigr)\ :\ s=x\cdot\omega(\varphi),\ \sigma\neq0\Bigr\}.$$
$\mathcal C$ is the graph of a diffeomorphism between conic open sets (the
Bolker condition), and $R^\#R=2\pi|D|^{-1}$ is an elliptic
pseudodifferential operator of order $-1$. In particular
$\|Rf\|_{H^{r+1/2}}\simeq\|f\|_{H^r}$ for $f$ supported in $D$.

*Derivation of $\mathcal C$.* Write $Rf(s,\varphi)=(2\pi)^{-1}\iint e^{i\sigma(s-x\cdot\omega(\varphi))}f(x)\,dx\,d\sigma$.
The phase $\Phi=\sigma(s-x\cdot\omega(\varphi))$ has
$\partial_s\Phi=\sigma$, $\partial_\varphi\Phi=-\sigma x\cdot\omega^\perp(\varphi)$,
$-\partial_x\Phi=\sigma\omega(\varphi)$, and the critical set
$\partial_\sigma\Phi=0$ is $s=x\cdot\omega(\varphi)$. ∎

**Theorem 4.3 (microlocal correspondence).** For $f\in\mathcal E'(D)$,
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

Write the sinogram covector as $\sigma(1,-t)$. The parameter
$t=x\cdot\omega^\perp(\varphi)$ is the position along the ray of the point
$x$ where the ray touches the edge. Projectivizing both sides gives the
**lifted canonical relation**
$$C:X\to Y,\qquad C(x,\theta)=(s,\varphi,t)=\bigl(x\cdot\omega(\theta),\ \theta,\ x\cdot\omega^\perp(\theta)\bigr),$$
where $Y$ (coordinates $(s,\varphi,t)$) is the cosphere bundle of the
sinogram domain, modulo the sign of $\sigma$. For each fixed $\theta$, $C$
is the rotation $x\mapsto R_{-\theta}x$. On fields,
$(Cu)(s,\varphi,t)=u(s\omega(\varphi)+t\omega^\perp(\varphi),\varphi)$
(`operators/canonical_relation.py`).

**Proposition 4.4 ($C$ is a volume-preserving strict contactomorphism).** Let
$\beta=ds-t\,d\varphi$ on $Y$, the restriction of the tautological form
$\sigma_sds+\sigma_\varphi d\varphi$ to $\sigma_s=1$. Then
$$C^*\beta=\alpha\qquad\text{and}\qquad C^*(ds\wedge d\varphi\wedge dt)=dx_1\wedge d\theta\wedge dx_2 .$$
Consequently $C$ maps Legendrian curves to Legendrian curves, positively
Legendrian ones to curves with $\dot s=t\dot\varphi$, and preserves mass and
boundary mass of currents (for the pushed-forward metric).

*Proof.* With $s=x\cdot\omega(\theta)$, $t=x\cdot\omega^\perp(\theta)$, $\varphi=\theta$:
$ds=\omega\cdot dx+(x\cdot\omega^\perp)\,d\theta=\omega\cdot dx+t\,d\theta$,
so $ds-t\,d\varphi=\omega(\theta)\cdot dx=\alpha$. The second identity holds
because $C$ is a rotation in each $\theta$-slice. ∎

**This is the central link between the prior and the data.** On the sinogram
side the Legendrian condition reads $ds=t\,d\varphi$. The sinogram
singularities lie on curves $s=h(\varphi)$, and the fibre coordinate of the
singularity is $t=h'(\varphi)$, the **slope** of the singular curve. A
singular curve in the sinogram determines its lift, and hence the image
edge, through its derivative:

**Corollary 4.5 (envelope formula).** If the sinogram singular set is locally
the graph $s=h(\varphi)$, the corresponding image edge is the envelope of
the lines $\{x\cdot\omega(\varphi)=h(\varphi)\}$:
$$z(\varphi)=h(\varphi)\,\omega(\varphi)+h'(\varphi)\,\omega^\perp(\varphi).$$
($h$ is the local support function of the edge.)

Local analysis of the sinogram determines $s$ and $\varphi$ of a singularity
sharply, but its slope $t=h'$ only coarsely (Section 5.3). Corollary 4.5
explains why: $t$ is a derivative *along* the singular curve, a non-local
quantity. Its accurate recovery requires following the curve, which is what
the Legendrian prior does.

**Proposition 4.6 (equivariance).** For a rotation
$\rho_\gamma(x,\theta)=(R_\gamma x,\theta+\gamma)$ and a translation
$\tau_y(x,\theta)=(x+y,\theta)$,
$$C\circ\rho_\gamma\circ C^{-1}:(s,\varphi,t)\mapsto(s,\varphi+\gamma,t),\qquad
C\circ\tau_y\circ C^{-1}:(s,\varphi,t)\mapsto(s+y\cdot\omega(\varphi),\varphi,t+y\cdot\omega^\perp(\varphi)).$$
Rotations of the image act on the lifted sinogram as pure shifts in $\varphi$.

*Proof.* $R_\gamma x\cdot\omega(\theta+\gamma)=x\cdot\omega(\theta)$ and
likewise for $\omega^\perp$. Translations are immediate. ∎

### 4.3 Fan beam

For a flat-detector fan beam with source $R_\beta(0,-R_s)$ and detector
point $R_\beta(u,R_d)$, $D=R_s+R_d$, the measurement $g(\beta,u)$ is the
integral of $f$ over the line through these two points.

**Proposition 4.7 (fan-beam canonical relation).** Let
$\gamma=\arctan(u/D)$ and $\Phi(\beta,u)=(R_s\sin\gamma,\ \beta-\gamma)$. Then
$g=\Phi^*Rf$, the ray direction (source to detector) is
$\omega^\perp(\beta-\gamma)$, and
$$\mathrm{WF}(g)=\Phi^*\mathrm{WF}(Rf)=\bigl\{(\beta,u;\ D\Phi^T\sigma(1,-t))\bigr\},\qquad D\Phi^T(1,-t)=\bigl(-t,\ s_u-t\,\varphi_u\bigr),$$
with $s_u=R_sD^2/(u^2+D^2)^{3/2}$ and $\varphi_u=-D/(u^2+D^2)$. The fibre
coordinate $t$ is intrinsic to the ray, so on lifted fields
$(C_{\mathrm{fan}}u)(\beta,u,t)=u\bigl(s\,\omega(\varphi)+t\,\omega^\perp(\varphi),\ \varphi\bmod\pi\bigr)$,
$(s,\varphi)=\Phi(\beta,u)$. Over a full $2\pi$ scan every line is measured
twice.

*Proof.* Direct computation of the line through the source and the detector
point. $\Phi$ is a local diffeomorphism, and wavefront sets pull back by
$(y,\eta)\mapsto(y,D\Phi(y)^T\eta)$ [H1, Thm 8.2.4]. ∎

Implementation: `operators/fan_beam.py`, checked against ODL ray geometry,
rebinning, bump placement and the covector direction in simulated fan data
(`tests/test_fan_beam.py`). For the walnut data the geometry was calibrated
against the dataset's system matrix, with correlation 0.9999
(`tests/test_walnut.py`).

### 4.4 Limited data

**Theorem 4.8 (visible singularities; Quinto [Q2], Frikel–Quinto [FQ]).** If
only angles $\varphi\in A\subset[0,\pi)$ are measured, the points of
$\mathrm{WF}(f)$ with $\theta\in\operatorname{int}A$ are recovered stably from
$R_Af$, while those with $\theta\notin\bar A$ are invisible ($R_A$ is not
microlocally elliptic there). Reconstruction by FBP adds spurious
singularities along lines at the boundary angles of $A$.

So with limited or sparse angular data, part of $\mathrm{WF}(f)$ must be
inferred from the rest. The Legendrian prior is what makes this possible:
it propagates orientation along curves (Theorem 3.8(ii)). This is the
regime in which the method should outperform local detection most clearly.

---

## 5. The measurement model

### 5.1 Directional multiscale transforms detect wavefront sets

**Theorem 5.1 (Kutyniok–Labate [KL]; Candès–Donoho [CD]).** Let
$\mathcal{SH}f(a,\vartheta,p)$ be the continuous shearlet (or curvelet)
transform at scale $a$, orientation $\vartheta$ and position $p$. Then
$(p,\vartheta)\notin\mathrm{WF}(f)$ if and only if
$\mathcal{SH}f(a,\vartheta',p')=O(a^N)$ as $a\to0$, for all $N$, uniformly in
a neighbourhood of $(p,\vartheta)$. At a jump discontinuity across a smooth
curve, the coefficients in the normal direction decay only like $a^{3/4}$.

So fine-scale coefficients of a directional transform are the natural
"measurements" of a wavefront set. Our analyzer $M$ (`operators/microlocal.py`)
is a single-scale version, adapted to the sinogram:

1. $\Lambda^{1/2}$ along $s$. Since $R$ has order $-\tfrac12$
   (Theorem 4.2), a jump in $f$ becomes a square-root singularity
   $\propto(h(\varphi)-s)_+^{1/2}$ in $Rf$. The half-order ramp returns it to
   a jump-type (order 0) singularity, so coefficient magnitudes scale like
   the contrast, up to a factor $\sqrt{\text{radius of curvature}}$.
2. A bank of anisotropic derivative-of-Gaussian kernels $K_{t_j}$:
   kernel $j$ differentiates along the conormal $(1,-t_j)$ and smooths along
   the tangent $(t_j,1)$ of a singular curve of slope $t_j$.
3. $c(s,\varphi,t_j)=(K_{t_j}*\Lambda^{1/2}y)(s,\varphi)$, resampled onto the
   lifted grid of $Y$.

**Proposition 5.2 (noise).** $M$ is linear before the modulus. For white
Gaussian sinogram noise of variance $\sigma^2$, the signed coefficients are
Gaussian with variance $\sigma^2\|K_{t_j}\circ\Lambda^{1/2}\|^2$, which is
computable (`coefficient_noise_std`).

### 5.2 The fibre response

**Proposition 5.3 (fibre response).** If $\Lambda^{1/2}y$ is locally a
straight jump of height $J$ across the line through $p=(s_0,\varphi_0)$ with
conormal $(1,-t_0)$, then
$$c(s_0,\varphi_0,t_j)=J\,r(t_j,t_0),\qquad r(t,t')=\int K_t(\Delta)\,\mathbf 1\{(1,-t')\cdot\Delta>0\}\,d\Delta .$$
For curved singular curves this holds to leading order, with relative error
$O(\kappa_Y\sigma_\tau^2/\sigma_n)$, where $\kappa_Y$ is the curvature of the
singular curve in $(s,\varphi)$ and $\sigma_n,\sigma_\tau$ are the kernel
widths.

*Proof.* Linearity and the definition of the kernel. The curved case follows
by comparing the curve with its tangent line over the kernel support. ∎

The function $r(\cdot,t_0)$ peaks at $t_0$ (normalization $r(t,t)=1$) but is
**wide**. Its width is set by the angular resolution of the kernels,
$\sim1/\text{aspect}$ in slope units. Longer kernels cannot help, because the
singular curves are curved (the error term above). Numerically
(`SinogramAnalyzer.fibre_response`), the measured coefficient profiles along
$t$ at true wavefront points match $r(\cdot,t_0)$ with median correlation
$0.993$; the full width at half maximum is about 30 cells on a 96-cell grid.

### 5.3 The data term

The fibre blur has a direct consequence. Pulled back by $C^{-1}$, each edge
point spreads along its own tangent line, because fixed $(s,\varphi)$ and
varying $t$ is a line in $\Omega$. A model that classifies every coefficient
independently ("is this voxel on the wavefront set?") therefore sees strong
evidence on every tangent line of every edge. In milestone 1 the posterior of
such a model exceeded 0.5 on 63% of the image. We call this the
**tangent-line ambiguity**. Proposition 5.3 removes it, because the blur along
the fibre is *known*:

**Definition 5.4 (measurement model).** Let $q\ge0$ be an amplitude field on
$X'$ (contrast times membership, Section 7.3), $v=Cq$, and
$(\mathcal R_tv)(s,\varphi,t)=\sum_{t'}|r(t,t')|\,v(s,\varphi,t')$. We model
$$|c|=\mathcal R_t\,C\,q+\varepsilon,\qquad\varepsilon\ \text{i.i.d. Laplace}(0,\kappa),$$
with negative log-likelihood
$D(q;c)=\kappa^{-1}\sum\bigl||c|-\mathcal R_tCq\bigr|$ (smoothed in the code;
`loss/data_discrepancy.py`).

Remarks. (a) Taking moduli loses the linearity of $M$: two singularities
on the same ray with different $t$ interfere. The heavy-tailed noise model
absorbs this and the curvature error. (b) The model is a deconvolution along
the fibre with a wide, smooth kernel, so it is ill-conditioned. Explaining a
streak with one point and explaining it with a smeared distribution along
the tangent line fit almost equally well. The prior makes the choice:
among all explanations, the Legendrian curve (the envelope, Corollary 4.5)
has the least energy, since tangent-line segments are straight horizontal
curves that pay length and endpoints. (c) A smooth background produces small
coefficients at fine scale (Theorem 5.1) and is not explained by $q$. This is
tested by the smooth backgrounds in the synthetic training data.

---

## 6. Discretization

### 6.1 Grids and symmetry

Fields live on the grid $\{\theta_k=k\pi/K\}\times\{\text{cell centres of }[-1,1]^2\}$
(`geometry/lifted.py`). The same grid is used for $Y$ with $(s,t)$ in place
of $x$, which is possible because $|t|\le1$ inside the unit disk.

**Lemma 6.1 (antipodal symmetry).** Let $A(x,\theta)=(x,\theta+\pi)$. Then
$A_*X_1=-X_1$, $A_*X_2=-X_2$, $A_*X_3=X_3$ and $A^*\alpha=-\alpha$. The
wavefront current of a real image is invariant under $A$ composed with
orientation reversal, which acts on densities as
$$(a,c,b)(\theta+\pi)=(a,c,-b)(\theta).$$
So on $X'=\Omega\times\mathbb{RP}^1$, $a$ and $c$ are functions while $b$ is
a section of the Möbius line bundle: it is $\pi$-*antiperiodic*.

*Proof.* $X_{1,2}(\theta+\pi)=-X_{1,2}(\theta)$ in Cartesian components, and
$\omega(\theta+\pi)=-\omega(\theta)$. Pushing $aX_1+cX_2+bX_3$ forward by
$A$ gives $-aX_1-cX_2+bX_3$ at $\theta+\pi$, and reversing the orientation
(to restore $a\ge0$) gives $aX_1+cX_2-bX_3$. ∎

### 6.2 Blurred currents

A discrete field has finite resolution. We represent a curve by its current
mollified with a Gaussian $\rho$ (2 cells in $x$ and $\theta$;
`data/phantoms.py:lifted_flux`).

**Proposition 6.2 (mollification).** Let $T*\rho$ denote componentwise
convolution in Cartesian coordinates of $\mathbb R^2\times\mathbb R/2\pi\mathbb Z$. Then
$\partial(T*\rho)=(\partial T)*\rho$, $\mathbf M_{\zeta=1}(T*\rho)\le\mathbf M_{\zeta=1}(T)$ and
$\mathbf M(\partial(T*\rho))\le\mathbf M(\partial T)$. Equality holds in the last
inequality when the endpoints are separated by more than the support of $\rho$.

*Proof.* Convolution commutes with constant-coefficient differential
operators. For $\zeta=1$ the norm $\sqrt{|v_x|^2+\xi^2v_\theta^2}$ is
independent of $\theta$, so Jensen's inequality applies. ∎

**Proposition 6.3 (horizontality defect).** Mollification does not preserve
the Legendrian condition. Smearing a sample with tangent $X_1(\theta_m)$
into the slice $\theta$ gives components $a\cos(\theta-\theta_m)$ along
$X_1(\theta)$ and $a\sin(\theta-\theta_m)$ along $X_2(\theta)$. Hence:

1. If $c$ is set to zero (a horizontal field), the divergence no longer
   commutes with mollification. A closed curve acquires a spurious boundary
   mass of order $\sigma_\theta/\sigma_x$ per unit length, independent of the
   grid size. Measured: 15.1 instead of 1.2 on the milestone-1 phantom.
2. With $c$ kept, $\mathbf M_g(T*\rho)\approx(1+\tfrac12(\zeta^2-1)\sigma_\theta^2)\,\mathbf M_{\zeta=1}(T*\rho)$
   to second order in $\sigma_\theta$: a small multiplicative bias, about 3%
   at $\zeta=2$.

So the representation must include the normal component $c$, and the
horizontality weight $\zeta$ must stay moderate. The Riemannian approximation
$\zeta<\infty$ is not just a numerical convenience; it is forced by finite
resolution.

**Proposition 6.4 (the discrete prior is proper and log-concave).** Let
$E_h$ be the discretized energy on the finite-dimensional space of grid
fields $(a,b,c)$ with $a\ge0$. Then $E_h$ is convex, and
$E_h(\tau)\ge\min(1,\zeta,\xi)\,h^2\Delta\theta\,\|\tau\|_{\ell^2\text{-sum}}$, so
$\int\exp(-\lambda E_h)<\infty$. The Gibbs prior is a proper log-concave
density.

*Proof.* The mass term is a sum of norms of linear images of $\tau$, hence
convex and coercive; the boundary term is a sum of absolute values of linear
maps. ∎

### 6.3 Finite differences and resolution

Derivatives are fourth-order central differences: zero extension in space,
periodic in $\theta$ for $a,c$, antiperiodic for $b$. The boundary mass of a
mollified closed curve (ideal: 0) is 10.6 with 2nd-order differences and a
1-cell blur, and 0.26 with 4th-order differences and a 2-cell blur
(`tests/test_regularization.py`). Network outputs must therefore be resolved
over about 2 cells in every direction, or they pay a spurious cost
proportional to their length.

---

## 7. Inference: MAP estimation and learned primal–dual

### 7.1 The posterior

Combining Definition 3.7 and Definition 5.4, the posterior over currents
and amplitudes given data $y$ is
$$p(T,q\mid y)\ \propto\ \exp\bigl(-D(q;My)-\lambda E(T)\bigr)\,\mathbf 1\{q\ \text{supported on}\ T\},$$
and the **MAP problem** is
$$\min_{T,\,q}\ D(q;My)+\lambda E(T).\tag{7.1}$$
For fixed coupling the problem is convex: $D$ is convex in $q$ and $E$
convex in $T$. The coupling (Section 7.3) is not.

**Proposition 7.2 (amortized MAP).** Let $F_\vartheta$ be a parametric map
from data to $(T,q)$ and
$\mathcal L(\vartheta)=\mathbb E_y\bigl[D(q_\vartheta(y);My)+\lambda E(T_\vartheta(y))\bigr]$.
If there is $\vartheta^*$ with $F_{\vartheta^*}(y)\in\arg\min(7.1)$ for a.e.
$y$, then $\vartheta^*$ minimizes $\mathcal L$, and every minimizer of
$\mathcal L$ solves (7.1) for a.e. $y$.

*Proof.* $\mathcal L(\vartheta)\ge\mathbb E_y[\min(7.1)]=\mathcal L(\vartheta^*)$,
with equality iff the integrand is minimal a.e. ∎

So **training the network on the unsupervised objective is the same as
learning a MAP solver** for the Bayesian model of Sections 3 and 5. No
labels are needed. The labels of the synthetic phantoms are used only for
validation.

### 7.2 Learned primal–dual

For $\min_uF(Ku)+G(u)$ the primal–dual hybrid gradient method of
Chambolle–Pock [CP1] iterates
$d\leftarrow\mathrm{prox}_{\sigma F^*}(d+\sigma K\bar u)$,
$u\leftarrow\mathrm{prox}_{\tau G}(u-\tau K^*d)$. Learned primal–dual (LPD)
[AÖ] unrolls a fixed number of iterations, replaces the proximal maps by
CNNs acting on $[d,Ku,\text{data}]$ and $[u,K^*d]$, and keeps several
channels of memory. Here $K=C$ (followed in the data term by the fixed
$\mathcal R_t$). The primal variable lives on the lifted image and the dual
on the lifted sinogram:
$$d_{i+1}=d_i+\Gamma_i\bigl([d_i,\ Cp_i,\ c]\bigr),\qquad p_{i+1}=p_i+\Lambda_i\bigl([p_i,\ C^{-1}d_{i+1}]\bigr).$$
We use $C^{-1}$ rather than the discrete adjoint. In the continuum they agree
($C$ is unitary, Proposition 4.4); discretely, the transpose of bilinear
resampling has moiré-like weights.

### 7.3 Architecture constraints from the geometry

* **Double covers.** The network works on $X=\Omega\times S^1$ and on the
  space of *oriented* lines ($\varphi\in[0,2\pi)$), both discretized with
  $2K$ slices and plain periodic convolutions. On the quotient
  $\varphi\in[0,\pi)$ the identification $(s,\varphi+\pi,t)\sim(-s,\varphi,-t)$
  includes a point reflection of the $(s,t)$-plane. A convolution kernel $k$
  descends to the quotient only if $k(\Delta s,\Delta\varphi,\Delta t)=k(-\Delta s,\Delta\varphi,-\Delta t)$,
  so "twisted" padding would create a seam. On the double cover no constraint
  arises. The input coefficients are extended by $c(s,\varphi+\pi,t)=c(-s,\varphi,-t)$.
* **Parity of the outputs.** The head outputs $(a,b,c,J)$ on $S^1$ and
  projects them onto antipodally symmetric currents,
  $a,c\mapsto\tfrac12(a(\theta)+a(\theta+\pi))$ and
  $b\mapsto\tfrac12(b(\theta)-b(\theta+\pi))$ (Lemma 6.1).
* **Positivity.** $a=a_{\mathrm{ref}}\,\mathrm{softplus}(\cdot)\ge0$, so the
  output is positively Legendrian up to its normal component, and
  Theorem 3.9 applies.
* **Membership and amplitude.** The classifier output is
  $m=1-e^{-a/a_{\mathrm{ref}}}$, with $a_{\mathrm{ref}}$ the peak line density
  of a curve blurred over two cells. The amplitude in the data term is
  $q=Jm$ with contrast $J\in(0,J_{\max}]$. The bound is necessary: with
  unbounded $J$ the network could send $a\to0$, $J\to\infty$ and make the
  prior free, since $E$ is 1-homogeneous in $a$.
* **Equivariance.** By Proposition 4.6, image rotations by multiples of
  $\pi/K$ act on the dual variables as cyclic shifts in $\varphi$, so the dual
  CNNs are exactly rotation equivariant. The primal CNNs are equivariant only
  under the combined action (spatial rotation plus $\theta$-shift), which
  ordinary convolutions respect approximately. SE(2) group convolutions
  [Bk] would make it exact.

---

## 8. Relation to prior work

* **Sub-Riemannian geometry of $SE(2)$ and perceptual completion.** Petitot
  [Pe] and Citti–Sarti [CS] model V1 as the contact manifold $X$. Duits,
  Boscain, Rossi and Sachkov [DBRS] study cuspless sub-Riemannian geodesics
  (positively Legendrian curves; cf. Theorem 3.9). Mumford [Mu] derives
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
  works. The differences here are that detection is *indirect* (from data, no
  image), *unsupervised* (amortized MAP of an explicit Bayesian model), and
  uses a Legendrian current prior with a lifted primal–dual architecture.
* **Microlocal analysis of tomography.** Quinto [Q1, Q2], Frikel–Quinto [FQ],
  Krishnan–Quinto [KQ] cover visible singularities and artifacts.
* **Learned reconstruction and self-supervision.** LPD [AÖ]; equivariant
  imaging [CTD] and Noise2Inverse [HPB] are alternatives for the unsupervised
  part (Section 9).

---

## 9. Open problems

1. **Sparse and limited angles.** The current analyzer needs dense angles.
   For sparse views, $t$ is not observable from a single projection at all,
   so the data term must be uninformative in $t$ and the prior has to do
   everything. This is the most interesting regime (Theorem 4.8) and needs
   its own analyzer.
2. **Multi-scale data term.** Theorem 5.1 characterizes the wavefront set by
   the decay across scales. A single scale cannot separate a weak edge from
   strong smooth curvature or texture. Ratios of coefficients across scales
   would give a contrast-invariant decay exponent.
3. **Coupling of membership and contrast.** The bounded-contrast
   parametrization $q=Jm$ (Section 7.3) makes the prior cost of weak edges
   low. A principled alternative is a hierarchical prior on $J$.
4. **Non-convexity and ghosts.** Convex relaxations of curve energies
   produce superpositions of curves [BPW]. Whether the network's MAP
   estimates are sharp should be monitored.
5. **Self-supervision across angles.** Predicting from an angle subset and
   scoring $D$ on the complement tests whether the prior generalizes across
   orientations (Theorem 4.8 makes this meaningful).
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
- [M] F. Morgan, *Geometric Measure Theory: A Beginner's Guide*, Academic Press.
- [Mu] D. Mumford, Elastica and computer vision, in *Algebraic Geometry and its Applications*, Springer, 1994.
- [Pe] J. Petitot, The neurogeometry of pinwheels as a sub-Riemannian contact structure, *J. Physiol. Paris* 97 (2003).
- [PS] E. Paolini, E. Stepanov, Decomposition of acyclic normal currents in a metric space, *J. Funct. Anal.* 263 (2012).
- [Q1] E. T. Quinto, The dependence of the generalized Radon transform on defining measures, *Trans. Amer. Math. Soc.* 257 (1980).
- [Q2] E. T. Quinto, Singularities of the X-ray transform and limited data tomography in $\mathbb R^2$ and $\mathbb R^3$, *SIAM J. Math. Anal.* 24 (1993).
- [Sm] S. K. Smirnov, Decomposition of solenoidal vector charges into elementary solenoids, and the structure of normal one-dimensional flows, *St. Petersburg Math. J.* 5 (1994).
- [WJ] L. Williams, D. Jacobs, Stochastic completion fields, *Neural Comput.* 9 (1997).
- [W] K. Hämäläinen et al., Tomographic X-ray data of a walnut, arXiv:1502.04064, 2015.
