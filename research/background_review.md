# Review of `research/background.md`

Scope: mathematical correctness, redundancy, clarity, and consistency with the code.
Line numbers refer to `background.md` as of 2026-10-06 (942 lines).

The review was cross-checked by:
- an independent second mathematical pass;
- a code-consistency pass (all cited tests pass: 19 passed);
- symbolic and numerical checks of the key computations.

Items are tagged:
- **[C]** critical: a wrong statement that the method's reasoning depends on;
- **[M]** major: a wrong or misleading statement, or a doc/code mismatch;
- **[m]** minor.

Each item gives **Where / Problem / Fix**. Suggested replacement text is indicative, not mandatory.

---

## 0. Verdict and global decisions

The geometric core is sound and well chosen. The following were verified and need no change:
- the contact structure on X;
- the Legendrian lift;
- the canonical relation of R;
- C*β = α;
- the frame and bracket computations;
- the fan-beam Jacobian;
- the antipodal parity.

The problems concentrate in three places:

1. **Section 3 (currents and prior).** The headline properties (sparsity, minimum cost per component) are proved for exact, integral, multiplicity-one Legendrian currents. They are then asserted for the network output, which is:
   - non-integral (diffuse, real-weighted);
   - not Legendrian (c ≠ 0, by design, Prop 6.3);
   - effectively contrast-weighted (Section 7.3).

   None of the three properties transfers. This is the most important correction.
2. **The narrative is out of date with the method.** Training uses the image-side, full-aperture data term (Remark 5.5) by default (`experiments/train_lpd.py:77`, `--data-term image`). But §1.1, §1.2, §7.1, §7.2 and §8 present the sinogram-local model (Def 5.4) as *the* method, and sell "no reconstruction" as a feature. The current model appears only in a Remark and in a §9 "Update".
3. **Notation is heavily overloaded** (see §4 below). This is the main clarity problem.

Recommended global restructuring, to be done first:

- **G1.** Present two data terms explicitly in Section 5:
  - **5.3 "Local sinogram model"** (Def 5.4) and its failure mode (the tangent-line ambiguity, ill-conditioning);
  - **5.4 "Full-aperture image-side model"**, promoted from Remark 5.5 to a Definition.

  Then update §1.2, §7.1, §7.2 (the LPD primal block also receives `b`: `[p, C⁻¹d, b]`, `models/lpd.py:72-76`) and §8 ("indirect, no image") accordingly. State honestly:
  - the image-side term uses FBP, i.e. a linear, elliptic (for full data) map of y;
  - with full data, FBP introduces no spurious singularities, so the "avoid reconstruction artifacts" argument (l.77–79) only applies to limited data;
  - the sinogram path is kept as network input.
- **G2.** Move all experiment logs and empirical numbers out of the theory into either a single "Numerical checks" subsection or `experiments/README.md`. This covers §9 "Status"/"Update" and the numbers in §5.2, Remark 5.5, Prop 6.3 and §6.3. Keep at most one sentence per claim in the theory, with a pointer.
- **G3.** Introduce a notation pass (§4 of this review) before any other edit, so later edits use the new symbols.
- **G4.** Move Lemma 6.1 (antipodal symmetry) into Section 2, right after Def 2.5. It is used in Cor 2.9 (a forward reference) and is geometry, not discretization.

---

## 1. Correctness

### Section 1 (Introduction)

**1.1 [M] Table §1.3 row 3 and l.113–114: "removing a point costs 2ν" is false.**
- Problem: Removing a single point changes neither the current nor its boundary: [γ′]+[γ″] = [γ], and the two new boundary atoms cancel. The Remark at l.370–374 itself says a measure-zero gap is invisible. "It needs the boundary term" is also wrong as an argument, since removing a point does not change ∂T either.
- Fix:
  - Row 3 becomes: "opening a gap of length ε costs 2ν − ε (Thm 3.8(iii))".
  - Replace l.113–114 with: "A length term alone *rewards* gaps (it saves ε). Only the boundary term makes a short gap cost something; gaps shorter than 2ν are penalized."

**1.2 [M] Table §1.3 row 5 and §1.2 l.86–90 ("sparsity", "plus ν times the number of endpoints", "has all the desired properties").**
- Problem: These hold only for integral multiplicity-one currents. See 1.10–1.12.
- Fix: Qualify both places: "for curves (integral currents of multiplicity one)". Add one sentence saying that, for the convex relaxation the network uses, sparsity is in total multiplicity (an ℓ¹/TV-type penalty), not in number of components.

**1.3 [m] l.73–77.**
- Problem: "C respects the contact structures … So the wavefront set can be inferred …". The inference rests on Thm 4.3 (bijectivity), not on contact preservation.
- Fix: Reorder so that the "so" follows Thm 4.3. Mention contact preservation as the reason the *prior* transfers to the data side.

### Section 2 (wavefront sets, contact geometry)

**1.4 [C] Prop 2.8 / Cor 2.9 (l.215, l.235): "piecewise C²" is not enough.**
- Problem: At a point where the boundary is C² but not C^∞ (e.g. a curvature jump with interior angle π), the localized Fourier transform decays only polynomially in non-normal directions, so the whole fibre is in WF. Cor 2.9 as stated is therefore wrong for piecewise-C² domains.
- Fix: Require piecewise C^∞ boundaries. Alternatively, define "corner" as any non-C^∞ boundary point and say so.

**1.5 [M] Cor 2.9: corner and junction set, and "no cancellation of jumps" is undefined.**
- Problem 1 (crossings): For a crossing f = 1_{x1>0} + 1_{x2>0}, WF at the origin contains only ±e₁, ±e₂, with no full fibre. Yet writing f as a sum of quadrant indicators makes Cor 2.9 predict a full fibre.
- Problem 2 (T-junctions): A T-junction point *does* carry a full fibre. So the bullet "At a T-junction the stem's lift simply ends: it has an endpoint" (l.248–249) is false at the level of sets.
- Background: Near a vertex p, f is homogeneous of degree 0 with jumps J_j across rays ψ_j. The degree −2 term of the localized Fourier transform is ∝ |ξ|⁻² Σ_j J_j tan(ψ_j − φ). So the full fibre is in WF iff the even part J(ψ) + J(ψ+π) of the angular jump function is not identically zero.
- Fix:
  - (a) State Cor 2.9 as an *inclusion* WF(f) ⊂ ⋃Γ_ℓ ∪ ⋃_{p∈V}{p}×S¹, where V is the set of non-C^∞ boundary points and junctions. Add that equality holds on the smooth parts of edges with nonzero jump, and that at p ∈ V the full fibre is present iff the even part of the angular jump function is nonzero (true for corners and T-junctions, false for transversal crossings).
  - (b) Rewrite the T-junction bullet. The *set* WF contains the full fibre at the junction. The *model* (the closed Legendrian lift with turning arcs, `phantoms.py corners='arc'`) represents only the turning arcs and drops the rest of the fibre. This is a modeling choice; state it once, near l.227–231.
  - (c) The labels use full fibres at corners (`data/synthetic.py:85-89`) while the current uses arcs (`data/phantoms.py:133-140`). Note that validation AUC therefore rewards corner directions the prior does not represent.
  - (d) The phantom generator produces X-crossings but no deliberate T-junctions (`phantoms.py:159`). The T-junction claim is untested; say so, or drop it.

**1.6 [M] Cor 2.9 l.239–241: "The images are again Legendrian, but negatively oriented (Lemma 6.1)".**
- Problem: Orientation is not a property of a set. Lemma 6.1 actually shows that the antipodal image, *with reversed orientation*, is positively Legendrian. It is the inward-normal lift traversed clockwise.
- Fix: "… together with their antipodal images, which are positively Legendrian when given the reversed orientation (Lemma 2.x, formerly 6.1)."

**1.7 [m] Def 2.4 l.168–171.**
- Problem: "(X, H) with a metric on H is a sub-Riemannian manifold" is fine. But "X ≅ SE(2)" only holds for Ω = ℝ².
- Fix: Write "X is an open subset of SE(2)".

**1.8 [m] Thm 2.6 citation [H1, Thm 8.1.5].**
- Problem: That theorem concerns densities on submanifolds (conormal WF of δ_{∂D}), not indicators.
- Fix: Cite it for ∇1_{D₀} = −n δ_{∂D₀} and add the one-line deduction, or cite [KQ] alone.

### Section 3 (currents, prior)

**1.9 [M] Def 3.1–3.3: Legendrian and positively Legendrian are defined only for absolutely continuous T = τ dX.**
- Problem: Curve currents [γ] are singular with respect to dX. So "a = line density" (l.285) is not a function, and Thm 3.9 (stated for "positively Legendrian normal 1-currents") applies to an empty definition in the case of interest.
- Fix:
  - Use the polar decomposition T = τ⃗ ‖T‖ (‖T‖ a measure, |τ⃗| = 1).
  - *Legendrian*: α(τ⃗) = 0 ‖T‖-a.e.
  - *Positively Legendrian*: additionally ⟨dx-part of τ⃗, ω^⊥(θ)⟩ ≥ 0 ‖T‖-a.e. (i.e. the X₁-coefficient a ≥ 0).
  - Write the energy as E(T) = ∫|τ⃗|_g d‖T‖ + ν‖∂T‖(X), and give the τ dX formula as the special case used on grids.

**1.10 [M] Thm 3.4 (Smirnov) is misstated.**
- Problems:
  - (a) A normal *cycle* decomposes into elementary solenoids, which in general are not closed finite-length curves (e.g. an irrational winding on a torus, which X = Ω × S¹ effectively contains). "Every normal 1-current T = ∫[γ]dμ over Lipschitz curves" is false.
  - (b) The useful form is T = A + Z, with A acyclic, Z a cycle, and M(T) = M(A) + M(Z). Then A = ∫[γ]dμ over simple arcs, with ‖T‖ = ∫‖[γ]‖dμ *as measures* and not only in total mass. The measure identity is what gives "tangents of μ-a.e. curve are tangents of T" (l.302–305).
  - (c) "2μ({γ})" (l.298) should be 2μ(Γ) = 2‖μ‖.
  - (d) "The theorem is local, so it applies on X through charts" (l.301) is invalid, because the decomposition is global.
- Fix:
  - Restate the theorem as in (b).
  - Justify its use on X by embedding X ↪ ℝ⁴ via (x, cos θ, sin θ) (curves stay in supp T), or cite [PS] (metric spaces).
  - **Simplest option:** Thm 3.9 only needs the case of integral currents. For an integral 1-cycle, the decomposition into closed Lipschitz curves is classical ([F, 4.2.25]). Use that and drop most of the Smirnov discussion (see 2.3).

**1.11 [C] Thm 3.9 does not apply to the network output; §7.3 l.814–816 claims it does.**
- Problem: The proof uses ż = λω^⊥(θ) with λ ≥ 0. With a normal component, ż = aω^⊥ + cω, and the derivative d/dt z·ω(θ₀) gets the extra term c cos(θ₀ − θ), which has no sign.
- Counterexample (checked numerically): a thin closed triangle loop with a ≥ 0 on every leg, c ≠ 0, θ ∈ {π/2, π/2 ± δ}, leg lengths ~ε/δ and δ ~ √ε. Its energy is ~0.06 at ε = 10⁻⁴ (ξ=1, ζ=2), against the claimed bound 2πξ ≈ 6.28. The second pass found a smooth counterexample with energy going to 0 linearly. Small loops are therefore nearly free once c ≠ 0, and c ≠ 0 is mandatory by Prop 6.3.
- Fix:
  - State Thm 3.9 for Legendrian currents (c = 0) only.
  - In §7.3 replace "and Theorem 3.9 applies" with: "Theorem 3.9 does not apply, since c ≠ 0. Small closed loops cost O(√(ζ ξ ℓ)) for loop size ℓ. The minimum cost per component is lost at the resolution scale, and only the ζ‖c‖ penalty limits such loops."
  - Optionally add the counterexample as a remark.

**1.12 [C] Thm 3.9's "N components cost ≥ N min(2ν, 2πξ)" and the Gibbs factor e^{−λ min(…)} per component.**
- Problem: E is 1-homogeneous. A component of weight w costs w·(L_ξ + νn). For real-valued (diffuse) currents, i.e. what a convex relaxation and the network produce, the bound reads E ≥ min(2ν, 2πξ)·‖μ‖, which is a bound on *total multiplicity* and not on the number of components. Weak spurious components are cheap. This is also the mechanism behind open problem 4 (superpositions).
- Fix: State the theorem for integral currents with multiplicity one, and add a remark covering:
  - the mass-proportional version for real weights;
  - the consequence that the relaxed prior is an ℓ¹/TV-type sparsity prior, not a cardinality prior;
  - a link to open problem 4, retitled "Non-integrality (ghosts)", since the issue there is relaxation, not non-convexity.

**1.13 [M] Thm 3.9 l.405–408: the "positivity is essential" example is invalid.**
- Problem: z(t) = p + sin t·ω^⊥(θ₀) at fixed θ₀ retraces itself. Its current is 0, so it cannot appear in a decomposition without cancellation.
- Fix: Use a non-retracing closed front with cusps, for example the thin triangle with legs along ω^⊥(θ₀), −ω^⊥(θ₀±δ) (λ changes sign), with lengths ε, ε/(2cos δ), ε/(2cos δ). Its total turning is 4δ and its length ~2ε, so its energy goes to 0.

**1.14 [m] Thm 3.9 proof l.394–397.**
- Problem: "its current vanishes. This contradicts the absence of cancellation" needs the fact that decomposition curves satisfy M([γ]) = L(γ) (simple curves). Include that in the restated decomposition theorem (1.10b).
- Fix: Add that property to the restated theorem.

**1.15 [M] Remark l.411–418 (TV) and §7.3 contradict each other.**
- Problem 1 (the TV identity): Σ|J_k| H¹(∂D_k) = TV(f) holds only when the boundaries are pairwise disjoint (up to H¹-null sets) and g = 0. Shared edges have jump |J_i − J_j|. In general TV(f) = ∫|∇g| + ∫_{S_f}|[f]| dH¹. If the antipodal copy is included, the mass is 2·TV. The identity is the BV structure theorem, not the coarea formula.
- Problem 2 (the contradiction): The remark says "we use the unweighted current in the prior". But §7.3 shows that at the optimum J = J_max and m = q/J_max, so a = −a_ref log(1 − q/J_max) ≈ a_ref q/J_max. The prior therefore acts on an approximately **contrast-weighted** current, which is the lifted TV, not the unweighted one.
- Fix:
  - Correct the identity's hypotheses.
  - Rewrite the last two sentences: "Ideally the prior acts on the unweighted current. With the parametrization of §7.3 it effectively acts on a contrast-weighted one, i.e. a lifted TV. See open problem 3."

**1.16 [m] Def 3.5 l.320–322: the wording is inverted.**
- Problem: Finite ζ *is* the Riemannian approximation; ζ → ∞ recovers the sub-Riemannian metric.
- Fix: Reword accordingly.

**1.17 [m] Def 3.7 "two equivalent ways".**
- Problem: A MAP functional and a discretized density are not equivalent constructions.
- Fix: "We use P only through (i) the MAP functional … and (ii) its discretization, where it is a proper density." Also add one sentence that samples from exp(−λE_h) do not look like curves (the familiar situation for TV-type priors; cf. Lassas–Siltanen 2004). The prior is a MAP device, and §7.1's language should not suggest sampling.

**1.18 [m] Thm 3.8(ii) "Among curves joining two points of X, E prefers SR geodesics".**
- Problem: This is tautological, and it is wrong for finite ζ, where minimizers are Riemannian g_ζ-geodesics. With positivity, minimizers may contain in-place rotations (cusp-free geodesics with keypoints [DBRS]).
- Fix: Qualify accordingly, or delete (see 2.2). Also note that ∫√(1+ξ²κ²)ds is a *total-turning* penalty with only weak (strictly convex) preference for spreading curvature. Corners cost a finite ξ·angle, which is intended, since cartoons have corners. Say so instead of "smoothness".

### Section 4 (tomography)

**1.19 [M] §4.2 l.460–468: C : X → Y is not well defined as written.**
- Problem: X has θ ∈ S¹, but φ ∈ [0, π) and Y is described as "the cosphere bundle … modulo the sign of σ". Projectivizing "both sides" is also inconsistent with X being a cosphere bundle.
- Fix: Define
  - Ỹ = ℝ_s × S¹_φ × ℝ_t (oriented lines; t = −σ_φ/σ_s is the affine chart of the *projectivized* cotangent bundle over σ_s ≠ 0);
  - C : X → Ỹ, a diffeomorphism onto {s² + t² < 1}.

  C intertwines A(x,θ) = (x, θ+π) with (s,φ,t) ↦ (−s, φ+π, −t), so it descends to X′ → Y = Ỹ/∼. This also prepares §7.3 (the double covers) and the notation for Thm 4.3, which should be stated on ℝ × S¹_φ.

**1.20 [M] Prop 4.4 l.475–477: the positivity statement is wrong.**
- Problem: "positively Legendrian ones [go] to curves with ṡ = tφ̇". That equation is just the Legendrian condition on Y.
- Fix: Add the pushed frame, which is useful for implementers:
  - C_*X₁ = ∂_t
  - C_*X₂ = ∂_s
  - C_*X₃ = ∂_φ + t∂_s − s∂_t

  Then: Legendrian ⇔ ṡ = tφ̇; positively Legendrian ⇔ additionally ṫ + sφ̇ ≥ 0 (this equals a).

**1.21 [m] Prop 4.4.**
- Problem: "preserves mass and boundary mass (for the pushed-forward metric)" is true for any diffeomorphism and so carries no information.
- Fix: Replace it with the frame formulas above. They say the left-invariant metric pushes to the metric with orthonormal-up-to-weights frame ∂_t, ∂_s, ∂_φ + t∂_s − s∂_t.

**1.22 [m] l.498–501 "t is a derivative along the singular curve, a non-local quantity".**
- Problem: A derivative is local.
- Fix: "Estimating t requires a window along the curve, and curvature limits the window (Prop 5.3), so t is resolved only coarsely."

**1.23 [m] Thm 4.8 l.549–551.**
- Problem: "it propagates orientation along curves (Theorem 3.8(ii))" overstates what is proved.
- Fix: "Minimal-energy completion across the invisible directions is a design hypothesis, not a theorem; it is open problem 1."

### Section 5 (measurement model)

**1.24 [M] §5.1 item 1 l.572–579: the square-root singularity only holds for edges of nonzero curvature.**
- Problem:
  - A straight segment maps under C to a single point (s₀, φ₀) with a vertical fibre in Y (C_*X₁ = ∂_t), giving a point singularity, not (h−s)₊^{1/2}.
  - At inflection points the sinogram curve has a cusp and is not a graph s = h(φ). This also affects Cor 4.5.
  - Λ^{1/2} applied to x₊^{1/2} gives c₁H(x) + c₂ log|x|, not a pure jump.
- Fix: State the curvature hypothesis, and mention that polygons (the phantoms) violate it. Replace "returns it to a jump-type singularity" with "returns it to an order-0 (jump plus logarithmic) singularity".

**1.25 [m] Prop 5.3.**
- Problem: With convolution and an odd kernel, c = −J·r unless K is applied as a correlation. The side on which the jump is positive is unspecified. This is harmless after the modulus, but the "J" here is the jump of Λ^{1/2}y, which is ∝ contrast·√(radius of curvature). It is not the image contrast J used in §7.3.
- Fix: Use a different symbol, or state the relation.

**1.26 [m] Prop 5.2 is not used anywhere.**
- Problem: The Laplace scale κ defaults to 1 (`loss/data_discrepancy.py`, `train_lpd.py:228`). `coefficient_noise_std` is a Monte-Carlo estimate used only by the milestone-1 baseline. The noise model of Def 5.4 (Laplace on |c|) is also inconsistent with Prop 5.2 (Gaussian signed coefficients, so |c| is folded-normal).
- Fix: Either connect the two ("κ is set from Prop 5.2" plus a code change), or delete Prop 5.2 (see 2.4).

**1.27 [m] Remark 5.5 l.660–667.**
- Problems:
  - R* should be R^# (or define R*).
  - The identity is W_θ ∗ R^#g = R^#((RW_θ) ∗_s g), with 1-D filters (RW_θ)(·,φ) ≈ cos(θ−φ)G′(s) for an isotropic derivative of Gaussian; "filters that span all angles" is vague.
  - The normalization 1/(√(2π)σ) is missing from the model formula.
  - For an isotropic W_θ, |r| = |cos(θ−θ_e)| is very wide in θ. State the actual (anisotropic, aspect 2) kernel.
- Fix: Correct all four points.

### Sections 6–7

**1.28 [m] Prop 6.2.**
- Problem: A Gaussian has unbounded support, so equality never holds exactly.
- Fix: Say "approximate equality when positive and negative endpoints are separated by several σ".

**1.29 [m] Prop 6.3(2) "about 3% at ζ=2".**
- Problem: This depends on K. It is ~1.5% measured at K = 48, and the formula gives ~6% at the training default K = 32.
- Fix: State K.

**1.30 [m] Prop 6.4.**
- Problem: "‖τ‖_{ℓ²-sum}" is the group norm Σ_cells|τ_cell|₂ (ℓ_{2,1}). Since ζ ≥ 1 by assumption, min(1, ζ, ξ) = min(1, ξ).
- Fix: Rename the norm and simplify the constant.

**1.31 [m] §6.3 "periodic in θ for a, c".**
- Problem: a and c are never differentiated in θ (the divergence is X₁a + X₂c + ∂_θb), so only b needs a θ boundary condition.
- Fix: Drop "periodic for a, c".

**1.32 [M] §7.1 l.761–767.**
- Problem: The posterior with "1{q supported on T}" is not what is implemented. The implementation is a deterministic coupling q = J·m(a), J ∈ (0, J_max], m = 1 − e^{−a/a_ref}.
- Fix: Write the actual objective, min_{τ,J} D(J·m(a); data) + λE(τ), and note that it is non-convex because of m. Also: equation (7.1) and "Proposition 7.2" share a number with no Proposition 7.1, so renumber.

**1.33 [m] Prop 7.2.**
- Problem: The conclusion "training = learning a MAP solver" holds only under realizability and global optimization of 𝓛.
- Fix: Say so in one clause.

**1.34 [M] §7.3 l.825–827: "It ranks voxels correctly (ROC AUC is invariant under monotone maps)" is unjustified.**
- Problem: AUC is invariant under a single *global* monotone map. Contrast weighting rescales voxels differently, so a weak true edge can rank below the blur or streak of a strong edge.
- Fix: Delete the claim, or replace it with "AUC is computed on a contrast-weighted score and therefore under-weights weak edges". Also replace "m ≈ min(1, q/J_max)" with "m = q/J_max when q ≤ J_max".

**1.35 [m] §7.3 equivariance.**
- Problem: The rotation equivariance on Y is about the origin only. On the Cartesian image grid only multiples of π/2 are exact symmetries. Translations act non-trivially on Y (Prop 4.6), so the dual CNNs are not translation equivariant. "Ordinary convolutions respect [the combined action] approximately" is unsupported.
- Fix: Say "do not respect it in general".

### References

**1.36 [m] References.**
- Problems:
  - [H2] and [W] are never cited (cite [W] at the walnut mention, l.537).
  - [M] has no year.
  - Check [H1, 8.1.5] (see 1.8).
- Fix: Correct the list as above.

---

## 2. Redundancy and simplification

**2.1 Merge Prop 3.6 and Thm 3.8 into one proposition, "Energy of curve configurations".**
- Thm 3.8(ii) is Prop 3.6 restated.
- (iii) and (iv) are the same computation (open vs closed). State one item: "deleting a sub-arc of length ε adds 2ν − ε, whether the curve is open or closed".
- (v) is a pointer only.
- Keep (i).
- The remark on lower semicontinuity stays.

**2.2 Shorten the "smoothness" discussion.**
- Drop the geodesic and association-field claim from the theorem.
- Move it to §8 as context.
- Replace it with the one-line honest statement from 1.18.

**2.3 Smirnov.**
- Once Thm 3.9 is restricted to integral currents (1.12), the general Smirnov theorem is only needed for the sentence "every positively Legendrian normal current is a superposition of positively Legendrian curves".
- Recommendation: keep one corrected sentence with [Sm, PS] as motivation, and base the proof of Thm 3.9 on the integral-cycle decomposition.

**2.4 Prop 5.2 (noise).**
- Delete it unless it is connected to κ (1.26).

**2.5 Prop 4.4, second identity.**
- The volume identity follows from C*β = α, since C*(β∧dβ) = α∧dα, β∧dβ = ds∧dφ∧dt and α∧dα = −dx₁∧dx₂∧dθ.
- State it as a one-line consequence instead of a separate claim.

**2.6 Prop 4.7 (fan beam) and the walnut calibration (l.534–538).**
- These are correct but implementation-level.
- Move them to an appendix "Fan-beam geometry". Keep one sentence in §4 saying that the fan-beam case is a reparametrization Φ of the lines, and that t is unchanged.

**2.7 Elastica variant (l.324–330).**
- Keep it, but shorten to two sentences. It is only there to justify the choice of the sub-Riemannian mass.

**2.8 §1.2 and §1.3 duplicate §3.**
- After the fixes, make §1.3's table the only place where the five desiderata are listed.
- §1.2 should not repeat the property list ("independence, smoothness, stability …"). Replace it with "see the table below".

**2.9 Empirical numbers and project history (G2).**
- Covers:
  - "milestone 1", the "63%" figure (l.624–626);
  - "the mixture model" (l.643);
  - the §5.2 correlations;
  - the §6.3 boundary-mass numbers;
  - §9 "Status" and "Update".
- The "mixture model" and "milestone 1" are never defined in this document. Either define them in one sentence (the per-voxel classifier baseline of `experiments/milestone1_canonical_relation.py`) or remove the references.
- Some quoted numbers are not checked by the cited tests:
  - `test_lpd.py` runs at n = 64 and asserts only the residual < 0.65 and the smear ordering; the correlation 0.85, the 31-cell value 0.56 and the 63-cell value 0.62 are not in the test.
  - `test_regularization.py` does not contain the "10.6" comparison.
  - `test_walnut.py` asserts > 0.999, not 0.9999.
- Either qualify the numbers as "measured once" or move them out with their provenance.

---

## 3. Clarity and coherence

**3.1 The story vs the method (G1).**
- The current text contradicts itself, which is the main coherence issue:
  - Prop 5.3 "removes" the ambiguity (l.627–628);
  - Remark (b) says it does not;
  - §9 says the streaks remain;
  - Remark 5.5 then replaces the model.
- After G1, present it as follows:
  - the local sinogram model and why it is ill-conditioned in t;
  - the full-aperture model that fixes this for dense angles;
  - limited data as the open case.
- Remark (b)'s claim that the Legendrian envelope "has the least energy among all explanations" is unproved, and the experiments show the MAP of Def 5.4 is not the truth. Label it as the design intent.

**3.2 Forward references.**
- Cor 2.9 → Lemma 6.1 (fix via G4).
- Def 3.5 → §6.2.
- Def 5.4 → §7.3: define q = J·m where q is first used, or say "amplitude field, parametrized in §7.3".

**3.3 X vs X′ vs Y vs Ỹ.**
- State once, in a short paragraph after the notation table:
  - which space each field lives on;
  - what the code stores (X′, θ ∈ [0, π));
  - what the network computes on (double covers with 2K slices);
  - which components are even or odd.
- §7.3's "double covers" bullet then becomes a pointer.

**3.4 Corners.**
- Make one explicit modeling statement near l.227–231:
  - the prior targets the closed Legendrian lift (edges plus turning arcs);
  - the remaining corner fibre, a weaker singularity, is not modeled;
  - labels currently include it (1.5c).

**3.5 State the hyperparameters that the text relies on.**
- Values, with their source:
  - K = 32 in training (48 in some tests);
  - ξ = 0.1, ν = 0.5 (training) / 1.0 (default), ζ = 2;
  - sinogram kernel aspect 6;
  - image analyzer σ = 1.5, aspect 2, edge width 2.6;
  - noise 0.01·max|y|;
  - sinogram size 3n × 2n.
- Put them in one table in the numerical section (G2), with source lines (`train_lpd.py:58-84`, `loss/regularization.py`, `operators/microlocal.py:109`, `data/synthetic.py:66,79`).
- Note that ν = 0.5 makes 2ν = 1 the bridging scale, i.e. the full image width on [−1, 1]². This deserves a sentence of interpretation, or a check that this is intended.

**3.6 Def 5.4.**
- Problem: The code uses the mean, not the sum (`data_discrepancy.py:180`). This only rescales κ.
- Fix: Mention it, or write "mean".

---

## 4. Notation overloads (fix first; G3)

| symbol | current meanings | suggestion |
|---|---|---|
| c | analyzer coefficients c = My; X₂-component of τ | coefficients → **m̂** or **y_M**; keep c for the component, or rename the component **n** ("normal") |
| b | X₃-component of τ; image-side coefficients b(x,θ) (Remark 5.5) | image coefficients → **β_img** or **u_W** (β is also the contact form on Y, so pick one) |
| ξ | covector (Def 2.1, Thm 2.6); turning length scale | turning scale → **ℓ_θ** or **ε_θ** |
| λ | Gibbs weight; Legendrian speed (Def 2.5, proof of Thm 3.9); fibre scalar (Thm 2.6) | speed → **v**; fibre scalar → **r** |
| κ | curvature; Laplace scale; κ_Y | Laplace scale → **s_ε** or **b_L** |
| σ | sinogram covector scale; noise std; kernel widths σ_n, σ_τ; mollifier σ_x, σ_θ; PDHG step | keep σ for covectors; noise → **σ_noise**; PDHG steps → **s, t̃** or drop |
| τ | density field; translation τ_y; PDHG step | translation → **T_y**; drop the PDHG step symbols |
| D | unit disk; D = R_s + R_d; D(q;c) data term; D₀, D_k domains; \|D\| (Fourier multiplier) | disk → **B**; source–detector distance → **R_{sd}**; data term → **𝒟**; multiplier → **\|∇\|** |
| J | contrast/jump; rotation by π/2 (Def 2.5); jump of Λ^{1/2}y (Prop 5.3) | rotation → write ω^⊥ directly; sinogram jump → **J_Y** |
| γ | curve; rotation angle (Prop 4.6); fan angle (Prop 4.7) | angles → **ψ**, **ϑ_f** |
| X | manifold; frame X_i | acceptable, but consider **M** or **SE** for the manifold if a rename is done anyway |
| r | fibre response; later angular response in Remark 5.5 | angular response → **r_θ** |
| (a, c, b) order | τ = aX₁ + cX₂ + bX₃ while the code signature is (a, b, c) | write τ = aX₁ + bX₃ + cX₂ throughout to match the code order |

The notation table at l.30–42 must be updated to match.

---

## 5. Suggested order of work

1. G3: the notation pass (§4).
2. G4: move Lemma 6.1 into Section 2.
3. The Section 3 fixes (1.9–1.15, 2.1–2.3). These are the mathematical core and settle what the prior actually guarantees.
4. The Section 2 and 4 fixes (1.4–1.6, 1.19–1.21).
5. G1: restructure Section 5 and update §1.2, §7.1, §7.2, §8 (1.24–1.27, 1.32, 3.1).
6. G2: move empirical numbers and logs out, and add the hyperparameter table (2.9, 3.5).
7. Minor items.

Check after editing: every claim in §1.2 and the §1.3 table must point to a result whose hypotheses (integral, c = 0, positivity) are stated. Wherever the network output does not meet them, the text must say so (§7.3).
