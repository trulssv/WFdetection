# Review of `research/presentation/wfdetection_beamer.tex`

The slides are reviewed for correctness, clarity, and notation consistency with
`research/background.md` (revised 2026-10-06).

`background.md` is the **reference** for notation. Where the slides need a symbol
that the document lacks, add it to both files (§1, "new in both").

Slide numbers count frames in source order: 1 = title. Line numbers refer to the `.tex`.

Tags:
- **[C]** wrong statement;
- **[M]** misleading, or inconsistent with the document or code;
- **[m]** minor.

---

## 1. Notation map (apply to both files)

| concept | slides now | use in both | notes |
|---|---|---|---|
| sinogram coordinates | $(\varphi,s,t)$, $y(\varphi,s)$ | $(s,\varphi,t)$, $y(s,\varphi)$ | code and document order (`operators/canonical_relation.py`) |
| sinogram coefficients | $c=My$, $c=\lvert My\rvert$ | $y_M=\lvert My\rvert$ | also $C^{-1}c\to C^{-1}y_M$ everywhere, including the slide text that describes figures |
| image-side coefficients | $b$, $b=A\,\mathrm{FBP}\,y$ | $y_W=\lvert W_\theta*\mathrm{FBP}\,y\rvert$ | |
| image analyzer | $A$ | $W$ (filters $W_\theta$) | $A$ is the antipodal map in the document |
| current components | $(a, b_T, c_T)$, $\tau=aX_1+c_TX_2+b_TX_3$ | $(a,b,c)$, $\tau=aX_1+bX_3+cX_2$ | possible once $b,c$ no longer denote coefficients |
| turning length scale | $\xi$ | $\ell_\theta$ (`--xi` in code) | $\xi$ is a covector in the document |
| kernel coordinates in $K_t$, $W_\theta$ | $a$ (across), $b$ (along) | $u_\perp$ (across), $u_\parallel$ (along) | **new in both**: the document only describes the kernels in words, so add these two symbols to §5.1 and §5.4 |
| kernel widths | $\sigma_n$, $\sigma_t$ | $\sigma_n$, $\sigma_\parallel$ | $\sigma_t$ clashes with the fibre coordinate $t$ |
| angular response (image side) | $r(\theta,\theta')$ | $r_\theta(\theta,\theta')$ | $r$ is the sinogram fibre response |
| residual argument in $\rho$ | $\rho(r)$ | $\rho(e)$ | **new in both**: add $\rho(e)=\sqrt{e^2+\delta^2}-\delta$, $\delta=10^{-2}$ (`loss/data_discrepancy.py:172`) to Defs 5.3 and 5.4 |
| image-side forward model | $\hat b(q)$, $F(Jm)$ | $\hat y_W=F_X(q)$ | **new in both**: name the prediction in Def 5.4. Use $\hat y_M=\mathcal R_tCq$ in Def 5.3 |
| normalization | $w$ | $w_X$ (image side), $w$ (sinogram) | as in the document |
| data term | $D$ | $\mathcal D_X$ (default), $\mathcal D_Y$ | |
| adjoint | $R^*$ | $R^\#$ | |
| ridge threshold | $t$ | $\eta$ | **new in both**: $t$ is the fibre coordinate. If the document mentions the ridge threshold, use $\eta$ |
| lifted spaces | "Sinogram side $Y=(\varphi,s,t)$", "Image side $X=(x,\theta)$" | $\tilde Y\ni(s,\varphi,t)$, $X\ni(x,\theta)$; stored on $Y$, $X'$ ($\theta,\varphi\in[0,\pi)$) | one footnote on slide 3 or slide 7 suffices |
| contrast bound | $J\in(0,1)$ | $J\in(0,J_{\max})$, $J_{\max}=1$ | code: $J=J_{\max}\,\mathrm{sigmoid}(\cdot)$, an open interval. The document (§7.1, §7.3) writes $(0,J_{\max}]$; change it to $(0,J_{\max})$ |
| theorem references | "Thm. 3.8–3.9" | "Prop. 3.7, Thm. 3.8" | after the revision |

After the pass, grep the `.tex` for `\xi`, `c_T`, `b_T`, `\hat b`, `F(`, `R^*`, `(\varphi,s` and the analyzer name `A`. No stale use should remain.

---

## 2. Correctness

**2.1 [C] Slide 2, "Goal" (l.29), and slide 25, "Takeaways".**
- Problem: "We do not try to reconstruct the whole image first". The default data term is computed on an FBP (slide 6 says so). This also contradicts the document (§5.4, §8).
- Fix: "We never compute a regularized or learned reconstruction. The data enter only through fixed linear filters (local sinogram filters, and oriented filters applied to the FBP), followed by a modulus."

**2.2 [M] Slide 4, "Pipeline Overview" (l.57).**
- Problem: "Everything after the analyzers is defined on the lifted *image* domain" contradicts slide 7, where the LPD dual variable $d$ lives on the sinogram side $\tilde Y$.
- Fix: "The output and the data term live on the lifted image domain. The LPD alternates between $\tilde Y$ (dual) and $X$ (primal)."

**2.3 [M] Slide 13, "Why Such A Regularizer?" (l.210, l.215). Sparsity row and footnote.**
- Problem: "every component costs at least $\min(2\nu,2\pi\xi)$" holds only for integral, exactly Legendrian ($c=0$), positively oriented curves (document Thm 3.8). It does **not** hold for the network output, which has $c\neq0$ and real weights (document §3.3). "Each requirement is a theorem about $E$" overstates.
- Fix:
  - Add "(for curves)" to row 5.
  - Change the footnote to: "Rows 1–4: Prop. 3.7; row 5: Thm 3.8, for curves only. For the relaxed network output, sparsity holds per unit weight (background §3.3)."

**2.4 [C] Slide 14, "Why Not A Simpler Regularizer?" (l.220).**
- Problem: "Removing a single point does not change the length, so it cannot express 'don't break curves' … This *needs* the endpoint term". Removing a point changes neither the length *nor* the boundary, and a zero-length gap is invisible to $E$ as well. This is the same error that was fixed in the document's §1.3.
- Fix: "A length term alone *rewards* a gap (it saves $\varepsilon$). Only the endpoint term makes a gap of length $\varepsilon$ cost $2\nu-\varepsilon>0$."

**2.5 [M] Slide 14 (l.221).**
- Problem: "$E$ is convex and 1-homogeneous. The MAP problem is well-posed." $E$ is convex, but the MAP objective (document eq. 7.1) is non-convex because of $q=J\,m(a)$, and well-posedness is not shown.
- Fix: "$E$ is convex and 1-homogeneous in $\tau$. The non-convexity sits only in the data coupling $q=J\,m(a)$."

**2.6 [M] Slide 25, "Takeaways" (l.376).**
- Problem: "while staying convex". The same issue as 2.5, and property 5 does not hold for the output (2.3).
- Fix: "a convex prior that has the five properties for curves".

**2.7 [M] Slide 12, "The Regularizer" (l.187–192).**
- Problems:
  - "$|\mathrm{div}\,\tau|$ = number of endpoints" holds only for curves of multiplicity one. In general it is the boundary mass.
  - "length in a sub-Riemannian metric": for finite $\zeta$ it is the Riemannian approximation.
  - "$c_T$ is the non-physical 'sideways' component" omits that $c\neq0$ is *required* at finite resolution (document Prop 6.2).
- Fix:
  - Write "boundary mass (= number of endpoints for curves)".
  - Write "length in a Riemannian approximation of the sub-Riemannian metric".
  - Add "kept because blurring creates it (Prop 6.2); penalized by $\zeta$".

**2.8 [M] Slide 6, "Image Analyzer" (l.87).**
- Problem: "$W_\theta\,\mathrm{FBP}=R^*\circ(\text{sinogram filters})$".
- Fix: Write $W_\theta*\mathrm{FBP}\,y=R^\#\bigl((RW_\theta)*_s\Lambda y\bigr)$. Use $R^\#$, and keep the ramp filter $\Lambda$ explicit.

**2.9 [m] Slide 5, "Sinogram Analyzer" (l.72).**
- Problem: The half-order ramp filter "turns the $\sqrt{\cdot}$-singularities of $Rf$ back into jumps". This is true only for edges of nonzero curvature, and the result is a jump plus a $\log$ term.
- Fix: "turns the $\sqrt{\cdot}$-singularities of $Rf$ (curved edges) into order-0, jump-like singularities".

**2.10 [M] Slide 23, "Current Limitations" (l.355). The first bullet is out of date.**
- Problem: "the present data term does not fully explain the true coefficients, so the MAP solution is not yet the true wavefront set". That was the finding for the *sinogram* model (now in `experiments/README.md`). For the image-side model with edge width 2.6, the true support fits as well as any $q$ (document §5.4, README). The remaining misfit is the one shown on slide 20:
  - over-prediction at crossings (the modulus is non-linear: two edges add before $|\cdot|$);
  - under-prediction along close parallel edges;
  - smooth intensity variation.
- Fix: Replace the bullet with that list.

**2.11 [M] Slide 21, "Learned Outputs" (l.321), and slide 23 (l.358).**
- Problem: "$J$ is saturated … so $m$ carries all of the detection". This is the predicted behaviour: at the optimum $J=J_{\max}$ (document §7.3). Its consequence is missing: $m=q/J_{\max}$ is **contrast-weighted**, so AUC and thresholds under-weight weak edges.
- Fix: Add one clause to slide 21, and make slide 23's coupling bullet match.

**2.12 [m] Slide 14 (l.224).**
- Problem: "$\nu$ the gap length the prior is willing to bridge". The bridged length is $2\nu$. With $\nu=0.5$ that is the radius of the object disk, so in practice $\nu$ acts as an endpoint penalty (document §6.4).
- Fix: Say so.

**2.13 [m] Slide 13, row 2 ("smoothness … grows with curvature").**
- Problem: The cost is a total-turning penalty, and a corner costs only $\ell_\theta\cdot$angle (slide 14 says this). Row 2 should not suggest a smoothness prior.
- Fix: Write "penalizes total turning; corners allowed".

**2.14 [m] Slide 15, "How We Evaluate" (l.234).**
- Problem: "Corners get all angles". Correct for the labels, but the prior only models the turning arc (document §2.3, "Modeling choice"), so AUC rewards corner directions the model does not represent.
- Fix: Add a half-sentence noting this.

**2.15 [M] Slide 22 (l.341) and `experiments/README.md`: inconsistent compute numbers.**
- Problem: The slides say "3.2 s/step and 17.4 GB" at $n=256$, $K=64$, batch 2. The README says "4.2 s/step, ~16 GB" for the same configuration.
- Fix: Reconcile from the source (`research/presentation/bench_scaling.py`, and the training logs under `outputs/`). Use one number in both places and state where it came from. If the number cannot be determined, keep the slide value and mark the README value as "earlier measurement".

**2.16 [m] Slide 6 (l.85, l.90).**
- Problem: "$K=64$ angles", "angular FWHM ≈ 90°".
  - $K$ is the number of $\theta$-slices on $[0,\pi)$, as in the document. Say "$K=64$ orientation slices".
  - The FWHM should be checked against `ImageDomainAnalyzer`/`angular_response` in `operators/image_analyzer.py`, or dropped.
- Fix: Apply both.

**2.17 [m] Slides 17–18 (l.276–282).**
- Problem: The numbers are for the $n=256$, $K=64$ production run. The document's README numbers are for $n=96$.
- Fix: State the grid on slide 18 (it already says $256\times256$; add $K=64$). Make sure the README has a row for this run, so the slide numbers have a source.

---

## 3. Clarity

**3.1 Slide 3 (l.42).**
- Problem: $\theta\in[0,\pi)$ is the stored range, but the network computes on $[0,2\pi)$.
- Fix: Add "(stored mod $\pi$ for real images)". Do not introduce more.

**3.2 Slide 7, "Where Does Each Quantity Live?"**
- Problem: It is the best place for the spaces.
- Fix: Write $\tilde Y=(s,\varphi,t)$ and $X=(x,\theta)$, and list $y_M$ on $\tilde Y$ and $y_W$ on $X$. The bullet "image coefficients $b=A\,\mathrm{FBP}\,y$" becomes "$y_W=|W_\theta*\mathrm{FBP}\,y|$". The statement that the data term uses $y_W$ and that $y_M$ is only an input should stay. It is correct and important.

**3.3 Slide 8, "What The Network Learns".**
- Problem: $a$ appears twice, as line density and inside "current components $(a,b_T,c_T)$".
- Fix: Merge into one item: "current components $(a,b,c)$; $a\ge0$ is the line density and gives $m=1-e^{-a/a_{\mathrm{ref}}}$".

**3.4 Slide 10, "Training Objective".**
- Problem: The data term is called $D$ and the residual function uses $r$, which collides with the angular response defined two lines earlier.
- Fix: Use $\mathcal D_X$ and $\rho(e)$ (see §1).

**3.5 Slide 20.**
- Problem: "$\hat b=F(Jm)$" introduces $F$ for the first time.
- Fix: Use $\hat y_W=F_X(q)$, the same symbol as slide 10.

**3.6 Slide 13 table.**
- Fix: Keep the table. Its rows now match the document's §1.3 table except for the hypotheses column. Add a short "for curves" qualifier rather than a column; the slide is already dense.

---

## 4. Changes to `background.md` for consistency

These are small; the document stays the reference.

1. **§5.1 and Def 5.3.** Write the kernel explicitly as $K_t\propto u_\perp e^{-u_\perp^2/2\sigma_n^2-u_\parallel^2/2\sigma_\parallel^2}$ ($u_\perp$ along the conormal $(1,-t)$, $u_\parallel$ along the tangent). Name the prediction $\hat y_M=\mathcal R_tCq$. Give the smoothed absolute value $\rho(e)=\sqrt{e^2+\delta^2}-\delta$, $\delta=10^{-2}$, once, and use it in both data terms.
2. **§5.4 and Def 5.4.** Write $W_\theta$ in the same form, with $u_\perp=\Delta x\cdot\omega(\theta)$ and $u_\parallel=\Delta x\cdot\omega^\perp(\theta)$. Name the prediction $\hat y_W=F_X(q)$.
3. **§7.1 and §7.3.** Change $J\in(0,J_{\max}]$ to $J\in(0,J_{\max})$, matching the code ($J_{\max}\,\mathrm{sigmoid}$). Keep "$J=J_{\max}$ at the optimum" as "$J\to J_{\max}$" (a supremum).
4. **Notation table.** Add $\hat y_M,\hat y_W$ / $F_X$, $\rho$, $u_\perp,u_\parallel$, $\sigma_\parallel$ (already present), and $\eta$ (ridge threshold) if it is used.
5. **§6.4 parameter table.** Keep it; the user likes it. Add one row "production run (slides): $n=256$, $K=64$, batch 2", with the reconciled timing from 2.15.
