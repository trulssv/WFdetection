# Presentation revision plan (2026-10-06)

This plan comes from the user's 13 comments on `wfdetection_beamer.tex`. `background.md` remains the reference for notation. Where the plan changes notation, change both files.

## A. Notation, in both files

1. **Filtering symbol.** Replace ⋆ (correlation) by ∗ ("filter with") in the slides, in `background.md` §5.1, and in the flowchart label (`make_assets.py`). Add one sentence in the document only: the kernels are odd, so |K⋆u| = |K∗u| exactly, and the code's `conv2d` is a correlation.
2. **Kernel widths.** Rename σ_n → σ_⊥ everywhere. This covers the slides, the document, the §6.4 parameter table, the notation table and figure labels.
3. **One generic kernel.** Define $\mathcal G_{\mathbf n}(u)\propto (u\cdot\mathbf n)\,e^{-(u\cdot\mathbf n)^2/2\sigma_\perp^2-(u\cdot\mathbf n^\perp)^2/2\sigma_\parallel^2}$ for a unit normal $\mathbf n$. Then:
   - $K_t=\mathcal G_{\mathbf n(t)}$ on $(s,\varphi)$, with $\mathbf n(t)=(1,-t)/\sqrt{1+t^2}$;
   - $W_\theta=\mathcal G_{\omega(\theta)}$ on $x$.

   Put this in `background.md` §5.1 (and refer to it from §5.4) and in the notation table. State the differences once: domain, how orientation is parametrized, aspect (6 vs 2), and pre-filter (Λ^{1/2} vs FBP).
4. **The modulus.** Say once what |·| means in the slides and in §5.1 (a sentence in §5.1 is enough):
   - it is the pointwise magnitude of the signed response;
   - it removes the sign of the jump (dark→bright vs bright→dark), which the wavefront set ignores (θ ~ θ+π);
   - it makes the coefficients comparable with q ≥ 0;
   - the cost is that edges on one ray add before the modulus is taken.

## B. New figures (`research/presentation/make_explainer_figs.py`, new script)

1. **`assets/fig_canonical_relation.png`**
   - Left panel, image domain:
     - the unit disk, with one ellipse edge;
     - a point x on the edge, with the normal ω(θ) drawn as an arrow;
     - the ray through x tangent to the edge (direction ω^⊥), with s (signed distance of the ray from the origin) and t (position of x along the ray) marked.
   - Right panel, sinogram:
     - the singular curve s = h(φ) of the same ellipse, computed analytically from its support function;
     - the point (s, φ = θ) with its tangent line of slope t = h′(φ) drawn.
   - Below the panels: $C(x,\theta)=(x\cdot\omega,\theta,x\cdot\omega^\perp)$.
   - Use matplotlib and analytic geometry only.
2. **`assets/fig_sinogram_analyzer.png`**
   - Left: a zoom of a real simulated sinogram (Λ^{1/2}y of the same ellipse; use `operators/microlocal.py` / `half_ramp` and the data simulation from the repo). Show one point (s₀, φ₀) on the singular curve.
   - Middle: three kernels K_t from `SinogramAnalyzer` (true slope t₀, and two wrong slopes), drawn as small images, each in its orientation.
   - Right: the signed and absolute response y_M(s₀, φ₀, t) against t, peaking at t₀, with a vertical line at t₀. Mark the FWHM (the "poor t-resolution").
   - Use the repo's actual analyzer code so that the figure is faithful.
3. **`assets/fig_pr_curves.png`**: precision–recall curves on the lifted voxels for C^{-1}y_M, y_W and m, over the validation set. Use the same validation set and model as `make_synthetic_figs.py`; reuse its `build`/`run_model`.

## C. Metrics (presentation scripts only; do not change training or evaluation code)

Add to `make_synthetic_figs.py`, or to the new script, and print the results:

- **AP:** average precision on the lifted voxels, with positives/negatives from `label_masks`, for C^{-1}y_M, y_W and m. Report the mean over the validation set.
- **Hard-negative AUC:** AUC with negatives restricted to voxels within 6 px (spatially) of the true wavefront set. Dilate the spatial projection of the positives by 6 px and intersect with the negatives.
- **Reference values** (measured on validation sample 0, to check the computation):
  - AUC: C^{-1}y_M 0.913, y_W 0.981.
  - Hard-negative AUC: C^{-1}y_M 0.797, y_W 0.882.
  - Positives are 0.17% of voxels and negatives 98.9%.
  - At the median positive score, 2.46% of the negatives of C^{-1}y_M exceed it (about 29 false voxels per true voxel).
  - The diagnostic script is in the session scratchpad; recompute rather than copying these numbers.

## D. Slide restructure (target: lower text density; split, don't cram)

New order. Keep existing figures where they fit.

1. Title
2. Goal in plain language
3. What is a wavefront set (lifted space, plot convention, θ legend)
4. Pipeline overview (flowchart)
5. **The canonical relation** (new; `fig_canonical_relation.png`; WF(Rf) = C(WF f); a rotation per angle slice; t = position along the ray = slope of the sinogram curve)
6. **Oriented derivative-of-Gaussian filters** (new): 𝒢_n, σ_⊥ and σ_∥, what ∗ and |·| mean, and that both analyzers are instances of 𝒢_n
7. Sinogram analyzer M: K_t = 𝒢_{n(t)}, how t enters (`fig_sinogram_analyzer.png`)
8. Limitation of M: poor t-resolution, so streaks along tangent lines (fibre response FWHM, and the C^{-1}y_M panel)
9. Image analyzer W: W_θ = 𝒢_{ω(θ)} on the FBP, the edge response formula, angular FWHM ≈ 83°. "Still a fixed linear map of y, then |·|", with a pointer to the backup slide for the identity.
10. Where does each quantity live (as now)
11. The network: LPD iterations, inputs y_M and y_W, the equations
12. **Network outputs** (split out):
    - the head outputs (a, b, c, J) directly on (x, θ), with no edge detection needed;
    - m = 1 − e^{−a/a_ref};
    - a_ref = 1/(2π σ_x σ_θ) with σ = 2 cells: the peak line density of a resolved unit curve, a fixed normalization (m ≈ 0.63 at its peak), not tuned;
    - q = Jm;
    - ridge extraction (non-maximum suppression) is used only for thin-curve evaluation.
13. Training objective: 𝒟_X and the forward model F_X (as now, less text)
14. **Desiderata** (moved here, before the regularizer): the table
15. The current T: τ = aX₁ + bX₃ + cX₂, and the meaning of a, b, c
16. The energy E(T): the formula, the parameters, and the link back to the desiderata rows
17. Why not a simpler regularizer (possibly split into 2 slides if dense)
18. Ground truth and baselines (from "How we evaluate"): analytic WF, corners, C^{-1}y_M, y_W, m
19. **Metric 1: ROC AUC.** The formula $\mathrm{AUC}=P(\text{score}(v^+)>\text{score}(v^-))$ (Mann–Whitney). Positives and negatives: soft label ≥ 0.5 vs < 0.01, with a gap. Why: it is threshold-free and ranks every voxel.
20. **Metric 2: thin-ridge precision, recall and F.**
    - Formulas: P = matched predicted ridge points / all predicted; R = matched true points / all true; F = 2PR/(P+R).
    - Matching tolerance: 1.5 px and ±1 θ bin. ODS: one best threshold η over the dataset.
    - F_x is spatial-only.
    - Why: the output we want is thin curves with correct orientation.
21. **Why AUC alone is misleading here**:
    - the class imbalance numbers from C;
    - the max-over-θ display exaggerates the streaks;
    - AP and hard-negative AUC as the complements;
    - `fig_pr_curves.png`.
22. Validation example: inputs and baselines
23. Validation example: LPD output and ridges
24. Validation summary: add the AP and hard-negative AUC numbers. Keep AUC and F.
25. Loss convergence
26. Inside the 5 LPD iterations
27. How well does the output explain the data
28. Learned outputs and where the prior pays
29. Walnut: qualitative results (as now)
30. **Evaluating on real data (plan)**. There is no ground truth; today the evaluation is qualitative, plus agreement with y_W. Proposal:
    - a pseudo ground truth from the 1200-projection reference, using a slower *independent* method (e.g. a multi-scale shearlet/curvelet decay test, Thm 5.1, or a long non-learned PDHG solve of the MAP problem);
    - evaluate the network on reduced data (fewer angles, more noise);
    - caveat: the pseudo ground truth must not be the same detector as a baseline.
31. Compute cost (less text)
32. Limitations
33. Next steps (add the real-data evaluation)
34. Takeaways
35. **Backup: where $W_\theta*\mathrm{FBP}\,y=R^\#\bigl((RW_\theta)*_s\Lambda y\bigr)$ comes from**:
    - R is the ray transform and R^# the backprojection (its adjoint);
    - FBP y = R^#(Λy);
    - ∗_s is a 1-D convolution in s per angle φ;
    - the Fourier slice / intertwining identity R(W∗f) = (RW) ∗_s Rf gives W ∗ R^#g = R^#((RW) ∗_s g).

Style:
- Keep the existing plain style.
- At most about 5 bullets per slide, and short ones.
- Use `\small` only where needed.
- Put backup slides after `\appendix`.
