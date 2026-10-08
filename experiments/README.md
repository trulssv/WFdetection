# experiments/

Runnable studies. Run them as modules from the repository root so the
packages resolve:

```bash
python -m experiments.<name> [options]
```

Outputs (figures, `metrics.json`) go to `outputs/<name>/`, which is not
tracked.

## `milestone1_canonical_relation`

A learning-free check of the whole chain on a fixed phantom (ellipses and
polygons):

1. render $f$, compute $y=Rf+\text{noise}$ with ODL/ASTRA (`--n-angles`,
   `--noise`);
2. compute $c=My$, the per-filter noise std, and the mixture posterior
   $P(\text{WF}\mid c)$;
3. compare $c$ with $C\,\mathrm{WF}(f)$ on the sinogram side, and
   $C^{-1}$posterior with $\mathrm{WF}(f)$ on the image side;
4. evaluate the curve prior on the true flux: the length term against the
   analytic value, and the endpoint mass with and without corner arcs.

Outputs:

* `overview.png`: phantom, sinogram, $\max_t My$, the true WF set
  (hue = orientation), the learning-free pull-back $C^{-1}|My|$, and
  $\max_\theta C^{-1}$posterior (which shows the tangent-line ambiguity);
* `lifted_sinogram_slices.png`: $(s,t)$ slices of $|My|$ next to the predicted
  $C\,\mathrm{WF}(f)$ at four angles;
* `metrics.json`: $t$-localization error, AUCs, prior terms.

### Results (default settings, 1% noise)

| quantity | value | interpretation |
|---|---|---|
| $t$-localization, mean / median abs. error (cells) | 0.03 / 0.55 | conventions correct, no bias |
| global argmax of $t$ within 2 cells | 81% | the rest sit on other edges tangent to the same ray |
| AUC, $My$ vs $C\,\mathrm{WF}$ (sinogram side) | 0.84 | limited by streaks in $t$ |
| AUC, $C^{-1}My$ vs $\mathrm{WF}$ (image side) | 0.85 | correct orientation, tangent-line streaks |
| AUC of pulled-back posterior / fraction > 0.5 | 0.73 / 63% | noise-only $p_0$ saturates on streaks |
| prior length ($\zeta=1$) vs analytic | 10.40 vs 10.74 | ok (3%) |
| endpoint mass: closed curves / without $c$ / without corner arcs | 1.2 / 15.1 / 15.0 (expected 14) | $c$ is necessary; endpoints counted correctly |

Conclusion: $C$, $M$, the ODL conventions and the prior behave as derived.
Local sinogram analysis fixes $(s,\varphi)$ of each singularity but leaves
$t$ (position along the ray) ambiguous. Resolving it is the curve prior's
role, and that is the main thing milestone 2 has to demonstrate.

Options: `--n` (default 128), `--n-theta` (64), `--n-angles` (400), `--noise`
(0.01, relative to max |y|), `--xi` (0.1).

## `train_lpd`: unsupervised training of the LPD network

```bash
python -m experiments.train_lpd --steps 4000 --lam 1e-4 --out outputs/train/run1
python -m experiments.train_lpd --resume outputs/train/run1 --steps 6000   # continue
```

Minimizes $D_{\text{data}}(Jm)+\lambda E(a,b,c)$ over synthetic data
(`data/synthetic.py`; random cartoons, smooth backgrounds, 1% noise, 3n dense
angles). This is amortized MAP estimation
([research/background.md](../research/background.md), Prop. 7.2); no labels
are used. The data term (`--data-term`) is one of:

* `image` (default): the full-aperture coefficients $b=|W\,\mathrm{FBP}\,y|$
  with the local model `OrientedEdgeForwardModel`;
* `sinogram`: Definition 5.4 on $c=My$;
* `both`.

Every `--val-every` steps it reports, on `--val-size` fixed held-out
phantoms:

* the ROC AUC of the soft membership $m$, of the learning-free $C^{-1}c$
  (`val_auc_baseline`) and of the learning-free $b$ (`val_auc_b`);
* for **thin estimates**, the F-score of the ridges of $m$ (`val_F`,
  `val_P`, `val_R`) and of $b$ (`val_F_b`) against the rasterized analytic
  wavefront set, at the dataset-optimal threshold. A point counts as correct
  within 1.5 cells and ±1 orientation cell (`evaluation/ridges.py`).
  `val_F_x` scores the spatial projection (the singular support) only.

It also writes `val_<step>.png` (phantom, thin true WF, both baselines, $m$,
and the ridges of $b$ and $m$), `log.jsonl` and `checkpoint.pt`.

A tqdm bar shows loss, D, E, the learning rate, and the time per step spent
on data generation vs. the network. Under `nohup` it prints one line every
30 s.

Main options:

| group | options |
|---|---|
| grid | `--n 96 --n-theta 32` |
| prior | `--lam` (weight), `--xi`, `--nu`, `--zeta` |
| network | `--n-iter 5 --hidden 24` |
| optimizer | `--batch 2`, `--lr 1e-3` (cosine schedule) |
| data | `--noise`, `--background` |
| image analyzer | `--sigma-cells 1.5 --aspect 2 --edge-width 2.6 --detrend 0` |
| other | `--no-amp` |

`--edge-width` is the width of an edge in $b$ that the forward model
assumes. It must include the FBP blur. With the kernel width alone (1.5), the
true wavefront set fits $b$ about 2.7× worse than an unrestricted $q$, and
the network widens its output to compensate. At 2.6 the true support fits as
well as any $q$ (no background, n=256).

Speed (RTX 6000 Ada, bf16, checkpointing):

| grid | batch | speed | GPU memory |
|---|---|---|---|
| n=96, K=32 | 8 | 0.9 s/step | 6 GB |
| n=256, K=64 | 2 | 3.2 s/step | 17.4 GB peak (23.8 GB reserved) |

The n=256 row is the production run of the slides (`outputs/train/n256_lam1e-4`,
λ = 1e-4, 1500 steps, 1.4 h): 3.20 s/step is the mean logged `sec_per_step`
of its `log.jsonl` (all six n=256 runs log 3.20–3.23 s/step), and the
benchmark `research/presentation/bench_scaling.py` (`assets/scaling.json`,
median of 8 steps, `torch.cuda.max_memory_allocated`) gives 3.21 s/step and
17.4 GB. An earlier measurement here said 4.2 s/step, ~16 GB. Validation of
that run at step 1500 (4 held-out phantoms, `val_size` 4): AUC 0.979 for $m$ (best 0.980 at step
750), 0.973 for $y_W$ (formerly $b$), 0.921 for $C^{-1}y_M$; ridge F 0.49
vs 0.77 for ridges of $y_W$; spatial-only F 0.95 vs 0.97.

Data generation costs 0.1–0.25 s per step. A forward and backward pass at
n=512, K=128 does not fit in 48 GB even at batch 1.

**Parallel sweeps:**
`COMMON_ARGS="--batch 8 --steps 2000" PYTHON=... bash experiments/sweep.sh NAME "--lam 3e-4" "--lam 1e-3" ...`
runs one configuration per GPU.

**Choosing λ.** $D$ is a mean over all lifted voxels and $E$ an integral, so
λ must be small (`loss/README.md`). An edge covers a fraction ∝ $1/n$ of the
voxels, so the best λ scales roughly like $1/n$. With the image term at
n=96, λ ≥ 3e-3 collapses to $m\equiv0$, and so does ν = 2 at λ = 3e-4.

### Results with the image-side data term (2026-10-02)

At n=96, the learning-free $b$ alone reaches a validation AUC of 0.963,
against 0.897 for $C^{-1}c$, and its ridges are thin curves with no tangent
lines.

λ sweep at n=96 (image term, batch 8, 2000 steps, `outputs/train/lam_*`):

| λ | peak val AUC (step) | final val AUC |
|---|---|---|
| 3e-4 | 0.970 (750) | 0.961 |
| 1e-3 | 0.953 (1500) | 0.953 |
| 3e-3, 1e-2 | collapse ($m\approx0$) | stopped |
| σ = 1.0, λ = 3e-4 | 0.976 (500) | (stopped at about 1500) |

The LPD output has no tangent-line streaks. In these runs the AUC drifts down
after its peak, because the model's edge width was too narrow (1.5 cells
instead of the 2.6 in $b$). The network widened $q$ to fill the gap, and also
fitted part of the smooth background's response. `--edge-width 2.6` fixes the
first problem. The second shrinks with resolution: for a smooth function the
fine-scale response is $O(\sigma)$, Theorem 5.1. Diagnostic: best D with
$q\ge0$ restricted to the true support vs. unrestricted, 4 validation
phantoms:

| n | edge width | background | true support / unrestricted |
|---|---|---|---|
| 96 | 1.5 | 0.3 | 0.0148 / 0.0042 |
| 96 | 1.5 | 0 | 0.0054 / 0.0040 |
| 256 | 1.5 | 0 | 0.0040 / 0.0029 |
| 256 | 2.6 | 0 | 0.0039 / 0.0040 |
| 256 | 2.6 | 0.3 | 0.0071 / 0.0038 |

For comparison, under the sinogram term the main network reaches D = 0.013,
while the truth with a fitted contrast scores 0.055, no better than a
31-cell tangent smear of it (0.054).

### Results with the sinogram data term (2026-10-02)

Validation AUC on 8 held-out phantoms (learning-free baseline $C^{-1}c$: 0.897):

| run | steps | val AUC | $E$ |
|---|---|---|---|
| λ = 1e-4 | 800 | 0.921 | 9.2 |
| λ = 3e-4 | 800 | 0.920 | 7.1 |
| λ = 1e-3 | 800 | 0.923 | 5.0 |
| λ = 1e-3, main (`outputs/train/main`) | 4000 | 0.914 (0.918 at 500) | 6.2 |

The sweep runs used $J_{\max}=2$, no forward-model normalization and a
hard-edged background. The main run includes all three fixes, so the rows are
not strictly comparable.

**The LPD beats the baseline consistently but modestly, and its output is
still streaky.** The diagnosis is decisive: on held-out phantoms, the
network's objective $D+\lambda E$ is about 3× *lower* than the objective at
the ground truth (0.014–0.023 vs 0.043–0.065), even with a separately fitted
contrast per shape. The network optimizes well, but **the MAP of the current
model is not the true wavefront set**. The forward model (Def. 5.4) does not
reproduce the measured coefficients accurately: its residual at the truth,
$D\approx0.05$, is far above what the network reaches by fitting
streak-like structure. The next step is a better data model, not a bigger
network: a multi-scale or curvature-adapted analyzer, modelling of
interference between singularities on the same ray, and per-edge contrast
(background §9).

### No contrast output, turning penalty and λ (2026-10-06)

The amplitude is now $q=a/a_\mathrm{ref}$ with no contrast $J$ (`--contrast`
restores $q=Jm$; old configs load with it). All runs: n=256, image data
term with image input, batch 2, 1500 steps, λ = 1e-4 unless stated.
Metrics from `python -m experiments.turning_diagnostics RUN ...` on the same
8 validation phantoms for every run: ROC AUC and ridge F of m; θ-concentration
= share of a's mass within ±3π/32 of its peak angle; "b vs true" =
correlation of b with the phantom's b = aκ (`lifted_flux`); κ-corr =
correlation of the centre-line curvature Σb/Σa with the truth.

| run | K | J | ℓθ | AUC | F | θ-conc | b vs true | κ-corr |
|---|---|---|---|---|---|---|---|---|
| `n256_lam1e-4` (old main) | 64 | yes | 0.1 | 0.983 | 0.53 | 0.59 | 0.14 | 0.27 |
| `n256_K64_lam1e-4_noJ_xi0.1` | 64 | no | 0.1 | 0.985 | 0.77 | 0.65 | 0.41 | 0.62 |
| `n256_K64_lam1e-4_noJ_xi0.3` | 64 | no | 0.3 | 0.985 | 0.77 | 0.71 | 0.69 | 0.83 |
| `n256_lam1e-4_J_xi0.1` | 32 | yes | 0.1 | 0.978 | 0.87 | 0.71 | 0.61 | 0.78 |
| `n256_lam1e-4_J_xi0.3` | 32 | yes | 0.3 | 0.982 | 0.86 | 0.63 | 0.40 | 0.44 |
| `n256_lam1e-4_noJ_xi0.1` | 32 | no | 0.1 | 0.985 | 0.88 | 0.76 | 0.68 | 0.82 |
| **`n256_lam1e-4_noJ_xi0.3`** (slides) | 32 | no | 0.3 | **0.987** | **0.89** | **0.77** | 0.68 | **0.85** |
| `n256_lam1e-4_noJ_xi1.0` | 32 | no | 1.0 | 0.987 | 0.86 | 0.74 | 0.58 | 0.81 |
| `n256_lam1e-4_noJ_elastica0.05` | 32 | no | elastica β=0.05 | 0.984 | 0.85 | 0.70 | 0.59 | 0.78 |

* **Removing J** helps at both K: F +0.24 at K=64, +0.01–0.03 at K=32, and
  b turns from a dipole across each edge (the "three lines") into the true
  aκ. With J and ℓθ = 0.3, b is mostly dipole again.
* **K = 32 beats K = 64** on ridge F (0.89 vs 0.77) at equal AUC and half
  the cost. Not yet understood; the 2-cell θ-resolution of the outputs is
  twice as wide in angle at K = 32.
* **Turning penalty:** ℓθ = 0.3 is best. ℓθ = 1.0 and the elastica shrink the
  true curvature (median curvature error 2.1 and 2.6 against 1.4; typical
  |κ| ≈ 2.8) and lower F; the elastica also rounds corners.
* A correctly blurred curved current also has div τ = 0, so the correlation
  of X₁a with −∂θb is high for good runs too (0.8–0.9); use "b vs true".

**λ sweep** (no J, ℓθ = 0.3, K = 32; `n256_noJ_xi0.3_lam*`, λ = 1e-4 is
`n256_lam1e-4_noJ_xi0.3`):

| λ | 0 | 1e-5 | 3e-5 | 5e-5 | **1e-4** | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.50 | 0.973 | 0.966 | 0.976 | 0.987 | 0.991 | 0.986 | 0.979 | 0.960 |
| F | 0.01 | 0.84 | 0.86 | 0.87 | **0.89** | 0.64 | 0.47 | 0.38 | 0 |
| b vs true | – | 0.20 | 0.67 | 0.70 | 0.68 | 0.15 | 0.12 | – | – |

λ = 0 diverges: b and c appear in no term, grow without bound (E ≈ 2e15)
and the network ends at a = 0. Below 1e-4 the prior is too weak to shape b
(at 1e-5, |div τ| is 3.5× higher). Above it the bands widen in x and θ,
b becomes a dipole again, weak edges fade (3e-3: m < 0.1 everywhere) and
the output collapses at 1e-2. AUC peaks at 3e-4 because it rewards a wide
soft band; F and the curvature metrics track the intended output.
Figures: `outputs/train/{turning_comparison,j_comparison,lambda_sweep_noJ_xi0.3}.png`.

## `inspect_lpd`: intermediate results of a trained network

```bash
python -m experiments.inspect_lpd --checkpoint outputs/train/run/checkpoint.pt --sample 0
```

Runs the network on one of the fixed validation phantoms with
`LiftedLPD(..., return_intermediates=True)` and writes the results to
`inspect_<sample>/` next to the checkpoint:

* `intermediates.npz`: inputs $c,b$; the primal/dual states after every
  iteration; the readout $m_i$ (the output head applied to the $i$-th primal
  state); the outputs $a,b,c,J,m,q=Jm$; the prediction $\hat b$; the prior's
  length density and $|\operatorname{div}\tau|$;
* `iterations.png`: per iteration, $m_i$, its ridges, $|p_i|$, and
  $|C^{-1}d_i|$;
* `final.png`: data vs. prediction, residuals (unexplained and
  overpredicted), $m$ and its ridges, $J$, the prior densities, and the
  orientation error at the true wavefront set; `info.json` has the numbers.

**What it shows** (n=256, λ = 1e-4, step 1000):

* $J$ sits at its bound $J_{\max}=1$ on every edge while $m\le0.47$, so the
  amplitude has to come from the width of $m$. Raising $J_{\max}$ to 4 made
  the run collapse to $m\equiv0$ (the prior cost of $m$ then dominates).
* Residuals concentrate at crossings. There the model, which adds
  magnitudes, overpredicts, whereas $|W\,\mathrm{FBP}\,y|$ of two crossing
  edges interferes.
* The orientation of $m$ is unbiased but noisy (median error 4 θ-cells at
  K=64, vs. 1 for $b$). The image term is almost blind to small θ shifts:
  the angular response is ~90° wide. So θ is set by the prior's
  horizontality, which is weak at small λ.

## `walnut_wavefront`: the FIPS walnut

```bash
python -m experiments.walnut_wavefront                                  # learning-free
python -m experiments.walnut_wavefront --checkpoint outputs/train/run1/checkpoint.pt --finetune 200
```

The 1200-projection fan-beam scan is rebinned to parallel geometry
(`FanToParallel`, Prop. 4.7) and analyzed by $M$. It shows the learning-free
$C^{-1}c$ and, with a checkpoint, the LPD membership. `--finetune N` continues
minimizing the unsupervised objective on the walnut itself, which is legitimate
because the objective needs no labels. The FBP of the rebinned data and the
dataset's reference FBP are shown for comparison only. Lines outside the
measured field of view ($|s|>0.98$) are masked: their truncation is a jump in
the data and would appear as a spurious wavefront set on a circle.

The FBP of the rebinned data matches the reference in orientation, scale and
detail. That is an end-to-end check of the fan-beam relation, the rebinning
and the geometry calibration on real data.

### Results (2026-10-02)

`outputs/walnut/` (learning-free, n=192) and `outputs/walnut_lpd/` (main
checkpoint, n=96, plus 200 unsupervised fine-tuning steps: $D$ 0.027 → 0.026).
The field-of-view mask removes the spurious ring. Both estimates are dominated
by tangent-line streaks of the bright shell, the same failure as on
phantoms. The LPD suppresses them somewhat but does not recover the envelope.
**A usable walnut wavefront set needs the improved data model above.**

## Numbers and logs moved from `research/background.md` (2026-10-06)

The theory document no longer quotes measurements; they are collected here
with their provenance. Unless a test asserts a value, it was **measured once**
and is not regression-checked.

**Numbering.** References above use the numbering of `background.md` before
the 2026-10-06 revision. Changes: Lemma 6.1 → Lemma 2.6; Prop 2.7, 2.8,
Cor 2.9 → Prop 2.8, 2.9, Cor 2.10; Prop 3.6 + Thm 3.8 → Prop 3.7; Thm 3.9 →
Thm 3.8; Prop 4.7 (fan beam) → Prop A.1; Thm 4.8 → Thm 4.7; Prop 5.2 (noise)
removed; Prop 5.3 → Prop 5.2; Def 5.4 (sinogram model, $\mathcal D_Y$ on
$y_M=|My|$, formerly $c$) → Def 5.3; Remark 5.5 → Def 5.4 (image-side model,
$y_W$, formerly $b$); Props 6.2–6.4 → 6.1–6.3; Prop 7.2 → Prop 7.1. The
turning scale `--xi` is $\ell_\theta$ in the text.

| claim (old location) | value | provenance |
|---|---|---|
| fibre response vs measured $t$-profiles at true WF points (§5.2) | median correlation 0.993; FWHM ≈ 30 cells on a 96-cell grid | `SinogramAnalyzer.fibre_response`; measured once |
| Def 5.4 (now 5.3) with the true amplitude at 2-cell resolution, ellipse (§5.2) | correlation 0.85; relative $\ell^1$ residual 0.56 (zero model: 1) | `tests/test_lpd.py` (n=64) asserts only residual < 0.65; 0.85 measured once |
| same $q$ smeared along tangent lines over 15 / 31 / 63 cells (§5.2) | residual 0.55 / 0.56 / 0.62 | test asserts only the ordering; values measured once |
| amplitude weighted by $\sqrt{\text{radius of curvature}}$ (§5.1) | no improvement of the Def 5.4 fit | measured once |
| per-voxel mixture posterior > 0.5 (§5.3, "milestone 1") | 63% of the image | `milestone1_canonical_relation` table above |
| image-side model: true support vs unrestricted $q$ (Remark 5.5) | equal fit at edge width 2.6; under Def 5.4 the truth fits 3–4× worse than the network's streaks | diagnostic table above |
| boundary mass of a closed curve with $c$ dropped (Prop 6.3(1), now 6.2(1)) | 15.1 instead of 1.2 | `milestone1_canonical_relation` table above |
| mass bias with $c$ kept, $\zeta=2$ (Prop 6.3(2), now 6.2(2)) | ≈1.5% measured at $K=48$; formula gives ≈3% at $K=48$, ≈6% at $K=32$ | measured once |
| boundary mass of a mollified closed curve (§6.3) | 10.6 (2nd order, 1-cell blur) vs 0.26 (4th order, 2-cell blur) | not asserted by `tests/test_regularization.py`; measured once |
| walnut geometry calibration (§4.3) | correlation 0.9999 with the system matrix | `tests/test_walnut.py` asserts > 0.999 |

**Status of the experiments (2026-10-02), formerly background §9.** Trained
by Prop 7.1 with the sinogram term, the LPD improves the ROC AUC over the
learning-free $C^{-1}y_M$ (0.914–0.923 vs 0.897), but its output keeps the
tangent-line streaks. On held-out phantoms the network's objective is about
3× lower than the objective at the ground truth: the solver works, but the
MAP of the local sinogram model is not the true wavefront set, and its
residual at the truth is far above the noise.

**Update, formerly background §9.** The full-aperture data term removes the
streaks. Its learning-free coefficients $y_W$ alone reach AUC 0.96–0.975
(vs 0.90 for $C^{-1}y_M$), and the LPD trained on it reaches 0.970–0.976 at
n=96. Thin estimates are ridges (non-maximum suppression across the curve and
in θ), scored by tolerance-based precision and recall (`evaluation/ridges.py`).
The remaining model error is the response to smooth intensity variation,
which decays with resolution (background Theorem 5.1).
