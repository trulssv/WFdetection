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

Minimizes $D(\mathcal R_tC(Jm);c)+\lambda E(a,b,c)$ over synthetic data
(`data/synthetic.py`; random cartoons, smooth backgrounds, 1% noise, 3n dense
angles). This is amortized MAP estimation
([research/background.md](../research/background.md), Prop. 7.2); no labels
are used. Every `--val-every` steps it reports, on 8 fixed held-out phantoms,
the ROC AUC of the membership $m$ against the analytic wavefront sets, next
to the AUC of the learning-free $C^{-1}c$. It also writes `val_<step>.png`,
`log.jsonl` and `checkpoint.pt`.

Main options: `--n 96 --n-theta 32` (grid), `--lam` (prior weight), `--xi`,
`--nu`, `--zeta` (prior), `--n-iter 5 --hidden 24` (network), `--batch 2`,
`--lr 1e-3` (cosine schedule), `--noise`, `--background`, `--no-amp`.
Speed: about 0.7 s/step on an RTX 5070 Laptop GPU (bf16, checkpointing).

**Choosing λ.** $D$ is a mean over all lifted voxels and $E$ an integral, so
λ must be small (`loss/README.md`). λ = 0.01 collapses to $m\equiv0$.

### Results (2026-10-02)

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
