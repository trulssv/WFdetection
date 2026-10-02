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
