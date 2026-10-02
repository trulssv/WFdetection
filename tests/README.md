# tests/

Run from the repository root with `pytest`. `pyproject.toml` adds the root to
`sys.path`. The tests need the `wfc` environment (torch, ODL, ASTRA).

| File | What it checks |
|---|---|
| `test_canonical_relation.py` | pointwise round trip; exact adjoint (dot-product test); a bump at $x_0$ lands at $(x_0\cdot\omega, x_0\cdot\omega^\perp)$ in every slice; inverse ≈ adjoint ≈ $C^{-1}$, norm preserved in the disk |
| `test_radon.py` | ODL convention $s=x\cdot\omega(\varphi)$ and index order; $Rf(s,\varphi+\pi)=Rf(-s,\varphi)$; autograd backward = matrix transpose $(w_X/w_Y)R^*$, not ODL's weighted adjoint |
| `test_microlocal.py` | integration test on a noise-free phantom: on smooth edges $My$ peaks at the $t$ predicted by $C$ (unbiased, median error ≤ 1 cell); AUC > 0.9 on both the sinogram and image sides; noise std and posterior |
| `test_regularization.py` | flux integrals equal length and total turning; length term ≈ analytic ($\zeta=1$) with a small bias for $\zeta=2$; endpoint mass ≈ 0 / 2 / 4 / 8 for closed / half / two arcs / square without corners; dropping $c$ breaks closedness; elastica gradients finite |
| `test_fan_beam.py` | fan line map vs ODL ray geometry; `line_to_fan` inverts both measurements; rebinning reproduces parallel data (< 2%); lifted fan relation moves a bump to the predicted $(\beta,u,t)$; predicted covector is normal to the singular curves of simulated fan data (< 5°) |
| `test_walnut.py` | ODL walnut geometry vs the dataset's system matrix (correlation > 0.999, scale = $L$); loader sanity. Skipped without the data |
| `test_lpd.py` | output shapes and parities ($a\ge0$, $b$ odd/$a,c$ even on the double cover); exact equivariance of the dual network under $\varphi$-shifts; forward model reproduces the measured streaks |
| `test_data_discrepancy.py` | gradient decreases membership on noise and increases it on edges; posterior monotone; masking |

`test_microlocal.py` and `test_radon.py` are the slowest (ODL projections and a
96-filter bank).
