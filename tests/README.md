# tests/

Run from the repository root with `pytest`. `pyproject.toml` adds the root to
`sys.path`. The tests need the `wfc` environment (torch, ODL, ASTRA).

| File | What it checks |
|---|---|
| `test_canonical_relation.py` | pointwise round trip; exact adjoint (dot-product test); a bump at $x_0$ lands at $(x_0\cdot\omega, x_0\cdot\omega^\perp)$ in every slice; inverse ≈ adjoint ≈ $C^{-1}$, norm preserved in the disk |
| `test_radon.py` | ODL convention $s=x\cdot\omega(\varphi)$ and index order; $Rf(s,\varphi+\pi)=Rf(-s,\varphi)$; autograd backward = matrix transpose $(w_X/w_Y)R^*$, not ODL's weighted adjoint |
| `test_microlocal.py` | integration test on a noise-free phantom: on smooth edges $My$ peaks at the $t$ predicted by $C$ (unbiased, median error ≤ 1 cell); AUC > 0.9 on both the sinogram and image sides; noise std and posterior |
| `test_regularization.py` | flux integrals equal length and total turning; length term ≈ analytic ($\zeta=1$) with a small bias for $\zeta=2$; endpoint mass ≈ 0 / 2 / 4 / 8 for closed / half / two arcs / square without corners; dropping $c$ breaks closedness; elastica gradients finite |
| `test_data_discrepancy.py` | gradient decreases membership on noise and increases it on edges; posterior monotone; masking |

`test_microlocal.py` and `test_radon.py` are the slowest (ODL projections and a
96-filter bank).
