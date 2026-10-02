# WFclassification

Unsupervised, indirect detection of the **wavefront set of a 2D image directly
from tomographic data**.

The wavefront set of a cartoon image is a union of *Legendrian curves* in the
cosphere bundle $X=\Omega\times S^1$ (position × normal direction). We
represent it as a 1-current on $X$ and place a Gibbs prior on it whose
energy is length plus endpoint count:
$E=\mathbf M_g(T)+\nu\,\mathbf M(\partial T)$. The data are compared with it
through the canonical relation $C$ of the ray transform, a strict
contactomorphism between the lifted image and the lifted sinogram. An
unrolled learned primal–dual network is trained, without labels, as an
amortized MAP solver.

**Start with [research/background.md](research/background.md)**: it derives
every component from first principles, with definitions, theorems and
proofs, and the code cites its numbering.

## Pipeline

```
sinogram y ──(fan beam: rebin, Prop 4.7)──▶ parallel y ──M──▶ c(φ,s,t)         lifted sinogram coefficients
                                                               │  D(R_t C q; c)   forward-model data term (Def 5.4)
LPD network: dual on (φ,s,t)  ⇄ C, C⁻¹ ⇄  primal on (θ,x)       │
                                                               ▼
                                     current τ = aX₁+cX₂+bX₃, contrast J, membership m
                                                               │  λ E(a,b,c)      curve prior (Def 3.5)
```

| Symbol | Meaning | Code |
|---|---|---|
| $C$ | canonical relation $(x,\theta)\mapsto(x\cdot\omega,\theta,x\cdot\omega^\perp)$; $C^*(ds-t\,d\varphi)=\alpha$ | [operators/canonical_relation.py](operators/canonical_relation.py) |
| $C_{\mathrm{fan}}$, rebinning | fan beam = parallel beam in other line coordinates | [operators/fan_beam.py](operators/fan_beam.py) |
| $M$, $r(t,t')$ | directional filter bank on the sinogram, and its fibre response | [operators/microlocal.py](operators/microlocal.py) |
| $R$ | ODL/ASTRA ray transform as a torch module | [operators/radon.py](operators/radon.py) |
| $E$ | curve prior: mass + boundary mass of the current | [loss/regularization.py](loss/regularization.py) |
| $D$ | Laplace forward-model data term (and the mixture baseline) | [loss/data_discrepancy.py](loss/data_discrepancy.py) |
| LPD | learned primal–dual on the double covers | [models/lpd.py](models/lpd.py) |

## Repository layout

| Directory | Contents |
|---|---|
| [research/](research/) | background and methods document |
| [geometry/](geometry/) | lifted grids, coordinate conventions, ODL geometries (parallel, fan) |
| [operators/](operators/) | canonical relations (parallel, fan), rebinning, microlocal analyzer, ray transform |
| [loss/](loss/) | curve prior and data terms |
| [models/](models/) | LPD network and lifted convolutions |
| [data/](data/) | phantoms with analytic wavefront sets, synthetic training data, walnut loader |
| [evaluation/](evaluation/) | metrics against analytic wavefront sets, plotting |
| [experiments/](experiments/) | milestone-1 study, training, walnut analysis; outputs go to `outputs/` |
| [tests/](tests/) | unit and integration tests (31) |

Each directory has its own README.

## Setup

ASTRA and ODL are not yet reliable on Python 3.13, and RTX 50xx GPUs (sm_120)
need a CUDA ≥ 12.8 build of torch:

```bash
conda create -n wfc python=3.12
conda activate wfc
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install odl astra-toolbox matplotlib scipy scikit-image pytest
```

For the walnut data, see [data/README.md](data/README.md) (about 27 MB from
Zenodo). Run everything from the repository root:

```bash
pytest                                                   # tests
python -m experiments.milestone1_canonical_relation      # learning-free check of the theory
python -m experiments.train_lpd --lam 1e-4 --out outputs/train/run1
python -m experiments.walnut_wavefront --checkpoint outputs/train/run1/checkpoint.pt
```

## Conventions (summary)

* $\Omega=[-1,1]^2$, discretized by cell centres; arrays are indexed `[i1, i2]`
  with $x_1$ along the first axis (ODL's convention). For plotting with
  `imshow`, transpose and use `origin="lower"`. The walnut figures use
  `origin="upper"` to match the dataset's documentation.
* Orientations live on $\mathbb{RP}^1$: $\theta_k=\pi k/K$. The flux component
  $b$ is antiperiodic. The network uses the double cover ($2K$ slices).
* Parallel beam: $Rf(s,\varphi)=\int f(s\omega(\varphi)+t\omega^\perp(\varphi))\,dt$,
  ODL's convention (verified in `tests/test_radon.py`).
* Objects must lie inside the unit disk, where $C$ is unitary.

## Status

1. **Milestone 1** *(done)*: numerical check of $C$, $M$ and the prior on
   phantoms with analytic wavefront sets
   ([experiments/README.md](experiments/README.md)).
2. **Theory** *(done)*: [research/background.md](research/background.md).
3. **Fan beam and walnut** *(done)*: fan-beam canonical relation and
   rebinning; walnut geometry calibrated against the dataset's system matrix.
4. **LPD and unsupervised training** *(first results)*: validation AUC
   0.914–0.923 vs 0.897 for the learning-free baseline, but the output keeps
   tangent-line streaks. Diagnosis: the network's objective is about 3× lower
   than the objective at the ground truth, so the data model (Def. 5.4), not
   the solver, is the bottleneck ([experiments/README.md](experiments/README.md)).
5. **Next**: sparse and limited angles (where the prior matters most), a
   multi-scale data term, and angle-split self-supervision
   (background §9).
