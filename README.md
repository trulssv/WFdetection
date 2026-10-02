# WFclassification

Unsupervised, indirect detection of the **wavefront set of a 2D image directly
from tomographic data**. The image wavefront set $\mathrm{WF}(f)$ is treated as a
union of curves in the lifted space $X = \Omega\times S^1$ (position × normal
direction). A Bayesian curve prior on that set is combined with an unsupervised
data term on the sinogram, and the two are linked through the **canonical
relation** of the ray transform.

## Pipeline

```
           sinogram y  ──M──▶  c(s,φ,t)            lifted sinogram coefficients
                                  │  data term D(Cu; c)
image WF field u(θ,x) ──C──▶  v(s,φ,t)            predicted sinogram WF
           │  curve prior R(u)
```

| Symbol | Meaning | Code |
|---|---|---|
| $u(\theta, x_1, x_2)$ | lifted image WF field on $\Omega\times\mathbb{RP}^1$ | layout `(B, C, n_theta, n, n)` |
| $C$ | canonical relation, $(Cu)(s,\varphi,t)=u(s\omega(\varphi)+t\omega^\perp(\varphi),\varphi)$ | [operators/canonical_relation.py](operators/canonical_relation.py) |
| $M$ | fixed directional filter bank on the sinogram | [operators/microlocal.py](operators/microlocal.py) |
| $R$ | ODL/ASTRA ray transform as a torch module | [operators/radon.py](operators/radon.py) |
| $\mathcal R(a,b,c)$ | curve prior (length + turning + endpoints) | [loss/regularization.py](loss/regularization.py) |
| $D$ | mixture pseudo-likelihood of coefficients | [loss/data_discrepancy.py](loss/data_discrepancy.py) |

## Repository layout

| Directory | Contents |
|---|---|
| [geometry/](geometry/) | lifted grids, coordinate conventions, ODL geometries |
| [operators/](operators/) | canonical relation $C$, microlocal analyzer $M$, ray transform |
| [loss/](loss/) | curve prior and unsupervised data term |
| [data/](data/) | phantoms with analytic wavefront sets (walnut loader planned) |
| [evaluation/](evaluation/) | metrics against analytic wavefront sets |
| [experiments/](experiments/) | runnable studies; outputs go to `outputs/` |
| [tests/](tests/) | unit and integration tests |

Each directory has its own README with details.

## Setup

ASTRA and ODL are not yet reliable on Python 3.13, and the RTX 50xx GPUs
(sm_120) need a CUDA ≥ 12.8 build of torch:

```bash
conda create -n wfc python=3.12
conda activate wfc
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install odl astra-toolbox matplotlib scipy scikit-image pytest
```

Run everything from the repository root:

```bash
pytest                                              # tests
python -m experiments.milestone1_canonical_relation # milestone 1 study
```

## Conventions (summary)

* $\Omega=[-1,1]^2$, discretized by cell centres; arrays are indexed `[i1, i2]`
  with $x_1$ along the first axis (ODL's convention). For plotting with
  `imshow`, transpose and use `origin="lower"`.
* Orientations live on $\mathbb{RP}^1$: $\theta_k=\pi k/K$. For real-valued
  images $(x,\xi)\in\mathrm{WF}\iff(x,-\xi)\in\mathrm{WF}$, so nothing is lost.
* Parallel beam: $Rf(s,\varphi)=\int f(s\omega(\varphi)+t\omega^\perp(\varphi))\,dt$
  with $\omega=(\cos\varphi,\sin\varphi)$ and $\omega^\perp=(-\sin\varphi,\cos\varphi)$.
  This is ODL's convention (verified in `tests/test_radon.py`).
* Objects must lie inside the unit disk, where $C$ is unitary.

## Roadmap

1. **Milestone 1** *(done)*: numerical check of $C$, $M$, the prior, and the
   sign conventions on phantoms with analytic WF. Results and findings are in
   [experiments/README.md](experiments/README.md). In short: the theory checks
   out; the current needs a normal component $c$ at finite resolution; local
   sinogram analysis leaves the position along the ray ambiguous, so the prior
   has to resolve it.
2. **Milestone 2**: LPD-style network with primal on $(\theta,x)$, dual on
   $(\varphi,s,t)$, and $C$, $C^{T}$ between them; unsupervised training with
   $D + \lambda\mathcal R$; angle-split self-supervision for validation.
3. **Milestone 3**: sparse-view and limited-angle settings (where the prior
   matters), an analyzer for sparse angles, and comparison against FBP edge
   detection.
4. **Milestone 4**: FIPS walnut data (fan beam, using the fan-to-parallel line
   reparametrization).
