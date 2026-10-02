# models/

The learned primal–dual (LPD) network, an amortized MAP solver for
$\min D(\mathcal R_tC(Jm);c)+\lambda E(a,b,c)$
([research/background.md](../research/background.md), §7).

## `lpd.py`: `LiftedLPD`

```
c (B,1,K,n,n) on (φ∈[0,π), s, t)
  └─ extend to φ∈[0,2π):  c(s,φ+π,t) = c(−s,φ,−t)
for i = 1..n_iter:
  d ← d + Γ_i([d, C p, c])          dual,   lifted sinogram of oriented lines (2K slices)
  p ← p + Λ_i([p, C⁻¹ d])           primal, lifted image Ω×S¹ (2K slices)
head(p) → (a, b, c, J) on S¹ → antipodal projection → fields on RP¹ (K slices)
```

| output | meaning | constraint |
|---|---|---|
| `a` | line density of the current $\tau=aX_1+cX_2+bX_3$ | $a=a_{\mathrm{ref}}\,\mathrm{softplus}\ge0$ (positively Legendrian, needed for Thm 3.9) |
| `b` | rotation rate $a\kappa$ | antipodally odd (Lemma 6.1) |
| `c` | normal component (blur) | antipodally even |
| `J` | contrast | $J_{\max}\,\mathrm{sigmoid}$, $J_{\max}=1$; bounded, because unbounded $J$ would make the prior free |
| `m` | membership $1-e^{-a/a_{\mathrm{ref}}}$, the classifier output | in $[0,1)$; contrast-weighted, $m\approx\min(1,q/J_{\max})$, since only $q=Jm$ is seen by the data (background §7.3) |

$a_{\mathrm{ref}}=1/(2\pi\sigma_x\sigma_\theta)$ with 2-cell blur, so a
well-resolved curve has $m\approx0.63$ on its centre line.

Design notes:

* **Double covers.** Both CNNs act on the double covers with plain periodic
  convolutions. On the $\varphi\in[0,\pi)$ quotient, the identification
  includes a reflection of the $(s,t)$-plane, so a "twisted" padding would
  be well defined only for reflection-symmetric kernels (§7.3).
* **Back-projection by $C^{-1}$**, not by the discrete adjoint: the transpose
  of bilinear resampling has moiré-like weights (`operators/README.md`).
* **Equivariance.** Rotations of the image by multiples of $\pi/K$ are cyclic
  $\varphi$-shifts of the dual variables (Prop. 4.6), so the dual CNNs are
  exactly equivariant. The primal CNNs are equivariant only approximately.
* **Memory.** Gradient checkpointing per iteration and bf16 autocast (in the
  training script). Defaults (n=96, K=32, 5 iterations, 24 hidden channels,
  283k parameters) need about 1.2 GB for a batch of 2.

## `layers.py`

`LiftedConv` (3D conv, periodic in the fibre, zero-padded in space),
`LiftedBlock` (residual CNN update), and the projections between $S^1$ and
$\mathbb{RP}^1$ (`s1_to_rp1_even`, `s1_to_rp1_odd`, `rp1_to_s1_even`,
`sinogram_rp1_to_s1`).
