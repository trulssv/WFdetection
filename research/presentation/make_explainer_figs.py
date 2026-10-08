"""Explainer figures and ranking metrics for the slides (no retraining).

    CUDA_VISIBLE_DEVICES=0 python research/presentation/make_explainer_figs.py [--only cr,analyzer,pr]

Writes to research/presentation/assets/:

    fig_canonical_relation.png  analytic: one ellipse edge, its ray (s, t) and its sinogram curve
    fig_sinogram_analyzer.png   the repo's SinogramAnalyzer on a simulated one-ellipse sinogram
    fig_pr_curves.png           precision-recall of C^{-1}y_M, y_W and m on the validation set

and prints the validation metrics used on the slides: ROC AUC, average precision
(AP) and hard-negative AUC (negatives within 6 px of the true wavefront set),
per sample and as means.  The model and validation set are those of
make_synthetic_figs.py (same checkpoint, torch seed, build, rng 12345).
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "research" / "presentation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse as EllipsePatch
from scipy.ndimage import binary_dilation, distance_transform_edt

from data.phantoms import Ellipse, Phantom
from evaluation.metrics import label_masks, roc_auc
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from operators.canonical_relation import CanonicalRelation
from operators.microlocal import SinogramAnalyzer, half_ramp
from operators.radon import RayTransform

import make_synthetic_figs as MS

ASSETS = ROOT / "research" / "presentation" / "assets"
DPI = 150
FS = 13

# one ellipse used by the two explainer figures
ELL = dict(center=(0.15, -0.1), axes=(0.55, 0.3), angle=0.5)
PHI0 = 1.15            # angle of the ray / edge normal shown


# --------------------------------------------------------------------------- geometry
def _rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def support(phi):
    """Support function h(phi) = max_{x in E} x . omega(phi) and touching point x(phi)."""
    phi = np.asarray(phi, float)
    w = np.stack([np.cos(phi), np.sin(phi)], -1)
    R = _rot(ELL["angle"])
    r1, r2 = ELL["axes"]
    wl = w @ R                                   # omega in the ellipse frame
    nrm = np.sqrt((r1 * wl[..., 0])**2 + (r2 * wl[..., 1])**2)
    xl = np.stack([r1**2 * wl[..., 0], r2**2 * wl[..., 1]], -1) / nrm[..., None]
    x = np.asarray(ELL["center"]) + xl @ R.T
    return np.sum(x * w, -1), x


def h_and_slope(phi):
    h, x = support(phi)
    wp = np.stack([-np.sin(phi), np.cos(phi)], -1)
    return h, np.sum(x * wp, -1), x            # s = h, t = h'(phi) = x . omega_perp


def fig_canonical_relation():
    s0, t0, x0 = h_and_slope(np.array(PHI0))
    w, wp = np.array([math.cos(PHI0), math.sin(PHI0)]), np.array([-math.sin(PHI0), math.cos(PHI0)])
    print(f"canonical relation fig: phi0 = theta = {PHI0:.3f}, x = ({x0[0]:.3f}, {x0[1]:.3f}), "
          f"s = {s0:.3f}, t = {t0:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    a = ax[0]
    a.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.6", lw=1, ls="--"))
    a.add_patch(EllipsePatch(ELL["center"], 2 * ELL["axes"][0], 2 * ELL["axes"][1],
                             angle=math.degrees(ELL["angle"]), facecolor="#e8eef6",
                             edgecolor="#16324f", lw=2))
    L = 1.15
    ray = np.array([s0 * w + u * wp for u in (-L, L)])
    a.plot(ray[:, 0], ray[:, 1], color="#c2571a", lw=2, label="ray (tangent to the edge)")
    foot = s0 * w
    a.annotate("", foot, (0, 0), arrowprops=dict(arrowstyle="-|>", color="#2a7f3f", lw=2))
    a.annotate("", x0, foot, arrowprops=dict(arrowstyle="-|>", color="#7b3294", lw=2))
    a.text(*(0.5 * foot + 0.07 * wp), "$s$", color="#2a7f3f", fontsize=FS + 3, ha="center", va="center")
    a.text(*(foot + 0.5 * t0 * wp + 0.08 * w), "$t$", color="#7b3294", fontsize=FS + 3,
           ha="center", va="center")
    a.annotate("", x0 + 0.32 * w, x0, arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    a.text(*(x0 + 0.40 * w), r"$\omega(\theta)$", fontsize=FS + 1, ha="center", va="center")
    a.plot(*x0, "o", color="k", ms=7)
    a.text(*(x0 + 0.06 * w - 0.10 * wp), "$x$", fontsize=FS + 2, ha="center", va="center")
    a.plot(0, 0, "+", color="k", ms=10)
    a.text(0.03, -0.07, "0", fontsize=FS)
    a.set_xlim(-1.1, 1.1), a.set_ylim(-1.1, 1.1), a.set_aspect("equal")
    a.set_xlabel("$x_1$", fontsize=FS), a.set_ylabel("$x_2$", fontsize=FS)
    a.set_title(r"Image: edge point $x$ with normal $\omega(\theta)$", fontsize=FS)
    a.legend(loc="lower left", fontsize=FS - 2, frameon=False)
    a.tick_params(labelsize=10)

    b = ax[1]
    phi = np.linspace(0, math.pi, 400)
    h, _, _ = h_and_slope(phi)
    hm, _, _ = h_and_slope(phi + math.pi)
    b.plot(h, phi, color="#16324f", lw=2, label=r"$s=h(\varphi)$")
    b.plot(-hm, phi, color="#16324f", lw=2, ls=":", label=r"$s=-h(\varphi+\pi)$")
    dp = 0.35
    pp = np.array([PHI0 - dp, PHI0 + dp])
    b.plot(s0 + t0 * (pp - PHI0), pp, color="#7b3294", lw=2.5,
           label=r"tangent, slope $t$")
    b.plot(s0, PHI0, "o", color="k", ms=7)
    b.annotate(r"$(s,\varphi=\theta)$" + "\n" + rf"$t=ds/d\varphi={t0:.2f}$", (s0, PHI0),
               (s0 - 0.75, PHI0 - 0.5), fontsize=FS,
               arrowprops=dict(arrowstyle="-", color="0.3"))
    b.set_xlim(-1.45, 1.2), b.set_ylim(0, math.pi)
    b.set_yticks([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi])
    b.set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    b.set_xlabel("detector position $s$", fontsize=FS), b.set_ylabel(r"angle $\varphi$", fontsize=FS)
    b.set_title("Sinogram: the edge is a curve, its slope is $t$", fontsize=FS)
    b.legend(loc="lower left", fontsize=FS - 2, frameon=True, framealpha=0.9)
    b.tick_params(labelsize=10)
    fig.suptitle(r"$C(x,\theta)=(x\cdot\omega(\theta),\ \theta,\ x\cdot\omega^\perp(\theta))=(s,\varphi,t)$",
                 y=0.03, fontsize=FS + 4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ASSETS / "fig_canonical_relation.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- analyzer
@torch.no_grad()
def fig_sinogram_analyzer(dev):
    n = 128
    grid = LiftedGrid(n, 32)
    geom = parallel_geometry(3 * n, 2 * n)
    R = RayTransform(image_space(n), geom)
    phis, s_det = sinogram_axes(geom)
    M = SinogramAnalyzer(grid, phis, s_det).to(dev)
    f = Phantom([Ellipse(ELL["center"], ELL["axes"], ELL["angle"], 1.0)]).render(n)
    y = R(torch.as_tensor(f, dtype=torch.float32)[None, None]).to(dev)
    torch.manual_seed(0)
    y = y + 0.01 * y.abs().max() * torch.randn_like(y)          # noise as in data/synthetic.py
    yl = half_ramp(y, M.ds)[0, 0].cpu().numpy()                 # Lambda^{1/2} y on (phi, s)
    sig = M.filter_native(y)[0][:, M.pad:-M.pad].cpu().numpy()  # (n_t, n_phi, n_det), signed
    ts = grid.coords()
    s0, t0, _ = h_and_slope(np.array(PHI0))
    k0 = int(np.argmin(np.abs(phis - PHI0)))
    # detector cell at the singular curve: the largest |response| of the matched kernel near s0
    near = np.where(np.abs(s_det - s0) <= 2 * M.ds)[0]
    j0 = int(np.argmin(np.abs(ts - t0)))
    i0 = int(near[np.argmax(np.abs(sig[j0, k0, near]))])
    prof = sig[:, k0, i0]
    absp = np.abs(prof)
    jmax = int(np.argmax(absp))
    half = absp[jmax] / 2
    above = np.where(absp >= half)[0]
    # contiguous run around the peak, linear interpolation at the crossings
    lo, hi = jmax, jmax
    while lo - 1 >= 0 and absp[lo - 1] >= half:
        lo -= 1
    while hi + 1 < len(absp) and absp[hi + 1] >= half:
        hi += 1
    tl = np.interp(half, [absp[lo - 1], absp[lo]], [ts[lo - 1], ts[lo]]) if lo > 0 else ts[0]
    th = np.interp(half, [absp[hi + 1], absp[hi]], [ts[hi + 1], ts[hi]]) if hi + 1 < len(ts) else ts[-1]
    fr = M.fibre_response().numpy()[:, j0]                      # predicted r(t, t0)
    print(f"analyzer fig: n={n}, sinogram {len(phis)}x{len(s_det)}, sigma_perp = {M.sigma_n:.4f}, "
          f"sigma_par = {M.sigma_t:.4f} (aspect 6), kernel {tuple(M.kernels.shape[-2:])}")
    print(f"analyzer fig: phi0 = {phis[k0]:.3f}, s0 analytic {s0:.3f} / cell {s_det[i0]:.3f}, "
          f"t0 analytic {t0:.3f}, argmax_t |response| = {ts[jmax]:.3f}, "
          f"FWHM in t = {th - tl:.3f} ({tl:.3f} .. {th:.3f}), above half: {len(above)} of {len(ts)} t-cells")

    wrong = [t0 - 0.6, t0 + 0.6]
    js = [j0] + [int(np.argmin(np.abs(ts - t))) for t in wrong]

    fig = plt.figure(figsize=(16, 5.0))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 0.62, 0.62, 0.62, 1.6], height_ratios=[1, 1],
                          wspace=0.32, hspace=0.35)
    a = fig.add_subplot(gs[:, 0])
    zs, zp = 0.35, 0.35
    ext = [s_det[0] - M.ds / 2, s_det[-1] + M.ds / 2, phis[0] - M.dphi / 2, phis[-1] + M.dphi / 2]
    v = np.quantile(np.abs(yl), 0.995)
    a.imshow(yl, origin="lower", extent=ext, aspect="equal", cmap="gray", vmin=-v, vmax=v,
             interpolation="nearest")
    pp = np.linspace(PHI0 - zp, PHI0 + zp, 200)
    hh, _, _ = h_and_slope(pp)
    a.plot(hh, pp, color="#f0a030", lw=1, ls="--", label=r"$s=h(\varphi)$")
    a.plot(s_det[i0], phis[k0], "o", ms=8, mfc="none", mec="#e03030", mew=2,
           label=r"$(s_0,\varphi_0)$")
    a.set_xlim(s0 - zs, s0 + zs), a.set_ylim(PHI0 - zp, PHI0 + zp)
    a.set_xlabel("$s$", fontsize=FS), a.set_ylabel(r"$\varphi$", fontsize=FS)
    a.set_title(r"$\Lambda^{1/2}y$ (zoom)", fontsize=FS)
    a.legend(loc="upper left", fontsize=FS - 3, framealpha=0.85)
    a.tick_params(labelsize=9)

    kern = M.kernels[:, 0].cpu().numpy()
    kp, ks = kern.shape[-2] // 2, kern.shape[-1] // 2
    kext = [-(ks + .5) * M.ds, (ks + .5) * M.ds, -(kp + .5) * M.dphi, (kp + .5) * M.dphi]
    for c, (j, lab) in enumerate(zip(js, ["true slope", "wrong slope", "wrong slope"])):
        ak = fig.add_subplot(gs[:, 1 + c])
        kk = kern[j]
        vk = np.abs(kk).max()
        ak.imshow(kk, origin="lower", extent=kext, cmap="RdBu_r", vmin=-vk, vmax=vk,
                  interpolation="nearest")
        ak.set_aspect("equal")
        ak.set_title(f"$K_t$, $t={ts[j]:.2f}$\n({lab})", fontsize=FS - 1,
                     color="#c02020" if c == 0 else "k")
        ak.set_xticks([]), ak.set_yticks([])
        ak.set_xlabel("$s$", fontsize=FS - 1)
        if c == 0:
            ak.set_ylabel(r"$\varphi$", fontsize=FS - 1)

    ar = fig.add_subplot(gs[:, 4])
    ar.axhline(0, color="0.7", lw=0.8)
    ar.plot(ts, prof, color="#4a7fb5", lw=1.5, label=r"signed $(K_t*\Lambda^{1/2}y)(s_0,\varphi_0)$")
    ar.plot(ts, absp, color="k", lw=2.2, label=r"$y_M(s_0,\varphi_0,t)=|\cdot|$")
    ar.plot(ts, np.abs(fr) * absp[j0] / max(abs(fr[j0]), 1e-12), color="#2a9d4a", lw=1.5, ls="--",
            label=r"predicted $|r(t,t_0)|$ (scaled)")
    ar.axvline(t0, color="#c02020", lw=1.5, ls=":")
    ar.text(t0 + 0.03, absp.max() * 1.12, f"$t_0={t0:.2f}$", color="#c02020", ha="left",
            fontsize=FS - 1)
    ar.annotate("", (th, half), (tl, half), arrowprops=dict(arrowstyle="<->", color="#e07020", lw=2))
    ar.text(th + 0.05, half + 0.04 * absp.max(), f"FWHM {th - tl:.2f}", color="#e07020", ha="left",
            fontsize=FS - 1)
    ar.set_xlim(ts[0], ts[-1])
    ymin = min(prof.min(), 0)
    ar.set_ylim(ymin - 0.05 * absp.max(), absp.max() * 1.8)
    ar.set_xlabel("slope / position along the ray $t$", fontsize=FS)
    ar.set_title(r"Response at $(s_0,\varphi_0)$ vs. $t$", fontsize=FS)
    ar.legend(loc="upper right", fontsize=FS - 4, framealpha=0.9)
    ar.tick_params(labelsize=9)
    out = ASSETS / "fig_sinogram_analyzer.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- metrics
def average_precision(score, pos, neg):
    """Step-wise AP over the voxels in pos | neg (as sklearn.average_precision_score):
    sum over distinct thresholds of (R_n - R_{n-1}) P_n."""
    y = np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)])
    s = np.concatenate([score[pos], score[neg]])
    return _ap_from(y, s)[0]


def _ap_from(y, s):
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]      # last index of each distinct score
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return ap, prec, rec


@torch.no_grad()
def metrics_and_pr(dev):
    ck = torch.load(str(ROOT / "outputs/train/n256_lam1e-4_noJ_xi0.3/checkpoint.pt"), map_location=dev,
                    weights_only=False)
    cfg = {**MS.LEGACY_DEFAULTS, **ck["config"]}
    amp = not cfg.get("no_amp", False)
    torch.manual_seed(cfg["seed"])
    grid, data, model, fwd, prior = MS.build(cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    C = CanonicalRelation(grid).to(dev)
    val = data.sample(np.random.default_rng(12345), cfg["val_size"], labels=True)
    nval = val["c"].shape[0]
    print(f"metrics: checkpoint step {ck['step']}, val set size {nval}")
    names = [r"$C^{-1}y_M$", "$y_W$", "LPD $m$"]
    keys = ["C^-1 y_M", "y_W", "m"]
    res = {k: {"auc": [], "ap": [], "hard": [], "hard_cb": [], "negfrac": []} for k in keys}
    pooled = {k: ([], []) for k in keys}
    for i in range(nval):
        batch = {k: v[i:i + 1] for k, v in val.items()}
        out = MS.run_model(model, batch, amp)
        pos, neg = label_masks(val["membership"][i, 0].numpy())
        pos2 = pos.any(0)
        near = (distance_transform_edt(~pos2) <= 6)[None] & neg          # Euclidean, 6 px
        near_cb = binary_dilation(pos2, iterations=6)[None] & neg          # as auc_diag.py
        scores = {"C^-1 y_M": C.inverse(batch["c"])[0, 0].cpu().numpy(),
                  "y_W": batch["b"][0, 0].cpu().numpy(),
                  "m": out["m"][0, 0].cpu().numpy()}
        print(f"sample {i}: pos {pos.mean() * 100:.2f}% neg {neg.mean() * 100:.1f}% of voxels; "
              f"hard negatives {near.sum() / neg.sum() * 100:.2f}% of negatives")
        for k in keys:
            sc = scores[k]
            r = res[k]
            r["auc"].append(roc_auc(sc, pos, neg))
            r["ap"].append(average_precision(sc, pos, neg))
            r["hard"].append(roc_auc(sc, pos, near))
            r["hard_cb"].append(roc_auc(sc, pos, near_cb))
            r["negfrac"].append(float(np.mean(sc[neg] > np.median(sc[pos]))))
            pooled[k][0].append(np.concatenate([np.ones(pos.sum(), bool), np.zeros(neg.sum(), bool)]))
            pooled[k][1].append(np.concatenate([sc[pos], sc[neg]]).astype(np.float32))
            print(f"  {k:9s} AUC {r['auc'][-1]:.3f}  AP {r['ap'][-1]:.3f}  hardAUC(eucl 6px) {r['hard'][-1]:.3f}"
                  f"  hardAUC(cityblock, auc_diag) {r['hard_cb'][-1]:.3f}  "
                  f"neg>median pos {r['negfrac'][-1] * 100:.2f}% "
                  f"(= {r['negfrac'][-1] * neg.sum() / (0.5 * pos.sum()):.1f} false voxels per true voxel "
                  f"above that threshold)")
    print(f"MEAN over {nval} validation samples:")
    for k in keys:
        r = res[k]
        print(f"  {k:9s} AUC {np.mean(r['auc']):.4f}  AP {np.mean(r['ap']):.4f}  "
              f"hardAUC {np.mean(r['hard']):.4f}  (cityblock {np.mean(r['hard_cb']):.4f})  "
              f"neg>median pos {np.mean(r['negfrac']) * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = ["#d08020", "#2a6fb5", "#b0306a"]
    base_rate = None
    for k, nm, c in zip(keys, names, cols):
        y = np.concatenate(pooled[k][0])
        s = np.concatenate(pooled[k][1])
        ap, prec, rec = _ap_from(y, s)
        base_rate = y.mean()
        print(f"pooled PR {k}: AP {ap:.4f} (positives {base_rate * 100:.3f}% of pos+neg voxels)")
        ax.plot(rec, prec, color=c, lw=2, label=f"{nm}  (mean AP {np.mean(res[k]['ap']):.2f})")
    ax.axhline(base_rate, color="0.5", lw=1, ls="--", label=f"chance ({base_rate * 100:.2f}% positives)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall", fontsize=FS), ax.set_ylabel("precision", fontsize=FS)
    ax.set_title(f"Precision-recall on lifted voxels ({nval} validation phantoms pooled)", fontsize=FS - 1)
    ax.legend(loc="upper right", fontsize=FS - 2, framealpha=0.9)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "fig_pr_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  (curves pooled over the validation set; AP in legend = "
          f"mean of per-sample AP)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="cr,analyzer,pr")
    args = ap.parse_args()
    which = set(args.only.split(","))
    ASSETS.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "cr" in which:
        fig_canonical_relation()
    if "analyzer" in which:
        fig_sinogram_analyzer(dev)
    if "pr" in which:
        metrics_and_pr(dev)


if __name__ == "__main__":
    main()
