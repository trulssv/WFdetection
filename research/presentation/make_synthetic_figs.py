"""Slide-sized synthetic-phantom figures from a trained LPD checkpoint (no retraining).

    CUDA_VISIBLE_DEVICES=0 python research/presentation/make_synthetic_figs.py \
        [--checkpoint outputs/train/n256_lam1e-4_noJ_xi0.3/checkpoint.pt] [--sample 0]

Reproduces the validation set of experiments/train_lpd.py (torch seed, build,
rng 12345) and writes to research/presentation/assets/:

    fig_val_inputs.png, fig_val_outputs.png, theta_legend.png,
    fig_iterations.png, fig_datafit.png, fig_outputs_prior.png

AUCs are for ``--sample``; F scores are dataset-optimal (ODS) over the whole
validation set, as in ``validate()``.  All numbers in titles are printed.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.metrics import label_masks, roc_auc
from evaluation.ridges import best_f, thin_estimate
from evaluation.visualize import orientation_rgb
from experiments.inspect_lpd import _norm_max
from experiments.train_lpd import LEGACY_DEFAULTS, build, run_model
from operators.canonical_relation import CanonicalRelation

ASSETS = ROOT / "research" / "presentation" / "assets"
FS = 13
DPI = 150


def show(ax, img, title=None, cmap=None, vmin=None, vmax=None):
    """Plot a (n, n) or (n, n, 3) field indexed [x1, x2] with x1 horizontal."""
    im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
    h = ax.imshow(im, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap if img.ndim == 2 else None,
                  vmin=vmin, vmax=vmax, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=FS)
    ax.set_xticks([]), ax.set_yticks([])
    return h


def row_figure(panels, path, cbar_for=None, cbar_label=None):
    """1 x len(panels) figure.  panels: (img, title, cmap, vmin, vmax)."""
    fig, ax = plt.subplots(1, len(panels), figsize=(16, 4.6))
    handles = [show(a, *p) for a, p in zip(ax, panels)]
    fig.tight_layout()
    if cbar_for is not None:
        box = ax[cbar_for[-1]].get_position()           # slim colorbar right of the last panel,
        cax = fig.add_axes([box.x1 + 0.006, box.y0, 0.008, box.height])   # panels keep their size
        cb = fig.colorbar(handles[cbar_for[-1]], cax=cax)
        cb.ax.tick_params(labelsize=10)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=11)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def theta_legend(K, path):
    """Hue legend matching orientation_rgb: hue = k / K at theta_k = pi k / K."""
    k = np.arange(K)
    hsv = np.stack([k / K, np.ones(K), np.ones(K)], -1)
    cmap = matplotlib.colors.ListedColormap(matplotlib.colors.hsv_to_rgb(hsv))
    fig, ax = plt.subplots(figsize=(4, 0.7))
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=matplotlib.colors.Normalize(0, np.pi),
                                          orientation="horizontal")
    cb.set_ticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    cb.set_ticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    cb.ax.tick_params(labelsize=11)
    cb.set_label(r"edge normal angle $\theta$", fontsize=12)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs/train/n256_lam1e-4_noJ_xi0.3/checkpoint.pt"))
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    ASSETS.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location=dev)
    cfg = {**LEGACY_DEFAULTS, **ck["config"]}
    amp = not cfg.get("no_amp", False)
    torch.manual_seed(cfg["seed"])                       # same RNG order as train_lpd.main
    grid, data, model, fwd, prior = build(cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    C = CanonicalRelation(grid).to(dev)
    C2 = CanonicalRelation(grid, double_cover=True).to(dev)
    val = data.sample(np.random.default_rng(12345), max(cfg["val_size"], args.sample + 1), labels=True)
    nval = val["c"].shape[0]
    print(f"checkpoint step {ck['step']}, n={cfg['n']}, K={cfg['n_theta']}, val set size {nval}, device {dev}")

    # ---- validation metrics (as in train_lpd.validate) ----
    ms, aucs, base, base_b = [], [], [], []
    for i in range(nval):
        batch = {k: v[i:i + 1] for k, v in val.items()}
        out = run_model(model, batch, amp)
        pos, neg = label_masks(val["membership"][i, 0].numpy())
        aucs.append(roc_auc(out["m"][0, 0].cpu().numpy(), pos, neg))
        base.append(roc_auc(C.inverse(batch["c"])[0, 0].cpu().numpy(), pos, neg))
        base_b.append(roc_auc(batch["b"][0, 0].cpu().numpy(), pos, neg))
        ms.append(out["m"][0, 0])
    trues = [val["wf_thin"][i, 0].numpy() for i in range(nval)]
    f_m = best_f(ms, trues, grid)
    f_b = best_f([val["b"][i, 0] for i in range(nval)], trues, grid)
    s = args.sample
    print(f"sample {s}: AUC C^-1 c = {base[s]:.3f}, AUC b = {base_b[s]:.3f}, AUC m = {aucs[s]:.3f}")
    print(f"mean over {nval}: AUC C^-1 c = {np.mean(base):.4f}, AUC b = {np.mean(base_b):.4f}, "
          f"AUC m = {np.mean(aucs):.4f}")
    print(f"ODS over {nval}: F(b) = {f_b['F']:.3f} (thr {f_b['threshold']:.2f}, P {f_b['P']:.3f}, "
          f"R {f_b['R']:.3f}, F_x {f_b['F_x']:.3f}); "
          f"F(m) = {f_m['F']:.3f} (thr {f_m['threshold']:.2f}, P {f_m['P']:.3f}, R {f_m['R']:.3f}, "
          f"F_x {f_m['F_x']:.3f})")

    m_np = ms[s].cpu().numpy()
    b_np = val["b"][s, 0].cpu().numpy()
    true_rgb = orientation_rgb(trues[s].astype(float))
    thin_m = thin_estimate(ms[s], grid, f_m["threshold"]).cpu().numpy()
    thin_b = thin_estimate(val["b"][s, 0], grid, f_b["threshold"]).cpu().numpy()
    print(f"sample {s}: max m = {m_np.max():.3f}")

    # ---- 1. inputs / baselines ----
    row_figure([
        (val["f"][s, 0].numpy(), "Phantom (ground truth image)", "gray", None, None),
        (true_rgb, "True wavefront set\n(analytic boundary + normals)", None, None, None),
        (orientation_rgb(C.inverse(val["c"][s:s + 1].to(dev))[0, 0].cpu().numpy()),
         f"Sinogram baseline $C^{{-1}}y_M$\n(AUC {base[s]:.3f})", None, None, None),
        (orientation_rgb(b_np), f"Image baseline $y_W$\n(AUC {base_b[s]:.3f})", None, None, None),
    ], ASSETS / "fig_val_inputs.png")

    # ---- 2. outputs ----
    row_figure([
        (true_rgb, "True wavefront set", None, None, None),
        (orientation_rgb(m_np), f"LPD membership $m$\n(AUC {aucs[s]:.3f})", None, None, None),
        (orientation_rgb((thin_b > 0).astype(float)), f"Thin ridges of $y_W$\n(F {f_b['F']:.3f})", None, None, None),
        (orientation_rgb((thin_m > 0).astype(float)), f"Thin ridges of LPD $m$\n(F {f_m['F']:.3f})", None, None, None),
    ], ASSETS / "fig_val_outputs.png")

    # ---- 3. theta legend ----
    theta_legend(grid.n_theta, ASSETS / "theta_legend.png")

    # ---- intermediates of sample s (as in inspect_lpd.inspect) ----
    batch = {k: v[s:s + 1] for k, v in val.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and batch["c"].is_cuda):
        o = model(batch["c"], batch["b"] if model.image_input else None, return_intermediates=True)
    o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}

    # ---- 4. iterations ----
    n_it = len(o["primal"])
    reads = [r["m"][0, 0].float().cpu().numpy() for r in o["readouts"]]
    prim = [_norm_max(p[0]) for p in o["primal"]]
    dual = [_norm_max(C2.inverse(d[:1].float())[0]) for d in o["dual"]]
    vp, vd = max(x.max() for x in prim), max(x.max() for x in dual)
    print("iterations: readout m_i shown with orientation_rgb, each normalized to its own max; "
          "max m_i = " + ", ".join(f"{r.max():.3f}" for r in reads))
    print(f"iterations: |p_i| row shares magma scale [0, {vp:.3g}] (max over iterations); per-iteration max = "
          + ", ".join(f"{x.max():.3g}" for x in prim))
    print(f"iterations: |C^-1 d_i| row shares magma scale [0, {vd:.3g}] (max over iterations); per-iteration max = "
          + ", ".join(f"{x.max():.3g}" for x in dual))
    fig, ax = plt.subplots(3, n_it, figsize=(16, 10))
    for i in range(n_it):
        show(ax[0, i], orientation_rgb(reads[i]), f"Iteration {i + 1}")
        show(ax[1, i], prim[i], cmap="magma", vmin=0, vmax=vp)
        show(ax[2, i], dual[i], cmap="magma", vmin=0, vmax=vd)
    for r, lab in enumerate(["Readout $m_i$\n(hue = θ)", "Primal state $|p_i|$",
                             "Dual state pulled back\n$|C^{-1}d_i|$"]):
        ax[r, 0].set_ylabel(lab, fontsize=FS)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_iterations.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(ASSETS / 'fig_iterations.png').relative_to(ROOT)}")

    # ---- 5. data fit ----
    q = o["q"]
    b_hat = fwd["image"](q)
    res = (batch["b"] - b_hat)[0, 0]
    under = res.clamp_min(0).amax(0).cpu().numpy()
    over = (-res).clamp_min(0).amax(0).cpu().numpy()
    vr = max(under.max(), over.max())
    bh = b_hat[0, 0].cpu().numpy()
    print(f"datafit: max b = {b_np.max():.3g}, max b_hat = {bh.max():.3g}, "
          f"max unexplained = {under.max():.3g}, max over-predicted = {over.max():.3g}, shared vmax = {vr:.3g}; "
          f"b and b_hat each normalized to own max in orientation_rgb")
    row_figure([
        (orientation_rgb(b_np), "Image data $y_W$", None, None, None),
        (orientation_rgb(bh), r"Predicted data $\hat y_W = F_X(q)$", None, None, None),
        (under, r"Unexplained: $\mathrm{max}_\theta\,(y_W-\hat y_W)_+$", "magma", 0, vr),
        (over, r"Over-predicted: $\mathrm{max}_\theta\,(\hat y_W-y_W)_+$", "magma", 0, vr),
    ], ASSETS / "fig_datafit.png", cbar_for=[2, 3])

    # ---- 6. outputs and prior ----
    m0 = o["m"][0, 0].cpu().numpy()
    if "J" in o:                                  # --contrast runs
        J0 = o["J"][0, 0].cpu().numpy()
        Jmap = np.take_along_axis(J0, m0.argmax(0)[None], 0)[0] * (m0.max(0) > 0.05)
        amp_title = "Contrast $J$ (where $m>0.05$)"
    else:
        Jmap = o["q"][0, 0].cpu().numpy().max(0)
        amp_title = r"Edge strength $q=a/a_\mathrm{ref}$ (max $\theta$)"
    length = prior.length_density(o["a"], o["b"], o["c"])[0, 0].amax(0).cpu().numpy()
    div = prior.divergence(o["a"], o["b"], o["c"]).abs()[0, 0].amax(0).cpu().numpy()
    sel = m0.max(0) > 0.05
    print(f"prior fig: amplitude where m>0.05: quantiles 1/10/50/90% = "
          + ", ".join(f"{v:.3f}" for v in np.quantile(Jmap[sel], [0.01, 0.1, 0.5, 0.9])))
    print(f"prior fig: amplitude panel where max m > 0.05, viridis [{Jmap.min():.3g}, {Jmap.max():.3g}]; "
          f"length density max_theta in [0, {length.max():.3g}]; |div tau| max_theta in [0, {div.max():.3g}]")
    row_figure([
        (orientation_rgb(m0), "Membership $m$ (hue = θ)", None, None, None),
        (Jmap, amp_title, "viridis", None, None),
        (length, "Prior: curve length density", "magma", 0, None),
        (div, r"Prior: endpoint density $|\mathrm{div}\,\tau|$", "magma", 0, None),
    ], ASSETS / "fig_outputs_prior.png")

    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
