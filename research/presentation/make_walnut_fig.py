"""Slide figure of the walnut wavefront set (n=256 LPD, fine-tuned).

Runs experiments/walnut_wavefront.main() (unchanged) into OUT, records wall-clock
time and peak GPU memory, then builds research/presentation/assets/fig_walnut.png
from the saved arrays (+ the dataset's reference FBP) and fig_walnut_fields.png,
the current tau = a X1 + c X2 + b X3 read out at the dominant angle.

    CUDA_VISIBLE_DEVICES=0 python -m research.presentation.make_walnut_fig [--skip-run]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.walnut import load_reference
from evaluation.visualize import orientation_rgb
from geometry.lifted import LiftedGrid
from models.lpd import reference_line_density
from experiments import walnut_wavefront

CKPT = "outputs/train/n256_lam1e-4_noJ_xi0.3/checkpoint.pt"
OUT = Path("outputs/walnut_lpd256_noJ")
FIG = Path("research/presentation/assets/fig_walnut.png")
FIG_FIELDS = Path("research/presentation/assets/fig_walnut_fields.png")
STEPS = 200


def run():
    sys.argv = ["walnut_wavefront", "--checkpoint", CKPT, "--finetune", str(STEPS), "--out", str(OUT)]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    walnut_wavefront.main()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    info = {"wall_clock_s_total": wall, "peak_gpu_mem_GiB": peak,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def figure():
    res = json.loads((OUT / "results.json").read_text())
    n = res["n"]
    ref = load_reference(n=n)
    np.save(OUT / "reference_fbp.npy", ref)
    base = np.load(OUT / "baseline.npy")
    b = np.load(OUT / "baseline_b.npy")
    rb = np.load(OUT / "ridges_b.npy").astype(float)
    m = np.load(OUT / "membership.npy")
    rm = np.load(OUT / "ridges_m.npy").astype(float)

    panels = [("Reference FBP\n(1200 projections)", ref, "gray"),
              ("Image baseline $y_W$\n(learning-free)", orientation_rgb(b), None),
              ("Thin ridges of $y_W$", orientation_rgb(rb), None),
              ("Sinogram baseline $C^{-1}y_M$\n(learning-free)", orientation_rgb(base), None),
              (f"LPD membership $m$\n(+{res.get('finetune_steps', STEPS)} fine-tuning steps; scaled to max)",
               orientation_rgb(m), None),
              ("Thin ridges of LPD $m$", orientation_rgb(rm), None)]
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for a, (title, img, cmap) in zip(ax.ravel(), panels):
        im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
        a.imshow(im, origin="upper", extent=[-1, 1, -1, 1], cmap=cmap)
        a.set_title(title, fontsize=13)
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {FIG}")


def figure_fields():
    """a, b, c on (theta, x) shown at theta*(x) = argmax_theta a, plus their theta-profiles at one pixel.

    a >= 0 is shown like m (brightness = max over theta, hue = argmax), since m is
    increasing in a.  b and c are signed, so a max-projection would be meaningless;
    they are read out at theta* and shown in units of a_ref, on one colour scale."""
    f = np.load(OUT / "fields.npz")
    a, b, c = f["a"], f["b"], f["c"]
    K, n, _ = a.shape
    a_ref = reference_line_density(LiftedGrid(n, K))
    m_max = (1 - np.exp(-a / a_ref)).max(0)
    k = a.argmax(0)
    at = lambda F: np.take_along_axis(F, k[None], 0)[0] / a_ref
    mask = m_max < 0.05
    b_s, c_s = np.ma.masked_where(mask, at(b)), np.ma.masked_where(mask, at(c))
    vb = np.quantile(np.abs(at(b)[~mask]), 0.99)
    sel = m_max > 0.2
    kappa = at(b)[sel] / at(a)[sel]
    print(f"fields: a_ref = {a_ref:.1f}; on m > 0.2 ({sel.mean():.1%} of pixels): "
          f"|b|/a_ref 50/99% = {np.quantile(np.abs(at(b)[sel]), [.5, .99]).round(3)}, "
          f"|c|/a_ref 50/99% = {np.quantile(np.abs(at(c)[sel]), [.5, .99]).round(4)}, "
          f"|kappa| = |b/a| 50/90% = {np.quantile(np.abs(kappa), [.5, .9]).round(2)}; colour scale +-{vb:.2f}")

    i, j = np.unravel_index(m_max.argmax(), m_max.shape)     # profile pixel: strongest detection
    xs = np.linspace(-1, 1, n, endpoint=False) + 1 / n
    th = np.arange(K) * np.pi / K

    fig, ax = plt.subplots(1, 5, figsize=(18.5, 4.6),
                           gridspec_kw={"width_ratios": [1, 1, 1, 0.05, 1.35], "wspace": 0.12})
    cax, ax = ax[3], [ax[0], ax[1], ax[2], ax[4]]
    maps = [(r"$a$: line density (hue = $\theta^*$)", orientation_rgb(a), None),
            (r"$b(x,\theta^*)/a_\mathrm{ref}$: turning", b_s, "RdBu_r"),
            (r"$c(x,\theta^*)/a_\mathrm{ref}$: sideways motion", c_s, "RdBu_r")]
    for axk, (title, img, cmap) in zip(ax, maps):
        im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
        axk.set_facecolor("0.85")
        h = axk.imshow(im, origin="upper", extent=[-1, 1, -1, 1], cmap=cmap, vmin=-vb if cmap else None,
                       vmax=vb if cmap else None, interpolation="nearest")
        axk.plot(xs[i], xs[j], "o", mfc="none", mec="k" if cmap else "w", ms=9, mew=1.5)
        axk.set_title(title, fontsize=13)
        axk.set_xticks([]), axk.set_yticks([])
    cb = fig.colorbar(h, cax=cax)
    pos = ax[3].get_position()
    ax[3].set_position([pos.x0 + 0.045, pos.y0 + 0.08, pos.width - 0.045, pos.height - 0.12])
    cb.ax.tick_params(labelsize=10)

    p = ax[3]
    for F, lab, col in [(a, "$a$", "k"), (b, "$b$", "C3"), (c, "$c$", "C0")]:
        p.plot(th, F[:, i, j] / a_ref, color=col, lw=2, label=lab)
    p.axvline(th[k[i, j]], color="0.5", ls="--", lw=1)
    p.axhline(0, color="0.7", lw=0.8)
    p.set_xlim(0, np.pi)
    p.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    p.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    p.set_xlabel(r"$\theta$ at the circled pixel ($\theta^*$ dashed)", fontsize=12)
    p.set_ylabel(r"field $/\,a_\mathrm{ref}$", fontsize=12)
    p.set_title("One pixel: the fibre over $x$", fontsize=13)
    p.legend(fontsize=11, frameon=False)
    p.set_aspect("auto")
    FIG_FIELDS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_FIELDS, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {FIG_FIELDS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true", help="only rebuild the figure")
    if not ap.parse_args().skip_run:
        run()
    figure()
    figure_fields()
