from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "research" / "presentation" / "assets"
FINAL_STEP = 1500  # step of outputs/train/n256_lam1e-4_noJ_xi0.3/checkpoint.pt


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            records.append(json.loads(line))
    return records


def make_convergence_plot() -> None:
    records = _read_jsonl(ROOT / "outputs" / "train" / "n256_lam1e-4_noJ_xi0.3" / "log.jsonl")
    train = [rec for rec in records if "loss" in rec]
    val = [rec for rec in records if "val_auc" in rec]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    axes[0].plot([rec["step"] for rec in train], [rec["loss"] for rec in train],
                 color="#0f4c5c", lw=2.0, label="total loss")
    axes[0].plot([rec["step"] for rec in train], [rec["D"] for rec in train],
                 color="#e36414", lw=1.6, label=r"data term $\mathcal{D}_X$")
    axes[0].plot([rec["step"] for rec in train],
                 [1e-4 * rec["E"] for rec in train],
                 color="#6a994e", lw=1.6, label=r"regularizer $\lambda E$")
    axes[0].set_title("Training convergence")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("objective value")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    steps = [rec["step"] for rec in val]
    axes[1].plot(steps, [rec["val_auc"] for rec in val], color="#1d3557", lw=2.2,
                 marker="o", label="LPD AUC")
    axes[1].hlines(val[0]["val_auc_b"], steps[0], steps[-1], color="#2a9d8f", lw=2,
                   linestyles="--", label="image baseline AUC")
    axes[1].hlines(val[0]["val_auc_baseline"], steps[0], steps[-1], color="#8d99ae", lw=2,
                   linestyles=":", label=r"sinogram baseline AUC")
    axes[1].plot(steps, [rec["val_F"] for rec in val], color="#c1121f", lw=2.2,
                 marker="s", label="LPD ridge F-score")
    axes[1].hlines(val[0]["val_F_b"], steps[0], steps[-1], color="#f4a261", lw=2,
                   linestyles="--", label="ridge baseline F-score")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png", dpi=180)
    plt.close(fig)


def make_metric_plot() -> None:
    records = _read_jsonl(ROOT / "outputs" / "train" / "n256_lam1e-4_noJ_xi0.3" / "log.jsonl")
    val = [rec for rec in records if rec.get("step") == FINAL_STEP and "val_auc" in rec][0]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))
    auc_labels = ["sinogram baseline\n$C^{-1}y_M$", "image baseline\n$y_W$", "LPD\nmembership $m$"]
    auc_values = [val["val_auc_baseline"], val["val_auc_b"], val["val_auc"]]
    auc_colors = ["#8d99ae", "#2a9d8f", "#1d3557"]
    axes[0].bar(auc_labels, auc_values, color=auc_colors)
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_title("ROC AUC on lifted voxels $(x,\\theta)$")
    axes[0].grid(axis="y", alpha=0.2)
    for idx, value in enumerate(auc_values):
        axes[0].text(idx, value + 0.003, f"{value:.3f}", ha="center", fontsize=10)

    f_labels = ["thin ridges of $y_W$", "thin ridges of LPD $m$"]
    f_values = [val["val_F_b"], val["val_F"]]
    f_colors = ["#f4a261", "#c1121f"]
    axes[1].bar(f_labels, f_values, color=f_colors)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Thin-ridge F-score (best threshold)")
    axes[1].grid(axis="y", alpha=0.2)
    for idx, value in enumerate(f_values):
        axes[1].text(idx, value + 0.03, f"{value:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(ASSETS / "metrics.png", dpi=180)
    plt.close(fig)


SINO_COLOR = "#d8f3dc"   # sinogram-domain quantities
IMAGE_COLOR = "#ffe8d6"  # image-domain quantities
_BOXES: list[tuple] = []


def _box(ax, xy, width, height, text, face, edge="#16324f", fontsize=12.5):
    x0, y0 = xy
    patch = FancyBboxPatch((x0, y0), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.5, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    label = ax.text(x0 + width / 2, y0 + height / 2, text, ha="center", va="center",
                    fontsize=fontsize, linespacing=1.35)
    _BOXES.append((patch, label, text))
    return patch


def _check_text_fits(fig) -> None:
    """Fail loudly if any label overflows its box (the old flowchart did)."""
    renderer = fig.canvas.get_renderer()
    for patch, label, text in _BOXES:
        box = patch.get_window_extent(renderer)
        tb = label.get_window_extent(renderer)
        if tb.x0 < box.x0 + 4 or tb.x1 > box.x1 - 4 or tb.y0 < box.y0 + 2 or tb.y1 > box.y1 - 2:
            raise RuntimeError(f"label does not fit its box: {text!r}")


def _arrow(ax, start, end, style="-|>", ls="-", color="#16324f", rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                                 lw=1.6, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def make_flowchart() -> None:
    _BOXES.clear()
    fig, ax = plt.subplots(figsize=(15.0, 6.2))
    ax.set_xlim(0, 17.6)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    data = "#d9f0ff"
    _box(ax, (0.1, 2.9), 2.4, 1.5, "Fan-beam data\n$g(\\beta,u)$", data)
    _box(ax, (3.0, 2.9), 2.6, 1.5, "Rebin to parallel\nsinogram $y(s,\\varphi)$", data)
    _box(ax, (6.2, 4.6), 3.6, 1.9,
         "Sinogram analyzer $M$\n$y_M=|K_t * \\Lambda^{1/2} y|$\non $(s,\\varphi,t)$", SINO_COLOR)
    _box(ax, (6.2, 0.8), 3.6, 1.9,
         "FBP + image analyzer $W$\n$y_W=|W_\\theta * \\mathrm{FBP}\\,y|$\non $(x,\\theta)$", IMAGE_COLOR)
    _box(ax, (10.5, 2.55), 3.3, 2.2,
         "Learned primal-dual\n5 unrolled iterations\ndual $d$: sinogram side\nprimal $p$: image side",
         "#f1e3f3")
    _box(ax, (14.4, 4.6), 3.1, 1.9, "Output on $(x,\\theta)$\ncurrent $(a,b,c)$\n$q=a/a_{\\rm ref}$,  membership $m$",
         IMAGE_COLOR)
    _box(ax, (14.4, 0.8), 3.1, 1.9, "Unsupervised loss\n$\\mathcal{D}_X(y_W,\\hat y_W(q))+\\lambda E(T)$",
         "#fff3b0")

    _arrow(ax, (2.5, 3.65), (3.0, 3.65))
    _arrow(ax, (5.6, 3.9), (6.2, 5.4))
    _arrow(ax, (5.6, 3.4), (6.2, 1.9))
    _arrow(ax, (9.8, 5.3), (10.5, 4.3))
    _arrow(ax, (9.8, 1.9), (10.5, 3.0))
    _arrow(ax, (13.8, 4.3), (14.4, 5.3))
    _arrow(ax, (15.95, 4.6), (15.95, 2.7))
    # y_W is also the target that the data term compares against
    _arrow(ax, (8.0, 0.8), (15.2, 0.8), ls="--", color="#b5651d", rad=0.12)
    ax.text(11.6, 0.05, "$y_W$ is also the data that $\\mathcal{D}_X$ compares $\\hat y_W(q)$ against",
            ha="center", fontsize=11, color="#b5651d")

    ax.add_patch(FancyBboxPatch((0.1, 6.85), 0.35, 0.3, boxstyle="square,pad=0",
                                facecolor=SINO_COLOR, edgecolor="#16324f"))
    ax.text(0.6, 7.0, "sinogram domain", va="center", fontsize=11)
    ax.add_patch(FancyBboxPatch((3.1, 6.85), 0.35, 0.3, boxstyle="square,pad=0",
                                facecolor=IMAGE_COLOR, edgecolor="#16324f"))
    ax.text(3.6, 7.0, "image domain (lifted: position $x$, normal angle $\\theta$)",
            va="center", fontsize=11)

    fig.tight_layout()
    fig.canvas.draw()
    _check_text_fits(fig)
    fig.savefig(ASSETS / "flowchart.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Figures from the checkpoint are made by make_synthetic_figs.py, make_walnut_fig.py
    # and bench_scaling.py.
    make_convergence_plot()
    make_metric_plot()
    make_flowchart()


if __name__ == "__main__":
    main()