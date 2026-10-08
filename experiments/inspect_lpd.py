"""Inspect a trained LPD network: intermediate states and the terms of the objective.

    python -m experiments.inspect_lpd --checkpoint outputs/train/run/checkpoint.pt [--sample 0]

Runs the network on validation phantom ``--sample`` (the same fixed held-out
set as in training, seed 12345) and writes to ``--out`` (default: next to the
checkpoint, ``inspect_<sample>/``):

* ``intermediates.npz``: inputs (c, b), the primal and dual states after each
  iteration, the readout (the output head applied to each primal state), the
  outputs (a, b_flux, c_normal, m, q, and J for --contrast runs), the predicted coefficients
  b_hat, and the prior's densities (length density, |div tau|); fields on RP^1
  have shape (K, n, n), states on the double cover (channels, 2K, n, n);
* ``iterations.png``: per iteration, the readout m_i, its ridges, the primal
  state (norm over channels, max over theta) and the dual state pulled back
  to the image (norm over channels, max over theta);
* ``final.png``: phantom, true wavefront set, the data b and its prediction
  b_hat, the residual, the amplitude q (or contrast J), prior densities, and the orientation error
  of the output at the true wavefront points.

Lifted fields are shown as max over theta (brightness) and argmax theta (hue).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.ridges import best_f, thin_estimate
from evaluation.visualize import orientation_rgb
from experiments.train_lpd import LEGACY_DEFAULTS, build, data_term
from loss.data_discrepancy import ForwardModelDiscrepancy
from models.layers import s1_to_rp1_even
from operators.canonical_relation import CanonicalRelation


def _norm_max(x):
    """(C, 2K, n, n) state -> (n, n): channel norm, max over the fibre."""
    return x.float().norm(dim=0).amax(dim=0).cpu().numpy()


def _show(ax, img, title, cmap=None):
    im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
    h = ax.imshow(im, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    if img.ndim == 2 and cmap not in (None, "gray"):
        plt.colorbar(h, ax=ax, fraction=0.046)


@torch.no_grad()
def inspect(model, batch, fwd, prior, grid, cfg, out_dir: Path, threshold=0.3):
    """Run ``model`` on a batch of size 1 and write intermediates and figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch["c"].is_cuda):
        o = model(batch["c"], batch["b"] if model.image_input else None, return_intermediates=True)
    o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}
    q = o["q"]
    b_hat = fwd["image"](q)
    disc = ForwardModelDiscrepancy(delta=1e-2)
    D, parts = data_term(q, batch, fwd, disc, cfg)
    E = prior(o["a"], o["b"], o["c"]).mean()
    length = prior.length_density(o["a"], o["b"], o["c"])
    div = prior.divergence(o["a"], o["b"], o["c"]).abs()
    C2 = CanonicalRelation(grid, double_cover=True).to(q.device)

    first = lambda x: x[0, 0].cpu().numpy()
    arrays = {"c": first(batch["c"]), "b": first(batch["b"]), "a": first(o["a"]), "b_flux": first(o["b"]),
              "c_normal": first(o["c"]), "m": first(o["m"]), "q": first(q),
              "b_hat": first(b_hat), "length_density": first(length), "abs_div": first(div),
              "primal": np.stack([p[0].cpu().numpy().astype(np.float16) for p in o["primal"]]),
              "dual": np.stack([d[0].cpu().numpy().astype(np.float16) for d in o["dual"]]),
              "readout_m": np.stack([first(r["m"]) for r in o["readouts"]])}
    if "J" in o:
        arrays["J"] = first(o["J"])
    if "wf_thin" in batch:
        arrays["wf_thin"] = batch["wf_thin"][0, 0].numpy()
    np.savez_compressed(out_dir / "intermediates.npz", **arrays)

    # per-iteration figure
    n_it = len(o["primal"])
    fig, ax = plt.subplots(n_it, 4, figsize=(16, 4 * n_it))
    ax = np.atleast_2d(ax)
    for i in range(n_it):
        m_i = o["readouts"][i]["m"][0, 0]
        ridge = (thin_estimate(m_i, grid, threshold) > 0).float().cpu().numpy()
        _show(ax[i, 0], orientation_rgb(m_i.cpu().numpy()), f"iteration {i + 1}: readout m (max {m_i.max():.2f})")
        _show(ax[i, 1], orientation_rgb(ridge), f"iteration {i + 1}: ridges of readout m")
        _show(ax[i, 2], _norm_max(o["primal"][i][0]), f"primal state |p_{i + 1}|", "magma")
        _show(ax[i, 3], _norm_max(C2.inverse(o["dual"][i][:1])[0]),
              f"dual state pulled back |C$^{{-1}}$d_{i + 1}|", "magma")
    fig.tight_layout()
    fig.savefig(out_dir / "iterations.png", dpi=80)
    plt.close(fig)

    # final figure
    m = o["m"][0, 0]
    res = (batch["b"] - b_hat)[0, 0]
    panels = []
    if "f" in batch:
        panels.append((batch["f"][0, 0].numpy(), "phantom", "gray"))
    if "wf_thin" in batch:
        panels.append((orientation_rgb(arrays["wf_thin"].astype(float)), "true WF (thin)", None))
    panels += [
        (orientation_rgb(arrays["b"]), "data b", None),
        (orientation_rgb(arrays["b_hat"]), r"prediction $\hat b$ = fwd(q)", None),
        (res.clamp_min(0).amax(0).cpu().numpy(), r"unexplained: max$_\theta$ (b − $\hat b$)$_+$", "magma"),
        ((-res).clamp_min(0).amax(0).cpu().numpy(), r"overpredicted: max$_\theta$ ($\hat b$ − b)$_+$", "magma"),
        (orientation_rgb(arrays["m"]), f"membership m (max {m.max():.2f})", None),
        (orientation_rgb((thin_estimate(m, grid, threshold) > 0).float().cpu().numpy()),
         f"ridges of m (threshold {threshold})", None),
        (np.take_along_axis(arrays["J"], arrays["m"].argmax(0)[None], 0)[0] * (arrays["m"].max(0) > 0.05),
         "contrast J at argmax θ (where m > 0.05)", "viridis") if "J" in arrays else
        (arrays["q"].max(0), "amplitude q = a / a_ref (max θ)", "viridis"),
        (arrays["length_density"].max(0), "prior: length density (max θ)", "magma"),
        (arrays["abs_div"].max(0), "prior: |div τ| (endpoints, max θ)", "magma"),
    ]
    info = {"D": float(D), **{k: float(v) for k, v in parts.items()}, "E": float(E)}
    if "wf_thin" in batch:
        K = grid.n_theta
        k_true, i1, i2 = np.nonzero(arrays["wf_thin"])
        err = np.full(arrays["m"].shape[1:], np.nan)
        dk = (arrays["m"][:, i1, i2].argmax(0) - k_true + K // 2) % K - K // 2
        err[i1, i2] = np.abs(dk)
        panels.append((err, "|θ error| of m at true WF (cells)", "viridis"))
        dkb = (arrays["b"][:, i1, i2].argmax(0) - k_true + K // 2) % K - K // 2
        info.update(theta_err_median_m=float(np.median(np.abs(dk))), theta_err_median_b=float(np.median(np.abs(dkb))),
                    **{f"{k}_m": v for k, v in best_f([m], [arrays["wf_thin"]], grid).items()})
    cols = 4
    rows = -(-len(panels) // cols)
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    ax = ax.ravel()
    for a in ax[len(panels):]:
        a.axis("off")
    for a, (img, title, cmap) in zip(ax, panels):
        _show(a, img, title, cmap)
    fig.suptitle("  ".join(f"{k}={v:.4g}" for k, v in info.items()), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "final.png", dpi=80)
    plt.close(fig)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample", type=int, default=0, help="index into the fixed validation set")
    ap.add_argument("--threshold", type=float, default=0.3, help="ridge threshold (fraction of max m)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=dev)
    cfg = {**LEGACY_DEFAULTS, **ck["config"]}
    grid, data, model, fwd, prior = build(cfg, dev)
    model.load_state_dict(ck["model"])
    val = data.sample(np.random.default_rng(12345), max(cfg["val_size"], args.sample + 1), labels=True)
    batch = {k: v[args.sample:args.sample + 1] for k, v in val.items()}
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"inspect_{args.sample}"
    info = inspect(model, batch, fwd, prior, grid, cfg, out, args.threshold)
    print(json.dumps(info, indent=2))
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
