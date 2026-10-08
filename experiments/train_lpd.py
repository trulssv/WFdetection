"""Unsupervised training of the lifted LPD network on synthetic tomography data.

    loss = D_data(q) + lambda E(a, b, c),  q = a / a_ref   (research/background.md, Sec. 7)

The data term (``--data-term``) is a Laplace forward-model discrepancy on

* ``image`` (default): the full-aperture coefficients b = |W FBP y|, modelled
  locally by G * q (``loss.data_discrepancy.OrientedEdgeForwardModel``);
* ``sinogram``: the sinogram coefficients c = M y, modelled by R_t C q
  (Definition 5.4).  Its fibre response is ~30 cells wide, so it cannot tell a
  curve from its tangent-line streaks;
* ``both``: image + mu * sinogram.

E is the curve prior; its turning (b) penalty is xi^2 b^2 inside the
square root of the length density (``--length subriemannian``) or
beta b^2 / a (``--length elastica``).  ``--contrast`` restores the old
amplitude q = J m with a learned contrast J.  No wavefront-set labels are used for training; the
analytic labels of the phantoms are used only for validation (ROC AUC of the
membership m, compared with the learning-free C^{-1} c and b).

Run from the repository root:

    python -m experiments.train_lpd --steps 4000 --out outputs/train/run1
    python -m experiments.train_lpd --resume outputs/train/run1   # continue

Writes config.json, log.jsonl, checkpoint.pt and validation figures val_<step>.png.
A tqdm bar shows loss, D, E, the learning rate and the time per step spent on
data generation and on the network (updated every 30 s when stderr is not a
terminal, e.g. under nohup).
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
from tqdm import tqdm

from data.synthetic import SyntheticTomography
from evaluation.metrics import label_masks, roc_auc
from evaluation.ridges import best_f, thin_estimate
from evaluation.visualize import orientation_rgb
from geometry.lifted import LiftedGrid
from loss.data_discrepancy import (ForwardModelDiscrepancy, MicrolocalForwardModel,
                                   OrientedEdgeForwardModel)
from loss.regularization import CurvePrior
from models.lpd import LiftedLPD
from operators.canonical_relation import CanonicalRelation


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/train/default")
    ap.add_argument("--resume", default=None, help="run directory to continue")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--n-theta", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--background", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.01, help="prior weight lambda")
    ap.add_argument("--xi", type=float, default=0.3)
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--zeta", type=float, default=2.0)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--contrast", action="store_true",
                    help="learned contrast J, q = J m (old architecture; default q = a / a_ref)")
    ap.add_argument("--j-max", type=float, default=1.0,
                    help="contrast bound with --contrast; q = J m must reach the edge strength of the normalized data")
    ap.add_argument("--length", choices=["subriemannian", "elastica"], default="subriemannian",
                    help="length term of the prior: sqrt(a^2 + zeta^2 c^2 + xi^2 b^2) or "
                         "alpha sqrt(a^2 + zeta^2 c^2) + beta b^2 / a")
    ap.add_argument("--beta", type=float, default=0.01, help="elastica turning weight")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast in the network")
    ap.add_argument("--data-term", choices=["image", "sinogram", "both"], default="image")
    ap.add_argument("--mu", type=float, default=0.5, help="weight of the sinogram term for 'both'")
    ap.add_argument("--no-image-input", dest="image_input", action="store_false",
                    help="do not feed b to the network's primal blocks")
    ap.add_argument("--sigma-cells", type=float, default=1.5, help="image analyzer width (cells)")
    ap.add_argument("--aspect", type=float, default=2.0, help="image analyzer aspect ratio")
    ap.add_argument("--edge-width", type=float, default=2.6,
                    help="width (cells) of an edge in b assumed by the forward model: kernel plus FBP blur")
    ap.add_argument("--detrend", type=float, default=0.0,
                    help="coarse/fine scale ratio subtracted by the image analyzer (0: off)")
    return ap.parse_args(argv)


# configs written before the image-side data term existed, or before J was removed
LEGACY_DEFAULTS = {"j_max": 1.0, "data_term": "sinogram", "mu": 0.5, "image_input": False,
                   "sigma_cells": 1.5, "aspect": 2.0, "detrend": 0.0, "edge_width": None,
                   "contrast": True, "length": "subriemannian", "beta": 0.01}


class _Lines:
    """File wrapper turning tqdm's carriage returns into newlines, so that a
    redirected log (nohup, remote runs) gets one progress line per refresh."""

    def __init__(self, f):
        self.f = f

    def write(self, s):
        s = s.replace("\r", "\n").replace("\n\n", "\n")
        return self.f.write(s)

    def flush(self):
        self.f.flush()


def progress(**kw):
    tty = sys.stderr.isatty()
    return tqdm(dynamic_ncols=tty, mininterval=0.5 if tty else 30, smoothing=0.05,
                file=sys.stderr if tty else _Lines(sys.stderr), **kw)


def build(cfg, device):
    grid = LiftedGrid(cfg["n"], cfg["n_theta"])
    data = SyntheticTomography(grid, noise=cfg["noise"], background=cfg["background"], device=device,
                               sigma_cells=cfg["sigma_cells"], aspect=cfg["aspect"],
                               detrend=cfg["detrend"])
    model = LiftedLPD(grid, n_iter=cfg["n_iter"], hidden=cfg["hidden"], J_max=cfg["j_max"],
                      image_input=cfg["image_input"], contrast=cfg["contrast"]).to(device)
    fwd = {"sinogram": MicrolocalForwardModel(CanonicalRelation(grid), data.M.fibre_response()).to(device),
           "image": OrientedEdgeForwardModel(grid, data.A.angular_response(),
                                             cfg["edge_width"] or cfg["sigma_cells"],
                                             sigma_coarse=data.A.sigma_coarse).to(device)}
    prior = CurvePrior(grid, xi=cfg["xi"], nu=cfg["nu"], zeta=cfg["zeta"], eps=1e-3,
                       length=cfg["length"], beta=cfg["beta"]).to(device)
    return grid, data, model, fwd, prior


def run_model(model, batch, amp):
    """Network in bf16 autocast (if enabled); outputs and losses in fp32."""
    c = batch["c"]
    b = batch["b"] if model.image_input else None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and c.is_cuda):
        out = model(c, b)
    return {k: v.float() for k, v in out.items()}


def data_term(q, batch, fwd, disc, cfg):
    """D_data(q) for the configured data term, plus its parts for logging."""
    parts = {}
    if cfg["data_term"] in ("image", "both"):
        parts["D_img"] = disc(fwd["image"](q), batch["b"])
    if cfg["data_term"] in ("sinogram", "both"):
        parts["D_sino"] = disc(fwd["sinogram"](q), batch["c"])
    D = parts.get("D_img", 0) + (cfg["mu"] if cfg["data_term"] == "both" else 1) * parts.get("D_sino", 0)
    return D, parts


def objective(out, batch, fwd, disc, prior, cfg):
    D, parts = data_term(out["q"], batch, fwd, disc, cfg)
    E = prior(out["a"], out["b"], out["c"]).mean()
    terms = {"D": D.item(), "E": E.item(), **{k: v.item() for k, v in parts.items()}}
    return D + cfg["lam"] * E, terms


@torch.no_grad()
def validate(model, val, fwd, disc, prior, cfg, C, path=None, amp=True):
    model.eval()
    aucs, base, base_b, logs, ms = [], [], [], [], []
    for i in range(val["c"].shape[0]):
        batch = {k: v[i:i + 1] for k, v in val.items()}
        out = run_model(model, batch, amp)
        _, terms = objective(out, batch, fwd, disc, prior, cfg)
        logs.append(terms)
        m_true = val["membership"][i, 0].numpy()
        pos, neg = label_masks(m_true)
        aucs.append(roc_auc(out["m"][0, 0].cpu().numpy(), pos, neg))
        u_base = C.inverse(batch["c"])[0, 0].cpu().numpy()
        base.append(roc_auc(u_base, pos, neg))
        b_np = batch["b"][0, 0].cpu().numpy()
        base_b.append(roc_auc(b_np, pos, neg))
        ms.append(out["m"][0, 0])
    trues = [val["wf_thin"][i, 0].numpy() for i in range(len(ms))]
    f_m = best_f(ms, trues, C.grid)
    f_b = best_f([val["b"][i, 0] for i in range(len(ms))], trues, C.grid)
    if path is not None:
        m_np, b_np = ms[0].cpu().numpy(), val["b"][0, 0].cpu().numpy()
        thin_m = thin_estimate(ms[0], C.grid, f_m["threshold"]).cpu().numpy()
        thin_b = thin_estimate(val["b"][0, 0], C.grid, f_b["threshold"]).cpu().numpy()
        panels = [(val["f"][0, 0].numpy(), "phantom"),
                  (orientation_rgb(trues[0].astype(float)), "true WF (thin, hue = θ)"),
                  (orientation_rgb(C.inverse(val["c"][:1])[0, 0].cpu().numpy()),
                   f"sinogram baseline $C^{{-1}}c$  AUC {base[0]:.3f}"),
                  (orientation_rgb(b_np), f"image baseline $b$  AUC {base_b[0]:.3f}"),
                  (orientation_rgb(m_np), f"LPD m (max {m_np.max():.2f})  AUC {aucs[0]:.3f}"),
                  (orientation_rgb((thin_b > 0).astype(float)), f"ridges of $b$  F {f_b['F']:.3f}"),
                  (orientation_rgb((thin_m > 0).astype(float)), f"ridges of LPD m  F {f_m['F']:.3f}")]
        fig, ax = plt.subplots(2, 4, figsize=(16, 8.6))
        ax = ax.ravel()
        ax[-1].axis("off")
        for a, (img, title) in zip(ax, panels):
            im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
            a.imshow(im, origin="lower", extent=[-1, 1, -1, 1], cmap="gray" if img.ndim == 2 else None)
            a.set_title(title, fontsize=10)
            a.set_xticks([]), a.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=90)
        plt.close(fig)
    model.train()
    rec = {"val_auc": float(np.mean(aucs)), "val_auc_baseline": float(np.mean(base)),
           "val_auc_b": float(np.mean(base_b)),
           "val_F": f_m["F"], "val_F_x": f_m["F_x"], "val_P": f_m["P"], "val_R": f_m["R"],
           "val_F_b": f_b["F"], "val_F_x_b": f_b["F_x"]}
    for k in logs[0]:
        rec[f"val_{k}"] = float(np.mean([l[k] for l in logs]))
    return rec


def main(argv=None):
    args = parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume:
        out_dir = Path(args.resume)
        cfg = {**LEGACY_DEFAULTS, **json.loads((out_dir / "config.json").read_text())}
        cfg["steps"] = args.steps
    else:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {k: v for k, v in vars(args).items() if k not in ("out", "resume")}
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    amp = not cfg.get("no_amp", False)
    grid, data, model, fwd, prior = build(cfg, dev)
    disc = ForwardModelDiscrepancy(delta=1e-2)
    C = CanonicalRelation(grid).to(dev)
    print(f"data term: {cfg['data_term']}, image input: {cfg['image_input']}, "
          f"{sum(p.numel() for p in model.parameters())} parameters, device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
    step = 0
    if args.resume and (out_dir / "checkpoint.pt").exists():
        ck = torch.load(out_dir / "checkpoint.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]

    val = data.sample(np.random.default_rng(12345), cfg["val_size"], labels=True)
    rng = np.random.default_rng(cfg["seed"] + step)
    log = open(out_dir / "log.jsonl", "a")
    pbar = progress(total=cfg["steps"], initial=step, desc=out_dir.name)
    t0 = time.time()
    while step < cfg["steps"]:
        t_start = time.time()
        batch = data.sample(rng, cfg["batch"])
        t_data = time.time()
        out = run_model(model, batch, amp)
        loss, terms = objective(out, batch, fwd, disc, prior, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        loss = float(loss)                     # synchronizes, so the timings are real
        t_end = time.time()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss:.4f}", D=f"{terms['D']:.4f}", E=f"{terms['E']:.2f}",
                         lr=f"{sched.get_last_lr()[0]:.1e}",
                         data=f"{t_data - t_start:.2f}s", net=f"{t_end - t_data:.2f}s", refresh=False)
        if step % 25 == 0:
            rec = {"step": step, "loss": loss, **terms, "time": time.time() - t0,
                   "sec_per_step": t_end - t_start, "sec_data": t_data - t_start}
            log.write(json.dumps(rec) + "\n"), log.flush()
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            rec = {"step": step, **validate(model, val, fwd, disc, prior, cfg, C,
                                            out_dir / f"val_{step:06d}.png", amp)}
            log.write(json.dumps(rec) + "\n"), log.flush()
            pbar.write(json.dumps(rec))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "config": cfg},
                       out_dir / "checkpoint.pt")
    pbar.close()


if __name__ == "__main__":
    main()
