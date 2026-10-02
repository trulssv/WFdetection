"""Unsupervised training of the lifted LPD network on synthetic tomography data.

    loss = D(R_t C (J m); c) + lambda E(a, b, c)          (research/background.md, Sec. 7)

D is the Laplace forward-model discrepancy on the normalized microlocal
coefficients c = M y, E the curve prior.  No wavefront-set labels are used for
training; the analytic labels of the phantoms are used only for validation
(ROC AUC of the membership m, compared with the learning-free pull-back
C^{-1} c).

Run from the repository root:

    python -m experiments.train_lpd --steps 4000 --out outputs/train/run1
    python -m experiments.train_lpd --resume outputs/train/run1   # continue

Writes config.json, log.jsonl, checkpoint.pt and validation figures val_<step>.png.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.synthetic import SyntheticTomography
from evaluation.metrics import label_masks, roc_auc
from evaluation.visualize import orientation_rgb
from geometry.lifted import LiftedGrid
from loss.data_discrepancy import ForwardModelDiscrepancy, MicrolocalForwardModel
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
    ap.add_argument("--xi", type=float, default=0.1)
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--zeta", type=float, default=2.0)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast in the network")
    return ap.parse_args(argv)


def build(cfg, device):
    grid = LiftedGrid(cfg["n"], cfg["n_theta"])
    data = SyntheticTomography(grid, noise=cfg["noise"], background=cfg["background"], device=device)
    model = LiftedLPD(grid, n_iter=cfg["n_iter"], hidden=cfg["hidden"]).to(device)
    fwd = MicrolocalForwardModel(CanonicalRelation(grid), data.M.fibre_response()).to(device)
    prior = CurvePrior(grid, xi=cfg["xi"], nu=cfg["nu"], zeta=cfg["zeta"], eps=1e-3).to(device)
    return grid, data, model, fwd, prior


def run_model(model, c, amp):
    """Network in bf16 autocast (if enabled); outputs and losses in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and c.is_cuda):
        out = model(c)
    return {k: v.float() for k, v in out.items()}


def objective(out, c, fwd, disc, prior, lam):
    c_hat = fwd(out["J"] * out["m"])
    D = disc(c_hat, c)
    E = prior(out["a"], out["b"], out["c"]).mean()
    return D + lam * E, {"D": D.item(), "E": E.item()}


@torch.no_grad()
def validate(model, val, fwd, disc, prior, lam, grid, C, path=None, amp=True):
    model.eval()
    aucs, base, logs = [], [], []
    for i in range(val["c"].shape[0]):
        c = val["c"][i:i + 1]
        out = run_model(model, c, amp)
        _, terms = objective(out, c, fwd, disc, prior, lam)
        logs.append(terms)
        m_true = val["membership"][i, 0].numpy()
        pos, neg = label_masks(m_true)
        aucs.append(roc_auc(out["m"][0, 0].cpu().numpy(), pos, neg))
        u_base = C.inverse(c)[0, 0].cpu().numpy()
        base.append(roc_auc(u_base, pos, neg))
        if i == 0 and path is not None:
            fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
            ext = [-1, 1, -1, 1]
            ax[0].imshow(val["f"][0, 0].numpy().T, origin="lower", extent=ext, cmap="gray")
            ax[0].set_title("phantom")
            ax[1].imshow(orientation_rgb(m_true).transpose(1, 0, 2), origin="lower", extent=ext)
            ax[1].set_title("true WF (hue = θ)")
            ax[2].imshow(orientation_rgb(u_base).transpose(1, 0, 2), origin="lower", extent=ext)
            ax[2].set_title(f"baseline $C^{{-1}}c$  AUC {base[-1]:.3f}")
            m_np = out["m"][0, 0].cpu().numpy()
            ax[3].imshow(orientation_rgb(m_np).transpose(1, 0, 2), origin="lower", extent=ext)
            ax[3].set_title(f"LPD m (max {m_np.max():.2f})  AUC {aucs[-1]:.3f}")
            for a in ax:
                a.set_xticks([]), a.set_yticks([])
            fig.tight_layout()
            fig.savefig(path, dpi=90)
            plt.close(fig)
    model.train()
    return {"val_auc": float(np.mean(aucs)), "val_auc_baseline": float(np.mean(base)),
            "val_D": float(np.mean([l["D"] for l in logs])),
            "val_E": float(np.mean([l["E"] for l in logs]))}


def main(argv=None):
    args = parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume:
        out_dir = Path(args.resume)
        cfg = json.loads((out_dir / "config.json").read_text())
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
    t0 = time.time()
    while step < cfg["steps"]:
        batch = data.sample(rng, cfg["batch"])
        out = run_model(model, batch["c"], amp)
        loss, terms = objective(out, batch["c"], fwd, disc, prior, cfg["lam"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        step += 1
        if step % 25 == 0:
            rec = {"step": step, "loss": float(loss), **terms, "time": time.time() - t0}
            log.write(json.dumps(rec) + "\n"), log.flush()
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            rec = {"step": step, **validate(model, val, fwd, disc, prior, cfg["lam"], grid, C,
                                            out_dir / f"val_{step:06d}.png", amp)}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "config": cfg},
                       out_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
