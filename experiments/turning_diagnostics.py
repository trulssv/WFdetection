"""Diagnostics of the turning component b of trained LPD networks.

    python -m experiments.turning_diagnostics outputs/train/run1 outputs/train/run2 ... [--out fig.png]

On the fixed validation set (seed 12345; ``--n-val`` samples, the same
phantoms for every run whatever its K) and on voxels of the predicted band
(m > ``--m-min``), reports per run:

* ``theta_conc``: fraction of a's mass at a pixel within +-3 pi / 32 of
  its argmax angle (+-3 bins at K = 32; 1 = sharp in theta);
* ``dipole_corr``: correlation of X1 a and -d_theta b.  The prior's endpoint
  term wants div tau = X1 a + d_theta b + X2 c ~ 0, and a band blurred in
  theta is closed by a b-dipole across the ridge.  The true blurred flux of
  a curved edge also has div tau = 0, so a high value alone does not mean
  b pays for blur; read it together with ``b_corr_true``;
* ``div_x1a``, ``div_tau``: mean |X1 a| and mean |div tau| (how much the
  dipole relieves the endpoint term);
* ``b_corr_true``: correlation of b with the true b = a kappa of the phantom
  (``data.phantoms.lifted_flux``, 2-cell blur), on the union of both bands;
* ``kappa_corr``, ``kappa_mae``: curvature kappa = sum b / sum a over
  +-3 pi / 32 around the true normal, at the true (thin) wavefront points away
  from polygon corners, against the same from the true flux.  The dipole is
  odd across the ridge, so it cancels here; this measures whether the centre
  line carries the curvature.

The optional figure shows, for validation sample 0, m and b at the argmax
angle of a for each run (the "turning plot").
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.phantoms import lifted_flux, random_phantom
from evaluation.metrics import label_masks, roc_auc
from evaluation.ridges import best_f
from experiments.train_lpd import LEGACY_DEFAULTS, build, run_model


def _corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def _window_sum(u, k, w=3):
    """sum over theta bins k - w .. k + w (cyclic) of u (K, n, n) at index field k (n, n);
    b is antiperiodic, so wrapped bins enter with a sign through ``sign``."""
    K = u.shape[0]
    return sum(torch.take_along_dim(u, ((k + o) % K)[None], 0)[0] for o in range(-w, w + 1))


def _window_sum_odd(u, k, w=3):
    K = u.shape[0]
    tot = 0
    for o in range(-w, w + 1):
        j = k + o
        sign = torch.where((j < 0) | (j >= K), -1.0, 1.0)
        tot = tot + sign * torch.take_along_dim(u, (j % K)[None], 0)[0]
    return tot


@torch.no_grad()
def diagnose(run: Path, dev, m_min=0.1, n_val=None):
    """Also returns ROC AUC and ridge F of m on the same samples (``auc``, ``F``)."""
    ck = torch.load(run / "checkpoint.pt", map_location=dev)
    cfg = {**LEGACY_DEFAULTS, **ck["config"]}
    grid, data, model, fwd, prior = build(cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    n_val = n_val or cfg["val_size"]
    val = data.sample(np.random.default_rng(12345), n_val, labels=True)
    rng = np.random.default_rng(12345)                 # phantoms are drawn first in sample()
    phantoms = [random_phantom(rng) for _ in range(n_val)]
    K = grid.n_theta
    w = max(round(3 * K / 32), 1)                      # +-3 pi / 32 in bins
    win, win_odd = (lambda u, k: _window_sum(u, k, w)), (lambda u, k: _window_sum_odd(u, k, w))
    aucs, ms, trues = [], [], []
    acc = {k: [] for k in ("conc_num", "conc_den", "x1a", "dth", "x1a_abs", "div_abs",
                           "b", "b_true", "kap", "kap_true")}
    first = None
    for i in range(n_val):
        batch = {k: v[i:i + 1].to(dev) if torch.is_tensor(v) else v for k, v in val.items()}
        o = run_model(model, batch, True)
        a, b, c, m = o["a"][0, 0], o["b"][0, 0], o["c"][0, 0], o["m"][0, 0]
        band = m > m_min
        pix = band.any(0)
        kstar = a.argmax(0)
        pos, neg = label_masks(val["membership"][i, 0].numpy())
        aucs.append(roc_auc(m.cpu().numpy(), pos, neg))
        ms.append(m), trues.append(val["wf_thin"][i, 0].numpy())
        acc["conc_num"].append(win(a, kstar)[pix])
        acc["conc_den"].append(a.sum(0)[pix])
        x1a = prior.x1_derivative(a)
        dth = prior.theta_derivative(b)
        acc["x1a"].append(x1a[band]), acc["dth"].append(-dth[band])
        acc["x1a_abs"].append(x1a[band].abs()), acc["div_abs"].append(prior.divergence(a, b, c)[band].abs())

        cs = phantoms[i].samples(grid.h / 2, "full")
        at, bt, _ = (torch.as_tensor(f, dtype=torch.float32, device=dev) for f in lifted_flux(grid, cs))
        union = band | (at > 0.1 * at.max())
        acc["b"].append(b[union]), acc["b_true"].append(bt[union])

        # curvature at true thin WF points with a single true normal (no corners, no crossings)
        thin = val["wf_thin"][i, 0].to(dev).bool()
        single = thin.sum(0) == 1
        ktrue = thin.float().argmax(0)
        den_t = win(at, ktrue)
        den = win(a, ktrue)
        sel = single & (den_t > 0.2 * at.max()) & (den > 0.05 * den.max())
        acc["kap"].append((win_odd(b, ktrue) / den)[sel])
        acc["kap_true"].append((win_odd(bt, ktrue) / den_t)[sel])
        if first is None:
            first = {"m": m.cpu().numpy(), "b_at": torch.take_along_dim(b, kstar[None], 0)[0].cpu().numpy(),
                     "mask": pix.cpu().numpy()}
    cat = {k: torch.cat(v) for k, v in acc.items()}
    kap, kap_t = cat["kap"], cat["kap_true"]
    res = {"K": K, "n_val": n_val, "auc": float(np.mean(aucs)), "F": best_f(ms, trues, grid)["F"],
           "theta_conc": float(cat["conc_num"].sum() / cat["conc_den"].sum()),
           "dipole_corr": _corr(cat["x1a"], cat["dth"]),
           "div_x1a": float(cat["x1a_abs"].mean()), "div_tau": float(cat["div_abs"].mean()),
           "b_corr_true": _corr(cat["b"], cat["b_true"]),
           "kappa_corr": _corr(kap, kap_t), "kappa_mae": float((kap - kap_t).abs().median()),
           "kappa_true_mad": float(kap_t.abs().median()), "n_kappa": int(kap.numel())}
    log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    last = [r for r in log if "val_auc" in r][-1]
    res.update({k: last[k] for k in ("step", "val_D", "val_E") if k in last})
    return res, first


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--m-min", type=float, default=0.1)
    ap.add_argument("--n-val", type=int, default=8, help="validation samples (same phantoms for all runs)")
    ap.add_argument("--out", default=None, help="figure path (turning plot of sample 0)")
    ap.add_argument("--json", default=None, help="write the table as JSON")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    table, figs = {}, {}
    for r in args.runs:
        table[r], figs[r] = diagnose(Path(r), dev, args.m_min, args.n_val)
        print(Path(r).name, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in table[r].items()}), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))
    if args.out:
        fig, ax = plt.subplots(2, len(args.runs), figsize=(4 * len(args.runs), 8), squeeze=False)
        for j, r in enumerate(args.runs):
            f = figs[r]
            ax[0, j].imshow(f["m"].max(0).T, origin="lower", cmap="gray")
            ax[0, j].set_title(f"{Path(r).name}\nmax$_\\theta$ m", fontsize=9)
            v = np.abs(f["b_at"][f["mask"]]).max() if f["mask"].any() else 1
            ax[1, j].imshow(np.where(f["mask"], f["b_at"], 0).T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            t = table[r]
            ax[1, j].set_title(f"b at argmax$_\\theta$ a\nF {t['F']:.2f}  conc {t['theta_conc']:.2f}  dipole {t['dipole_corr']:.2f}"
                               f"  κ-corr {t['kappa_corr']:.2f}", fontsize=9)
            for a in ax[:, j]:
                a.set_xticks([]), a.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out, dpi=90)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
