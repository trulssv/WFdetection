"""Wavefront set of the FIPS walnut from its fan-beam data.

Pipeline (research/background.md, Sections 4.3 and 5):

    fan sinogram g(beta, u)  --rebin (Prop. 4.7)-->  y(phi, s)  --M-->  c(phi, s, t)
    learning-free estimate:   C^{-1} c
    learned estimate:         LPD(c), optionally fine-tuned on this sample by
                              minimizing the unsupervised objective D + lambda E

The 1200-projection scan is used: the analyzer needs dense angles.  The FBP
of the rebinned data and the dataset's high-resolution reference FBP are
shown for comparison (they are not used by the method).

Run from the repository root (data in data/raw/walnut, see data/README.md):

    python -m experiments.walnut_wavefront                       # baseline only
    python -m experiments.walnut_wavefront --checkpoint outputs/train/run/checkpoint.pt \\
        [--finetune 200]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.synthetic import normalize_coefficients
from data.walnut import load_reference, load_walnut
from evaluation.visualize import orientation_rgb
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from loss.data_discrepancy import ForwardModelDiscrepancy, MicrolocalForwardModel
from loss.regularization import CurvePrior
from models.lpd import LiftedLPD
from operators.canonical_relation import CanonicalRelation
from operators.fan_beam import FanToParallel
from operators.microlocal import SinogramAnalyzer
from operators.radon import RayTransform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="default: 192, or the checkpoint's n")
    ap.add_argument("--n-theta", type=int, default=None)
    ap.add_argument("--det-binning", type=int, default=4)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--finetune", type=int, default=0, help="unsupervised steps on the walnut")
    ap.add_argument("--out", default="outputs/walnut")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=dev)
        cfg = ck["config"]
    n = args.n or (cfg["n"] if cfg else 192)
    K = args.n_theta or (cfg["n_theta"] if cfg else 48)
    grid = LiftedGrid(n, K)

    # data: fan -> parallel (dense angles over [0, pi), symmetric detector)
    d = load_walnut(1200, det_binning=args.det_binning)
    fan = d["geometry"]
    betas, us = sinogram_axes(fan)
    par = parallel_geometry(n_angles=3 * n, n_det=2 * n, det_radius=1.0)
    phis, s = sinogram_axes(par)
    rebin = FanToParallel(betas, us, fan.src_radius, fan.det_radius, phis, s).to(dev)
    g = torch.as_tensor(d["sinogram"], device=dev)[None, None]
    y = rebin(g)
    fbp = RayTransform(image_space(n), par).fbp(y.cpu())[0, 0].numpy()

    M = SinogramAnalyzer(grid, phis, s).to(dev)
    C = CanonicalRelation(grid).to(dev)
    with torch.no_grad():
        c = normalize_coefficients(M(y))
        base = C.inverse(c)[0, 0].cpu().numpy()

    results = {"n": n, "n_theta": K}
    m = None
    if cfg is not None:
        model = LiftedLPD(grid, n_iter=cfg["n_iter"], hidden=cfg["hidden"]).to(dev)
        model.load_state_dict(ck["model"])
        fwd = MicrolocalForwardModel(C, M.fibre_response()).to(dev)
        prior = CurvePrior(grid, xi=cfg["xi"], nu=cfg["nu"], zeta=cfg["zeta"], eps=1e-3).to(dev)
        disc = ForwardModelDiscrepancy(delta=1e-2)

        def objective():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(c)
            o = {k: v.float() for k, v in o.items()}
            D = disc(fwd(o["J"] * o["m"]), c)
            E = prior(o["a"], o["b"], o["c"]).mean()
            return o, D, E

        if args.finetune:
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)
            model.train()
            for it in range(args.finetune):
                _, D, E = objective()
                loss = D + cfg["lam"] * E
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if (it + 1) % 50 == 0:
                    print(f"finetune {it + 1}: D={D.item():.4f} E={E.item():.2f}", flush=True)
        model.eval()
        with torch.no_grad():
            o, D, E = objective()
        m = o["m"][0, 0].cpu().numpy()
        results.update({"D": D.item(), "E": E.item(), "finetune_steps": args.finetune})

    ref = load_reference(n=n)

    ext = [-1, 1, -1, 1]
    panels = [("reference FBP (1200 proj., dataset)", ref, "gray"),
              ("FBP of rebinned data", fbp, "gray"),
              (r"learning-free $C^{-1}c$ (hue = θ)", orientation_rgb(base), None)]
    if m is not None:
        panels.append(("LPD membership m" + (f" (+{args.finetune} unsup. steps)" if args.finetune else ""),
                       orientation_rgb(m, vmax=1), None))
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.2))
    for a, (title, img, cmap) in zip(ax, panels):
        im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
        a.imshow(im, origin="upper", extent=ext, cmap=cmap)
        a.set_title(title)
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "walnut_wavefront.png", dpi=110)
    np.save(out / "baseline.npy", base)
    if m is not None:
        np.save(out / "membership.npy", m)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"written to {out}/ (images shown with origin='upper', matching the dataset's figures)")


if __name__ == "__main__":
    main()
