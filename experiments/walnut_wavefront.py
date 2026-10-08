"""Wavefront set of the FIPS walnut from its fan-beam data.

Pipeline (research/background.md, Sections 4.3 and 5):

    fan sinogram g(beta, u)  --rebin (Prop. 4.7)-->  y(phi, s)  --M-->  c(phi, s, t)
                                                       y  --W FBP-->  b(theta, x)
    learning-free estimates:  C^{-1} c (sinogram analyzer), b (full aperture)
    learned estimate:         LPD(c, b), optionally fine-tuned on this sample by
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
from evaluation.ridges import thin_estimate
from evaluation.visualize import orientation_rgb
from experiments.train_lpd import LEGACY_DEFAULTS, data_term, progress
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from loss.data_discrepancy import (ForwardModelDiscrepancy, MicrolocalForwardModel,
                                   OrientedEdgeForwardModel)
from loss.regularization import CurvePrior
from models.lpd import LiftedLPD
from operators.canonical_relation import CanonicalRelation
from operators.fan_beam import FanToParallel
from operators.image_analyzer import ImageDomainAnalyzer
from operators.microlocal import SinogramAnalyzer
from operators.radon import RayTransform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="default: 192, or the checkpoint's n")
    ap.add_argument("--n-theta", type=int, default=None)
    ap.add_argument("--det-binning", type=int, default=4)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--finetune", type=int, default=0, help="unsupervised steps on the walnut")
    ap.add_argument("--detrend", type=float, default=4.0, help="image analyzer detrending (no checkpoint)")
    ap.add_argument("--ridge-threshold", type=float, default=0.1,
                    help="thin estimates: ridges above this fraction of the maximum")
    ap.add_argument("--out", default="outputs/walnut")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=dev)
        cfg = {**LEGACY_DEFAULTS, **ck["config"]}
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
    # Lines with |s| > s_fov are not measured: the truncation is a jump in y
    # and would show up as a spurious wavefront set on the circle |x| = s_fov.
    s_fov = fan.src_radius * math.sin(math.atan(us.max() / (fan.src_radius + fan.det_radius)))
    margin = 3 * M.sigma_t + 2 * grid.h
    s_lift = grid.coords_torch(device=dev)
    fov = (s_lift.abs() < s_fov - margin).float().view(1, 1, 1, -1, 1)
    # image side: the FBP has a jump on the circle |x| = s_fov, mask it the same way
    A = ImageDomainAnalyzer(grid, cfg["sigma_cells"] if cfg else 1.5, cfg["aspect"] if cfg else 2.0,
                            cfg["detrend"] if cfg else args.detrend).to(dev)
    x = grid.coords_torch(device=dev)
    r = torch.sqrt(x[:, None]**2 + x[None, :]**2)
    fov_img = (r < s_fov - (3 * A.sigma_t + 2) * grid.h).float()
    with torch.no_grad():
        c = normalize_coefficients(M(y) * fov)
        base = C.inverse(c)[0, 0].cpu().numpy()
        b = normalize_coefficients(A(torch.as_tensor(fbp, device=dev)[None, None]) * fov_img)
        base_b = b[0, 0].cpu().numpy()

    results = {"n": n, "n_theta": K, "s_fov": s_fov}
    m = None
    if cfg is not None:
        model = LiftedLPD(grid, n_iter=cfg["n_iter"], hidden=cfg["hidden"], J_max=cfg["j_max"],
                          image_input=cfg["image_input"], contrast=cfg["contrast"]).to(dev)
        model.load_state_dict(ck["model"])
        fwd = {"sinogram": MicrolocalForwardModel(C, M.fibre_response()).to(dev),
               "image": OrientedEdgeForwardModel(grid, A.angular_response(),
                                                 cfg["edge_width"] or cfg["sigma_cells"],
                                                 sigma_coarse=A.sigma_coarse).to(dev)}
        prior = CurvePrior(grid, xi=cfg["xi"], nu=cfg["nu"], zeta=cfg["zeta"], eps=1e-3,
                           length=cfg["length"], beta=cfg["beta"]).to(dev)
        disc = ForwardModelDiscrepancy(delta=1e-2)
        batch = {"c": c, "b": b}

        def objective():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(c, b if cfg["image_input"] else None)
            o = {k: v.float() for k, v in o.items()}
            D, _ = data_term(o["q"], batch, fwd, disc, cfg)
            E = prior(o["a"], o["b"], o["c"]).mean()
            return o, D, E

        if args.finetune:
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)
            model.train()
            pbar = progress(iterable=range(args.finetune), desc="finetune")
            for _ in pbar:
                _, D, E = objective()
                loss = D + cfg["lam"] * E
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}", D=f"{D.item():.4f}", E=f"{E.item():.2f}")
        model.eval()
        with torch.no_grad():
            o, D, E = objective()
        m = o["m"][0, 0].cpu().numpy()
        fields = {k: o[k][0, 0].cpu().numpy() for k in ("a", "b", "c", "q", "J") if k in o}
        results.update({"D": D.item(), "E": E.item(), "finetune_steps": args.finetune})

    ref = load_reference(n=n)
    t = args.ridge_threshold
    ridge_b = (thin_estimate(torch.as_tensor(base_b), grid, t) > 0).numpy().astype(float)
    ridge_m = None if m is None else (thin_estimate(torch.as_tensor(m), grid, t) > 0).numpy().astype(float)

    ext = [-1, 1, -1, 1]
    panels = [("reference FBP (1200 proj., dataset)", ref, "gray"),
              ("FBP of rebinned data", fbp, "gray"),
              (r"learning-free $C^{-1}c$ (hue = θ)", orientation_rgb(base), None),
              (r"learning-free full-aperture $b$", orientation_rgb(base_b), None),
              (r"ridges of $b$ (thin)", orientation_rgb(ridge_b), None)]
    if m is not None:
        panels.append(("LPD membership m" + (f" (+{args.finetune} unsup. steps)" if args.finetune else ""),
                       orientation_rgb(m, vmax=1), None))
        panels.append(("ridges of LPD m (thin)", orientation_rgb(ridge_m), None))
    cols = 3 if len(panels) <= 6 else 4
    rows = math.ceil(len(panels) / cols)
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 5.2 * rows))
    ax = ax.ravel()
    for a in ax[len(panels):]:
        a.axis("off")
    for a, (title, img, cmap) in zip(ax, panels):
        im = img.T if img.ndim == 2 else img.transpose(1, 0, 2)
        a.imshow(im, origin="upper", extent=ext, cmap=cmap)
        a.set_title(title)
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "walnut_wavefront.png", dpi=110)
    np.save(out / "baseline.npy", base)
    np.save(out / "baseline_b.npy", base_b)
    np.save(out / "ridges_b.npy", ridge_b.astype(bool))
    if m is not None:
        np.save(out / "membership.npy", m)
        np.save(out / "ridges_m.npy", ridge_m.astype(bool))
        # the current tau = a X1 + c X2 + b X3 and the contrast J, each (K, n, n) on (theta, x)
        np.savez_compressed(out / "fields.npz", **fields)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"written to {out}/ (images shown with origin='upper', matching the dataset's figures)")


if __name__ == "__main__":
    main()
