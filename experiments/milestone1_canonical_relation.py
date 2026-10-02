"""Milestone 1: numerical check of the canonical-relation pipeline.

    phantom f with analytic WF(f)  --ODL-->  sinogram y = R f + noise
    y  --M-->  lifted sinogram coefficients c(s, phi, t)
    WF(f)  --C-->  predicted lifted sinogram WF

and the classical, learning-free WF estimate C^{-1} M y on the image side.
Also evaluates the curve prior on the true flux of the phantom.

Run from the repository root:

    python -m experiments.milestone1_canonical_relation [--n 128] [--noise 0.01]

Figures and a metrics summary are written to outputs/milestone1/.
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

from data.phantoms import (CurveSamples, Ellipse, Phantom, Polygon, lifted_flux,
                           lifted_membership)
from evaluation.metrics import label_masks, roc_auc, t_localization
from geometry.lifted import LiftedGrid
from geometry.tomo import image_space, parallel_geometry, sinogram_axes
from loss.data_discrepancy import MixtureLikelihood
from loss.regularization import CurvePrior
from operators.canonical_relation import CanonicalRelation
from operators.microlocal import SinogramAnalyzer
from operators.radon import RayTransform


def default_phantom():
    return Phantom([
        Ellipse((0.0, 0.0), (0.75, 0.6), 0.2, 0.5),
        Ellipse((-0.25, 0.15), (0.3, 0.18), 0.6, 0.6),
        Ellipse((0.3, -0.25), (0.12, 0.22), -0.4, -0.4),
        Polygon([[0.05, 0.3], [0.45, 0.2], [0.3, 0.5]], 0.4),
        Polygon([[-0.45, -0.35], [-0.15, -0.4], [-0.1, -0.15], [-0.4, -0.1]], -0.3),
    ])


def orientation_rgb(field, grid):
    """Max over theta as brightness, argmax theta as hue (orientation mod pi)."""
    mx = field.max(axis=0)
    hue = field.argmax(axis=0) / grid.n_theta
    hsv = np.stack([hue, np.ones_like(hue), mx / (mx.max() + 1e-12)], -1)
    return matplotlib.colors.hsv_to_rgb(hsv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n-theta", type=int, default=64)
    ap.add_argument("--n-angles", type=int, default=400)
    ap.add_argument("--noise", type=float, default=0.01,
                    help="sinogram noise std relative to max |y|")
    ap.add_argument("--xi", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/milestone1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    n = args.n
    grid = LiftedGrid(n=n, n_theta=args.n_theta)
    phantom = default_phantom()

    # data
    geom = parallel_geometry(n_angles=args.n_angles, n_det=2 * n)
    R = RayTransform(image_space(n), geom)
    f = torch.as_tensor(phantom.render(n), dtype=torch.float32)
    y_clean = R(f[None, None]).to(dev)
    sigma = args.noise * float(y_clean.abs().max())
    y = y_clean + sigma * torch.randn_like(y_clean)

    # operators
    phis, s_det = sinogram_axes(geom)
    M = SinogramAnalyzer(grid, phis, s_det).to(dev)
    C = CanonicalRelation(grid).to(dev)

    with torch.no_grad():
        c = M(y)
        sigma0 = M.coefficient_noise_std(sigma, device=dev)
        lik = MixtureLikelihood(sigma0.cpu(), edge_scale=float(c.quantile(0.999)) / 2).to(dev)
        post = lik.posterior(c, prior=0.05)       # learning-free WF posterior, sinogram side
        u_post = C.inverse(post)                  # ... pulled back to the image side
        u_raw = C.inverse(c)                      # raw coefficients pulled back

    # ground truth
    # t-localization on smooth edges only (a straight edge maps to a single
    # sinogram point whose WF contains a whole interval of t)
    cs_smooth = CurveSamples.concat([sh.samples(grid.h / 2) for sh in phantom.shapes
                                     if isinstance(sh, Ellipse)])
    m_true = torch.as_tensor(lifted_membership(grid, phantom.samples(grid.h / 2, "full")),
                             dtype=torch.float32, device=dev)
    v_true = C(m_true[None, None])[0, 0]

    c_np = c[0, 0].cpu().numpy()
    u_raw_np, u_post_np = u_raw[0, 0].cpu().numpy(), u_post[0, 0].cpu().numpy()
    v_np, m_np = v_true.cpu().numpy(), m_true.cpu().numpy()

    loc = t_localization(c_np, grid, cs_smooth.x, cs_smooth.theta, window=8)
    pos_s, neg_s = label_masks(v_np)
    pos_i, neg_i = label_masks(m_np)
    metrics = {
        "noise_sigma": sigma,
        "t_localization_median_abs_error_cells": float(np.median(np.abs(loc["local_error"]))),
        "t_localization_mean_error_cells": float(np.mean(loc["local_error"])),
        "t_global_hit_rate": loc["global_hit_rate"],
        "auc_sinogram_side_coefficients": roc_auc(c_np, pos_s, neg_s),
        "auc_image_side_pulled_back_coefficients": roc_auc(u_raw_np, pos_i, neg_i),
        "auc_image_side_pulled_back_posterior": roc_auc(u_post_np, pos_i, neg_i),
        "posterior_fraction_above_0.5_image_side": float(np.mean(u_post_np > 0.5)),
    }

    # curve prior on the true flux
    prior = CurvePrior(grid, xi=args.xi, nu=1.0).double()
    cs_arc = phantom.samples(grid.h / 4, corners="arc")
    a, b, c_ = (torch.as_tensor(x) for x in lifted_flux(grid, cs_arc))
    cs_none = phantom.samples(grid.h / 4, corners="none")
    a0, b0, c0 = (torch.as_tensor(x) for x in lifted_flux(grid, cs_none))
    metrics["prior_length_term_zeta1"] = float(CurvePrior(grid, xi=args.xi, zeta=1.0)
                                               .double().mass(a, b, c_))
    metrics["prior_length_analytic"] = cs_arc.subriemannian_length(args.xi)
    metrics["prior_endpoints_closed_curves"] = float(prior.boundary_mass(a, b, c_))
    metrics["prior_endpoints_closed_curves_without_c"] = float(prior.boundary_mass(a, b))
    metrics["prior_endpoints_without_corner_arcs"] = float(prior.boundary_mass(a0, b0, c0))
    n_edges = sum(len(sh.vertices) for sh in phantom.shapes if isinstance(sh, Polygon))
    metrics["expected_endpoints_without_corner_arcs"] = 2 * n_edges

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- figures ---------------------------------------------------------
    ext = [-1, 1, -1, 1]
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))
    ax[0, 0].imshow(f.numpy().T, origin="lower", extent=ext, cmap="gray")
    ax[0, 0].set_title("phantom f")
    ax[0, 1].imshow(y[0, 0].cpu().numpy().T, origin="lower", aspect="auto", cmap="gray",
                    extent=[0, math.pi, s_det[0], s_det[-1]])
    ax[0, 1].set_title(r"sinogram $y$ ($\varphi$ horizontal)")
    ax[0, 2].imshow(c_np.max(axis=2).T, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, math.pi, -1, 1])
    ax[0, 2].set_title(r"$\max_t\, M y$")
    ax[1, 0].imshow(orientation_rgb(m_np, grid).transpose(1, 0, 2), origin="lower", extent=ext)
    ax[1, 0].set_title(r"true WF (hue = $\theta$)")
    ax[1, 1].imshow(orientation_rgb(u_raw_np, grid).transpose(1, 0, 2), origin="lower", extent=ext)
    ax[1, 1].set_title(r"$C^{-1} |My|$  (no learning, hue = $\theta$)")
    ax[1, 2].imshow(u_post_np.max(axis=0).T, origin="lower", extent=ext, cmap="magma",
                    vmin=0, vmax=1)
    ax[1, 2].set_title(r"$\max_\theta\, C^{-1}$ posterior: tangent-line ambiguity")
    for a_ in ax.flat:
        a_.set_xticks([]), a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)

    # lifted sinogram slices: coefficients vs prediction in the (s, t) plane
    ks = np.linspace(0, grid.n_theta, 4, endpoint=False).astype(int)
    fig, ax = plt.subplots(2, len(ks), figsize=(4 * len(ks), 8))
    for col, k in enumerate(ks):
        ax[0, col].imshow(c_np[k].T, origin="lower", extent=ext, cmap="magma")
        ax[0, col].set_title(rf"$|My|$, $\varphi$={math.degrees(grid.thetas()[k]):.0f}°  (s →, t ↑)")
        ax[1, col].imshow(v_np[k].T, origin="lower", extent=ext, cmap="magma")
        ax[1, col].set_title(r"predicted $C\,\mathrm{WF}(f)$")
    for a_ in ax.flat:
        a_.set_xticks([]), a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "lifted_sinogram_slices.png", dpi=110)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
