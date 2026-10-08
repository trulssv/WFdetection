"""Diagnostics for the canonical relation C and the LPD dual state."""
import sys
from pathlib import Path

ROOT = Path("/workspace/WFdetection")
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.metrics import roc_auc
from experiments.train_lpd import LEGACY_DEFAULTS, build
from geometry.tomo import sinogram_axes
from operators.canonical_relation import CanonicalRelation

torch.set_grad_enabled(False)
dev = "cuda"
ck = torch.load(ROOT / "outputs/train/n256_lam1e-4/checkpoint.pt", map_location=dev)
cfg = {**LEGACY_DEFAULTS, **ck["config"]}
torch.manual_seed(cfg["seed"])
grid, data, model, fwd, prior = build(cfg, dev)
model.load_state_dict(ck["model"]); model.eval()
n, K = grid.n, grid.n_theta
C = CanonicalRelation(grid).to(dev)

# ---------------- A: ray transform convention s = x . omega(phi) ----------------
cc = -1 + (np.arange(n) + 0.5) * 2 / n
X1, X2 = np.meshgrid(cc, cc, indexing="ij")
x0 = np.array([0.45, 0.2])
f = ((X1 - x0[0])**2 + (X2 - x0[1])**2 < 0.03**2).astype(np.float32)
y = data.R(torch.as_tensor(f)[None, None]).cpu().numpy()[0, 0]          # (n_angles, n_det)
phis, s_axis = sinogram_axes(data.geometry)
s_peak = s_axis[y.argmax(1)]
s_pred = x0[0] * np.cos(phis) + x0[1] * np.sin(phis)
print(f"[A] ray transform: max |s_peak - x0.omega(phi)| = {np.abs(s_peak - s_pred).max():.4f} "
      f"(detector cell {s_axis[1]-s_axis[0]:.4f}); "
      f"with omega -> (sin,cos): {np.abs(s_peak - (x0[0]*np.sin(phis)+x0[1]*np.cos(phis))).max():.3f}")

# ---------------- B: does C map true WF onto the peaks of c = M y ? ----------------
val = data.sample(np.random.default_rng(12345), 4, labels=True)
u = val["membership"].to(dev)                     # soft true WF on (theta, x1, x2)
c = val["c"].to(dev)                              # analyzer output on (phi, s, t)
Cu = C(u)
variants = {
    "as implemented": c,
    "t -> -t": c.flip(4),
    "s -> -s": c.flip(3),
    "s,t -> -s,-t": c.flip(3).flip(4),
    "swap s<->t": c.transpose(3, 4),
}
print("[B] AUC of c in sinogram space against C(true WF), per variant of the convention:")
for name, v in variants.items():
    aucs = []
    for i in range(u.shape[0]):
        lab = Cu[i, 0].cpu().numpy()
        aucs.append(roc_auc(v[i, 0].cpu().numpy(), lab >= 0.5, lab < 0.01))
    print(f"      {name:16s} AUC = {np.mean(aucs):.3f}")
# round trip
rt = C.inverse(C(u))
inside = torch.as_tensor((X1**2 + X2**2) < 0.95**2, device=dev)
err = ((rt - u)[..., inside].norm() / u[..., inside].norm()).item()
print(f"[B] round trip ||C^-1 C u - u|| / ||u|| inside unit disk = {err:.3f}")

# ---------------- C: raw dual state in its own coordinates ----------------
o = model(c[:1], val["b"][:1].to(dev), return_intermediates=True)
d = o["dual"][-1][0].float().norm(dim=0).cpu().numpy()       # (2K, n, n) on (phi in [0,2pi), s, t)
p = o["primal"][-1][0].float().norm(dim=0).cpu().numpy()     # (2K, n, n) on (theta, x1, x2)
yy = val["y"][0, 0].cpu().numpy()
cs = c[0, 0].cpu().numpy(); cus = Cu[0, 0].cpu().numpy()
print(f"[C] dual shape {d.shape}, primal shape {p.shape}")

fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
def sino(a, img, title, phimax=np.pi):
    a.imshow(img.T, origin="lower", aspect="auto", extent=[0, phimax, -1, 1], cmap="magma")
    a.set_title(title); a.set_xlabel(r"$\varphi$"); a.set_ylabel("$s$")
sino(ax[0, 0], yy, r"measured sinogram $y(\varphi,s)$")
ax[0, 0].images[0].set_extent([0, np.pi, s_axis[0], s_axis[-1]])
sino(ax[0, 1], cs.max(2), r"analyzer $\max_t\, c(\varphi,s,t)$")
sino(ax[0, 2], cus.max(2), r"true WF pushed forward $\max_t\, C u$")
sino(ax[0, 3], d.max(2), r"raw dual state $\max_t |d_5|(\varphi,s)$", 2 * np.pi)
k0 = K // 3
def sl(a, img, title, xl, yl):
    a.imshow(img.T, origin="lower", extent=[-1, 1, -1, 1], cmap="magma")
    a.set_title(title); a.set_xlabel(xl); a.set_ylabel(yl)
phi0 = k0 * np.pi / K
sl(ax[1, 0], cs[k0], rf"$c$ at fixed $\varphi={np.degrees(phi0):.0f}^\circ$: $(s,t)$ slice", "$s$", "$t$")
sl(ax[1, 1], d[k0], rf"$|d_5|$ at fixed $\varphi={np.degrees(phi0):.0f}^\circ$: $(s,t)$ slice", "$s$", "$t$")
sl(ax[1, 2], p[k0], rf"$|p_5|$ at fixed $\theta={np.degrees(phi0):.0f}^\circ$: $(x_1,x_2)$", "$x_1$", "$x_2$")
sl(ax[1, 3], C.inverse(torch.as_tensor(cs[None, None], device=dev))[0, 0, k0].cpu().numpy(),
   rf"$C^{{-1}}c$ at $\theta={np.degrees(phi0):.0f}^\circ$ (rotated back)", "$x_1$", "$x_2$")
fig.tight_layout(); fig.savefig(OUT / "check_C.png", dpi=80)
print("wrote", OUT / "check_C.png")
