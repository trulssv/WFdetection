"""Training cost and memory scaling of the lifted LPD (for the Beamer talk).

1. Mines outputs/train/*/{config.json,log.jsonl} for logged step times.
2. Benchmarks one training step exactly as in experiments/train_lpd.py
   (data.sample -> bf16-autocast forward -> D + lambda E -> backward ->
   grad clipping -> AdamW + cosine step) for n in {64, ..., 384}, K = n/4,
   batch 2, plus a batch sweep at n = 96 (K = 32, as in the n = 96 runs).
   Each configuration runs in its own subprocess, so an OOM is isolated.
3. Fits power laws vs. the number of lifted voxels N = K n^2.
4. Writes assets/fig_scaling.png and assets/scaling.json.

Run from the repository root on one GPU:

    CUDA_VISIBLE_DEVICES=1 python -m research.presentation.bench_scaling
    python -m research.presentation.bench_scaling --plot-only   # re-plot from json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
REF_CFG = ROOT / "outputs/train/n256_lam1e-4/config.json"
N_WARM, N_TIMED = 3, 8


# --------------------------------------------------------------------- logs
def mine_logs():
    rows = []
    for d in sorted(glob.glob(str(ROOT / "outputs/train/*/"))):
        d = Path(d)
        if not (d / "config.json").exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        recs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if l.strip()]
        tr = [r for r in recs if "time" in r]
        if not tr:
            continue
        timed = [r for r in tr if "sec_per_step" in r]
        last = tr[-1]
        rows.append({
            "run": d.name, "n": cfg["n"], "K": cfg["n_theta"], "batch": cfg["batch"],
            "steps_cfg": cfg["steps"], "steps_done": last["step"],
            "sec_per_step": float(np.mean([r["sec_per_step"] for r in timed])) if timed else None,
            "sec_data": float(np.mean([r["sec_data"] for r in timed])) if timed else None,
            # wall clock incl. validation / checkpointing
            "wall_sec_per_step": last["time"] / last["step"],
            "wall_hours": last["time"] / 3600,
        })
    return rows


# ---------------------------------------------------------------- benchmark
def bench_one(n, K, batch):
    """One configuration; runs inside a subprocess.  Returns a dict."""
    import torch
    sys.path.insert(0, str(ROOT))
    from experiments.train_lpd import LEGACY_DEFAULTS, build, objective, run_model
    from data.synthetic import normalize_coefficients, smooth_background
    from data.phantoms import random_phantom
    from loss.data_discrepancy import ForwardModelDiscrepancy

    dev = "cuda"
    cfg = {**LEGACY_DEFAULTS, **json.loads(REF_CFG.read_text())}
    cfg.update(n=n, n_theta=K, batch=batch)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    amp = not cfg.get("no_amp", False)
    grid, data, model, fwd, prior = build(cfg, dev)
    disc = ForwardModelDiscrepancy(delta=1e-2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
    n_params = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(0)
    sync = torch.cuda.synchronize

    def data_breakdown():
        """data.sample() split into its stages (same code, with timers)."""
        t = {}
        t0 = time.perf_counter()
        phantoms = [random_phantom(rng) for _ in range(batch)]
        f = np.stack([p.render(n) + (smooth_background(rng, n, amplitude=data.background)
                                     if data.background else 0) for p in phantoms])
        f = torch.as_tensor(f, dtype=torch.float32)[:, None]
        t1 = time.perf_counter(); t["phantom_cpu"] = t1 - t0
        y = data.R(f).to(dev)
        sig = data.noise * y.abs().flatten(1).max(1).values.view(-1, 1, 1, 1)
        y = y + sig * torch.randn(y.shape, device=dev)
        sync(); t2 = time.perf_counter(); t["radon_noise"] = t2 - t1
        with torch.no_grad():
            c = normalize_coefficients(data.M(y))
            sync(); t3 = time.perf_counter(); t["sino_analyzer_M"] = t3 - t2
            fb = data.R.fbp(y)
            sync(); t4 = time.perf_counter(); t["fbp"] = t4 - t3
            b = normalize_coefficients(data.A(fb))
            sync(); t5 = time.perf_counter(); t["image_analyzer_A"] = t5 - t4
        return t

    def train_step():
        sync(); t0 = time.perf_counter()
        batch_ = data.sample(rng, batch)
        sync(); t1 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        out = run_model(model, batch_, amp)
        loss, terms = objective(out, batch_, fwd, disc, prior, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        float(loss)
        sync(); t2 = time.perf_counter()
        return t1 - t0, t2 - t1, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    res = {"n": n, "K": K, "batch": batch, "N": K * n * n, "n_params": n_params,
           "n_iter": cfg["n_iter"], "hidden": cfg["hidden"], "amp_bf16": amp,
           "gpu": torch.cuda.get_device_name(0)}
    try:
        for _ in range(N_WARM):
            train_step()
        td, tg, mem, memr = [], [], [], []
        for _ in range(N_TIMED):
            a, b_, m, r = train_step()
            td.append(a), tg.append(b_), mem.append(m), memr.append(r)
        res.update(sec_data=float(np.median(td)), sec_gpu=float(np.median(tg)),
                   sec_gpu_std=float(np.std(tg)), sec_step=float(np.median(np.add(td, tg))),
                   train_peak_GB=max(mem) / 1e9, train_reserved_GB=max(memr) / 1e9)
        # data generation by stage
        parts = [data_breakdown() for _ in range(5)]
        res["data_parts"] = {k: float(np.median([p[k] for p in parts])) for k in parts[0]}
        # data generation peak (M and A on the lifted grid)
        torch.cuda.reset_peak_memory_stats()
        batch_ = data.sample(rng, batch)
        res["data_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
        # inference: eval mode (no checkpointing), no_grad
        model.eval()
        with torch.no_grad():
            run_model(model, batch_, amp)
            sync()
            torch.cuda.reset_peak_memory_stats()
            ti = []
            for _ in range(5):
                sync(); t0 = time.perf_counter()
                run_model(model, batch_, amp)
                sync(); ti.append(time.perf_counter() - t0)
        res.update(infer_peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                   sec_infer=float(np.median(ti)), status="ok")
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="OOM", error=str(e).splitlines()[0])
    return res


def run_all():
    configs = [(n, n // 4, 2) for n in (64, 96, 128, 192, 256, 320, 384)]
    configs += [(96, 32, b) for b in (2, 4, 8)]
    results = []
    for n, K, b in configs:
        print(f"n={n} K={K} batch={b} ...", flush=True)
        p = subprocess.run([sys.executable, "-m", "research.presentation.bench_scaling",
                            "--single", str(n), str(K), str(b)],
                           cwd=ROOT, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            r = json.loads(line[-1][7:])
        else:
            tail = (p.stderr or "").strip().splitlines()[-3:]
            r = {"n": n, "K": K, "batch": b, "N": K * n * n, "status": "failed",
                 "error": " | ".join(tail)}
            if any("out of memory" in t.lower() for t in tail):
                r["status"] = "OOM"
        print("   ", {k: v for k, v in r.items() if k != "data_parts"}, flush=True)
        results.append(r)
    return results


# ---------------------------------------------------------------- fit / plot
def fit(N, y):
    s, c = np.polyfit(np.log(N), np.log(y), 1)
    return float(s), float(np.exp(c))


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    sw = [r for r in out["bench"] if r["batch"] == 2 and r["K"] == r["n"] // 4 and r["status"] == "ok"]
    sw.sort(key=lambda r: r["N"])
    oom = [r for r in out["bench"] if r["batch"] == 2 and r["K"] == r["n"] // 4 and r["status"] != "ok"]
    oom.sort(key=lambda r: r["N"])
    N = np.array([r["N"] for r in sw], float)
    fits = out["fits"]
    blue, orange, green, grey = "#2a6fdb", "#e07b00", "#2f9e44", "#666666"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    Nf = np.geomspace(N.min() / 1.2, N.max() * 1.2, 50)

    ax = axs[0]
    tg = np.array([r["sec_gpu"] for r in sw]); tt = np.array([r["sec_step"] for r in sw])
    ax.plot(N, tt, "o-", color=blue, label="full step (data + network)")
    ax.plot(N, tg, "s--", color=green, ms=5, label="network only (fwd + bwd + Adam)")
    s, c = fits["sec_step"]["slope"], fits["sec_step"]["coef"]
    ax.plot(Nf, c * Nf**s, ":", color=grey, lw=1.2)
    ax.text(0.04, 0.70, f"$t \\propto N^{{{s:.2f}}}$", transform=ax.transAxes, color=grey, fontsize=12)
    ax.set_title("Training time per step vs. lifted grid size")
    ax.set_ylabel("seconds / step (batch 2)")

    ax = axs[1]
    mt = np.array([r["train_peak_GB"] for r in sw]); mi = np.array([r["infer_peak_GB"] for r in sw])
    ax.plot(N, mt, "o-", color=blue, label="training (checkpointed)")
    ax.plot(N, mi, "^-", color=orange, label="inference (no grad)")
    for key, col in (("train_peak_GB", blue), ("infer_peak_GB", orange)):
        s, c = fits[key]["slope"], fits[key]["coef"]
        ax.plot(Nf, c * Nf**s, ":", color=grey, lw=1.2)
    ax.text(0.04, 0.70, f"train $\\propto N^{{{fits['train_peak_GB']['slope']:.2f}}}$\n"
            f"infer $\\propto N^{{{fits['infer_peak_GB']['slope']:.2f}}}$",
            transform=ax.transAxes, color=grey, fontsize=12, va="center")
    ax.axhline(48, color="#b00020", lw=1, ls="--")
    oom_txt = "".join(f";  n={r['n']}, K={r['K']}: OOM" for r in oom)
    ax.text(N.min(), 48 * 0.62, "48 GB card limit" + oom_txt, color="#b00020", fontsize=11)
    ax.set_title("Peak GPU memory (batch 2)")
    ax.set_ylabel("GB (PyTorch allocator)")

    for ax, key in zip(axs, ("sec_step", "train_peak_GB")):
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("lifted voxels $N = K n^2$")
        prod = [r for r in sw if r["n"] == 256]
        if prod:
            v = prod[0][key]
            ax.plot(prod[0]["N"], v, "o", ms=15, mfc="none", mec="#b00020", mew=2,
                    label="production run (n=256, K=64)")
            ax.annotate(f"{v:.1f} s" if key == "sec_step" else f"{v:.1f} GB", (prod[0]["N"], v),
                        xytext=(-58, 8), textcoords="offset points", color="#b00020", fontsize=11)
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ax.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3, 10, 30]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xticks(N)
        ax.set_xticklabels([f"n={r['n']}\nK={r['K']}" for r in sw], fontsize=11)
        ax.minorticks_off()
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(loc="lower right", frameon=False)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(N)
        sec.set_xticklabels([f"{x / 1e6:.2g}M" for x in N], fontsize=11)
        sec.minorticks_off()
        sec.set_xlabel("N (lifted voxels per sample)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_scaling.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", nargs=3, type=int, metavar=("N", "K", "BATCH"))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.single:
        print("RESULT " + json.dumps(bench_one(*args.single)), flush=True)
        return
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / "scaling.json"
    if args.plot_only:
        plot(json.loads(path.read_text()))
        return
    out = {"logs": mine_logs(), "bench": run_all(),
           "protocol": f"{N_WARM} warmup + {N_TIMED} timed steps (median), config of "
                       f"{REF_CFG.relative_to(ROOT)} with n, K, batch overridden; "
                       "peak memory = torch.cuda.max_memory_allocated (excludes ASTRA buffers)"}
    sw = [r for r in out["bench"] if r["batch"] == 2 and r["K"] == r["n"] // 4 and r["status"] == "ok"]
    N = np.array([r["N"] for r in sw], float)
    out["fits"] = {}
    for key in ("sec_step", "sec_gpu", "sec_data", "train_peak_GB", "infer_peak_GB", "sec_infer"):
        s, c = fit(N, np.array([r[key] for r in sw]))
        out["fits"][key] = {"slope": s, "coef": c}
    # fit on the larger grids only (small grids are overhead-dominated)
    big = [r for r in sw if r["n"] >= 128]
    if len(big) >= 2:
        Nb = np.array([r["N"] for r in big], float)
        for key in ("sec_gpu", "train_peak_GB"):
            out["fits"][key + "_n>=128"] = {"slope": fit(Nb, np.array([r[key] for r in big]))[0]}
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["fits"], indent=2))
    plot(out)


if __name__ == "__main__":
    main()
