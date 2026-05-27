"""Two-panel gradient figure for the MLSP paper.

Panel A: rho_xy(t) = leaf x/y gradient ratio trajectory for Eq.6 on
  recovered (LR yx, 100%) vs unrecovered (RL xy, 0%) target -- shows
  the transient reversal that distinguishes success from failure.

Panel B: branch-wise rho_LR = grad_left/grad_right for V16 on chain
  vs balanced targets -- shows that balanced topologies have
  persistent left-branch dominance (failure correlate), while chains
  stay near-balanced (recovery correlate).

Data:
  Panel A: re-measured by plot_gradient.py (re-run if results stale).
  Panel B: scratch/phase2_results/rho_lr_{chain,balanced}.json.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import time

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "figure.dpi": 200,
})

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
GRAY   = "#BBBBBB"

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR.parent))

# ---------------------------------------------------------------- Panel A
from master_formula import EmlMasterFormula, DTYPE, eml_torch
from scratch.sr_systematic import (
    filter_real_domain,
    TRAIN_LO, TRAIN_HI, TRAIN_STEP,
    SEARCH_ITERS, LR, GRAD_CLIP, TAU_SEARCH,
)

e_val = np.exp(1.0)
def _eml(a, b): return np.exp(a) - np.log(b)

TARGETS_A = {
    "paper": ("LR $(y,x)$, 100\\%",
              lambda x, y: _eml(_eml(1.0, _eml(y, x)), 1.0)),
    "T1":    ("RL $(x,y)$, 0\\%",
              lambda x, y: e_val - np.exp(x) + np.log(y)),
}


def make_data(target_fn):
    xs = np.arange(TRAIN_LO, TRAIN_HI + TRAIN_STEP * 0.5, TRAIN_STEP)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    xx, yy = xx.ravel(), yy.ravel()
    xx, yy, tt, n = filter_real_domain(xx, yy, target_fn)
    return (torch.tensor(xx, dtype=torch.float64),
            torch.tensor(yy, dtype=torch.float64),
            torch.tensor(tt, dtype=DTYPE), n)


def measure_panel_a(target_key, n_seeds=10,
                    checkpoints=(100, 200, 300, 500, 750, 1000, 1500, 2000)):
    _, fn = TARGETS_A[target_key]
    xt, yt, tt, _ = make_data(fn)
    cp_set = set(checkpoints)
    ratios = {cp: [] for cp in checkpoints}
    for s in range(n_seeds):
        torch.manual_seed(137 + s)
        m = EmlMasterFormula(depth=3, n_vars=2, op_fn=eml_torch)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        for it in range(1, max(checkpoints) + 1):
            opt.zero_grad()
            pred, _, _ = m(xt, yt, tau_leaf=TAU_SEARCH, tau_gate=TAU_SEARCH)
            loss = torch.mean((pred - tt).abs() ** 2).real
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if it in cp_set:
                for n, p in m.named_parameters():
                    if n == "leaf_logits" and p.grad is not None:
                        lx = p.grad.data[:, 1].norm(2).item()
                        ly = p.grad.data[:, 2].norm(2).item()
                        ratios[it].append(lx / ly if ly > 1e-15 else float('inf'))
            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
            opt.step()
    return checkpoints, ratios


# ---------------------------------------------------------------- Panel B
def load_panel_b():
    chain = json.load(open(THIS_DIR / "phase2_results/rho_lr_chain.json"))
    bal = json.load(open(THIS_DIR / "phase2_results/rho_lr_balanced.json"))
    # Aggregate per (arch, target): median + IQR over seeds*iters (whole search trajectory)
    def agg(rows):
        from collections import defaultdict
        g = defaultdict(list)
        for r in rows:
            v = r.get("rho_LR")
            if v is not None and np.isfinite(v):
                g[r["target"]].append(v)
        # Return as ordered list with median/q25/q75
        out = []
        for tgt, vals in g.items():
            v = np.array(sorted(vals))
            med = np.median(v)
            q25, q75 = np.percentile(v, [25, 75])
            out.append((tgt, med, q25, q75, len(v)))
        return out
    return agg(chain), agg(bal)


def main():
    cache_path = THIS_DIR / "panel_a_trajectory_cache.json"
    if cache_path.exists():
        print(f"Panel A: loading cached trajectories from {cache_path.name}")
        results_a = {k: tuple(v) for k, v in json.load(open(cache_path)).items()}
    else:
        print("Panel A: measuring trajectories (10 seeds x 2 targets)...")
        results_a = {}
        for tk in ["paper", "T1"]:
            t0 = time.time()
            cps, ratios = measure_panel_a(tk)
            means = [float(np.mean(ratios[cp])) for cp in cps]
            sems = [float(np.std(ratios[cp]) / np.sqrt(len(ratios[cp]))) for cp in cps]
            results_a[tk] = (list(cps), means, sems)
            print(f"  {tk}: {time.time()-t0:.0f}s")
        with open(cache_path, "w") as f:
            json.dump(results_a, f)

    print("Panel B: loading rho_LR for chain vs balanced...")
    chain, bal = load_panel_b()

    # ---- Plot ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.5, 2.0),
                                    gridspec_kw={"width_ratios": [1.0, 1.0],
                                                 "wspace": 0.55})

    # Panel A
    for tk, color, marker in [("paper", BLUE, "o"), ("T1", ORANGE, "s")]:
        cps, means, sems = results_a[tk]
        desc, _ = TARGETS_A[tk]
        axA.errorbar(cps, means, yerr=sems, color=color, marker=marker,
                     markersize=3.5, linewidth=1.0, capsize=2, label=desc)
    axA.axhline(1.0, color=GRAY, lw=0.7, ls="--", label="$x/y = 1$")
    axA.set_xlabel("Training iteration")
    axA.set_ylabel(r"Leaf gradient $\nabla_x / \nabla_y$")
    axA.set_xlim(0, 2100)
    axA.legend(loc="upper right", fontsize=5.5, framealpha=0.85,
               handletextpad=0.3, borderpad=0.3, labelspacing=0.2)
    axA.text(-0.18, 1.02, "(a)", transform=axA.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    # Panel B: bar chart of median rho_LR for each (arch,target) cell
    # Order: chains first (LR, RL, RR), then balanced
    label_map = {
        "EML_LR_yx": "EML LR",
        "EML_RL_xy": "EML RL",
        "EML_RR_yx": "EML RR",
        "SML_LR_yx": "SML LR",
        "SML_RL_xy": "SML RL",
        "EML_Bal_xy": "EML Bal",
        "SML_Bal_xy": "SML Bal",
        "SML_Bal_yx": "SML Bal y$\\leftrightarrow$x",
    }
    chain_order = ["EML_LR_yx", "EML_RL_xy", "EML_RR_yx", "SML_LR_yx", "SML_RL_xy"]
    bal_order = ["EML_Bal_xy", "SML_Bal_xy", "SML_Bal_yx"]
    chain_d = {t: (m, q25, q75) for (t, m, q25, q75, _) in chain}
    bal_d = {t: (m, q25, q75) for (t, m, q25, q75, _) in bal}

    xs = []
    meds = []
    errs_lo = []
    errs_hi = []
    colors = []
    labels = []
    for i, t in enumerate(chain_order):
        if t not in chain_d:
            continue
        m, q25, q75 = chain_d[t]
        xs.append(len(xs))
        meds.append(m)
        errs_lo.append(max(0, m - q25))
        errs_hi.append(q75 - m)
        colors.append(BLUE)
        labels.append(label_map[t])
    # Gap between chain and balanced groups
    gap_pos = len(xs)
    for i, t in enumerate(bal_order):
        if t not in bal_d:
            continue
        m, q25, q75 = bal_d[t]
        xs.append(len(xs) + 0.6)  # extra gap
        meds.append(m)
        errs_lo.append(max(0, m - q25))
        errs_hi.append(q75 - m)
        colors.append(ORANGE)
        labels.append(label_map[t])

    axB.bar(xs, meds, yerr=[errs_lo, errs_hi], color=colors,
            edgecolor="black", linewidth=0.4, capsize=2, width=0.7)
    axB.axhline(1.0, color=GRAY, lw=0.7, ls="--")
    axB.set_xticks(xs)
    axB.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    axB.set_ylabel(r"V16 $\rho_{LR}$ (med, IQR)", fontsize=6.5)
    axB.set_yscale("log")
    # Group annotations
    chain_max = gap_pos - 1 if gap_pos > 0 else 0
    axB.axvspan(-0.5, chain_max + 0.5, alpha=0.06, color=BLUE)
    axB.axvspan(chain_max + 0.6, max(xs) + 0.5, alpha=0.06, color=ORANGE)
    axB.text(chain_max / 2.0, axB.get_ylim()[1] * 0.6, "chains",
             ha="center", fontsize=7, color=BLUE, fontweight="bold")
    axB.text((chain_max + 0.6 + max(xs)) / 2.0, axB.get_ylim()[1] * 0.6,
             "balanced", ha="center", fontsize=7, color=ORANGE,
             fontweight="bold")
    axB.text(-0.18, 1.02, "(b)", transform=axB.transAxes,
             fontsize=9, fontweight="bold", va="bottom")

    outdir = Path(__file__).resolve().parents[5] / "domains" / "math" / "docs" / "odrzywolek2026"
    fig.savefig(outdir / "fig_odrzywolek2026_gradient.pdf",
                bbox_inches="tight")
    fig.savefig(outdir / "fig_odrzywolek2026_gradient.png",
                bbox_inches="tight", dpi=200)
    print(f"Saved 2-panel gradient figure to {outdir}")


if __name__ == "__main__":
    main()
