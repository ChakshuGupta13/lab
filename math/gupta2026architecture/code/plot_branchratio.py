"""Fig 4: V16 branch-wise rho_LR for chain vs balanced targets.

Single-column figure (formerly Panel B of plot_gradient_2panel.py).
Reads scratch/phase2_results/rho_lr_{chain,balanced}.json.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "figure.dpi": 200,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#BBBBBB"

THIS_DIR = Path(__file__).parent

def agg(rows):
    g = defaultdict(list)
    for r in rows:
        v = r.get("rho_LR")
        if v is not None and np.isfinite(v):
            g[r["target"]].append(v)
    out = []
    for tgt, vals in g.items():
        v = np.array(vals)
        med = np.median(v)
        q25, q75 = np.percentile(v, [25, 75])
        out.append((tgt, med, q25, q75))
    return out

chain = agg(json.load(open(THIS_DIR / "phase2_results/rho_lr_chain.json")))
bal = agg(json.load(open(THIS_DIR / "phase2_results/rho_lr_balanced.json")))

LABEL = {
    "EML_LR_yx": "EML LR",
    "EML_RL_xy": "EML RL",
    "EML_RR_yx": "EML RR",
    "SML_LR_yx": "SML LR",
    "SML_RL_xy": "SML RL",
    "EML_Bal_xy": "EML Bal",
    "SML_Bal_xy": "SML Bal",
    "SML_Bal_yx": r"SML Bal $y\!\leftrightarrow\!x$",
}
chain_order = ["EML_LR_yx", "EML_RL_xy", "EML_RR_yx", "SML_LR_yx", "SML_RL_xy"]
bal_order = ["EML_Bal_xy", "SML_Bal_xy", "SML_Bal_yx"]

chain_d = {t: (m, q25, q75) for (t, m, q25, q75) in chain}
bal_d = {t: (m, q25, q75) for (t, m, q25, q75) in bal}

xs, meds, errs_lo, errs_hi, colors, labels = [], [], [], [], [], []
for t in chain_order:
    if t not in chain_d:
        continue
    m, q25, q75 = chain_d[t]
    xs.append(len(xs))
    meds.append(m); errs_lo.append(max(0, m - q25)); errs_hi.append(q75 - m)
    colors.append(BLUE); labels.append(LABEL[t])
gap_pos = len(xs)
for t in bal_order:
    if t not in bal_d:
        continue
    m, q25, q75 = bal_d[t]
    xs.append(len(xs) + 0.6)
    meds.append(m); errs_lo.append(max(0, m - q25)); errs_hi.append(q75 - m)
    colors.append(ORANGE); labels.append(LABEL[t])

fig, ax = plt.subplots(figsize=(3.4, 2.0), constrained_layout=True)
ax.bar(xs, meds, yerr=[errs_lo, errs_hi], color=colors,
       edgecolor="black", linewidth=0.4, capsize=2, width=0.7)
ax.axhline(1.0, color=GRAY, lw=0.7, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel(r"V16 $\rho_{LR}$ (median, IQR)")
ax.set_yscale("log")
# Group annotations: place ABOVE plot using transAxes coords so they don't
# fight with axis tick labels at the top of the plotting area.
chain_max_x = gap_pos - 1 if gap_pos > 0 else 0
ax.axvspan(-0.5, chain_max_x + 0.5, alpha=0.06, color=BLUE)
ax.axvspan(chain_max_x + 0.6, max(xs) + 0.5, alpha=0.06, color=ORANGE)
# Convert to display coords
n_total = max(xs) + 0.5 - (-0.5)
chain_center = ((-0.5) + (chain_max_x + 0.5)) / 2
bal_center = ((chain_max_x + 0.6) + (max(xs) + 0.5)) / 2
chain_frac = (chain_center - (-0.5)) / n_total
bal_frac = (bal_center - (-0.5)) / n_total
ax.text(chain_frac, 1.02, "chains", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7, color=BLUE, fontweight="bold")
ax.text(bal_frac, 1.02, "balanced", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7, color=ORANGE, fontweight="bold")

outdir = Path(__file__).resolve().parents[5] / "domains" / "math" / "docs" / "odrzywolek2026"
fig.savefig(outdir / "fig_odrzywolek2026_branchratio.pdf", bbox_inches="tight")
fig.savefig(outdir / "fig_odrzywolek2026_branchratio.png", bbox_inches="tight", dpi=200)
print(f"Saved to {outdir / 'fig_odrzywolek2026_branchratio.pdf'}")
