"""Generate recovery-rate heatmap for EML, SML, and RML operators.

Data source: README.md d3 matrix, SML chain matrix, RML chain matrix.
Output: domains/math/docs/odrzywolek2026/fig_odrzywolek2026_heatmap.pdf
"""

import os
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np

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

# Colorblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"

# =====================================================================
# Data from README.md — recovery rates (%)
# =====================================================================
# Row order: LR(yx), LR(xy), RL(xy), RL(yx), RR(xy), RR(yx), Bal
# Column order: Eq.6, V16, Hybrid
# NaN = not tested / not applicable

# EML (d3 matrix + balanced)
# paper=LR(yx), T8=LR(xy), T1=RL(xy), T1_yx=RL(yx),
# T4=RR(xy), T4_yx=RR(yx), Bal=0%
eml = np.array([
    [100.0,  17.0,   0.4],   # LR (yx) — paper
    [  1.6,  16.0,   0.4],   # LR (xy) — T8
    [  0.0,  99.6,  96.0],   # RL (xy) — T1
    [  0.0,  99.6,  95.3],   # RL (yx) — T1_yx (V16 = same by symmetry)
    [  0.0,  39.5,   0.0],   # RR (xy) — T4
    [100.0,  39.5,   0.0],   # RR (yx) — T4_yx (V16 = same by symmetry)
    [  0.0,   0.0,   0.0],   # Balanced
])

# SML chain matrix + balanced
sml = np.array([
    [  0.0,  84.0,   0.8],   # LR (yx) — S_LR
    [  0.0,  83.0,   0.8],   # LR (xy) — S_LR_xy
    [100.0, 100.0, 100.0],   # RL (xy) — S_RL
    [  0.0, 100.0, 100.0],   # RL (yx) — S_RL_yx
    [100.0,  99.6,  89.8],   # RR (xy) — S_RR
    [100.0,  98.8,  88.7],   # RR (yx) — S_RR_yx
    [  0.0,   0.0, np.nan],  # Balanced (Hybrid not tested for EML bal)
])

# RML chain matrix (only 4 targets tested, no balanced)
rml = np.array([
    [  0.0,   2.0,   1.2],   # LR (yx) — R_LR
    [  0.0,   0.8,   2.3],   # LR (xy) — R_LR_xy
    [  0.0,   0.0,   0.0],   # RL (xy) — R_RL
    [  0.0,   0.0,   0.0],   # RL (yx) — R_RL_yx
    [np.nan, np.nan, np.nan], # RR (xy) — not tested
    [np.nan, np.nan, np.nan], # RR (yx) — not tested
    [np.nan, np.nan, np.nan], # Balanced — not tested
])

row_labels = [
    "LR $(y,x)$",
    "LR $(x,y)$",
    "RL $(x,y)$",
    "RL $(y,x)$",
    "RR $(x,y)$",
    "RR $(y,x)$",
    "Balanced",
]
col_labels = ["Eq.6", "V16", "Hybrid"]
panel_labels = ["(a) EML", "(b) SML", "(c) RML"]
datasets = [eml, sml, rml]


fig, axes = plt.subplots(1, 3, figsize=(3.5, 2.0), sharey=True,
                         gridspec_kw={"wspace": 0.06})

for ax, data, plabel in zip(axes, datasets, panel_labels):
    # Mask NaN for display
    masked = np.ma.array(data, mask=np.isnan(data))

    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100,
                   aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=5, color="#666666")
            else:
                val = data[i, j]
                color = "white" if val < 20 or val > 80 else "black"
                txt = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=5, color=color, fontweight="bold")

    # NaN cells: gray hatching
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=True, facecolor="#DDDDDD",
                             edgecolor="#AAAAAA", linewidth=0.5))

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6)
    ax.set_xlabel(plabel, fontsize=7, fontweight="bold")

    # Horizontal separator before Balanced row
    ax.axhline(y=5.5, color="black", linewidth=0.6)

axes[0].set_yticks(range(len(row_labels)))
axes[0].set_yticklabels(row_labels, fontsize=6)

# Colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Rec. (\\%)", fontsize=7)
cbar.ax.tick_params(labelsize=6)

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "docs", "odrzywolek2026")
fig.savefig(os.path.join(outdir, "fig_odrzywolek2026_heatmap.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(outdir, "fig_odrzywolek2026_heatmap.png"),
            bbox_inches="tight", dpi=200)
print("Saved heatmap to docs/odrzywolek2026/fig_odrzywolek2026_heatmap.{pdf,png}")
