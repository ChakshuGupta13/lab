"""Generate the validation-selector figure (2 panels).

Panel (a): Pooled recovery on jointly-run synthetic subset.
           V16 baseline vs validation selector.
Panel (b): Shockley diode per-architecture + selector.

Output: domains/math/docs/odrzywolek2026/fig_odrzywolek2026_selector.pdf
"""

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "figure.dpi": 200,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
TEAL = "#4477AA"
ROSE = "#CC6677"
GRAY = "#BBBBBB"

# --- Data ---
# Panel (a): Jointly-run subset (3 targets where V16 + another arch present)
# V16 is the only arch on all 3 targets.
panel_a_labels = ["V16\nbaseline", "Validation\nselector"]
panel_a_rates = [264 / 768, 385 / 768]  # 34.4%, 50.1%
panel_a_colors = [GRAY, BLUE]

# Panel (b): Shockley diode (32 seeds each)
panel_b_labels = ["V16", "Eq.\\,6", "Hybrid", "Selector"]
panel_b_rates = [0 / 32, 18 / 32, 22 / 32, 28 / 32]  # 0%, 56.2%, 68.8%, 87.5%
panel_b_colors = [GRAY, ORANGE, TEAL, BLUE]

# --- Figure ---
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.5, 2.0),
                                  gridspec_kw={"width_ratios": [1, 1.6]})

# Panel (a)
bars_a = ax_a.bar(range(len(panel_a_labels)), [r * 100 for r in panel_a_rates],
                  color=panel_a_colors, edgecolor="black", linewidth=0.5, width=0.6)
ax_a.set_xticks(range(len(panel_a_labels)))
ax_a.set_xticklabels(panel_a_labels, fontsize=8)
ax_a.set_ylabel("Recovery rate (\\%)")
ax_a.set_ylim(0, 100)
ax_a.set_title("(a) Jointly-run subset", fontsize=9, pad=4)
# Annotate bars
for bar, rate in zip(bars_a, panel_a_rates):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
              f"{rate*100:.1f}\\%", ha="center", va="bottom", fontsize=7)

# Panel (b)
bars_b = ax_b.bar(range(len(panel_b_labels)), [r * 100 for r in panel_b_rates],
                  color=panel_b_colors, edgecolor="black", linewidth=0.5, width=0.6)
ax_b.set_xticks(range(len(panel_b_labels)))
ax_b.set_xticklabels(panel_b_labels, fontsize=8)
ax_b.set_ylim(0, 100)
ax_b.set_title("(b) Shockley diode", fontsize=9, pad=4)
# Annotate bars
for bar, rate in zip(bars_b, panel_b_rates):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
              f"{rate*100:.1f}\\%", ha="center", va="bottom", fontsize=7)

plt.tight_layout(pad=0.4)

out_dir = Path(__file__).resolve().parents[3] / "docs" / "odrzywolek2026"
out_path = out_dir / "fig_odrzywolek2026_selector.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
