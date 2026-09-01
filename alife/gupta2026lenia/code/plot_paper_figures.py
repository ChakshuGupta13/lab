"""Publication figures for the Lenia collision-logic paper, rendered with the
PGF backend in the repo's figure-conventions style (colorblind-safe palette,
single-column width, no on-figure titles -- captions live in the .tex).

Regenerates vector PDFs into ../../docs/lenia-compute/figures/:
  fig_lenia_substrate.pdf   one Lenia update step (A, K, U=K*A, growth G(U))
  fig_lenia_outcomes.pdf    the four collision outcomes as max-time projections
  fig_lenia_opmap.pdf       head-on operating map (outcome over phi x b)
  fig_lenia_gate.pdf        INHIBIT gate truth-table montage (4 fields)
  fig_lenia_robustness.pdf  Delta-phi=1 block/leak heatmap (24 phases x b)

Each figure recomputes its data from the simulator, so one command reproduces it.

Run:  python plot_paper_figures.py
"""
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "figure.dpi": 200,
})

COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
          "gray": "#BBBBBB", "purple": "#CC79A7"}
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "docs", "lenia-compute", "figures")

import collide
import orbium_collide
from orbium_phase_map import phase_advance, breath_period, CODE
from gate import run_gate


def fig_outcomes(seed, rule_kw, vA, gm):
    """The four collision outcomes as max-over-time projections of real
    simulator runs. Every panel is one (geometry, b) pair labelled directly by
    the outcome word used in the operating map: survive, annihilate, miss,
    merge. Anchors the outcome vocabulary before the reader meets the
    (phi, b) heatmap."""
    spd = float(np.hypot(*vA))
    G, D = 361, 70
    N = int(D / spd) + 110
    snaps = list(range(0, N, 2))
    # (label, geom, b): survive/annihilate/miss use head-on; merge is
    # perpendicular-only (chosen because it is the geometry the gate uses in
    # section 3.2). The b values are drawn from the head-on operating map
    # (fig:opmap): survive and annihilate from the mixed interior, miss from
    # the clean band (b>=21, phase-uniform miss) so the exemplar matches the
    # region where a miss is structurally guaranteed.
    cases = [("survive",     "headon",  0),
             ("annihilate",  "headon",  7),
             ("miss",        "headon",  22),
             ("merge",       "perp_cw", 0)]

    def proj(geom, b):
        seedB, vB = collide.partner(seed, vA, geom, rule_kw)
        res = collide.run(seed, vA, seedB, vB, rule_kw, b, gm,
                          D=D, N=N, G=G, snaps=snaps, track=True)
        frames = res["frames"]
        return np.max(np.stack([frames[t] for t in sorted(frames)]), axis=0)

    projs = [proj(g, b) for _, g, b in cases]
    both = np.maximum.reduce(projs)
    ys, xs = np.where(both > 0.05)
    y0, y1 = max(0, ys.min() - 10), min(G, ys.max() + 11)
    x0, x1 = max(0, xs.min() - 10), min(G, xs.max() + 11)

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    for ax, (label, _, _), pr in zip(axes.ravel(), cases, projs):
        ax.imshow(pr[y0:y1, x0:x1], cmap="viridis", vmin=0, vmax=1,
                  interpolation="bilinear")
        ax.set_title(label, fontsize=9, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(OUTDIR, "fig_lenia_outcomes.pdf"),
                bbox_inches="tight", backend="pgf")
    plt.close(fig)
    print("saved fig_lenia_outcomes.pdf")


def fig_opmap(seed, rule_kw, vA, gm):
    """Head-on operating map: collision outcome over phase phi x impact param b."""
    # local codes: the shared CODE maps merge1 and other to the same value,
    # which would mislabel unclassified cells as merges in the legend.
    CODE5 = {"annihilate": 0, "SURVIVE2": 1, "miss": 2, "merge1": 3, "other": 4}
    LABELS = [("annihilate", "blue"), ("survive", "orange"), ("miss", "gray"),
              ("merge", "green"), ("other", "purple")]
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "scratch", "opmap_phase_stats.json")
    if os.path.exists(cache):
        import json
        d = json.load(open(cache))
        phis, bs, grid = d["phis"], d["bs"], np.asarray(d["grid"], int)
    else:
        spd = float(np.hypot(*vA))
        P = breath_period(seed, rule_kw)
        G, D = 361, 70
        N = int(D / spd) + 110
        phis = list(range(0, int(round(P)), max(1, int(round(P / 8)))))  # phi=P duplicates phi=0
        bs = list(range(0, 25))
        grid = np.zeros((len(phis), len(bs)), int)
        for i, phi in enumerate(phis):
            B0 = phase_advance(seed, rule_kw, phi)
            seedB = np.ascontiguousarray(collide.GEOMS["headon"](B0))
            vB = collide.measure_velocity(seedB, rule_kw)
            for j, b in enumerate(bs):
                res = collide.run(seed, vA, seedB, vB, rule_kw, b, gm, D=D, N=N, G=G, track=True)
                grid[i, j] = CODE5[collide.label_outcome(res, gm)]
            print(f"  opmap phi={phi}", flush=True)
    # 0 annihilate, 1 survive2, 2 miss, 3 merge, 4 other
    cmap = ListedColormap([COLORS[c] for _, c in LABELS])
    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=4,
              extent=[bs[0] - 0.5, bs[-1] + 0.5, -0.5, len(phis) - 0.5],
              interpolation="nearest")
    ax.set_xlabel(r"impact parameter $b$ (px)")
    ax.set_ylabel(r"breath phase $\phi$")
    ax.set_yticks(range(len(phis)))
    ax.set_yticklabels([str(p) for p in phis])
    present = sorted(set(grid.ravel().tolist()))
    legend = [Patch(facecolor=COLORS[LABELS[k][1]], label=LABELS[k][0])
              for k in present]
    ax.legend(handles=legend, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), frameon=False, columnspacing=1.0,
              handlelength=1.0, handletextpad=0.5)
    fig.savefig(os.path.join(OUTDIR, "fig_lenia_opmap.pdf"), bbox_inches="tight", backend="pgf")
    plt.close(fig)
    print("saved fig_lenia_opmap.pdf")


def fig_gate(seed, rule_kw, vA, gm):
    """Two-panel max-over-time projection of the field: the signal passes to the
    output window when the control is absent, and is deflected off the line when
    the control is present. The streak is the glider's whole path."""
    import lenia
    import jax.numpy as jnp
    from gate import stamp

    seedC, vC = collide.partner(seed, vA, "perp_cw", rule_kw)
    G, D, b_ctrl = 361, 70, 5
    spd = float(np.hypot(*vA)); tau = D / spd
    N = int(tau) + 90
    P = np.array([G / 2.0, G / 2.0])
    nC = np.array([-vC[1], vC[0]]) / (np.hypot(*vC) + 1e-9)
    out_center = P + np.asarray(vA) * (N - tau)
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)

    def maxproj(C_on):
        field = stamp(G, seed, *(P - np.asarray(vA) * tau))
        if C_on:
            pC = P - np.asarray(vC) * tau + b_ctrl * nC
            field = field + stamp(G, seedC, *pC)
        _, traj = lenia.rollout(jnp.asarray(field), Kf, rule, N)
        return np.max(np.asarray(traj), axis=0)

    m0 = maxproj(False)
    m1 = maxproj(True)
    both = np.maximum(m0, m1)
    ys, xs = np.where(both > 0.05)
    y0, y1 = max(0, ys.min() - 8), min(G, ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(G, xs.max() + 9)

    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.95))
    panels = [(axs[0], m0, r"$C$ absent: $\mathrm{out}=1$"),
              (axs[1], m1, r"$C$ present: $\mathrm{out}=0$")]
    for ax, m, lab in panels:
        ax.imshow(m[y0:y1, x0:x1], cmap="viridis", vmin=0, vmax=1,
                  interpolation="nearest")
        oy, ox = out_center[0] - y0, out_center[1] - x0
        # radius-20 circle = the detector's actual acceptance region (collide.run)
        ax.add_patch(plt.Circle((ox, oy), 20, fill=False,
                                edgecolor="#E69F00", lw=1.1, ls="--"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, fontsize=8, pad=2)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(OUTDIR, "fig_lenia_gate.pdf"), bbox_inches="tight", backend="pgf")
    plt.close(fig)
    print("saved fig_lenia_gate.pdf (output window dashed)")


def fig_robustness(seed, rule_kw, vA, gm):
    """Block/leak over all 24 breath phases x impact parameter b (Delta-phi=1)."""
    phis = list(range(0, 24))
    band = list(range(0, 14))
    blk = np.zeros((len(phis), len(band)), bool)
    for i, phi in enumerate(phis):
        C0 = phase_advance(seed, rule_kw, phi)
        seedC_p = np.ascontiguousarray(collide.GEOMS["perp_cw"](C0))
        vC_p = collide.measure_velocity(seedC_p, rule_kw)
        for j, b in enumerate(band):
            o = run_gate(seed, vA, seedC_p, vC_p, rule_kw, gm, 1, 1, b)
            blk[i, j] = (o == 0)
        print(f"  robust phi={phi}", flush=True)
    # 0 leak (orange), 1 block (blue)
    cmap = ListedColormap([COLORS["orange"], COLORS["blue"]])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.imshow(blk.astype(int), aspect="auto", origin="lower", cmap=cmap,
              vmin=0, vmax=1, interpolation="nearest",
              extent=[band[0] - 0.5, band[-1] + 0.5, phis[0] - 0.5, phis[-1] + 0.5])
    ax.axvline(8.5, color="k", lw=1.0, ls="--")
    ax.set_xlabel(r"impact parameter $b$ (px)")
    ax.set_ylabel(r"breath phase $\phi$")
    legend = [Patch(facecolor=COLORS["blue"], label=r"blocks ($\mathrm{out}=0$)"),
              Patch(facecolor=COLORS["orange"], label=r"leaks ($\mathrm{out}=1$)")]
    ax.legend(handles=legend, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.20), frameon=False, columnspacing=1.0,
              handlelength=1.0, handletextpad=0.5)
    fig.savefig(os.path.join(OUTDIR, "fig_lenia_robustness.pdf"), bbox_inches="tight", backend="pgf")
    plt.close(fig)
    print("saved fig_lenia_robustness.pdf")


def fig_substrate(seed, rule_kw, vA, gm):
    """One update step as a left-to-right dataflow: the glider field A, the ring
    kernel K, the potential U=K*A (ring-average), and the growth G(U). Grounds the
    rule (Eq. update) and the ring shape of K."""
    import lenia
    import jax.numpy as jnp
    G2, CROP = 161, 22
    rule = lenia.Rule(grid=G2, **rule_kw)
    Kf = lenia.kernel_fft(rule)
    s0, s1 = seed.shape
    oy, ox = (G2 - s0) // 2, (G2 - s1) // 2
    A = jnp.zeros((G2, G2)).at[oy:oy + s0, ox:ox + s1].set(jnp.asarray(seed))
    for _ in range(24):
        A = lenia.step(A, Kf, rule)
    c0 = np.asarray(lenia.center_of_mass(A))
    Astep = A
    for _ in range(10):
        Astep = lenia.step(Astep, Kf, rule)
    v = np.asarray(lenia.center_of_mass(Astep)) - c0
    vh = v / (np.hypot(*v) + 1e-9)
    U = np.asarray(lenia.potential(A, Kf))
    Gm = np.asarray(lenia._growth(lenia.potential(A, Kf), rule))
    A = np.asarray(A)
    h = G2 // 2
    sl = (slice(h - CROP, h + CROP + 1), slice(h - CROP, h + CROP + 1))
    Ac, Uc, Gc = A[sl], U[sl], Gm[sl]
    K = np.asarray(lenia.kernel(rule))
    kr = int(rule.kr) + 4
    Kk = K[h - kr:h + kr + 1, h - kr:h + kr + 1]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05))
    ctr = CROP
    axes[0].imshow(Ac, cmap="viridis", vmin=0, vmax=1, origin="lower", interpolation="bilinear")
    axes[0].annotate("", xy=(ctr + 12 * vh[1], ctr + 12 * vh[0]),
                     xytext=(ctr - 12 * vh[1], ctr - 12 * vh[0]),
                     arrowprops=dict(arrowstyle="-|>", color="white", lw=1.3))
    axes[0].set_title(r"field $A$", fontsize=9)
    axes[1].imshow(Kk / Kk.max(), cmap="viridis", origin="lower", interpolation="bilinear")
    axes[1].set_title(r"kernel $K$ (a ring)", fontsize=9)
    axes[2].imshow(Uc, cmap="viridis", origin="lower", interpolation="bilinear")
    axes[2].set_title(r"potential $U=K*A$", fontsize=9)
    axes[3].imshow(Gc, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", interpolation="bilinear")
    axes[3].set_title(r"growth $G(U)$", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(os.path.join(OUTDIR, "fig_lenia_substrate.pdf"), bbox_inches="tight", backend="pgf")
    plt.close(fig)
    print("saved fig_lenia_substrate.pdf")


def main():
    import sys
    which = set(sys.argv[1:]) or {"substrate", "outcomes", "gate", "opmap", "robustness"}
    os.makedirs(OUTDIR, exist_ok=True)
    seed, rule_kw = orbium_collide.orbium()
    vA = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    if "substrate" in which:
        print("rendering substrate figure ..."); fig_substrate(seed, rule_kw, vA, gm)
    if "outcomes" in which:
        print("rendering outcome taxonomy ..."); fig_outcomes(seed, rule_kw, vA, gm)
    if "gate" in which:
        print("rendering gate projection ..."); fig_gate(seed, rule_kw, vA, gm)
    if "opmap" in which:
        print("rendering operating map ..."); fig_opmap(seed, rule_kw, vA, gm)
    if "robustness" in which:
        print("rendering robustness heatmap ..."); fig_robustness(seed, rule_kw, vA, gm)


if __name__ == "__main__":
    main()
