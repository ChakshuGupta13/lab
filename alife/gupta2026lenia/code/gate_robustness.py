"""Rigorous gate-window robustness: the inhibit action S=1, C=1 -> 0 swept over
ALL 24 breath phases (Delta-phi = 1) x a wide impact-parameter band, to locate
the EXACT zero-leak operating window.

gate.py samples only 9 phases (Delta-phi = 3); that undersamples the window
edges. This experiment defends or narrows the operating window against the full
breath cycle. A b value is in the window iff the gate blocks (out = 0) at every
one of the 24 phases. Emits a phase x b leak heatmap.

Run:  python gate_robustness.py
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import collide
import orbium_collide
from orbium_phase_map import phase_advance
from gate import run_gate


def main():
    seed, rule_kw = orbium_collide.orbium()
    vA = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)

    phis = list(range(0, 24))          # Delta-phi = 1: every breath phase
    band = list(range(0, 14))          # wide enough to catch both window edges
    print(f"INHIBIT robustness: Delta-phi=1, {len(phis)} phases x b={band[0]}..{band[-1]} "
          f"({len(phis) * len(band)} gate runs)")
    blk = np.zeros((len(phis), len(band)), bool)
    for i, phi in enumerate(phis):
        C0 = phase_advance(seed, rule_kw, phi)
        seedC_p = np.ascontiguousarray(collide.GEOMS["perp_cw"](C0))
        vC_p = collide.measure_velocity(seedC_p, rule_kw)
        row = ""
        for j, b in enumerate(band):
            o = run_gate(seed, vA, seedC_p, vC_p, rule_kw, gm, 1, 1, b)
            blk[i, j] = (o == 0)
            row += "." if blk[i, j] else "L"
        print(f"  phi={phi:2d} b={band[0]:2d}..{band[-1]:2d}: {row}", flush=True)

    window = [band[j] for j in range(len(band)) if blk[:, j].all()]
    correct = int(blk.sum())
    total = blk.size
    print(f">>> inhibit blocks in {correct}/{total} (phase x b) points "
          f"({100 * correct / total:.0f}%)")
    if window:
        print(f">>> zero-leak window (ALL 24 phases): b={window[0]}..{window[-1]} "
              f"({len(window)}px wide)")
    else:
        print(">>> NO b value blocks at all 24 phases")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.imshow(blk.astype(int), aspect="auto", origin="lower", cmap="cividis",
              vmin=0, vmax=1,
              extent=[band[0] - 0.5, band[-1] + 0.5, phis[0] - 0.5, phis[-1] + 0.5])
    ax.set_xlabel("impact parameter b (px)")
    ax.set_ylabel("relative breath phase")
    fig.savefig("assets/gate_robustness.png", dpi=120, bbox_inches="tight")
    print("saved assets/gate_robustness.png")


if __name__ == "__main__":
    main()
