"""O3c: 2-gate cascade composability probe.

Two inhibit/deflection gates in SERIES on one signal track:
  - gate 1 at region P1, control C1 (perp_cw, offset b)
  - gate 2 at region P2 = P1 + vS*GAP downstream, control C2 (offset b)
The signal that survives gate 1 glides on to P2 where C2 can deflect it. Output
read at the far window out_center = P2 + vS*TAIL. Composition predicts

    out = S AND NOT C1 AND NOT C2.

NOTE (review): an AND-NOT chain outputs 1 ONLY when no
control fires -- i.e. only when no survivor exists -- so it is never required to
emit a 1 in the presence of debris. This demonstrates I/O-level series
composition + GAP-robustness, but it STRUCTURALLY DODGES (does not solve) the
survivor-absorption problem: it shows gates chain, not that live deflected
survivors are harmless to a downstream gate that must output 1.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lenia
import collide
import orbium_collide
from gate import stamp
from orbium_phase_map import phase_advance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _output(A1, A2, out_center, vS, gm):
    """1 iff a glider-mass blob sits in the output window moving in vS's dir."""
    b1 = collide.classify(np.asarray(A1), gm)
    b2 = collide.classify(np.asarray(A2), gm)
    indeg = np.degrees(np.arctan2(vS[1], vS[0]))
    for m, y, x in b2:
        if m < 0.6 * gm or np.hypot(y - out_center[0], x - out_center[1]) > 20 or not b1:
            continue
        k = min(range(len(b1)), key=lambda i: (b1[i][1] - y) ** 2 + (b1[i][2] - x) ** 2)
        ang = np.degrees(np.arctan2((x - b1[k][2]) / 20.0, (y - b1[k][1]) / 20.0))
        if abs((ang - indeg + 180) % 360 - 180) < 35:
            return 1
    return 0


def run_cascade(seed, vS, seedC, vC, rule_kw, gm, S, C1, C2,
                b=5, D=70, GAP=110, TAIL=90, G=461, P1=(150.0, 150.0), return_field=False):
    spd = float(np.hypot(*vS)); tau1 = D / spd; tau2 = tau1 + GAP
    N = int(tau2) + TAIL
    P1 = np.array(P1); P2 = P1 + np.asarray(vS) * GAP
    nC = np.array([-vC[1], vC[0]]) / (np.hypot(*vC) + 1e-9)

    field = np.zeros((G, G))
    if S:
        field = field + stamp(G, seed, *(P1 - np.asarray(vS) * tau1))
    if C1:
        field = field + stamp(G, seedC, *(P1 - np.asarray(vC) * tau1 + b * nC))
    if C2:
        field = field + stamp(G, seedC, *(P2 - np.asarray(vC) * tau2 + b * nC))

    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    A1 = lenia.rollout(jnp.asarray(field), Kf, rule, N - 20)[0]
    A2 = lenia.rollout(A1, Kf, rule, 20)[0]
    out_center = P2 + np.asarray(vS) * TAIL
    out = _output(A1, A2, out_center, vS, gm)
    return (out, np.asarray(A2)) if return_field else out


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    seedC, vC = collide.partner(seed, vS, "perp_cw", rule_kw)
    print("2-gate cascade  out should = S AND NOT C1 AND NOT C2:")
    allok = True
    for S in (0, 1):
        for C1 in (0, 1):
            for C2 in (0, 1):
                o = run_cascade(seed, vS, seedC, vC, rule_kw, gm, S, C1, C2)
                exp = int(S and not C1 and not C2)
                ok = (o == exp); allok &= ok
                print(f"  S={S} C1={C1} C2={C2} -> out={o}  (expect {exp})  "
                      f"{'ok' if ok else 'XX MISMATCH'}")
    print(f">>> cascade {'I/O CHAIN CORRECT' if allok else 'BROKEN'} (out = S AND NOT C1 AND NOT C2)")
    print("    NOTE: an AND-NOT chain emits 1 only when no survivor exists, so this")
    print("    DODGES (does not solve) survivor-absorption (review).")

    # phase robustness: pre-evolve BOTH controls by the SAME phi -- only the
    # diagonal of the 2-control phase torus; independent phases are untested.
    print("\nphase robustness (both controls share one phi; diagonal only, coarse):")
    for phi in range(0, 24, 4):
        C0 = phase_advance(seed, rule_kw, phi)
        sC = np.ascontiguousarray(collide.GEOMS["perp_cw"](C0))
        vCp = collide.measure_velocity(sC, rule_kw)
        bad = 0
        for S in (0, 1):
            for C1 in (0, 1):
                for C2 in (0, 1):
                    o = run_cascade(seed, vS, sC, vCp, rule_kw, gm, S, C1, C2)
                    bad += (o != int(S and not C1 and not C2))
        print(f"  phi={phi:2d}: {8 - bad}/8 rows correct")

    # gap robustness: GAP is a step count; the px separation is spd*GAP,
    # = {37,49,67,85} px for GAP in (60,80,110,140), as reported in the paper
    print("\ngap robustness (truth table at each inter-gate spacing):")
    for gap in (60, 80, 110, 140):
        bad = 0
        for S in (0, 1):
            for C1 in (0, 1):
                for C2 in (0, 1):
                    o = run_cascade(seed, vS, seedC, vC, rule_kw, gm,
                                    S, C1, C2, GAP=gap)
                    bad += (o != int(S and not C1 and not C2))
        print(f"  GAP={gap:3d}: {8 - bad}/8 rows correct")

    # montage of the 4 S=1 cases (the informative rows)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for (C1, C2), ax in zip([(0, 0), (0, 1), (1, 0), (1, 1)], axs):
        o, fld = run_cascade(seed, vS, seedC, vC, rule_kw, gm, 1, C1, C2, return_field=True)
        ax.imshow(fld, cmap="viridis", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"S=1 C1={C1} C2={C2} -> {o}", fontsize=10)
    fig.suptitle("Orbium 2-gate cascade (out = S AND NOT C1 AND NOT C2)", fontsize=12)
    fig.savefig("assets/orbium_cascade.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    print("saved assets/orbium_cascade.png")


if __name__ == "__main__":
    main()
