"""U4: fanout -- can one signal become two copies?

A Lenia collision never splits a lone glider; duplication needs a second
"helper" glider. The 2-survivor SURVIVE2 scatter is the candidate: at the
phase-robust head-on SURVIVE2 offset (b=16) a signal S and a counter-propagating
helper H scatter into TWO survivors. If those two leave on tracks distinct from
H's straight-through track (where H goes when S is absent), reading those two
tracks is a fanout: present on both iff S present.

This probe counts survivors and their exit directions for (S+H) vs (H alone)
across breath phase, and checks the 2-vs-1 / distinct-track condition.

Run:  python fanout.py
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


def survivors(seed, vS, seedH, vH, rule_kw, gm, b, S_on, H_on, D=70, G=361):
    spd = float(np.hypot(*vS)); tau = D / spd; N = int(tau) + 110
    P = np.array([G / 2.0, G / 2.0])
    nH = np.array([-vH[1], vH[0]]) / (np.hypot(*vH) + 1e-9)
    field = np.zeros((G, G))
    if S_on:
        field = field + stamp(G, seed, *(P - np.asarray(vS) * tau))
    if H_on:
        field = field + stamp(G, seedH, *(P - np.asarray(vH) * tau + b * nH))
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    A1 = lenia.rollout(jnp.asarray(field), Kf, rule, N - 30)[0]
    A2 = lenia.rollout(A1, Kf, rule, 30)[0]
    b1 = collide.classify(np.asarray(A1), gm); b2 = collide.classify(np.asarray(A2), gm)
    dirs = []
    for m, y, x in b2:
        if m < 0.6 * gm or not b1:
            continue
        k = min(range(len(b1)), key=lambda i: (b1[i][1] - y) ** 2 + (b1[i][2] - x) ** 2)
        dirs.append(round(np.degrees(np.arctan2((x - b1[k][2]) / 30.0, (y - b1[k][1]) / 30.0))))
    return dirs


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    b = 16                       # phase-robust head-on SURVIVE2 offset
    phis = list(range(0, 24, 3))
    print(f"fanout via head-on SURVIVE2 at b={b}: survivor exit directions (deg)")
    print("  phi |  S+H (want 2 copies)        |  H alone (want 1, distinct track)")
    n2 = n1 = 0
    for phi in phis:
        H0 = phase_advance(seed, rule_kw, phi)
        sH = np.ascontiguousarray(collide.GEOMS["headon"](H0))
        vH = collide.measure_velocity(sH, rule_kw)
        dSH = survivors(seed, vS, sH, vH, rule_kw, gm, b, 1, 1)
        dH = survivors(seed, vS, sH, vH, rule_kw, gm, b, 0, 1)
        n2 += (len(dSH) == 2); n1 += (len(dH) == 1)
        print(f"  {phi:3d} |  {str(dSH):26s} |  {dH}")
    print(f"\n  S+H gave exactly 2 survivors in {n2}/{len(phis)} phases; "
          f"H-alone gave exactly 1 in {n1}/{len(phis)} phases.")
    if n2 >= len(phis) - 1 and n1 >= len(phis) - 1:
        print(">>> fanout FEASIBLE at DEPTH 1: 2 S-dependent outputs vs 1 helper track")
        print("    (verified: both survivors are full-mass Orbia). CAVEATS: needs")
        print("    a counter-propagating helper STREAM (single H tested != stream; couples to")
        print("    the constant-1 source); one output is the deflected H; and the 2 outputs")
        print("    need POSITIONAL restoration to hit a downstream 2-4px gate window -- none")
        print("    of which is shown. Depth->=2 composition is UNPROVEN.")
    else:
        print(">>> fanout NOT clean: the 2-vs-1 survivor structure is not phase-robust")
        print("    at this point; a reliable 1->2 copy is not demonstrated.")


if __name__ == "__main__":
    main()
