"""U2: routing / turning -- can a deflected signal be AIMED?

Wiring gates requires (a) a straight "wire" (a lone glider glides straight to the
next gate -- trivially true) and (b) TURNS (redirect a signal to a gate not on
its straight path). The deflection gate turns a signal when the control fires; a
fixed "mirror" glider M can serve as a turn element. For routing to work the
turned output must leave in a REPRODUCIBLE direction (small spread over breath
phase), else it cannot be aimed at a downstream gate.

This probe measures the exit direction(s) of the survivor(s) of a signal-mirror
deflection at a fixed offset, across breath phase, and reports the spread.

Run:  python routing.py
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


def deflect_exits(seed, vS, seedM, vM, rule_kw, gm, b, D=70, G=361):
    spd = float(np.hypot(*vS)); tau = D / spd; N = int(tau) + 110
    P = np.array([G / 2.0, G / 2.0])
    nM = np.array([-vM[1], vM[0]]) / (np.hypot(*vM) + 1e-9)
    field = stamp(G, seed, *(P - np.asarray(vS) * tau)) + \
        stamp(G, seedM, *(P - np.asarray(vM) * tau + b * nM))
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
        dirs.append(np.degrees(np.arctan2((x - b1[k][2]) / 30.0, (y - b1[k][1]) / 30.0)))
    return dirs


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    in_ang = np.degrees(np.arctan2(vS[1], vS[0]))
    print(f"signal in-direction = {in_ang:.0f} deg")
    print("STRAIGHT WIRE: a lone glider glides straight -- trivially reproducible.\n")

    phis = list(range(0, 24, 3))
    print("TURN via mirror-deflection: survivor exit direction(s) vs breath phase")
    for b in (4, 5, 6):
        seedM, vM = collide.partner(seed, vS, "perp_cw", rule_kw)
        all_main = []
        rows = []
        for phi in phis:
            M0 = phase_advance(seed, rule_kw, phi)
            sM = np.ascontiguousarray(collide.GEOMS["perp_cw"](M0))
            vMp = collide.measure_velocity(sM, rule_kw)
            dirs = deflect_exits(seed, vS, sM, vMp, rule_kw, gm, b)
            rows.append((phi, [round(d) for d in dirs]))
            if dirs:
                all_main.append(max(dirs, key=lambda d: 1))  # record all; summarize below
        # summarise the dominant exit direction spread (circular std over phases)
        flat = [d for _, ds in rows for d in ds]
        if flat:
            ang = np.radians(flat)
            R = np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))
            circ_std = np.degrees(np.sqrt(-2 * np.log(R + 1e-12)))
            print(f"  b={b}: exits={[(p, ds) for p, ds in rows]}")
            print(f"        all exit angles mean~{np.degrees(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))):.0f}deg "
                  f"circular_std={circ_std:.0f}deg  (small std => aimable turn)")
    print("\n>>> routing verdict: straight wires trivial; turn is aimable at DEPTH 1 iff")
    print("    the exit spread is small (see per-b circular_std). CAVEAT: ~8deg jitter")
    print("    over a 70px hop is ~10px (2-3x the gate's operating band), and Orbium does")
    print("    NOT restore position/direction (only shape/speed) -- so multi-turn routing")
    print("    accumulates jitter with no correction and is UNPROVEN.")


if __name__ == "__main__":
    main()
