"""U5: depth-2 delivery -- does a TURNED signal land within a downstream gate
window, or does the no-restoration jitter scatter it past?

Routing.py showed a deflection turn has ~8-14deg exit-angle jitter over breath
phase. The decisive depth-2 question: a gate's operating window is ~7px wide
(gate.py, b=2..8). If the turned signal's LANDING position at a realistic
downstream distance spreads wider than that window, a second gate cannot reliably
catch it -- confirming "no positional/direction restoration" as a hard blocker
for depth>=2 routing. This records the survivor's landing position across phase
at increasing downstream distances and compares the spread to the 7px window.

Run:  python depth2.py
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

GATE_WINDOW = 7.0   # px, the inhibit gate's phase-robust operating band (b=2..8)


def landing(seed, vS, seedM, vM, rule_kw, gm, b, extra, D=70, G=461):
    """Survivor (y,x) position `extra` steps after the collision time tau."""
    spd = float(np.hypot(*vS)); tau = D / spd; N = int(tau) + extra
    P = np.array([G / 2.0, G / 2.0])
    nM = np.array([-vM[1], vM[0]]) / (np.hypot(*vM) + 1e-9)
    field = stamp(G, seed, *(P - np.asarray(vS) * tau)) + \
        stamp(G, seedM, *(P - np.asarray(vM) * tau + b * nM))
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    A = lenia.rollout(jnp.asarray(field), Kf, rule, N)[0]
    bl = collide.classify(np.asarray(A), gm)
    bl = [(m, y, x) for m, y, x in bl if m > 0.6 * gm]
    if not bl:
        return None
    # the survivor furthest from P along the exit (largest displacement)
    m, y, x = max(bl, key=lambda t: (t[1] - P[0]) ** 2 + (t[2] - P[1]) ** 2)
    return np.array([y, x])


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    seedM, vM = collide.partner(seed, vS, "perp_cw", rule_kw)
    phis = list(range(0, 24, 3))
    b = 5
    print(f"depth-2 delivery: turned-signal landing spread vs downstream distance "
          f"(gate window = {GATE_WINDOW:.0f}px)")
    for extra in (40, 80, 120):
        pts = []
        for phi in phis:
            M0 = phase_advance(seed, rule_kw, phi)
            sM = np.ascontiguousarray(collide.GEOMS["perp_cw"](M0))
            vMp = collide.measure_velocity(sM, rule_kw)
            p = landing(seed, vS, sM, vMp, rule_kw, gm, b, extra)
            if p is not None:
                pts.append(p)
        pts = np.array(pts)
        if len(pts) < 2:
            print(f"  +{extra} steps: too few survivors detected ({len(pts)})")
            continue
        c = pts.mean(0)
        dev = np.hypot(*(pts - c).T)               # radial deviation from centroid
        dist = float(np.hypot(*(c - 230.5)))       # downstream distance from grid centre
        spread = float(dev.max())
        ok = spread <= GATE_WINDOW / 2
        print(f"  +{extra} steps (~{dist:.0f}px downstream): {len(pts)} landings, "
              f"max spread {spread:.1f}px  -> {'within' if ok else 'EXCEEDS'} gate window")
    print(f"\n>>> if the spread EXCEEDS ~{GATE_WINDOW/2:.0f}px (half-window) a downstream gate")
    print("    cannot reliably catch the turned signal -> no positional restoration is a")
    print("    real depth>=2 routing blocker. Straight wires (no turn) are unaffected.")


if __name__ == "__main__":
    main()
