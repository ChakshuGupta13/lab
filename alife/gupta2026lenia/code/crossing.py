"""U3: signal crossing -- can two signal wires cross?

A planar circuit needs wires to cross. Two Orbium signals on crossing tracks
(S_A along vA; S_B = a perp_cw-oriented glider along vB) are aimed at the same
crossing point P. If they arrive together they collide (Lenia solitons are not
transparent). We sweep the arrival-time offset dt between them and ask whether
BOTH reach their own output windows. dt=0 is a simultaneous crossing; large dt
means one has cleared P before the other arrives.

Verdict: if only large dt works, crossings need temporal scheduling (no
simultaneous crossing); if even large dt fails, crossing is impossible.

Run:  python crossing.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lenia
import collide
import orbium_collide
from gate import stamp


def _reaches(A1, A2, center, v, gm):
    b1 = collide.classify(np.asarray(A1), gm); b2 = collide.classify(np.asarray(A2), gm)
    indeg = np.degrees(np.arctan2(v[1], v[0]))
    for m, y, x in b2:
        if m < 0.6 * gm or np.hypot(y - center[0], x - center[1]) > 22 or not b1:
            continue
        k = min(range(len(b1)), key=lambda i: (b1[i][1] - y) ** 2 + (b1[i][2] - x) ** 2)
        ang = np.degrees(np.arctan2((x - b1[k][2]) / 20.0, (y - b1[k][1]) / 20.0))
        if abs((ang - indeg + 180) % 360 - 180) < 40:
            return 1
    return 0


def run_cross(seed, vA, seedB, vB, rule_kw, gm, dt, D=80, TAIL=90, G=461):
    spdA = float(np.hypot(*vA)); tauA = D / spdA
    P = np.array([G / 2.0, G / 2.0])
    # A arrives at P at tauA; B arrives at tauA+dt
    pA = P - np.asarray(vA) * tauA
    pB = P - np.asarray(vB) * (tauA + dt)
    field = stamp(G, seed, *pA) + stamp(G, seedB, *pB)
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    N = int(tauA + abs(dt)) + TAIL
    A1 = lenia.rollout(jnp.asarray(field), Kf, rule, N - 20)[0]
    A2 = lenia.rollout(A1, Kf, rule, 20)[0]
    outA = P + np.asarray(vA) * (N - tauA)
    outB = P + np.asarray(vB) * (N - tauA - dt)
    return _reaches(A1, A2, outA, vA, gm), _reaches(A1, A2, outB, vB, gm)


def main():
    seed, rule_kw = orbium_collide.orbium()
    vA = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    seedB = np.ascontiguousarray(collide.GEOMS["perp_cw"](seed))
    vB = collide.measure_velocity(seedB, rule_kw)
    print(f"crossing: vA={vA.round(2)} (in {np.degrees(np.arctan2(vA[1],vA[0])):.0f}deg), "
          f"vB={vB.round(2)} (in {np.degrees(np.arctan2(vB[1],vB[0])):.0f}deg)")
    print("arrival-time offset dt vs (A reaches out_A, B reaches out_B):")
    clean = None
    for dt in range(0, 90, 10):
        a, b = run_cross(seed, vA, seedB, vB, rule_kw, gm, dt)
        tag = "BOTH cross" if (a and b) else ("A only" if a else ("B only" if b else "both lost"))
        print(f"  dt={dt:3d}: A={a} B={b}  ({tag})")
        if clean is None and a and b:
            clean = dt
    if clean == 0:
        print(">>> simultaneous crossing WORKS (unexpected for non-integrable solitons).")
    elif clean is not None:
        print(f">>> NO simultaneous crossing; both signals survive only when separated by "
              f"dt>={clean} steps -> crossings need TEMPORAL SCHEDULING.")
    else:
        print(">>> crossing FAILS at all tested dt -> wire crossing not supported.")


if __name__ == "__main__":
    main()
