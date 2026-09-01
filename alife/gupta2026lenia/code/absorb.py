"""O4: the decisive survivor-absorption / interference probe.

The 2-gate cascade (cascade.py) structurally dodged the question of whether a
live DEFLECTED SURVIVOR corrupts a signal that must output 1 (review W1). Two
measurements probe whether bounded-Lenia Orbium gates can scale. OUTCOME
(review): both were INCONCLUSIVE -- TEST 1's keep-away is
the trivial geometric miss distance, TEST 2 searched only head-on of 5
geometries. Survivor management is left OPEN, neither solved nor shown fatal.

  TEST 1 (keep-away): a crossing glider passes the signal at lateral offset b.
    At what b does the signal reliably SURVIVE to its output window (= output 1
    WITH a glider present nearby)? Small keep-away => debris is manageable by
    layout (route survivors > keep-away from live signals). This directly tests
    "can a gate emit 1 amid debris?".

  TEST 2 (eater): can a head-on collision ANNIHILATE a survivor (mass -> 0) so
    it is truly absorbed, reliably across breath phase? Phase-robust annihilation
    => a usable eater; phase-fragile (annihilate vs merge1 flips with phase) =>
    no clean absorber.

Run:  python absorb.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lenia
import collide
import orbium_collide
from gate import run_gate
from orbium_phase_map import phase_advance


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    phis = list(range(0, 24, 6))

    # ---- TEST 1: keep-away distance (output 1 amid a passing crossing glider) ----
    print("TEST 1 keep-away: does the signal SURVIVE (out=1) vs crossing-glider offset b?")
    survive_b = None
    for b in range(0, 33, 2):
        n = 0
        for phi in phis:
            C0 = phase_advance(seed, rule_kw, phi)
            sC = np.ascontiguousarray(collide.GEOMS["perp_cw"](C0))
            vCp = collide.measure_velocity(sC, rule_kw)
            n += run_gate(seed, vS, sC, vCp, rule_kw, gm, 1, 1, b)
        flag = "survives" if n == len(phis) else ("blocked" if n == 0 else "mixed")
        print(f"  b={b:2d}: signal out=1 in {n}/{len(phis)} phases  ({flag})")
        if survive_b is None and n == len(phis):
            survive_b = b
    print(f">>> keep-away ~{survive_b}px ~= r95+R (~9+13) = the geometric MISS distance")
    print("    (review): this is the trivial 'gliders miss beyond")
    print("    interaction range' spacing constraint, NOT debris tolerated near a live")
    print("    signal -- and it is not clean (interaction re-appears at larger b). Says")
    print("    nothing non-trivial about debris tolerance.")

    # ---- TEST 2: eater (phase-robust head-on annihilation) ----
    print("\nTEST 2 eater: head-on annihilation (final mass ratio < 0.1) across phase?")
    best = None
    for b in range(0, 22, 2):
        n = 0
        for phi in phis:
            E0 = phase_advance(seed, rule_kw, phi)
            sE = np.ascontiguousarray(collide.GEOMS["headon"](E0))
            vEp = collide.measure_velocity(sE, rule_kw)
            res = collide.run(seed, vS, sE, vEp, rule_kw, b, gm, D=70, N=224, G=361, track=True)
            n += (res["ratio"] < 0.10)
        print(f"  b={b:2d}: annihilates {n}/{len(phis)} phases")
        if n == len(phis):
            best = b
    if best is not None:
        print(f">>> phase-robust eater at b={best}: a head-on glider absorbs the survivor "
              f"every phase (true absorption available).")
    else:
        print(">>> no 4/4 head-on annihilation found -- BUT only HEAD-ON of 5 D4")
        print("    geometries was searched here; mirror_ud reaches 3/4 at b=8 (review),")
        print("    untested. Eater feasibility is OPEN (under-powered negative), not ruled")
        print("    out. Survivor management overall is OPEN: neither solved nor shown fatal")
        print("    (on a large grid with bounded compute time, survivors exit the active")
        print("    region; only the periodic torus would bring them back).")


if __name__ == "__main__":
    main()
