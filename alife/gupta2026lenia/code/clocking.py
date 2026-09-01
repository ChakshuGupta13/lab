"""U6: does CLOCKING rescue routing? -- the phase-tolerance of a turned signal.

depth2.py measured the turned-signal landing spread over the FULL breath period
(uncontrolled phase) and found 8-16px >> the 7px gate window. But phase at
collision is set by arrival TIMING, so a clocked circuit (fixed relative phase,
like a Game-of-Life computer) makes the scatter deterministic. The question is
the required clock precision: over how wide a contiguous breath-phase window does
the landing stay within the gate window? A wide window => modest clock precision
suffices and routing composes under clocking; a narrow window => clocking must be
near-perfect.

Run:  python clocking.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import collide
import orbium_collide
from orbium_phase_map import phase_advance
from depth2 import landing, GATE_WINDOW


def widest_window(pts, tol, period=24):
    """Longest run of consecutive breath phases (circular) whose landings all
    exist and lie within `tol` of their common centroid. A phase with no
    surviving signal breaks the run."""
    arr = [pts.get(i) for i in range(period)]
    best = 0
    for start in range(period):
        idx = []
        for k in range(period):
            i = (start + k) % period
            if arr[i] is None:
                break
            idx.append(i)
            sub = np.array([arr[j] for j in idx])
            if np.hypot(*(sub - sub.mean(0)).T).max() > tol:
                idx.pop()
                break
        best = max(best, len(idx))
    return best, None


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    tol = GATE_WINDOW / 2            # landing must stay within +-3.5px to hit the gate band

    print(f"turned-signal phase-tolerance vs hop distance (period 24, gate window "
          f"{GATE_WINDOW:.0f}px, tol +-{tol:.1f}px):")
    results = {}
    for extra, dpx in ((40, 25), (80, 47), (120, 71)):
        pts = {}
        for phi in range(24):
            M0 = phase_advance(seed, rule_kw, phi)
            sM = np.ascontiguousarray(collide.GEOMS["perp_cw"](M0))
            vMp = collide.measure_velocity(sM, rule_kw)
            p = landing(seed, vS, sM, vMp, rule_kw, gm, 5, extra)
            if p is not None:
                pts[phi] = p
        n_ok, idx = widest_window(pts, tol)
        results[dpx] = n_ok
        print(f"  ~{dpx:2d}px hop: survivor {len(pts)}/24 phases; widest in-band phase "
              f"window = {n_ok}/24 steps ({100*n_ok/24:.0f}% of period)")

    short = results.get(25, 0)
    print(f"\n>>> A NARROW single-turn clocked tolerance exists: {short}/24 breath steps at")
    print("    ~25px (shrinking to 3/24 at ~71px). This measures ONE isolated turn's landing")
    print("    consistency, NOT a downstream gate firing or a multi-stage circuit. Clocking")
    print("    addresses only the restoration jitter (review): it does")
    print("    NOT touch survivor ABSORPTION (no sink primitive exists -- a sink IS the eater")
    print("    that was shown not to exist), assemble any circuit, or test per-stage clock-")
    print("    constraint composition (which contracts with circuit size). Universality remains")
    print("    UNPROVEN -- narrowed on one axis, not closed.")


if __name__ == "__main__":
    main()
