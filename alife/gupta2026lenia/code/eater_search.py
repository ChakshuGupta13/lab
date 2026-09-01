"""U1 (universality step 1): proper EATER search.

The absorption probe (absorb.py) only checked head-on; the review flagged it
under-powered (mirror_ud reaches 3/4 at b=8, untested). Here we search ALL five
D4 collision geometries x impact parameter x breath phase for a PHASE-ROBUST
annihilator: an (geometry, b) where colliding the signal with that partner drives
the total mass to ~0 (ratio < 0.1) for EVERY breath phase. Such a partner is an
"eater" -- it absorbs a survivor, solving survivor management.

Run:  python eater_search.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import collide
import orbium_collide
from orbium_phase_map import phase_advance


def main():
    seed, rule_kw = orbium_collide.orbium()
    vS = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    phis = list(range(0, 24, 3))      # 8 breath phases over the period
    bs = list(range(0, 25, 2))        # 13 impact parameters

    print(f"eater search: annihilation (ratio<0.1) over 6 geoms x {len(bs)} b x "
          f"{len(phis)} phases")
    found = []
    for geom in collide.GEOMS:
        grid = np.zeros((len(phis), len(bs)), bool)
        for i, phi in enumerate(phis):
            C0 = phase_advance(seed, rule_kw, phi)
            sB = np.ascontiguousarray(collide.GEOMS[geom](C0))
            vB = collide.measure_velocity(sB, rule_kw)
            for j, b in enumerate(bs):
                res = collide.run(seed, vS, sB, vB, rule_kw, b, gm, D=70, N=224, G=361, track=True)
                grid[i, j] = res["ratio"] < 0.10
        robust = [bs[j] for j in range(len(bs)) if grid[:, j].all()]
        bestj = max(range(len(bs)), key=lambda j: grid[:, j].sum())
        print(f"  {geom:10s}: phase-robust annihilate b={robust}  "
              f"(best col b={bs[bestj]}: {grid[:, bestj].sum()}/{len(phis)}; "
              f"overall {100 * grid.mean():.0f}%)")
        if robust:
            found.append((geom, robust))

    if found:
        print(f"\n>>> EATER FOUND: {found}")
        print("    a phase-robust annihilator exists -> survivor absorption is available.")
    else:
        print("\n>>> NO phase-robust eater across all 6 geometries: annihilation is")
        print("    phase-fragile everywhere (annihilate vs merge1 flips with phase).")
        print("    Survivor absorption by collision is not clean; routing survivors off")
        print("    a large grid (bounded compute time) remains the fallback.")


if __name__ == "__main__":
    main()
