"""O2: Orbium-Orbium collision study in bounded (standard) Lenia.

The construction go/no-go. Do two Orbia ever survive a collision as solitons
(reflect or pass through)? A reliable, input-dependent soliton-preserving
outcome is the basis for a logic gate. The substrate is mass-BOUNDED, so there
is no explosion; outcomes are annihilate / merge1 / PASS (2 survive AND
interacted) / miss (2 survive, no interaction) / other.

Odd grid => the D4 partner transforms are EXACT (rotation centre on a cell),
fixing the even-grid asymmetry the review flagged for the AL study.

Run:  python orbium_collide.py [geom]
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lenia
import collide

SEEDS = "assets/orbium_seeds.npz"


def orbium(which="lores"):
    d = np.load(SEEDS)
    rule_kw = dict(kr=float(d["R"]), muT=float(d[f"{which}_mu"]),
                   sigmaT=float(d[f"{which}_sigma"]), dt=float(d["dt"]),
                   asymptotic=False, kernel_core="poly", growth_core="poly")
    return np.asarray(d[which]), rule_kw


def survivor_dirs(f0, f1, dt_frames, gm):
    """Outgoing velocity direction (deg) of each survivor between two late frames."""
    b0 = collide.classify(f0, gm); b1 = collide.classify(f1, gm)
    out = []
    for _, y, x in b1:
        if not b0:
            continue
        j = min(range(len(b0)), key=lambda k: (b0[k][1] - y) ** 2 + (b0[k][2] - x) ** 2)
        vy = (y - b0[j][1]) / dt_frames; vx = (x - b0[j][2]) / dt_frames
        out.append(np.degrees(np.arctan2(vx, vy)))
    return out


def main():
    seed, rule_kw = orbium()
    vA = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    geom = sys.argv[1] if len(sys.argv) > 1 else "headon"
    seedB, vB = collide.partner(seed, vA, geom, rule_kw)
    spd = float(np.hypot(*vA))
    G, D = 361, 70
    N = int(D / spd) + 110          # classify ~110 steps post-collision, before any torus wrap
    in_ang = np.degrees(np.arctan2(vA[1], vA[0]))
    print(f"Orbium  geom={geom}  vA={vA.round(3)} vB={vB.round(3)} "
          f"|v|={spd:.3f}  glider_mass={gm:.1f}  (tau={D/spd:.0f}, N={N}, in_dir={in_ang:.0f} deg)")
    for b in range(0, 22, 3):
        snaps = [N - 31, N - 1]
        res = collide.run(seed, vA, seedB, vB, rule_kw, b, gm, D=D, N=N, G=G, snaps=snaps, track=True)
        lab = collide.label_outcome(res, gm)
        extra = ""
        if lab == "SURVIVE2":
            dirs = survivor_dirs(res["frames"][N - 31], res["frames"][N - 1], 30, gm)
            extra = f"  out_dirs={[round(a) for a in dirs]} deg"
        print(f"  b={b:2d}  {lab:13s}  ratio={res['ratio']:.2f}  "
              f"blobs={len(res['blobs'])}  nbmin={res['nblob_min']}{extra}")


if __name__ == "__main__":
    main()
