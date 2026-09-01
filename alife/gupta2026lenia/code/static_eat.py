"""A (last probe, step 2): test the still-life as an EATER / REFLECTOR / WALL.

static_eater.py found a stationary stable still-life in the Orbium rule. The
decisive question for the absorption blocker: when an Orbium runs into it, is the
Orbium ABSORBED (gone) and the still-life restored (an eater) -- or REFLECTED
(turned, still-life survives) -- or does the wall get destroyed? Either eater or
reflector would break a boundary blocker. Tested across impact parameter and
breath phase.

Run:  python static_eat.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lenia
import collide
import orbium_collide
from static_eater import disk, find_fixedpoint
from gate import stamp
from orbium_phase_map import phase_advance


def make_stilllife(rule128):
    """Regenerate the still-life and crop it to a localized patch."""
    sl, res = find_fixedpoint(disk(8, 0.25), rule128)
    A = lenia.recenter(jnp.asarray(sl))
    h = sl.shape[0] // 2
    patch = np.asarray(A[h - 24:h + 24, h - 24:h + 24])
    return patch, res, float(patch.sum())


def main():
    seedO, rule_kw = orbium_collide.orbium()
    rule128 = lenia.Rule(grid=128, **rule_kw)
    sl, res, slm = make_stilllife(rule128)
    vO = collide.measure_velocity(seedO, rule_kw)
    gm = collide.lone_mass(seedO, rule_kw)
    print(f"still-life: fixedpt res={res:.4f} mass={slm:.1f}; Orbium gm={gm:.1f} |v|={np.hypot(*vO):.3f}")

    # verify the still-life persists + stays localized on a large grid, alone
    G = 301
    rule = lenia.Rule(grid=G, **rule_kw); Kf = lenia.kernel_fft(rule)
    A = jnp.asarray(stamp(G, sl, G / 2, G / 2))
    m0 = float(jnp.sum(A))
    A = lenia.rollout(A, Kf, rule, 400)[0]
    bl = collide.classify(np.asarray(A), slm)
    c = lenia.center_of_mass(A)
    drift = float(np.hypot(c[0] - G / 2, c[1] - G / 2))
    print(f"verify (alone, 400 steps): mass {m0:.1f}->{float(jnp.sum(A)):.1f}, "
          f"blobs={len(bl)}, CoM drift={drift:.1f}px  "
          f"-> {'persistent localized wall' if len(bl)==1 and drift<5 else 'NOT a stable wall'}")

    # eater/reflector test: fire an Orbium into the wall at the grid centre
    print("\nfire Orbium into the wall (D=80); outcome across impact parameter b:")
    D = 80; spd = float(np.hypot(*vO)); tau = D / spd; N = int(tau) + 120
    P = np.array([G / 2.0, G / 2.0])
    n = np.array([-vO[1], vO[0]]) / spd
    snaps = {}
    for b in range(0, 22, 3):
        field = stamp(G, sl, P[0], P[1]) + stamp(G, seedO, *(P - vO * tau + b * n))
        A = lenia.rollout(jnp.asarray(field), Kf, rule, N)[0]
        mass = float(jnp.sum(A))
        bl = collide.classify(np.asarray(A), gm)
        big = [(m, y, x) for m, y, x in bl if m > 0.4 * gm]
        wall_here = any(np.hypot(y - P[0], x - P[1]) < 18 for _, y, x in big)
        movers = [(m, y, x) for m, y, x in big if np.hypot(y - P[0], x - P[1]) >= 18]
        # absorbed: wall remains, no mover (Orbium gone); reflected: wall + 1 mover
        if wall_here and not movers and abs(mass - slm) < 0.4 * gm:
            tag = "ABSORBED (eater!)"
        elif wall_here and movers:
            tag = f"reflected/passed ({len(movers)} mover)"
        elif not wall_here:
            tag = "WALL DESTROYED"
        else:
            tag = "other"
        print(f"  b={b:2d}: final mass={mass:.0f} (wall~{slm:.0f}+gm~{gm:.0f}={slm+gm:.0f})  "
              f"wall_intact={wall_here} movers={len(movers)}  -> {tag}")
        if b in (0, 9):
            snaps[b] = np.asarray(A)

    fig, axs = plt.subplots(1, len(snaps), figsize=(5 * len(snaps), 5))
    axs = np.atleast_1d(axs)
    for ax, (b, fld) in zip(axs, snaps.items()):
        ax.imshow(fld, cmap="viridis", vmin=0, vmax=1); ax.axis("off"); ax.set_title(f"b={b}")
    fig.suptitle("Orbium into the still-life wall")
    fig.savefig("assets/orbium_wall.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    print("saved assets/orbium_wall.png")


if __name__ == "__main__":
    main()
