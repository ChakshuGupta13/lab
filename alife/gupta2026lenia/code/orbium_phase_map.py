"""O3a diagnostic: is there a ROBUST operating basin for an Orbium-collision gate?

Maps collision outcome over (longitudinal breath phase phi) x (impact parameter
b) at Delta-b=1, per geometry. A wide b-band whose outcome is STABLE across phi
is a robust gate operating point. If every band is knife-edge (2-3 px) or flips
with phi, a reliable gate is unlikely and we reframe (characterisation paper).

phi is realised by pre-evolving partner B by phi steps in isolation (advancing
its internal breath clock) before the D4 transform, so A and B meet at relative
breath phase phi.

Run:  python orbium_phase_map.py [geom]
"""
import sys

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

CODE = {"annihilate": 0, "SURVIVE2": 1, "miss": 2, "merge1": 3, "other": 3}
GLYPH = {0: "A", 1: "S", 2: ".", 3: "F"}   # Annihilate / Survive2 / miss / Fuse(merge1)
CMAP = plt.matplotlib.colors.ListedColormap(["#1b1b3a", "#f2c14e", "#4a7ba6", "#c0392b"])


def breath_period(seed, rule_kw, G=200, N=160):
    """Dominant internal breathing period (steps) via radius-of-gyration FFT."""
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    s = seed.shape[0]; o = (G - s) // 2
    A = jnp.zeros((G, G)).at[o:o + s, o:o + s].set(jnp.asarray(seed))
    ax = jnp.arange(G); h = G // 2
    rg = []
    for _ in range(N):
        A = lenia.step(A, Kf, rule)
        cr = lenia.recenter(A)
        m = jnp.sum(cr) + 1e-9
        cy = jnp.sum(jnp.sum(cr, 1) * (ax - h) ** 2) / m
        cx = jnp.sum(jnp.sum(cr, 0) * (ax - h) ** 2) / m
        rg.append(float(jnp.sqrt(cy + cx)))
    rg = np.array(rg[N // 4:]); rg -= rg.mean()
    sp = np.abs(np.fft.rfft(rg)); freqs = np.fft.rfftfreq(len(rg))
    k = 1 + int(np.argmax(sp[1:]))
    return (1.0 / freqs[k]) if freqs[k] > 0 else float("nan")


def phase_advance(seed, rule_kw, phi, patch=34, G=160):
    """Orbium pre-evolved phi steps (breath clock advanced), cropped to a patch."""
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    s = seed.shape[0]; o = (G - s) // 2
    A = jnp.zeros((G, G)).at[o:o + s, o:o + s].set(jnp.asarray(seed))
    if phi > 0:
        A = lenia.rollout(A, Kf, rule, phi)[0]
    A = lenia.recenter(A)
    h = G // 2; hp = patch // 2
    return np.asarray(A[h - hp:h + hp, h - hp:h + hp])


def main():
    geom = sys.argv[1] if len(sys.argv) > 1 else "headon"
    seed, rule_kw = orbium_collide.orbium()
    vA = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    spd = float(np.hypot(*vA))
    P = breath_period(seed, rule_kw)
    G, D = 361, 70
    N = int(D / spd) + 110
    phis = list(range(0, int(round(P)) + 2, max(1, int(round(P / 8)))))
    bs = list(range(0, 25))
    print(f"geom={geom} |v|={spd:.3f} gm={gm:.1f} breath_period={P:.1f}  "
          f"phis={phis}  b=0..{bs[-1]} (db=1)  G={G} D={D} N={N}")

    grid = np.zeros((len(phis), len(bs)), int)
    for i, phi in enumerate(phis):
        B0 = phase_advance(seed, rule_kw, phi)
        seedB = np.ascontiguousarray(collide.GEOMS[geom](B0))
        vB = collide.measure_velocity(seedB, rule_kw)
        row = []
        for j, b in enumerate(bs):
            res = collide.run(seed, vA, seedB, vB, rule_kw, b, gm, D=D, N=N, G=G, track=True)
            lab = collide.label_outcome(res, gm)
            grid[i, j] = CODE.get(lab, 3)
            row.append(GLYPH[grid[i, j]])
        print(f"  phi={phi:3d}: " + "".join(row))

    nS = int((grid == 1).sum()); nA = int((grid == 0).sum())
    nMiss = int((grid == 2).sum()); nF = int((grid == 3).sum())
    print(f"\n4-state counts: SURVIVE2={nS} annihilate={nA} miss={nMiss} merge1/Fuse={nF}")

    # A gate needs two phase-ROBUST output states keyed by position b:
    #   output 1 = SURVIVE2 for every phi;  output 0 = DESTRUCTIVE (annihilate OR
    #   merge1) for every phi.  (Pure-annihilate is the wrong metric -- merge1 is
    #   also a destroyed-input outcome.)
    destructive = (grid == 0) | (grid == 3)
    surv = [bs[j] for j in range(len(bs)) if np.all(grid[:, j] == 1)]
    destr = [bs[j] for j in range(len(bs)) if np.all(destructive[:, j])]
    print(f"phase-robust SURVIVE2 b-cols (all {len(phis)} phi): {surv}")
    print(f"phase-robust DESTRUCTIVE b-cols (all {len(phis)} phi): {destr}")
    if surv and destr:
        print(f">>> GATE-FEASIBLE: position b keys a phase-robust switch "
              f"(survive at b={surv}, destroy at b={destr}).")
    else:
        print(">>> NO phase-robust position switch; check clocked (fixed-phi) windows.")

    # clocked option: per phi, widest contiguous SURVIVE2 and DESTRUCTIVE runs
    def widest(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    print("clocked (fixed-phi) widest windows  [SURVIVE2 | DESTRUCTIVE]:")
    for i, phi in enumerate(phis):
        print(f"  phi={phi:3d}:  {widest(grid[i] == 1)}px | {widest(destructive[i])}px")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(grid, aspect="auto", cmap=CMAP, vmin=0, vmax=3,
              extent=[bs[0] - 0.5, bs[-1] + 0.5, len(phis) - 0.5, -0.5])
    ax.set_xlabel("impact parameter b (px)"); ax.set_ylabel("phase index")
    ax.set_yticks(range(len(phis))); ax.set_yticklabels(phis)
    ax.set_title(f"Orbium {geom}: outcome map  (yellow=SURVIVE2, dark=annihilate, blue=miss)")
    fig.savefig(f"assets/orbium_phasemap_{geom}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"saved assets/orbium_phasemap_{geom}.png")


if __name__ == "__main__":
    main()
