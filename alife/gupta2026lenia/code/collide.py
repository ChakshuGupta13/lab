"""Glider collision harness with D4 partner geometries.

For an Asymptotic-Lenia glider (seed S, velocity v), the dynamics are
equivariant under the dihedral group D4 of the square (the convolution
kernel is radially symmetric, so rotating/reflecting a glider gives another
glider moving in the transformed direction). We exploit this to build
collision partners by EXACT, interpolation-free array transforms — no
re-discovery needed.

Each (glider, geometry) pair is collided over an impact-parameter sweep and
classified into one of:
    annihilate   - mass -> 0
    explode      - runaway labyrinthine growth (AL has no mass bound)
    SURVIVE2     - the two gliders interact AND both survive as solitons
                   (reflection, transmission, or scatter -- the outgoing
                   direction is measured separately; construction-relevant)
    miss         - two solitons survive but never interacted (no gate)
    merge1       - a single soliton survives
    other        - everything else

The SURVIVE2 label is the headline: its presence (with a non-trivial
impact-parameter basin) is evidence FOR a Lenia logic gate; its universal
absence across gliders x geometries is evidence for the obstruction.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from scipy import ndimage

jax.config.update("jax_enable_x64", True)
import lenia

SEED = 96                       # seed patch is SEED x SEED
THRESH = 0.10                   # cell "alive" threshold for blob labelling
GLIDER = "assets/glider_al_v1.npz"

# D4 partner transforms (exact, interpolation-free). Identity is omitted:
# two identical gliders moving the same way never converge.
GEOMS = {
    "headon":    lambda S: S[::-1, ::-1],   # rotate 180  -> moves -v
    "perp_cw":   lambda S: np.rot90(S, 1),   # rotate  90
    "perp_ccw":  lambda S: np.rot90(S, 3),   # rotate 270
    "mirror_lr": lambda S: S[:, ::-1],       # reflect across vertical axis
    "mirror_ud": lambda S: S[::-1, :],       # reflect across horizontal axis
    "transpose": lambda S: S.T,              # reflect across main diagonal
}


def measure_velocity(seed, rule_kw, steps=40, G=256):
    """Run a lone glider a few steps on a clean grid and fit its CoM velocity."""
    seed = jnp.asarray(seed)
    s = seed.shape[0]
    o = (G - s) // 2
    A = jnp.zeros((G, G)).at[o:o + s, o:o + s].set(seed)
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    rows, cols = [], []
    for _ in range(steps):
        c = lenia.center_of_mass(A)
        rows.append(float(c[0])); cols.append(float(c[1]))
        A = lenia.step(A, Kf, rule)
    ts = np.arange(steps)
    vr = float(np.polyfit(ts, rows, 1)[0])
    vc = float(np.polyfit(ts, cols, 1)[0])
    return np.array([vr, vc])


def partner(seed, v, geom, rule_kw):
    """Build a collision partner from a D4 transform; measure its velocity."""
    Sb = np.ascontiguousarray(GEOMS[geom](np.asarray(seed)))
    vb = measure_velocity(Sb, rule_kw)
    return Sb, vb


def lone_mass(seed, rule_kw, settle=40, G=256):
    """Steady-state mass of a single glider after a short settle."""
    seed = jnp.asarray(seed)
    s = seed.shape[0]; o = (G - s) // 2
    A = jnp.zeros((G, G)).at[o:o + s, o:o + s].set(seed)
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    A = lenia.rollout(A, Kf, rule, settle)[0]
    return float(jnp.sum(A))


def place(A, patch, cy, cx):
    """Add `patch` (square) into A with its centre at (cy, cx)."""
    s = patch.shape[0]
    y0 = int(round(cy - s / 2)); x0 = int(round(cx - s / 2))
    y0 = max(0, min(A.shape[0] - s, y0)); x0 = max(0, min(A.shape[1] - s, x0))
    return A.at[y0:y0 + s, x0:x0 + s].add(jnp.asarray(patch))


def setup(seedA, vA, seedB, vB, b, D, G):
    """Aim both gliders at the grid centre; offset B by b perpendicular to vB.

    Without interaction, A passes through P and B through P + b*nB at the same
    crossing time tau = D/|vA|, so b=0 is a direct hit for any geometry.
    """
    P = np.array([G / 2.0, G / 2.0])
    vA = np.asarray(vA, float); vB = np.asarray(vB, float)
    tau = D / (np.hypot(*vA) + 1e-9)
    nB = np.array([-vB[1], vB[0]]) / (np.hypot(*vB) + 1e-9)
    posA = P - vA * tau
    posB = P - vB * tau + b * nB
    A = jnp.zeros((G, G))
    A = place(A, seedA, posA[0], posA[1])
    A = place(A, seedB, posB[0], posB[1])
    return A


def classify(field, glider_mass):
    """Label connected alive-blobs; return [(mass, y, x), ...] sorted by mass."""
    field = np.asarray(field)
    lab, n = ndimage.label(field > THRESH)
    out = []
    for i in range(1, n + 1):
        m = float(field[lab == i].sum())
        if m > 0.30 * glider_mass:
            ys, xs = np.where(lab == i)
            out.append((m, float(ys.mean()), float(xs.mean())))
    out.sort(key=lambda t: -t[0])
    return out


def run(seedA, vA, seedB, vB, rule_kw, b, glider_mass, D=120, N=280, G=720, snaps=None, track=False):
    """Collide A and B at impact parameter b; track mass and final blobs.

    track=True classifies periodically to detect interaction even when total
    mass is conserved (a reflection in a bounded substrate keeps mass ~constant
    but momentarily merges the two blobs into one).
    """
    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    A = setup(seedA, vA, seedB, vB, b, D, G)
    twogm = 2.0 * glider_mass
    masses = []
    frames = {}
    snaps = snaps or []
    nblob_min = 2
    for t in range(N):
        m = float(jnp.sum(A))
        masses.append(m)
        if t in snaps:
            frames[t] = np.asarray(A)
        if track and t % 4 == 0:
            big = [1 for mm, _, _ in classify(A, glider_mass) if mm > 0.5 * glider_mass]
            nblob_min = min(nblob_min, len(big))
        A = lenia.step(A, Kf, rule)
    masses = np.array(masses)
    blobs = classify(A, glider_mass)
    # interaction = mass deviated, OR the two solitons merged into one at some t
    dev = np.abs(masses - twogm) / twogm
    interacted = bool(dev.max() > 0.15 or (track and nblob_min < 2))
    return dict(masses=masses, blobs=blobs, interacted=interacted, nblob_min=nblob_min,
                ratio=float(masses[-1] / twogm), frames=frames)


def label_outcome(res, glider_mass):
    """Map a run() result to a single outcome label (see module docstring)."""
    ratio = res["ratio"]; blobs = res["blobs"]; gm = glider_mass
    if ratio < 0.10:
        return "annihilate"
    if ratio > 1.80:
        return "explode"
    bigs = [m for m, _, _ in blobs if m > 0.50 * gm]
    if len(bigs) >= 2 and 0.70 < ratio < 1.70:
        return "SURVIVE2" if res["interacted"] else "miss"
    if len(bigs) == 1 and 0.30 < ratio < 1.70:
        return "merge1"
    return "other"


def sweep(seedA, vA, seedB, vB, rule_kw, glider_mass, bs, **kw):
    """Run an impact-parameter sweep; return {b: (label, res)}."""
    out = {}
    for b in bs:
        res = run(seedA, vA, seedB, vB, rule_kw, b, glider_mass, **kw)
        out[b] = (label_outcome(res, glider_mass), res)
    return out


def main():
    d = np.load(GLIDER)
    seed = np.array(d["seed"]); v = np.array(d["v"])
    rule_kw = dict(kr=float(d["kr"]), muK=float(d["muK"]), sigmaK=float(d["sigmaK"]),
                   muT=float(d["muT"]), sigmaT=float(d["sigmaT"]), dt=float(d["dt"]))
    gm = lone_mass(seed, rule_kw)

    geom = sys.argv[1] if len(sys.argv) > 1 else "headon"
    seedB, vB = partner(seed, v, geom, rule_kw)
    print(f"geom={geom}  vA={v.round(3)}  vB={vB.round(3)}  glider_mass~{gm:.1f}")
    for b in range(0, 64, 8):
        res = run(seed, v, seedB, vB, rule_kw, b, gm)
        print(f"  b={b:3d}  {label_outcome(res, gm):12s}  "
              f"ratio={res['ratio']:.2f}  blobs={len(res['blobs'])}  "
              f"interacted={res['interacted']}")


if __name__ == "__main__":
    main()
