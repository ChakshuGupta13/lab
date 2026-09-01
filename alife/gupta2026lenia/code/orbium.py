"""Standard (mass-bounded) Lenia: the canonical Orbium glider.

Loads the verbatim Orbium seeds (parse_orbium.py) and runs them under Chan's
exact convention (poly-bump kernel, 1/2-factor Gaussian growth, clipped update
A<-clip(A+dt*G,0,1)). Standard Lenia is mass-BOUNDED (state in [0,1]), so unlike
Asymptotic Lenia it cannot explode on collision -- the substrate for the
soliton-collision logic-gate construction. Chan 2020 documents Orbium
reflections, the feasibility anchor.

Run:  python orbium.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lenia

SEEDS = "assets/orbium_seeds.npz"


def orbium_rule(which="lores", grid=256):
    """Build the canonical Orbium Rule: poly bump kernel + Chan quad4 poly growth."""
    d = np.load(SEEDS)
    mu = float(d[f"{which}_mu"]); sigma = float(d[f"{which}_sigma"])
    rule = lenia.Rule(grid=grid, kr=float(d["R"]), muT=mu, sigmaT=sigma,
                      dt=float(d["dt"]), asymptotic=False, kernel_core="poly",
                      growth_core="poly")
    return rule, np.asarray(d[which])


def place_center(seed, grid):
    s0, s1 = seed.shape
    oy = (grid - s0) // 2; ox = (grid - s1) // 2
    return jnp.zeros((grid, grid)).at[oy:oy + s0, ox:ox + s1].set(jnp.asarray(seed))


def glide_test(which, grid=256, N=200, verbose=True):
    rule, seed = orbium_rule(which, grid)
    Kf = lenia.kernel_fft(rule)
    A = place_center(seed, grid)
    h = grid // 2
    masses, rows, cols, peaks, locf = [], [], [], [], []
    for t in range(N):
        m = float(jnp.sum(A)); c = lenia.center_of_mass(A)
        cr = lenia.recenter(A)
        win = cr[h - 20:h + 20, h - 20:h + 20]
        masses.append(m); rows.append(float(c[0])); cols.append(float(c[1]))
        peaks.append(float(jnp.max(win))); locf.append(float(jnp.sum(win) / (m + 1e-9)))
        A = lenia.step(A, Kf, rule)
    masses = np.array(masses); rows = np.array(rows); cols = np.array(cols)
    lo = N // 8
    ts = np.arange(N)[lo:]
    cr_ = np.polyfit(ts, rows[lo:], 1); cc_ = np.polyfit(ts, cols[lo:], 1)
    lin = np.std(rows[lo:] - np.polyval(cr_, ts)) + np.std(cols[lo:] - np.polyval(cc_, ts))
    speed = float(np.hypot(cr_[0], cc_[0]))
    mr = float(masses[-1] / masses[lo])
    local = float(min(locf[lo:]))
    if verbose:
        print(f"{which:6s}: |v|={speed:.3f} v=({cr_[0]:+.3f},{cc_[0]:+.3f}) "
              f"lin={lin:.3f}px  mass {masses[lo]:.1f}->{masses[-1]:.1f} (mr={mr:.3f})  "
              f"local={local:.3f} peak={peaks[-1]:.3f}")
    return dict(which=which, speed=speed, vrow=float(cr_[0]), vcol=float(cc_[0]),
                lin=float(lin), mass_ratio=mr, mass0=float(masses[lo]), local=local,
                peak=float(peaks[-1]))


def main():
    print("Canonical Orbium (poly kernel + poly quad4 growth) gliding test (grid=256, N=200):")
    for which in ("lores", "hires"):
        glide_test(which)


if __name__ == "__main__":
    main()
