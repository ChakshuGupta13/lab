"""A (last probe): static still-life "eater"/wall search.

An eater in Game-of-Life is a STILL-LIFE (a stationary stable pattern), not a
collision -- a glider runs into it, is absorbed, and the still-life restores.
That is the one absorption mechanism not yet tested here (the eater_search only
tried moving collision partners). Step 1: does the Orbium RULE support a
stationary stable pattern at all? Relax candidate blobs (disks / Gaussian bumps)
and report which settle to a non-moving, mass-stable, persistent spot.

If a still-life is found, static_eat.py will test it as an absorber / reflector.

Run:  python static_eater.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lenia
import collide
import orbium_collide
from gate import stamp

G = 128


def rigorous_verify(A_full, rule_kw, Gbig=301, N=400):
    """Crop a candidate, re-stamp on a LARGE grid, roll N steps, and require it to
    stay localized (1 blob), low-drift, mass-stable, alive. The weak small-grid
    check gives false positives (grid-specific near-fixed-points that decay)."""
    A = lenia.recenter(jnp.asarray(A_full))
    h = A_full.shape[0] // 2
    patch = np.asarray(A[h - 26:h + 26, h - 26:h + 26])
    if patch.sum() < 5:
        return False, "dead-precrop"
    rule = lenia.Rule(grid=Gbig, **rule_kw)
    Kf = lenia.kernel_fft(rule)
    A2 = jnp.asarray(stamp(Gbig, patch, Gbig / 2, Gbig / 2))
    m0 = float(jnp.sum(A2))
    masses = []
    for _ in range(N):
        masses.append(float(jnp.sum(A2)))
        A2 = lenia.step(A2, Kf, rule)
    mf = float(jnp.sum(A2))
    c = lenia.center_of_mass(A2)
    drift = float(np.hypot(c[0] - Gbig / 2, c[1] - Gbig / 2))
    bl = collide.classify(np.asarray(A2), m0)
    cv = float(np.std(masses[N // 4:]) / (np.mean(masses[N // 4:]) + 1e-9))
    ok = mf > 5 and 0.8 < mf / (m0 + 1e-9) < 1.2 and drift < 5 and len(bl) == 1 and cv < 0.03
    return ok, f"mass {m0:.0f}->{mf:.0f} drift={drift:.0f}px blobs={len(bl)} cv={cv:.3f}"


def disk(r, val=1.0):
    ax = np.arange(G) - G // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return (val * ((yy ** 2 + xx ** 2) < r * r)).astype(float)


def bump(r, val=1.0):
    ax = np.arange(G) - G // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    d = np.sqrt(yy ** 2 + xx ** 2) / r
    return (val * np.exp(-(d ** 2)) * (d < 2)).astype(float)


def annulus(r0, r1, val=0.5):
    ax = np.arange(G) - G // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    d = np.sqrt(yy ** 2 + xx ** 2)
    return (val * ((d >= r0) & (d < r1))).astype(float)


def find_fixedpoint(warm, rule, steps=700, lr=0.02):
    """Gradient descent for a stationary pattern: minimise ||step(A)-A||^2 with a
    mass anchor (the v=0 glider equation that found the glider)."""
    Kf = lenia.kernel_fft(rule)
    A0 = jnp.clip(jnp.asarray(warm), 1e-4, 1 - 1e-4)
    raw = jnp.log(A0) - jnp.log(1 - A0)
    mtarget = float(jnp.sum(A0))

    def loss(raw):
        A = jax.nn.sigmoid(raw)
        An = lenia.step(A, Kf, rule)
        res = jnp.sum((An - A) ** 2) / (jnp.sum(A ** 2) + 1e-9)
        mpen = ((jnp.sum(A) - mtarget) / (mtarget + 1e-9)) ** 2
        return res + 0.3 * mpen, res

    gl = jax.jit(jax.value_and_grad(loss, has_aux=True))
    m = jnp.zeros_like(raw); v = jnp.zeros_like(raw)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        (val_, res), g = gl(raw)
        t = s + 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        raw = raw - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    A = jax.nn.sigmoid(raw)
    return np.asarray(A), float(res)


def relax(seed, rule, N=320):
    Kf = lenia.kernel_fft(rule)
    A = jnp.asarray(seed)
    rows, cols, masses, peaks = [], [], [], []
    for t in range(N):
        c = lenia.center_of_mass(A)
        rows.append(float(c[0])); cols.append(float(c[1]))
        masses.append(float(jnp.sum(A))); peaks.append(float(jnp.max(A)))
        A = lenia.step(A, Kf, rule)
    lo = N // 2
    vr = np.polyfit(np.arange(N)[lo:], rows[lo:], 1)[0]
    vc = np.polyfit(np.arange(N)[lo:], cols[lo:], 1)[0]
    speed = float(np.hypot(vr, vc))
    m0, mf = masses[lo], masses[-1]
    mr = mf / (m0 + 1e-9)
    return dict(A=np.asarray(A), speed=speed, mass0=m0, massf=mf, mr=mr,
                peak=peaks[-1], cv=float(np.std(masses[lo:]) / (np.mean(masses[lo:]) + 1e-9)))


def main():
    _, rule_kw = orbium_collide.orbium()
    rule = lenia.Rule(grid=G, **rule_kw)
    print("still-life search (Orbium rule): relax candidate blobs, want |v|~0, mass stable, alive")
    cands = [("disk", r, disk(r)) for r in (4, 6, 8, 10, 12, 14, 18, 24)] + \
            [("bump", r, bump(r)) for r in (6, 10, 14, 20)]
    found = []
    for kind, r, seed in cands:
        if seed.sum() < 1:
            continue
        res = relax(seed, rule)
        stationary = res["speed"] < 0.05
        alive = res["peak"] > 0.3 and res["massf"] > 5
        stable = 0.5 < res["mr"] < 1.6 and res["cv"] < 0.05
        flag = "STILL-LIFE" if (stationary and alive and stable) else ""
        print(f"  {kind} r={r:2d}: |v|={res['speed']:.3f} mass {res['mass0']:.0f}->{res['massf']:.0f} "
              f"(mr={res['mr']:.2f} cv={res['cv']:.3f}) peak={res['peak']:.2f}  {flag}")
        if stationary and alive and stable:
            found.append((kind, r))
    # Phase 2: gradient-descent fixed-point search from MANY warm starts, each
    # RIGOROUSLY verified on a large grid over 400 steps. (The weak 200-step / G=128
    # check gave a FALSE POSITIVE: a grid-specific near-fixed-point that actually
    # decays to 0 and drifts 200+px when re-stamped on a larger grid.)
    print("\nfixed-point search + RIGOROUS large-grid verify:")
    warms = []
    for r0 in (2, 3, 4, 5):
        for w in (3, 5):
            warms.append((f"ann{r0}-{r0+w}", annulus(r0, r0 + w, 0.4)))
    for r in (5, 7, 9, 11, 13):
        for d in (0.2, 0.3):
            warms.append((f"ldisk{r}@{d}", disk(r, d)))
    for name, w in warms:
        if w.sum() < 1:
            continue
        A, res = find_fixedpoint(w, rule, steps=500)
        ok, info = rigorous_verify(A, rule_kw)
        if ok:
            print(f"  {name:12s}: res={res:.4f}  VERIFIED STILL-LIFE  ({info})")
            found.append(name)
    print(f"  searched {len(warms)} warm starts (none printed above = none verified).")

    if found:
        print(f"\n>>> VERIFIED still-life(s): {found} -> static eater/reflector possible; test next.")
    else:
        print(f"\n>>> NO PERIOD-1 still-life from {len(warms)} fixed-point searches (and naive")
        print("    blobs die). NOTE (review): this rules out only")
        print("    PERIOD-1 fixed points + naive seeds. A STATIONARY BREATHER -- the natural")
        print("    analogue, since Orbium itself breathes -- is NOT representable by")
        print("    ||step(A)-A||^2 and is untested, as are catalytic/multi-body/phase-scheduled")
        print("    absorbers. So: no static absorber FOUND, not proven absent. Survivor")
        print("    absorption remains UNDEMONSTRATED (and is not the decisive blocker --")
        print("    signal RESTORATION is, since it also kills billiard-ball layouts).")


if __name__ == "__main__":
    main()
