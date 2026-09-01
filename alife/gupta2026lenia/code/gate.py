"""O3b: a collision-based DEFLECTION switch giving out = S AND NOT C in bounded Lenia.

A control glider C, aimed at the signal glider S with perpendicular offset b_ctrl,
DEFLECTS S off its output track. verified (review): at the
operating point a full-mass soliton SURVIVES every breath phase and is merely
routed away from the output window -- it is NOT destroyed. Absent C, S passes
straight through. Reading "a signal on the output line" as the output bit gives

    output = (S on output line) = (S present) AND NOT (C present)   -- inhibit / AND-NOT.

Output detector: a glider-mass blob in the downstream output window MOVING in S's
direction (a straight-through signal). out=0 means S was DEFLECTED off the line,
not annihilated -- the live off-track survivor is a real composition obstacle (it
must be absorbed or routed in any multi-gate circuit). This is a billiard-ball-
model-style deflection primitive, not a destructive gate.

Run:  python gate.py
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
from orbium_phase_map import phase_advance


def fourier_shift(A, dy, dx):
    """Shift a 2-D field by (dy, dx) pixels (sub-pixel) via FFT phase ramp."""
    n0, n1 = A.shape
    FY, FX = np.meshgrid(np.fft.fftfreq(n0), np.fft.fftfreq(n1), indexing="ij")
    ph = np.exp(-2j * np.pi * (FY * dy + FX * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(A) * ph))


def stamp(G, patch, cy, cx):
    """Place patch on a G x G field with its CoM at EXACT sub-pixel (cy, cx),
    eliminating the integer-rounding registration artifact."""
    patch = np.asarray(patch); s0, s1 = patch.shape
    ay = np.arange(s0); ax = np.arange(s1); m = patch.sum() + 1e-12
    py = (patch.sum(1) * ay).sum() / m; px = (patch.sum(0) * ax).sum() / m
    iy = max(0, min(G - s0, int(round(cy - py))))
    ix = max(0, min(G - s1, int(round(cx - px))))
    field = np.zeros((G, G)); field[iy:iy + s0, ix:ix + s1] = patch
    return fourier_shift(field, cy - (iy + py), cx - (ix + px))


def run_gate(seed, vA, seedC, vC, rule_kw, gm, S_on, C_on, b_ctrl,
             D=70, G=361, N=None, return_field=False):
    spd = float(np.hypot(*vA)); tau = D / spd
    if N is None:
        N = int(tau) + 90
    P = np.array([G / 2.0, G / 2.0])
    nC = np.array([-vC[1], vC[0]]) / (np.hypot(*vC) + 1e-9)

    field = np.zeros((G, G))
    if S_on:
        pS = P - np.asarray(vA) * tau
        field = field + stamp(G, seed, pS[0], pS[1])
    if C_on:
        pC = P - np.asarray(vC) * tau + b_ctrl * nC
        field = field + stamp(G, seedC, pC[0], pC[1])
    A = jnp.asarray(field)

    rule = lenia.Rule(grid=G, **{"asymptotic": True, **rule_kw})
    Kf = lenia.kernel_fft(rule)
    A1 = lenia.rollout(A, Kf, rule, N - 20)[0]
    A2 = lenia.rollout(A1, Kf, rule, 20)[0]

    out_center = P + np.asarray(vA) * (N - tau)
    b1 = collide.classify(np.asarray(A1), gm)
    b2 = collide.classify(np.asarray(A2), gm)
    # output = a glider-mass blob near out_center in A2, moving ~vA (straight signal)
    indeg = np.degrees(np.arctan2(vA[1], vA[0]))
    out = 0
    for m, y, x in b2:
        if m < 0.6 * gm or np.hypot(y - out_center[0], x - out_center[1]) > 20 or not b1:
            continue
        k = min(range(len(b1)), key=lambda i: (b1[i][1] - y) ** 2 + (b1[i][2] - x) ** 2)
        vy, vx = (y - b1[k][1]) / 20.0, (x - b1[k][2]) / 20.0
        ang = np.degrees(np.arctan2(vx, vy))
        if abs((ang - indeg + 180) % 360 - 180) < 35:      # moving in S's direction
            out = 1; break
    return (out, np.asarray(A2)) if return_field else out


def main():
    seed, rule_kw = orbium_collide.orbium()
    vA = collide.measure_velocity(seed, rule_kw)
    gm = collide.lone_mass(seed, rule_kw)
    seedC, vC = collide.partner(seed, vA, "perp_cw", rule_kw)
    b_ctrl = 5                                     # centre of the phase-robust block band

    print(f"INHIBIT gate  vS={vA.round(2)} vC={vC.round(2)} gm={gm:.1f} b_ctrl={b_ctrl}")
    print("truth table (output should = S AND NOT C):")
    tt = {}
    for S_on in (0, 1):
        for C_on in (0, 1):
            o = run_gate(seed, vA, seedC, vC, rule_kw, gm, S_on, C_on, b_ctrl)
            tt[(S_on, C_on)] = o
            exp = S_on and not C_on
            mark = "ok" if o == exp else "XX MISMATCH"
            print(f"  S={S_on} C={C_on} -> out={o}  (expect {int(exp)})  {mark}")
    ok = all(tt[(s, c)] == (s and not c) for s in (0, 1) for c in (0, 1))
    print(f">>> truth table {'CORRECT' if ok else 'WRONG'} for inhibit (S AND NOT C)")
    print("    (mechanism: C DEFLECTS S off the output line; the signal soliton")
    print("     survives every phase as a live off-track glider, not destroyed)")

    # robustness of the inhibit ACTION (S=1,C=1 -> 0) over relative breath phase
    # x control offset (the other 3 rows are phase-trivial). A composable gate
    # must block regardless of the uncontrolled phase of the incoming control.
    phis = list(range(0, 25, 3)); band = list(range(2, 14))
    print(f"\ninhibit action S=1,C=1 -> 0 over phase x b_ctrl ('.'=blocked ok, 'L'=leaked):")
    blk = np.zeros((len(phis), len(band)), bool)
    for i, phi in enumerate(phis):
        C0 = phase_advance(seed, rule_kw, phi)
        seedC_p = np.ascontiguousarray(collide.GEOMS["perp_cw"](C0))
        vC_p = collide.measure_velocity(seedC_p, rule_kw)
        row = ""
        for j, b in enumerate(band):
            o = run_gate(seed, vA, seedC_p, vC_p, rule_kw, gm, 1, 1, b)
            blk[i, j] = (o == 0)
            row += "." if blk[i, j] else "L"
        print(f"  phi={phi:3d} b={band[0]:2d}..{band[-1]:2d}: {row}")
    correct = int(blk.sum()); total = blk.size
    robust = [band[j] for j in range(len(band)) if blk[:, j].all()]
    print(f">>> inhibit blocks in {correct}/{total} (phase x b) points ({100*correct/total:.0f}%)")
    print(f">>> phase-robust block band (all {len(phis)} phi): b={robust}  "
          f"-> a {len(robust)}px-wide reliable inhibit operating window")

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    for (S_on, C_on), ax in zip([(1, 0), (1, 1), (0, 1), (0, 0)], axs.flat):
        o, fld = run_gate(seed, vA, seedC, vC, rule_kw, gm, S_on, C_on, b_ctrl, return_field=True)
        ax.imshow(fld, cmap="viridis", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"S={S_on} C={C_on}  ->  out={o}", fontsize=11)
    fig.suptitle("Orbium INHIBIT gate (out = S AND NOT C): final fields", fontsize=12)
    fig.savefig("assets/orbium_inhibit_gate.png", dpi=85, bbox_inches="tight"); plt.close(fig)
    print("saved assets/orbium_inhibit_gate.png")


if __name__ == "__main__":
    main()
