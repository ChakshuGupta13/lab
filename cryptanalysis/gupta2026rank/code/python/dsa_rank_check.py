#!/usr/bin/env python3
"""Verify the complete-NTT (ML-DSA) rank ceiling derived in
docs/pqc-fault-sat/notes/dsa-manual-proof.md.

Tests rank D_K = n - 1 and ker D_K = span(e_0) at n in {8, 16, 32}
under both zeroing (zeta -> 0) and general perturbation
(zeta -> zeta + 1). One fault per layer at group g = 0.

Run from this directory:  python3 dsa_rank_check.py
"""
import numpy as np
import random


def pow_mod(b, e, mod):
    r, b = 1, b % mod
    while e:
        if e & 1:
            r = (r * b) % mod
        b = (b * b) % mod
        e >>= 1
    return r


def rank_mod_p(A, p):
    A = A.copy() % p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        pivot = next((rr for rr in range(r, rows) if A[rr, c] % p != 0), None)
        if pivot is None:
            continue
        A[[r, pivot]] = A[[pivot, r]]
        inv = pow_mod(int(A[r, c]), p - 2, p)
        for rr in range(rows):
            if rr != r and A[rr, c] % p != 0:
                f = (A[rr, c] * inv) % p
                A[rr] = (A[rr] - f * A[r]) % p
        r += 1
    return r


def run_case(n, q, zeta_root, seed=42):
    """Return list of (label, rank, e0_in_ker, e1_in_ker) for (zeroing, general)."""
    random.seed(seed)
    m = int(np.log2(n))

    # Random nonzero twiddles per (layer, group).
    twiddles = {}
    for ell in range(m):
        length = n // (2 ** (ell + 1))
        ngroups = n // (2 * length)
        for g in range(ngroups):
            twiddles[(ell, g)] = pow_mod(zeta_root, random.randint(1, 2 * n - 1), q)

    def ntt(x, faults):
        y = list(x)
        for ell in range(m):
            length = n // (2 ** (ell + 1))
            ngroups = n // (2 * length)
            for g in range(ngroups):
                base = g * 2 * length
                z = faults.get((ell, g), twiddles[(ell, g)])
                for s in range(length):
                    a = y[base + s]
                    b = y[base + length + s]
                    y[base + s] = (a + z * b) % q
                    y[base + length + s] = (a - z * b) % q
        return y

    inv2 = pow_mod(2, q - 2, q)

    def invntt(y):
        """Gentleman-Sande inverse of the complete NTT (unfaulted twiddles)."""
        x = list(y)
        for ell in reversed(range(m)):
            length = n // (2 ** (ell + 1))
            ngroups = n // (2 * length)
            for g in range(ngroups):
                base = g * 2 * length
                z = twiddles[(ell, g)]
                zinv = pow_mod(z, q - 2, q)
                for s in range(length):
                    u = x[base + s]
                    v = x[base + length + s]
                    x[base + s] = ((u + v) * inv2) % q
                    x[base + length + s] = ((u - v) * inv2 * zinv) % q
        return x

    # Roundtrip sanity (complete NTT shape): INTT o NTT = id.
    _rng = random.Random(seed + 1000)
    probe = [_rng.randint(0, q - 1) for _ in range(n)]
    assert invntt(ntt(probe, {})) == probe, f"ML-DSA roundtrip FAILED (n={n}, q={q})"
    print(f"n={n:>3} q={q:>4}  roundtrip INTT o NTT = id: PASS")

    # Fault one group (g = 0) per layer.
    fault_zeroing = {(ell, 0): 0 for ell in range(m)}
    fault_general = {}
    for ell in range(m):
        orig = twiddles[(ell, 0)]
        new = (orig + 1) % q
        if new == 0:
            new = (orig + 2) % q
        fault_general[(ell, 0)] = new

    out = []
    for label, faults in [("zeroing", fault_zeroing), ("general", fault_general)]:
        DK = np.zeros((n, n), dtype=int)
        for i in range(n):
            e = [0] * n
            e[i] = 1
            yu = ntt(e, {})
            yf = ntt(e, faults)
            DK[:, i] = [(yu[j] - yf[j]) % q for j in range(n)]
        rk = rank_mod_p(DK, q)
        e0_in = all(DK[j, 0] % q == 0 for j in range(n))
        e1_in = all(DK[j, 1] % q == 0 for j in range(n))
        out.append((label, rk, e0_in, e1_in))
    return out


if __name__ == "__main__":
    # Primes chosen so 2n | q-1 (NTT-friendly).
    cases = [(8, 17, 3), (16, 97, 5), (32, 193, 5)]
    all_ok = True
    for n, q, root in cases:
        for label, rk, e0_in, e1_in in run_case(n, q, root):
            ok = (rk == n - 1) and e0_in and (not e1_in)
            all_ok &= ok
            print(
                f"n={n:>3} q={q:>4} {label:>8}: "
                f"rank={rk:>3} (expect {n - 1:>3}), "
                f"e0\u2208ker={e0_in}, e1\u2208ker={e1_in}  "
                f"{'OK' if ok else 'FAIL'}"
            )
    raise SystemExit(0 if all_ok else 1)
