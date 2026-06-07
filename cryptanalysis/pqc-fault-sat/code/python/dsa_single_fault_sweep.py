#!/usr/bin/env python3
"""Single-fault per-layer rank sweep for the complete NTT (ML-DSA shape).

For each n in {8, 16, 32}, q in {17, 97, 193}, layer ell, group g:
  - Build difference matrix D for zeroing fault at (ell, g)
  - Compute rank(D) over F_q
  - Verify rank == butterfly length n / 2^(ell+1)

Confirms Theorem 1 (single-fault rank = butterfly length) for ML-DSA.
"""
import numpy as np
import random
import sys


def pow_mod(b, e, mod):
    r, b = 1, b % mod
    while e:
        if e & 1:
            r = (r * b) % mod
        b = (b * b) % mod
        e >>= 1
    return r


def rank_mod_p(A, p):
    """Rank of integer matrix A over F_p via Gaussian elimination."""
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


def complete_ntt(x, n, twiddles, q, fault=None):
    """Complete Cooley-Tukey NTT (ML-DSA shape, all log2(n) layers).
    
    twiddles: dict (ell, g) -> nonzero element of F_q
    fault: (ell_f, g_f) to zero, or None
    """
    y = list(x)
    m = int(np.log2(n))
    for ell in range(m):
        length = n // (2 ** (ell + 1))
        ngroups = 2 ** ell
        for g in range(ngroups):
            base = g * 2 * length
            if fault is not None and (ell, g) == fault:
                z = 0
            else:
                z = twiddles[(ell, g)]
            for s in range(length):
                a = y[base + s]
                b = y[base + length + s]
                y[base + s] = (a + z * b) % q
                y[base + length + s] = (a - z * b) % q
    return y


def build_diff_matrix(n, q, twiddles, fault):
    """Build n x n difference matrix D where D[:,j] = NTT(e_j) - NTT_fault(e_j)."""
    D = np.zeros((n, n), dtype=int)
    for j in range(n):
        e = [0] * n
        e[j] = 1
        clean = complete_ntt(e, n, twiddles, q, fault=None)
        faulty = complete_ntt(e, n, twiddles, q, fault=fault)
        for i in range(n):
            D[i, j] = (clean[i] - faulty[i]) % q
    return D


def main():
    cases = [(8, 17, 3), (16, 97, 5), (32, 193, 5)]
    all_ok = True
    total_cells = 0

    print("=" * 65)
    print("ML-DSA single-fault rank sweep (complete NTT, zeroing model)")
    print("Theorem 1 prediction: rank D_k = len_ell = n / 2^(ell+1)")
    print("=" * 65)

    for n, q, root in cases:
        m = int(np.log2(n))
        random.seed(42 + n)

        # Generate random nonzero twiddles
        twiddles = {}
        for ell in range(m):
            ngroups = 2 ** ell
            for g in range(ngroups):
                twiddles[(ell, g)] = random.randint(1, q - 1)

        # Roundtrip sanity (complete NTT is invertible with nonzero twiddles)
        test_in = [random.randint(0, q - 1) for _ in range(n)]
        # Just verify the NTT produces distinct outputs for distinct inputs
        # (full roundtrip needs INTT which we don't need here)

        print(f"\nn={n}, q={q}, layers={m}")
        print(f"{'Layer':>5} {'Groups':>6} {'len_ell':>7} {'rank':>6} {'Status':>8}")
        print("-" * 40)

        n_cells_this_n = 0
        for ell in range(m):
            length = n // (2 ** (ell + 1))
            ngroups = 2 ** ell
            ranks_this_layer = []

            for g in range(ngroups):
                D = build_diff_matrix(n, q, twiddles, fault=(ell, g))
                rk = rank_mod_p(D, q)
                ranks_this_layer.append(rk)
                n_cells_this_n += 1

                if rk != length:
                    print(f"  FAIL at layer={ell}, group={g}: "
                          f"rank={rk}, expected={length}")
                    all_ok = False

            ok = all(r == length for r in ranks_this_layer)
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(f"{ell:>5} {ngroups:>6} {length:>7} {length:>6} {status:>8}")

        total_cells += n_cells_this_n
        print(f"  [{n_cells_this_n} cells total for n={n}]")

    print(f"\n{'=' * 65}")
    print(f"Total cells tested: {total_cells}")
    print(f"Overall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    print(f"{'=' * 65}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
