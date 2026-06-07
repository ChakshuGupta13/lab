#!/usr/bin/env python3
"""Universal single-fault check for Theorem 1 (rank D_k = len_{ell(k)}).

Theorem 1 claims rank D_k = len_{ell(k)} for EVERY faulted twiddle index k
under ARBITRARY perturbation zeta'_k != zeta_k.  The two existing scripts each
test a slice:

  - generalization_check.py (G2): arbitrary delta, but only k = 2^ell (group 0).
  - dsa_single_fault_sweep.py:     every (layer, group), but zeroing only.

This script closes the remaining corner: EVERY single-fault index k, each with
a random non-zero delta AND a near-extreme delta (zeta'_k = zeta_k+1), for both
NTT shapes.  A FAIL here would refute Theorem 1 in full generality.
"""
import random

Q_KEM = 3329
Q_DSA = 8380417
ZETA_DSA = 1753


def bitrev(x, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def zetas_kem(n):
    root = pow(17, 512 // (2 * n), Q_KEM)
    half = n // 2
    lg = half.bit_length() - 1
    return [pow(root, bitrev(i, lg), Q_KEM) for i in range(half)]


def zetas_dsa(n):
    lg = n.bit_length() - 1
    return [0] + [pow(ZETA_DSA, bitrev(k, lg), Q_DSA) for k in range(1, n)]


def ntt(x, n, zetas, q, min_length, subst):
    """Cooley-Tukey forward NTT; subst maps twiddle index k -> replacement."""
    a = list(x)
    k, length = 1, n // 2
    while length >= min_length:
        start = 0
        while start < n:
            z = subst.get(k, zetas[k])
            for j in range(start, start + length):
                t = (z * a[j + length]) % q
                a[j + length] = (a[j] - t) % q
                a[j] = (a[j] + t) % q
            k += 1
            start += 2 * length
        length //= 2
    return a


def diff_rank(n, zetas, q, min_length, k, zeta_prime):
    """rank over F_q of D_k(x) = NTT(x) - NTT_k(x), columns = images of e_j."""
    cols_clean, cols_fault = [], []
    for j in range(n):
        e = [1 if i == j else 0 for i in range(n)]
        cols_clean.append(ntt(e, n, zetas, q, min_length, {}))
        cols_fault.append(ntt(e, n, zetas, q, min_length, {k: zeta_prime}))
    # D[i][j] = clean[j][i] - fault[j][i]
    D = [[(cols_clean[j][i] - cols_fault[j][i]) % q for j in range(n)]
         for i in range(n)]
    # Gaussian elimination over F_q
    r = 0
    for c in range(n):
        piv = next((rr for rr in range(r, n) if D[rr][c] % q), None)
        if piv is None:
            continue
        D[r], D[piv] = D[piv], D[r]
        inv = pow(D[r][c], q - 2, q)
        for rr in range(n):
            if rr != r and D[rr][c] % q:
                f = (D[rr][c] * inv) % q
                D[rr] = [(D[rr][j] - f * D[r][j]) % q for j in range(n)]
        r += 1
    return r


def run(label, n, zetas, q, min_length, kmax):
    random.seed(1234 + n)
    ok = True
    n_cells = 0
    per_layer = {}
    for k in range(1, kmax + 1):
        ell = k.bit_length() - 1               # floor(log2 k)
        L = n // (2 ** (ell + 1))               # len_{ell(k)}
        for zp in [(zetas[k] - random.randint(1, q - 1)) % q,  # random delta
                   (zetas[k] + 1) % q]:                         # delta = -1
            if zp == zetas[k]:
                continue
            r = diff_rank(n, zetas, q, min_length, k, zp)
            n_cells += 1
            if r != L:
                ok = False
                print(f"  FAIL {label} k={k} ell={ell} zeta'={zp}: "
                      f"rank={r} expected={L}")
        per_layer.setdefault(ell, L)
    layers = ", ".join(f"L{e}={per_layer[e]}" for e in sorted(per_layer))
    print(f"{label}: n={n} q={q} k=1..{kmax}  {n_cells} cells  "
          f"[{layers}]  {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("Theorem 1 universal single-fault check (every k, arbitrary delta)")
    print("=" * 70)
    a = run("ML-KEM (incomplete)", 256, zetas_kem(256), Q_KEM, 2, 256 // 2 - 1)
    b = run("ML-DSA (complete)  ", 32, zetas_dsa(32), Q_DSA, 1, 32 - 1)
    c = run("ML-DSA (complete)  ", 64, zetas_dsa(64), Q_DSA, 1, 64 - 1)
    print("=" * 70)
    print("OVERALL:", "ALL PASS" if (a and b and c) else "SOME FAIL")
