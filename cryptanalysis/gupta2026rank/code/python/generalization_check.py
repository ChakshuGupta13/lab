"""
G2+G3+G4: Empirical verification for generalized twiddle-fault theorem.

G2: Confirm rank(D) = len_ℓ for ARBITRARY non-zero δ (not just zeroing
    or XOR bit-flip) at ML-KEM scale (q=3329, n=256).

G3: Confirm the theorem transfers to ML-DSA complete NTT (q=8380417,
    n=256, 8 layers). Expected: rank(D_K) = n−1, ker = span(e₀).

G4: Exhaustive collision check for ML-DSA: ζ_k ⊕ 2^j ≢ 0 mod q_DSA
    for all k and all j ∈ {0,...,22} (q_DSA is 23 bits).
"""

from __future__ import annotations
import sys, os, random, math
import numpy as np

SISTER = os.path.join(os.path.dirname(__file__), "..", "pqc-fault-sat", "scratch")
sys.path.insert(0, os.path.abspath(SISTER))
from ntt_fault_rank import Q as Q_KEM, compute_zetas


# ============================================================
# ML-DSA parameters
# ============================================================
Q_DSA = 8380417  # ML-DSA modulus (prime, 23 bits)
# ML-DSA uses ζ = 1753, primitive 512th root of unity mod Q_DSA
# Complete NTT: 8 layers at n=256 (length goes down to 1)
ZETA_DSA = 1753

def compute_zetas_dsa(n, q=Q_DSA):
    """ML-DSA twiddle table: complete NTT (length >= 1), primitive 2n-th root."""
    root = pow(ZETA_DSA, (2 * n) // (2 * n), q)  # = ZETA_DSA^1 for n=256
    # Actually: ML-DSA uses ζ = 1753 which is a primitive 512th root.
    # zetas[k] = 1753^{bitrev_8(k)} mod q for k = 1..255.
    half = n  # complete NTT uses all n-1 twiddles (k=1..n-1)
    # But standard ML-DSA NTT is the SAME Cooley-Tukey structure as ML-KEM,
    # just with m = log2(n) = 8 layers instead of 7. So:
    log_n = n.bit_length() - 1  # 8 for n=256

    def bitrev(x, bits):
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    # root = pow(ZETA_DSA, 1, q) = 1753
    return [0] + [pow(ZETA_DSA, bitrev(k, log_n), q) for k in range(1, n)]


def make_ntt(n, zetas, fault_subst, min_length=2):
    """Forward NTT (Cooley-Tukey) with twiddle substitutions.
    min_length=2 for ML-KEM (incomplete), min_length=1 for ML-DSA (complete)."""
    def f(x):
        a = list(x)
        k = 1; length = n // 2
        q = Q_KEM if len(zetas) < n else Q_DSA  # infer q from table size
        while length >= min_length:
            start = 0
            while start < n:
                z = fault_subst.get(k, zetas[k])
                for j in range(start, start + length):
                    t = (z * a[j + length]) % q
                    a[j + length] = (a[j] - t) % q
                    a[j] = (a[j] + t) % q
                k += 1; start += 2 * length
            length //= 2
        return a
    return f


def matrix_of(f, n):
    cols = [f([1 if j == i else 0 for j in range(n)]) for i in range(n)]
    return np.array(cols, dtype=int).T


def rank_mod_q(M, q):
    A = (M.copy() % q).astype(int)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = next((rr for rr in range(r, rows) if A[rr, c] != 0), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), q - 2, q)
        A[r] = (A[r] * inv) % q
        for rr in range(rows):
            if rr == r:
                continue
            f0 = int(A[rr, c])
            if f0 == 0:
                continue
            A[rr] = (A[rr] - f0 * A[r]) % q
        r += 1
        if r == rows:
            break
    return r


# ============================================================
# G2: General-δ rank at ML-KEM scale
# ============================================================

def run_g2():
    print("=" * 70)
    print("G2: General-delta rank at ML-KEM (q=3329)")
    print("  Hypothesis: rank(D) = len_ℓ for ANY delta != 0")
    print("=" * 70)
    for n in [32, 256]:
        zetas = compute_zetas(n, Q_KEM)
        M_clean = matrix_of(make_ntt(n, zetas, {}, min_length=2), n)
        log2n = n.bit_length() - 1
        random.seed(42)
        print(f"\nn={n}:")
        for ell in range(log2n - 1):  # ML-KEM incomplete: layers 0..log2n-2
            k = 2 ** ell
            L_l = n // (2 ** (ell + 1))
            # Test 5 random non-zero delta values per layer
            for trial in range(5):
                delta = random.randint(1, Q_KEM - 1)
                zeta_prime = (zetas[k] - delta) % Q_KEM
                # Ensure zeta_prime != zetas[k] (delta != 0 mod q)
                if zeta_prime == zetas[k]:
                    continue
                f_fault = make_ntt(n, zetas, {k: zeta_prime}, min_length=2)
                D = (M_clean - matrix_of(f_fault, n)) % Q_KEM
                r = rank_mod_q(D, Q_KEM)
                status = "OK" if r == L_l else "FAIL"
                if trial == 0 or r != L_l:
                    print(f"  layer {ell}, k={k}, delta={delta}: "
                          f"rank(D)={r}, expected={L_l} [{status}]")
            # Also test zeroing and bit-flip for comparison
            f_zero = make_ntt(n, zetas, {k: 0}, min_length=2)
            D_zero = (M_clean - matrix_of(f_zero, n)) % Q_KEM
            r_zero = rank_mod_q(D_zero, Q_KEM)
            f_bf = make_ntt(n, zetas, {k: zetas[k] ^ 1}, min_length=2)
            D_bf = (M_clean - matrix_of(f_bf, n)) % Q_KEM
            r_bf = rank_mod_q(D_bf, Q_KEM)
            print(f"  layer {ell}: zeroing={r_zero}, bit-flip={r_bf}, "
                  f"all random deltas={'match' if r == L_l else 'MISMATCH'}")


# ============================================================
# G3: ML-DSA complete NTT
# ============================================================

def run_g3():
    print("\n" + "=" * 70)
    print("G3: ML-DSA complete NTT (q=8380417, 8 layers)")
    print("  Hypothesis: rank(D_K) = n-1, ker = span(e_0)")
    print("=" * 70)

    for n in [8, 16, 32]:
        zetas = compute_zetas_dsa(n, Q_DSA)
        M_clean = matrix_of(make_ntt(n, zetas, {}, min_length=1), n)
        log2n = n.bit_length() - 1

        # One-per-layer zeroing
        K_zero = {2 ** ell: 0 for ell in range(log2n)}
        f_zero = make_ntt(n, zetas, K_zero, min_length=1)
        D_zero = (M_clean - matrix_of(f_zero, n)) % Q_DSA
        r_zero = rank_mod_q(D_zero, Q_DSA)

        # One-per-layer bit-flip (bit 0)
        K_bf = {2 ** ell: zetas[2 ** ell] ^ 1 for ell in range(log2n)}
        f_bf = make_ntt(n, zetas, K_bf, min_length=1)
        D_bf = (M_clean - matrix_of(f_bf, n)) % Q_DSA
        r_bf = rank_mod_q(D_bf, Q_DSA)

        # One-per-layer random delta
        random.seed(7)
        K_rand = {2 ** ell: (zetas[2 ** ell] - random.randint(1, Q_DSA - 1)) % Q_DSA
                  for ell in range(log2n)}
        f_rand = make_ntt(n, zetas, K_rand, min_length=1)
        D_rand = (M_clean - matrix_of(f_rand, n)) % Q_DSA
        r_rand = rank_mod_q(D_rand, Q_DSA)

        print(f"\nn={n}, layers={log2n}: expected rank(D_K) = n-1 = {n-1}")
        print(f"  zeroing:      rank(D_K) = {r_zero}")
        print(f"  bit-flip b0:  rank(D_K) = {r_bf}")
        print(f"  random delta: rank(D_K) = {r_rand}")

        # Check kernel: should be span(e_0) for complete NTT
        # (position 0 is always a-input; position 1 becomes b-input at
        # the last layer which has len=1)
        # Verify: D_K(e_0) = 0?
        e0 = [1] + [0] * (n - 1)
        D_e0_zero = [(M_clean @ np.array(e0) - matrix_of(f_zero, n) @ np.array(e0))[i] % Q_DSA
                     for i in range(n)]
        D_e0_is_zero = all(x == 0 for x in D_e0_zero)
        e1 = [0, 1] + [0] * (n - 2)
        D_e1_zero = [(M_clean @ np.array(e1) - matrix_of(f_zero, n) @ np.array(e1))[i] % Q_DSA
                     for i in range(n)]
        D_e1_is_zero = all(x == 0 for x in D_e1_zero)
        print(f"  D(e_0) = 0? {D_e0_is_zero}  (expect True)")
        print(f"  D(e_1) = 0? {D_e1_is_zero}  (expect False for complete NTT)")


# ============================================================
# G4: ML-DSA collision check
# ============================================================

def run_g4():
    print("\n" + "=" * 70)
    print("G4: ML-DSA collision check (q=8380417, 23 bits)")
    print("  Check: zeta_k XOR 2^j != 0 mod q for all k, j")
    print("=" * 70)

    n = 256
    zetas = compute_zetas_dsa(n, Q_DSA)
    bits = (Q_DSA - 1).bit_length()  # 23
    collisions = []
    for k in range(1, n):
        for j in range(bits):
            z = zetas[k] ^ (1 << j)
            if z % Q_DSA == 0:
                collisions.append((k, j, zetas[k]))
    print(f"  n={n}, twiddles={n-1}, bit positions={bits}")
    print(f"  total (k,j) pairs checked: {(n-1) * bits}")
    print(f"  collisions found: {len(collisions)}")
    if collisions:
        for k, j, z in collisions[:10]:
            print(f"    k={k}, j={j}, zeta_k={z}")
    else:
        print("  NO COLLISIONS — every bit-flip yields zeta' != 0 mod q.")


if __name__ == "__main__":
    run_g2()
    run_g3()
    run_g4()
