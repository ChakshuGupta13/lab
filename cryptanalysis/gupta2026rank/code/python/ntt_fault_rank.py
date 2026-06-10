#!/usr/bin/env python3
"""
ntt_fault_rank.py — Rank analysis of twiddle-zeroing faults in ML-KEM NTT.

For each fault position k (twiddle zetas[k] zeroed), computes:
- Which NTT layer and group the fault is in
- The rank of the induced leakage matrix over GF(q)
  (= number of secret-key coefficients leaked from one chosen-ciphertext query)

Then checks multi-fault (K=2) combinations for full-rank recovery.

Usage: python ntt_fault_rank.py [16|256]
"""
import sys
from math import gcd

Q = 3329  # ML-KEM modulus


def modinv(a, q=Q):
    """Modular inverse via Fermat's little theorem (q prime)."""
    return pow(a % q, q - 2, q)


def compute_zetas(n, q=Q):
    """Compute NTT twiddle factors in plain form (non-Montgomery).
    Matches ML-KEM reference: Cooley-Tukey, bit-reversed order.
    Returns list of n//2 values; NTT uses indices 1..n//2-1.
    """
    root = pow(17, 512 // (2 * n), q)  # primitive 2n-th root of unity
    half = n // 2
    log_half = half.bit_length() - 1

    def bitrev(x, bits):
        r = 0
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    return [pow(root, bitrev(i, log_half), q) for i in range(half)]


def ntt(r_in, n, zetas, q=Q, fault_k=-1, fault_set=None):
    """Forward NTT (Cooley-Tukey, len from n/2 down to 2).
    fault_k >= 1: zero the single twiddle zetas[fault_k].
    fault_set (set or sequence): zero ALL twiddles whose index is in this set
        (simultaneous multi-fault model — see paper §6 'Simultaneous vs separate').
    If both are given, fault_set takes precedence.
    """
    r = [x % q for x in r_in]
    k = 1
    length = n // 2
    fset = set(fault_set) if fault_set is not None else None
    while length >= 2:
        start = 0
        while start < n:
            if fset is not None:
                z = 0 if k in fset else zetas[k]
            else:
                z = 0 if k == fault_k else zetas[k]
            for j in range(start, start + length):
                t = (z * r[j + length]) % q
                r[j + length] = (r[j] - t) % q
                r[j] = (r[j] + t) % q
            k += 1
            start += 2 * length
        length //= 2
    return r


def invntt(r_in, n, zetas, q=Q):
    """Inverse NTT (Gentleman-Sande, len from 2 up to n/2)."""
    r = [x % q for x in r_in]
    half = n // 2
    k = half - 1
    length = 2
    while length <= n // 2:
        start = 0
        while start < n:
            z = zetas[k]
            k -= 1
            for j in range(start, start + length):
                t = r[j]
                r[j] = (t + r[j + length]) % q
                r[j + length] = (r[j + length] - t) % q
                r[j + length] = (z * r[j + length]) % q
            start += 2 * length
        length *= 2
    # Scale by (n/2)^{-1} mod q
    scale = modinv(n // 2, q)
    return [(x * scale) % q for x in r]


def fault_info(k_fault, n):
    """Return (layer, butterfly_len, group_index) for fault at k_fault."""
    layer = 0
    k_count = 0
    length = n // 2
    while length >= 2:
        groups = n // (2 * length)
        if k_fault <= k_count + groups:
            return layer, length, k_fault - k_count - 1
        k_count += groups
        layer += 1
        length //= 2
    return -1, -1, -1


def matrix_rank_gfq(rows, ncols, q=Q):
    """Rank of a matrix over GF(q) via Gaussian elimination.
    rows: list of lists (each row has ncols entries).
    """
    A = [[c % q for c in row] for row in rows]
    m = len(A)
    pivots = 0
    for col in range(ncols):
        pivot_row = -1
        for row in range(pivots, m):
            if A[row][col] % q != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        A[pivots], A[pivot_row] = A[pivot_row], A[pivots]
        inv = modinv(A[pivots][col], q)
        A[pivots] = [(c * inv) % q for c in A[pivots]]
        for row in range(m):
            if row != pivots and A[row][col] % q != 0:
                f = A[row][col]
                A[row] = [(A[row][j] - f * A[pivots][j]) % q for j in range(ncols)]
        pivots += 1
    return pivots


def compute_diff_matrix(n, zetas, k_fault, q=Q):
    """Compute the n×n difference matrix D where D[i][j] = (NTT(e_j)[i] - NTT_fault(e_j)[i]) mod q.
    Each column j is the NTT-domain difference when input is unit vector e_j.
    """
    D = [[0] * n for _ in range(n)]
    for j in range(n):
        e_j = [0] * n
        e_j[j] = 1
        c = ntt(e_j, n, zetas, q)
        f = ntt(e_j, n, zetas, q, fault_k=k_fault)
        for i in range(n):
            D[i][j] = (c[i] - f[i]) % q
    return D


def check_one_per_layer_multifault(n, zetas, q=Q, num_selections=6, seed=20260606):
    """One fault per layer (incomplete NTT): verify rank = n-2 and that the
    kernel includes span(e_0, e_1) (columns 0,1 of D_K are zero, nullity 2),
    under both twiddle-zeroing and general perturbation (zeta' = zeta + 1).
    Reproduces the paper's n=256 'six representative selections' claim.
    """
    import random as _random
    rng = _random.Random(seed)
    by_layer = {}
    for k in range(1, n // 2):
        layer, _, _ = fault_info(k, n)
        by_layer.setdefault(layer, []).append(k)
    layers = sorted(by_layer)
    selections = [
        [by_layer[l][0] for l in layers],                 # all-first
        [by_layer[l][-1] for l in layers],                # all-last
        [by_layer[l][len(by_layer[l]) // 2] for l in layers],  # all-mid
    ]
    while len(selections) < num_selections:
        selections.append([rng.choice(by_layer[l]) for l in layers])

    print(f"\n{'='*55}")
    print(f"One-per-layer multi-fault: n={n} ({len(selections)} selections)")
    print(f"Expect rank={n-2}, kernel=span(e_0,e_1), nullity 2")
    print(f"{'='*55}")
    all_ok = True
    for sel in selections:
        fset = set(sel)
        # general-perturbation table: zeta' = zeta + 1 (delta = -1 != 0 mod q)
        zetas_pert = list(zetas)
        for k in sel:
            zetas_pert[k] = (zetas[k] + 1) % q
        for model in ("zeroing", "general"):
            D = [[0] * n for _ in range(n)]
            for j in range(n):
                e_j = [0] * n
                e_j[j] = 1
                c = ntt(e_j, n, zetas, q)
                if model == "zeroing":
                    f = ntt(e_j, n, zetas, q, fault_set=fset)
                else:
                    f = ntt(e_j, n, zetas_pert, q)
                for i in range(n):
                    D[i][j] = (c[i] - f[i]) % q
            rank = matrix_rank_gfq(D, n)
            e0 = all(D[i][0] % q == 0 for i in range(n))
            e1 = all(D[i][1] % q == 0 for i in range(n))
            ok = rank == n - 2 and e0 and e1
            all_ok &= ok
            print(f"  sel={sorted(sel)} [{model:>7}]: rank={rank} (exp {n-2}), "
                  f"e0,e1 in ker={e0 and e1}, nullity={n-rank}  "
                  f"{'OK' if ok else 'FAIL'}")
    print(f"  -> {'ALL OK' if all_ok else 'FAILURE'}")
    return all_ok


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    assert n & (n - 1) == 0 and 4 <= n <= 256, "n must be power of 2 in [4,256]"

    zetas = compute_zetas(n)
    num_faults = n // 2 - 1
    num_layers = (n // 2).bit_length() - 1  # log2(n) - 1

    # --- Roundtrip verification ---
    import random
    random.seed(42)
    test = [random.randint(0, Q - 1) for _ in range(n)]
    assert invntt(ntt(test, n, zetas), n, zetas) == test, "NTT roundtrip FAILED"
    print(f"NTT roundtrip: PASS (n={n})")

    # --- Single-fault rank analysis ---
    print(f"\n{'='*55}")
    print(f"Single-fault rank analysis: n={n}, q={Q}")
    print(f"NTT layers={num_layers}, fault positions k=1..{num_faults}")
    print(f"{'='*55}")
    print(f"{'k':>4} {'layer':>5} {'len':>5} {'rank':>6} {'frac':>8}")
    print("-" * 35)

    ranks = {}  # k -> rank
    layer_ranks = {}  # layer -> [ranks]

    for k_fault in range(1, num_faults + 1):
        D = compute_diff_matrix(n, zetas, k_fault)
        rank = matrix_rank_gfq(D, n)
        layer, length, group = fault_info(k_fault, n)

        ranks[k_fault] = rank
        layer_ranks.setdefault(layer, []).append(rank)

        print(f"{k_fault:4d} {layer:5d} {length:5d} {rank:6d} {rank/n:8.2f}")

    # --- Summary per layer ---
    print(f"\n--- Layer summary ---")
    for layer in sorted(layer_ranks):
        r = layer_ranks[layer]
        length = n // (2 ** (layer + 1))
        print(f"  Layer {layer} (len={length:3d}): {len(r):3d} positions, "
              f"rank = {min(r)}-{max(r)} (mean {sum(r)/len(r):.1f})")

    # --- Multi-fault (K=2) rank analysis ---
    if n <= 64:  # Only for small n (otherwise too many pairs)
        print(f"\n{'='*55}")
        print(f"Multi-fault (K=2) rank analysis")
        print(f"{'='*55}")

        # Pick one representative per layer
        layer_reps = {}
        for k in range(1, num_faults + 1):
            layer, _, _ = fault_info(k, n)
            if layer not in layer_reps:
                layer_reps[layer] = k

        best_pair = None
        best_rank = 0

        for l1, k1 in sorted(layer_reps.items()):
            for l2, k2 in sorted(layer_reps.items()):
                if l2 <= l1:
                    continue
                D1 = compute_diff_matrix(n, zetas, k1)
                D2 = compute_diff_matrix(n, zetas, k2)
                # Stack: 2n rows × n cols
                combined = D1 + D2
                rank = matrix_rank_gfq(combined, n)

                if rank > best_rank:
                    best_rank = rank
                    best_pair = (k1, l1, k2, l2)

                print(f"  k1={k1}(L{l1}) + k2={k2}(L{l2}): rank={rank}/{n}")

        if best_pair:
            k1, l1, k2, l2 = best_pair
            print(f"\nBest pair: k={k1}(L{l1}) + k={k2}(L{l2}) → rank={best_rank}/{n}")
            if best_rank == n:
                print("  → FULL KEY RECOVERY possible with 2 faults!")
            else:
                print(f"  → Partial recovery ({best_rank}/{n} coefficients)")
    else:
        print(f"\n(Skipping K=2 exhaustive analysis for n={n}; use targeted pairs)")
        # For n=256, test specific layer pairs
        print(f"\n--- Targeted K=2 pairs (one per layer) ---")
        layer_reps = {}
        for k in range(1, num_faults + 1):
            layer, _, _ = fault_info(k, n)
            if layer not in layer_reps:
                layer_reps[layer] = k

        for l1, k1 in sorted(layer_reps.items()):
            D1 = compute_diff_matrix(n, zetas, k1)
            for l2, k2 in sorted(layer_reps.items()):
                if l2 <= l1:
                    continue
                D2 = compute_diff_matrix(n, zetas, k2)
                combined = D1 + D2
                rank = matrix_rank_gfq(combined, n)
                print(f"  k1={k1}(L{l1}) + k2={k2}(L{l2}): rank={rank}/{n}")

        # One-per-layer multi-fault (paper Theorem 2 / section 6):
        # six representative selections -> rank n-2, kernel span(e_0, e_1).
        check_one_per_layer_multifault(n, zetas)


if __name__ == '__main__':
    main()
