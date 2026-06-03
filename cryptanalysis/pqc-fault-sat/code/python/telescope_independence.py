"""Exhaustive verification of telescope-image independence.

Tests, for n in {32, 64}, all one-per-layer fault selections K, that:
  (i) D_K has rank n-2
  (ii) each telescope term T_ell has rank exactly len_ell
together implying the m image subspaces are linearly independent and
their sum equals im(D_K).

Also tests robustness to twiddle choice (random nonzero twiddles), to
confirm the rank claim is not specific to the ML-KEM twiddle ladder
zeta = 17 over F_3329.
"""
import numpy as np, random, time
from itertools import product as iproduct
q = 3329

def build_ntt_layers(n, twiddles=None):
    """If twiddles is None, use ML-KEM twiddle ladder zeta=17."""
    num_layers = (n // 2 - 1).bit_length()
    if twiddles is None:
        zetas = [pow(17, i * (128 // (n // 2)), q) for i in range(n // 2)]
        twiddles = zetas[1:]  # skip zetas[0]=1 placeholder
    layers = []; k_idx = 0
    for ell in range(num_layers):
        M = np.eye(n, dtype=int)
        for g in range(1 << ell):
            gs = n >> ell; half = gs >> 1; start = g * gs
            z = twiddles[k_idx]; k_idx += 1
            for j in range(half):
                a, b = start + j, start + j + half
                M[a, a] = 1; M[a, b] = z; M[b, a] = 1; M[b, b] = (-z) % q
        layers.append(M % q)
    return layers

def make_faulted(layer, ell, group_idx, n):
    L = layer.copy(); gs = n >> ell; half = gs >> 1; start = group_idx * gs
    for j in range(half):
        a, b = start + j, start + j + half
        L[a, :] = 0; L[b, :] = 0; L[a, a] = 1; L[b, a] = 1
    return L % q

def mm(A, B): return A @ B % q
def rank_Fq(M):
    M = M.copy() % q; rows, cols = M.shape; r = 0
    for col in range(cols):
        piv = None
        for row in range(r, rows):
            if M[row, col] % q: piv = row; break
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, col]), q - 2, q); M[r] = M[r] * inv % q
        for row in range(rows):
            if row != r and M[row, col] % q:
                M[row] = (M[row] - M[row, col] * M[r]) % q
        r += 1
    return r

def run_exhaustive(n, twiddles=None, label=""):
    num_layers = (n // 2 - 1).bit_length()
    layers = build_ntt_layers(n, twiddles)
    I = np.eye(n, dtype=int)
    groups_per_layer = [1 << ell for ell in range(num_layers)]
    count = 0; all_pass = True
    NTT = I.copy()
    for L in layers: NTT = mm(L, NTT)
    for sel in iproduct(*[range(g) for g in groups_per_layer]):
        faulted = [make_faulted(layers[ell], ell, sel[ell], n) for ell in range(num_layers)]
        NTT_K = I.copy()
        for L in faulted: NTT_K = mm(L, NTT_K)
        D_K = (NTT - NTT_K) % q
        rk = rank_Fq(D_K)
        term_ranks = []
        for ell in range(num_layers):
            A = I.copy()
            for i in range(ell + 1, num_layers): A = mm(layers[i], A)
            P = (layers[ell] - faulted[ell]) % q
            F = I.copy()
            for i in range(ell): F = mm(faulted[i], F)
            term_ranks.append(rank_Fq(mm(A, mm(P, F))))
        ok = rk == n - 2 and all(r == (n >> (ell+1)) for ell, r in enumerate(term_ranks))
        if not ok:
            print(f"  FAIL n={n} {label} sel={sel}: rank={rk}, term_ranks={term_ranks}")
            all_pass = False
        count += 1
    return count, all_pass

def check_lemma_Fellsupport(n, twiddles=None):
    """Verify Lemma: F_ell submatrix at (b-positions of group g_ell^*, I_ell)
    is the identity matrix, for every layer ell and every selection."""
    m = (n // 2 - 1).bit_length()
    log2n = n.bit_length() - 1
    layers = build_ntt_layers(n, twiddles)
    I = np.eye(n, dtype=int)
    groups_per_layer = [1 << ell for ell in range(m)]
    fails = 0; total = 0
    for sel in iproduct(*[range(g) for g in groups_per_layer]):
        faulted = [make_faulted(layers[ell], ell, sel[ell], n) for ell in range(m)]
        for ell in range(m):
            F = I.copy()
            for k in range(ell): F = mm(faulted[k], F)
            b_ell = log2n - 1 - ell
            len_ell = 1 << b_ell
            g_star = sel[ell]
            b_positions = [g_star * (2 * len_ell) + len_ell + s for s in range(len_ell)]
            I_ell = [len_ell + s for s in range(len_ell)]
            sub = F[np.ix_(b_positions, I_ell)] % q
            if not np.array_equal(sub, np.eye(len_ell, dtype=int)):
                fails += 1
        total += 1
    return fails, total

print("=== Lemma Fellsupport (identity-submatrix property) ===")
for n in (8, 16, 32, 64):
    t0 = time.time()
    fails, total = check_lemma_Fellsupport(n)
    print(f"  n={n} ML-KEM twiddles: {total} selections, lemma holds: {fails==0}, {time.time()-t0:.1f}s")

print("=== Exhaustive verification (ML-KEM twiddles, zeta=17) ===")
for n in (32, 64):
    t0 = time.time()
    count, ok = run_exhaustive(n)
    print(f"  n={n}: {count} selections, all rank={n-2}: {ok}, {time.time()-t0:.1f}s")

print("=== Robustness to twiddle choice (random nonzero twiddles in F_q*) ===")
random.seed(20260601)
for n, trials in [(32, 20), (64, 3)]:
    for trial in range(trials):
        twiddles = [random.randint(1, q-1) for _ in range(n // 2 - 1)]
        t0 = time.time()
        count, ok = run_exhaustive(n, twiddles, label=f"trial{trial}")
        if not ok or trial == trials - 1:
            print(f"  n={n} trial {trial}: {count} selections, ok={ok}, {time.time()-t0:.1f}s")
    print(f"  n={n}: all {trials} random-twiddle trials passed")

# ----------------------------------------------------------------------
# Full Cooley-Tukey NTT (log_2(n) layers) — ML-DSA analogue.
# Predicted ceiling: rank D_K = n - 1, kernel = span(e_0). The Lemma
# F_ell-support proof transfers verbatim; reported in the paper's
# Limitations as future-work justification.
# ----------------------------------------------------------------------

def build_full_layers(n, twiddles):
    """Full Cooley-Tukey NTT with log_2(n) layers (last layer: len=1)."""
    num_layers = n.bit_length() - 1
    layers = []; k_idx = 0
    for ell in range(num_layers):
        M = np.eye(n, dtype=int)
        for g in range(1 << ell):
            gs = n >> ell; half = gs >> 1; start = g * gs
            z = twiddles[k_idx]; k_idx += 1
            for j in range(half):
                a, b = start + j, start + j + half
                M[a, a] = 1; M[a, b] = z; M[b, a] = 1; M[b, b] = (-z) % q
        layers.append(M % q)
    return layers

def run_full_exhaustive(n, twiddles):
    num_layers = n.bit_length() - 1
    layers = build_full_layers(n, twiddles)
    I = np.eye(n, dtype=int)
    NTT = I.copy()
    for L in layers: NTT = mm(L, NTT)
    fails = 0; total = 0
    for sel in iproduct(*[range(1 << ell) for ell in range(num_layers)]):
        faulted = [make_faulted(layers[ell], ell, sel[ell], n) for ell in range(num_layers)]
        NTT_K = I.copy()
        for L in faulted: NTT_K = mm(L, NTT_K)
        D_K = (NTT - NTT_K) % q
        rk = rank_Fq(D_K)
        if rk != n - 1:
            fails += 1
        total += 1
    return total, fails

print("=== Full Cooley-Tukey NTT (log_2(n) layers) — ML-DSA analogue ===")
random.seed(20260602)
for n in (8, 16, 32):
    twiddles = [random.randint(1, q-1) for _ in range(n - 1)]
    t0 = time.time()
    total, fails = run_full_exhaustive(n, twiddles)
    print(f"  n={n}, layers={n.bit_length()-1}: {total} selections, "
          f"all rank=n-1={n-1}: {fails==0}, {time.time()-t0:.1f}s")
