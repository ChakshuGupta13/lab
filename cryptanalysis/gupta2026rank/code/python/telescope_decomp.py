"""Telescope with correct composition order."""
import numpy as np
q = 3329

def build_ntt_layers(n):
    num_layers = (n // 2 - 1).bit_length()
    zetas = [pow(17, i * (128 // (n // 2)), q) for i in range(n // 2)]
    layers = []
    k_idx = 1
    for ell in range(num_layers):
        M = np.eye(n, dtype=int)
        num_groups = 1 << ell
        group_size = n >> ell
        half = group_size >> 1
        for g in range(num_groups):
            start = g * group_size
            z = zetas[k_idx]; k_idx += 1
            for j in range(half):
                a, b = start + j, start + j + half
                M[a, a] = 1; M[a, b] = z
                M[b, a] = 1; M[b, b] = (-z) % q
        layers.append(M % q)
    return layers

def mm(A, B): return A @ B % q

def rank_Fq(M):
    M = M.copy() % q; rows, cols = M.shape; r = 0
    for col in range(cols):
        piv = None
        for row in range(r, rows):
            if M[row, col] % q: piv = row; break
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, col]), q - 2, q)
        M[r] = M[r] * inv % q
        for row in range(rows):
            if row != r and M[row, col] % q:
                M[row] = (M[row] - M[row, col] * M[r]) % q
        r += 1
    return r

for n in [8, 16, 32]:
    num_layers = (n // 2 - 1).bit_length()
    layers = build_ntt_layers(n)
    I = np.eye(n, dtype=int)

    # Fault first group at each layer
    faulted = []
    for ell in range(num_layers):
        L = layers[ell].copy()
        group_size = n >> ell; half = group_size >> 1
        for j in range(half):
            a, b = j, j + half
            L[a, :] = 0; L[b, :] = 0
            L[a, a] = 1; L[b, a] = 1
        faulted.append(L % q)

    # NTT = L_{num-1} @ ... @ L_0
    NTT = I.copy()
    for L in layers: NTT = mm(L, NTT)
    NTT_K = I.copy()
    for L in faulted: NTT_K = mm(L, NTT_K)
    D_K = (NTT - NTT_K) % q

    # Telescope: T_ell = A_ell @ P_ell @ F_ell
    # A_ell = L_{num-1} @ ... @ L_{ell+1}  (unfaulted above)
    # P_ell = L_ell - L_ell^*
    # F_ell = L_{ell-1}^* @ ... @ L_0^*    (faulted below)
    S = np.zeros((n, n), dtype=int)
    for ell in range(num_layers):
        A = I.copy()
        for i in range(ell + 1, num_layers):
            A = mm(layers[i], A)
        P = (layers[ell] - faulted[ell]) % q
        F = I.copy()
        for i in range(ell):
            F = mm(faulted[i], F)
        term = mm(A, mm(P, F))
        r = rank_Fq(term)
        len_ell = n >> (ell + 1)
        S = (S + term) % q
        print(f"  n={n} layer {ell}: rank(A·P·F) = {r}, len = {len_ell}, match: {r == len_ell}")

    match = np.all((S - D_K) % q == 0)
    rank_DK = rank_Fq(D_K)
    print(f"  n={n}: telescope verified={match}, rank D_K={rank_DK}, n-2={n-2}")
    print()
