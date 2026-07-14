#!/usr/bin/env python3
"""Fully-numba transfer-matrix DP: setup + inner loop both JIT-compiled.

The prior numba attempt (fast_dp_nb.py) got 2.7x because Python setup was
91% of the total (19s of cell_ext calls at m=10). This version encodes columns
as an int8 (K, m) matrix + precomputed adjacency arrays, and JITs the triple
enumeration and boundary-extras enumeration.

Correctness: reproduces Ed_DP baseline at m in {3,4,5,6}, d_cap=10.
"""
from itertools import product
import time

import numba
import numpy as np


# -------------------------------------------------------------------- column setup
def cols_of_int(m):
    """Return proper 3-colorings of P_m as an (K, m) int8 array."""
    lst = [c for c in product(range(3), repeat=m)
           if all(c[i] != c[i + 1] for i in range(m - 1))]
    return np.array(lst, dtype=np.int8)


def build_adjacency(cols):
    """For each column a, list column indices b with a[i] != b[i] for all i."""
    K, m = cols.shape
    adj_lists = []
    for a in range(K):
        row = cols[a]
        # b is compatible if cols[b] differs in every position from row
        mask = np.all(cols != row[None, :], axis=1)
        adj_lists.append(np.where(mask)[0].astype(np.int32))
    adj_size = np.array([len(l) for l in adj_lists], dtype=np.int32)
    max_deg = adj_size.max()
    adj = np.full((K, max_deg), -1, dtype=np.int32)
    for a in range(K):
        adj[a, :adj_size[a]] = adj_lists[a]
    return adj, adj_size


# -------------------------------------------------------------------- jitted core
@numba.njit(cache=True, inline='always')
def _cext_interior(cols, b, a, c, m):
    """cell_ext for interior column b with both left=a and right=c present."""
    count = 0
    for i in range(m):
        first = cols[a, i]                      # left is always present
        if cols[c, i] != first:                 # right must match
            continue
        if i > 0 and cols[b, i - 1] != first:   # up (if present)
            continue
        if i < m - 1 and cols[b, i + 1] != first:  # down (if present)
            continue
        count += 1
    return count


@numba.njit(cache=True, inline='always')
def _cext_boundary(cols, b, other, m, other_is_left):
    """cell_ext for a boundary column b with only one horizontal neighbor.
    If other_is_left, only left=other is present (right absent).
    Else only right=other is present (left absent)."""
    count = 0
    for i in range(m):
        first = cols[other, i]                   # only one horizontal neighbor
        if i > 0 and cols[b, i - 1] != first:
            continue
        if i < m - 1 and cols[b, i + 1] != first:
            continue
        count += 1
    return count


@numba.njit(cache=True)
def _boundary_extras(cols, adj, adj_size, m):
    """Return ext_L[state], ext_R[state] where state = state_offset[a] + k for
    (a, b=adj[a,k]) valid pairs. State indices are cumulative over a."""
    K = cols.shape[0]
    state_offset = np.empty(K + 1, dtype=np.int32)
    state_offset[0] = 0
    for a in range(K):
        state_offset[a + 1] = state_offset[a] + adj_size[a]
    N = state_offset[K]
    ext_L = np.empty(N, dtype=np.int8)
    ext_R = np.empty(N, dtype=np.int8)
    for a in range(K):
        for k in range(adj_size[a]):
            b = adj[a, k]
            idx = state_offset[a] + k
            # first col a finalized as left boundary (right neighbor = b, no left neighbor)
            ext_L[idx] = _cext_boundary(cols, a, b, m, False)
            # last col b finalized as right boundary (left neighbor = a, no right)
            ext_R[idx] = _cext_boundary(cols, b, a, m, True)
    return state_offset, ext_L, ext_R


@numba.njit(cache=True)
def _enumerate_triples(cols, adj, adj_size, state_offset, d_cap, m):
    """Enumerate valid (a, b, c) triples with cell_ext(b, a, c) <= d_cap.
    Two-pass: count, then fill."""
    K = cols.shape[0]
    # Count
    n_triples = 0
    for a in range(K):
        for k in range(adj_size[a]):
            b = adj[a, k]
            for l in range(adj_size[b]):
                c = adj[b, l]
                eb = _cext_interior(cols, b, a, c, m)
                if eb <= d_cap:
                    n_triples += 1
    # Fill
    src = np.empty(n_triples, dtype=np.int32)
    dst = np.empty(n_triples, dtype=np.int32)
    shift = np.empty(n_triples, dtype=np.int8)
    idx = 0
    for a in range(K):
        off_a = state_offset[a]
        for k in range(adj_size[a]):
            b = adj[a, k]
            src_id = off_a + k
            off_b = state_offset[b]
            for l in range(adj_size[b]):
                c = adj[b, l]
                eb = _cext_interior(cols, b, a, c, m)
                if eb > d_cap:
                    continue
                src[idx] = src_id
                dst[idx] = off_b + l
                shift[idx] = eb
                idx += 1
    return src, dst, shift


@numba.njit(cache=True)
def _transition(hist, new_hist, src, dst, shift, D):
    n = src.shape[0]
    for i in range(n):
        s = src[i]
        d = dst[i]
        e = shift[i]
        for k in range(D - e):
            new_hist[d, k + e] += hist[s, k]


@numba.njit(cache=True)
def _close(hist, ext_R, D, d_cap):
    N = hist.shape[0]
    tot = np.zeros(D, dtype=np.int64)
    for i in range(N):
        eb = ext_R[i]
        if eb > d_cap:
            continue
        for k in range(D - eb):
            tot[k + eb] += hist[i, k]
    return tot


def Ed_DP_capped_nb2(m: int, nmax: int, d_cap: int):
    cols = cols_of_int(m)
    adj, adj_size = build_adjacency(cols)
    state_offset, ext_L, ext_R = _boundary_extras(cols, adj, adj_size, m)
    src, dst, shift = _enumerate_triples(cols, adj, adj_size, state_offset, d_cap, m)
    N = state_offset[-1]
    D = d_cap + 1

    hist = np.zeros((N, D), dtype=np.int64)
    for idx in range(N):
        e0 = ext_L[idx]
        if e0 <= d_cap:
            hist[idx, e0] += 1
    new_hist = np.zeros_like(hist)

    def close_to_dict(state):
        tot = _close(state, ext_R, D, d_cap)
        return {int(d): int(tot[d] // 3) for d in range(D) if tot[d]}

    out = {2: close_to_dict(hist)}
    for n in range(3, nmax + 1):
        new_hist.fill(0)
        _transition(hist, new_hist, src, dst, shift, D)
        hist, new_hist = new_hist, hist
        out[n] = close_to_dict(hist)
    return out


if __name__ == "__main__":
    from column_dp import Ed_DP as Ed_DP_ref
    from fast_dp import Ed_DP_capped

    # warm up JIT
    _ = Ed_DP_capped_nb2(3, 5, 5)

    print("=== correctness (numba2 vs Ed_DP baseline) ===")
    for m in (3, 4, 5, 6):
        nmax = m + 3
        d_cap = 10
        ref = Ed_DP_ref(m, nmax)
        nb = Ed_DP_capped_nb2(m, nmax, d_cap)
        ok = all(ref[n].get(d, 0) == nb[n].get(d, 0)
                 for n in range(2, nmax + 1) for d in range(d_cap + 1))
        print(f"  m={m}: numba2={ok}")

    print("\n=== timing: pure-Python fast vs numba2 ===")
    for m, nmax, d_cap in [(8, 11, 8), (10, 10, 8), (11, 11, 9), (12, 11, 10), (13, 11, 10)]:
        if m <= 10:
            t = time.time(); Ed_DP_capped(m, nmax, d_cap); t_fast = time.time() - t
        else:
            t_fast = float('nan')
        t = time.time(); Ed_DP_capped_nb2(m, nmax, d_cap); t_nb = time.time() - t
        print(f"  m={m:2d} nmax={nmax:2d} d_cap={d_cap:2d}:  "
              f"fast={t_fast:7.2f}s  numba2={t_nb:7.2f}s"
              f"  speedup={t_fast/max(t_nb, 1e-9):.1f}x" if not np.isnan(t_fast)
              else f"  m={m:2d} nmax={nmax:2d} d_cap={d_cap:2d}:  numba2={t_nb:7.2f}s")
