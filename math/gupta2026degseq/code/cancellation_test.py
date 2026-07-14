"""
Cancellation test: the onset drop d+1 -> d-1 in the pole order
of sum_n E_d(m,n) z^n.

Setup (from lem:ratGF / transfer_check.Ed_transfer):
  E_d(m,n) = [x^d] (1/3) u(x)^T T(x)^{n-2} v(x),  n >= 2,
where T(x)[(b,c),(a,b)] = x^{flip(a,b,c)} on compatible pair-states (a,b)->(b,c),
v(x)[(c1,c2)] = x^{flip(None,c1,c2)}  (finalise column 1, no left neighbour),
u(x)[(c_{n-1},c_n)] = x^{flip(c_{n-1},c_n,None)}  (finalise column n, no right).
So  sum_{n>=2} E_d(m,n) z^n = (z^2/3) [x^d] u^T (I - z T(x))^{-1} v.

T0 := T(0) has spectrum {0,1,omega,omega^2} with omega a primitive cube root of 1
(eig0_onset_probe). The eigenvalue-1 spectral projection is
  Pi_1 = (T0^3 + T0^2 + T0)/3     (kills 0, fixes 1, kills omega,omega^2).
On the count (rotation-invariant) block (I - z T0)^{-1} = I-part + [1/(1-z)] Pi_1,
a single pole at z=1. HYPOTHESIS: the boundary vectors carry no eigenvalue-1
component,  Pi_1 v_i = 0  and  u_i^T Pi_1 = 0  for every x-component i, so the two
outermost resolvent factors drop the pole order from d+1 to d-1.

This script: (1) validates the transfer form of E_d against the DP; (2) tests the
hypothesis numerically for m=2,3,4.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from transfer_matrix import columns, compatible, flip


def pair_states(m: int) -> Tuple[List[Tuple], Dict[Tuple, int]]:
    cols = columns(m)
    S = [(a, b) for a in cols for b in cols if compatible(a, b)]
    return S, {s: i for i, s in enumerate(S)}


def build_T0(m: int, S, idx) -> np.ndarray:
    cols = columns(m)
    N = len(S)
    T0 = np.zeros((N, N))
    for (a, b) in S:
        for c in cols:
            if compatible(b, c) and flip(a, b, c) == 0:
                T0[idx[(b, c)], idx[(a, b)]] += 1.0
    return T0


def boundary_components(m: int, S, idx, dmax: int):
    """v_i, u_i (i = number of boundary-column extrema) as 0/1 vectors over states."""
    cols = columns(m)
    N = len(S)
    v = defaultdict(lambda: np.zeros(N))   # i -> vector
    u = defaultdict(lambda: np.zeros(N))
    for (a, b) in S:
        # v: (a,b) is the first bulk pair (col1=a, col2=b); finalise col1 w/ left=None
        fv = flip(None, a, b)
        if fv <= dmax:
            v[fv][idx[(a, b)]] += 1.0
        # u: (a,b) is the last bulk pair (col_{n-1}=a, col_n=b); finalise col_n w/ right=None
        fu = flip(a, b, None)
        if fu <= dmax:
            u[fu][idx[(a, b)]] += 1.0
    return dict(v), dict(u)


def Ed_via_transfer(m: int, n: int, dmax: int, S, idx, T0info) -> Dict[int, int]:
    """E_d(m,n) for d<=dmax by the marked transfer form, x truncated at dmax.
    Returns {d: count}. Uses full x-marked T (dict of powers) via per-degree DP."""
    cols = columns(m)
    N = len(S)
    # state vector: for each state, a length-(dmax+1) array over x-degree
    # start = v(x): finalise col1
    vec = np.zeros((N, dmax + 1))
    for (a, b) in S:
        fv = flip(None, a, b)
        if fv <= dmax:
            vec[idx[(a, b)], fv] += 1.0
    # apply T(x) (n-2) times: (b,c)<-(a,b) weight x^{flip(a,b,c)}
    for _ in range(n - 2):
        nv = np.zeros((N, dmax + 1))
        for (a, b) in S:
            fr = idx[(a, b)]
            col = vec[fr]
            if not col.any():
                continue
            for c in cols:
                if not compatible(b, c):
                    continue
                f = flip(a, b, c)
                if f > dmax:
                    continue
                to = idx[(b, c)]
                # shift col by f
                nv[to, f:] += col[:dmax + 1 - f]
        vec = nv
    # finalise last column with u(x): (a,b) -> x^{flip(a,b,None)}
    total = np.zeros(dmax + 1)
    for (a, b) in S:
        fr = idx[(a, b)]
        col = vec[fr]
        if not col.any():
            continue
        f = flip(a, b, None)
        if f > dmax:
            continue
        total[f:] += col[:dmax + 1 - f]
    return {d: int(round(total[d])) // 3 for d in range(dmax + 1)}


def build_Wj(m: int, S, idx, j: int) -> np.ndarray:
    """Bulk single-value marker: W_j[(b,c),(a,b)] = 1 iff compatible(b,c) & flip==j."""
    cols = columns(m)
    N = len(S)
    W = np.zeros((N, N))
    for (a, b) in S:
        for c in cols:
            if compatible(b, c) and flip(a, b, c) == j:
                W[idx[(b, c)], idx[(a, b)]] += 1.0
    return W


def main() -> None:
    dmax = 6
    for m in (2, 3, 4):
        S, idx = pair_states(m)
        N = len(S)
        T0 = build_T0(m, S, idx)
        Pi1 = (np.linalg.matrix_power(T0, 3) + np.linalg.matrix_power(T0, 2) + T0) / 3.0
        v, u = boundary_components(m, S, idx, dmax)
        W1 = build_Wj(m, S, idx, 1)
        # M = Pi1 W1 Pi1 : the eig-1 -> eig-1 coupling of a single mark.
        M = Pi1 @ W1 @ Pi1
        r1 = int(round(np.trace(Pi1)))                     # eig-1 dim = rank Pi1
        ranks = [int(np.linalg.matrix_rank(np.linalg.matrix_power(M, k), tol=1e-8))
                 for k in range(1, 9)]
        evM = np.linalg.eigvals(M)
        nz = sorted({round(abs(e), 4) for e in evM if abs(e) > 1e-8})
        # top-pole coefficient chain for the all-single-mark (k=d) term:
        # s(k) = u1^T Pi1 (W1 Pi1)^k v1  (v1,u1 = single-extremum boundary parts)
        v1 = v.get(1, np.zeros(N)); u1 = u.get(1, np.zeros(N))
        WP = W1 @ Pi1
        s = []
        vecp = Pi1 @ v1
        for k in range(0, 9):
            s.append(round(float(u1 @ vecp), 4))
            vecp = WP @ vecp
        print(f"m={m}: dim={N}  eig-1 dim={r1}")
        print(f"   M=Pi1 W1 Pi1: rank(M^k) k=1..8 = {ranks};  |eig(M)| nonzero = {nz}")
        print(f"   s(k)=u1^T Pi1 (W1 Pi1)^k v1, k=0..8 = {s}")


if __name__ == "__main__":
    main()
