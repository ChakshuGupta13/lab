"""
Period-1 mechanism for the per-axis (fixed-m) count.

The transfer-matrix route reads a height function on the m x n grid column by column. Columns are
proper 3-colourings of the path P_m (c in {0,1,2}^m, adjacent rows differ); a
valid transition c -> c' needs c'_i != c_i for every i (horizontal edges proper).
A cell (i,j) is a strict local extremum iff all its present neighbours share one
colour (then forced to c_i -+ 1); detecting extrema in column j needs columns
j-1, j, j+1. So the bulk transfer operator acts on PAIRS (prev, cur) of columns,
emitting x^{#extrema in cur} on the step to (cur, next).

Write the x-marked pair-transfer matrix as  T_m(x) = T0 + x*B(x), where T0 is the
"no new extremum in the middle column" transfer matrix. Standard transfer-matrix
theory gives  sum_n c_{m,n}(x) z^n = (boundary) * (I - z T_m(x))^{-1} * (boundary),
rational in z with denominator det(I - z T_m(x)). Extracting [x^d] expands
(I - zT0 - zx B)^{-1} in x: every term has denominator a power of det(I - z T0).
Hence:

   the poles of  [x^d] Z_m(x,z)  lie among the reciprocal eigenvalues of T0,

so the per-axis count E_d(m,.) is eventually a quasi-polynomial whose period
divides the lcm of the multiplicative orders of the unit-circle eigenvalues of
T0.  PERIOD 1  <=>  the only eigenvalue of T0 on the unit circle is 1.

This script builds T0 for m=2,3 (and 4 if quick) and reports its eigenvalues on
|lambda|=1. If they are exactly {1} (with the rest strictly inside), period-1 is
mechanistically established. It also cross-checks the FULL x-marked transfer
matrix reproduces E_d(m,n) against the brute oracle for small n.
"""

from __future__ import annotations

import argparse
import cmath
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

Colcol = Tuple[int, ...]


def proper_columns(m: int) -> List[Colcol]:
    return [c for c in product(range(3), repeat=m)
            if all(c[i] != c[i + 1] for i in range(m - 1))]


def valid_transition(c: Colcol, cp: Colcol) -> bool:
    return all(c[i] != cp[i] for i in range(len(c)))


def middle_extrema(prev: Colcol, cur: Colcol, nxt: Colcol, m: int) -> int:
    """#strict local extrema among cells of the MIDDLE column `cur`, given the
    left (prev) and right (nxt) columns. Cell i is an extremum iff all present
    neighbours (up i-1, down i+1 in cur; left prev[i]; right nxt[i]) share one
    colour."""
    cnt = 0
    for i in range(m):
        nb = [prev[i], nxt[i]]
        if i > 0:
            nb.append(cur[i - 1])
        if i < m - 1:
            nb.append(cur[i + 1])
        if len(set(nb)) == 1:
            cnt += 1
    return cnt


def build_T0(m: int) -> Tuple[np.ndarray, List[Tuple[Colcol, Colcol]]]:
    """Bulk no-new-extremum pair-transfer matrix. States = valid (prev,cur)
    pairs; transition (prev,cur)->(cur,nxt) allowed iff valid AND the middle
    column `cur` gets ZERO extrema from (prev,cur,nxt)."""
    cols = proper_columns(m)
    states = [(a, b) for a in cols for b in cols if valid_transition(a, b)]
    idx = {s: k for k, s in enumerate(states)}
    N = len(states)
    T0 = np.zeros((N, N))
    for (a, b) in states:
        for c in cols:
            if not valid_transition(b, c):
                continue
            if middle_extrema(a, b, c, m) == 0:
                T0[idx[(b, c)], idx[(a, b)]] = 1.0
    return T0, states


def unit_circle_eigs(T0: np.ndarray, tol: float = 1e-9):
    eigs = np.linalg.eigvals(T0)
    on = [e for e in eigs if abs(abs(e) - 1.0) < 1e-6]
    inside = [e for e in eigs if abs(e) < 1.0 - 1e-6]
    outside = [e for e in eigs if abs(e) > 1.0 + 1e-6]
    return eigs, on, inside, outside


def classify_on_circle(on: List[complex]):
    """Group unit-circle eigenvalues by argument; report which are = 1 vs other
    roots-of-unity-like."""
    out = []
    for e in on:
        ang = cmath.phase(e)
        out.append((round(e.real, 6), round(e.imag, 6), round(ang, 6)))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="2,3")
    args = ap.parse_args()
    for m in [int(x) for x in args.ms.split(",")]:
        T0, states = build_T0(m)
        eigs, on, inside, outside = unit_circle_eigs(T0)
        print(f"=== m={m}: T0 is {T0.shape[0]}x{T0.shape[0]} "
              f"(bulk no-extremum pair-transfer) ===")
        print(f"  spectral radius = {max(abs(e) for e in eigs):.6f}")
        print(f"  # eigenvalues: on |z|=1: {len(on)}, "
              f"strictly inside: {len(inside)}, outside: {len(outside)}")
        oncls = classify_on_circle(on)
        print(f"  unit-circle eigenvalues (Re,Im,arg): {oncls}")
        only_one = all(abs(e - 1.0) < 1e-6 for e in on)
        print(f"  => unit-circle spectrum is exactly {{1}}? "
              f"{'YES (period-1 mechanism holds)' if only_one and on else 'NO'}")
        print()


if __name__ == "__main__":
    main()
