#!/usr/bin/env python3
"""Delta-parameterized barrier LP: extends §5 (which is cubic-only) to any Delta.

For Delta-regular graphs, the natural analog of the cubic potential Phi = 4*beta + P (§4)
is Phi_Delta = (Delta+1)*beta + P_1  (mult-0 weight Delta+1, mult-1 weight 1, others 0).
Flipping an M-edge changes Phi_Delta by
    Delta*p + q - (Delta*a + b),
where a = #mult-0 in up, b = #mult-1 in up, p = #mult-1 in cp, q = #mult-2 in cp.
Local-min filter: Delta*p + q >= Delta*a + b.

Edge-types: (cp, up) where cp is a multiset of mults from {1..Delta} of size <=Delta-1
            and up is a multiset of mults from {0..Delta-1} of size <=Delta-1.

Half-edge consistency (per mult j): sum #j(cp)*x * (Delta-j) = sum #j(up)*x * j.
  (mult-j vertex has j chosen incidences and Delta-j unchosen.)

Objective: max sum #0(up)*x / Delta = beta.
  * beta = 0 => LINEAR DISCHARGING CLOSES the Delta-regular case (proof).
  * beta > 0 => barrier extends; §5's story generalizes.
"""
import itertools, sys
import numpy as np
from scipy.optimize import linprog
import argparse

def enum_types(Delta):
    """Return list of (cp, up) with cp mults in 1..Delta size 0..Delta-1, up mults 0..Delta-1 size 0..Delta-1."""
    def profs(vals, maxsize):
        out = [()]
        for s in range(1, maxsize + 1):
            out += list(itertools.combinations_with_replacement(vals, s))
        return out
    CP = profs(tuple(range(1, Delta + 1)), Delta - 1)
    UP = profs(tuple(range(0, Delta)), Delta - 1)
    return CP, UP

def cnt(p, v): return sum(1 for x in p if x == v)

def solve(Delta):
    CP, UP = enum_types(Delta)
    types = []
    for cp in CP:
        for up in UP:
            a = cnt(up, 0); b = cnt(up, 1); p = cnt(cp, 1); q = cnt(cp, 2)
            L = Delta * p + q
            G = Delta * a + b
            if L >= G:
                types.append((cp, up))
    T = len(types)
    print(f"Delta={Delta}: edge-types (L>=G filter): {T}", flush=True)
    if T == 0:
        print(f"  no type survives => LP infeasible => Delta-regular CLOSED!", flush=True)
        return

    # objective: max sum #0(up)*x
    c = np.array([-cnt(up, 0) for (cp, up) in types], dtype=float)

    # half-edge consistency: for j=1..Delta-1,
    #   sum #j(cp)*x * (Delta-j) = sum #j(up)*x * j
    #   => (Delta-j)*sum #j(cp)*x - j*sum #j(up)*x = 0
    # (for j=0: only up-side, no constraint; for j=Delta: only cp-side, no constraint)
    rows = []
    for j in range(1, Delta):
        row = np.array([(Delta - j) * cnt(cp, j) - j * cnt(up, j) for (cp, up) in types], dtype=float)
        if np.any(row != 0):
            rows.append(row)
    # normalization: sum x = 1
    rows.append(np.ones(T))
    b_eq = np.zeros(len(rows)); b_eq[-1] = 1.0
    A_eq = np.vstack(rows)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * T, method="highs")
    if not res.success:
        print(f"  LP {res.message}", flush=True)
        if "infeasible" in res.message.lower():
            print(f"  >>> Delta={Delta} LP INFEASIBLE => LINEAR DISCHARGING PROVES CONJECTURE for Delta={Delta} !!", flush=True)
        return
    obj = -res.fun
    beta = obj / Delta
    print(f"  max sum #0(up)*x = {obj:.6f}   =>   beta = {beta:.6f}", flush=True)
    if beta < 1e-8:
        print(f"  >>> beta = 0 => LINEAR DISCHARGING PROVES CONJECTURE for Delta={Delta} !!", flush=True)
    else:
        # Show support
        nonzero = [(i, res.x[i]) for i in range(T) if res.x[i] > 1e-6]
        print(f"  witness support ({len(nonzero)} types, top 5):")
        for i, xi in sorted(nonzero, key=lambda kv: -kv[1])[:5]:
            print(f"    x={xi:.4f}  cp={types[i][0]} up={types[i][1]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("deltas", type=int, nargs="+")
    args = ap.parse_args()
    for D in args.deltas:
        solve(D)
        print()
