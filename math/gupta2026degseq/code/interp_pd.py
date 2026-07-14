#!/usr/bin/env python3
"""Exact interpolation of p_d(m,n) from transfer-matrix E_d samples.

p_d is symmetric, total degree D=d-2, valid on the high region {m,n>=d-1}.
By m<->n symmetry a triangular unisolvent node set suffices:
    nodes = {(d-1+a, d-1+b) : 0<=a<=b, a+b<=D}
and every node has first coord <= second, so we only ever run full column
sweeps Ed_series(m, .) for m = d-1 .. floor((3d-4)/2), reading larger-n
values off the small-m slices.  (Unisolvent: node count == #symmetric coeffs,
collocation determinant nonzero.)

Validation mode (default): interpolate d=4..7 and diff against the known
polynomials.  Max m for d=7 is 8 -> cheap (~30s total).  The expensive
d=8 run (max m=10) is gated behind --dmax 8.
"""
import argparse
import sys
from fractions import Fraction

import sympy as sp

sys.path.insert(0, ".")
from fast_dp import Ed_DP_capped  # capped transfer-matrix DP; ~10x faster

m, n = sp.symbols("m n")

# ---- known polynomials for validation --------------------------------------
R = sp.Rational
KNOWN = {
    4: 2*(m**2+n**2) + 6*m*n - 10*(m+n) - 4,
    5: R(2,3)*(m**3+n**3) + 2*(m**2+n**2) + 50*m*n - R(392,3)*(m+n) + 264,
    6: (R(1,6)*(m**4+n**4) + R(1,3)*(m**3+n**3) + 38*(m**2*n+m*n**2)
        - R(229,6)*(m**2+n**2) - 272*m*n - R(103,3)*(m+n) + 1176),
    7: (R(1,30)*(m**5+n**5) + R(37,3)*(m**3*n+m*n**3) + R(7,6)*(m**3+n**3)
        + 18*m**2*n**2 + 25*(m**2*n+m*n**2) - 496*(m**2+n**2)
        - R(3818,3)*m*n + R(14354,5)*(m+n) - 904),
}


def nodes_for(d):
    """Triangular unisolvent nodes (a<=b) and the per-m max-n needed."""
    D = d - 2
    nd = []
    for a in range(0, D // 2 + 1):
        for b in range(a, D - a + 1):
            nd.append((d - 1 + a, d - 1 + b))
    return nd


def sample_Ed(d, verbose=True):
    """Return {(x,y): E_d(x,y)} on the unisolvent nodes (symmetrised)."""
    nd = nodes_for(d)
    by_m = {}
    for (x, y) in nd:
        by_m.setdefault(x, 0)
        by_m[x] = max(by_m[x], y)
    vals = {}
    for x in sorted(by_m):
        nmax = by_m[x]
        if verbose:
            print(f"    Ed_DP_capped(m={x}, nmax={nmax}, d_cap={d}) ...", flush=True)
        series = Ed_DP_capped(x, nmax, d)    # {n: {d: E_d(x,n)}}, capped
        for (px, py) in nd:
            if px == x:
                v = series[py].get(d, 0)
                vals[(px, py)] = v
                vals[(py, px)] = v           # symmetry
    return vals


def interpolate(d, vals):
    """Solve for the symmetric total-degree-(d-2) polynomial through vals."""
    D = d - 2
    # symmetric orbit basis: {(i,j): i<=j, i+j<=D}
    orbits = [(i, j) for j in range(D + 1) for i in range(j + 1) if i + j <= D]
    def basis(i, j, x, y):
        return x**i * y**j + (x**j * y**i if i != j else 0)
    # square linear system over the a<=b nodes
    pts = sorted({(x, y) for (x, y) in vals if x <= y})
    assert len(pts) == len(orbits), (len(pts), len(orbits))
    A = sp.zeros(len(pts), len(orbits))
    rhs = sp.zeros(len(pts), 1)
    for r, (x, y) in enumerate(pts):
        for c, (i, j) in enumerate(orbits):
            A[r, c] = basis(i, j, x, y)
        rhs[r] = vals[(x, y)]
    sol = A.solve(rhs)
    poly = sum(sol[c] * (m**i * n**j + (m**j * n**i if i != j else 0))
               for c, (i, j) in enumerate(orbits))
    return sp.expand(poly)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmax", type=int, default=7,
                    help="interpolate d=4..dmax (d>=8 is the expensive path)")
    args = ap.parse_args()
    for d in range(4, args.dmax + 1):
        print(f"[d={d}] max m = {(3*d-4)//2}, nodes = {len(nodes_for(d))}")
        vals = sample_Ed(d)
        pd = interpolate(d, vals)
        if d in KNOWN:
            diff = sp.expand(pd - KNOWN[d])
            print(f"    interpolated == known p_{d}: {diff == 0}")
            if diff != 0:
                print(f"    !! DIFF: {diff}")
        else:
            print(f"    p_{d} = {pd}")
        print()


if __name__ == "__main__":
    main()
