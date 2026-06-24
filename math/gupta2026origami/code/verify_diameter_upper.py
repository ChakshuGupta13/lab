"""
Diameter upper bound — reduction probe.

We established (diameter_analysis.py, cited recoloring-distance formula verified vs
BFS): for two OFG vertices with height functions h, h',
    d(h,h') = (1/2) min_{K in Z} sum_v | h'(v) - h(v) - K |.
Set phi = (h' - h)/2.  On each grid edge {u,v}, h and h' each change by +-1, so
phi(u)-phi(v) in {-1,0,1}: phi is an INTEGER 1-LIPSCHITZ function on the grid.
And  d(h,h') = min_{K'} sum_v |phi(v) - K'| =: disp(phi)  (median dispersion).

Hence  diam OFG(M_{m,n}) = max over REALIZABLE phi of disp(phi).
Since {realizable phi} subset {integer 1-Lipschitz phi}, we get
    diam <= max over ALL integer 1-Lipschitz phi of disp(phi).
If that max equals D(m,n) = disp(antidiagonal a(i,j)=i+j), then (with the proved
lower bound diam >= D) we conclude diam = D EXACTLY — a pure lattice fact, no
origami realizability needed.

THIS SCRIPT decides the approach: enumerate ALL integer 1-Lipschitz phi (phi(0,0)=0)
on small grids and check  max disp(phi) == D(m,n), and that the antidiagonal attains it.
Also checks the proof MECHANISM: per-threshold tail domination
    #{phi >= med+t} + #{phi <= med-t}  <=  #{a >= med_a+t} + #{a <= med_a-t}   (all t>=1),
which would give disp(phi) <= disp(a) termwise (the isoperimetric route).
"""

import argparse
import statistics


def disp(vals):
    """min_K sum |x-K| = sum(top half) - sum(bottom half)."""
    s = sorted(vals)
    N = len(s)
    return sum(s[N - N // 2:]) - sum(s[:N // 2])


def median_int(vals):
    s = sorted(vals)
    N = len(s)
    return s[N // 2]  # an integer median minimiser (lower-median is fine)


def enumerate_lipschitz(m, n):
    """DFS over integer 1-Lipschitz phi with phi(0,0)=0, yielding value-tuples."""
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    # neighbors already assigned earlier in row-major order constrain each cell
    prev_nbrs = []
    for k, (i, j) in enumerate(cells):
        pn = []
        for (di, dj) in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            c2 = (i + di, j + dj)
            if c2 in idx and idx[c2] < k:
                pn.append(idx[c2])
        prev_nbrs.append(pn)

    phi = [None] * len(cells)
    out = []

    def bt(k):
        if k == len(cells):
            out.append(tuple(phi))
            return
        if not prev_nbrs[k]:
            phi[k] = 0
            bt(k + 1)
            phi[k] = None
            return
        lo = max(phi[p] for p in prev_nbrs[k]) - 1
        hi = min(phi[p] for p in prev_nbrs[k]) + 1
        for val in range(lo, hi + 1):
            phi[k] = val
            bt(k + 1)
        phi[k] = None

    bt(0)
    return cells, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="2:2,2:3,3:3,2:4,3:4,2:5")
    args = ap.parse_args()
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        cells, phis = enumerate_lipschitz(m, n)
        a = [i + j for (i, j) in cells]
        D = disp(a)
        best = -1
        argmax_is_antidiag = False
        tail_dom_ok = True
        med_a = median_int(a)
        amax = max(a)
        for phi in phis:
            d = disp(phi)
            if d > best:
                best = d
            # per-threshold tail domination vs antidiagonal
            med = median_int(phi)
            for t in range(1, amax + 2):
                lhs = sum(1 for x in phi if x >= med + t) + sum(1 for x in phi if x <= med - t)
                rhs = sum(1 for x in a if x >= med_a + t) + sum(1 for x in a if x <= med_a - t)
                if lhs > rhs:
                    tail_dom_ok = False
        # is the antidiagonal an argmax?
        argmax_is_antidiag = (disp(a) == best)
        print(f"m={m} n={n}: #1-Lip phi={len(phis)}  D(m,n)=disp(antidiag)={D}  "
              f"max disp(phi)={best}  "
              f"[max==D? {best == D}]  [antidiag attains? {argmax_is_antidiag}]  "
              f"[tail-domination holds? {tail_dom_ok}]")


if __name__ == "__main__":
    main()
