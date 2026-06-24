"""
Verify the Cone Classification that powers the Prop 4 proof.

Cone Lemma (proved): a height function with a unique local maximum q equals the
distance-cone  h(v) = h(q) - d(q,v).  Dually for a unique local minimum.

Each grid vertex q gives exactly one canonical cone (unique-max-at-q). Its degree
= 1 + (number of LOCAL maxima of d(q,.)), and the local maxima of d(q,.) are:
  q corner   -> 1 (opposite corner)        -> degree 2
  q edge     -> 2 (two opposite-side corners) -> degree 3
  q interior -> 4 (all four corners)        -> degree 5

This script checks, by DIRECT enumeration of OFG vertices, that:
  C1: the set of unique-max vertices = the mn cones, with the degree split
      4 x deg2, 2(m+n-4) x deg3, (m-2)(n-2) x deg5;
  C2: #local maxima of d(q,.) matches 1/2/4 by boundary kind of q (pure lattice
      fact, no enumeration).
"""

import argparse
from collections import Counter

from ofg import grid_neighbors, proper_3colorings, canon
from heights import height_function


def gd(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def dist_local_maxima(q, m, n):
    cells = [(i, j) for i in range(m) for j in range(n)]
    nb = {}
    for (i, j) in cells:
        nb[(i, j)] = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= i + di < m and 0 <= j + dj < n]
    return [v for v in cells if all(gd(q, w) < gd(q, v) for w in nb[v])]


def c2_lattice(m, n):
    corners = {(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)}
    tally = {"corner": set(), "edge": set(), "interior": set()}
    for i in range(m):
        for j in range(n):
            q = (i, j)
            k = ("corner" if q in corners
                 else "edge" if i in (0, m - 1) or j in (0, n - 1)
                 else "interior")
            tally[k].add(len(dist_local_maxima(q, m, n)))
    return {k: sorted(v) for k, v in tally.items()}


def c1_enumerate(m, n):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    seen = set()
    uniqmax_deg = Counter()
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        h = height_function(cc, m, n, nbrs, cells, idx)
        hv = {cells[k]: h[k] for k in range(len(cells))}
        maxs = [c for c in cells if all(hv[nb] < hv[c] for nb in nbrs[c])]
        deg = sum(1 for c in cells if len({hv[nb] for nb in nbrs[c]}) <= 1)
        if len(maxs) == 1:
            uniqmax_deg[deg] += 1
    return uniqmax_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="3:3,3:4,4:4,3:5,4:5,2:5")
    args = ap.parse_args()
    print("== C2: #local maxima of d(q,.) by boundary kind (pure lattice) ==")
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        print(f"  m={m} n={n}: {c2_lattice(m, n)}   (expect corner:[1] edge:[2] interior:[4])")
    print()
    print("== C1: degree split of the unique-max (cone) vertices ==")
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        d = c1_enumerate(m, n)
        exp = {2: 4, 3: 2 * (m + n - 4), 5: (m - 2) * (n - 2)}
        exp = {k: v for k, v in exp.items() if v > 0}
        ok = (dict(d) == exp)
        print(f"  m={m} n={n}: uniq-max degrees={dict(sorted(d.items()))}  "
              f"expect={exp}  [{'OK' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
