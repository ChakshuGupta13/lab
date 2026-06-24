"""
Direct validation of the #deg-2 = 4 THEOREM's structural conclusion:
every degree-2 OFG vertex is a corner gradient h(i,j) = h(corner) + d(corner,(i,j)),
with its unique local min and unique local max at OPPOSITE corners.

This backs the monotone-path proof:
  unique min p, unique max q  =>  D=h(q)-h(p) <= d(p,q) (shortest path)
  and D >= d(p,v)+d(q,v) for all v (monotone descend/ascend paths)
  => D = d(p,q) and every v on a p-q geodesic => p,q opposite corners
  => h(v) = h(p) + d(p,v)  (corner gradient).
"""

import argparse
from collections import deque

from ofg import grid_neighbors, proper_3colorings, canon
from heights import height_function, local_extrema


def grid_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def check(m, n):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    corners = {(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)}

    seen = set()
    deg2 = []
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        # degree via flippability
        deg = sum(1 for k, cell in enumerate(cells)
                  if len({cc[idx[nb]] for nb in nbrs[cell]}) <= 1)
        if deg == 2:
            deg2.append(cc)

    ok = True
    for cc in deg2:
        h = height_function(cc, m, n, nbrs, cells, idx)
        hvals = {cells[k]: h[k] for k in range(len(cells))}
        # find the unique local min and max
        mins = [c for c in cells if all(hvals[nb] > hvals[c] for nb in nbrs[c])]
        maxs = [c for c in cells if all(hvals[nb] < hvals[c] for nb in nbrs[c])]
        cond_unique = (len(mins) == 1 and len(maxs) == 1)
        p, q = mins[0], maxs[0]
        cond_corners = (p in corners and q in corners)
        cond_opposite = (grid_dist(p, q) == (m - 1) + (n - 1))
        # gradient: h(v) = h(p) + d(p,v) for all v
        cond_gradient = all(hvals[c] - hvals[p] == grid_dist(p, c) for c in cells)
        if not (cond_unique and cond_corners and cond_opposite and cond_gradient):
            ok = False
            print(f"    COUNTEREXAMPLE m={m} n={n}: min={p} max={q} "
                  f"unique={cond_unique} corners={cond_corners} "
                  f"opposite={cond_opposite} gradient={cond_gradient}")
    return len(deg2), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="2:2,2:3,3:2,2:4,3:3,4:2,2:5,3:4,4:3,5:2,3:5,4:4,5:3")
    args = ap.parse_args()
    print("== #deg-2 vertices are EXACTLY the 4 opposite-corner gradients ==")
    allok = True
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        cnt, ok = check(m, n)
        allok &= ok and (cnt == 4)
        print(f"  m={m} n={n}: #deg2={cnt} all-corner-gradients={ok} "
              f"[{'OK' if ok and cnt == 4 else 'FAIL'}]")
    print(f"  -> {'THEOREM conclusion holds on all cases' if allok else 'FAILED'}")


if __name__ == "__main__":
    main()
