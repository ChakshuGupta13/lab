"""
Height-function analysis of OFG(M_{m,n}) — the proof tool.

Every proper 3-coloring gamma of a grid (4-cycle-generated, so all 3-colorings have
height functions; Eppstein 2019, Cereceda 2007) lifts to a height function
h: V -> Z with |h(u)-h(v)| = 1 on edges and color = h mod 3, fixed by h(0,0)=0.

KEY LEMMA (to verify here): a grid vertex v is flippable under gamma
  <=>  v is a strict local extremum of h (all neighbors above, or all below).
Hence  degree of gamma in OFG  ==  number of local extrema of h.

This script:
  1. cross-checks degree (from ofg flippability) == #local extrema (from h),
     for every OFG vertex, m,n in a small range -> asserts the lemma;
  2. tabulates #deg-2 and #deg-3 across m,n to confirm the conjectured
     closed forms  #deg-2 = 4  and  #deg-3 = 4(m+n-4).
"""

import argparse
from collections import deque

from ofg import grid_neighbors, proper_3colorings, canon


def height_function(coloring, m, n, nbrs, cells, idx):
    """Lift a proper 3-coloring to h with h(cell0)=0; assert consistency."""
    h = {0: 0}
    dq = deque([0])
    while dq:
        k = dq.popleft()
        ck = coloring[k]
        for nb in nbrs[cells[k]]:
            kk = idx[nb]
            diff = (coloring[kk] - ck) % 3           # in {1,2}
            delta = 1 if diff == 1 else -1
            hv = h[k] + delta
            if kk in h:
                if h[kk] != hv:                      # would violate 4-cycle consistency
                    raise AssertionError("height function inconsistent")
            else:
                h[kk] = hv
                dq.append(kk)
    return h


def local_extrema(h, m, n, nbrs, cells, idx):
    cnt = 0
    for k, cell in enumerate(cells):
        hv = h[k]
        nh = [h[idx[nb]] for nb in nbrs[cell]]
        if all(x > hv for x in nh) or all(x < hv for x in nh):
            cnt += 1
    return cnt


def degree_via_flip(coloring, nbrs, cells, idx):
    """Flippability degree, identical rule to ofg.build_ofg."""
    deg = 0
    for k, cell in enumerate(cells):
        nb_colors = {coloring[idx[nb]] for nb in nbrs[cell]}
        if len(nb_colors) <= 1:
            deg += 1
    return deg


def analyze(m, n, cross_check=True):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}

    seen = set()
    deg_dist = {}
    mismatches = 0
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        deg = degree_via_flip(cc, nbrs, cells, idx)
        if cross_check:
            h = height_function(cc, m, n, nbrs, cells, idx)
            ext = local_extrema(h, m, n, nbrs, cells, idx)
            if ext != deg:
                mismatches += 1
        deg_dist[deg] = deg_dist.get(deg, 0) + 1
    return deg_dist, mismatches, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-mn", default="2:2,2:3,2:4,3:3,3:4,4:4,2:5,3:5",
                    help="m:n cases to cross-check degree==#extrema")
    ap.add_argument("--count-cases",
                    default="2:2,2:3,2:4,2:5,3:3,3:4,3:5,4:4,4:5,5:5,6:6,2:6,4:6",
                    help="m:n cases to tabulate #deg2,#deg3")
    args = ap.parse_args()

    print("== KEY LEMMA cross-check: degree(gamma) == #local extrema of h ==")
    all_ok = True
    for tok in args.cross_mn.split(","):
        m, n = map(int, tok.split(":"))
        deg_dist, mism, V = analyze(m, n, cross_check=True)
        ok = (mism == 0)
        all_ok &= ok
        print(f"  m={m} n={n}: V={V} mismatches={mism}  [{'OK' if ok else 'FAIL'}]")
    print(f"  -> LEMMA {'HOLDS on all checked cases' if all_ok else 'FAILED'}")

    print()
    print("== #deg-2 and #deg-3 vs conjectured closed forms ==")
    print(f"  {'m':>2} {'n':>2} {'#deg2':>6} {'pred4':>5} {'#deg3':>6} {'4(m+n-4)':>9}")
    for tok in args.count_cases.split(","):
        m, n = map(int, tok.split(":"))
        deg_dist, _, V = analyze(m, n, cross_check=False)
        d2 = deg_dist.get(2, 0)
        d3 = deg_dist.get(3, 0)
        pred3 = 4 * (m + n - 4)
        f2 = "OK" if d2 == 4 else "!!"
        f3 = "OK" if d3 == pred3 else "!!"
        print(f"  {m:>2} {n:>2} {d2:>6} {'4':>5}[{f2}] {d3:>6} {pred3:>9}[{f3}]")


if __name__ == "__main__":
    main()
