"""
Verify the ENVELOPE LEMMA (the universal tool for the degree-4 count).

CLAIM (general, any number of extrema). Let h be a height function on the grid
G_{m,n} (|h(u)-h(v)|=1 on every edge). Let P be its set of strict local minima
and Q its set of strict local maxima. Then for every vertex v,
    h(v) = min_{p in P} [ h(p) + d(p,v) ]   (lower envelope of min-cones)
         = max_{q in Q} [ h(q) - d(q,v) ]   (upper envelope of max-cones),
where d is the grid graph distance.

This script checks the identity for EVERY height function (every OFG vertex),
not just the (2 min, 2 max) ones, across many grids incl. asymmetric, and
separately confirms it for the degree-4 slice.
"""
import argparse
from ofg import grid_neighbors, proper_3colorings, canon
from heights import height_function


def gdist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def check(m, n):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    seen = set()
    n_all = n_bad_all = 0
    n_d4 = n_bad_d4 = 0
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        h = height_function(cc, m, n, nbrs, cells, idx)
        hv = {cells[k]: h[k] for k in range(len(cells))}
        mins = [c for c in cells if all(hv[nb] > hv[c] for nb in nbrs[c])]
        maxs = [c for c in cells if all(hv[nb] < hv[c] for nb in nbrs[c])]
        ok = True
        for v in cells:
            lo = min(hv[p] + gdist(p, v) for p in mins)
            hi = max(hv[q] - gdist(q, v) for q in maxs)
            if not (lo == hv[v] == hi):
                ok = False
                break
        n_all += 1
        if not ok:
            n_bad_all += 1
        if (len(mins), len(maxs)) == (2, 2):
            n_d4 += 1
            if not ok:
                n_bad_d4 += 1
    return n_all, n_bad_all, n_d4, n_bad_d4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="2:2,2:3,3:3,2:4,3:4,4:4,2:5,3:5,4:5,2:6,3:6")
    args = ap.parse_args()
    grand_bad = 0
    print(f"  {'m':>2} {'n':>2} {'#OFGvert':>9} {'bad(all)':>9} {'#deg4':>6} {'bad(deg4)':>9}")
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        a, ba, d4, bd4 = check(m, n)
        grand_bad += ba
        flag = "OK" if ba == 0 else "FAIL"
        print(f"  {m:>2} {n:>2} {a:>9} {ba:>9} {d4:>6} {bd4:>9}  [{flag}]")
    print(f"\n>>> ENVELOPE LEMMA: {'HOLDS on every height function, all cases' if grand_bad == 0 else 'FAILED'}")


if __name__ == "__main__":
    main()
