"""
Verify a candidate closed form for #deg-4 on HELD-OUT cases (not used to derive it).

Derivation used m,n in {2..5}x{2..6}. Candidate (for the interior regime min(m,n)>=3):
    #deg-4 = 2m^2 + 2n^2 + 6mn - 10(m+n) - 4.
The boundary family m=2 (or n=2) is a SEPARATE edge regime (Hull): #deg-4(2,n) = 2n^2-2n-4
for n>=3 (his Table 1 d=4 row, n>=3).

Held-out checks: (3,7),(4,7),(6,6) for the interior formula; (2,7) for the boundary.
"""

import argparse

from ofg import grid_neighbors, proper_3colorings, canon
from heights import height_function


def count_deg4(m, n):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    seen = set()
    c4 = 0
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        h = height_function(cc, m, n, nbrs, cells, idx)
        hv = {cells[k]: h[k] for k in range(len(cells))}
        deg = sum(1 for c in cells if len({hv[nb] for nb in nbrs[c]}) <= 1)
        if deg == 4:
            c4 += 1
    return c4


def interior(m, n):
    return 2 * m * m + 2 * n * n + 6 * m * n - 10 * (m + n) - 4


def boundary2(n):
    return 2 * n * n - 2 * n - 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="3:7,4:7,2:7")
    args = ap.parse_args()
    print("== held-out #deg-4 verification ==")
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        actual = count_deg4(m, n)
        if min(m, n) >= 3:
            pred = interior(m, n); lbl = "interior 2m^2+2n^2+6mn-10(m+n)-4"
        elif m == 2:
            pred = boundary2(n); lbl = "boundary 2n^2-2n-4"
        elif n == 2:
            pred = boundary2(m); lbl = "boundary 2m^2-2m-4"
        else:
            pred = None; lbl = "n/a"
        ok = (actual == pred)
        print(f"  m={m} n={n}: actual={actual} pred={pred} [{lbl}]  [{'OK' if ok else 'MISMATCH'}]")


if __name__ == "__main__":
    main()
