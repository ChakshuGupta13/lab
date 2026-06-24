"""
Fast degree-5 (2,3) counter via the cone-pair bijection.

A (2,3) height function = min of two cones with exactly 2 minima and 3 maxima.
Same bijection as degree-4 but with 3 maxima instead of 2.
No brute-force coloring needed — enumerate (p1,p2,delta) directly.
"""
import argparse
from collections import Counter
from ofg import grid_neighbors


def gdist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def kind(cell, m, n):
    i, j = cell
    c = (i in (0, m - 1)) + (j in (0, n - 1))
    return "C" if c == 2 else "E" if c == 1 else "I"


def count_23(m, n):
    cells = [(i, j) for i in range(m) for j in range(n)]
    nbrs = grid_neighbors(m, n)
    groups = Counter()
    for ia in range(len(cells)):
        for ib in range(ia + 1, len(cells)):
            p1, p2 = cells[ia], cells[ib]
            D = gdist(p1, p2)
            d1 = {v: gdist(p1, v) for v in cells}
            d2 = {v: gdist(p2, v) for v in cells}
            for delta in range(-D, D + 1):
                if (delta - D) % 2 != 0:
                    continue
                h = {v: min(d1[v], delta + d2[v]) for v in cells}
                # quick check: is h a valid height function?
                ok = True
                for v in cells:
                    for w in nbrs[v]:
                        if abs(h[v] - h[w]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                mins = [v for v in cells if all(h[w] > h[v] for w in nbrs[v])]
                if set(mins) != {p1, p2}:
                    continue
                maxs = [v for v in cells if all(h[w] < h[v] for w in nbrs[v])]
                if len(maxs) == 3:
                    mk = "".join(sorted(kind(c, m, n) for c in mins))
                    xk = "".join(sorted(kind(c, m, n) for c in maxs))
                    groups[(mk, xk)] += 1
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="3:3,3:4,4:4,3:5,4:5,5:5,3:6,4:6,3:7,5:6,4:7,6:6,3:8,5:7")
    args = ap.parse_args()
    allkeys = set()
    table = {}
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        g = count_23(m, n)
        table[(m, n)] = g
        allkeys |= set(g)
        total = sum(g.values())
        print(f"m={m} n={n}: #(2,3)={total}")
    print("\n== per-group counts ==")
    for key in sorted(allkeys):
        row = "  ".join(f"{m}x{n}:{table[(m,n)].get(key,0):>4}"
                        for (m, n) in sorted(table))
        print(f"  {key[0]}|{key[1]}: {row}")
    # total (2,3) + formula fitting
    print("\n== totals ==")
    for (m, n) in sorted(table):
        total = sum(table[(m, n)].values())
        print(f"  {m}x{n}: #(2,3)={total:>4}")


if __name__ == "__main__":
    main()
