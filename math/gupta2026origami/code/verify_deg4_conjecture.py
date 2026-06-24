"""
COMPREHENSIVE verification that the six per-family closed forms (each derived
from the cone-pair bijection + ridge analysis) reproduce the brute-force
degree-4 family counts exactly, and that they sum to Conjecture 6.1.

Family closed forms (min(m,n) >= 3):
  CC|EE = EE|CC = 4
  CE|EE = EE|CE = 8(m+n-6)
  CE|CE = 2(m-2)(m-3) + 2(n-2)(n-3) + 16
  EE|EE = 2(m-2)(n-2) + 4(m-3)(n-3)
  sum   = 2m^2 + 2n^2 + 6mn - 10(m+n) - 4
"""
import argparse
from collections import Counter
from ofg import grid_neighbors, proper_3colorings, canon
from heights import height_function


def kind(cell, m, n):
    i, j = cell
    c = (i in (0, m - 1)) + (j in (0, n - 1))
    return "C" if c == 2 else "E" if c == 1 else "I"


def brute_families(m, n):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    seen = set()
    fams = Counter()
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        h = height_function(cc, m, n, nbrs, cells, idx)
        hv = {cells[k]: h[k] for k in range(len(cells))}
        mins = [c for c in cells if all(hv[nb] > hv[c] for nb in nbrs[c])]
        maxs = [c for c in cells if all(hv[nb] < hv[c] for nb in nbrs[c])]
        if (len(mins), len(maxs)) != (2, 2):
            continue
        key = ("".join(sorted(kind(c, m, n) for c in mins)) + "|" +
               "".join(sorted(kind(c, m, n) for c in maxs)))
        fams[key] += 1
    return fams


def closed_forms(m, n):
    return {
        "CC|EE": 4,
        "EE|CC": 4,
        "CE|EE": 8 * (m + n - 6),
        "EE|CE": 8 * (m + n - 6),
        "CE|CE": 2 * (m - 2) * (m - 3) + 2 * (n - 2) * (n - 3) + 16,
        "EE|EE": 2 * (m - 2) * (n - 2) + 4 * (m - 3) * (n - 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="3:3,3:4,4:4,3:5,4:5,5:5,3:6,4:6,5:6,6:6,3:7,4:7,3:8")
    args = ap.parse_args()
    allok = True
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        bf = brute_families(m, n)
        cf = closed_forms(m, n)
        keys = set(bf) | set(cf)
        row_ok = all(bf.get(k, 0) == cf.get(k, 0) for k in keys)
        total_bf = sum(bf.values())
        total_cf = sum(cf.values())
        conj = 2 * m * m + 2 * n * n + 6 * m * n - 10 * (m + n) - 4
        ok = row_ok and total_bf == total_cf == conj
        allok &= ok
        diffs = {k: (bf.get(k, 0), cf.get(k, 0)) for k in keys if bf.get(k, 0) != cf.get(k, 0)}
        print(f"m={m} n={n}: total brute={total_bf} closed={total_cf} conj={conj} "
              f"[{'OK' if ok else 'FAIL ' + str(diffs)}]")
    print(f"\n>>> ALL SIX FAMILY CLOSED FORMS vs BRUTE FORCE: "
          f"{'EXACT MATCH, sum = Conjecture 6.1' if allok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
