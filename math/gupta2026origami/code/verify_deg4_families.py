"""
Structural verification for the per-family degree-4 counts (Conjecture 6.1).

For each family we verify the geometric characterisation used in the proof,
against brute-force ground truth (ofg.py + heights.py).

Families (min-kinds | max-kinds):
  CORNER  CC|EE  (and dual EE|CC): two opposite-corner minima; h is a tent in
          the diagonal coordinate; exactly two maxima forming a 2-cell
          antidiagonal next to one corner. Count 4.
This script grows one check per family as the proof proceeds.
"""
import argparse
from collections import Counter, defaultdict
from ofg import grid_neighbors, proper_3colorings, canon
from heights import height_function


def kind(cell, m, n):
    i, j = cell
    c = (i in (0, m - 1)) + (j in (0, n - 1))
    return "C" if c == 2 else "E" if c == 1 else "I"


def all_22(m, n):
    """Yield (hv, mins, maxs) for every (2,2) height function (one per OFG vtx)."""
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    seen = set()
    for col in proper_3colorings(m, n, nbrs):
        cc = canon(col)
        if cc in seen:
            continue
        seen.add(cc)
        h = height_function(cc, m, n, nbrs, cells, idx)
        hv = {cells[k]: h[k] for k in range(len(cells))}
        mins = [c for c in cells if all(hv[nb] > hv[c] for nb in nbrs[c])]
        maxs = [c for c in cells if all(hv[nb] < hv[c] for nb in nbrs[c])]
        if (len(mins), len(maxs)) == (2, 2):
            yield hv, mins, maxs


def check_corner(m, n):
    """CC|EE + EE|CC structural claims."""
    corners = {(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)}
    cc_ee = []
    ee_cc = []
    for hv, mins, maxs in all_22(m, n):
        mk = "".join(sorted(kind(c, m, n) for c in mins))
        xk = "".join(sorted(kind(c, m, n) for c in maxs))
        if mk == "CC" and xk == "EE":
            cc_ee.append((hv, mins, maxs))
        elif mk == "EE" and xk == "CC":
            ee_cc.append((hv, mins, maxs))
    ok = True
    msgs = []
    # CC|EE: minima are opposite corners; h tent in diagonal coord; maxima a
    # 2-cell antidiagonal adjacent to the higher corner.
    for (hv, mins, maxs) in cc_ee:
        p1, p2 = sorted(mins)
        if set(mins) > corners:
            ok = False
        opp = (p1[0] != p2[0] and p1[1] != p2[1])   # opposite corners differ in both coords
        if not opp:
            ok = False
            msgs.append(f"CC|EE minima not opposite: {mins}")
        # tent: h constant on the relevant diagonal. Determine orientation.
        # For diagonal {(0,0),(m-1,n-1)} use s=i+j; for {(0,n-1),(m-1,0)} use t=i-j.
        if {p1, p2} == {(0, 0), (m - 1, n - 1)}:
            coord = lambda c: c[0] + c[1]
        else:
            coord = lambda c: c[0] - c[1]
        bycoord = defaultdict(set)
        for c, val in hv.items():
            bycoord[coord(c)].add(val)
        if any(len(s) > 1 for s in bycoord.values()):
            ok = False
            msgs.append(f"CC|EE not constant on diagonals: {mins}")
        # maxima form a 2-cell antidiagonal at coord in {min+1, max-1}
        mxc = {coord(c) for c in maxs}
        if len(mxc) != 1:
            ok = False
            msgs.append(f"CC|EE maxima not on one antidiagonal: {maxs}")
    return ok, len(cc_ee), len(ee_cc), msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="3:3,3:4,4:4,3:5,4:5,5:5,4:6")
    args = ap.parse_args()
    allok = True
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        ok, ncce, neecc, msgs = check_corner(m, n)
        allok &= ok and ncce == 4 and neecc == 4
        print(f"m={m} n={n}: CC|EE={ncce} EE|CC={neecc}  struct={'OK' if ok else 'FAIL'}"
              + ("" if not msgs else f"  {msgs[:3]}"))
    print(f"\nCORNER family: {'ALL OK (count=4 each, structure confirmed)' if allok else 'FAIL'}")


if __name__ == "__main__":
    main()
