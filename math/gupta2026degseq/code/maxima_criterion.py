"""
The k-cone Maxima Criterion (generalised Ridge Lemma), Lemma lem:maxima.

The Ridge Lemma (2 cones) characterises the strict local
maxima of h = min(cone_1, delta+cone_2). Its proof rests on a local mechanism:
across an edge each cone changes by +-1 and the cone-differences have fixed
parity, so an INACTIVE cone (value > E by the parity gap, hence >= E+2) cannot
produce a lower neighbour, while an ACTIVE cone (value == E) does iff the step
moves toward its apex. We lift this to k cones.

  Active set at v:  A(v) = { s : c_s + dgrid(p_s, v) = E(v) }.

  MAXIMA CRITERION (Lemma 1b).  For the lower envelope E = min_s (c_s +
  dgrid(p_s, .)) that is a Z-height function (parity), a cell v=(i,j) is a
  strict local maximum of E iff in each PRESENT neighbour direction some active
  cone points back toward its apex:
     down  (i<m-1):  exists s in A(v) with r_s > i
     up    (i>0):    exists s in A(v) with r_s < i
     right (j<n-1):  exists s in A(v) with c_s_col > j
     left  (j>0):    exists s in A(v) with c_s_col < j
  (apex p_s = (r_s, c_s_col); "present" = the neighbour exists on the grid.)

This script checks the criterion reproduces the directly-computed strict-local-
max set EXACTLY, over MANY k-cone envelopes (k=2,3,4) on several grids, for all
parity-compatible active offset tuples. (We do NOT require min-set = apexes here
-- the criterion is about maxima of ANY lower envelope of distance cones, so we
test it on every parity-valid offset tuple that yields a height function.)
"""

from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Tuple

Cell = Tuple[int, int]


def gdist(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(m: int, n: int) -> Dict[Cell, List[Cell]]:
    nb = {}
    for i in range(m):
        for j in range(n):
            lst = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < n:
                    lst.append((ii, jj))
            nb[(i, j)] = lst
    return nb


def envelope(apexes: List[Cell], offs: List[int],
             cells: List[Cell]) -> Dict[Cell, int]:
    return {v: min(offs[s] + gdist(apexes[s], v) for s in range(len(apexes)))
            for v in cells}


def direct_maxima(E: Dict[Cell, int], nbrs) -> set:
    return {v for v in E if all(E[w] < E[v] for w in nbrs[v])}


def criterion_maxima(E: Dict[Cell, int], apexes, offs, m, n, nbrs) -> set:
    """Maxima predicted by the k-cone criterion (active set + 4 directions)."""
    out = set()
    for v in E:
        i, j = v
        Ev = E[v]
        A = [s for s in range(len(apexes))
             if offs[s] + gdist(apexes[s], v) == Ev]
        # four directional conditions
        down = (i == m - 1) or any(apexes[s][0] > i for s in A)
        up = (i == 0) or any(apexes[s][0] < i for s in A)
        right = (j == n - 1) or any(apexes[s][1] > j for s in A)
        left = (j == 0) or any(apexes[s][1] < j for s in A)
        if down and up and right and left:
            out.add(v)
    return out


def is_height_function(E, nbrs) -> bool:
    for v in E:
        for w in nbrs[v]:
            if abs(E[v] - E[w]) != 1:
                return False
    return True


def test_grid(m: int, n: int, ks: List[int], max_tuples: int = None):
    cells = [(i, j) for i in range(m) for j in range(n)]
    nbrs = neighbors(m, n)
    checked = 0
    mismatches = 0
    hf = 0
    for k in ks:
        for apex_idx in itertools.combinations(range(len(cells)), k):
            apexes = [cells[t] for t in apex_idx]
            D = [[gdist(apexes[s], apexes[t]) for t in range(k)]
                 for s in range(k)]
            # offsets: off[0]=0; off[t] in (-D0t, D0t) parity-compatible; with
            # pairwise parity for s<t. Enumerate all such tuples.
            ranges = []
            for t in range(k):
                if t == 0:
                    ranges.append([0])
                else:
                    ranges.append([c for c in range(-D[0][t] + 1, D[0][t])
                                   if (c - D[0][t]) % 2 == 0])
            for combo in itertools.product(*ranges):
                # pairwise parity check
                ok = True
                for s in range(k):
                    for t in range(s + 1, k):
                        if (combo[s] - combo[t] - D[s][t]) % 2 != 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                E = envelope(apexes, list(combo), cells)
                if not is_height_function(E, nbrs):
                    continue
                hf += 1
                direct = direct_maxima(E, nbrs)
                pred = criterion_maxima(E, apexes, list(combo), m, n, nbrs)
                checked += 1
                if direct != pred:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"    MISMATCH {m}x{n} k={k} apexes={apexes} "
                              f"offs={combo}: direct={sorted(direct)} "
                              f"pred={sorted(pred)}")
                if max_tuples and checked >= max_tuples:
                    break
    return checked, hf, mismatches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="3:3,3:4,4:4,3:5,4:5")
    ap.add_argument("--ks", default="2,3,4")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    print("=== k-cone Maxima Criterion verification ===")
    print(f"ks={ks}; checking criterion == direct strict-local-max set\n")
    tot_checked = tot_mis = 0
    for tok in args.grids.split(","):
        m, n = map(int, tok.split(":"))
        checked, hf, mis = test_grid(m, n, ks)
        tot_checked += checked
        tot_mis += mis
        tag = "OK" if mis == 0 else f"{mis} MISMATCH"
        print(f"  {m}x{n}: {checked} height-fn envelopes checked "
              f"(k in {ks}); {tag}")
    print()
    if tot_mis == 0:
        print(f"=== CRITERION VERIFIED on {tot_checked} envelopes (0 mismatch) ===")
    else:
        print(f"=== {tot_mis}/{tot_checked} MISMATCH ===")


if __name__ == "__main__":
    main()
