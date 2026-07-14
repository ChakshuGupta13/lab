"""
Envelope Structure Theorem for OFG(M_{m,n}) (Theorem thm:envelope).

Verifies, by exhaustive enumeration on small grids, the structural claims the
envelope construction rests on.

Background:
  OFG(M_{m,n}) vertices  <->  integer height functions h on the grid G_{m,n}
  (0-based cells (i,j), 0<=i<m, 0<=j<n) with |h(u)-h(v)|=1 across each edge and
  h(0,0)=0.  The degree of a vertex = number of strict local extrema of h
  (Lemma degree-extrema).  A strict local minimum: all neighbours higher; a
  strict local maximum: all neighbours lower.

Claims verified here (these become Theorem/Lemma statements in the paper):

  (E1) [Envelope]  h(v) = min_{p in P} ( h(p) + dgrid(p,v) )  for every cell v,
       where P = strict-local-min set.  (This is the committed Envelope Lemma;
       re-checked here as a foundation.)

  (E2) [Minima = apexes of the envelope]  Building the lower envelope of the
       distance-cones seated at P with offsets c_p = h(p), its strict local
       minima recomputed from scratch equal P exactly.

  (E3) [Active condition is LINEAR and characterises min-ness]  Define p in P
       "active" iff  c_p < c_{p'} + dgrid(p',p)  for every other p' in P.  Then
       the active apexes are EXACTLY the strict local minima of the envelope.
       This predicate is a boolean combination of linear inequalities in the
       apex coordinates (through dgrid) and the offsets -- the Presburger-
       encodable condition the construction needs.

  (E4) [Parity]  for p,p' in P:  c_p - c_{p'} ≡ dgrid(p,p')  (mod 2).

  (E5) [O(d) parametrization, independent of (m,n)]  with a:=|P|, b:=|Q|,
       d=a+b, the data (P, (c_p)_{p in P}) has 2a coordinates + (a-1) offsets
       (after fixing one offset by normalisation) = 3a-1 integer degrees of
       freedom, and a <= d-1.  Using colour inversion to take the SMALLER
       extremum side gives min(a,b) <= floor(d/2) apexes, so the parameter
       count is O(d), INDEPENDENT of m,n.  THIS is the reduction that brings
       the count inside Barvinok-Woods (a fixed counted-variable dimension),
       in contrast to the naive mn height variables (which grow with the
       parameters and are therefore outside the Presburger framework).

The script also tabulates, per degree d, the observed split multiset
{(a,b)=(|P|,|Q|)}, and cross-checks the height-function count against the
committed brute-force OFG oracle (ofg.analyze) when importable.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

Cell = Tuple[int, int]


def dgrid(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(m: int, n: int) -> Dict[Cell, List[Cell]]:
    nbrs: Dict[Cell, List[Cell]] = {}
    for i in range(m):
        for j in range(n):
            lst = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < n:
                    lst.append((ii, jj))
            nbrs[(i, j)] = lst
    return nbrs


def enum_height_functions(m: int, n: int) -> List[Dict[Cell, int]]:
    """All integer height functions h on G_{m,n} with h(0,0)=0 and
    |h(u)-h(v)|=1 across each grid edge.  Row-major backtracking: a new cell
    must differ by exactly 1 from each already-placed neighbour (left, up)."""
    cells = [(i, j) for i in range(m) for j in range(n)]
    h: Dict[Cell, int] = {}
    out: List[Dict[Cell, int]] = []

    def bt(k: int) -> None:
        if k == len(cells):
            out.append(dict(h))
            return
        (i, j) = cells[k]
        cons = []
        if j > 0:
            cons.append(h[(i, j - 1)])
        if i > 0:
            cons.append(h[(i - 1, j)])
        if not cons:
            h[(i, j)] = 0
            bt(k + 1)
            del h[(i, j)]
            return
        cand = {cons[0] - 1, cons[0] + 1}
        for c in cons[1:]:
            cand &= {c - 1, c + 1}
        for v in sorted(cand):
            h[(i, j)] = v
            bt(k + 1)
            del h[(i, j)]

    bt(0)
    return out


def extrema(h: Dict[Cell, int], nbrs: Dict[Cell, List[Cell]]):
    P = [v for v in h if all(h[w] > h[v] for w in nbrs[v])]
    Q = [v for v in h if all(h[w] < h[v] for w in nbrs[v])]
    return P, Q


def verify_one(h: Dict[Cell, int], cells: List[Cell],
               nbrs: Dict[Cell, List[Cell]]) -> Tuple[int, int, int]:
    """Check E1-E5 on a single height function. Returns (d, a, b)."""
    P, Q = extrema(h, nbrs)
    a, b = len(P), len(Q)
    d = a + b

    # E1 Envelope (min over min-cones) and its dual (max over max-cones)
    for v in cells:
        env = min(h[p] + dgrid(p, v) for p in P)
        assert env == h[v], f"E1 min-envelope fail at {v}"
        dual = max(h[q] - dgrid(q, v) for q in Q)
        assert dual == h[v], f"E1 max-envelope fail at {v}"

    # E2 rebuild envelope from (P, offsets), recompute its strict minima
    env = {v: min(h[p] + dgrid(p, v) for p in P) for v in cells}
    env_minset = {v for v in cells if all(env[w] > env[v] for w in nbrs[v])}
    assert env_minset == set(P), "E2 envelope minset != P"

    # E3 active predicate (linear) == being a strict local min of the envelope
    active = {p for p in P
              if all(h[p] < h[p2] + dgrid(p2, p) for p2 in P if p2 != p)}
    assert active == set(P), "E3 active set != P"

    # E4 parity
    for p in P:
        for p2 in P:
            assert (h[p] - h[p2] - dgrid(p, p2)) % 2 == 0, "E4 parity fail"

    # E5 a <= d-1 (and dually b <= d-1); min(a,b) <= floor(d/2)
    if d >= 2:
        assert a <= d - 1 and b <= d - 1, "E5 side bound fail"
        assert min(a, b) <= d // 2, "E5 floor(d/2) fail"

    return d, a, b


def try_ofg_vertex_count(m: int, n: int):
    """Cross-check #height-functions vs committed brute OFG oracle, if importable."""
    sib = os.path.join(os.path.dirname(__file__), "..", "origami-flipgraph")
    sib = os.path.abspath(sib)
    if sib not in sys.path:
        sys.path.insert(0, sib)
    try:
        import ofg  # type: ignore
        info = ofg.analyze(m, n, want_diameter=False)
        return info["V"], info["deg_dist"]
    except Exception as e:  # noqa: BLE001
        return None, f"ofg import/analyze failed: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="3:3,3:4,4:4,3:5,4:5",
                    help="comma list of m:n grids to verify")
    ap.add_argument("--cross-check-ofg", action="store_true",
                    help="compare height-function count + deg dist vs ofg.analyze")
    args = ap.parse_args()

    print("=== Envelope Structure Theorem verification ===")
    print("Checking E1 (envelope), E2 (minima=apexes), E3 (active=linear),")
    print("E4 (parity), E5 (O(d) params) on every height function.\n")

    all_ok = True
    for tok in args.grids.split(","):
        m, n = map(int, tok.split(":"))
        hfs = enum_height_functions(m, n)
        cells = [(i, j) for i in range(m) for j in range(n)]
        nbrs = neighbors(m, n)

        deg_hist: Counter = Counter()
        split_hist: Dict[int, Counter] = defaultdict(Counter)
        max_a = 0
        for h in hfs:
            d, a, b = verify_one(h, cells, nbrs)
            deg_hist[d] += 1
            split_hist[d][(a, b)] += 1
            max_a = max(max_a, min(a, b))

        print(f"--- grid {m}x{n}:  {len(hfs)} height functions  "
              f"(all E1-E5 PASS)  max min(a,b)={max_a} ---")
        for d in sorted(deg_hist):
            splits = ", ".join(f"({a},{b}):{c}"
                               for (a, b), c in sorted(split_hist[d].items()))
            print(f"    deg {d}: {deg_hist[d]:6d}   splits[{splits}]")

        if args.cross_check_ofg:
            V, dd = try_ofg_vertex_count(m, n)
            if V is None:
                print(f"    [ofg cross-check skipped: {dd}]")
            else:
                match_V = (V == len(hfs))
                match_deg = all(deg_hist.get(k, 0) == dd.get(k, 0)
                                for k in set(deg_hist) | set(dd))
                print(f"    [ofg: V={V} {'OK' if match_V else 'MISMATCH'};  "
                      f"deg-dist {'OK' if match_deg else 'MISMATCH'}]")
                all_ok = all_ok and match_V and match_deg
        print()

    print("=== ALL CHECKS PASSED ===" if all_ok else "=== MISMATCH DETECTED ===")


if __name__ == "__main__":
    main()
