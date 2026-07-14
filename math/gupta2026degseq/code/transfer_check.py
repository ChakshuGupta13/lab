"""
The full x-marked column transfer DP reproduces the OFG degree distribution
E_d(m,n). Validates the extremum bookkeeping that underlies the period-1
spectral computation (transfer_period.py).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from transfer_period import proper_columns, valid_transition
from envelope_structure import enum_height_functions, extrema, neighbors


def edge_extrema(col, m, left, right) -> int:
    """#strict local extrema of `col` given neighbour columns left/right
    (None = grid edge, neighbour absent)."""
    cnt = 0
    for i in range(m):
        nb = []
        if i > 0:
            nb.append(col[i - 1])
        if i < m - 1:
            nb.append(col[i + 1])
        if left is not None:
            nb.append(left[i])
        if right is not None:
            nb.append(right[i])
        if len(set(nb)) == 1:
            cnt += 1
    return cnt


def Ed_transfer(m: int, n: int) -> dict:
    """OFG degree distribution {d: count} for the m x n grid via the x-marked
    column DP. State = (prev_col, cur_col); a column is finalised (extrema
    counted) once both its horizontal neighbours are known/absent. Colourings
    are counted with a free corner, so OFG count = colourings / 3."""
    cols = proper_columns(m)
    cur = defaultdict(Counter)
    for c0 in cols:
        cur[(None, c0)][0] += 1
    for _ in range(1, n):
        nxt = defaultdict(Counter)
        for (a, b), cdict in cur.items():
            for c in cols:
                if not valid_transition(b, c):
                    continue
                ext = edge_extrema(b, m, a, c)  # finalise column b
                for xd, ct in cdict.items():
                    nxt[(b, c)][xd + ext] += ct
        cur = nxt
    total = Counter()
    for (a, b), cdict in cur.items():
        ext = edge_extrema(b, m, a, None)  # finalise last column (no right)
        for xd, ct in cdict.items():
            total[xd + ext] += ct
    return {d: ct // 3 for d, ct in total.items()}


def main() -> None:
    print("=== transfer DP vs brute OFG degree distribution ===")
    ok = True
    for m in (2, 3):
        for n in range(2, 7):
            td = Ed_transfer(m, n)
            nbrs = neighbors(m, n)
            bd = Counter()
            for h in enum_height_functions(m, n):
                P, Q = extrema(h, nbrs)
                bd[len(P) + len(Q)] += 1
            match = all(td.get(k, 0) == bd.get(k, 0) for k in set(td) | set(bd))
            ok = ok and match
            print(f"  m={m} n={n}: transfer={dict(sorted(td.items()))} "
                  f"brute={dict(sorted(bd.items()))} [{'OK' if match else 'FAIL'}]")
    print("ALL MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
