"""
Per-split degree + H2 (pure-power top) reading.

For each extremum split (a,b) with a+b=d, empirically determine:
  - the total degree of N_{(a,b)}(m,n) as a polynomial in (m,n);
  - WHICH split achieves the maximal degree d-2 (the "dominant" split);
  - whether the top-degree part is a pure power m^{d-2}+n^{d-2} (H2, for d>=5)
    or carries genuine mixed monomials m^a n^b, a,b>=1 (the d=4 exception).

Method.  By the Envelope Structure Theorem a height function with
exactly a minima and b maxima is the lower envelope of a distance-cones; taking
the smaller side k=min(a,b) (colour inversion h<->-h gives N_{(a,b)}=N_{(b,a)}),
we enumerate k-cone min-envelopes:
  - choose k apex cells,
  - offsets c_1=0, c_2..c_k in parity-compatible bounded ranges,
  - keep those whose envelope is a valid Z-height function with min-set EXACTLY
    the k apexes and exactly max(a,b) maxima.
This is polynomial-in-grid for fixed k (the Phase-1a O(d)-variable reduction),
so rows m0 x n with small m0 are cheap even for k=3.

For each split we fit:
  - per-row 1-variable polynomials N_{(a,b)}(m0, .) in n  (Christensen per-m
    polynomiality; these double as the boundary-stripe data);
  - the generic 2-D symmetric polynomial by interpolating rows m0 >= tau_d=d-1,
    reading off total degree and the top-degree monomials (H2 test).

Cross-checked against the brute split-binned enumerator of envelope_structure.py
on every small grid.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Tuple

from envelope_structure import (
    enum_height_functions, extrema, neighbors, dgrid as _dg,
)

Cell = Tuple[int, int]

# On-disk cache for count_split_envelope (reused by the per-split degree
# diagnostic AND hold-out generation; avoids recomputing the
# expensive k=3 envelope sweeps).
_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".split_cache.json")
_CACHE: Dict[str, int] = {}
if os.path.exists(_CACHE_PATH):
    try:
        with open(_CACHE_PATH) as _f:
            _CACHE = json.load(_f)
    except Exception:  # noqa: BLE001
        _CACHE = {}


def _cache_save() -> None:
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_CACHE, f)
    os.replace(tmp, _CACHE_PATH)


def gdist(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def count_split_envelope(a: int, b: int, m: int, n: int) -> int:
    """Cached wrapper around _count_split_envelope."""
    if a > b:
        return count_split_envelope(b, a, m, n)  # N_{(a,b)} = N_{(b,a)}
    key = f"{a},{b},{m},{n}"
    if key in _CACHE:
        return _CACHE[key]
    val = _count_split_envelope(a, b, m, n)
    _CACHE[key] = val
    return val


def _count_split_envelope(a: int, b: int, m: int, n: int) -> int:
    """#height functions on G_{m,n} with exactly a strict minima and b strict
    maxima, via k=a min-cones (caller guarantees a<=b)."""
    k = a  # number of min-cones (a <= b)
    cells = [(i, j) for i in range(m) for j in range(n)]
    nbrs = neighbors(m, n)
    total = 0

    for apex_idx in combinations(range(len(cells)), k):
        apexes = [cells[t] for t in apex_idx]
        D = [[gdist(apexes[s], apexes[t]) for t in range(k)] for s in range(k)]
        dist = [{v: gdist(p, v) for v in cells} for p in apexes]

        # offsets: c[0]=0; c[t] (t>=1) in (-D[0][t], D[0][t]) with parity
        # c[t] ≡ D[0][t] (mod 2). Enumerate recursively, pruning on pairwise
        # active+parity as soon as two offsets are fixed.
        c = [0] * k

        def rec(t: int) -> int:
            if t == k:
                # all offsets fixed & active+parity-consistent; build envelope
                env = {v: min(c[s] + dist[s][v] for s in range(k)) for v in cells}
                # valid Z-height function?
                for v in cells:
                    hv = env[v]
                    for w in nbrs[v]:
                        if abs(hv - env[w]) != 1:
                            return 0
                # min-set exactly the apexes?
                mins = {v for v in cells if all(env[w] > env[v] for w in nbrs[v])}
                if mins != set(apexes):
                    return 0
                maxs = {v for v in cells if all(env[w] < env[v] for w in nbrs[v])}
                return 1 if len(maxs) == b else 0
            sub = 0
            lo, hi = -D[0][t] + 1, D[0][t] - 1
            for ct in range(lo, hi + 1):
                if (ct - D[0][t]) % 2 != 0:
                    continue
                # pairwise active + parity vs already-fixed offsets s<t
                ok = True
                for s in range(t):
                    if abs(c[s] - ct) >= D[s][t]:
                        ok = False
                        break
                    if (c[s] - ct - D[s][t]) % 2 != 0:
                        ok = False
                        break
                if not ok:
                    continue
                c[t] = ct
                sub += rec(t + 1)
            return sub

        total += rec(1)
    return total


def brute_split_bins(m: int, n: int) -> Dict[Tuple[int, int], int]:
    """Ground truth: enumerate ALL height functions, bin by (|P|,|Q|)."""
    nbrs = neighbors(m, n)
    out: Dict[Tuple[int, int], int] = {}
    for h in enum_height_functions(m, n):
        P, Q = extrema(h, nbrs)
        key = (len(P), len(Q))
        out[key] = out.get(key, 0) + 1
    return out


def cross_check(grids: List[Tuple[int, int]]) -> bool:
    print("=== cross-check: envelope split counter vs brute split bins ===")
    ok_all = True
    for (m, n) in grids:
        bins = brute_split_bins(m, n)
        for (a, b), want in sorted(bins.items()):
            if a == 0 or b == 0:
                continue
            got = count_split_envelope(a, b, m, n)
            tag = "OK" if got == want else "FAIL"
            if got != want:
                ok_all = False
                print(f"  {m}x{n} split({a},{b}): envelope={got} brute={want} {tag}")
        print(f"  {m}x{n}: all splits checked "
              f"({'OK' if ok_all else 'MISMATCH'})")
    print()
    return ok_all


# ---------------------------------------------------------------------------
# Symmetric-polynomial fitting (exact rational), generalised from fit_deg6.py
# ---------------------------------------------------------------------------

def sym_basis(max_deg: int) -> List[Tuple[int, int]]:
    """Ordered (i,j), i>=j>=0, i+j<=max_deg. (i,j)->symmetrised monomial."""
    return [(i, j) for s in range(max_deg + 1)
            for i in range(s, (s // 2) - 1, -1) for j in [s - i] if i >= j]


def basis_val(i: int, j: int, m: int, n: int) -> int:
    if i == j:
        return m ** i * n ** i
    return m ** i * n ** j + m ** j * n ** i


def fit_symmetric(points: List[Tuple[int, int, int]], max_deg: int):
    """Exact least-squares (normal equations) symmetric fit; returns coeffs
    dict {(i,j):Fraction} or None if singular."""
    basis = sym_basis(max_deg)
    nb = len(basis)
    if len(points) < nb:
        return None
    A = [[Fraction(basis_val(i, j, m, n)) for (i, j) in basis]
         for (m, n, _) in points]
    bvec = [Fraction(v) for (_, _, v) in points]
    ATA = [[sum(A[r][s] * A[r][t] for r in range(len(A))) for t in range(nb)]
           for s in range(nb)]
    ATb = [sum(A[r][s] * bvec[r] for r in range(len(A))) for s in range(nb)]
    aug = [ATA[s] + [ATb[s]] for s in range(nb)]
    for col in range(nb):
        piv = next((r for r in range(col, nb) if aug[r][col] != 0), None)
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(nb):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [aug[r][t] - f * aug[col][t] for t in range(nb + 1)]
    return {basis[s]: aug[s][-1] for s in range(nb)}


def residuals(coeffs, points) -> int:
    bad = 0
    for (m, n, v) in points:
        pred = sum(c * basis_val(i, j, m, n) for (i, j), c in coeffs.items())
        if pred != v:
            bad += 1
    return bad


def fit_poly_1d(pts: List[Tuple[int, int]], max_deg: int):
    """Exact 1-var fit y=f(x); pts=[(x,y)]; returns coeff list [c0..] or None."""
    nb = max_deg + 1
    if len(pts) < nb:
        return None
    A = [[Fraction(x) ** k for k in range(nb)] for (x, _) in pts]
    bvec = [Fraction(y) for (_, y) in pts]
    ATA = [[sum(A[r][s] * A[r][t] for r in range(len(A))) for t in range(nb)]
           for s in range(nb)]
    ATb = [sum(A[r][s] * bvec[r] for r in range(len(A))) for s in range(nb)]
    aug = [ATA[s] + [ATb[s]] for s in range(nb)]
    for col in range(nb):
        piv = next((r for r in range(col, nb) if aug[r][col] != 0), None)
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(nb):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [aug[r][t] - f * aug[col][t] for t in range(nb + 1)]
    return [aug[s][-1] for s in range(nb)]


def degree_of_1d(coeffs) -> int:
    for k in range(len(coeffs) - 1, -1, -1):
        if coeffs[k] != 0:
            return k
    return -1


def top_monomials(coeffs, deg: int):
    """Return the (i,j)->coeff entries at total degree == deg (nonzero)."""
    return {(i, j): c for (i, j), c in coeffs.items() if i + j == deg and c != 0}


# ---------------------------------------------------------------------------
# Focused per-split degree + H2 reading
# ---------------------------------------------------------------------------

def row_sweep(a: int, b: int, m0: int, n_lo: int, n_hi: int):
    """N_{(a,b)}(m0, n) for n in [n_lo, n_hi]; returns [(n, value)]."""
    out = [(n, count_split_envelope(a, b, m0, n)) for n in range(n_lo, n_hi + 1)]
    _cache_save()
    return out


def diagnose_split(a: int, b: int, rows: Dict[int, Tuple[int, int]],
                   gen_thresh: int, max_deg: int):
    """rows: m0 -> (n_lo, n_hi). Fit each row's 1-var poly in n; assemble the
    generic 2-D symmetric poly from rows m0>=gen_thresh. Print findings."""
    d = a + b
    print(f"--- split ({a},{b})  [d={d}, k=min={min(a,b)}] ---")
    data2d: List[Tuple[int, int, int]] = []
    t0 = time.time()
    for m0 in sorted(rows):
        lo, hi = rows[m0]
        rs = row_sweep(a, b, m0, lo, hi)
        cf = fit_poly_1d(rs, max_deg)
        if cf is None:
            print(f"    row m0={m0}: insufficient pts")
        else:
            deg = degree_of_1d(cf)
            lead = cf[deg] if deg >= 0 else 0
            tag = "GENERIC" if m0 >= gen_thresh else "stripe"
            print(f"    row m0={m0} [{tag}]: deg_n={deg}  lead={lead}  "
                  f"(n={lo}..{hi})")
        for (n, v) in rs:
            data2d.append((m0, n, v))
            if m0 != n:
                data2d.append((n, m0, v))  # symmetry: N is symmetric in (m,n)
    # generic 2-D fit on points with min(m,n) >= gen_thresh
    gen_pts = [(m, n, v) for (m, n, v) in data2d
               if min(m, n) >= gen_thresh]
    # dedup
    seen = {}
    for (m, n, v) in gen_pts:
        seen[(m, n)] = v
    gen_pts = [(m, n, v) for (m, n), v in seen.items()]
    coeffs = fit_symmetric(gen_pts, max_deg)
    if coeffs is None:
        print(f"    [2-D generic fit: insufficient/again singular "
              f"({len(gen_pts)} pts)]  t={time.time()-t0:.1f}s")
        print()
        return
    bad = residuals(coeffs, gen_pts)
    deg = max((i + j for (i, j), c in coeffs.items() if c != 0), default=-1)
    tops = top_monomials(coeffs, deg)
    pure = set(tops) <= {(deg, 0)}  # only m^deg+n^deg term
    print(f"    2-D generic (min>= {gen_thresh}, {len(gen_pts)} pts, "
          f"residuals={bad}): TOTAL DEGREE={deg}")
    tops_str = ", ".join(
        (f"{c}*m^{i}n^{j}+sym" if i != j else f"{c}*m^{i}n^{i}")
        for (i, j), c in sorted(tops.items()))
    print(f"      top-degree monomials: {tops_str}")
    print(f"      H2 pure-power top (m^{deg}+n^{deg} only)?  "
          f"{'YES' if pure else 'NO (mixed)'}")
    print(f"    [t={time.time()-t0:.1f}s]")
    print()
    return coeffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-grids", default="3:3,3:4,4:4,3:5,4:5")
    ap.add_argument("--only-cross", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="run the focused per-split degree+H2 diagnostic")
    args = ap.parse_args()

    if not args.diagnose:
        grids = [tuple(map(int, t.split(":")))
                 for t in args.cross_grids.split(",")]
        t0 = time.time()
        ok = cross_check(grids)
        print(f"[cross-check {'PASSED' if ok else 'FAILED'} "
              f"in {time.time()-t0:.1f}s]")
        return

    print("=== Per-split degree + H2 diagnostic ===\n")
    # d=4: (2,2). tau=3.
    diagnose_split(2, 2, {3: (3, 12), 4: (4, 12), 5: (5, 12)},
                   gen_thresh=3, max_deg=4)
    # d=5: (2,3). tau=4.  (1,4) is a cone: (m-2)(n-2), degree 2 (analytic).
    diagnose_split(2, 3, {4: (4, 12), 5: (5, 12), 6: (6, 12)},
                   gen_thresh=4, max_deg=4)
    # d=6: (2,4) and (3,3). tau=5.
    diagnose_split(2, 4, {5: (5, 12), 6: (6, 12), 7: (7, 12)},
                   gen_thresh=5, max_deg=5)
    diagnose_split(3, 3, {5: (5, 11), 6: (6, 11), 7: (7, 10)},
                   gen_thresh=5, max_deg=5)


if __name__ == "__main__":
    main()
