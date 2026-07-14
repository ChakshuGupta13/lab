"""
d=7 confirmation: the BALANCED split (3,4) reaches total degree 5 = d-2 with
pure-power top C(m^5+n^5), and uniquely dominates E_7.

Confirms the structural law at d=7:
  - splits at d=7 with a<=b: (2,5) [confirmed degree 3, subdominant] and the
    balanced (3,4). (1,6) is impossible (a cone has degree 1+kappa, kappa<=4).
  - so E_7 = 2 N(2,5) + 2 N(3,4); the leading degree must come from (3,4) alone.

Uses the numpy-fast, brute-validated counter (fast_split.count_split_fast).
"""

from __future__ import annotations

import time
from fractions import Fraction

from fast_split import count_split_fast, _dist_arrays
from split_diagnostic import fit_symmetric, residuals, top_monomials, fit_poly_1d, degree_of_1d


def n34(m: int, n: int, cache: dict) -> int:
    key = (min(m, n), max(m, n))
    if key not in cache:
        cache[key] = count_split_fast(3, 4, key[0], key[1])
    return cache[key]


def main() -> None:
    tau = 6  # d-1
    cache: dict = {}

    # Grid set: rows m0 = 6..9, n up to 14, min(m,n) >= 6.
    rows = {6: range(6, 15), 7: range(7, 15), 8: range(8, 15), 9: range(9, 14)}

    print("=== d=7 balanced (3,4) confirmation ===")
    t0 = time.time()

    # per-row 1-var degree in n (should be d-2 = 5)
    pts = []
    for m0 in sorted(rows):
        rs = [(n, n34(m0, n, cache)) for n in rows[m0]]
        cf = fit_poly_1d([(n, v) for (n, v) in rs], 6)
        deg = degree_of_1d(cf) if cf else -1
        lead = cf[deg] if cf and deg >= 0 else None
        print(f"  row m0={m0}: deg_n={deg}  lead={lead}  "
              f"(n={min(rows[m0])}..{max(rows[m0])})  [{time.time()-t0:.1f}s]")
        for (n, v) in rs:
            pts.append((m0, n, v))
            if m0 != n:
                pts.append((n, m0, v))

    # dedup (m,n)
    seen = {}
    for (m, n, v) in pts:
        seen[(m, n)] = v
    allpts = [(m, n, v) for (m, n), v in seen.items()]

    # hold out 5 high points
    holdouts = [(6, 14), (7, 13), (8, 12), (9, 11), (9, 13)]
    holdouts = [(m, n) for (m, n) in holdouts if (m, n) in seen]
    fit_pts = [(m, n, v) for (m, n, v) in allpts if (m, n) not in holdouts]

    coeffs = fit_symmetric(fit_pts, max_deg=6)  # generous basis; degree FOUND
    if coeffs is None:
        print("  [fit singular]")
        return
    bad = residuals(coeffs, fit_pts)
    deg = max((i + j for (i, j), c in coeffs.items() if c != 0), default=-1)
    tops = top_monomials(coeffs, deg)
    pure = set(tops) <= {(deg, 0)}
    print()
    print(f"  N(3,4): 2-D fit on {len(fit_pts)} pts (min>={tau}), "
          f"deg-6 basis, residuals={bad}")
    print(f"    TOTAL DEGREE = {deg}   (expect 5 = d-2)")
    tstr = ", ".join((f"{c}*(m^{i}n^{j}+sym)" if i != j else f"{c}*m^{i}n^{i}")
                     for (i, j), c in sorted(tops.items()))
    print(f"    top-degree monomials: {tstr}")
    print(f"    H2 pure-power top (m^5+n^5 only)?  {'YES' if pure else 'NO (mixed)'}")

    # hold-out validation
    print(f"  hold-out validation ({len(holdouts)} pts held out before fit):")
    allok = True
    for (m, n) in holdouts:
        pred = sum(c * (m**i * n**j + (m**j * n**i if i != j else 0))
                   for (i, j), c in coeffs.items())
        # basis_val inline
        pred = sum(c * (m**i * n**i if i == j else m**i*n**j + m**j*n**i)
                   for (i, j), c in coeffs.items())
        act = seen[(m, n)]
        ok = (pred == act)
        allok = allok and ok
        print(f"    ({m},{n}): actual={act}  pred={pred}  [{'OK' if ok else 'DIFF'}]")
    print(f"  hold-outs: {'ALL OK' if allok else 'MISMATCH'}")
    print(f"[total {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
