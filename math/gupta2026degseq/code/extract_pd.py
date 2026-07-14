"""
Extract the exact symmetric polynomial p_d(m,n) on the high region
{m,n >= d-1}, and the boundary-correction stripe for min(m,n) < d-1.

  E_d(m,n) = sum_{a<=b, a+b=d} (2-[a=b]) N_{(a,b)}(m,n)   (colour inversion).

Splits with k=min(a,b)<=3 use the validated numpy counter (fast_split). The
balanced (4,4) split for d=8 needs a k=4 counter (provided here, same envelope
construction); it is the expensive part and is gated separately.

3a: fit p_d on a high-region grid (min>=d-1) with held-out validation; print the
    exact rational symmetric polynomial.
3b: for each stripe value s in {2..d-2}, the per-row count E_d(s, .) is a single
    polynomial in n (per-axis result); the boundary correction is
    B_d^{(s)}(n) = E_d(s,n) - p_d(s,n), an explicit polynomial in n.
"""

from __future__ import annotations

import argparse
import time
from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from fast_split import count_split_fast, _dist_arrays, _num_strict_maxima
from split_diagnostic import (fit_symmetric, residuals, basis_val, sym_basis,
                              fit_poly_1d, degree_of_1d)


# --- k=4 envelope counter (only needed for d=8 balanced split (4,4)) ----------
def count_44(m: int, n: int) -> int:
    """N_{(4,4)}(m,n): 4-cone min-envelopes with exactly 4 minima (the apexes)
    and 4 maxima. By the Envelope Structure Theorem, active+parity => min-set is
    exactly the apexes, so the only check is #maxima==4."""
    dist = _dist_arrays(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    total = 0
    for idx in combinations(range(len(cells)), 4):
        ps = [cells[t] for t in idx]
        darr = [dist[p] for p in ps]
        D = [[int(darr[s][ps[t]]) for t in range(4)] for s in range(4)]
        base0 = darr[0]
        for c1 in range(-D[0][1] + 1, D[0][1]):
            if (c1 - D[0][1]) & 1:
                continue
            e01 = np.minimum(base0, c1 + darr[1])
            for c2 in range(-D[0][2] + 1, D[0][2]):
                if (c2 - D[0][2]) & 1 or abs(c1 - c2) >= D[1][2]:
                    continue
                e012 = np.minimum(e01, c2 + darr[2])
                for c3 in range(-D[0][3] + 1, D[0][3]):
                    if (c3 - D[0][3]) & 1:
                        continue
                    if abs(c1 - c3) >= D[1][3] or abs(c2 - c3) >= D[2][3]:
                        continue
                    E = np.minimum(e012, c3 + darr[3])
                    if _num_strict_maxima(E) == 4:
                        total += 1
    return total


def E_d(d: int, m: int, n: int) -> int:
    total = 0
    for a in range(1, d // 2 + 1):
        b = d - a
        if a > b:
            continue
        fac = 2 - (1 if a == b else 0)
        if a == b == 4:
            total += fac * count_44(m, n)
        else:
            total += fac * count_split_fast(a, b, m, n)
    return total


def fmt_poly(coeffs: Dict[Tuple[int, int], Fraction]) -> str:
    parts = []
    for (i, j), c in sorted(coeffs.items()):
        if c == 0:
            continue
        if i == j == 0:
            parts.append(f"{c}")
        elif i == j:
            parts.append(f"{c}*m^{i}n^{i}")
        else:
            parts.append(f"{c}*(m^{i}n^{j}+m^{j}n^{i})")
    return " + ".join(parts) if parts else "0"


def extract_pd(d: int, lo: int, hi: int, holdout_k: int = 5):
    """Fit p_d on min>=d-1 grid [lo,hi]^2; hold out the holdout_k largest points."""
    tau = d - 1
    data = {}
    t0 = time.time()
    for m in range(lo, hi + 1):
        for n in range(m, hi + 1):
            if min(m, n) < tau:
                continue
            v = E_d(d, m, n)
            data[(m, n)] = v
            data[(n, m)] = v
    pts = [(m, n, v) for (m, n), v in data.items()]
    pts_sorted = sorted(pts, key=lambda t: -(t[0] + t[1]))
    holds = pts_sorted[:holdout_k]
    fitpts = [p for p in pts if p not in holds]
    coeffs = fit_symmetric(fitpts, d - 2)
    if coeffs is None:
        return {"err": f"singular ({len(fitpts)} pts)"}
    bad = residuals(coeffs, fitpts)
    hold_ok = all(sum(c * basis_val(i, j, m, n) for (i, j), c in coeffs.items())
                  == v for (m, n, v) in holds)
    deg = max((i + j for (i, j), c in coeffs.items() if c != 0), default=-1)
    return {"coeffs": coeffs, "n_fit": len(fitpts), "residuals": bad,
            "holds": holds, "hold_ok": hold_ok, "deg": deg,
            "t": time.time() - t0}


def boundary_stripes(d: int, p_d: Dict[Tuple[int, int], Fraction],
                     n_lo: int, n_hi: int):
    """For each stripe s in {2..d-2}: per-row poly E_d(s,.) and correction."""
    tau = d - 1
    out = []
    for s in range(2, tau):  # s < d-1
        row = []
        for n in range(max(n_lo, s), n_hi + 1):
            row.append((n, E_d(d, s, n)))
        cf = fit_poly_1d(row, d)
        deg = degree_of_1d(cf) if cf else -1
        # correction B(n) = E_d(s,n) - p_d(s,n) as a 1-var poly in n
        corr = []
        for (n, v) in row:
            pv = sum(c * basis_val(i, j, s, n) for (i, j), c in p_d.items())
            corr.append((n, v - pv))
        ccf = fit_poly_1d(corr, d)
        cdeg = degree_of_1d(ccf) if ccf else -1
        # residual check: does the fitted row poly reproduce the data?
        rowok = cf is not None and all(
            sum(cf[k] * Fraction(n)**k for k in range(len(cf))) == v
            for (n, v) in row)
        out.append({"s": s, "row_deg": deg, "corr_deg": cdeg,
                    "corr_coeffs": ccf, "row_ok": rowok,
                    "nonzero_corr": any(v != 0 for _, v in corr)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="4,5,6")
    ap.add_argument("--hi", type=int, default=0, help="0 => d-1+6")
    args = ap.parse_args()
    for d in [int(x) for x in args.ds.split(",")]:
        tau = d - 1
        hi = args.hi if args.hi else tau + 6
        print(f"\n========== d={d} (threshold tau={tau}) ==========")
        r = extract_pd(d, tau, hi)
        if "err" in r:
            print(f"  p_{d}: {r['err']}")
            continue
        print(f"  p_{d}(m,n) [fit {r['n_fit']} pts min>={tau}, "
              f"residuals={r['residuals']}, deg={r['deg']}, t={r['t']:.1f}s]:")
        print(f"    {fmt_poly(r['coeffs'])}")
        print(f"  hold-outs ({len(r['holds'])}): "
              f"{'ALL OK' if r['hold_ok'] else 'FAIL'}")
        # boundary stripes
        print(f"  boundary stripes (min(m,n)=s < {tau}):")
        for st in boundary_stripes(d, r["coeffs"], tau, hi + 2):
            cc = fmt_poly_1d(st["corr_coeffs"]) if st["corr_coeffs"] else "?"
            print(f"    s={st['s']}: E_d(s,.) deg={st['row_deg']} "
                  f"[{'ok' if st['row_ok'] else 'FAIL'}]; "
                  f"correction B(n) deg={st['corr_deg']}"
                  f"{'' if st['nonzero_corr'] else '  (ZERO: no correction needed)'}")
            if st["nonzero_corr"]:
                print(f"        B_{d}^(s={st['s']})(n) = {cc}")


def fmt_poly_1d(coeffs) -> str:
    if coeffs is None:
        return "?"
    terms = []
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        terms.append(f"{c}" if k == 0 else f"{c}*n^{k}")
    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    main()
