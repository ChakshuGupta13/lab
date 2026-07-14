#!/usr/bin/env python3
"""
Master verification for "One construction for the Miura-ori flip-graph degree
sequence" (arXiv:2607.05567).

Reproduces the paper's headline numerical claims and prints PASS/FAIL for each.
Runs in a few minutes (degrees d <= 6). The heavier per-claim scripts
(envelope_structure.py, maxima_criterion.py, transfer_check.py,
boundary_lemma_verify.py, relation_B_complete_verify.py, column_dp.py,
verify_separable.py) run their own checks standalone; the p_9/p_10 closed forms
use the numba transfer-matrix chain (compute_p10.py).

    python3 verify_claims.py
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial

from extract_pd import E_d, extract_pd, boundary_stripes
from column_dp import Ed_DP

# Baxter numbers Bax(k) (OEIS A001181): 1, 2, 6, 22, 92, ...
BAXTER = {1: 1, 2: 2, 3: 6, 4: 22, 5: 92}


def _report(name: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    return ok


def two_engine_Ed(dmax: int = 7, rng: int = 6) -> bool:
    """E_d(m,n) from the envelope split-counter equals E_d from the independent
    column transfer-matrix DP, for every degree d and grid in range.  The two
    routes (Theorem thm:envelope; Lemma lem:ratGF) never disagree."""
    for m in range(2, rng):
        dp = Ed_DP(m, rng)
        for n in range(2, rng):
            for d in range(2, dmax + 1):
                if E_d(d, m, n) != dp[n].get(d, 0):
                    return False
    return True


def symmetry(dmax: int = 7, rng: int = 7) -> bool:
    """The degree sequence is symmetric in the two axes: E_d(m,n) = E_d(n,m)."""
    return all(
        E_d(d, m, n) == E_d(d, n, m)
        for m in range(2, rng)
        for n in range(2, rng)
        for d in range(2, dmax + 1)
    )


def m2n_vertex_count(rng: int = 7) -> bool:
    """OFG(M_{2,n}) has 2*3^(n-1) vertices (Christensen-Hull et al. 2025); the
    degree counts sum to this, reproducing the known m=2 result."""
    return all(
        sum(Ed_DP(2, n)[n].values()) == 2 * 3 ** (n - 1) for n in range(2, rng)
    )


def closed_forms(ds=(4, 5, 6, 7)) -> bool:
    """On the high region m,n >= d-1 the count is a single symmetric polynomial
    p_d of total degree d-2 (Theorem thm:poly), with top-degree part
    4/(d-2)! (m^{d-2}+n^{d-2}) for d >= 5 (Proposition prop:leading).  One step
    below threshold (s = d-2) the boundary correction is linear with leading
    coefficient -4*Bax(d-3) (Conjecture conj:baxter)."""
    ok = True
    for d in ds:
        r = extract_pd(d, d - 1, d - 1 + 6)
        if "err" in r:
            ok &= _report(f"p_{d} closed form", False)
            continue
        fit_ok = r["deg"] == d - 2 and not r["residuals"] and r["hold_ok"]
        ok &= _report(
            f"p_{d}: symmetric polynomial, total degree {d - 2}, residual 0, hold-out OK",
            fit_ok,
        )
        if d >= 5:
            top = r["coeffs"].get((d - 2, 0), Fraction(0))
            want = Fraction(4, factorial(d - 2))
            ok &= _report(
                f"p_{d}: top-degree coefficient = 4/(d-2)! = {want}", top == want
            )
        stripe = [x for x in boundary_stripes(d, r["coeffs"], d - 1, d - 1 + 8)
                  if x["s"] == d - 2]
        if stripe and (d - 3) in BAXTER:
            cc = stripe[0]["corr_coeffs"]
            lead = cc[1] if len(cc) > 1 else Fraction(0)
            want = Fraction(-4 * BAXTER[d - 3])
            ok &= _report(
                f"p_{d}: boundary correction at s=d-2 has leading coeff -4*Bax(d-3) = {want}",
                lead == want,
            )
    return ok


def main() -> int:
    print("=" * 70)
    print("Verifying the Miura-ori flip-graph degree-sequence claims")
    print("=" * 70)
    print("\n--- structural / cross-engine ---")
    results = [
        ("E_d via two independent engines (envelope vs transfer-matrix DP) agree",
         two_engine_Ed(dmax=6)),
        ("degree sequence is axis-symmetric  E_d(m,n) = E_d(n,m)", symmetry(dmax=6)),
        ("M_{2,n} vertex count = 2*3^(n-1)  (Christensen-Hull et al.)",
         m2n_vertex_count()),
    ]
    ok = all(_report(name, val) for name, val in results)

    print("\n--- closed forms, leading coefficients, boundary corrections ---")
    ok &= closed_forms(ds=(4, 5, 6))

    print()
    print("=" * 70)
    print("ALL CLAIMS VERIFIED" if ok else "SOME CHECKS FAILED")
    print("Heavier checks (d=7 balanced split, d=8 leading term, p_9/p_10 via")
    print("the numba transfer matrix) run standalone: confirm_d7.py,")
    print("d8_leading_check.py, compute_p10.py.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
