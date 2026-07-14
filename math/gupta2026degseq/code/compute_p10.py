#!/usr/bin/env python3
"""Compute p_10(m,n) via numba2 fast DP + exact interpolation.

Reuses interp_pd.py's node/interpolation infrastructure but swaps the DP
backend to Ed_DP_capped_nb2 (fully JIT-compiled, ~60x faster than pure Python
at m=10, unlocks m=13 in ~2 minutes).

Also validates on p_9 first (sanity: fast numba must reproduce p_9 that we
computed via pure-Python fast DP) before spending time on p_10.
"""
import sys
import time

sys.path.insert(0, ".")

import sympy as sp

from fast_dp_nb2 import Ed_DP_capped_nb2

m_sym, n_sym = sp.symbols("m n")
R = sp.Rational

# ---- reuse interp_pd.nodes_for and interpolate, but supply our own sample fn
from interp_pd import nodes_for, interpolate


def sample_Ed_nb(d, verbose=True):
    """Same shape as interp_pd.sample_Ed but uses the numba-jit DP."""
    nd = nodes_for(d)
    by_m = {}
    for (x, y) in nd:
        by_m.setdefault(x, 0)
        by_m[x] = max(by_m[x], y)
    vals = {}
    for x in sorted(by_m):
        nmax = by_m[x]
        if verbose:
            t = time.time()
            print(f"    Ed_DP_capped_nb2(m={x}, nmax={nmax}, d_cap={d}) ...",
                  flush=True)
        series = Ed_DP_capped_nb2(x, nmax, d)     # {n: {d: E_d(x,n)}}
        if verbose:
            print(f"      -> {time.time()-t:.2f}s", flush=True)
        for (px, py) in nd:
            if px == x:
                v = series[py].get(d, 0)
                vals[(px, py)] = v
                vals[(py, px)] = v
    return vals


def compute_pd(d):
    print(f"\n[d={d}] max m = {(3*d-4)//2}, nodes = {len(nodes_for(d))}", flush=True)
    vals = sample_Ed_nb(d)
    return interpolate(d, vals)


if __name__ == "__main__":
    # warm up JIT
    print("Warming numba2 JIT ...", flush=True)
    _ = Ed_DP_capped_nb2(3, 5, 5)

    # ---- 1. Validate: reproduce p_9 exactly ---------------------------------
    print("\n" + "=" * 74)
    print("VALIDATION: reproduce p_9 via numba2 DP")
    print("=" * 74)
    t0 = time.time()
    p9_nb = compute_pd(9)
    print(f"  p_9 computed in {time.time()-t0:.1f}s")

    # Known p_9 from prior compute_p9.py run
    p9_known = (R(1,1260)*(m_sym**7+n_sym**7) - R(1,180)*(m_sym**6+n_sym**6)
                + R(11,15)*(m_sym**5*n_sym+m_sym*n_sym**5)
                + R(97,180)*(m_sym**5+n_sym**5)
                + (m_sym**4*n_sym**2+m_sym**2*n_sym**4)
                + R(7,2)*(m_sym**4*n_sym+m_sym*n_sym**4)
                - R(155,36)*(m_sym**4+n_sym**4)
                + 2*m_sym**3*n_sym**3
                + 99*(m_sym**3*n_sym**2+m_sym**2*n_sym**3)
                + R(449,3)*(m_sym**3*n_sym+m_sym*n_sym**3)
                - R(68852,45)*(m_sym**3+n_sym**3)
                - 400*m_sym**2*n_sym**2
                - R(16657,2)*(m_sym**2*n_sym+m_sym*n_sym**2)
                + R(640274,45)*(m_sym**2+n_sym**2)
                + R(220166,5)*m_sym*n_sym
                + R(4101458,105)*(m_sym+n_sym)
                - 368848)
    diff = sp.expand(p9_nb - p9_known)
    print(f"  numba2 p_9 == known p_9:  {diff == 0}")
    if diff != 0:
        print(f"  !! DIFF: {diff}")
        sys.exit(1)

    # ---- 2. Compute p_10 ----------------------------------------------------
    print("\n" + "=" * 74)
    print("COMPUTE p_10")
    print("=" * 74)
    t0 = time.time()
    p10 = compute_pd(10)
    total = time.time() - t0
    print(f"\n  p_10 computed in {total:.1f}s\n")
    print("  p_10(m,n) =")
    print(f"    {p10}\n")

    # ---- 3. Prediction check ------------------------------------------------
    c_m8  = p10.coeff(m_sym, 8).coeff(n_sym, 0)
    c_m7  = p10.coeff(m_sym, 7).coeff(n_sym, 0)
    c_m6  = p10.coeff(m_sym, 6).coeff(n_sym, 0)   # sub-top: prior 2 predicts 2(7-10)/7! = -6/5040 = -1/840
    c_n8  = p10.coeff(n_sym, 8).coeff(m_sym, 0)

    print("=" * 74)
    print("PREDICTION CHECK  (p_10 vs priors)")
    print("=" * 74)

    def check(label, observed, predicted):
        ok = observed == predicted
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label:<52s} observed={observed}  predicted={predicted}")
        return ok

    check("Prior 1  coeff m^8  (= 4/8!)",     c_m8, R(4, sp.factorial(8)))
    check("Prior 1  coeff n^8  (symmetry)",   c_n8, R(4, sp.factorial(8)))
    check("Prior 2  coeff m^7  (= 2(7-10)/7! = -6/5040 = -1/840)",
                                              c_m7, R(2 * (7 - 10), sp.factorial(7)))
    # Prior 1b: no top-layer mixed monomials
    print("\n  Prior 1b: top-layer (i+j=8, i,j>=1) mixed monomials should be 0:")
    all_zero = True
    for i in range(1, 5):
        j = 8 - i
        c = p10.coeff(m_sym, i).coeff(n_sym, j)
        ok = c == 0
        all_zero &= ok
        mark = "0 ✓" if ok else f"{c} FAIL"
        print(f"     m^{i} n^{j}: {mark}")
    print(f"  Prior 1b {'PASS' if all_zero else 'FAIL'}")

    # sub-sub-top: what's coeff of m^6? (= 4/(d-4)! = 4/6! = ... no that's not a prior)
    # The "16 conjecture" says [m^{d-3}]R_d = 16/(d-3)! for d>=5.
    # Cross-check by computing R_10 = p_10 - single_side and extracting coeff m^7.
    def binom_poly(x, k):
        if k == 0: return sp.Integer(1)
        return sp.expand(sp.prod(x - i for i in range(k)) / sp.factorial(k))
    single_side_10 = sp.expand(4 * binom_poly(m_sym - 2, 8) + 4 * binom_poly(n_sym - 2, 8))
    R10 = sp.expand(p10 - single_side_10)
    c_m7_R = R10.coeff(m_sym, 7).coeff(n_sym, 0)
    check('"16 conjecture" coeff m^7 in R_10  (= 16/7!)',
          c_m7_R, R(16, sp.factorial(7)))

    print("\n  For the record — pure-power coeffs of p_10:")
    for k in range(8, -1, -1):
        c = p10.coeff(m_sym, k).coeff(n_sym, 0)
        print(f"     coeff m^{k}: {c}")
