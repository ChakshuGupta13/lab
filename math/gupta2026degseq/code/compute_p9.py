#!/usr/bin/env python3
"""Compute p_9(m,n) via the fast capped DP + exact interpolation, then check
the three concrete predictions from Path A:

  Prior 1  (paper's Prop:singleside): coeff m^7 = 1/1260 = 4/7!
  Prior 2  (period-1 c=1 EGF, R_d top-pure): coeff m^6 = -1/180
  Weak 2b  (top-mixed linear (88-2d)/(d-4)!):  coeff m^5 n = 7/12

Sweeps: m=8 (nmax=14), m=9 (nmax=13), m=10 (nmax=12), m=11 (nmax=11).
"""
import sys
import time
sys.path.insert(0, ".")

import sympy as sp
from interp_pd import nodes_for, sample_Ed, interpolate

m, n = sp.symbols("m n")
R = sp.Rational

d = 9
t0 = time.time()
print(f"[d={d}] max m = {(3*d-4)//2}, nodes = {len(nodes_for(d))}", flush=True)
vals = sample_Ed(d)
p9 = interpolate(d, vals)
dt = time.time() - t0
print(f"\n  p_9 computed in {dt:.1f}s")
print(f"\n  p_9(m,n) =")
print(f"    {p9}\n")

# ---- prediction check --------------------------------------------------------
c_m7  = p9.coeff(m, 7).coeff(n, 0)
c_m6  = p9.coeff(m, 6).coeff(n, 0)
c_m5n = p9.coeff(m, 5).coeff(n, 1)
c_m5  = p9.coeff(m, 5).coeff(n, 0)
c_n7  = p9.coeff(n, 7).coeff(m, 0)

# also check no mixed at top (Prior 1b): m^7 n, m^6 n^2, m^5 n^3, m^4 n^4, ..., top layer only
top_layer_mixed = [(i, 7 - i) for i in range(1, 4)]  # (1,6), (2,5), (3,4)  by symmetry check other halves too

print("=" * 78)
print("PREDICTION CHECK  (Path A priors vs p_9 observation)")
print("=" * 78)

def check(label, observed, predicted):
    ok = observed == predicted
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:<50s} observed={observed}  predicted={predicted}")
    return ok

check("Prior 1  coeff m^7  (=coeff n^7 by symmetry)", c_m7, R(4, sp.factorial(7)))
check("Prior 1  coeff n^7  (symmetry sanity)",        c_n7, R(4, sp.factorial(7)))
check("Prior 2  coeff m^6  (=coeff n^6 by symmetry)", c_m6, R(-1, 180))
check("Weak 2b  coeff m^5 n (=orbit (1,5))",          c_m5n, R(7, 12))

# Prior 1b: no top-layer mixed monomials in p_9 (total degree 7 mixed = 0)
print("\n  Prior 1b: top-layer mixed coeffs (total degree 7, i+j=7 with i,j>=1) should be 0:")
all_zero = True
for i in range(1, 4):
    j = 7 - i
    coef = p9.coeff(m, i).coeff(n, j)
    ok = coef == 0
    all_zero &= ok
    mark = "  0 ✓" if ok else f"  {coef} FAIL"
    print(f"     m^{i} n^{j}: {mark}")
print(f"  Prior 1b {'PASS' if all_zero else 'FAIL'}")

# c=1 pure Prior 2 sanity: also check other pure powers just for record
print("\n  For the record — other pure-power coeffs of p_9:")
for k in range(7, -1, -1):
    c = p9.coeff(m, k).coeff(n, 0)
    print(f"     coeff m^{k}: {c}")
