"""Grounding checks for the strengthened Section 9 validation sentence and the
d=8 leading-behaviour remark (both in docs/origami-flipgraph-sequel/main.tex).

Two independent recomputations of E_d(m,n) are used:
  * the envelope counter (extract_pd.E_d), which produces p_d, and
  * the column transfer matrix (transfer_matrix.vpoly, = Lemma ratGF),
which is independent of the envelope construction.

CHECK 1 (Section 9):  the transfer matrix reproduces every p_d (d=2..7) on all
                      grids up to 9x9.
CHECK 2 (Remark):     at the first open degree d=8, E_8(m,.) has degree d-2=6 in
                      n with leading coefficient C(8)=4/6!=1/180 and vanishing
                      m^5 n coefficient, for m=7 and m=8.

Runtime ~5 min (the 9x9 transfer-matrix evaluation dominates).
"""
from __future__ import annotations

import sys
from fractions import Fraction as F
from math import factorial

sys.path.insert(0, ".")
from transfer_matrix import vpoly


# --- paper's closed forms p_2..p_7 (Section 9) ------------------------------
def p2(m, n): return F(4)
def p3(m, n): return 4 * (m + n) - 16
def p4(m, n): return 2 * (m**2 + n**2) + 6 * m * n - 10 * (m + n) - 4
def p5(m, n): return F(2, 3) * (m**3 + n**3) + 2 * (m**2 + n**2) + 50 * m * n - F(392, 3) * (m + n) + 264
def p6(m, n): return (F(1, 6) * (m**4 + n**4) + F(1, 3) * (m**3 + n**3) + 38 * (m**2 * n + m * n**2)
                      - F(229, 6) * (m**2 + n**2) - 272 * m * n - F(103, 3) * (m + n) + 1176)
def p7(m, n): return (F(1, 30) * (m**5 + n**5) + F(37, 3) * (m**3 * n + m * n**3) + F(7, 6) * (m**3 + n**3)
                      + 18 * m**2 * n**2 + 25 * (m**2 * n + m * n**2) - 496 * (m**2 + n**2)
                      - F(3818, 3) * m * n + F(14354, 5) * (m + n) - 904)
PD = {2: p2, 3: p3, 4: p4, 5: p5, 6: p6, 7: p7}


def fit6(pairs):
    """Exact degree-6 interpolation of 7 (n, value) pairs; returns coeffs a_0..a_6."""
    ns = [n for n, _ in pairs][:7]
    A = [[F(ns[i])**j for j in range(7)] + [F(pairs[i][1])] for i in range(7)]
    for c in range(7):
        piv = next(r for r in range(c, 7) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        inv = A[c][c]
        A[c] = [x / inv for x in A[c]]
        for r in range(7):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [A[r][k] - f * A[c][k] for k in range(8)]
    return [A[i][7] for i in range(7)]


def check1_section9(grids=((5, 5), (6, 6), (7, 7), (8, 8), (9, 9))):
    print("CHECK 1 (Section 9): transfer matrix reproduces p_2..p_7 up to 9x9")
    ok = True
    for (m, n) in grids:
        P = vpoly(m, n)
        bad = [d for d in range(2, 8) if P.get(d, 0) != int(PD[d](m, n))]
        print(f"  {m}x{n}: {'OK' if not bad else 'FAIL ' + str(bad)}")
        ok = ok and not bad
    return ok


def check2_remark():
    print("CHECK 2 (Remark): d=8 leading behaviour at m=7,8")
    C8 = F(4, factorial(6))                      # = 1/180
    rows = {}
    for m in (7, 8):
        rows[m] = fit6([(n, vpoly(m, n).get(8, 0)) for n in range(m, m + 7)])
    a6 = {m: rows[m][6] for m in (7, 8)}         # coeff of n^6 (pure power)
    a5 = {m: rows[m][5] for m in (7, 8)}         # coeff of n^5 = c_05 + c_15*m
    c15 = a5[8] - a5[7]                           # coeff of mixed monomial m^5 n
    print(f"  leading n^6 coeff:  m=7 {a6[7]}  m=8 {a6[8]}   C(8)=1/180: "
          f"{a6[7] == C8 and a6[8] == C8}")
    print(f"  mixed m^5 n coeff c_15 = a5(8)-a5(7) = {c15}  vanishes: {c15 == 0}")
    return a6[7] == C8 and a6[8] == C8 and c15 == 0


if __name__ == "__main__":
    ok1 = check1_section9()
    ok2 = check2_remark()
    print(f"\nALL CHECKS {'PASS' if ok1 and ok2 else 'FAIL'}")
    sys.exit(0 if ok1 and ok2 else 1)
