#!/usr/bin/env python3
"""
TxGraffiti C1 -- COMPLETE the last case Delta=3 via an elementary regime proof.

The Delta=3 case of C1 reduces (via alpha>=R Favaron, alpha>=W Caro-Wei) to the
pure degree-sequence inequality
    L'(3):  a <= 2W = n1 + (2/3) n2 + (1/2) n3
for any multiset of degrees in {1,2,3} with integer m = (n1+2n2+3n3)/2 (i.e. n1+n3 even),
where a = annihilation number = max{ j : sum of j smallest degrees <= m } and
n_k = #{vertices of degree k}.  (No graphicality needed -- pure arithmetic.)

CLAIMED CLOSED FORM for a (ascending degrees: n1 ones, n2 twos, n3 threes):
  Regime I   (n1 > 2n2+3n3, i.e. m < n1):           a = floor(m)
  Regime II  (n1 <= m < n1+2n2, i.e. 2n2+3n3>=n1 and 3n3 < n1+2n2):
                                                    a = n1 + floor((2n2+3n3-n1)/4)
  Regime III (m >= n1+2n2, i.e. 3n3 >= n1+2n2):     a = n1 + n2 + floor((3n3-n1-2n2)/6)

This script verifies, over ALL (n1,n2,n3) with n1+n3 even and n=n1+n2+n3 <= NMAX:
  (1) the closed-form a matches the directly-computed annihilation number a;
  (2) a <= 2W holds exactly (Fraction), and locates equality cases;
  (3) the regime boundaries partition correctly.

Also (4) sanity: confirm a <= 2W fails for SOME Delta=2 multiset (degrees in {1,2}),
showing the cubic-cap argument is Delta=3-specific.

Usage: python c1_delta3_proof.py [NMAX]   (default 200)
"""
import sys
from fractions import Fraction as F


def annihilation_counts(n1, n2, n3):
    """Direct annihilation: a = max j with (sum of j smallest degrees) <= m."""
    degs = [1] * n1 + [2] * n2 + [3] * n3
    m = (n1 + 2 * n2 + 3 * n3) // 2
    tot = a = 0
    for d in degs:                      # already ascending
        tot += d
        if tot <= m:
            a += 1
        else:
            break
    return a, m


def a_closed_form(n1, n2, n3):
    m = (n1 + 2 * n2 + 3 * n3) / 2      # may be x.0; use integer m below
    M = (n1 + 2 * n2 + 3 * n3) // 2
    if n1 > 2 * n2 + 3 * n3:            # Regime I: m < n1
        return M, "I"
    if 3 * n3 < n1 + 2 * n2:            # Regime II: n1 <= m < n1+2n2
        return n1 + (2 * n2 + 3 * n3 - n1) // 4, "II"
    return n1 + n2 + (3 * n3 - n1 - 2 * n2) // 6, "III"   # Regime III: m >= n1+2n2


def main(nmax):
    bad_formula = []
    viol = []
    eq_cases = []
    regime_count = {"I": 0, "II": 0, "III": 0}
    checked = 0

    for n3 in range(0, nmax + 1):
        for n2 in range(0, nmax + 1 - n3):
            for n1 in range(0, nmax + 1 - n3 - n2):
                if n1 + n2 + n3 == 0:
                    continue
                if (n1 + n3) % 2 != 0:        # need n1+2n2+3n3 even => m integer
                    continue
                if n3 == 0 and n2 == 0 and n1 == 0:
                    continue
                checked += 1
                a_dir, m = annihilation_counts(n1, n2, n3)
                a_cf, reg = a_closed_form(n1, n2, n3)
                regime_count[reg] += 1
                if a_dir != a_cf:
                    if len(bad_formula) < 20:
                        bad_formula.append((n1, n2, n3, a_dir, a_cf, reg, m))
                W2 = F(n1) + F(2 * n2, 3) + F(n3, 2)        # 2W
                if a_dir > W2:
                    if len(viol) < 20:
                        viol.append((n1, n2, n3, a_dir, float(W2)))
                elif a_dir == W2:
                    if len(eq_cases) < 30:
                        eq_cases.append((n1, n2, n3, a_dir))

    print(f"Checked {checked} multisets (degrees in {{1,2,3}}, n1+n3 even, n<= {nmax}).")
    print(f"Regime partition: I={regime_count['I']}  II={regime_count['II']}  III={regime_count['III']}")
    print(f"Closed-form a mismatches: {len(bad_formula)}")
    for e in bad_formula:
        print("   MISMATCH (n1,n2,n3,a_dir,a_cf,reg,m):", e)
    print(f"L'(3) a<=2W violations: {len(viol)}")
    for e in viol:
        print("   VIOLATION:", e)
    print(f"Equality a==2W cases (sample): {len(eq_cases)} e.g. {eq_cases[:12]}")
    print("   (expected: n1=0, i.e. cubic n3 even with n2=0, and {2,3}-mixtures with n1=0)")

    # (4) Delta=2 sanity: degrees in {1,2}, show vehicle a<=(Delta-1)W = W
    #     can FAIL (=> the vehicle's Delta>=3 restriction is necessary, and
    #     Delta=2 is handled separately in Theorem 2 via a=alpha).
    print("\nDelta=2 sanity (degrees in {1,2}): a vs W where W=n1/2+n2/3"
          " (vehicle target at Delta=2 is a<=(Delta-1)W = W)")
    d2fail = 0
    for n2 in range(0, 40):
        for n1 in range(0, 40):
            if n1 + n2 == 0 or n1 % 2 != 0:    # m=(n1+2n2)/2 integer => n1 even
                continue
            degs = [1] * n1 + [2] * n2
            m = (n1 + 2 * n2) // 2
            tot = a = 0
            for d in degs:
                tot += d
                if tot <= m: a += 1
                else: break
            W = F(n1, 2) + F(n2, 3)
            if a > W:
                d2fail += 1
    print(f"   Delta=2 multisets with a>W: {d2fail} (nonzero => vehicle"
          f" inequality fails at Delta=2, handled separately by a=alpha)")


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(nmax)
