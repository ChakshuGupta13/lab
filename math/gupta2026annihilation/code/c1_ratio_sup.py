#!/usr/bin/env python3
"""
Test the candidate UNIFORM tight bound  a/W <= (Delta+1)/2  for ALL Delta>=1,
versus the paper's  a <= (Delta-1)W  (proved only for Delta>=3).

a, W defined exactly as in c1_delta4.py:
  ascending degree sequence d_1<=...<=d_n, m = (sum d_i)/2,
  a = largest j with d_1+...+d_j <= m,
  W = sum_i 1/(d_i+1).

For each maximum degree Delta, enumerate ALL graphic degree sequences with
entries in {1,...,Delta}, max entry exactly Delta, n<=NMAX, and record:
  rho(Delta) = max a/W,  the maximizing sequence, and whether it ever
  exceeds (Delta+1)/2.

Usage: python c1_ratio_sup.py [NMAX] [DMAX]   (defaults 16, 8)
"""
import sys
from fractions import Fraction as F
from itertools import combinations_with_replacement


def annihilation(asc, m):
    tot = a = 0
    for x in asc:
        tot += x
        if tot <= m:
            a += 1
        else:
            break
    return a


def erdos_gallai_ok(asc):
    d = sorted(asc, reverse=True)
    n = len(d)
    if sum(d) % 2:
        return False
    pref = 0
    for k in range(1, n + 1):
        pref += d[k - 1]
        rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
        if pref > rhs:
            return False
    return True


def gen_sequences(n, D):
    """All nondecreasing degree sequences length n, entries in 1..D, max == D."""
    for combo in combinations_with_replacement(range(1, D + 1), n):
        if combo[-1] != D:        # max degree must be exactly D
            continue
        yield combo


def main(nmax, dmax):
    print(f"NMAX={nmax}  DMAX={dmax}")
    print(f"{'D':>3} {'rho=max a/W':>14} {'(D+1)/2':>9} {'D-1':>5} "
          f"{'rho<=(D+1)/2?':>14} {'argmax seq (n; counts)':>34}")
    overall_violation = False
    for D in range(1, dmax + 1):
        best = F(0)
        best_seq = None
        cap = F(D + 1, 2)
        viol = False
        for n in range(2, nmax + 1):
            for asc in gen_sequences(n, D):
                s = sum(asc)
                if s % 2:
                    continue
                if not erdos_gallai_ok(asc):
                    continue
                m = s // 2
                a = annihilation(asc, m)
                W = sum(F(1, d + 1) for d in asc)
                r = F(a) / W
                if r > best:
                    best = r
                    best_seq = (n, asc)
                if r > cap:
                    viol = True
                    overall_violation = True
        # compact count description of the argmax
        n_bs, seq = best_seq
        counts = {}
        for d in seq:
            counts[d] = counts.get(d, 0) + 1
        cdesc = f"n={n_bs}; " + ",".join(f"{k}^{v}" for k, v in sorted(counts.items()))
        flag = "YES" if best <= cap else "NO  <-- VIOLATION"
        print(f"{D:>3} {str(best):>14} {str(cap):>9} {D-1:>5} {flag:>14} {cdesc:>34}")
    print()
    if overall_violation:
        print("RESULT: (Delta+1)/2 is NOT a uniform upper bound -- found a/W > (Delta+1)/2.")
    else:
        print("RESULT: a/W <= (Delta+1)/2 held for EVERY graphic sequence tested.")
        print("        Tight: Delta-regular on even n attains it (a/W = (Delta+1)/2).")


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    dmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(nmax, dmax)
