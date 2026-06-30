#!/usr/bin/env python3
"""
VERIFY the proposed proof of  a <= ((Delta+1)/2) W  step by step.

Proof (Boppana-style, transplanted to the annihilation head):
  H = the a smallest-degree vertices (the annihilation head).
  (1) sum_{H} d <= m   [def of a];  2m = sum_all d  =>  sum_H d <= sum_T d.
  (2) tail degrees <= Delta  =>  sum_T d <= Delta (n-a)  =>  sum_H d <= Delta(n-a).
  (3) H_k = #head vertices of degree k;  sum_k H_k = a,  sum_k k H_k = sum_H d.
      so  n-a >= (1/Delta) sum_k k H_k.
  (4) W = sum_k H_k/(k+1) + sum_tail 1/(d+1)
        >= sum_k H_k/(k+1) + (n-a)/(Delta+1)
        >= sum_k H_k [ 1/(k+1) + k/(Delta(Delta+1)) ].
  (5) integer k in [0,Delta]:  1/(k+1) + k/(Delta(Delta+1)) >= 2/(Delta+1),
      equality at k = Delta-1 and k = Delta.
  (6) => W >= (2/(Delta+1)) a  =>  a <= ((Delta+1)/2) W.

This script:
  A. checks step (5) pointwise inequality, exactly, for Delta=1..60 and all
     integer k in [0,Delta]; reports the min and where equality holds.
  B. on ALL graphic degree sequences (Erdos-Gallai) with max degree Delta,
     n<=NMAX, verifies the FULL chain link by link with exact Fractions:
       - sum_H d <= m
       - sum_H d <= Delta(n-a)
       - the W lower bound L := sum_k H_k/(k+1) + (n-a)/(Delta+1) satisfies
         W >= L  AND  L >= (2/(Delta+1)) a  AND  a <= ((Delta+1)/2) W.
     Any failure printed.

Usage: python c1_ratio_proof_check.py [NMAX] [DMAX]   (defaults 16, 8)
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


def check_pointwise(dmax):
    """Step (5): 1/(k+1) + k/(D(D+1)) >= 2/(D+1) for integer k in [0,D]."""
    worst = None
    eqpts = {}
    for D in range(1, dmax + 1):
        target = F(2, D + 1)
        eq = []
        for k in range(0, D + 1):
            val = F(1, k + 1) + F(k, D * (D + 1))
            if val < target:
                print(f"  POINTWISE FAIL D={D} k={k}: {val} < {target}")
                return False
            if val == target:
                eq.append(k)
            slack = val - target
            if worst is None or slack < worst[0]:
                worst = (slack, D, k)
        eqpts[D] = eq
    print(f"  pointwise OK for D=1..{dmax}; min slack {float(worst[0]):.3e} "
          f"at D={worst[1]},k={worst[2]}")
    # report equality points for a few D
    for D in [1, 2, 3, 4, 5, 8, 20]:
        if D in eqpts:
            print(f"    D={D}: equality at k={eqpts[D]} (expect [D-1, D] = "
                  f"[{D-1},{D}])")
    return True


def gen_sequences(n, D):
    for combo in combinations_with_replacement(range(1, D + 1), n):
        if combo[-1] != D:
            continue
        yield combo


def check_chain(nmax, dmax):
    total = fails = 0
    tight = 0
    for D in range(1, dmax + 1):
        for n in range(2, nmax + 1):
            for asc in gen_sequences(n, D):
                s = sum(asc)
                if s % 2 or not erdos_gallai_ok(asc):
                    continue
                total += 1
                m = s // 2
                a = annihilation(asc, m)
                W = sum(F(1, d + 1) for d in asc)
                head = asc[:a]
                tail = asc[a:]
                sumH = sum(head)
                # link (1)
                ok1 = sumH <= m
                # link (2)
                ok2 = sumH <= D * (n - a)
                # head degree-class counts
                Hk = {}
                for d in head:
                    Hk[d] = Hk.get(d, 0) + 1
                assert sum(Hk.values()) == a
                assert sum(k * v for k, v in Hk.items()) == sumH
                # link (4): the explicit lower bound L on W
                L = sum(F(v, k + 1) for k, v in Hk.items()) + F(n - a, D + 1)
                okW = W >= L
                # the per-class collapse: L' = sum_k H_k (1/(k+1)+k/(D(D+1)))
                Lp = sum(v * (F(1, k + 1) + F(k, D * (D + 1)))
                         for k, v in Hk.items())
                okLLp = L >= Lp  # since (n-a)/(D+1) >= (1/(D(D+1))) sum k H_k
                # link (5)+(6): L' >= (2/(D+1)) a
                ok56 = Lp >= F(2, D + 1) * a
                # final
                okfin = F(a) <= F(D + 1, 2) * W
                if F(a) == F(D + 1, 2) * W:
                    tight += 1
                if not (ok1 and ok2 and okW and okLLp and ok56 and okfin):
                    fails += 1
                    if fails <= 10:
                        print(f"  FAIL D={D} n={n} seq={asc}: "
                              f"ok1={ok1} ok2={ok2} okW={okW} okLLp={okLLp} "
                              f"ok56={ok56} okfin={okfin} a={a} W={W} L={L} Lp={Lp}")
    print(f"  checked {total} graphic sequences; {fails} failures; "
          f"{tight} attain equality a=((D+1)/2)W")
    return fails == 0


def main(nmax, dmax):
    print(f"=== Step (5) pointwise inequality, D=1..60 ===")
    p_ok = check_pointwise(60)
    print(f"=== Full chain on graphic sequences, NMAX={nmax} DMAX={dmax} ===")
    c_ok = check_chain(nmax, dmax)
    print()
    print("PROOF VERIFIED end-to-end." if (p_ok and c_ok)
          else "PROOF HAS A GAP -- see failures above.")


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    dmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(nmax, dmax)
