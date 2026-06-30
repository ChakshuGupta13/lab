#!/usr/bin/env python3
"""
TxGraffiti C1 subclass proof validation.

Claim (this session): C1 (Delta*alpha >= a+R) holds for every connected graph
with Delta>=2 that falls in either:
  Case A:  Delta == 2                  -- because a == alpha there
  Case B:  Delta >= 3 and alpha >= n/2 -- because a<=n-1 < 2alpha <= (Delta-1)alpha,
                                          so (Delta-1)alpha >= a, then +alpha>=R.
The residual OPEN core is:
  Case C:  Delta >= 3 and alpha < n/2.

This script classifies every connected graph (Delta>=2) up to n into A/B/C and
verifies the load-bearing facts:
  A:  a == alpha           (so a+R <= 2alpha = Delta*alpha)
  B:  2*alpha >= n  and  (Delta-1)*alpha >= a
  C:  count them; confirm C1 still holds; report the tightest C1 margin
      (Delta*alpha - (a+R)) and the lemma L margin ((Delta-1)alpha - a).

If A/B never fail their load-bearing facts and C1 holds throughout, the proof for
A and B is empirically sound and the hard core is exactly C.

Usage: python c1_subclass.py [nmin] [nmax]   (default 3 9)
"""
import sys
import subprocess
from fractions import Fraction
import networkx as nx


def geng_stream(n):
    proc = subprocess.Popen(["geng", "-qc", str(n)], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        g6 = line.strip()
        if g6:
            yield g6, nx.from_graph6_bytes(g6.encode())
    proc.wait()


def independence_number(G):
    return max(len(c) for c in nx.find_cliques(nx.complement(G)))


def annihilation_number(degs, m):
    asc = sorted(degs)
    tot, a = 0, 0
    for d in asc:
        tot += d
        if tot <= m:
            a += 1
        else:
            break
    return a


def residue(degs):
    seq = sorted(degs, reverse=True)
    while seq and seq[0] > 0:
        d = seq[0]
        seq = seq[1:]
        for k in range(d):
            seq[k] -= 1
        seq.sort(reverse=True)
    return len(seq)


def analyze(nmin, nmax):
    for n in range(nmin, nmax + 1):
        cntA = cntB = cntC = 0
        A_aeqalpha_fail = []     # Case A but a != alpha  (should be EMPTY)
        B_key_fail = []          # Case B but (Delta-1)alpha < a or 2alpha < n  (EMPTY)
        C_c1_fail = []           # Case C but C1 violated  (EMPTY -> C1 has no CE)
        C_tight = None           # tightest C1 margin in residual
        C_L_fail = 0             # in C, does lemma L (a<=(Delta-1)alpha) ever fail?
        anyC1fail = []

        for g6, G in geng_stream(n):
            deg = dict(G.degree())
            degs = list(deg.values())
            m = G.number_of_edges()
            Delta = max(degs)
            if Delta < 2:
                continue
            alpha = independence_number(G)
            a = annihilation_number(degs, m)
            R = residue(degs)
            c1_margin = Delta * alpha - (a + R)
            if c1_margin < 0:
                anyC1fail.append((g6, Delta, alpha, a, R))

            if Delta == 2:
                cntA += 1
                if a != alpha:
                    A_aeqalpha_fail.append((g6, a, alpha, sorted(degs)))
            elif 2 * alpha >= n:        # Delta>=3, alpha>=n/2
                cntB += 1
                if (Delta - 1) * alpha < a or 2 * alpha < n:
                    B_key_fail.append((g6, Delta, alpha, a))
            else:                        # Delta>=3, alpha<n/2  (residual)
                cntC += 1
                if c1_margin < 0:
                    C_c1_fail.append((g6, Delta, alpha, a, R))
                if a > (Delta - 1) * alpha:
                    C_L_fail += 1
                if C_tight is None or c1_margin < C_tight[0]:
                    C_tight = (c1_margin, g6, Delta, alpha, a, R, sorted(degs))

        total = cntA + cntB + cntC
        print(f"\n===== n = {n}  ({total} connected graphs, Delta>=2) =====")
        print(f"  Case A (Delta=2):                {cntA:7d}   "
              f"a!=alpha failures: {len(A_aeqalpha_fail)}")
        print(f"  Case B (Delta>=3, alpha>=n/2):   {cntB:7d}   "
              f"key-step failures: {len(B_key_fail)}")
        print(f"  Case C (Delta>=3, alpha<n/2):    {cntC:7d}   "
              f"C1 violations: {len(C_c1_fail)}   L-fails: {C_L_fail}")
        print(f"  TOTAL C1 violations (all cases): {len(anyC1fail)}")
        if A_aeqalpha_fail:
            print("   !! A a!=alpha:", A_aeqalpha_fail[:5])
        if B_key_fail:
            print("   !! B key-step fail:", B_key_fail[:5])
        if C_c1_fail:
            print("   !! C C1 violation:", C_c1_fail[:5])
        if C_tight:
            print(f"  residual (Case C) tightest C1 margin = {C_tight[0]}  "
                  f"at g6={C_tight[1]} Delta={C_tight[2]} alpha={C_tight[3]} "
                  f"a={C_tight[4]} R={C_tight[5]} degseq={C_tight[6]}")


if __name__ == "__main__":
    nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    analyze(nmin, nmax)
