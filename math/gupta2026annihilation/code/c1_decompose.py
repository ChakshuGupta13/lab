#!/usr/bin/env python3
"""
TxGraffiti C1 hard-core DECOMPOSITION probe.

Master target (=> C1 via the known alpha >= R, Favaron 1991):
  L:  a <= (Delta-1)*alpha       for connected G, Delta>=2.
L does NOT involve R, so it is attackable with independence-number machinery.

Caro-Wei (Caro 1979, Wei 1981):  alpha >= CW := sum_i 1/(d_i+1) >= n/(Delta+1).

Strata of the proof of L (Delta>=3):
  S1  regular            : a = floor(n/2); (Delta-1)alpha >= (Delta-1)/(Delta+1)*n >= n/2 >= a.
  S2  d_a <= Delta-2     : (Delta-1) sum_{i<=a} 1/(d_i+1) >= (Delta-1)*a/(d_a+1) >= a.
  S3  d_a >= Delta-1     : residual ("top-heavy"); first-a Caro-Wei terms insufficient.
  (d_a = the a-th smallest degree, 1-indexed; a<=n-1 always for graphs with an edge.)

Unifying degree-sequence strengthening (removes all graph structure if true):
  L': a <= (Delta-1)*CW.   L' => L (since alpha >= CW). Pure function of the degree sequence.

This script, over all connected graphs with Delta>=2 up to n:
  - confirms L holds (0 violations expected; it's C1's reduction),
  - TESTS L' (the key question): does Caro-Wei alone suffice? where is it tight / does it fail?
  - stratifies into A (Delta=2)/S1/S2/S3 and reports the hard-core residual {Delta>=3, alpha<n/2}
    broken down by stratum, plus within S3-hardcore how many have a > floor(n/2).

Usage: python c1_decompose.py [nmin] [nmax]   (default 3 9)
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


def annihilation_number(degs_desc_or_any, m):
    asc = sorted(degs_desc_or_any)
    tot, a = 0, 0
    for d in asc:
        tot += d
        if tot <= m:
            a += 1
        else:
            break
    return a


def analyze(nmin, nmax, max_ex=8):
    for n in range(nmin, nmax + 1):
        cntA = cntS1 = cntS2 = cntS3 = 0
        L_viol = 0
        Lp_viol = []                 # L' violations (g6, data) -- THE KEY OUTPUT
        Lp_tight = []                # L' equality cases
        hc = dict(total=0, S1=0, S2=0, S3=0)   # hard-core {Delta>=3, alpha<n/2}
        s3hc_above_floor = 0          # S3-hardcore with a > floor(n/2)
        s3hc_examples = []

        for g6, G in geng_stream(n):
            deg = dict(G.degree())
            degs = sorted(deg.values())          # ascending
            m = G.number_of_edges()
            Delta = degs[-1]
            delta = degs[0]
            if Delta < 2:
                continue
            alpha = independence_number(G)
            a = annihilation_number(degs, m)      # a <= n-1
            CW = sum(Fraction(1, d + 1) for d in degs)
            d_a = degs[a - 1]                     # a-th smallest (1-indexed); a>=1
            regular = (delta == Delta)

            # L and L'
            if a > (Delta - 1) * alpha:
                L_viol += 1
            lp_lhs, lp_rhs = a, (Delta - 1) * CW
            if lp_lhs > lp_rhs:
                if len(Lp_viol) < max_ex:
                    Lp_viol.append((g6, Delta, alpha, a, float(CW), degs))
            elif lp_lhs == lp_rhs:
                if len(Lp_tight) < max_ex:
                    Lp_tight.append((g6, Delta, alpha, a, degs))

            # stratum
            if Delta == 2:
                stratum = "A"; cntA += 1
            elif regular:
                stratum = "S1"; cntS1 += 1
            elif d_a <= Delta - 2:
                stratum = "S2"; cntS2 += 1
            else:
                stratum = "S3"; cntS3 += 1

            # hard core
            if Delta >= 3 and 2 * alpha < n:
                hc["total"] += 1
                if stratum in ("S1", "S2", "S3"):
                    hc[stratum] += 1
                if stratum == "S3":
                    if a > n // 2:
                        s3hc_above_floor += 1
                    if len(s3hc_examples) < max_ex:
                        s3hc_examples.append((g6, Delta, alpha, a, n // 2, d_a, degs))

        tot = cntA + cntS1 + cntS2 + cntS3
        print(f"\n===== n = {n}  ({tot} connected graphs, Delta>=2) =====")
        print(f"  strata: A(D=2)={cntA}  S1(reg)={cntS1}  S2(d_a<=D-2)={cntS2}  S3(top-heavy)={cntS3}")
        print(f"  L: a<=(D-1)alpha   violations = {L_viol}   (expect 0)")
        print(f"  L': a<=(D-1)*CaroWei  violations = {len(Lp_viol)}   <<< KEY")
        if Lp_viol:
            print("     L' VIOLATORS:")
            for e in Lp_viol:
                print("       ", e)
        if Lp_tight:
            print(f"  L' equality cases (sample): {[(e[0], 'D=%d a=%d alpha=%d'%(e[1],e[3],e[2])) for e in Lp_tight]}")
        print(f"  HARD CORE {{Delta>=3, alpha<n/2}}: total={hc['total']}  "
              f"-> S1={hc['S1']}  S2={hc['S2']}  S3(residual)={hc['S3']}")
        print(f"     of S3-hardcore, a>floor(n/2) (truly irregular): {s3hc_above_floor}")
        if s3hc_examples:
            print("     S3-hardcore samples (g6,D,alpha,a,floor(n/2),d_a,degs):")
            for e in s3hc_examples[:5]:
                print("       ", e)


if __name__ == "__main__":
    nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    analyze(nmin, nmax)
