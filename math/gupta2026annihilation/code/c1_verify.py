#!/usr/bin/env python3
"""
TxGraffiti Conjecture 1 deep-scope verification.

C1 (Davila-Brimkov-Pepper, arXiv:2507.17780, open since 2016):
  G nontrivial connected, Delta>=2  =>  Delta*alpha >= a + R   (integer form of
  alpha >= (a+R)/Delta), and the bound is sharp.

Invariants (all exact integers):
  Delta  max degree
  alpha  independence number = max maximal-clique of complement(G)
  a      annihilation number = max j : (sum of j smallest degrees) <= m
  R      residue             = #zeros after Havel-Hakimi terminates

Reduction lemma under test:
  L:  a <= (Delta-1)*alpha
  Fact: C1 holds via the known alpha>=R (Favaron 1991) iff L holds, because
        Delta*alpha = (Delta-1)alpha + alpha >= a + alpha >= a + R  <=  L and alpha>=R.
  So this script also records where L holds / fails: the L-FAIL set is exactly the
  set of graphs whose C1 truth needs strictly more than alpha>=R.

For each n it reports:
  - #connected graphs
  - C1 violations split by Delta>=2 vs Delta==1 (confirms the nontrivial reading)
  - C1 equality graphs (Delta*alpha == a+R): degree sequence + invariants
  - L violations (Delta>=2) and L equality graphs
  - cross-tab: among C1-equality graphs, how many are also L-equality

Usage: python c1_verify.py [nmin] [nmax]   (default 3 9)
Requires: geng on PATH, networkx.
"""
import sys
import subprocess
from collections import Counter
import networkx as nx


def geng_stream(n, connected=True):
    args = ["geng", "-q"]
    if connected:
        args.append("-c")
    args.append(str(n))
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        g6 = line.strip()
        if g6:
            yield g6, nx.from_graph6_bytes(g6.encode())
    proc.wait()


def independence_number(G):
    """alpha(G) = max maximal clique of complement(G)."""
    comp = nx.complement(G)
    return max(len(c) for c in nx.find_cliques(comp))


def annihilation_number(degs, m):
    """max j : sum of j smallest degrees <= m."""
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
    """#zeros after Havel-Hakimi terminates."""
    seq = sorted(degs, reverse=True)
    while seq and seq[0] > 0:
        d = seq[0]
        seq = seq[1:]
        if d > len(seq):
            raise ValueError("non-graphic during HH")
        for k in range(d):
            seq[k] -= 1
        seq.sort(reverse=True)
    return len(seq)


def analyze(nmin, nmax, list_eq=True, max_list=12):
    for n in range(nmin, nmax + 1):
        cnt = 0
        c1_viol_d2 = []      # C1 violations with Delta>=2  (should be EMPTY)
        c1_viol_d1 = 0       # C1 'violations' with Delta==1 (the K2 degenerate)
        c1_eq = []           # Delta*alpha == a+R, Delta>=2
        L_viol = []          # a > (Delta-1)alpha, Delta>=2
        L_eq = 0             # a == (Delta-1)alpha, Delta>=2
        both_eq = 0          # C1-equality AND L-equality
        c1_eq_degseq = Counter()

        for g6, G in geng_stream(n):
            cnt += 1
            deg = dict(G.degree())
            degs = list(deg.values())
            m = G.number_of_edges()
            Delta = max(degs)
            alpha = independence_number(G)
            a = annihilation_number(degs, m)
            R = residue(degs)

            lhs = Delta * alpha
            rhs = a + R

            if Delta == 1:
                if lhs < rhs:
                    c1_viol_d1 += 1
                continue  # Delta==1 excluded from the conjecture's scope

            # ---- Delta >= 2 from here ----
            if lhs < rhs:
                if len(c1_viol_d2) < max_list:
                    c1_viol_d2.append((g6, Delta, alpha, a, R, sorted(degs)))
            elif lhs == rhs:
                c1_eq.append((g6, Delta, alpha, a, R, tuple(sorted(degs, reverse=True))))
                c1_eq_degseq[tuple(sorted(degs, reverse=True))] += 1

            # reduction lemma L: a <= (Delta-1)alpha
            Llhs = a
            Lrhs = (Delta - 1) * alpha
            if Llhs > Lrhs:
                if len(L_viol) < max_list:
                    L_viol.append((g6, Delta, alpha, a, R, sorted(degs)))
            elif Llhs == Lrhs:
                L_eq += 1
                if lhs == rhs:
                    both_eq += 1

        print(f"\n===== n = {n}  ({cnt} connected graphs) =====")
        print(f"C1 (Delta>=2): violations = {len(c1_viol_d2)}   "
              f"equality = {len(c1_eq)}")
        if c1_viol_d2:
            print("   !!! C1 VIOLATORS (Delta>=2):", c1_viol_d2)
        print(f"C1 (Delta==1) degenerate violations (expect K2 only): {c1_viol_d1}")
        print(f"L: a<=(Delta-1)alpha (Delta>=2): violations = {len(L_viol)}   "
              f"equality = {L_eq}")
        if L_viol:
            print("   L VIOLATORS:", L_viol)
        print(f"   among C1-equality graphs, also L-equality: {both_eq}/{len(c1_eq)}")
        if list_eq and c1_eq:
            print(f"   C1-equality degree-sequence multiset (top): "
                  f"{c1_eq_degseq.most_common(8)}")
            # show a few concrete equality graphs
            for rec in c1_eq[:max_list]:
                g6, Delta, alpha, a, R, ds = rec
                print(f"     EQ g6={g6} Delta={Delta} alpha={alpha} a={a} R={R} "
                      f"degseq={ds}")


if __name__ == "__main__":
    nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    analyze(nmin, nmax)
