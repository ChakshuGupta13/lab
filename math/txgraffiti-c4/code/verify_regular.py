#!/usr/bin/env python3
"""Verify the regular-graph lemma: C4 holds for all regular graphs.

Confirms on all connected r-regular graphs with n <= 9:
  H(G) = n/2                            (exact)
  mu*(G) <= H(G)                         (C4 holds)
  Equality mu* = n/2 iff K_{2t} or K_{t,t}  (Sumner 1979)
"""
from fractions import Fraction
from graph_utils import (
    geng_regular, harmonic_index, mu_star_linegraph,
    is_complete, is_balanced_complete_bipartite,
)


def verify_regular(nmax=9):
    total = 0
    H_fails = 0
    c4_fails = 0
    equality_graphs = []

    for n in range(2, nmax + 1):
        for r in range(1, n):
            if (n * r) % 2 != 0:
                continue
            for g6, G in geng_regular(n, r):
                total += 1
                H = harmonic_index(G)
                if H != Fraction(n, 2):
                    H_fails += 1
                    print(f"  H != n/2: n={n} r={r} {g6} H={H}")

                mu = mu_star_linegraph(G)
                if mu > H:
                    c4_fails += 1
                    print(f"  C4 FAILS: n={n} r={r} {g6} mu*={mu} H={H}")

                if Fraction(mu) == Fraction(n, 2):
                    kind = ("K_n" if is_complete(G)
                            else "K_t,t" if is_balanced_complete_bipartite(G)
                            else "OTHER")
                    equality_graphs.append((n, r, g6, kind))

    print(f"Checked {total} connected regular graphs, n=2..{nmax}.")
    print(f"H != n/2 failures: {H_fails}")
    print(f"C4 violations: {c4_fails}")
    print(f"\nEquality mu* = n/2 = H ({len(equality_graphs)} graphs):")
    for n, r, g6, kind in equality_graphs:
        print(f"  n={n} r={r} {g6:10} -> {kind}")
    non_sumner = [e for e in equality_graphs if e[3] == "OTHER"]
    print(f"\nGraphs with mu*=n/2 that are neither K_n nor K_t,t: "
          f"{len(non_sumner)} (Sumner predicts 0)")


if __name__ == "__main__":
    verify_regular()
