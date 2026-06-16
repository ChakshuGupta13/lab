#!/usr/bin/env python3
"""Verify the friendship-graph family F_k and the crossover at k=4.

Confirms:
  mu*(F_k) = k                          (brute-force)
  H(F_k)   = 2k/(k+1) + k/2            (exact, cross-checked with closed form)
  C4 holds iff k <= 3                   (equality at k=3)
  mu*/H -> 2 as k -> infinity           (asymptotic)
"""
from fractions import Fraction
from graph_utils import (
    friendship_graph, harmonic_index, mu_star_bruteforce,
    H_friendship_closed,
)


def verify_friendship():
    print(f"{'k':>2} {'n':>3} {'m':>3} {'mu*':>5} {'=k?':>4} "
          f"{'H':>12} {'closed ok?':>10} {'C4':>8} {'ratio':>8}")
    print("-" * 68)
    for k in range(2, 9):
        G = friendship_graph(k)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        mu = mu_star_bruteforce(G)
        H = harmonic_index(G)
        Hc = H_friendship_closed(k)
        holds = "holds" if mu <= H else "FAILS"
        ratio = float(Fraction(mu) / H)
        print(f"{k:>2} {n:>3} {m:>3} {mu:>5} {str(mu == k):>4} "
              f"{str(H):>12} {str(H == Hc):>10} {holds:>8} {ratio:>8.4f}")

    print("\nAsymptotic ratio mu*/H along F_k:")
    for k in [3, 10, 100, 1000, 10000]:
        Hc = H_friendship_closed(k)
        print(f"  k={k:>5}: mu*/H = {float(Fraction(k) / Hc):.6f}")
    print("  limit: mu*/H -> 2")


if __name__ == "__main__":
    verify_friendship()
