#!/usr/bin/env python3
"""Verify the universal lower bound H(G) < 4 mu*(G) and the resulting
two-sided band for trees.

Theorem (lower bound).  Every graph G with at least one edge satisfies
    H(G) < 4 mu*(G),
equivalently mu*(G) > (1/4) H(G).  The constant 4 is best possible: the
balanced double star D(k,k) has mu* = 1 and H = 4k/(k+2) + 1/(k+1) -> 4, so
the ratio H/mu* approaches 4 from below but never reaches it.

Combined with the tree upper bound mu*(T) < (3/2) H(T) (see
verify_subdivided_star.py), every nontrivial tree satisfies the two-sided band
    (1/4) H(T) < mu*(T) < (3/2) H(T),
with both constants best possible -- the lower by D_k, the upper by S_k.

Confirms:
  H(G) < 4 mu*(G)  for every connected graph on n <= 8 vertices (exhaustive)
  H(D(k,k)) = 4k/(k+2) + 1/(k+1) -> 4   while mu* = 1            (sharpness)
  (1/4) H(T) < mu*(T) < (3/2) H(T)  for every tree on n <= 16    (two-sided)
"""
from fractions import Fraction

import networkx as nx

from graph_utils import (
    geng_connected, harmonic_index, mu_star_linegraph, mu_star_bruteforce,
    double_star, H_double_star_closed,
)


def verify_lower_bound_exhaustive(nmax=8):
    """Every connected graph on 2 <= n <= nmax vertices satisfies H < 4 mu*.
    Reports, per order, the largest ratio H/mu* (the closest approach to the
    bound 4) and the graph attaining it."""
    print(f"Exhaustive lower bound: H(G) < 4 mu*(G), connected graphs n <= {nmax}")
    print(f"{'n':>3} {'#graphs':>9} {'max H/mu*':>10} {'< 4?':>5} "
          f"{'argmax (graph6)':>16}")
    print("-" * 52)
    all_ok = True
    tested = 0
    for n in range(2, nmax + 1):
        ngraphs = 0
        worst = Fraction(0)
        worst_g6 = None
        for g6, G in geng_connected(n):
            ngraphs += 1
            H = harmonic_index(G)
            mu = mu_star_linegraph(G)
            if not (H < 4 * mu):
                all_ok = False
            r = H / mu
            if r > worst:
                worst, worst_g6 = r, g6
        ok = worst < 4
        all_ok = all_ok and ok
        tested += ngraphs
        print(f"{n:>3} {ngraphs:>9} {float(worst):>10.5f} {str(ok):>5} "
              f"{worst_g6:>16}")
    # guard against a vacuous pass if geng emits nothing
    all_ok = all_ok and tested > 0
    print(f"\nGraphs enumerated (must be > 0): {tested}")
    print(f"All satisfy H < 4 mu*: {all_ok}")
    return all_ok


def verify_double_star_sharp():
    """The balanced double star D(k,k) has mu* = 1 and H -> 4, so the constant
    4 in H < 4 mu* cannot be lowered.  Cross-checks mu* by brute force and H
    against its closed form."""
    print("\nSharpness: balanced double star D(k,k), mu* = 1, H/mu* -> 4")
    print(f"{'k':>5} {'n':>4} {'mu*':>4} {'H':>16} {'closed ok?':>10} "
          f"{'H/mu*':>10}")
    print("-" * 54)
    ok = True
    for k in [1, 2, 3, 5, 10, 100, 1000]:
        D = double_star(k, k)
        n = D.number_of_nodes()
        # brute force only where it is cheap; the identity method otherwise
        mu = mu_star_bruteforce(D) if k <= 10 else mu_star_linegraph(D)
        H = harmonic_index(D)
        Hc = H_double_star_closed(k)
        if mu != 1 or H != Hc or not (H < 4 * mu):
            ok = False
        print(f"{k:>5} {n:>4} {mu:>4} {str(H):>16} {str(H == Hc):>10} "
              f"{float(H / mu):>10.6f}")
    print("  limit: H/mu* -> 4 from below (constant 4 best possible)")
    return ok


def verify_two_sided_trees(nmax=16):
    """Every tree on 2 <= n <= nmax vertices satisfies the two-sided band
    (1/4) H(T) < mu*(T) < (3/2) H(T).  The minimum ratio mu*/H is attained by
    the balanced double star, the maximum by the subdivided star."""
    print(f"\nTwo-sided band: (1/4) H(T) < mu*(T) < (3/2) H(T), trees n <= {nmax}")
    print(f"{'n':>3} {'#trees':>8} {'min mu*/H':>10} {'max mu*/H':>10} "
          f"{'band ok?':>8}")
    print("-" * 44)
    all_ok = True
    for n in range(2, nmax + 1):
        lo = Fraction(3, 2)
        hi = Fraction(0)
        ok = True
        for T in nx.nonisomorphic_trees(n):
            H = harmonic_index(T)
            mu = mu_star_linegraph(T)
            if not (Fraction(1, 4) * H < mu < Fraction(3, 2) * H):
                ok = False
            r = Fraction(mu) / H
            lo = min(lo, r)
            hi = max(hi, r)
        all_ok = all_ok and ok
        n_trees = sum(1 for _ in nx.nonisomorphic_trees(n))
        print(f"{n:>3} {n_trees:>8} {float(lo):>10.5f} {float(hi):>10.5f} "
              f"{str(ok):>8}")
    print(f"\nAll trees satisfy the two-sided band: {all_ok}")
    print("  min mu*/H -> 1/4 (balanced double star), "
          "max mu*/H -> 3/2 (subdivided star)")
    return all_ok


if __name__ == "__main__":
    import sys
    ok = verify_lower_bound_exhaustive()
    ok = verify_double_star_sharp() and ok
    ok = verify_two_sided_trees() and ok
    sys.exit(0 if ok else 1)
