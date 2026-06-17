#!/usr/bin/env python3
"""Verify the subdivided-star family S_k and the triangle-free/bipartite
resolution of TxGraffiti Conjecture 4.

S_k is the star K_{1,k} with every edge subdivided once (a spider with k legs
of length 2): one hub of degree k, k middle vertices of degree 2, k leaf
vertices of degree 1.  It is a tree, hence bipartite and triangle-free.

Confirms:
  mu*(S_k) = k                          (brute-force)
  H(S_k)   = 2k/(k+2) + 2k/3            (exact, cross-checked with closed form)
  C4 fails iff k >= 5                    (equality at k=4, on nine vertices)
  mu*/H -> 3/2 as k -> infinity          (bounded, unlike the windmill family)
  S_k is a tree (=> bipartite, triangle-free) for every k
"""
from fractions import Fraction

import networkx as nx

from graph_utils import (
    subdivided_star, harmonic_index, mu_star_bruteforce,
    H_subdivided_star_closed,
)


def verify_subdivided_star():
    print(f"{'k':>2} {'n':>3} {'m':>3} {'mu*':>5} {'=k?':>4} "
          f"{'H':>12} {'closed ok?':>10} {'C4':>8} {'tree?':>6} {'ratio':>8}")
    print("-" * 78)
    for k in range(2, 11):
        G = subdivided_star(k)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        mu = mu_star_bruteforce(G)
        H = harmonic_index(G)
        Hc = H_subdivided_star_closed(k)
        is_tree = nx.is_tree(G)
        holds = "holds" if mu <= H else "FAILS"
        ratio = float(Fraction(mu) / H)
        print(f"{k:>2} {n:>3} {m:>3} {mu:>5} {str(mu == k):>4} "
              f"{str(H):>12} {str(H == Hc):>10} {holds:>8} "
              f"{str(is_tree):>6} {ratio:>8.4f}")

    print("\nSmallest tree counterexample is S_5 (n = 11):")
    G = subdivided_star(5)
    H = harmonic_index(G)
    mu = mu_star_bruteforce(G)
    print(f"  mu*(S_5) = {mu}, H(S_5) = {H} = {float(H):.4f}, "
          f"bipartite = {nx.is_bipartite(G)}, "
          f"triangle-free = {sum(nx.triangles(G).values()) == 0}")

    print("\nAsymptotic ratio mu*/H along S_k:")
    for k in [4, 10, 100, 1000, 10000]:
        Hc = H_subdivided_star_closed(k)
        print(f"  k={k:>5}: mu*/H = {float(Fraction(k) / Hc):.6f}")
    print("  limit: mu*/H -> 3/2 (bounded)")


def verify_tree_conjecture(nmax=16):
    """Exhaustive check that mu*(T) < (3/2) H(T) for every tree T on n <= nmax
    vertices, and that the subdivided star attains the maximum ratio at each
    odd order n = 2k+1."""
    from graph_utils import mu_star_linegraph
    print(f"\nConjecture check: mu*(T) < (3/2) H(T) for all trees, n <= {nmax}")
    print(f"{'n':>3} {'#trees':>8} {'max mu*/H':>10} {'<3/2?':>6} "
          f"{'argmax=S_k?':>12}")
    for n in range(2, nmax + 1):
        best = Fraction(0)
        best_T = None
        all_ok = True
        for T in nx.nonisomorphic_trees(n):
            H = harmonic_index(T)
            r = Fraction(mu_star_linegraph(T)) / H
            if r >= Fraction(3, 2):
                all_ok = False
            if r > best:
                best, best_T = r, T
        is_Sk = (n - 1) % 2 == 0 and nx.is_isomorphic(
            best_T, subdivided_star((n - 1) // 2))
        n_trees = sum(1 for _ in nx.nonisomorphic_trees(n))
        print(f"{n:>3} {n_trees:>8} {float(best):>10.5f} {str(all_ok):>6} "
              f"{str(is_Sk):>12}")


if __name__ == "__main__":
    verify_subdivided_star()
    verify_tree_conjecture()
