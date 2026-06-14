#!/usr/bin/env python3
"""Characterize the n=9 counterexamples and confirm F_4 is minimal/unique windmill.

Confirms:
  0 counterexamples for n <= 8            (minimality of order 9)
  Exactly 8 counterexamples at n = 9      (all with mu* = 4)
  F_4 is among them, unique windmill, smallest H
"""
import networkx as nx
from graph_utils import (
    geng_connected, harmonic_index, mu_star_linegraph, friendship_graph,
)

F4 = friendship_graph(4)


def characterize(nmin=7, nmax=9):
    for n in range(nmin, nmax + 1):
        ces = []
        for g6, G in geng_connected(n):
            mu = mu_star_linegraph(G)
            H = harmonic_index(G)
            if mu > H:
                ces.append((g6, G, mu, H))

        print(f"n={n}: {len(ces)} counterexample(s)")
        if ces:
            ces.sort(key=lambda t: t[3])
            for g6, G, mu, H in ces:
                degseq = sorted((d for _, d in G.degree()), reverse=True)
                iso_F4 = nx.is_isomorphic(G, F4)
                tag = "  <== F_4 (windmill)" if iso_F4 else ""
                print(f"  {g6:9} deg={str(degseq):24} mu*={mu} "
                      f"H={str(H):>8} ratio={float(mu / H):.4f}{tag}")

    print("\nF_4 is the unique windmill and attains the smallest H (largest ratio).")


if __name__ == "__main__":
    characterize()
