#!/usr/bin/env python3
"""Verify the subquartic enumeration and the equality classification.

The paper conjectures that every connected graph of maximum degree at most
four (a "subquartic" graph) satisfies mu* <= H, and observes that the
inequality is met with equality by exactly six such graphs on up to nine
vertices.  This script is the exhaustive evidence on 2 <= n <= 9; the paper
reports the same no-counterexample outcome up to n = 11.

Confirms:
  no connected graph with Delta <= 4 on 2 <= n <= 9 violates mu* <= H
  exactly six such graphs attain mu* = H:
    K_2, C_4, K_4, K_{3,3}, K_{4,4}   (regular, randomly matchable)
    S_4                               (the unique non-regular extremal graph)
"""
from graph_utils import geng_maxdeg, harmonic_index, mu_star_linegraph


# The six subquartic graphs with mu* = H, keyed by (n, sorted degree sequence).
EXPECTED_EQUALITY = {
    (2, (1, 1)): "K_2",
    (4, (2, 2, 2, 2)): "C_4",
    (4, (3, 3, 3, 3)): "K_4",
    (6, (3, 3, 3, 3, 3, 3)): "K_{3,3}",
    (8, (4, 4, 4, 4, 4, 4, 4, 4)): "K_{4,4}",
    (9, (1, 1, 1, 1, 2, 2, 2, 2, 4)): "S_4",
}


def verify_subquartic(nmax=9):
    """Single pass per order: count Delta<=4 graphs, count counterexamples to
    mu* <= H, and collect the graphs meeting equality."""
    print(f"Subquartic enumeration: mu* <= H for Delta <= 4, n <= {nmax}")
    print(f"{'n':>3} {'#graphs(D<=4)':>14} {'#counterex.':>12} "
          f"{'#equality':>10}")
    print("-" * 42)
    total_ce = 0
    equality = []
    tested = 0
    for n in range(2, nmax + 1):
        ngraphs = nce = neq = 0
        for g6, G in geng_maxdeg(n, 4):
            ngraphs += 1
            H = harmonic_index(G)
            mu = mu_star_linegraph(G)
            if mu > H:
                nce += 1
            elif mu == H:
                neq += 1
                degseq = tuple(sorted(d for _, d in G.degree()))
                equality.append((n, degseq))
        total_ce += nce
        tested += ngraphs
        print(f"{n:>3} {ngraphs:>14} {nce:>12} {neq:>10}")

    print(f"\nTotal counterexamples (Delta <= 4, n <= {nmax}): {total_ce}")

    print("\nGraphs meeting mu* = H:")
    found = {}
    for n, degseq in equality:
        name = EXPECTED_EQUALITY.get((n, degseq), "UNEXPECTED")
        found[(n, degseq)] = name
        print(f"  n={n:>2}  deg={degseq}  ->  {name}")

    # tested > 0 guards against a vacuous pass if geng emits nothing; the count
    # check closes the dict-masking hole (two non-isomorphic graphs sharing one
    # (n, degseq) key would make len(equality) exceed the expected six).
    ok_tested = tested > 0
    ok_ce = total_ce == 0
    ok_eq = found == EXPECTED_EQUALITY and len(equality) == len(EXPECTED_EQUALITY)
    print(f"\nGraphs enumerated (must be > 0): {tested}")
    print(f"No subquartic counterexample: {ok_ce}")
    print(f"Equality set is exactly {{K_2, C_4, K_4, K_{{3,3}}, K_{{4,4}}, S_4}}: "
          f"{ok_eq}")
    return ok_tested and ok_ce and ok_eq


if __name__ == "__main__":
    import sys
    sys.exit(0 if verify_subquartic() else 1)
