"""Verification for the counting identity added to the paper.

Claim: for ANY maximal matching M of a Delta-regular graph on n vertices, with
q = the number of edges inside V(M) that are not in M,

        |M| = (Delta*n + 2q) / (4*Delta - 2),

hence |M| >= Delta*n/(4*Delta-2) with equality iff q = 0, i.e. iff M is an
INDUCED matching.

Also checks the claim made about thm:cex: its 50-vertex cubic graph attains the
floor, so its minimum maximal matching is induced.
"""

import itertools
from fractions import Fraction

import networkx as nx


def maximal_matchings(G):
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    out = []

    def rec(i, used, cur):
        if i == len(edges):
            for u, v in edges:
                if u not in used and v not in used:
                    return
            out.append(tuple(cur))
            return
        u, v = edges[i]
        if u not in used and v not in used:
            cur.append((u, v))
            rec(i + 1, used | {u, v}, cur)
            cur.pop()
        rec(i + 1, used, cur)

    rec(0, frozenset(), [])
    return out


def audit(G, name, sample_all=True):
    degs = {d for _, d in G.degree()}
    assert len(degs) == 1, f"{name} not regular"
    D = degs.pop()
    n = G.number_of_nodes()
    bad_id = bad_eq = 0
    best = None
    for M in maximal_matchings(G):
        S = {x for e in M for x in e}
        q = G.subgraph(S).number_of_edges() - len(M)
        # identity
        if Fraction(D * n + 2 * q, 4 * D - 2) != len(M):
            bad_id += 1
        # equality <=> induced matching
        induced = (G.subgraph(S).number_of_edges() == len(M))
        at_floor = (Fraction(len(M)) == Fraction(D * n, 4 * D - 2))
        if at_floor != induced:
            bad_eq += 1
        if best is None or len(M) < best[0]:
            best = (len(M), q, induced)
    gam_e, q_min, ind = best
    print(f"  {'OK ' if bad_id == 0 and bad_eq == 0 else 'FAIL'} {name:20s} "
          f"D={D} n={n:3d} gamma_e={gam_e:3d} q_min={q_min:3d} "
          f"floor={float(Fraction(D*n,4*D-2)):6.2f} induced={ind} "
          f"[identity viol={bad_id}, equality viol={bad_eq}]")
    return bad_id + bad_eq


def cex_graph():
    """The thm:cex construction: incidence graph of a balanced (3,2,2) formula.
    Any balanced formula gives the same structural shape; the 20-clause
    unsatisfiable one is what makes it a counterexample, which is irrelevant to
    the counting identity being checked here."""
    import random
    D, v = 3, 15
    c = 2 * v * (D - 1) // D
    slots = [(x, s) for x in range(v) for s in (True, False) for _ in range(D - 1)]
    rnd = random.Random(7)
    for _ in range(20000):
        rnd.shuffle(slots)
        cl = [slots[i * D:(i + 1) * D] for i in range(c)]
        if all(len({x for x, _ in q}) == D for q in cl):
            break
    G = nx.Graph()
    for x in range(v):
        G.add_edge(("v", x, True), ("v", x, False))
    for i, q in enumerate(cl):
        for x, s in q:
            G.add_edge(("c", i), ("v", x, s))
    M = [(("v", x, True), ("v", x, False)) for x in range(v)]
    return G, M


total = 0
print("identity + equality characterisation, all maximal matchings enumerated:")
for G, name in [
    (nx.complete_graph(4), "K4"),
    (nx.cycle_graph(6), "C6"),
    (nx.cycle_graph(9), "C9"),
    (nx.petersen_graph(), "Petersen"),
    (nx.complete_bipartite_graph(3, 3), "K3,3"),
    (nx.hypercube_graph(3), "Q3"),
    (nx.circulant_graph(10, [1, 2]), "C10(1,2) 4-reg"),
    (nx.cartesian_product(nx.complete_graph(3), nx.complete_graph(2)), "prism"),
    (nx.cartesian_product(nx.complete_graph(3), nx.complete_graph(3)), "rook"),
    (nx.complete_graph(6), "K6"),
]:
    total += audit(nx.convert_node_labels_to_integers(G), name)

print("\nthm:cex construction (50-vertex cubic incidence graph):")
G, M = cex_graph()
D, n = 3, G.number_of_nodes()
S = {x for e in M for x in e}
q = G.subgraph(S).number_of_edges() - len(M)
floor = Fraction(D * n, 4 * D - 2)
print(f"  n={n}  |M|={len(M)}  q={q}  floor={floor}  "
      f"at_floor={Fraction(len(M)) == floor}  induced={q == 0}  "
      f"regular={ {d for _, d in G.degree()} }")

print(f"\ntotal violations across all tests: {total}")
