"""Shared graph invariant computations for TxGraffiti C4 refutation.

All arithmetic on the harmonic index uses fractions.Fraction for exact
equality/violation detection (no floating-point rounding).

Requires: networkx >= 3.0, nauty geng on PATH.
"""
import subprocess
from fractions import Fraction
from itertools import combinations

import networkx as nx


# ---------------------------------------------------------------------------
# Graph generation (nauty geng)
# ---------------------------------------------------------------------------

def geng_connected(n):
    """Yield (graph6_string, nx.Graph) for all connected graphs on n vertices."""
    proc = subprocess.Popen(
        ["geng", "-qc", str(n)], stdout=subprocess.PIPE, text=True
    )
    for line in proc.stdout:
        g6 = line.strip()
        if g6:
            yield g6, nx.from_graph6_bytes(g6.encode())
    proc.wait()


def geng_regular(n, r):
    """Yield (graph6_string, nx.Graph) for all connected r-regular graphs on n vertices."""
    out = subprocess.run(
        ["geng", "-qc", f"-d{r}", f"-D{r}", str(n)],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines():
        g6 = line.strip()
        if g6:
            yield g6, nx.from_graph6_bytes(g6.encode())


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def harmonic_index(G):
    """Exact harmonic index H(G) = sum_{uv in E} 2/(d(u)+d(v))."""
    deg = dict(G.degree())
    return sum(
        (Fraction(2, deg[u] + deg[v]) for u, v in G.edges()),
        Fraction(0),
    )


def mu_star_linegraph(G):
    """mu*(G) via the identity mu*(G) = i(L(G))."""
    if G.number_of_edges() == 0:
        return 0
    L = nx.line_graph(G)
    comp = nx.complement(L)
    return min(len(c) for c in nx.find_cliques(comp))


def _is_matching(edges):
    seen = set()
    for u, v in edges:
        if u in seen or v in seen:
            return False
        seen.update((u, v))
    return True


def _matching_is_maximal(G, edges):
    sat = set()
    for u, v in edges:
        sat.update((u, v))
    return all(u in sat or v in sat for u, v in G.edges())


def mu_star_bruteforce(G):
    """mu*(G) by exhaustive search (independent of the line-graph method)."""
    E = list(G.edges())
    if not E:
        return 0
    for k in range(1, len(E) + 1):
        for sub in combinations(E, k):
            if _is_matching(sub) and _matching_is_maximal(G, sub):
                return k
    return len(E)


# ---------------------------------------------------------------------------
# Named graph families
# ---------------------------------------------------------------------------

def friendship_graph(k):
    """F_k: k triangles sharing one hub vertex.  n = 2k+1, m = 3k."""
    G = nx.Graph()
    hub = 0
    for i in range(k):
        a, b = 2 * i + 1, 2 * i + 2
        G.add_edge(hub, a)
        G.add_edge(hub, b)
        G.add_edge(a, b)
    return G


def generalized_windmill(m, k):
    """G_{m,k} = K_{m,2k} + perfect matching on the 2k-side.
    m hubs, 2k blade vertices, k blade edges.  G_{1,k} = F_k."""
    G = nx.Graph()
    hubs = [("h", j) for j in range(m)]
    blade = [("b", i, s) for i in range(k) for s in (0, 1)]
    G.add_nodes_from(hubs + blade)
    for h in hubs:
        for b in blade:
            G.add_edge(h, b)
    for i in range(k):
        G.add_edge(("b", i, 0), ("b", i, 1))
    return G


def subdivided_star(k):
    """S_k: the star K_{1,k} with every edge subdivided once (a spider with k
    legs of length 2).  One hub of degree k, k middle vertices of degree 2,
    k leaf vertices of degree 1.  n = 2k+1, m = 2k.  A tree, hence bipartite
    and triangle-free."""
    G = nx.Graph()
    hub = "h"
    for i in range(k):
        mid, leaf = ("m", i), ("l", i)
        G.add_edge(hub, mid)
        G.add_edge(mid, leaf)
    return G


# ---------------------------------------------------------------------------
# Closed-form harmonic indices
# ---------------------------------------------------------------------------

def H_friendship_closed(k):
    """Closed form: H(F_k) = 2k/(k+1) + k/2."""
    return Fraction(2 * k, k + 1) + Fraction(k, 2)


def H_generalized_closed(m, k):
    """Closed form: H(G_{m,k}) = k/(m+1) + 4km/(2k+m+1)."""
    return Fraction(k, m + 1) + Fraction(4 * k * m, 2 * k + m + 1)


def H_subdivided_star_closed(k):
    """Closed form: H(S_k) = 2k/(k+2) + 2k/3.
    The k hub-middle edges each join a degree-k hub to a degree-2 middle
    (weight 2/(k+2)); the k middle-leaf edges each join degree 2 to degree 1
    (weight 2/3)."""
    return Fraction(2 * k, k + 2) + Fraction(2 * k, 3)


# ---------------------------------------------------------------------------
# Graph classification helpers
# ---------------------------------------------------------------------------

def is_complete(G):
    n = G.number_of_nodes()
    return G.number_of_edges() == n * (n - 1) // 2


def is_balanced_complete_bipartite(G):
    if not nx.is_bipartite(G):
        return False
    A, B = nx.bipartite.sets(G)
    return len(A) == len(B) and G.number_of_edges() == len(A) * len(B)
