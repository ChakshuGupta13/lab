"""
Origami flip graph OFG(M_{m,n}) enumerator via the 3-coloring reconfiguration
bijection.

Background (R7-verified from Christensen, Hull, O'Neil, Pappano, Ter-Saakov, Yang,
"The Origami flip graph of the 2xn Miura-ori", arXiv:2506.19700, 2025; bijection
originally Ginepro-Hull 2014, J. Integer Seq.):

  Locally valid MV assignments of the m x n Miura-ori M_{m,n}  <->  proper 3-colorings
  of the m x n grid graph (one vertex precolored).  A "face flip" of face (i,j)
  corresponds to recoloring grid-vertex (i,j); the face is flippable iff that vertex's
  color can be changed while keeping the coloring proper.

  Hence  OFG(M_{m,n})  ==  R_3(grid_{m x n}) / Z_3  (global color-rotation quotient):
    - vertices = proper 3-colorings, canonicalized so corner (0,0) has color 0
                 (one representative per global "+c mod 3" orbit)
    - face (i,j) flippable  <=>  its grid-neighbors use <= 1 distinct color
                 (then the recolor is forced to the unique other available color)
    - degree(gamma) = number of flippable faces
    - edge {canon(gamma), canon(flip(gamma,(i,j)))} for each flippable (i,j)

Validation targets (Hull et al., M_{2,n}, must reproduce EXACTLY):
    V       = 2 * 3^(n-1)                 (Thm 3.1)
    E       = 8 * (n+1) * 3^(n-3)         (Thm 3.2)
    diameter= ceil(n^2 / 2)               (Cor 5.9)
    degrees = Table 1 distribution        (Sec 4.2)

Then run the open case m = 3.
"""

import argparse
import math
from collections import Counter

import networkx as nx


def grid_neighbors(m, n):
    nbrs = {}
    for i in range(m):
        for j in range(n):
            lst = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < n:
                    lst.append((ii, jj))
            nbrs[(i, j)] = lst
    return nbrs


def proper_3colorings(m, n, nbrs):
    """Yield every proper 3-coloring as a row-major tuple over the m*n vertices."""
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}
    coloring = [None] * len(cells)

    def bt(k):
        if k == len(cells):
            yield tuple(coloring)
            return
        used = set()
        for nb in nbrs[cells[k]]:
            kk = idx[nb]
            if kk < k:
                used.add(coloring[kk])
        for col in (0, 1, 2):
            if col not in used:
                coloring[k] = col
                yield from bt(k + 1)
        coloring[k] = None

    yield from bt(0)


def canon(coloring):
    """Global-rotation canonical form: shift so vertex 0 has color 0."""
    s = coloring[0]
    return tuple((c - s) % 3 for c in coloring)


def build_ofg(m, n):
    nbrs = grid_neighbors(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    idx = {c: k for k, c in enumerate(cells)}

    G = nx.Graph()
    degree_of = {}
    total = 0

    for col in proper_3colorings(m, n, nbrs):
        total += 1
        ccol = canon(col)
        if ccol in degree_of:
            continue
        deg = 0
        nbr_reps = []
        for k, cell in enumerate(cells):
            nb_colors = {ccol[idx[nb]] for nb in nbrs[cell]}
            if len(nb_colors) <= 1:
                cur = ccol[k]
                avail = [x for x in (0, 1, 2) if x not in nb_colors]
                # avail = [cur, other]; pick the other
                other = avail[0] if avail[1] == cur else avail[1]
                newc = list(ccol)
                newc[k] = other
                nbr_reps.append(canon(tuple(newc)))
                deg += 1
        degree_of[ccol] = deg
        G.add_node(ccol)
        for r in nbr_reps:
            G.add_edge(ccol, r)

    return G, degree_of, total


def analyze(m, n, want_diameter=True):
    G, degree_of, total = build_ofg(m, n)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    assert V == total // 3, f"vertex/orbit mismatch: V={V} total/3={total // 3}"
    # degree consistency: sum of flippable-face counts == 2|E| (handshake)
    sumdeg = sum(degree_of.values())
    deg_from_graph = sum(d for _, d in G.degree())
    distinct = Counter(degree_of.values())

    info = {
        "m": m, "n": n, "V": V, "E": E,
        "total_3col": total,
        "sum_flippable": sumdeg,
        "graph_degsum": deg_from_graph,
        "deg_dist": dict(sorted(distinct.items())),
        "connected": nx.is_connected(G),
    }
    if want_diameter and nx.is_connected(G):
        info["diameter"] = nx.diameter(G)
    return info


def hull_expected(n):
    V = 2 * 3 ** (n - 1)
    # E = 8(n+1)3^(n-3); use rationals to avoid fractional intermediate for n<3
    E = 8 * (n + 1) * 3 ** (n - 3) if n >= 3 else None
    if n == 2:
        E = 8
    diam = math.ceil(n * n / 2)
    return V, E, diam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--nmax", type=int, default=7)
    ap.add_argument("--nmin", type=int, default=1)
    ap.add_argument("--no-diam", action="store_true")
    args = ap.parse_args()

    print(f"=== OFG(M_{{{args.m},n}}) for n={args.nmin}..{args.nmax} ===")
    for n in range(args.nmin, args.nmax + 1):
        info = analyze(args.m, n, want_diameter=not args.no_diam)
        line = (f"n={n}: V={info['V']} E={info['E']} "
                f"conn={info['connected']} "
                f"diam={info.get('diameter', '-')} "
                f"sum_flip={info['sum_flippable']} 2E={2*info['E']}")
        print(line)
        print(f"    deg_dist={info['deg_dist']}")
        if args.m == 2:
            Vh, Eh, dh = hull_expected(n)
            okV = (info["V"] == Vh)
            okE = (info["E"] == Eh) if Eh is not None else "n/a"
            okD = (info.get("diameter") == dh)
            print(f"    HULL check: V {info['V']}=={Vh}?{okV}  "
                  f"E {info['E']}=={Eh}?{okE}  diam {info.get('diameter')}=={dh}?{okD}")


if __name__ == "__main__":
    main()
