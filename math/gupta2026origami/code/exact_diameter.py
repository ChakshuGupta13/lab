"""
Exact diameter via iFUB (iterative Fringe Upper Bound).

The min-degree-source shortcut (max eccentricity over minimum-degree vertices) is
only a LOWER bound on the diameter unless the diametral pair includes a
minimum-degree vertex (proven for m=2 by Hull et al., unproven for m>=3). iFUB
computes the EXACT diameter with a small number of BFS sweeps (not all-pairs), so
it scales to the V=18k..160k graphs that nx.diameter cannot handle in reasonable
time.

Also re-checks the m=3 degree-sequence polynomiality for d=5,6.
"""

import argparse
from collections import deque, defaultdict

import numpy as np

from ofg import build_ofg


def bfs_eccentricity(adj, src):
    dist = {src: 0}
    dq = deque([src])
    far = 0
    while dq:
        u = dq.popleft()
        du = dist[u]
        for v in adj[u]:
            if v not in dist:
                dist[v] = du + 1
                if du + 1 > far:
                    far = du + 1
                dq.append(v)
    return far, dist


def ifub_diameter(G):
    adj = {u: list(G[u]) for u in G.nodes}
    s = next(iter(adj))
    # double sweep to land on a peripheral vertex u
    _, dist_s = bfs_eccentricity(adj, s)
    u = max(dist_s, key=dist_s.get)
    ecc_u, dist_u = bfs_eccentricity(adj, u)

    by_level = defaultdict(list)
    for v, d in dist_u.items():
        by_level[d].append(v)

    lb = ecc_u
    i = ecc_u
    bfs_count = 2
    while i >= 1:
        if 2 * (i - 1) <= lb:           # no fringe vertex can beat current lb
            break
        for v in by_level[i]:
            ecc_v, _ = bfs_eccentricity(adj, v)
            bfs_count += 1
            if ecc_v > lb:
                lb = ecc_v
        i -= 1
    return lb, bfs_count


def fit_poly_degree(xs, ys):
    """Lowest polynomial degree (<=len-1) that fits (xs,ys) exactly over the rationals."""
    for deg in range(0, len(xs)):
        # need deg+1 points to fit a degree-deg poly; check it predicts the rest
        coeffs = np.polyfit(xs[: deg + 1], ys[: deg + 1], deg)
        pred = np.polyval(coeffs, xs)
        if np.allclose(pred, ys, atol=1e-6):
            return deg
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="3:5,3:6,3:7,3:8,4:4,4:5,4:6,5:4")
    args = ap.parse_args()

    print("== EXACT diameter (iFUB) ==")
    print(f"{'m':>2} {'n':>2} {'V':>8} {'diam_exact':>10} {'formula':>8} {'bfs':>6}")
    for tok in args.cases.split(","):
        m, n = map(int, tok.split(":"))
        G, deg, _ = build_ofg(m, n)
        V = G.number_of_nodes()
        diam, nb = ifub_diameter(G)
        if m == 3:
            f = (3 * n * n) // 4 + 2
        elif m == 4:
            f = n * n + 4 + (n % 2)
        else:
            f = None
        ok = "==OK" if (f is not None and diam == f) else ("(m=5)" if m == 5 else "!!MISMATCH")
        print(f"{m:>2} {n:>2} {V:>8} {diam:>10} {str(f):>8} {nb:>6}  {ok}")

    print()
    print("== m=3 degree-sequence polynomiality: #deg-d vs degree d-2 poly ==")
    # collect #deg-d counts across n for fixed m=3
    ns = list(range(2, 9))
    dd_by_n = {}
    for n in ns:
        G, deg, _ = build_ofg(3, n)
        c = defaultdict(int)
        for d in deg.values():
            c[d] += 1
        dd_by_n[n] = c
    for d in range(2, 9):
        # use n large enough that v_n^d is in its polynomial regime (n >= ceil(d/2)+1)
        nmin = (d + 1) // 2 + 1
        xs = [n for n in ns if n >= nmin]
        ys = [dd_by_n[n][d] for n in xs]
        if len(xs) < 2:
            print(f"  d={d}: too few points (n>= {nmin})")
            continue
        pd = fit_poly_degree(np.array(xs, float), np.array(ys, float))
        exp = d - 2
        flag = "OK" if pd == exp else f"got {pd}, expected {exp}"
        print(f"  d={d}: counts(n={xs})={ys} -> poly degree {pd} (expect d-2={exp})  [{flag}]")


if __name__ == "__main__":
    main()
