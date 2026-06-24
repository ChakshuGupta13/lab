"""
Consolidated, machine-checked verification of every numerical claim in the paper
for OFG(M_{m,n}). Run after ofg.py is validated. Uses the min-degree-source
BFS diameter shortcut (validated equal to the exact all-pairs diameter for every
(m,n) with V small enough to check directly).

Prints PASS/FAIL for each claim. NO claim should rely on a single data point.
"""

import math
import networkx as nx

from ofg import build_ofg


def diam_shortcut(G, degree_of):
    """max eccentricity over minimum-degree vertices (== true diameter on all
    cross-checked instances)."""
    mind = min(degree_of.values())
    srcs = [v for v, d in degree_of.items() if d == mind]
    best = 0
    for s in srcs:
        L = nx.single_source_shortest_path_length(G, s)
        best = max(best, max(L.values()))
    return best


def collect(m, nmax):
    """return {n: (V, E, diam, deg_dist)} for n=1..nmax."""
    out = {}
    for n in range(1, nmax + 1):
        G, degree_of, total = build_ofg(m, n)
        V, E = G.number_of_nodes(), G.number_of_edges()
        assert V == total // 3
        dd = {}
        for d in degree_of.values():
            dd[d] = dd.get(d, 0) + 1
        out[n] = (V, E, diam_shortcut(G, degree_of), dd)
    return out


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    allpass = True

    print("== m=2 : reproduce Hull et al. exactly ==")
    d2 = collect(2, 7)
    for n in range(1, 8):
        V, E, dia, dd = d2[n]
        allpass &= check(f"n={n} V=2*3^(n-1)={2*3**(n-1)}", V == 2 * 3 ** (n - 1))
        allpass &= check(f"n={n} diam=ceil(n^2/2)={math.ceil(n*n/2)}",
                         dia == math.ceil(n * n / 2))
    for n in range(2, 8):
        V, E, dia, dd = d2[n]
        allpass &= check(f"n={n} E=8(n+1)3^(n-3)={8*(n+1)*3**(n-3)}",
                         E == 8 * (n + 1) * 3 ** (n - 3))
    # degree-2 count constant = 4 (n>=2); max degree = 2n appears twice
    for n in range(2, 8):
        dd = d2[n][3]
        allpass &= check(f"n={n} #deg2==4", dd.get(2) == 4)
        allpass &= check(f"n={n} maxdeg=2n={2*n} twice", dd.get(2 * n) == 2)

    print("== m=3 : OPEN case ==")
    d3 = collect(3, 8)
    Vs = [d3[n][0] for n in range(1, 9)]
    print(f"  V(M_3,n) = {Vs}   (OEIS A052913)")
    allpass &= check("V(n)=5V(n-1)-2V(n-2), V1=4,V2=18",
                     all(Vs[i] == 5 * Vs[i - 1] - 2 * Vs[i - 2] for i in range(2, 8))
                     and Vs[0] == 4 and Vs[1] == 18)
    for n in range(1, 9):
        dia = d3[n][2]
        f3 = math.floor(3 * n * n / 4) + 2
        allpass &= check(f"n={n} diam=floor(3n^2/4)+2={f3} (got {dia})", dia == f3)
    # degree-sequence structure generalizes
    for n in range(2, 9):
        dd = d3[n][3]
        allpass &= check(f"n={n} #deg2==4", dd.get(2) == 4)
        allpass &= check(f"n={n} maxdeg=3n={3*n} twice", dd.get(3 * n) == 2)
        allpass &= check(f"n={n} #deg3==4(n-1)={4*(n-1)}", dd.get(3) == 4 * (n - 1))
    # deg-4 count is the degree-2 polynomial 2n^2+8n-16
    for n in range(2, 9):
        dd = d3[n][3]
        allpass &= check(f"n={n} #deg4==2n^2+8n-16={2*n*n+8*n-16}",
                         dd.get(4) == 2 * n * n + 8 * n - 16)

    print("== m=4 : OPEN case ==")
    d4 = collect(4, 6)
    for n in range(2, 7):
        dia = d4[n][2]
        f4 = n * n + 4 + (n % 2)
        allpass &= check(f"n={n} diam=n^2+4+[odd]={f4} (got {dia})", dia == f4)

    print("== symmetry D(m,n)=D(n,m) and diagonal D(m,m)=(m^3-m)/3 ==")
    D = {}
    for (m, n) in [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3),
                   (2, 5), (5, 2), (3, 5), (5, 3), (3, 3), (4, 4)]:
        G, deg, _ = build_ofg(m, n)
        D[(m, n)] = diam_shortcut(G, deg)
    for (a, b) in [((2, 3), (3, 2)), ((2, 4), (4, 2)), ((3, 4), (4, 3)),
                   ((2, 5), (5, 2)), ((3, 5), (5, 3))]:
        allpass &= check(f"D{a}==D{b}  ({D[a]}=={D[b]})", D[a] == D[b])
    for m in (2, 3, 4):
        val = (m ** 3 - m) // 3
        allpass &= check(f"D({m},{m})=(m^3-m)/3={val} (got {D[(m,m)]})",
                         D[(m, m)] == val)
    # D(5,5)=40=(5^3-5)/3 is confirmed by exact_diameter.py (V=193662, too large
    # for the builds above); see that script for the exact iFUB computation.

    print()
    print("ALL CLAIMS PASS" if allpass else "SOME CLAIM FAILED")


if __name__ == "__main__":
    main()
