#!/usr/bin/env python3
"""Deterministic-construction probe for Baste's conjecture gamma <= gamma_e at Delta=3 (cubic).

The conjecture reduces to: every maximal matching M of an r-regular graph admits a
DOMINATING TRANSVERSAL -- one endpoint per M-edge (set D, |D|=|M|) dominating the
unsaturated independent set Z = V\\V(M). D need NOT be independent (gamma-version, not
i-version), so it is a pure covering problem. The lopsided LLL proves existence for
r >= 7; r <= 6 needs a deterministic argument.

Question: does a SIMPLE deterministic RULE always yield a dominating transversal for
cubic graphs? 'Yes' => a concrete proof target. 'No' => the min-B exchange / discharging
(Baste Thm 3, claw-free only) is really needed.

'at-risk' u in Z: no two of its neighbours are matched to each other (else u is
auto-covered whichever endpoint is chosen). Only at-risk vertices can fail to be covered.
"""
import sys, subprocess, itertools


def cubic_graphs(n):
    out = subprocess.run(["geng", "-qc", "-d3", "-D3", str(n)],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.strip():
            yield g6_to_adj(line.strip(), n)


def g6_to_adj(g6, n):
    data = g6.encode()
    idx = 1
    bits = []
    for c in data[idx:]:
        bits.extend((c - 63) >> k & 1 for k in range(5, -1, -1))
    adj = [set() for _ in range(n)]
    p = 0
    for j in range(1, n):
        for i in range(j):
            if p < len(bits) and bits[p]:
                adj[i].add(j); adj[j].add(i)
            p += 1
    return adj


def maximal_matchings(adj, n):
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    res = set()

    def rec(sat, chosen):
        first = None
        for k, (i, j) in enumerate(edges):
            if i not in sat and j not in sat:
                first = k; break
        if first is None:
            res.add(frozenset(chosen)); return
        i, j = edges[first]
        rec(sat | {i, j}, chosen + [(i, j)])          # include e
        for a, b in edges:                             # exclude e: saturate an endpoint otherwise
            if (a in (i, j) or b in (i, j)) and (a, b) != (i, j) and a not in sat and b not in sat:
                rec(sat | {a, b}, chosen + [(a, b)])
    rec(frozenset(), [])
    return res


def at_risk(adj, M, n):
    sat = set(itertools.chain.from_iterable(M))
    partner = {}
    for i, j in M:
        partner[i] = j; partner[j] = i
    ar = []
    for u in range(n):
        if u in sat:
            continue
        nb = adj[u]
        if any(partner.get(x) in nb for x in nb):      # two neighbours matched => auto-covered
            continue
        ar.append((u, tuple(nb)))
    return ar


def cov_map(ar):
    cov = {}
    for idx, (_u, nb) in enumerate(ar):
        for x in nb:
            cov.setdefault(x, set()).add(idx)
    return cov


def all_covered(D, ar):
    return all(any(x in D for x in nb) for _u, nb in ar)


def greedy_maxcov(M, ar, order=None):
    Ms = sorted(M)
    if order is not None:
        Ms = [Ms[k] for k in order]
    cov = cov_map(ar)
    D, covered = set(), set()
    for i, j in Ms:
        ci = cov.get(i, set()) - covered
        cj = cov.get(j, set()) - covered
        pick = i if len(ci) >= len(cj) else j
        D.add(pick); covered |= cov.get(pick, set())
    return all_covered(D, ar)


def greedy_sorted(M, ar):
    Ms = sorted(M)
    cov = cov_map(ar)
    order = sorted(range(len(Ms)),
                   key=lambda k: -max(len(cov.get(Ms[k][0], set())),
                                      len(cov.get(Ms[k][1], set()))))
    return greedy_maxcov(M, ar, order)


def exists_dominating(M, ar):
    if not ar:
        return True
    Ms = sorted(M)
    cov = cov_map(ar)
    target = set(range(len(ar)))
    for bits in itertools.product((0, 1), repeat=len(Ms)):
        covered = set()
        for k, b in enumerate(bits):
            covered |= cov.get(Ms[k][b], set())
        if covered == target:
            return True
    return False


def main():
    ns = [int(x) for x in sys.argv[1:]] or [6, 8, 10, 12, 14]
    for n in ns:
        ngraphs = tot = with_ar = ex_ok = gr_ok = grs_ok = gr_fail_ex = 0
        for adj in cubic_graphs(n):
            ngraphs += 1
            for M in maximal_matchings(adj, n):
                tot += 1
                ar = at_risk(adj, M, n)
                if ar:
                    with_ar += 1
                ex = exists_dominating(M, ar)
                gr = greedy_maxcov(M, ar)
                grs = greedy_sorted(M, ar)
                ex_ok += ex; gr_ok += gr; grs_ok += grs
                if ex and not (gr or grs):
                    gr_fail_ex += 1
        print(f"n={n:2d}: graphs={ngraphs:5d} matchings={tot:7d} with-atrisk={with_ar:6d} | "
              f"EXISTS={ex_ok}/{tot} GREEDY={gr_ok}/{tot} GREEDY-SORT={grs_ok}/{tot} "
              f"both-greedy-fail-but-exists={gr_fail_ex}")


if __name__ == "__main__":
    main()
