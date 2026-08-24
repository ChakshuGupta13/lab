"""Perturb the n>=50 EQUALITY cases (gamma = gamma_e) hunting a Baste counterexample.

Equality cases are one step from a counterexample, so they are the right seeds.
GP(25,2), GP(25,12) (n=50) and GP(26,2) (n=52) attain gamma = gamma_e and lie
beyond our n<=48 theorem.  They also CONTAIN CLAWS (vertex i has i-1, i+1, i+k
pairwise non-adjacent), so they fall outside the known claw-free equality
characterisation.

Move: a cubic-preserving 2-swap -- replace disjoint edges (a,b),(c,d) by
(a,c),(b,d) -- keeping the graph simple, cubic and connected.
Test: gamma_e, then ONE SAT call "is there a dominating set of size gamma_e?".
UNSAT there means gamma > gamma_e, i.e. a counterexample to Baste.
Fragility (few minimum dominating sets) is the hill-climb gradient.
"""
import itertools
import random
import sys
import time

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical195 as Solver


def edges_of(adj):
    return sorted({(min(u, v), max(u, v)) for u in range(len(adj)) for v in adj[u]})


def connected(adj, n):
    seen, stack = {0}, [0]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def gen_petersen(k, j):
    n = 2 * k
    adj = [[] for _ in range(n)]

    def link(a, b):
        if b not in adj[a]:
            adj[a].append(b)
            adj[b].append(a)
    for i in range(k):
        link(i, (i + 1) % k)
        link(i, i + k)
        link(i + k, (i + j) % k + k)
    return adj


def greedy_mm(edges):
    used, c = set(), 0
    for (u, v) in edges:
        if u not in used and v not in used:
            used.add(u)
            used.add(v)
            c += 1
    return c


def gamma_e(adj):
    n = len(adj)
    edges = edges_of(adj)
    m = len(edges)
    inc = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        inc[u].append(i)
        inc[v].append(i)

    def leq(k):
        s = Solver()
        for v in range(n):
            for a, b in itertools.combinations(inc[v], 2):
                s.add_clause([-(a + 1), -(b + 1)])
        for (u, v) in edges:
            s.add_clause([(j + 1) for j in set(inc[u] + inc[v])])
        for cl in CardEnc.atmost(lits=list(range(1, m + 1)), bound=k,
                                 vpool=IDPool(start_from=m + 1),
                                 encoding=EncType.seqcounter).clauses:
            s.add_clause(cl)
        r = s.solve()
        s.delete()
        return r
    best = greedy_mm(edges)
    k = best - 1
    while k >= 0 and leq(k):
        best = k
        k -= 1
    return best


def dom_count(adj, k, cap):
    """Number of dominating sets of size <= k, capped (0 => gamma > k)."""
    n = len(adj)
    s = Solver()
    for v in range(n):
        s.add_clause([v + 1] + [u + 1 for u in adj[v]])
    for cl in CardEnc.atmost(lits=list(range(1, n + 1)), bound=k, top_id=n,
                             encoding=EncType.seqcounter).clauses:
        s.add_clause(cl)
    c = 0
    while c < cap and s.solve():
        mod = s.get_model()
        s.add_clause([-v for v in range(1, n + 1) if mod[v - 1] > 0])
        c += 1
    s.delete()
    return c


def two_swap(adj, rng):
    n = len(adj)
    edges = edges_of(adj)
    for _ in range(80):
        (a, b), (c, d) = rng.sample(edges, 2)
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            c, d = d, c
        if c in adj[a] or d in adj[b]:
            continue
        new = [list(x) for x in adj]
        new[a].remove(b); new[b].remove(a)
        new[c].remove(d); new[d].remove(c)
        new[a].append(c); new[c].append(a)
        new[b].append(d); new[d].append(b)
        if all(len(x) == 3 and len(set(x)) == 3 for x in new) and connected(new, n):
            return new
    return None


def evaluate(adj, cap=8):
    """(gamma_e, #minimum dominating sets). count 0 => COUNTEREXAMPLE."""
    ge = gamma_e(adj)
    return ge, dom_count(adj, ge, cap)


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    seeds = [("GP(25,2)", gen_petersen(25, 2)),
             ("GP(25,12)", gen_petersen(25, 12)),
             ("GP(26,2)", gen_petersen(26, 2))]
    rng = random.Random(20260728)
    t0 = time.time()
    per = budget / len(seeds)
    for name, seed in seeds:
        ge, cnt = evaluate(seed)
        print(f"\n=== seed {name}: n={len(seed)} gamma_e={ge} "
              f"#minDom={cnt} (0 would be a counterexample) ===")
        cur, best, it, ts = (seed, cnt), cnt, 0, time.time()
        while time.time() - ts < per:
            it += 1
            cand = two_swap(cur[0], rng)
            if cand is None:
                continue
            g2, c2 = evaluate(cand)
            if c2 == 0:
                print(f"  *** COUNTEREXAMPLE at iter {it}: gamma > "
                      f"gamma_e={g2} ***")
                print(f"  edges = {edges_of(cand)}")
                sys.exit(0)
            if g2 > ge:                       # gamma_e drifted up: not useful
                continue
            if c2 <= cur[1]:
                cur = (cand, c2)
                if c2 < best:
                    best = c2
                    print(f"  iter {it:5d}  gamma_e={g2}  #minDom={c2}  "
                          f"[{time.time()-ts:.0f}s]")
            elif rng.random() < 0.05:
                cur = (cand, c2)
        print(f"  iterations={it}  fewest #minDom reached={best} (need 0)")
    print(f"\ntotal {time.time()-t0:.0f}s -- no counterexample found")
