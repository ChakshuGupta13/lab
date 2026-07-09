"""Verify the graph-deletion induction structure for Favaron's R <= alpha,
against the EXACT residueAux definition in Invariants.lean.

residueAux (Lean):
  [] -> 0
  0 :: s -> 1 + s.length
  d :: rest (d != 0) -> residueAux (havelHakimiStep (d :: rest))
havelHakimiStep (d :: rest): (rest.splitAt d) -> decrement first d by 1 (nat, so 0-1=0)
  -> (decremented ++ remaining).mergeSort (>=)   [descending]

Crux to verify: for a max-degree vertex v, R(G) <= R(G - v).
Then R(G) <= R(G-v) <= alpha(G-v) <= alpha(G) closes by induction.
"""
import itertools
import random


def havel_hakimi_step(s):
    # s is a list; assume caller passes it; mimic Lean exactly
    if not s:
        return []
    d = s[0]
    rest = s[1:]
    to_decrement = rest[:d]
    remaining = rest[d:]
    decremented = [max(x - 1, 0) for x in to_decrement]  # nat subtraction
    out = decremented + remaining
    out.sort(reverse=True)  # descending, matches mergeSort (>=)
    return out


def residue_aux(l):
    # exact port; l must be descending-sorted for meaningfulness
    while True:
        if not l:
            return 0
        if l[0] == 0:
            return 1 + (len(l) - 1)  # 1 + s.length  where s = l[1:]
        l = havel_hakimi_step(l)


def degrees_desc(adj, n):
    ds = [sum(adj[i]) for i in range(n)]
    ds.sort(reverse=True)
    return ds


def residue_graph(adj, n):
    return residue_aux(degrees_desc(adj, n))


def alpha_graph(adj, verts):
    # brute force max independent set over given vertex list
    best = 0
    vl = list(verts)
    for r in range(len(vl), -1, -1):
        if r <= best:
            break
        found = False
        for S in itertools.combinations(vl, r):
            ok = True
            for a, b in itertools.combinations(S, 2):
                if adj[a][b]:
                    ok = False
                    break
            if ok:
                found = True
                break
        if found:
            best = max(best, r)
            break
    return best


def random_graph(n, p, rng):
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i][j] = adj[j][i] = 1
    return adj


def induced(adj, verts):
    # return adjacency restricted to verts (as full n x n but only verts matter)
    return adj


def main():
    rng = random.Random(12345)
    fails_main = 0
    fails_mono = 0
    fails_mono_any = 0
    trials = 0
    worst = []
    for _ in range(40000):
        n = rng.randint(1, 8)
        p = rng.random()
        adj = random_graph(n, p, rng)
        verts = list(range(n))
        R = residue_graph(adj, n)
        A = alpha_graph(adj, verts)
        trials += 1
        if R > A:
            fails_main += 1
            if len(worst) < 5:
                worst.append(("MAIN", degrees_desc(adj, n), R, A))
        # deletion step: the crux R(G) <= R(G-v) for a max-degree v.
        # NOTE (adversary 2026-07-08): the crux is FALSE on edgeless graphs
        # (max degree 0): there R(G)=n but R(G-v)=n-1, so n<=n-1 fails. The Lean
        # statement therefore requires 0 < G.degree v; the branch guard below
        # (max degree >= 1) tests EXACTLY that positive-degree region.
        degs = [sum(adj[i]) for i in range(n)]
        if max(degs, default=0) >= 1:
            v = max(range(n), key=lambda i: degs[i])  # a max-degree vertex, deg v >= 1
            rem = [i for i in verts if i != v]
            # residue of G - v: degrees within rem
            ds_rem = sorted((sum(adj[i][j] for j in rem) for i in rem), reverse=True)
            Rv = residue_aux(ds_rem)
            if not (R <= Rv):
                fails_mono += 1
                if len([w for w in worst if w[0] == "MONO"]) < 5:
                    worst.append(("MONO", degrees_desc(adj, n), R, Rv))
            # also test: does SOME vertex v work (weaker requirement)?
            any_ok = False
            for u in verts:
                rem_u = [i for i in verts if i != u]
                ds_u = sorted((sum(adj[i][j] for j in rem_u) for i in rem_u), reverse=True)
                if R <= residue_aux(ds_u):
                    any_ok = True
                    break
            if not any_ok:
                fails_mono_any += 1

    print(f"trials={trials}")
    print(f"R <= alpha FAILURES: {fails_main}")
    print(f"R(G) <= R(G - maxdeg_v) FAILURES: {fails_mono}")
    print(f"no vertex u with R(G) <= R(G-u) FAILURES: {fails_mono_any}")
    for w in worst:
        print("  ", w)


if __name__ == "__main__":
    main()
