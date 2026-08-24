#!/usr/bin/env python3
"""ADVERSARY independent verification of thm:cex (the 50-vertex graph).
Everything is recomputed from scratch. The formula is treated as untrusted input.
"""
import itertools
from pysat.solvers import Cadical195 as Solver
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

# The formula the paper builds its graph from (Berman-Karpinski-Scott 20-clause).
F = [[1,4,-6],[2,-4,5],[2,-5,6],[4,5,6],[-4,-5,-6],
     [1,7,-9],[-2,-7,8],[-2,-8,9],[7,8,9],[-7,-8,-9],
     [-1,10,-12],[3,-10,11],[3,-11,12],[10,11,12],[-10,-11,-12],
     [-1,13,-15],[-3,-13,14],[-3,-14,15],[13,14,15],[-13,-14,-15]]

print("="*70)
print("PART A: thm:cex — the 50-vertex counterexample graph")
print("="*70)

# ---- A1: formula structure ----
nvar = max(abs(l) for c in F for l in c)
nclause = len(F)
print(f"[A1] #clauses={nclause}  #vars={nvar}  (paper: 20 clauses, 15 vars)")
assert nclause == 20 and nvar == 15
# 3-CNF, distinct vars per clause
for c in F:
    assert len(c) == 3, f"clause not width 3: {c}"
    assert len({abs(l) for l in c}) == 3, f"repeated var in clause: {c}"
# each literal exactly twice
from collections import Counter
lit = Counter(l for c in F for l in c)
bad = [(v,s) for v in range(1,nvar+1) for s in (v,-v) if lit[s]!=2]
print(f"[A1] every literal occurs exactly twice: {not bad}  (violations: {bad})")
assert not bad

# ---- A2: unsatisfiable (my own solve) ----
with Solver(bootstrap_with=F) as s:
    sat = s.solve()
print(f"[A2] formula SAT? {sat}  (paper needs UNSAT)")
assert sat is False

# ---- A3: build incidence graph from the paper's definition ----
# var i -> edge (a_i,b_i); clause j -> z_j; literal +i in clause j -> z_j~a_i; -i -> z_j~b_i
def a(i): return 2*(i-1)
def b(i): return 2*(i-1)+1
def z(j): return 2*nvar + j
n = 2*nvar + nclause
adj = [set() for _ in range(n)]
def add(u,v): adj[u].add(v); adj[v].add(u)
M = []
for i in range(1,nvar+1):
    add(a(i),b(i)); M.append((a(i),b(i)))
for j,c in enumerate(F):
    for l in c:
        i = abs(l)
        add(z(j), a(i) if l>0 else b(i))
print(f"[A3] n = {n}  (paper: 50)")
assert n == 50

# ---- A4: cubic, simple, connected ----
degs = [len(s) for s in adj]
cubic = all(d==3 for d in degs)
simple = all(u not in adj[u] for u in range(n)) and all(len(adj[u])==len(set(adj[u])) for u in range(n))
# connected
seen={0}; st=[0]
while st:
    u=st.pop()
    for v in adj[u]:
        if v not in seen: seen.add(v); st.append(v)
conn = len(seen)==n
print(f"[A4] cubic={cubic}  simple={simple}  connected={conn}  degset={set(degs)}")
assert cubic and simple and conn

# ---- A5: M is a maximal matching, |M|=15 ----
Z = set(range(2*nvar, n))  # the clause vertices, unsaturated by M
# maximality <=> Z independent
Zind = all(v not in adj[u] for u in Z for v in Z if v>u)
print(f"[A5] |M|={len(M)}  Z(unsat)={len(Z)}  Z independent (=> M maximal)? {Zind}")
assert len(M)==15 and Zind

# ---- A6/A7: min maximal matching size and uniqueness (my own SAT model) ----
# edge vars; matching (<=1 incident per vertex) + maximality (every edge covered)
edges = sorted({(min(u,v),max(u,v)) for u in range(n) for v in adj[u]})
eid = {e:k+1 for k,e in enumerate(edges)}   # 1-based lits
inc = [[] for _ in range(n)]                # incident edge-ids per vertex
for (u,v) in edges:
    inc[u].append(eid[(u,v)]); inc[v].append(eid[(u,v)])

def maximal_matching_clauses():
    cl = []
    # matching: at most one incident edge per vertex
    for u in range(n):
        if len(inc[u])>=2:
            for e,f in itertools.combinations(inc[u],2):
                cl.append([-e,-f])
    # maximality: every edge e={u,v} has some selected edge incident to u or v
    for (u,v) in edges:
        cover = set(inc[u]) | set(inc[v])
        cl.append(sorted(cover))
    return cl

def min_maximal_matching():
    base = maximal_matching_clauses()
    m = len(edges)
    best = None
    for k in range(1, m+1):
        vp = IDPool(start_from=m+1)
        card = CardEnc.atmost(lits=list(range(1,m+1)), bound=k, vpool=vp, encoding=EncType.seqcounter)
        with Solver(bootstrap_with=base+card.clauses) as s:
            if s.solve():
                best = k; break
    return best

mmm = min_maximal_matching()
print(f"[A6] minimum maximal matching size = {mmm}  (paper: 15 = gamma_e)")
assert mmm == 15

# uniqueness: enumerate all maximal matchings of size exactly 15
def all_min_maximal(k):
    base = maximal_matching_clauses()
    m = len(edges)
    vp = IDPool(start_from=m+1)
    card = CardEnc.equals(lits=list(range(1,m+1)), bound=k, vpool=vp, encoding=EncType.seqcounter)
    sols = []
    with Solver(bootstrap_with=base+card.clauses) as s:
        while s.solve():
            model = s.get_model()
            chosen = frozenset(e for e in range(1,m+1) if model[e-1]>0)
            sols.append(chosen)
            # with exact-15 cardinality, forbidding "all of chosen true" removes exactly this MMM
            s.add_clause([-e for e in chosen])
            if len(sols)>6: break
    return sols

mins = all_min_maximal(15)
Mset = frozenset(eid[(min(u,v),max(u,v))] for (u,v) in M)
is_M = (len(mins)==1 and mins[0]==Mset)
print(f"[A7] # minimum maximal matchings of size 15 = {len(mins)}  unique&equals M? {is_M}")

# ---- A8: gamma(G) via SAT min dominating set ----
def dominating_clauses():
    cl = []
    for v in range(n):
        cl.append([v+1] + [u+1 for u in adj[v]])   # v dominated
    return cl

def gamma():
    base = dominating_clauses()
    for k in range(1, n+1):
        vp = IDPool(start_from=n+1)
        card = CardEnc.atmost(lits=list(range(1,n+1)), bound=k, vpool=vp, encoding=EncType.seqcounter)
        with Solver(bootstrap_with=base+card.clauses) as s:
            if s.solve():
                return k, [v for v in range(n) if s.get_model()[v]>0]
    return None,None

g, dset = gamma()
print(f"[A8] gamma(G) = {g}  (paper: 14)   dominating set size {len(dset)}")
assert g == 14

# ---- A9: gamma < gamma_e, Baste HOLDS; the 14-set is not a transversal of M ----
print(f"[A9] gamma={g} < gamma_e={mmm}: Baste HOLDS (14<15). "
      f"14-set cannot be a transversal (would need |M|=15 vertices).")
assert g < mmm

# ---- A10: Phi(G,M) equals the formula and is UNSAT (no dominating transversal) ----
# By construction Phi = F; re-confirm no transversal of M dominates Z by brute reasoning:
# each transversal = choice of one endpoint per M-edge = truth assignment to 15 vars.
# u (clause vertex) covered  <=> some neighbour chosen <=> its clause satisfied.
# So a dominating transversal <=> F satisfiable. F is UNSAT (A2) => none exists.
print(f"[A10] Phi(G,M)=F is UNSAT => no dominating transversal of M. "
      f"With M unique min maximal matching, the reduction cannot prove gamma<=gamma_e here.")
print("\nPART A COMPLETE — all assertions passed." if is_M else
      "\nPART A: uniqueness needs manual check (see A7).")
