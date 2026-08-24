"""Is Baste's inequality sharp only at Delta=3, or at Delta=4,5,6 too?

For a Delta-regular graph the counting bound is gamma_e >= Delta*n/(2(2Delta-1)):
  Delta=3: 3n/10 = 0.3000n   Delta=5: 5n/18 = 0.2778n
  Delta=4: 2n/7  = 0.2857n   Delta=6: 6n/22 = 0.2727n

At Delta=3 equality gamma = gamma_e holds on an infinite family (GP(k,2),
k != 3 mod 5).  This scans structured Delta-regular families for equality at
Delta = 4,5,6, and also reports whether each equality case attains the counting
floor (the "tight family") -- most do not, which is why proving the tight family
would not prove the conjecture.
"""
import sys
from fractions import Fraction

from baste_perturb_equality import dom_count, edges_of, gamma_e  # noqa: E402


def circulant(n, S):
    adj = [[] for _ in range(n)]
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i and j not in adj[i]:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def torus(m, k):
    n = m * k
    adj = [[] for _ in range(n)]

    def link(a, b):
        if a != b and b not in adj[a]:
            adj[a].append(b)
            adj[b].append(a)
    for r in range(m):
        for c in range(k):
            v = r * k + c
            link(v, r * k + (c + 1) % k)
            link(v, ((r + 1) % m) * k + c)
    return adj


def hypercube(d):
    n = 1 << d
    adj = [[] for _ in range(n)]
    for v in range(n):
        for b in range(d):
            adj[v].append(v ^ (1 << b))
    return adj


def report(name, adj, delta):
    n = len(adj)
    if any(len(set(a)) != delta for a in adj):
        return None
    ge = gamma_e(adj)
    if dom_count(adj, ge, 1) == 0:
        print(f"  *** {name}: gamma > gamma_e={ge}   COUNTEREXAMPLE")
        return "CE"
    equal = dom_count(adj, ge - 1, 1) == 0
    floor = Fraction(delta * n, 2 * (2 * delta - 1))
    tight = Fraction(ge) == floor
    if equal:
        print(f"      {name:18s} n={n:3d}  gamma = gamma_e = {ge:3d}   EQUALITY"
              f"   counting floor {float(floor):.2f} "
              f"{'(TIGHT)' if tight else '(not tight)'}")
    return "EQ" if equal else "strict"


if __name__ == "__main__":
    stats = {}
    for delta, fams in (
        (4, [(f"C_{n}(1,2)", circulant(n, (1, 2))) for n in range(6, 34)]
            + [(f"C_{n}(1,3)", circulant(n, (1, 3))) for n in range(8, 30)]
            + [(f"C_{n}(1,4)", circulant(n, (1, 4))) for n in range(10, 28)]
            + [(f"T({m}x{k})", torus(m, k)) for m in (3, 4, 5, 6) for k in (3, 4, 5, 6)]
            + [("Q4", hypercube(4))]),
        (5, [(f"C_{n}(1,2,{n//2})", circulant(n, (1, 2, n // 2)))
             for n in range(8, 34, 2)]
            + [(f"C_{n}(1,3,{n//2})", circulant(n, (1, 3, n // 2)))
               for n in range(10, 32, 2)]
            + [("Q5", hypercube(5))]),
        (6, [(f"C_{n}(1,2,3)", circulant(n, (1, 2, 3))) for n in range(8, 30)]
            + [(f"C_{n}(1,2,4)", circulant(n, (1, 2, 4))) for n in range(10, 26)]),
    ):
        print(f"\n=== Delta = {delta}  (counting floor "
              f"{delta}n/{2*(2*delta-1)}) ===")
        c = {"EQ": 0, "strict": 0, "CE": 0}
        for name, adj in fams:
            r = report(name, adj, delta)
            if r:
                c[r] += 1
        stats[delta] = c
        print(f"  tested={sum(c.values())}  equality={c['EQ']}  "
              f"strict={c['strict']}  counterexamples={c['CE']}")
    print("\n=== summary ===")
    for d, c in stats.items():
        print(f"  Delta={d}: {c['EQ']} equality / {sum(c.values())} tested, "
              f"{c['CE']} counterexamples")
