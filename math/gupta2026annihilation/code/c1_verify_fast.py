#!/usr/bin/env python3
"""
Fast exhaustive check of Theorem 1 (Delta*alpha >= a + R) computing the
independence number directly, over all connected graphs in a given order range.

The vetted reference `c1_verify.py` computes alpha via networkx max-clique of the
complement, which is too slow to reach order ten (11.7M graphs at n=10). This
script keeps the SAME annihilation/residue definitions but replaces the alpha
computation with an in-house g6 parser plus a bitset maximum-independent-set
recursion (vertices <= 10 fit in a machine word), and avoids networkx in the hot
loop entirely.

Scope of Theorem 1: nontrivial connected graphs other than K_2, i.e. every
connected graph with Delta >= 2, i.e. every connected graph on >= 3 vertices.

Modes
  --check        cross-validate the in-house alpha (own g6 parser + bitset MIS)
                 against c1_verify.py (networkx max-clique of the complement) on
                 all connected graphs for n in [nmin, nmax]; must match exactly.
                 (a and R are byte-identical copies of the vetted functions, so
                 their --check comparison is a consistency check, not an
                 independent oracle.)
  (default)      run the Theorem-1 check; report per-order counts (compare to
                 OEIS A001349) and total violations (expected 0).

Usage:
  python c1_verify_fast.py --check 3 8        # correctness gate
  python c1_verify_fast.py 3 10               # full run to order ten
Requires: geng on PATH (nauty). networkx only in --check mode.
"""
import sys
import subprocess


# ---- annihilation number and residue: copied verbatim from c1_verify.py ----
def annihilation_number(degs, m):
    """max j : sum of j smallest degrees <= m."""
    asc = sorted(degs)
    tot, a = 0, 0
    for d in asc:
        tot += d
        if tot <= m:
            a += 1
        else:
            break
    return a


def residue(degs):
    """#zeros after Havel-Hakimi terminates."""
    seq = sorted(degs, reverse=True)
    while seq and seq[0] > 0:
        d = seq[0]
        seq = seq[1:]
        if d > len(seq):
            raise ValueError("non-graphic during HH")
        for k in range(d):
            seq[k] -= 1
        seq.sort(reverse=True)
    return len(seq)


# ---- in-house graph6 parser (McKay format, n <= 62) -> adjacency bitmasks ----
def parse_g6(s):
    data = [ord(c) - 63 for c in s]
    n = data[0]
    adj = [0] * n
    idx = 0
    bitbuf = 0
    nbits = 0
    payload = data[1:]
    pi = 0

    def next_bit():
        nonlocal bitbuf, nbits, pi
        if nbits == 0:
            bitbuf = payload[pi]
            pi += 1
            nbits = 6
        nbits -= 1
        return (bitbuf >> nbits) & 1

    for j in range(1, n):
        for i in range(j):
            if next_bit():
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return n, adj


# ---- bitset maximum independent set (alpha) ----
def alpha_bitset(n, adj):
    full = (1 << n) - 1

    def rec(avail):
        if avail == 0:
            return 0
        # pivot: highest-degree-in-subgraph vertex shrinks the include-branch most
        best_v = -1
        best_pop = -1
        a = avail
        while a:
            v = (a & -a).bit_length() - 1
            a &= a - 1
            pop = bin(adj[v] & avail).count("1")
            if pop > best_pop:
                best_pop = pop
                best_v = v
        v = best_v
        without_v = avail & ~(1 << v)
        # exclude v
        best = rec(without_v)
        # include v: drop v and its neighbours
        inc = 1 + rec(without_v & ~adj[v])
        return inc if inc > best else best

    return rec(full)


def geng_stream(n):
    proc = subprocess.Popen(["geng", "-qc", str(n)],
                            stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        g6 = line.strip()
        if g6:
            yield g6
    proc.wait()


def run_check(nmin, nmax):
    """Cross-validate alpha/a/R against the vetted networkx reference."""
    import c1_verify as ref  # networkx-based, vetted
    import networkx as nx
    bad = 0
    total = 0
    for n in range(nmin, nmax + 1):
        for g6 in geng_stream(n):
            G = nx.from_graph6_bytes(g6.encode())
            degs = [d for _, d in G.degree()]
            m = G.number_of_edges()
            n2, adj = parse_g6(g6)
            a_fast = alpha_bitset(n2, adj)
            a_ref = ref.independence_number(G)
            if a_fast != a_ref:
                bad += 1
                if bad <= 10:
                    print(f"  ALPHA MISMATCH g6={g6} fast={a_fast} ref={a_ref}")
            # a and R use copied functions; sanity check against ref module
            if annihilation_number(degs, m) != ref.annihilation_number(degs, m):
                bad += 1
                print(f"  A MISMATCH g6={g6}")
            if residue(degs) != ref.residue(degs):
                bad += 1
                print(f"  R MISMATCH g6={g6}")
            total += 1
        print(f"  n={n}: validated {total} graphs cumulative", flush=True)
    print(f"CHECK: {total} graphs, {bad} mismatches -> "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


def run_theorem(nmin, nmax):
    """Direct Theorem-1 check computing alpha; report counts and violations."""
    grand = 0
    viol = 0
    viol_examples = []
    for n in range(nmin, nmax + 1):
        cnt = 0
        for g6 in geng_stream(n):
            nn, adj = parse_g6(g6)
            degs = [bin(row).count("1") for row in adj]
            m = sum(degs) // 2
            Delta = max(degs)
            if Delta < 2:
                continue  # K_2 (Delta=1) is the excluded degenerate case
            a = annihilation_number(degs, m)
            R = residue(degs)
            al = alpha_bitset(nn, adj)
            if Delta * al < a + R:
                viol += 1
                if len(viol_examples) < 10:
                    viol_examples.append((g6, Delta, al, a, R))
            cnt += 1
        grand += cnt
        print(f"  n={n}: {cnt} connected graphs with Delta>=2 checked "
              f"(cumulative {grand})", flush=True)
    print(f"TOTAL connected graphs Delta>=2, order {nmin}..{nmax}: {grand}")
    print(f"Theorem 1 violations (Delta*alpha < a+R): {viol}")
    for e in viol_examples:
        print("  VIOL", e)
    print("VERDICT:", "THEOREM 1 HOLDS (0 violations)" if viol == 0
          else f"FAILS ({viol})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--check":
        nmin = int(args[1]) if len(args) > 1 else 3
        nmax = int(args[2]) if len(args) > 2 else 8
        ok = run_check(nmin, nmax)
        sys.exit(0 if ok else 1)
    else:
        nmin = int(args[0]) if len(args) > 0 else 3
        nmax = int(args[1]) if len(args) > 1 else 10
        run_theorem(nmin, nmax)
