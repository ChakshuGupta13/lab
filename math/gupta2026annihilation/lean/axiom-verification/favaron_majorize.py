"""Pin down the correct majorization target lemma by exhaustive enumeration over
graphical sequences (small n).

Majorization: a >= b (a majorizes b) if sorted-desc, same sum, and every prefix sum
of a >= prefix sum of b.

Claim S (residue Schur-convex on graphical): if a, b BOTH graphical, same length,
a majorizes b, then residue(a) >= residue(b).

Claim H (HH preserves majorization on graphical): a,b graphical same length, a maj b,
head-processing... test residue directly instead (Claim S is what we need).

The crux needs: sigma (decrement neighbours) majorizes pi' (decrement Delta-largest),
both graphical -> residue(sigma) >= residue(pi').  Since residue(G)=residue(pi') and
residue(G-v)=residue(sigma), that gives R(G) <= R(G-v).
"""
import itertools


def hh_step(s):
    if not s:
        return []
    d = s[0]
    rest = s[1:]
    dec = [max(x - 1, 0) for x in rest[:d]]
    out = dec + rest[d:]
    out.sort(reverse=True)
    return out


def residue_aux(l):
    l = sorted(l, reverse=True)
    while True:
        if not l:
            return 0
        if l[0] == 0:
            return 1 + (len(l) - 1)
        l = hh_step(l)


def is_graphical(seq):
    s = sorted(seq, reverse=True)
    if sum(s) % 2 != 0:
        return False
    while s and s[0] > 0:
        d = s[0]
        s = s[1:]
        if d > len(s):
            return False
        for i in range(d):
            s[i] -= 1
            if s[i] < 0:
                return False
        s.sort(reverse=True)
    return all(x == 0 for x in s)


def majorizes(a, b):
    # a, b sorted desc, same length, same sum; a majorizes b
    if sum(a) != sum(b) or len(a) != len(b):
        return False
    pa = pb = 0
    for x, y in zip(a, b):
        pa += x
        pb += y
        if pa < pb:
            return False
    return True


def all_graphical(n, maxdeg):
    seqs = []
    for combo in itertools.product(range(maxdeg + 1), repeat=n):
        s = tuple(sorted(combo, reverse=True))
        if s in seqs:
            continue
        if is_graphical(s):
            seqs.append(s)
    return sorted(set(seqs))


for n in range(1, 8):
    seqs = all_graphical(n, n - 1)
    failS = 0
    exS = []
    for a in seqs:
        for b in seqs:
            if a == b:
                continue
            if majorizes(list(a), list(b)):
                # a majorizes b, both graphical
                if residue_aux(a) < residue_aux(b):
                    failS += 1
                    if len(exS) < 4:
                        exS.append((a, residue_aux(a), b, residue_aux(b)))
    tag = "OK (Schur-convex holds)" if failS == 0 else f"FAILS ({failS})"
    print(f"n={n}: {len(seqs)} graphical seqs, Claim S {tag}")
    for e in exS:
        print("    a=%s R=%d  maj  b=%s R=%d  (R(a)<R(b)!)" % e)
