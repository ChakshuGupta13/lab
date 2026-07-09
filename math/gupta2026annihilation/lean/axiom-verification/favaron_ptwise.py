"""Test a majorization-free decomposition of the crux.

Claim A (residueAux pointwise monotone): for sorted-desc l1, l2 of equal length with
  l1[i] <= l2[i] for all i, residueAux l1 <= residueAux l2.
Claim B (decrement-largest is pointwise-minimal): sort_desc(T, k-largest decremented)
  <= sort_desc(T, any k-subset decremented), pointwise.

If BOTH hold, the crux R(HH) <= R(G-v) follows with NO majorization, NO graphicality:
  decrement-largest (=HH) <=ptwise decrement-neighbours (=G-v) [B], then residueAux
  monotone [A].
"""
import itertools
import random


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


def dec_subset(T, idxs):
    out = list(T)
    for i in idxs:
        out[i] = max(out[i] - 1, 0)
    return out


def k_largest_idxs(T, k):
    order = sorted(range(len(T)), key=lambda i: (T[i], -i), reverse=True)
    return set(order[:k])


def ptwise_le(a, b):
    return all(x <= y for x, y in zip(a, b))


# ---- Claim A: residueAux pointwise monotone on sorted-desc equal-length lists ----
rng = random.Random(7)
failA = 0
exA = []
for _ in range(300000):
    m = rng.randint(1, 8)
    l2 = sorted([rng.randint(0, 7) for _ in range(m)], reverse=True)
    # build l1 <= l2 pointwise, sorted desc
    l1 = sorted([rng.randint(0, l2[i]) for i in range(m)], reverse=True)
    # ensure still <= after independent sort: enforce pointwise by clamping
    l1 = [min(l1[i], l2[i]) for i in range(m)]
    if not (l1 == sorted(l1, reverse=True) and ptwise_le(l1, l2)):
        continue
    if residue_aux(l1) > residue_aux(l2):
        failA += 1
        if len(exA) < 8:
            exA.append((list(l1), residue_aux(l1), list(l2), residue_aux(l2)))
print(f"Claim A (residueAux ptwise-monotone): FAILURES={failA}")
for e in exA:
    print("   l1=%s R=%d   l2=%s R=%d" % e)

# ---- Claim B: decrement-k-largest is pointwise-minimal among k-subset decrements ----
failB = 0
exB = []
for _ in range(200000):
    m = rng.randint(1, 7)
    T = [rng.randint(0, 6) for _ in range(m)]
    k = rng.randint(0, m)
    big = k_largest_idxs(T, k)
    Lbig = sorted(dec_subset(T, big), reverse=True)
    for S in itertools.combinations(range(m), k):
        Ls = sorted(dec_subset(T, set(S)), reverse=True)
        if not ptwise_le(Lbig, Ls):
            failB += 1
            if len(exB) < 8:
                exB.append((list(T), k, sorted(big), Lbig, list(S), Ls))
            break
print(f"Claim B (decrement-largest ptwise-minimal): FAILURES={failB}")
for e in exB:
    print("   T=%s k=%d big=%s Lbig=%s  S=%s Ls=%s" % e)
