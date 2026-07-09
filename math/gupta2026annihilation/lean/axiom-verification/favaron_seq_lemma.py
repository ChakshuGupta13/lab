"""Test the PURE SEQUENCE claim behind the crux:

  For a base multiset T (list of nats) and count k, is
    residueAux(sort_desc(decrement the k LARGEST of T))
    <= residueAux(sort_desc(decrement ANY k-subset S of T)) ?

If YES universally (arbitrary T, k, S), the Favaron crux is a pure sequence
lemma with NO graphicality hypothesis -> far easier to formalize.

Also test the atomic-swap form:
  moving one decrement from position i (value a) to position j (value b<=a),
  i.e. "decrement a larger element instead of a smaller" -> residue does not increase.
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


def decrement_subset(T, idxs):
    out = list(T)
    for i in idxs:
        out[i] = max(out[i] - 1, 0)
    return out


def k_largest_idxs(T, k):
    order = sorted(range(len(T)), key=lambda i: T[i], reverse=True)
    return set(order[:k])


def main():
    rng = random.Random(999)
    fails = 0
    ex = []
    trials = 0
    for _ in range(200000):
        m = rng.randint(1, 7)
        T = [rng.randint(0, 6) for _ in range(m)]
        k = rng.randint(0, m)
        big = k_largest_idxs(T, k)
        Rbig = residue_aux(decrement_subset(T, big))
        # test against all k-subsets (or a sample if too many)
        subsets = list(itertools.combinations(range(m), k))
        for S in subsets:
            trials += 1
            Rs = residue_aux(decrement_subset(T, set(S)))
            if Rbig > Rs:
                fails += 1
                if len(ex) < 8:
                    ex.append((list(T), k, sorted(big), Rbig, list(S), Rs))
                break
    print(f"subset-comparisons trials={trials}")
    print(f"FAILURES (decrement-k-largest gave residue > some other k-subset): {fails}")
    for e in ex:
        print("  T=%s k=%d big=%s Rbig=%d  S=%s Rs=%d" % e)

    # atomic swap test: T sorted desc, pick i<j (so T[i]>=T[j]); compare
    #   decrement i (larger)  vs  decrement j (smaller)
    print("\n=== atomic swap: decrement larger vs smaller single element ===")
    fails2 = 0
    ex2 = []
    for _ in range(200000):
        m = rng.randint(2, 8)
        T = sorted([rng.randint(0, 6) for _ in range(m)], reverse=True)
        i = rng.randint(0, m - 1)
        j = rng.randint(0, m - 1)
        if T[i] < T[j]:
            i, j = j, i  # ensure T[i] >= T[j]
        if i == j:
            continue
        Ri = residue_aux(decrement_subset(T, {i}))  # decrement larger
        Rj = residue_aux(decrement_subset(T, {j}))  # decrement smaller
        if Ri > Rj:
            fails2 += 1
            if len(ex2) < 8:
                ex2.append((list(T), i, T[i], Ri, j, T[j], Rj))
    print(f"FAILURES (decrement-larger residue > decrement-smaller): {fails2}")
    for e in ex2:
        print("  T=%s  dec_i idx%d val%d ->R=%d   dec_j idx%d val%d ->R=%d" % e)


if __name__ == "__main__":
    main()
