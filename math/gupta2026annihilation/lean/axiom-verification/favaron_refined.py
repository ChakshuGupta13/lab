"""Find the MINIMAL constraint under which
    residueAux(decrement the k-largest of T) <= residueAux(decrement any valid k-subset)
holds. The max-degree graph deletion provides:
  (C1) k = Delta >= max(T)          [v is a MAX-degree vertex]
  (C2) decremented entries are all POSITIVE (>=1)  [neighbours have degree>=1]
  (C3) T has >= k positive entries  [v has k neighbours]
  (C4) (Delta :: T) is graphical    [full Erdos-Gallai]

Test which combination kills all counterexamples.
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


def is_graphical(seq):
    s = sorted(seq, reverse=True)
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


def k_largest_idxs(T, k):
    order = sorted(range(len(T)), key=lambda i: (T[i], -i), reverse=True)
    return set(order[:k])


def test(constraint_name, check):
    rng = random.Random(2026)
    fails = 0
    tested = 0
    ex = []
    for _ in range(300000):
        m = rng.randint(1, 7)
        T = [rng.randint(0, 6) for _ in range(m)]
        k = rng.randint(0, m)
        if not check(T, k):
            continue
        big = k_largest_idxs(T, k)
        # big must be all-positive to be a legal decrement target under C2
        if any(T[i] == 0 for i in big):
            continue
        Rbig = residue_aux(dec_subset(T, big))
        pos = [i for i in range(m) if T[i] > 0]
        for S in itertools.combinations(pos, k):
            tested += 1
            Rs = residue_aux(dec_subset(T, set(S)))
            if Rbig > Rs:
                fails += 1
                if len(ex) < 6:
                    ex.append((list(T), k, sorted(big), Rbig, list(S), Rs))
                break
    print(f"[{constraint_name}] tested={tested} FAILURES={fails}")
    for e in ex:
        print("   T=%s k=%d big=%s Rbig=%d S=%s Rs=%d" % e)


# C2+C3 only: decrement positives, T has >=k positives
test("C2+C3 (positives only)",
     lambda T, k: sum(1 for x in T if x > 0) >= k)

# C1+C2+C3: also k >= max(T)
test("C1+C2+C3 (k>=maxT, positives)",
     lambda T, k: k >= (max(T) if T else 0) and sum(1 for x in T if x > 0) >= k)

# C4: (k :: T) graphical  (full)
test("C4 ((k::T) graphical)",
     lambda T, k: is_graphical([k] + T))
