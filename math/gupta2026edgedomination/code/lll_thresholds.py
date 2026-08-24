#!/usr/bin/env python3
r"""LLL thresholds for gamma <= gamma_e on r-regular graphs, and a brute-force
validation of the atomic lopsided-LLL negative-dependency lemma.

Setup (covering transversal of a maximal matching M of an r-regular graph G):
- For each M-edge e_i = {p_i, q_i}, pick one endpoint uniformly at random -> D.
- An "at-risk" vertex u in Z = V \ V(M) has all r neighbours on r DISTINCT M-edges,
  one endpoint each; u is uncovered iff every such edge picks the OTHER endpoint.
- Bad event B_u = "u uncovered" is an ATOM: B_u = AND_i (Z_{e_i} = away-from-u),
  Pr[B_u] = 2^-r.

Dependencies:
- FULL (share-a-variable, Moser-Tardos variable model): B_u ~ B_{u'} iff they share
  an M-edge. Worst-case degree d_full = r(2r-3): per edge, u's-endpoint side has <= r-2
  other Z-neighbours, opposite endpoint has <= r-1; sum (r-2)+(r-1)=2r-3, times r edges.
- LOPSIDED (atomic conflict graph): two atoms are lopsi-dependent iff they DISAGREE on a
  shared variable. B_u and B_{u'} share edge e; they CONFLICT iff u,u' are adjacent to
  OPPOSITE endpoints of e (want opposite picks); if adjacent to the SAME endpoint they
  AGREE (both want e to pick the other endpoint) -> positively correlated -> dropped.
  Only the opposite-endpoint (<= r-1 per edge) side survives: d_lopsi = r(r-1).

The lopsided LLL is valid here because the events are atoms over INDEPENDENT variables:
for such atoms, conditioning on avoiding any set of NON-conflicting (agreeing) atoms does
not increase Pr[B_u] (verified by brute force below). Only EXISTENCE of a good transversal
is needed (existential Erdos-Spencer lopsided LLL), not an algorithm.
"""
import math
from itertools import product

E = math.e


def shearer_regular(p, d):
    """Exact Shearer threshold for a d-regular dependency graph: p <= (d-1)^(d-1)/d^d."""
    if d == 0:
        return True
    pstar = (d - 1) ** (d - 1) / (d ** d)
    return p <= pstar


def threshold_table():
    print("r |  p=2^-r  | FULL d=r(2r-3)      | LOPSI d=r(r-1)")
    print("  |          | e*p*(d+1)  Shearer  | e*p*(d+1)  Shearer")
    for r in range(3, 14):
        p = 2.0 ** (-r)
        dF, dL = r * (2 * r - 3), r * (r - 1)
        symF, symL = E * p * (dF + 1), E * p * (dL + 1)
        print(f"{r:2d}| {p:.5f} | {symF:6.3f} {'OK' if symF <= 1 else 'x ':>2}"
              f"  {'OK' if shearer_regular(p, dF) else 'x ':>3}    |"
              f" {symL:6.3f} {'OK' if symL <= 1 else 'x ':>2}"
              f"  {'OK' if shearer_regular(p, dL) else 'x ':>3}")
    print()
    for label, ok in [
        ("FULL  sym", lambda r: E * 2 ** -r * (r * (2 * r - 3) + 1) <= 1),
        ("FULL  Shearer", lambda r: shearer_regular(2.0 ** -r, r * (2 * r - 3))),
        ("LOPSI sym", lambda r: E * 2 ** -r * (r * (r - 1) + 1) <= 1),
        ("LOPSI Shearer", lambda r: shearer_regular(2.0 ** -r, r * (r - 1))),
    ]:
        r = 3
        while r < 60 and not ok(r):
            r += 1
        print(f"  threshold {label:14s}: r >= {r}")


def validate_negative_dependency():
    """Brute-force check of the crux lemma on explicit tiny instances.

    Each atom is a dict {edge_index: chosen_endpoint} over independent uniform
    variables Z_i in {0,1}. Verify: conditioning on avoiding a set S of atoms that
    each AGREE with B (no shared variable disagrees) never increases Pr[B]; and that
    a CONFLICTING atom can increase it (so it must be kept).
    """
    def pr(atoms_true, atoms_false, nvars):
        """Pr[ (AND atoms_true) AND (AND NOT atoms_false) ] under uniform product."""
        num = 0
        total = 2 ** nvars
        for assign in product([0, 1], repeat=nvars):
            if all(assign[i] == v for a in atoms_true for i, v in a.items()) and \
               all(not all(assign[i] == v for i, v in a.items()) for a in atoms_false):
                num += 1
        return num / total

    def conditional(B, avoid, nvars):
        denom = pr([], avoid, nvars)
        if denom == 0:
            return 0.0
        return pr([B], avoid, nvars) / denom

    nvars = 4  # 4 independent edges
    B = {0: 1, 1: 1}                      # B_u: edge0->1, edge1->1  (Pr=1/4)
    agreeing = [
        {0: 1, 2: 1},                    # shares edge0, agrees (wants 1); extra edge2
        {1: 1, 3: 1},                    # shares edge1, agrees; extra edge3
        {0: 1, 1: 1, 2: 0},              # shares edge0,edge1 both agree; extra edge2
        {2: 0, 3: 1},                    # shares NO edge with B (independent)
    ]
    conflicting = [
        {0: 0, 2: 1},                    # shares edge0, DISAGREES (wants 0)
        {1: 0},                          # shares edge1, DISAGREES
    ]
    base = 0.25
    print(f"Pr[B] = {base}")
    print("  agreeing / independent atoms avoided -> Pr[B | .] must be <= Pr[B]:")
    ok_all = True
    # every subset of the agreeing set
    from itertools import combinations
    for k in range(1, len(agreeing) + 1):
        for S in combinations(agreeing, k):
            c = conditional(B, list(S), nvars)
            flag = "OK" if c <= base + 1e-12 else "VIOLATION"
            if c > base + 1e-12:
                ok_all = False
                print(f"    avoid {list(S)} -> {c:.4f}  {flag}")
    print(f"    all {2**len(agreeing)-1} nonempty agreeing-subsets satisfy <= Pr[B]: {ok_all}")
    print("  conflicting atoms avoided -> Pr[B | .] may EXCEED Pr[B] (must be kept):")
    for a in conflicting:
        c = conditional(B, [a], nvars)
        print(f"    avoid {a} -> {c:.4f}  {'increases' if c > base + 1e-12 else 'no-increase'}")
    return ok_all


if __name__ == "__main__":
    print("=== LLL thresholds (full vs lopsided dependency) ===")
    threshold_table()
    print()
    print("=== Crux lemma: atomic lopsided negative-dependency (brute force) ===")
    ok = validate_negative_dependency()
    print()
    print("LOAD-BEARING CLAIM validated:" if ok else "LEMMA FAILED:",
          "agreeing atoms are droppable; conflicting atoms are not.")
