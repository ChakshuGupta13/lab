"""
Relation (B): verifies the full counting identity
T0 v1 = v1 and the uniqueness mechanism, closing onset <= d-1.

The identity, for every compatible (b,c):
    #{ valid a } = [ b has exactly one boundary extremum (right c) ],
where a valid a is compatible with b, has exactly one boundary extremum (right b),
and satisfies flip(a,b,c)=0 (destroys all of b's extrema).

Complete proof (heights: b<->beta with steps e; c<->gamma, sigma=gamma-beta;
a<->alpha with steps e', rho=beta-alpha; all +-1 walks):

1. PINNING. To destroy b's extremum at cell i (colour kappa_i), and be compatible,
   a_i must be the third colour: a_i=b_i+1 at a max (rho_i=-1), b_i-1 at a min
   (rho_i=+1).
2. CRUX-ON-a (the g-walk flip argument, proved for any boundary column): a valid a
   has alpha UNIMODAL -- one local max at some p (e'=+1 before p, -1 after) if a's
   extremum is a max, one local min if a min.
3. rho CONSTANT (uniqueness). From e'_i = e_i - (rho_{i+1}-rho_i):
   - i<p (e'=+1): rho_{i+1}-rho_i = e_i-1 in {0,-2}  => rho non-increasing on [0,p];
   - i>=p (e'=-1): rho_{i+1}-rho_i = e_i+1 in {0,+2}  => rho non-decreasing on [p,.].
   a's extremum at p is a max => rho_p=-1. Exactly-one-extremum forbids corner
   extrema: cell 0 (e'_0=+1) needs rho_0=-1; cell m-1 (e'_{m-2}=-1) needs rho_{m-1}=-1.
   Non-increasing from rho_0=-1 to rho_p=-1 and non-decreasing back to rho_{m-1}=-1,
   with rho in {+-1}, forces rho == -1. So a = b+1. (min case: rho==+1, a=b-1.)
4. SHIFT ANALYSIS. a=b+1 (a above b everywhere) destroys b's maxima but not minima,
   and has extrema = b's local maxima; so it is valid iff b has exactly one extremum
   and it is a max. Symmetrically a=b-1 iff exactly one min. Hence #valid a = [b has
   exactly one extremum] (Case 1 -> the matching shift, unique; Case 2 (>=2) -> none).

This is Q0 v1 = 0 (v1 in eig-1 of T0), the numerator ingredient of onset <= d-1.
"""
from __future__ import annotations

from transfer_matrix import columns, compatible, flip
from cancellation_test import pair_states


def bdy_count(a, b, m):
    n = 0
    for i in range(m):
        nb = []
        if i > 0:
            nb.append(a[i - 1])
        if i < m - 1:
            nb.append(a[i + 1])
        nb.append(b[i])
        if len(set(nb)) == 1:
            n += 1
    return n


def main() -> None:
    print("relation (B): full identity  #valid a = [b has exactly 1 extremum]")
    all_ok = True
    for m in range(2, 9):
        cols = columns(m)
        S, _ = pair_states(m)
        ident_ok = shift_ok = True
        for (b, c) in S:
            nb_ex = bdy_count(b, c, m)
            valid = [a for a in cols
                     if compatible(a, b) and bdy_count(a, b, m) == 1
                     and flip(a, b, c) == 0]
            if len(valid) != (1 if nb_ex == 1 else 0):
                ident_ok = False
            for a in valid:                      # every valid a is a uniform shift
                diffs = {(a[i] - b[i]) % 3 for i in range(m)}
                if len(diffs) != 1 or 0 in diffs:
                    shift_ok = False
        ok = ident_ok and shift_ok
        all_ok = all_ok and ok
        print(f"  m={m}: T0 v1 = v1: {ident_ok}   every valid a a shift: {shift_ok}"
              f"   [{'OK' if ok else 'FAIL'}]")
    print(f"\nrelation (B) fully verified for m=2..8: {all_ok}")


if __name__ == "__main__":
    main()
