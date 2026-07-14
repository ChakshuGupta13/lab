"""
Verification for the boundary-column lemma.

Lemma: a boundary column (first/last column of the grid, i.e. no neighbour on one
side) of any grid height function carries at least one strict local extremum.

Two independent checks:
  (1) GROUND TRUTH -- over all compatible colour-column pairs (a,b), the minimum
      number of extrema of a boundary column equals 1 (never 0).
  (2) The PROOF MODEL -- the lemma is proved by reducing "column a has no extremum"
      to a constraint system on the height-steps e_i = h_{i+1}-h_i in {+1,-1} and
      the left/right signs sigma_i = g_i - h_i in {+1,-1}:
        * g-walk validity : e_i + (sigma_{i+1}-sigma_i) in {+1,-1}   for all i
        * no extremum at cell 0      : e_0 != sigma_0
        * no extremum at interior i  : NOT( -e_{i-1} = e_i = sigma_i )
        * no extremum at cell m-1    : e_{m-2} = sigma_{m-1}
      The proof shows this system is unsatisfiable (sigma must be constant, then a
      corner forces a contradiction). This script confirms 0 solutions, matching
      the ground-truth minimum of 1.
"""
from __future__ import annotations

from itertools import product

from transfer_matrix import columns, compatible, flip


def ground_truth_min_extrema(m: int) -> int:
    """Minimum extrema of a boundary column over all compatible pairs (a,b)."""
    cols = columns(m)
    mn = 10 ** 9
    for a in cols:
        for b in cols:
            if compatible(a, b):
                mn = min(mn, flip(None, a, b), flip(a, b, None))
    return mn


def no_extremum_system_solutions(m: int) -> int:
    """Number of (e, sigma) satisfying g-walk validity AND no-extremum at every
    cell. The proof asserts this is 0 for all m; this brute-forces it."""
    sols = 0
    for e in product((1, -1), repeat=m - 1):
        for s in product((1, -1), repeat=m):
            if any(e[i] + (s[i + 1] - s[i]) not in (1, -1) for i in range(m - 1)):
                continue                              # g-walk invalid
            if e[0] == s[0]:
                continue                              # cell 0 is an extremum
            if any(-e[i - 1] == e[i] == s[i] for i in range(1, m - 1)):
                continue                              # an interior cell is an extremum
            if e[m - 2] != s[m - 1]:
                continue                              # cell m-1 is an extremum
            sols += 1                                 # all cells non-extremum
    return sols


def main() -> None:
    print("boundary-column lemma verification")
    print(f"{'m':>3} | {'min extrema (ground truth)':>26} | "
          f"{'no-extremum (e,sigma) solutions':>31}")
    ok = True
    for m in range(2, 12):
        gt = ground_truth_min_extrema(m) if m <= 10 else None   # 6^m enumeration cap
        sysN = no_extremum_system_solutions(m)
        gt_s = "-" if gt is None else str(gt)
        flag = "" if (sysN == 0 and (gt is None or gt >= 1)) else "  <-- FAIL"
        ok = ok and sysN == 0 and (gt is None or gt >= 1)
        print(f"{m:>3} | {gt_s:>26} | {sysN:>31}{flag}")
    print(f"\nlemma holds (min>=1 and 0 no-extremum solutions): {ok}")


if __name__ == "__main__":
    main()
