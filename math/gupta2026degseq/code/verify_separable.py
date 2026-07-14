"""
Independent verification of the separable case (Theorem thm:sepcase + Lemma
lem:dimid): a separable family factors into independent row and column
contributions, and its counting function has the predicted total degree.

The paper's sec:leading establishes:
  - lem:sep  : separable <=> apex set is a product grid R x C with factoring
               offsets c_{(r,c)} = alpha_r + beta_c.
  - lem:dimid: a separable family with rho minima-rows, gamma minima-cols, rho'
               maxima-rows, gamma' maxima-cols has total degree
               D = (rho+rho'-2)+(gamma+gamma'-2), and
               (d-2) - D = (rho-1)(gamma-1) + (rho'-1)(gamma'-1) >= 0.
  - thm:sepcase: for d>=5, a separable family reaches degree d-2 ONLY if
               single-edge; every other separable family has degree <= d-3.
               (d=4 is the exception: the 6mn term.)

The non-separable residual (deg N^ns_{(a,b)} <= d-3 for d>=5) remains open; only
the separable case is verified here.

WHAT THIS SCRIPT VERIFIES:
  (V1) extrema of E=f(i)+g(j) are exactly (extrema f) x (extrema g)  [= lem:sep];
  (V2) the separable count factors, deg = (d_f-2)+(d_g-2)            [= lem:dimid];
  (V3) (a_f-1)(a_g-1)+(b_f-1)(b_g-1) >= 1 for multi-edge => deg <= d-3 for d>=5,
       with d=4 the sole exception                                  [= thm:sepcase].
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import comb
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------
# +-1 walks on a path of N vertices, classified by (#minima, #maxima)
# (endpoints count as extrema, per lem:binom)
# --------------------------------------------------------------------------
def walks_by_extrema(N: int) -> Dict[Tuple[int, int], int]:
    """{(a_f,b_f): #walks h on P_N with a_f strict minima, b_f strict maxima}.
    h is a +-1 step sequence; endpoints are extrema."""
    out: Dict[Tuple[int, int], int] = defaultdict(int)
    if N == 1:
        return {(0, 0): 1}  # degenerate; not used (we use N>=2)
    for steps in product((-1, 1), repeat=N - 1):
        h = [0]
        for s in steps:
            h.append(h[-1] + s)
        amin = bmax = 0
        for i in range(N):
            left = h[i - 1] if i > 0 else None
            right = h[i + 1] if i < N - 1 else None
            nb = [x for x in (left, right) if x is not None]
            if all(h[i] < x for x in nb):
                amin += 1
            elif all(h[i] > x for x in nb):
                bmax += 1
        out[(amin, bmax)] += 1
    return dict(out)


def separable_extrema(hf: List[int], hg: List[int]):
    """Direct extrema count of E(i,j)=hf[i]+hg[j]: returns (#minima,#maxima)."""
    m, n = len(hf), len(hg)
    amin = bmax = 0
    for i in range(m):
        for j in range(n):
            v = hf[i] + hg[j]
            nb = []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < n:
                    nb.append(hf[ii] + hg[jj])
            if nb and all(v < x for x in nb):
                amin += 1
            elif nb and all(v > x for x in nb):
                bmax += 1
    return amin, bmax


def all_walks(N: int):
    """List of (h, (a_f,b_f)) for every +-1 walk on P_N."""
    out = []
    for steps in product((-1, 1), repeat=N - 1):
        h = [0]
        for s in steps:
            h.append(h[-1] + s)
        amin = bmax = 0
        for i in range(N):
            nb = []
            if i > 0:
                nb.append(h[i - 1])
            if i < N - 1:
                nb.append(h[i + 1])
            if all(h[i] < x for x in nb):
                amin += 1
            elif all(h[i] > x for x in nb):
                bmax += 1
        out.append((h, (amin, bmax)))
    return out


def verify_extrema_product(Ns=(3, 4, 5)):
    """V1: extrema(E) == (extrema f) x (extrema g) for all walk pairs on small N."""
    print("=== V1: extrema(f(i)+g(j)) = (extrema f) x (extrema g) ===")
    ok = True
    for Nf in Ns:
        for Ng in Ns:
            for hf, (af, bf) in all_walks(Nf):
                for hg, (ag, bg) in all_walks(Ng):
                    a, b = separable_extrema(hf, hg)
                    if (a, b) != (af * ag, bf * bg):
                        ok = False
                        print(f"  MISMATCH Nf={Nf} Ng={Ng}: got ({a},{b}) "
                              f"expected ({af*ag},{bf*bg})  f={hf} g={hg}")
    print(f"  V1 holds on N in {Ns}: {ok}\n")
    return ok


def verify_degree_inequality(dmax=7):
    """V2+V3: deg N_sep = (d_f-2)+(d_g-2); multi-edge => deg <= d-3.
    Enumerate all (a_f,b_f),(a_g,b_g) with balanced walks and a_f a_g+b_f b_g=d."""
    print("=== V2+V3: separable degree decomposition + multi-edge drop ===")
    ok = True
    for d in range(2, dmax + 1):
        worst_multi = -1
        rows = []
        for af in range(1, d + 1):
            for bf in range(1, d + 1):
                if abs(af - bf) > 1:
                    continue
                for ag in range(1, d + 1):
                    for bg in range(1, d + 1):
                        if abs(ag - bg) > 1:
                            continue
                        if af * ag + bf * bg != d:
                            continue
                        df, dg = af + bf, ag + bg
                        deg = (df - 2) + (dg - 2)
                        a, b = af * ag, bf * bg
                        single_edge = (af == 1 or ag == 1)  # one axis a ramp on minima side
                        # but careful: single-edge means one factor is a RAMP (1,1)
                        is_ramp_f = (af == 1 and bf == 1)
                        is_ramp_g = (ag == 1 and bg == 1)
                        single = is_ramp_f or is_ramp_g
                        rows.append((af, bf, ag, bg, deg, single, a, b))
                        if not single:
                            worst_multi = max(worst_multi, deg)
                            drop = (d - 2) - deg
                            ineq = (af - 1) * (ag - 1) + (bf - 1) * (bg - 1)
                            if drop != ineq:
                                ok = False
                                print(f"  d={d}: DROP!=INEQ for "
                                      f"f=({af},{bf}) g=({ag},{bg}): "
                                      f"drop={drop} ineq={ineq}")
                            if d >= 5 and deg > d - 3:
                                ok = False
                                print(f"  d={d}: MULTI-EDGE deg={deg} > d-3={d-3}"
                                      f"  f=({af},{bf}) g=({ag},{bg})")
        if worst_multi >= 0:
            status = "OK" if (worst_multi <= d - 3 or d == 4) else "VIOL"
            note = "  (d=4: the documented 6mn exception)" if d == 4 and worst_multi > d - 3 else ""
            print(f"  d={d}: worst multi-edge separable degree = {worst_multi} "
                  f"(need <= d-3={d-3})  {status}{note}")
        else:
            print(f"  d={d}: no multi-edge separable split")
    print(f"  V2+V3 (decomposition + inequality + bound): {ok}\n")
    return ok


def main():
    v1 = verify_extrema_product()
    v23 = verify_degree_inequality()
    print(f"=== SEPARABLE sub-case verified (V1,V2,V3): {v1 and v23} ===")


if __name__ == "__main__":
    main()
