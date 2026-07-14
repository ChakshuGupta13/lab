"""
Column TRANSFER MATRIX for v^d_{m,n} = E_d (number of degree-d
vertices of OFG(M_{m,n}) = number of anchored grid height functions with exactly
d = a+b strict local extrema). This is the transfer object of Lemma lem:ratGF.

It builds the column-to-column transfer operator T_m (equivalently the singular
variety det(I - A(x,y)) at (1,1)) and validates it against the known counts.

Brute enumeration of height functions is exponential in
m*n; the transfer matrix computes v^d_{m,n} in time poly(n) (matrix power) for
fixed m, so it reaches far larger d / grids.

MODEL (verified against christensen2025 line 333 + earlier colourability probe):
- A grid height function with +-1 steps <-> proper 3-colouring c = h mod 3 (the
  +-1 lift always closes up around unit faces, so the bijection is exact).
- Anchor h(0,0)=0  <=>  colour of vertex (0,0) is 0.
- A vertex is a strict local extremum (flippable face) iff ALL its existing grid
  neighbours share a single colour (all c+1 => strict min, all c-1 => strict max).
- deg of the flip-graph vertex = #flippable = #extrema = a+b = d, so the number of
  degree-d vertices v^d_{m,n} = sum_{a+b=d} N_{a,b}(m,n) = E_d(m,n).

TRANSFER CONSTRUCTION (two-column window):
- A column colouring is a tuple in {0,1,2}^m with consecutive entries distinct
  (proper colouring of the column path).
- Adjacent columns p, c are compatible iff p_i != c_i for every row i.
- #extrema in the MIDDLE column c, given left=p and right=x (either may be None at
  the grid boundary), is flip(p, c, x): per row i, gather the colours of the
  existing neighbours up (i-1,c), down (i+1,c), left (i,p), right (i,x); the vertex
  is an extremum iff those colours are all equal.
- v^d = sum over column sequences of t^{#extrema}; we keep the whole t-polynomial
  (a dict {power: count}) and read off the coefficient of t^d.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Dict, List, Optional, Tuple

Col = Tuple[int, ...]
Poly = Dict[int, int]   # t-polynomial: {exponent: coefficient}


# --------------------------------------------------------------------------
# t-polynomial helpers (exact integer arithmetic)
# --------------------------------------------------------------------------
def p_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = defaultdict(int)
    for i, ci in a.items():
        for j, cj in b.items():
            out[i + j] += ci * cj
    return dict(out)


def p_add(a: Poly, b: Poly) -> Poly:
    out = defaultdict(int, a)
    for j, cj in b.items():
        out[j] += cj
    return {k: v for k, v in out.items() if v != 0}


# --------------------------------------------------------------------------
# columns and the local extremum count
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def columns(m: int) -> Tuple[Col, ...]:
    """All proper colourings of the height-m column path (consecutive distinct)."""
    out: List[Col] = []
    for tup in product(range(3), repeat=m):
        if all(tup[i] != tup[i + 1] for i in range(m - 1)):
            out.append(tup)
    return tuple(out)


def compatible(p: Col, c: Col) -> bool:
    return all(pi != ci for pi, ci in zip(p, c))


def flip(p: Optional[Col], c: Col, x: Optional[Col]) -> int:
    """#strict extrema in middle column c given left p, right x (None at boundary)."""
    m = len(c)
    cnt = 0
    for i in range(m):
        nb = []
        if i > 0:
            nb.append(c[i - 1])
        if i < m - 1:
            nb.append(c[i + 1])
        if p is not None:
            nb.append(p[i])
        if x is not None:
            nb.append(x[i])
        # extremum iff all existing neighbours share one colour
        if nb and all(v == nb[0] for v in nb):
            cnt += 1
    return cnt


# --------------------------------------------------------------------------
# v^d generating polynomial for fixed (m, n): coefficient of t^d is v^d_{m,n}
# --------------------------------------------------------------------------
def vpoly(m: int, n: int, anchor: bool = True) -> Poly:
    """Full t-polynomial sum_d v^d_{m,n} t^d (anchored: colour of (0,0) fixed=0).
    Memoised so repeated finite-difference fits reuse column sequences."""
    return _vpoly_cached(m, n, anchor)


@lru_cache(maxsize=None)
def _vpoly_cached(m: int, n: int, anchor: bool) -> Poly:
    cols = columns(m)
    if n == 1:
        # single column: each vertex's neighbours are only up/down in-column
        acc: Poly = {}
        for c in cols:
            if anchor and c[0] != 0:
                continue
            e = flip(None, c, None)
            acc = p_add(acc, {e: 1})
        return acc

    # state = ordered compatible pair (prev, cur); weight carries finalized columns
    # start: place col0, col1; finalize col0 via flip(None, col0, col1)
    state: Dict[Tuple[Col, Col], Poly] = defaultdict(dict)
    for c0 in cols:
        if anchor and c0[0] != 0:
            continue
        for c1 in cols:
            if not compatible(c0, c1):
                continue
            w = {flip(None, c0, c1): 1}
            key = (c0, c1)
            state[key] = p_add(state.get(key, {}), w)

    # finalize middle columns 1..n-2 by transitions (p,c)->(c,x): emit flip(p,c,x)
    for _ in range(n - 2):
        nxt: Dict[Tuple[Col, Col], Poly] = defaultdict(dict)
        for (p, c), w in state.items():
            for x in cols:
                if not compatible(c, x):
                    continue
                w2 = p_mul(w, {flip(p, c, x): 1})
                key = (c, x)
                nxt[key] = p_add(nxt.get(key, {}), w2)
        state = nxt

    # finalize last column n-1 via flip(prev, last, None)
    acc: Poly = {}
    for (p, c), w in state.items():
        w2 = p_mul(w, {flip(p, c, None): 1})
        acc = p_add(acc, w2)
    return acc


def v_d(m: int, n: int, d: int, anchor: bool = True) -> int:
    return vpoly(m, n, anchor).get(d, 0)


# --------------------------------------------------------------------------
# exact n-polynomial of v^d_{m,n} at fixed m (Fraction finite differences)
# --------------------------------------------------------------------------
from fractions import Fraction
from math import factorial


def fit_n_poly(m: int, d: int, n0: int, deg: int):
    """Exact coeffs [a_0..a_deg] (standard basis, powers of n) of v^d_{m,.} from
    deg+1 consecutive values at n0..n0+deg, VERIFIED by re-fit at n0+1 and a
    zero (deg+1)-th finite difference. Returns (coeffs, ok)."""
    def coeffs_from(start):
        ys = [Fraction(v_d(m, start + k, d)) for k in range(deg + 1)]
        diff = [ys[:]]
        for _ in range(deg):
            prev = diff[-1]
            diff.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
        coeff = [Fraction(0)] * (deg + 1)
        for r in range(deg + 1):
            d0 = diff[r][0]
            if d0 == 0:
                continue
            poly = [Fraction(1)]                 # prod_{i<r} (n - (start+i))
            for i in range(r):
                shift = start + i
                newp = [Fraction(0)] * (len(poly) + 1)
                for k, c in enumerate(poly):
                    newp[k + 1] += c
                    newp[k] += c * (-shift)
                poly = newp
            scale = d0 / factorial(r)
            for k, c in enumerate(poly):
                coeff[k] += c * scale
        return coeff

    c0 = coeffs_from(n0)
    c1 = coeffs_from(n0 + 1)
    ys = [v_d(m, n0 + k, d) for k in range(deg + 2)]
    dd = ys[:]
    for _ in range(deg + 1):
        dd = [dd[i + 1] - dd[i] for i in range(len(dd) - 1)]
    ok = (c0 == c1) and (dd[0] == 0)
    return c0, ok


def analyse_b2(splits_d=(5, 6, 7)):
    """For each d, fit v^d_{m,n} in n at fixed in-region m (m>=d-1) and test the
    TOP-TWO-LAYER part of the P1 cap: a_{d-2}=C_d and a_{d-3} are m-CONSTANT.
    NOTE: this tests ONLY the j=d-2 and j=d-3 conditions of G2 (the leading and
    first sub-leading n-coefficients). The full G2 (all mixed m^j n^{d-2-j}=0 for
    1<=j<=d-3) needs deg_m a_j <= d-3-j for the LOWER j too, which requires >=3
    widths and is NOT tested here (it is covered for d<=7 by the separate cone-
    enumeration work g2_cell_dim.py)."""
    print("=== b2: P1 cap via fast transfer matrix (in-region m >= d-1) ===")
    print("  G2 => coeff of n^{d-2} = C_d constant in m; coeff of n^{d-3} const")
    for d in splits_d:
        deg = d - 2
        ms = [mm for mm in range(d - 1, 8)]      # in-region, engine-feasible
        if not ms:
            continue
        print(f"\n  d={d} (n-degree {deg}); in-region m in {ms}:")
        lead, sub = {}, {}
        for m in ms:
            n0 = max(d + 1, m + 1)
            coeffs, ok = fit_n_poly(m, d, n0, deg)
            lead[m] = coeffs[deg]
            sub[m] = coeffs[deg - 1] if deg >= 1 else None
            print(f"    m={m}: fit OK={ok};  a_{deg}={coeffs[deg]}"
                  f"   a_{deg-1}={coeffs[deg-1] if deg>=1 else '-'}")
        lead_const = len(set(lead.values())) == 1
        sub_vals = [v for v in sub.values() if v is not None]
        sub_const = (len(set(sub_vals)) == 1) if sub_vals else None
        print(f"    => a_(d-2) m-constant C_{d}: {lead_const} "
              f"(values {sorted(set(lead.values()))})")
        print(f"    => a_(d-3) m-constant: {sub_const} "
              f"(values {sorted(set(sub_vals))})")


def brute_vpoly(m: int, n: int) -> Poly:
    """Enumerate proper 3-colourings with (0,0)=0; bucket by #extrema. O(3^{mn})."""
    cells = [(i, j) for i in range(m) for j in range(n)]

    def nbrs(i, j):
        out = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < m and 0 <= jj < n:
                out.append((ii, jj))
        return out

    col: Dict[Tuple[int, int], int] = {}
    acc: Poly = defaultdict(int)

    def bt(k: int):
        if k == len(cells):
            e = 0
            for (i, j) in cells:
                ns = [col[u] for u in nbrs(i, j)]
                if ns and all(v == ns[0] for v in ns):
                    e += 1
            acc[e] += 1
            return
        i, j = cells[k]
        for ccol in range(3):
            if (i, j) == (0, 0) and ccol != 0:
                continue
            ok = True
            for u in nbrs(i, j):
                if u in col and col[u] == ccol:
                    ok = False
                    break
            if ok:
                col[(i, j)] = ccol
                bt(k + 1)
                del col[(i, j)]

    bt(0)
    return {k: v for k, v in acc.items() if v != 0}


# --------------------------------------------------------------------------
# validation driver
# --------------------------------------------------------------------------
def validate():
    print("=== VALIDATION 1: transfer vpoly == brute vpoly (anchored) ===")
    ok = True
    for (m, n) in [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (2, 5)]:
        tv = vpoly(m, n)
        bv = brute_vpoly(m, n)
        match = tv == bv
        ok = ok and match
        print(f"  {m}x{n}: transfer={dict(sorted(tv.items()))}")
        print(f"        brute   ={dict(sorted(bv.items()))}  -> {'OK' if match else 'MISMATCH'}")
    print(f"  ALL MATCH: {ok}\n")

    print("=== VALIDATION 2: known closed forms for v^2, v^3, v^4 (min>=3) ===")
    print("  v^2 = 4;  v^3 = 4(m+n-4);  v^4 = 2m^2+2n^2+6mn-10(m+n)-4")
    ok2 = True
    for (m, n) in [(3, 3), (3, 4), (4, 4), (3, 5), (4, 5)]:
        v2, v3, v4 = v_d(m, n, 2), v_d(m, n, 3), v_d(m, n, 4)
        e2, e3, e4 = 4, 4 * (m + n - 4), 2*m*m + 2*n*n + 6*m*n - 10*(m+n) - 4
        good = (v2 == e2 and v3 == e3 and v4 == e4)
        ok2 = ok2 and good
        print(f"  {m}x{n}: v2={v2}(={e2}) v3={v3}(={e3}) v4={v4}(={e4})  "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"  ALL MATCH: {ok2}\n")
    return ok and ok2


def analyse_leadconst(dmax=8):
    """Discovery check: leading constant C_d (coeff of n^{d-2} in v^d, m-constant
    in-region) satisfies C_d = 4/(d-2)!  (EGF sum_d C_d z^{d-2} = 4 e^z).
    Verified against known v^2,v^3,v^4 and computed d=5..dmax at max feasible m."""
    print("=== leading-constant discovery: C_d = 4/(d-2)!  (EGF 4 e^z) ===")
    known = {2: Fraction(4), 3: Fraction(4), 4: Fraction(2)}  # coeff of m^{d-2}
    ok = True
    for d in range(2, dmax + 1):
        pred = Fraction(4, 1) / factorial(d - 2)
        if d in known:
            got = known[d]
            src = "paper v^d"
        else:
            m = min(d - 1, 7)            # one in-region, engine-feasible width
            deg = d - 2
            n0 = max(d + 1, m + 1)
            coeffs, fit_ok = fit_n_poly(m, d, n0, deg)
            got = coeffs[deg]
            src = f"fit m={m}"
            if not fit_ok:
                ok = False
        match = (got == pred)
        ok = ok and match
        print(f"  d={d}: C_d={got}  pred 4/(d-2)!={pred}  [{src}]  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"  => C_d = 4/(d-2)! holds d=2..{dmax}: {ok}\n")
    return ok


if __name__ == "__main__":
    import sys
    if "--b2" in sys.argv:
        analyse_b2()
    elif "--lead" in sys.argv:
        analyse_leadconst()
    else:
        validate()
