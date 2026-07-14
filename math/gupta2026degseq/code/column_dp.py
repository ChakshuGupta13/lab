"""
Column transfer-matrix engine and rational-GF verifier (Lemma lem:quotient): for
every m>=2, the generating function sum_{n>=2} E_d(m,n) z^n has its only pole at
z=1, so E_d(m,.) is eventually a polynomial in n.

Provides the core enumeration primitives (cols_of, trans_ok, cell_ext, Ed_DP)
reused by the p_d transfer-matrix chain. The check: is the only pole of the GF
at z=1, for every (m,d)? Equivalently, is E_d(m,.) eventually a polynomial (all
poles at z=1) with no geometric/oscillatory tail that would put a pole at z!=1?

Strategy (three independent enumerators, then GF analysis):
 (0) Direct HEIGHT-FUNCTION enumerator (corner fixed) -- validates the whole
     mod-3 reduction end to end, NOT using transfer_period.py at all.
 (1) From-scratch column-sequence brute enumerator (colourings/3).
 (2) My own pair-DP for larger n.
 (3) Cross-check (0)=(1)=(2)=their Ed_series (imported) on shared cells.
 (4) Per (m,d): E_d table, finite-difference triangle, minimal k with
     Delta^k == 0 on the tail (=> denominator (1-z)^k, sole pole z=1), and a
     sympy factorisation of the minimal TAIL recurrence's char poly (must be
     (x-1)^{d-1}: ALL roots at 1).  Any root off 1  => CRITICAL.
 (5) d=2 boundary; max-d / zero-GF edge (GF==0 has NO pole -> possible overclaim).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from typing import Dict, List, Optional, Tuple

import sympy as sp

Col = Tuple[int, ...]

# --------------------------------------------------------------------------- #
# (0) DIRECT height-function enumerator (no mod-3, no transfer_period).        #
#     Grid cells (i,j): i in 0..m-1 rows, j in 0..n-1 cols. Corner h[0][0]=0.  #
#     |h[a]-h[b]|=1 on every grid edge.  Count strict local extrema.           #
# --------------------------------------------------------------------------- #
def Ed_heightfuncs(m: int, n: int) -> Counter:
    """Return Counter{d: #height functions on m x n grid, corner h(0,0)=0,
    with exactly d strict local extrema}.  Pure integer DFS."""
    cells = [(i, j) for j in range(n) for i in range(m)]
    # order cells column-major so each new cell has >=1 already-placed neighbour
    order = cells
    pos = {c: k for k, c in enumerate(order)}

    def nbrs(i, j):
        out = []
        if i > 0: out.append((i - 1, j))
        if i < m - 1: out.append((i + 1, j))
        if j > 0: out.append((i, j - 1))
        if j < n - 1: out.append((i, j + 1))
        return out

    h: Dict[Tuple[int, int], int] = {}
    tally: Counter = Counter()

    def count_extrema() -> int:
        d = 0
        for (i, j) in cells:
            v = h[(i, j)]
            nb = [h[x] for x in nbrs(i, j)]
            if all(w < v for w in nb) or all(w > v for w in nb):
                d += 1
        return d

    def dfs(k: int):
        if k == len(order):
            tally[count_extrema()] += 1
            return
        (i, j) = order[k]
        placed = [(a, b) for (a, b) in nbrs(i, j) if (a, b) in h]
        if not placed:  # only the corner
            h[(i, j)] = 0
            dfs(k + 1)
            del h[(i, j)]
            return
        # candidate values: differ by +-1 from every placed neighbour
        base = h[placed[0]]
        for val in (base - 1, base + 1):
            if all(abs(val - h[p]) == 1 for p in placed):
                h[(i, j)] = val
                dfs(k + 1)
                del h[(i, j)]

    dfs(0)
    return tally


# --------------------------------------------------------------------------- #
# (1) From-scratch column brute enumerator (colourings / 3).                   #
# --------------------------------------------------------------------------- #
def cols_of(m: int) -> List[Col]:
    return [c for c in product(range(3), repeat=m)
            if all(c[i] != c[i + 1] for i in range(m - 1))]


def trans_ok(a: Col, b: Col) -> bool:
    return all(a[i] != b[i] for i in range(len(a)))


def cell_ext(col: Col, m: int, left: Optional[Col], right: Optional[Col]) -> int:
    c = 0
    for i in range(m):
        nb = []
        if i > 0: nb.append(col[i - 1])
        if i < m - 1: nb.append(col[i + 1])
        if left is not None: nb.append(left[i])
        if right is not None: nb.append(right[i])
        if len(set(nb)) == 1:
            c += 1
    return c


def Ed_colouring_brute(m: int, n: int) -> Counter:
    """Exhaustive over all length-n admissible column sequences; extrema counted
    with the column's actual present neighbours; result divided by 3."""
    cols = cols_of(m)
    tally: Counter = Counter()

    def rec(seq: List[Col]):
        if len(seq) == n:
            d = 0
            for j in range(n):
                left = seq[j - 1] if j > 0 else None
                right = seq[j + 1] if j < n - 1 else None
                d += cell_ext(seq[j], m, left, right)
            tally[d] += 1
            return
        for c in cols:
            if not seq or trans_ok(seq[-1], c):
                seq.append(c)
                rec(seq)
                seq.pop()

    rec([])
    return Counter({d: v // 3 for d, v in tally.items()})


# --------------------------------------------------------------------------- #
# (2) My own pair-DP (independent of transfer_period.Ed_series structure).     #
#     State after committing columns 0..j = (col_{j-1}, col_j, extrema so far   #
#     for columns 0..j-1 finalised).  Column j finalised when j+1 known.        #
# --------------------------------------------------------------------------- #
def Ed_DP(m: int, nmax: int) -> Dict[int, Counter]:
    cols = cols_of(m)
    trans = {a: [b for b in cols if trans_ok(a, b)] for a in cols}
    out: Dict[int, Counter] = {}
    # state dict: (prev,cur) -> Counter{extrema_finalised: colourings}
    st: Dict[Tuple[Col, Col], Counter] = defaultdict(Counter)
    for a in cols:
        for b in trans[a]:
            st[(a, b)][cell_ext(a, m, None, b)] += 1   # col0 = left boundary
    def close(state):
        tot: Counter = Counter()
        for (a, b), cd in state.items():
            eb = cell_ext(b, m, a, None)               # last col = right boundary
            for e, ct in cd.items():
                tot[e + eb] += ct
        return Counter({d: c // 3 for d, c in tot.items()})
    out[2] = close(st)
    for n in range(3, nmax + 1):
        nxt: Dict[Tuple[Col, Col], Counter] = defaultdict(Counter)
        for (a, b), cd in st.items():
            for c in trans[b]:
                eb = cell_ext(b, m, a, c)              # finalise interior col b
                dst = nxt[(b, c)]
                for e, ct in cd.items():
                    dst[e + eb] += ct
        st = nxt
        out[n] = close(st)
    return out


# --------------------------------------------------------------------------- #
# GF / finite-difference analysis                                             #
# --------------------------------------------------------------------------- #
def fdiff(seq: List[Fraction], k: int) -> List[Fraction]:
    s = list(seq)
    for _ in range(k):
        s = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    return s


def min_zero_diff(vals: List[Fraction], kmax: int, tail: int = 4) -> Optional[int]:
    """Smallest k such that Delta^k vals is all-zero on its last `tail` entries.
    Returns None if none up to kmax."""
    for k in range(0, kmax + 1):
        d = fdiff(vals, k)
        if len(d) >= tail and all(x == 0 for x in d[-tail:]):
            return k
    return None


def tail_recurrence_charpoly(vals: List[Fraction], n_start: int, onset: int,
                             maxL: int = 8):
    """Fit the minimal constant-coeff linear recurrence a_{n+L}=sum c_j a_{n+j}
    holding for all n>=onset (tail), over Q.  Return (L, charpoly factored) or
    (None,None).  charpoly = x^L - sum c_j x^j ; its roots are the poles' recips."""
    # indices: vals[k] corresponds to n = n_start + k
    tail_idx = [k for k in range(len(vals)) if n_start + k >= onset]
    for L in range(1, maxL + 1):
        # unknowns c_0..c_{L-1}; equations for each n with n, n+L in tail range
        rows = []
        rhs = []
        for k in tail_idx:
            if k + L < len(vals):
                rows.append([vals[k + j] for j in range(L)])
                rhs.append(vals[k + L])
        if len(rows) < L:
            continue
        M = sp.Matrix(rows)
        b = sp.Matrix(rhs)
        # least-squares exact: solve normal equations; require EXACT consistency
        try:
            sol, params = M.gauss_jordan_solve(b)
        except Exception:
            continue
        if params.free_symbols:
            # underdetermined: pick particular (params=0)
            sol = sol.subs({s: 0 for s in params.free_symbols})
        # verify recurrence holds EXACTLY on all tail equations
        ok = all(sum(sol[j] * rows[r][j] for j in range(L)) == rhs[r]
                 for r in range(len(rows)))
        if not ok:
            continue
        x = sp.symbols('x')
        char = x**L - sum(sol[j] * x**j for j in range(L))
        return L, sp.factor(char), [sp.nsimplify(c) for c in sol]
    return None, None, None


def build_gf_denom(vals: List[Fraction], n_start: int, k: int):
    """Given eventually-poly seq with Delta^k==0 on tail, build EXACT rational
    GF sum a_n z^n = P(z)/(1-z)^k and return (P(z) factored, denom, reconstruct-
    ok).  P(z)=G(z)(1-z)^k has coeffs p_t = sum_j (-1)^j C(k,j) a_{t-j}; must be
    a finite polynomial (p_t=0 for large t)."""
    z = sp.symbols('z')
    a = {n_start + i: vals[i] for i in range(len(vals))}
    nmin, nmax = n_start, n_start + len(vals) - 1
    pcoeff = {}
    for t in range(nmin, nmax + 1):
        s = sp.Integer(0)
        for j in range(0, k + 1):
            if t - j in a:
                s += (-1)**j * sp.binomial(k, j) * a[t - j]
            elif t - j >= nmin:
                s = None
                break
        if s is not None:
            pcoeff[t] = sp.nsimplify(s)
    # P is a genuine polynomial iff pcoeff[t]==0 for all t beyond some bound.
    nonzero_t = sorted(t for t, v in pcoeff.items() if v != 0)
    # tail of pcoeff (large t) must be zero
    checkable = [t for t in pcoeff if t >= nmin + k]  # p_t fully determined
    tail_ok = all(pcoeff[t] == 0 for t in checkable if t > (max(nonzero_t) if nonzero_t else nmin))
    P = sum(pcoeff[t] * z**t for t in pcoeff)
    denom = (1 - z)**k
    # reconstruct check: series of P/denom must reproduce a_n
    G = sp.series(P / denom, z, 0, nmax + 1).removeO()
    recon_ok = all(sp.expand(G).coeff(z, nn) == a[nn] for nn in range(nmin, nmax + 1))
    return sp.factor(P), denom, recon_ok, nonzero_t


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("STEP 1  cross-check enumerators (independent) on shared cells")
    print("=" * 78)

    mism = 0
    for m in (2, 3):
        nmax_naive = {2: 6, 3: 4}[m]
        dp = Ed_DP(m, nmax_naive)
        for n in range(2, nmax_naive + 1):
            hf = Ed_heightfuncs(m, n)          # (0) direct height functions
            cb = Ed_colouring_brute(m, n)      # (1) colour brute /3
            dd = dp[n]                          # (2) my DP
            alld = set(hf) | set(cb) | set(dd)
            for d in sorted(alld):
                vals = (hf[d], cb[d], dd[d])
                if len(set(vals)) != 1:
                    mism += 1
                    print(f"  MISMATCH m={m} n={n} d={d}: "
                          f"height={hf[d]} colourbrute={cb[d]} myDP={dd[d]}")
        print(f"  m={m}: cross-checked n=2..{nmax_naive} across 3 methods "
              f"(height-fn / colour-brute / DP)")
    print(f"  total mismatches = {mism}  "
          f"[{'OK' if mism == 0 else 'FAIL'}]")

    print()
    print("=" * 78)
    print("STEP 2  sole-pole check per (m,d)")
    print("=" * 78)
    NMAX = 16
    results = {}
    for m in (2, 3, 4):
        ser = Ed_DP(m, NMAX)
        ns = sorted(ser.keys())
        dmax_here = max(max(c) for c in ser.values() if c)
        print(f"\n----- m={m}  (n=2..{NMAX}; observed max d = {dmax_here}) -----")
        for d in range(2, min(dmax_here, 7) + 1):
            vals = [Fraction(ser[n].get(d, 0)) for n in ns]
            # table
            tbl = "  ".join(f"{int(v)}" for v in vals)
            # minimal k with Delta^k -> 0 on tail
            k = min_zero_diff(vals, kmax=d + 3, tail=4)
            # onset = first n where degree-(k-1) poly matches to the end
            expect_k = d - 1 if d >= 2 else 0
            # finite-diff triangle top rows
            fd_rows = []
            for kk in range(0, (k if k is not None else d) + 1):
                row = fdiff(vals, kk)
                fd_rows.append((kk, row[-6:]))
            # tail recurrence char poly (adversarial: all roots must be x=1)
            onset_guess = max(2, d - 1)
            L, char, coeffs = tail_recurrence_charpoly(vals, n_start=ns[0],
                                                       onset=onset_guess, maxL=d + 2)
            # exact GF denominator
            if k is not None and k >= 1:
                P, denom, recon_ok, nz = build_gf_denom(vals, ns[0], k)
            else:
                P, denom, recon_ok, nz = (None, None, None, None)
            print(f"  d={d}:  E_d(m,n) for n=2..{NMAX} = {tbl}")
            print(f"        minimal k with Delta^k=0 on tail: {k}   "
                  f"(claim: k=d-1={expect_k}, i.e. denom (1-z)^{expect_k}, deg n = {d-2})")
            if k is not None:
                print(f"        => GF denominator (1-z)^{k}; ALL poles at z=1: "
                      f"{'YES' if k is not None else 'NO'}")
            if char is not None:
                roots = sp.roots(char)
                offone = {r: mult for r, mult in roots.items() if sp.simplify(r - 1) != 0}
                print(f"        tail recurrence order L={L}, charpoly factored = {char}")
                print(f"        roots = {roots}   roots-off-1 = {offone}  "
                      f"[{'CRITICAL off-1 pole' if offone else 'all roots at x=1 OK'}]")
            if P is not None:
                print(f"        exact GF = [{P}] / {denom}   reconstruct_matches={recon_ok}")
            results[(m, d)] = dict(k=k, expect_k=expect_k, char=char,
                                   offone=bool(char is not None and
                                               any(sp.simplify(r - 1) != 0 for r in sp.roots(char))))

    print()
    print("=" * 78)
    print("STEP 3  d=2 boundary  &  max-d / zero-GF edge")
    print("=" * 78)
    for m in (2, 3, 4):
        ser = Ed_DP(m, NMAX)
        e2 = [ser[n].get(2, 0) for n in sorted(ser)]
        print(f"  m={m}: E_2(m,n), n=2..{NMAX} = {e2}   "
              f"(paper says E_2==4 const; GF 4z^2/(1-z), sole pole z=1)")
        # is E_2 constant 4?
        print(f"        E_2 constant==4 ? {all(v == 4 for v in e2)}")
    # max-d edge: for m=2 find the largest d with E_d>0 for some n, and whether
    # there is a 'gap' d where E_d==0 for ALL n<=NMAX (=> GF has NO pole).
    for m in (2, 3, 4):
        ser = Ed_DP(m, NMAX)
        maxd = max(max(c) for c in ser.values() if c)
        zero_ds = [d for d in range(2, maxd + 2)
                   if all(ser[n].get(d, 0) == 0 for n in sorted(ser))]
        print(f"  m={m}: observed max d (n<=%d) = {maxd}; "
              f"d-values that are 0 for ALL n in range = {zero_ds}" % NMAX)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    bad = [(m, d) for (m, d), r in results.items() if r["offone"]]
    kbad = [(m, d, r["k"], r["expect_k"]) for (m, d), r in results.items()
            if r["k"] is None or r["k"] != r["expect_k"]]
    print(f"  (m,d) with a pole OFF z=1 (CRITICAL): {bad if bad else 'NONE'}")
    print(f"  (m,d) where minimal k != d-1 (degree/order note): "
          f"{kbad if kbad else 'NONE — every GF denom is exactly (1-z)^(d-1)'}")


if __name__ == "__main__":
    main()
