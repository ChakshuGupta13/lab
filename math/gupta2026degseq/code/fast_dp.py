#!/usr/bin/env python3
"""Optimised transfer-matrix DP for E_d(m,n), capped at d_cap.

We only ever want E_d for d <= d_cap (=<= 11 in practice). Extrema counts
along the sweep are non-decreasing, so any state whose partial-extrema count
already exceeds d_cap contributes nothing to E_d for d <= d_cap and can be
dropped. That turns each state's histogram from an O(mn)-entry Counter into
a fixed-length list of size d_cap+1.

Other tweaks: lru_cache on cell_ext, precomputed boundary extras, list-backed
histograms indexed by extrema count instead of a Counter dict.
"""
from functools import lru_cache
from typing import Dict, List, Tuple
import time

from column_dp import cols_of, trans_ok, cell_ext as _cell_ext_raw


def Ed_DP_capped(m: int, nmax: int, d_cap: int) -> Dict[int, Dict[int, int]]:
    cols = cols_of(m)
    trans = {a: [b for b in cols if trans_ok(a, b)] for a in cols}

    @lru_cache(maxsize=None)
    def cext(col, left, right):
        return _cell_ext_raw(col, m, left, right)

    ext_L = {(a, b): cext(a, None, b) for a in cols for b in trans[a]}
    ext_R = {(b, a): cext(b, a, None) for a in cols for b in trans[a]}

    def new_hist():
        return [0] * (d_cap + 1)

    st: Dict[Tuple, List[int]] = {}
    for a in cols:
        for b in trans[a]:
            e0 = ext_L[(a, b)]
            if e0 <= d_cap:
                key = (a, b)
                h = st.get(key)
                if h is None:
                    h = new_hist()
                    st[key] = h
                h[e0] += 1

    def close(state):
        tot = [0] * (d_cap + 1)
        for (a, b), h in state.items():
            eb = ext_R[(b, a)]
            if eb > d_cap:
                continue
            for e in range(d_cap + 1 - eb):
                ct = h[e]
                if ct:
                    tot[e + eb] += ct
        return {d: tot[d] // 3 for d in range(d_cap + 1) if tot[d]}

    out: Dict[int, Dict[int, int]] = {2: close(st)}
    for n in range(3, nmax + 1):
        nxt: Dict[Tuple, List[int]] = {}
        for (a, b), h in st.items():
            for c in trans[b]:
                eb = cext(b, a, c)
                if eb > d_cap:
                    continue
                key = (b, c)
                dst = nxt.get(key)
                if dst is None:
                    dst = new_hist()
                    nxt[key] = dst
                # shift-add: dst[e+eb] += h[e] for e in [0, d_cap-eb]
                limit = d_cap + 1 - eb
                for e in range(limit):
                    ct = h[e]
                    if ct:
                        dst[e + eb] += ct
        st = nxt
        out[n] = close(st)
    return out


if __name__ == "__main__":
    from column_dp import Ed_DP as Ed_DP_ref

    # 1. correctness against Ed_DP on small m
    print("=== correctness (fast vs Ed_DP baseline) ===")
    for m in (3, 4, 5, 6):
        nmax = m + 3
        d_cap = 10
        ref = Ed_DP_ref(m, nmax)
        fast = Ed_DP_capped(m, nmax, d_cap)
        ok = True
        mismatches = []
        for n in range(2, nmax + 1):
            for d in range(d_cap + 1):
                if ref[n].get(d, 0) != fast[n].get(d, 0):
                    ok = False
                    mismatches.append((n, d, ref[n].get(d, 0), fast[n].get(d, 0)))
        print(f"  m={m} d_cap={d_cap}: ok={ok}"
              + ("" if ok else f"  mismatches[:3]={mismatches[:3]}"))

    # 2. timing: fast at m=8 vs baseline (measured 15.75s previously)
    print("\n=== timing ===")
    t = time.time(); Ed_DP_ref(8, 11); t_ref8 = time.time() - t
    t = time.time(); Ed_DP_capped(8, 11, 8); t_fast8 = time.time() - t
    print(f"  m=8 nmax=11 d_cap=8:  baseline={t_ref8:.2f}s  fast={t_fast8:.2f}s"
          f"  speedup={t_ref8/max(t_fast8,1e-9):.1f}x")

    # 3. push toward m=10 (d=8 interpolation ceiling)
    for m in (9, 10):
        t = time.time(); Ed_DP_capped(m, m, 8); dt = time.time() - t
        print(f"  m={m}  nmax={m}  d_cap=8:  fast={dt:.2f}s")
