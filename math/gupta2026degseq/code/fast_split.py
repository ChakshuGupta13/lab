"""
Fast k-cone split counter (numpy), for the d=7 balanced (3,4) confirmation and
later hold-out generation.

Speedups over split_diagnostic.count_split_envelope (pure Python):
  - By the Envelope Structure Theorem: once all k
    offsets are ACTIVE and pairwise PARITY-compatible, the envelope E is
    automatically a Z-height function AND its strict-min set is EXACTLY the k
    apexes. So the ONLY per-configuration check is "#strict maxima == b".
    (No validity scan, no min-set scan -- the two most expensive checks drop.)
  - Distance-from-cell arrays precomputed once per cell (numpy int16).
  - Maxima counted by a vectorised strict-greater-than-all-neighbours test.

Validated against split_diagnostic.count_split_envelope (which is itself
cross-checked against the brute OFG enumerator) on small grids before use.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

Cell = Tuple[int, int]

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".split_cache_fast.json")
_CACHE: Dict[str, int] = {}
if os.path.exists(_CACHE_PATH):
    try:
        with open(_CACHE_PATH) as _f:
            _CACHE = json.load(_f)
    except Exception:  # noqa: BLE001
        _CACHE = {}


def _cache_save() -> None:
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_CACHE, f)
    os.replace(tmp, _CACHE_PATH)


def _dist_arrays(m: int, n: int) -> Dict[Cell, np.ndarray]:
    ii = np.arange(m).reshape(m, 1)
    jj = np.arange(n).reshape(1, n)
    out: Dict[Cell, np.ndarray] = {}
    for a in range(m):
        for b in range(n):
            out[(a, b)] = (np.abs(ii - a) + np.abs(jj - b)).astype(np.int32)
    return out


def _num_strict_maxima(E: np.ndarray) -> int:
    """Count cells strictly greater than every existing 4-neighbour."""
    m, n = E.shape
    is_max = np.ones((m, n), dtype=bool)
    # up neighbour exists for rows 1..m-1
    is_max[1:, :] &= E[1:, :] > E[:-1, :]
    is_max[:-1, :] &= E[:-1, :] > E[1:, :]
    is_max[:, 1:] &= E[:, 1:] > E[:, :-1]
    is_max[:, :-1] &= E[:, :-1] > E[:, 1:]
    return int(is_max.sum())


def count_split_fast(a: int, b: int, m: int, n: int,
                     dist: Dict[Cell, np.ndarray] = None) -> int:
    """#height functions on G_{m,n} with exactly a minima, b maxima
    (a<=b assumed; uses k=a min-cones). Numpy-accelerated."""
    if a > b:
        return count_split_fast(b, a, m, n, dist)
    k = a
    if dist is None:
        dist = _dist_arrays(m, n)
    cells = [(i, j) for i in range(m) for j in range(n)]
    total = 0

    if k == 1:
        # single cone: a min at p, b = 1 + (#maxima of dist(p,.)). Just scan.
        for p in cells:
            E = dist[p]
            if _num_strict_maxima(E) == b:
                total += 1
        return total

    for idx in combinations(range(len(cells)), k):
        ps = [cells[t] for t in idx]
        darr = [dist[p] for p in ps]
        # pairwise distances D[s][t] = dist[ps[s]] at ps[t]
        D = [[int(darr[s][ps[t]]) for t in range(k)] for s in range(k)]

        if k == 2:
            D01 = D[0][1]
            for c1 in range(-D01 + 1, D01):
                if (c1 - D01) & 1:
                    continue
                E = np.minimum(darr[0], c1 + darr[1])
                if _num_strict_maxima(E) == b:
                    total += 1
        elif k == 3:
            D01, D02, D12 = D[0][1], D[0][2], D[1][2]
            base0 = darr[0]
            for c1 in range(-D01 + 1, D01):
                if (c1 - D01) & 1:
                    continue
                arr1 = c1 + darr[1]
                e01 = np.minimum(base0, arr1)
                lo2, hi2 = -D02 + 1, D02 - 1
                for c2 in range(lo2, hi2 + 1):
                    if (c2 - D02) & 1:
                        continue
                    if abs(c1 - c2) >= D12:
                        continue
                    E = np.minimum(e01, c2 + darr[2])
                    if _num_strict_maxima(E) == b:
                        total += 1
        else:
            raise NotImplementedError("k>=4 not needed here")
    return total


def count_cached(a: int, b: int, m: int, n: int) -> int:
    if a > b:
        return count_cached(b, a, m, n)
    key = f"{a},{b},{m},{n}"
    if key in _CACHE:
        return _CACHE[key]
    val = count_split_fast(a, b, m, n)
    _CACHE[key] = val
    _cache_save()
    return val


def validate(grids: List[Tuple[int, int]], splits: List[Tuple[int, int]]):
    """Cross-check the fast counter against split_diagnostic.count_split_envelope
    (brute-validated) on small grids."""
    from split_diagnostic import count_split_envelope as slow
    print("=== validate fast vs slow (brute-checked) counter ===")
    ok = True
    for (m, n) in grids:
        for (a, b) in splits:
            f = count_split_fast(a, b, m, n)
            s = slow(a, b, m, n)
            tag = "OK" if f == s else "FAIL"
            if f != s:
                ok = False
            print(f"  {m}x{n} ({a},{b}): fast={f} slow={s} {tag}")
    print("=== fast counter VALIDATED ===" if ok else "=== MISMATCH ===")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--time", default="", help="a:b:m:n to time a single count")
    args = ap.parse_args()
    if args.validate:
        validate([(3, 3), (3, 4), (4, 4), (3, 5), (4, 5)],
                 [(2, 2), (2, 3), (3, 3), (2, 4)])
    if args.time:
        a, b, m, n = map(int, args.time.split(":"))
        t0 = time.time()
        v = count_split_fast(a, b, m, n)
        print(f"count_split_fast({a},{b},{m},{n}) = {v}  "
              f"[{time.time()-t0:.2f}s]")


if __name__ == "__main__":
    main()
