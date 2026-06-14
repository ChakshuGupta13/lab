#!/usr/bin/env python3
"""Exhaustive verification of Conjecture 4 on all connected graphs up to n vertices.

For each n, reports: count of connected graphs, number of C4 violations,
maximum mu*/H ratio, and any counterexamples found.

Connected counts are cross-checked against OEIS A001349.

Usage: python verify_c4_exhaustive.py [nmin] [nmax]   (default 2 9)
"""
import sys
from graph_utils import geng_connected, harmonic_index, mu_star_linegraph

# OEIS A001349: number of connected graphs on n labelled/unlabelled vertices
OEIS_A001349 = {
    1: 1, 2: 1, 3: 2, 4: 6, 5: 21, 6: 112,
    7: 853, 8: 11117, 9: 261080, 10: 11716571,
}


def verify(nmin, nmax):
    for n in range(nmin, nmax + 1):
        cnt = 0
        violations = []
        max_ratio = None

        for g6, G in geng_connected(n):
            cnt += 1
            mu = mu_star_linegraph(G)
            H = harmonic_index(G)
            ratio = mu / H if H > 0 else None

            if mu > H:
                violations.append((g6, mu, str(H), float(H)))

            if ratio is not None:
                if max_ratio is None or ratio > max_ratio:
                    max_ratio = ratio

        # OEIS cross-check
        expected = OEIS_A001349.get(n)
        oeis_ok = cnt == expected if expected else "n/a"

        print(f"n={n}: {cnt} connected graphs, {len(violations)} C4 violations, "
              f"max mu*/H = {float(max_ratio):.6f}, OEIS A001349 check = {oeis_ok}")
        if violations:
            for g6, mu, Hs, Hf in violations:
                print(f"  CE: {g6}  mu*={mu}  H={Hs} ({Hf:.4f})")


if __name__ == "__main__":
    nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    verify(nmin, nmax)
