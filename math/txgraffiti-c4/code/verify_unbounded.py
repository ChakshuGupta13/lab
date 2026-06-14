#!/usr/bin/env python3
"""Verify the unbounded-separation family G_{m,k} = K_{m,2k} + matching.

Confirms:
  mu*(G_{m,k}) = k                           (line-graph + brute-force)
  H(G_{m,k}) = k/(m+1) + 4km/(2k+m+1)       (exact, cross-checked)
  mu*/H -> m+1 as k -> infinity              (unbounded over m)
  mu* <= 2H is FALSE                         (m=2, k=50 gives ratio 2.43)
"""
from fractions import Fraction
from graph_utils import (
    generalized_windmill, harmonic_index,
    mu_star_linegraph, mu_star_bruteforce,
    H_generalized_closed,
)


def verify_unbounded():
    print("=== G_{m,k} family: mu* and H verification ===")
    print(f"{'m':>2} {'k':>2} {'n':>3} {'mu*_lg':>6} {'mu*_bf':>6} {'=k?':>4} "
          f"{'H':>10} {'closed ok?':>10} {'ratio':>7}")
    print("-" * 70)
    for m in (1, 2, 3, 4):
        for k in range(2, 7):
            G = generalized_windmill(m, k)
            n = G.number_of_nodes()
            mu_lg = mu_star_linegraph(G)
            mu_bf = mu_star_bruteforce(G) if n <= 12 else None
            H = harmonic_index(G)
            Hc = H_generalized_closed(m, k)
            ratio = float(Fraction(mu_lg) / H)
            bf_str = str(mu_bf) if mu_bf is not None else "-"
            print(f"{m:>2} {k:>2} {n:>3} {mu_lg:>6} {bf_str:>6} "
                  f"{str(mu_lg == k):>4} {str(H):>10} {str(H == Hc):>10} "
                  f"{ratio:>7.3f}")

    print("\n=== Ratio exceeds 2? (refutes mu* <= 2H) ===")
    for m in (2, 3, 4):
        for k in (10, 50, 100):
            Hc = H_generalized_closed(m, k)
            ratio = float(Fraction(k) / Hc)
            flag = "  <-- > 2" if ratio > 2 else ""
            print(f"  m={m} k={k:>4}: mu*/H = {ratio:.4f}{flag}")
        print(f"  limit (k->inf) for m={m}: mu*/H -> {m + 1}")

    print("\nConclusion: mu*/H is unbounded. No constant c gives mu* <= c*H.")


if __name__ == "__main__":
    verify_unbounded()
