#!/usr/bin/env python3
"""
TxGraffiti C1 -- the structure-free residual inequality L'.

After the decomposition (c1_decompose.py), C1 reduces to:
    Delta = 2 : a = alpha          (trivial; paths/cycles)
    Delta >= 3: L'  :  a <= (Delta-1) * sum_i 1/(d_i+1)        [Caro-Wei strengthening]
L' implies L: a<=(Delta-1)alpha (since alpha >= sum 1/(d_i+1) by Caro 1979 / Wei 1981),
and L implies C1 (since alpha >= R, Favaron 1991). L' depends ONLY on the degree
sequence -- no graph, no independence number, no residue.

This script tests L' directly over ALL graphic degree sequences (Erdos-Gallai),
delta>=1 (connected domain), Delta>=3, up to length n. No graph realization or
independence-number computation is needed, so it runs far past the geng range.

It separately confirms that L' FAILS for some Delta=2 sequences (where it is not
needed) -- documenting that Caro-Wei is sufficient exactly on the Delta>=3 side.

Usage: python c1_lprime.py [nmin] [nmax]   (default 4 13)
"""
import sys
from fractions import Fraction


def erdos_gallai_ok(asc):
    """asc: non-decreasing degree list. True iff graphic (Erdos-Gallai + even sum)."""
    d = sorted(asc, reverse=True)
    n = len(d)
    s = sum(d)
    if s % 2 != 0:
        return False
    pref = 0
    # suffix helper for sum min(d_i,k)
    for k in range(1, n + 1):
        pref += d[k - 1]
        rhs = k * (k - 1)
        for i in range(k, n):
            rhs += min(d[i], k)
        if pref > rhs:
            return False
    return True


def annihilation(asc, m):
    tot, a = 0, 0
    for x in asc:
        tot += x
        if tot <= m:
            a += 1
        else:
            break
    return a


def check_n(n):
    """Enumerate non-decreasing sequences d_1<=...<=d_n, 1<=d_i<=n-1, even sum,
    graphic; test L' for those with Delta>=3. Returns (count_d3, viol_d3 list,
    tight_d3 count, viol_d2 list)."""
    cnt3 = tight3 = 0
    viol3 = []
    viol2 = []
    seq = [0] * n

    def rec(pos, prev):
        nonlocal cnt3, tight3
        if pos == n:
            asc = seq[:]
            Delta = asc[-1]
            if Delta < 2:
                return
            s = sum(asc)
            if s % 2 != 0:
                return
            if not erdos_gallai_ok(asc):
                return
            m = s // 2
            a = annihilation(asc, m)
            CW = sum(Fraction(1, x + 1) for x in asc)
            lhs, rhs = a, (Delta - 1) * CW
            if Delta == 2:
                if lhs > rhs and len(viol2) < 6:
                    viol2.append((tuple(asc), a, float(rhs)))
                return
            # Delta >= 3
            cnt3 += 1
            if lhs > rhs:
                if len(viol3) < 20:
                    viol3.append((tuple(asc), Delta, a, float(rhs)))
            elif lhs == rhs:
                tight3 += 1
            return
        # choose d[pos] in [max(prev,1), n-1]
        for v in range(max(prev, 1), n):
            seq[pos] = v
            rec(pos + 1, v)

    rec(0, 1)
    return cnt3, viol3, tight3, viol2


def main(nmin, nmax):
    print(f"{'n':>3} {'#seq(D>=3)':>11} {'L viol':>7} {'tight':>6} {'D=2 L-prime fails':>18}")
    for n in range(nmin, nmax + 1):
        cnt3, viol3, tight3, viol2 = check_n(n)
        flag = "OK" if not viol3 else f"!!! {len(viol3)}"
        print(f"{n:>3} {cnt3:>11} {flag:>7} {tight3:>6}   {len(viol2)} e.g. "
              f"{viol2[0] if viol2 else '-'}")
        if viol3:
            print("    L' (Delta>=3) VIOLATORS:")
            for e in viol3:
                print("       ", e)
    print("\nKey: 'L viol' over Delta>=3 must be 0 (=> Caro-Wei suffices for the "
          "whole hard side).\nDelta=2 violations are expected and harmless (handled by a=alpha).")


if __name__ == "__main__":
    nmin = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    main(nmin, nmax)
