"""Parse the canonical Orbium seeds verbatim from Chan's Lenia.m (fetched to
/tmp/lenia_ref.m) into assets/orbium_seeds.npz. No hand-typing of the arrays.

Source: https://github.com/Chakazul/Lenia  Matlab/Lenia.m  (Orbium bicaudatus).
Low-res:  R=13 peaks=[1] mu=0.15 sigma=0.014 dt=0.1  (textbook Lenia paper).
High-res: R=13 peaks=[1] mu=0.17 sigma=0.015 dt=0.1.
Kernel core = quad4 polynomial bump (4r(1-r))^4; growth = gaussian 2*exp(-(n-mu)^2/(2 sigma^2))-1.
"""
import re
import numpy as np

SRC = "/tmp/lenia_ref.m"
txt = open(SRC).read()


def parse_cells(line):
    m = re.search(r"cells=\[(.*?)\];", line)
    rows = [r for r in m.group(1).split(";") if r.strip() != ""]
    arr = np.array([[float(x) for x in r.split(",")] for r in rows])
    return arr


def grab(mu):
    pat = rf"name='Orbium'; if isLoadCell; R=13;peaks=\[1\];mu={mu};.*?cells=\[.*?\]; end"
    for line in txt.splitlines():
        if re.search(pat, line):
            return parse_cells(line)
    raise RuntimeError(f"Orbium mu={mu} not found")


lo = grab("0.15")
hi = grab("0.17")
print(f"low-res  mu=0.15 sigma=0.014: shape={lo.shape} sum={lo.sum():.3f} max={lo.max():.3f}")
print(f"high-res mu=0.17 sigma=0.015: shape={hi.shape} sum={hi.sum():.3f} max={hi.max():.3f}")

np.savez("assets/orbium_seeds.npz",
         lores=lo, lores_mu=0.15, lores_sigma=0.014,
         hires=hi, hires_mu=0.17, hires_sigma=0.015,
         R=13.0, dt=0.1)
print("saved assets/orbium_seeds.npz")
