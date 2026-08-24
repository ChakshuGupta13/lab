#!/usr/bin/env python3
"""Fractional LP test for whether ANY linear discharging over local edge-types can prove
the cubic covering claim (Phi=4b+P has no beta>=1 local minimum).

Model a hypothetical Phi-local-minimum as a fractional density over M-edge TYPES. An
M-edge k has chosen endpoint ch (covers its at-risk neighbours, each mult in {1,2,3}) and
unchosen un (does NOT cover its at-risk neighbours, each mult in {0,1,2}); each endpoint
has <=2 at-risk neighbours (cubic, minus the M-partner). Edge-type = (ch-profile, un-profile).

Per-edge local-min constraint (validated dPhi=L-G): L(cp)=3*#1(cp)+#2(cp) >= G(up)=3*#0(up)+#1(up).
Half-edge consistency (a mult-t at-risk vertex is chosen-neighbour of t edges, unchosen-
neighbour of 3-t edges):
  mult-1: (#1 incidences on un side) = 2*(#1 on ch side)
  mult-2: (#2 incidences on ch side) = 2*(#2 on un side)
(mult-0 only un-side, mult-3 only ch-side -- no cross constraint.)

Maximise beta = (1/3)*sum(#0(up)*e) over nonneg edge densities e with sum(e)=1, subject to
the per-edge filter and the two consistency equalities. max=0 => LINEAR DISCHARGING PROVES
beta=0 (a real proof, since this LP RELAXES the graph problem). max>0 => no such linear
discharging can work (barrier), and the maximiser is the fractional witness.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

def profiles(vals):
    out=[()]
    for s in (1,2):
        out += list(itertools.combinations_with_replacement(vals,s))
    return out

CP = profiles((1,2,3))     # ch-side at-risk neighbour mults
UP = profiles((0,1,2))     # un-side at-risk neighbour mults
def cnt(p,v): return sum(1 for x in p if x==v)

types=[]
for cp in CP:
    for up in UP:
        L=3*cnt(cp,1)+cnt(cp,2); G=3*cnt(up,0)+cnt(up,1)
        if L>=G:                       # only non-improving edges appear in a local-min
            types.append((cp,up))
n=len(types)
# objective: maximise sum(#0(up)*e)  -> minimise -that
c=np.array([-cnt(up,0) for (cp,up) in types],dtype=float)
# consistency equalities A_eq e = 0, plus normalisation sum e = 1
row1=np.array([cnt(up,1)-2*cnt(cp,1) for (cp,up) in types],dtype=float)  # mult-1
row2=np.array([cnt(cp,2)-2*cnt(up,2) for (cp,up) in types],dtype=float)  # mult-2
norm=np.ones(n)
A_eq=np.vstack([row1,row2,norm]); b_eq=np.array([0.0,0.0,1.0])
res=linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=[(0,None)]*n,method="highs")
print(f"edge-types (L>=G): {n}")
print(f"LP status: {res.message}")
maxbeta_times3 = -res.fun if res.success else float('nan')
print(f"max  sum(#0*e)  = {maxbeta_times3:.6f}   (= 3*beta at total-edges=1)")
if res.success and maxbeta_times3 < 1e-7:
    print(">>> max beta = 0  => LINEAR DISCHARGING PROVES beta=0  (PROOF!)")
elif res.success:
    print(">>> max beta > 0  => NO linear discharging over these types works (barrier). Witness:")
    for i,(cp,up) in enumerate(types):
        if res.x[i]>1e-6: print(f"    e={res.x[i]:.4f}  ch-profile={cp} un-profile={up}  L={3*cnt(cp,1)+cnt(cp,2)} G={3*cnt(up,0)+cnt(up,1)}")
