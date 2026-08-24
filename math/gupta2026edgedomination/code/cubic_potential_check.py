#!/usr/bin/env python3
"""Cache (beta,P,m2,m3) for base + every 1-flip of every |B|>=1 transversal, then test
many integer potentials Phi = a*beta + b*P + c*m2 + d*m3 for '0 beta>=1 local minima'.
A clean integer Phi with 0 stuck => elementary greedy proof candidate."""
import sys, itertools
from cubic_construction_probe import cubic_graphs, maximal_matchings, at_risk, cov_map

def hist(bits, Ms, cov, n_ar):
    m=[0]*n_ar
    for k,b in enumerate(bits):
        for idx in cov.get(Ms[k][b], ()): m[idx]+=1
    b0=b1=b2=b3=0
    for x in m:
        if x==0:b0+=1
        elif x==1:b1+=1
        elif x==2:b2+=1
        else:b3+=1
    return (b0,b1,b2,b3)

def main():
    ns=[int(x) for x in sys.argv[1:]] or [14]
    # cache: list of (base_tuple, [flip_tuples])
    cache=[]
    for n in ns:
        for adj in cubic_graphs(n):
            for M in maximal_matchings(adj,n):
                Ms=sorted(M); ar=at_risk(adj,Ms,n)
                if not ar: continue
                cov=cov_map(ar); n_ar=len(ar); mm=len(Ms)
                for bits in itertools.product((0,1),repeat=mm):
                    base=hist(bits,Ms,cov,n_ar)
                    if base[0]==0: continue
                    flips=[]
                    for k in range(mm):
                        b2=list(bits); b2[k]^=1
                        flips.append(hist(b2,Ms,cov,n_ar))
                    cache.append((base,flips))
    print(f"ns={ns}: cached |B|>=1 states = {len(cache)}", flush=True)
    def val(t,coef): return coef[0]*t[0]+coef[1]*t[1]+coef[2]*t[2]+coef[3]*t[3]
    # candidate integer potentials (a,b,c,d) on (beta,P,m2,m3)
    cands=[(2,1,0,0),(3,1,0,0),(4,1,0,0),(1,1,0,0),
           (3,0,-1,-1),(4,0,-1,-1),(2,0,-1,-1),
           (3,1,-1,-1),(2,1,-1,-1),(4,1,-1,-1),
           (3,2,1,0),(4,2,1,0),(2,1,0,-1),(3,1,0,-1),
           (4,2,0,-1),(5,2,0,-1),(3,1,-1,-2),(4,1,-1,-2)]
    for coef in cands:
        stuck=0
        for base,flips in cache:
            cur=val(base,coef)
            if not any(val(f,coef)<cur for f in flips): stuck+=1
        tag=" <== ELEMENTARY-PROOF CANDIDATE" if stuck==0 else ""
        print(f"  Phi = {coef[0]}*b + {coef[1]}*P + {coef[2]}*m2 + {coef[3]}*m3 : stuck={stuck}{tag}", flush=True)

if __name__=="__main__":
    main()
