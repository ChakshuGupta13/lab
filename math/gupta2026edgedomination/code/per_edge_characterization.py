import sys, itertools
from cubic_construction_probe import cubic_graphs, maximal_matchings, at_risk, cov_map

def mults(bits,Ms,cov,nar):
    m=[0]*nar
    for k,b in enumerate(bits):
        for idx in cov.get(Ms[k][b],()): m[idx]+=1
    return m

bad=0; mismatch=0
for n in (10,12):
  for adj in cubic_graphs(n):
    for M in maximal_matchings(adj,n):
        Ms=sorted(M); ar=at_risk(adj,Ms,n)
        if not ar: continue
        cov=cov_map(ar); nar=len(ar); mm=len(Ms)
        # at-risk vertex -> its index, and its neighbour list
        arv=[u for (u,_nb) in ar]; nb=[set(_nb) for (_u,_nb) in ar]
        # coverage set per M-endpoint already in cov (endpoint -> at-risk idx set)
        for bits in itertools.product((0,1),repeat=mm):
            m=mults(bits,Ms,cov,nar)
            b0=sum(1 for x in m if x==0)
            if b0==0: continue
            bad+=1
            b1=sum(1 for x in m if x==1); cur=4*b0+b1
            for k in range(mm):
                ch=Ms[k][bits[k]]; un=Ms[k][1-bits[k]]
                # at-risk neighbours of ch / un = cov[ch]/cov[un] (indices)
                covch=cov.get(ch,set()); covun=cov.get(un,set())
                a_k=sum(1 for i in covun if m[i]==0)
                b_k=sum(1 for i in covun if m[i]==1)
                p_k=sum(1 for i in covch if m[i]==1)
                q_k=sum(1 for i in covch if m[i]==2)
                G=3*a_k+b_k; L=3*p_k+q_k
                # actual dPhi
                b2=list(bits); b2[k]^=1; m2=mults(b2,Ms,cov,nar)
                c0=sum(1 for x in m2 if x==0); c1=sum(1 for x in m2 if x==1)
                dphi=(4*c0+c1)-cur
                if dphi != (L-G): mismatch+=1
print(f"bad-transversals checked (n=10,12): {bad}")
print(f"edges where dPhi != L-G : {mismatch}  {'<== FORMULA VALID' if mismatch==0 else '*** FORMULA WRONG ***'}")
