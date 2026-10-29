# n7s41: reads the n7s22 H200 DIAG 81391142 (stamped tk2f, layer routings): /tmp/n7s22/run/check.json + /tmp/n7s22/run/submission.py (routings); run from the dir holding routes.py
import numpy as np
exec(open('routes.py').read().replace("print(","(lambda *a,**k:None)("))
G=132
def local_pos(n):
    pos=np.zeros(n); 
    for b in range(G):
        a0=n*b//G; a1=n*(b+1)//G; pos[a0:a1]=np.arange(a1-a0)
    return pos
for (load,seed),c in out.items():
    if load not in ('decode 16','decode 32') : continue
    c=np.array(c); ng=(c+7)//8
    res=[]
    for RPE in (256,128):
        n=int(ng.sum())*RPE; pos=local_pos(n); base=0; ds=[]
        for g in ng:
            for gi in range(1,g):
                for r in range(RPE):
                    ds.append(pos[base+gi*RPE+r]-pos[base+(gi-1)*RPE+r])  # later group minus earlier
            base+=g*RPE
        ds=np.array(ds)
        res.append(f"RPE{RPE}: rows/CTA {n/G:5.2f} dk median {np.median(ds):5.2f} |dk|<1: {np.mean(np.abs(ds)<1):4.2f} dk<0: {np.mean(ds<0):4.2f}" if len(ds) else f"RPE{RPE}: none")
    print(load,seed,'E',len(c),'x',int(ng.sum())-len(c),' | '.join(res))
