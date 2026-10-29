# n7s41: reads the n7s22 H200 DIAG 81391142 (stamped tk2f, layer routings): /tmp/n7s22/run/check.json + /tmp/n7s22/run/submission.py (routings); run from the dir holding routes.py
import numpy as np
exec(open('routes.py').read().replace("print(","(lambda *a,**k:None)("))
G=132; tot={}
for (load,seed),c in out.items():
    if load not in ('decode 1','verify 1','decode 16','decode 32'): continue
    ng=int(((np.array(c)+7)//8).sum()); L=[]
    for RPE,tr in ((256,2.06),(128,2.18)):
        n=ng*RPE; f=n/G-np.floor(n/G); L.append((1-f)*tr if f>0 else 0)
    tot.setdefault(load,[]).append(sum(L))
for k,v in tot.items(): print(k, 'quantization tail us/call per seed', ' '.join(f'{x:.2f}' for x in v), 'mean %.2f'%np.mean(v))
