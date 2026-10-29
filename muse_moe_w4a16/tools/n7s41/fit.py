# n7s41: reads the n7s22 H200 DIAG 81391142 (stamped tk2f, layer routings): /tmp/n7s22/run/check.json + /tmp/n7s22/run/submission.py (routings); run from the dir holding routes.py
import json,re,numpy as np
exec(open('routes.py').read().replace("print(","(lambda *a,**k:None)("))
d=json.load(open('/tmp/n7s22/run/check.json'))
def walk(x):
    if isinstance(x,str):
        if 'diagtl' in x: yield x
    elif isinstance(x,dict):
        for v in x.values(): yield from walk(v)
    elif isinstance(x,list):
        for v in x: yield from walk(v)
rows=[];seen=set()
for t in walk(d):
    for l in t.split('\n'):
        m=re.search(r'diagtl (\w+ \d+) s(\d+) T(\d+) E(\d+) total ([\d.]+): (.*?)  norm',l)
        if not m or l in seen: continue
        seen.add(l)
        f={k:tuple(map(float,v.split('/'))) for k,v in re.findall(r'(\w+) ([\d.]+/[\d.]+/[\d.]+)',m.group(6))}
        c=np.array(out[(m.group(1),int(m.group(2)))]); E=len(c); g8=int(((c+7)//8).sum())
        rows.append((m.group(1),int(m.group(2)),E,g8-E,f['ph1end'][1]-f['prod1st'][1],f['ph1end'][2]-f['prod1st'][1],f['ph2data'][1]-f['ph1end'][1],f['end'][1]-f['ph2data'][1],f['end'][2]-f['ph2data'][1],f['end'][2]))
A=np.array([[r[2],r[3],1] for r in rows],float)
for name,k in (('ph1 med',4),('ph1 max',5),('trans',6),('ph2 med',7),('ph2 max',8),('end max',9)):
    y=np.array([r[k] for r in rows]); co,res,_,_=np.linalg.lstsq(A,y,rcond=None)
    print(f"{name:8s} = {co[0]:5.2f} x E + {co[1]:5.2f} x extra + {co[2]:5.2f}   resid sd {np.std(y-A@co):.2f}")
for r in rows: print(r[0],r[1],'E',r[2],'x',r[3],' '.join(f'{v:5.1f}' for v in r[4:]))
print('---- per extra group residual (fit on extra-free inputs)')
base=[r for r in rows if r[3]==0]
A0=np.array([[r[2],1] for r in base],float)
for name,k in (('ph1 med',4),('ph1 max',5),('ph2 med',7),('ph2 max',8),('end max',9)):
    y0=np.array([r[k] for r in base]); co=np.linalg.lstsq(A0,y0,rcond=None)[0]
    print(name, f"{co[0]:.2f} x E + {co[1]:.2f}", ' '.join(f"{r[0][-2:]}s{r[1]}:{(r[k]-co[0]*r[2]-co[1])/r[3]:5.2f}" for r in rows if r[3]))
