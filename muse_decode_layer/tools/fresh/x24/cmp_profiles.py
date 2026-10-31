import json,glob,sys
def load(f):
    d=json.load(open(f)); out={}
    for b in d['benchmarks']:
        c=b['case']; k=(c["load"] if "load" in c else c.get("name"))+" "+c.get("layer","")+" b"+str(c.get("batch"))
        out[k]=(b['median_ns']/1e3, b['baseline_ns']/1e3)
    return d['summary'].get('speedup'),out
best={}
for pat in sys.argv[1:]:
    for f in glob.glob(pat):
        try: s,o=load(f)
        except Exception as e: continue
        if s is None: continue
        tag=pat.split('/')[3]
        for k,(m,bl) in o.items():
            if (tag,k) not in best or m<best[(tag,k)][0]: best[(tag,k)]=(m,bl,f.split('.')[-2][:8],s)
keys=sorted({k for _,k in best})
for k in keys:
    print(k[:70], *["%s %.1f(%s %.3f)"%(t[:5],*[best[(t,k)][i] for i in (0,)],best[(t,k)][2],best[(t,k)][3]) for t in sorted({t for t,_ in best}) if (t,k) in best])
