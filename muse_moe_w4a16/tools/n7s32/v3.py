import json,glob,os,re,statistics as st
P="/home/daytona/r-layer/muse_decode_layer/profiles/"
res={}
for f in sorted(glob.glob(P+"*.json"),key=os.path.getmtime):
    d=json.load(open(f))
    if d.get("problem")!="muse_decode_layer" or "diag" in f: continue
    j=d["job_id"][:8]
    med={}
    for b in d["benchmarks"]:
        c=b["case"]; med[(c["load"],c["layer"])]=(b["median_ns"]/1e3,b["baseline_ns"]/1e3)
    for p in d["profiles"]:
        c=p["case"]; k=(c["load"],c["layer"])
        ex=sum(float(m.group(1)) for m in re.finditer(r"^void moe_(?:twm|tk2\w*)<[^\n]*?\s([\d.]+) µs",p["report"],re.M))
        busy=re.search(r"GPU busy[^\d]*([\d.]+)",p["report"]); 
        res.setdefault(k,[]).append((j,med[k][0],med[k][1],ex,float(busy.group(1)) if busy else 0))
import numpy as np
for k,v in res.items():
    a=np.array([x[1:] for x in v])
    ours,sg,ex,busy=a.T
    r=np.corrcoef(ours,ex)[0,1]; rs=np.corrcoef(ours,sg)[0,1]
    slope=np.polyfit(ex,ours,1)[0]
    print("%-10s %-7s n=%d ours %.1f sd %.1f (%.1f%%) | expert %.1f sd %.1f | corr(ours,expert) %.2f slope %.2f | sg %.1f sd %.1f%% corr(ours,sg) %.2f | ours-expert sd %.1f"%(k[0],k[1],len(v),ours.mean(),ours.std(),100*ours.std()/ours.mean(),ex.mean(),ex.std(),r,slope,sg.mean(),100*sg.std()/sg.mean(),rs,(ours-ex).std()))
