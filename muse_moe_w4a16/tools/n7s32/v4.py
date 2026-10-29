import json,glob,os,numpy as np
P="/home/daytona/r-layer/muse_decode_layer/profiles/"
C={}
for f in sorted(glob.glob(P+"submission.*.json"),key=os.path.getmtime):
    d=json.load(open(f)); j=d.get("job_id","?")[:8]
    if "benchmarks" not in d: continue
    for b in d["benchmarks"]:
        c=b["case"]; k="%s%s"%(c["load"].replace("decode ","d").replace("verify ","v"),c["layer"][:2])
        o=np.array(b["runs_ns"])/1e3; s=np.array(b["baseline_runs_ns"])/1e3
        C.setdefault(k,[]).append((j,np.corrcoef(o,s)[0,1],np.median(s/o),b["speedup"],np.median(o),np.median(s)))
for k,v in C.items():
    cs=[x[1] for x in v]; sp=np.array([x[3] for x in v])
    print("%-6s per-rep corr(ours,sg) median %.2f [%.2f..%.2f] | run speedup mean %.3f sd %.1f%% min %.3f max %.3f"%(k,np.median(cs),min(cs),max(cs),sp.mean(),100*sp.std()/sp.mean(),sp.min(),sp.max()))
import sys
for k in sys.argv[1:]:
    for x in C[k]: print(k,x[0],"corr %.2f spd %.3f ours %.1f sg %.1f"%(x[1],x[3],x[4],x[5]))
