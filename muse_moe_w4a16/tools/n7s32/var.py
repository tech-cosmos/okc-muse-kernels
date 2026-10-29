import json,glob,os,re,statistics as st
P="/home/daytona/r-layer/muse_decode_layer/profiles/"
out=[]
for f in sorted(glob.glob(P+"*.json"),key=os.path.getmtime):
    d=json.load(open(f)); 
    if d.get("problem")!="muse_decode_layer": continue
    j=d["job_id"][:8]; lg=f[:-5]+".log"
    log=open(lg).read() if os.path.exists(lg) else ""
    drops=re.findall(r"(\S+) dropped",log)
    picks={}
    for m in re.finditer(r"tune T=(\d+) (\w+):.*-> (.*)",log): picks[(int(m.group(1)),m.group(2))]=m.group(3).strip()
    rows={}
    for b in d["benchmarks"]:
        c=b["case"]; k=(c["load"].replace("decode ","d").replace("verify ","v"),c["layer"][:2])
        r=sorted(b.get("runs_ns",[])); 
        rows[k]=dict(med=b["median_ns"]/1e3,sg=b["baseline_ns"]/1e3,r=[x/1e3 for x in r],T=c["batch"]*(1 if 'decode' in c['load'] else 1),load=c["load"])
    out.append((j,os.path.basename(f),d["summary"].get("speedup"),drops,picks,rows,os.path.getmtime(f)))
import pickle;pickle.dump(out,open("/tmp/n7s32/var.pkl","wb"))
keys=sorted(out[-1][5])
print("job      spd   "+" ".join(("%s%s"%k).rjust(8) for k in keys))
for j,fn,s,dr,pk,rows,t in out:
    print(j,("%.3f"%s if s else "  -  "),"ours "+" ".join(("%.1f"%rows[k]["med"]).rjust(8) if k in rows else "   -    " for k in keys), "drops:",",".join(dr) if dr else "-", "" if pk else "(no tune log)")
print()
for j,fn,s,dr,pk,rows,t in out:
    print(j,"iqr/med% "+" ".join(("%.1f"%(100*(st.quantiles(rows[k]["r"],n=4)[2]-st.quantiles(rows[k]["r"],n=4)[0])/rows[k]["med"]) if k in rows and len(rows[k]["r"])>4 else "-").rjust(6) for k in keys))
