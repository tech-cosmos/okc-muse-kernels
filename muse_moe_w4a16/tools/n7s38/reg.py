import json,glob,os,re
P="/home/daytona/r-layer/muse_decode_layer/profiles/"
fs=sorted(glob.glob(P+"*.json"),key=os.path.getmtime)[-20:]
print("job      tunT8sl | v1sl ours/sg ratio | v1fu ours/sg ratio | d16sl ours/sg ratio | d16fu ours/sg ratio | d1sl ours/sg")
for f in fs:
    d=json.load(open(f))
    if "job_id" not in d: continue
    j=d["job_id"][:8]
    log=open(f[:-5]+".log").read() if os.path.exists(f[:-5]+".log") else ""
    m=re.search(r"tune T=8 sliding: default ([0-9.]+)",log); t8=m.group(1) if m else "-"
    rows={}
    for b in d["benchmarks"]:
        c=b["case"]; k=c["load"].replace("decode ","d").replace("verify ","v")+c["layer"][:2]
        rows[k]=(b["median_ns"]/1e3,b["baseline_ns"]/1e3)
    s=f"{j} {t8:>7} "
    for k in ["v1sl","v1fu","d16sl","d16fu","d1sl"]:
        o,g=rows.get(k,(0,1)); s+=f"| {o:6.1f} {g:6.1f} {g/o:5.3f} "
    print(s, f"{d['summary'].get('speedup')}")
