import json,glob,os,re,sys
P="/home/daytona/r-layer/muse_decode_layer/profiles/"
want=sys.argv[1:] or ["verify 32 sliding"]
for f in sorted(glob.glob(P+"submission.*.json"),key=os.path.getmtime):
    d=json.load(open(f))
    if "benchmarks" not in d: continue
    j=d["job_id"][:8]
    sp={(b["case"]["load"]+" "+b["case"]["layer"]):(b["speedup"],b["median_ns"]/1e3) for b in d["benchmarks"]}
    for p in d["profiles"]:
        k=p["case"]["load"]+" "+p["case"]["layer"]
        if k not in want: continue
        ks=[]
        for line in p["report"].splitlines():
            m=re.match(r"(.{60})\s+([\d.]+) µs",line)
            if m and ("moe" in m.group(1) or "total" in m.group(1) or "busy" in m.group(1)): ks.append("%s=%s"%(m.group(1).split("(")[0].replace("void ","").strip()[:22],m.group(2)))
        b=re.search(r"GPU busy.*?([\d.]+) µs",p["report"])
        print(j,k,"spd %.3f ours %.1f"%sp[k],"busy",b.group(1) if b else "-"," ".join(ks))
