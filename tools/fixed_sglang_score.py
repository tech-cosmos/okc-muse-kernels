import json,glob,math
jobs={"3e3b9fd3":"v153 SUBMIT (board)","60f02748":"v153 check","4787d92e":"v165 check","b87bd6aa":"v165 submit","725fc518":"v163+twm submit","3e39a051":"v162 check","7693598b":"v163 submit","af87d466":"v157 check","93027e57":"v166 check","59013cf5":"v167 check","a5b4ddab":"v168 check","8b4e6cf7":"v172 check","ae6c4bd3":"v172 submit","2d1f9256":"v174 check","d8fc46c5":"v175 check","68c3ba66":"v175 submit"}
rows={}
for j,n in jobs.items():
    fs=glob.glob("/home/daytona/r-layer/muse_decode_layer/profiles/submission.%s*.json"%j)
    if not fs: continue
    d=json.load(open(fs[0])); b=d["benchmarks"]
    rows[j]=(n,{(x["case"]["load"],x["case"]["layer"]):(x["median_ns"]/1e3,x["baseline_ns"]/1e3) for x in b})
keys=sorted(rows["3e3b9fd3"][1].keys())
print("job      name                ", " ".join(("%s %s"%(k[0].replace("decode ","d").replace("verify ","v"),k[1][:2])).rjust(8) for k in keys))
for j,(n,r) in rows.items():
    print(j, n.ljust(20), " ".join(("%.1f"%r[k][0]).rjust(8) if k in r else "   -    " for k in keys), " ours")
    print(j, "".ljust(20), " ".join(("%.1f"%r[k][1]).rjust(8) if k in r else "   -    " for k in keys), " sglang")
print()
sg={k:sum(r[k][1] for _,r in rows.values())/len(rows) for k in keys}
loads=sorted(set(k[0] for k in keys))
def score(f):
    return math.exp(sum(math.log((46*sg[(l,"sliding")]+16*sg[(l,"full")])/(46*f(l,"sliding")+16*f(l,"full"))) for l in loads)/len(loads))
for j,(n,r) in rows.items():
    s_fixed=score(lambda l,t:r[(l,t)][0])
    s_own=math.exp(sum(math.log((46*r[(l,"sliding")][1]+16*r[(l,"full")][1])/(46*r[(l,"sliding")][0]+16*r[(l,"full")][0])) for l in loads)/len(loads))
    print(j,n.ljust(20),"vs avg SGLang %.4f   vs own-run SGLang %.4f"%(s_fixed,s_own))
