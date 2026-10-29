import json,glob,os,statistics as st
P="/home/daytona/r-layer/muse_decode_layer/profiles/"
def load(j):
    f=glob.glob(P+"*%s*.json"%j)[0]; d=json.load(open(f)); R={}
    for b in d["benchmarks"]:
        c=b["case"]; k="%s%s"%(c["load"].replace("decode ","d").replace("verify ","v"),c["layer"][:2])
        R[k]=b
    return R
import sys
for j in sys.argv[2:]:
    b=load(j)[sys.argv[1]]
    r=[x/1e3 for x in b["runs_ns"]]; s=[x/1e3 for x in b["baseline_runs_ns"]]
    print(j,"ours med %.1f min %.1f max %.1f | sg med %.1f min %.1f iqr%% %.1f"%(st.median(r),min(r),max(r),st.median(s),min(s),100*(st.quantiles(s,n=4)[2]-st.quantiles(s,n=4)[0])/st.median(s)))
    print("  ours seq:"," ".join("%.0f"%x for x in r))
    print("  sg   seq:"," ".join("%.0f"%x for x in s))
    print("  keys:",[k for k in b if k not in("runs_ns","baseline_runs_ns")])
