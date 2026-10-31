import json, sys, statistics as st
ev=[e for e in json.load(open(sys.argv[1]))["traceEvents"] if e.get("cat")=="kernel"]
ev.sort(key=lambda e:e["ts"]); ev=[e for e in ev if "sum" not in e["name"].lower() and "reduce" not in e["name"].lower()]
N=int(sys.argv[2]); k=len(ev)//N; per=[ev[i*k:(i+1)*k] for i in range(N)]
names=[e["name"][:20] for e in per[0]]
ti=[j for j,n in enumerate(names) if "tk2f" in n][0]
rows=[]
for r in per:
    t0=r[0]["ts"]; rows.append(([e["ts"]-t0 for e in r],[e["ts"]+e["dur"]-t0 for e in r]))
for j in range(ti-3,ti+2):
    print(names[j], "end:", " ".join(f"{x[1][j]:.1f}" for x in rows[:12]))
print("tk2f start", " ".join(f"{x[0][ti]:.1f}" for x in rows[:12]))
a=[x[1][ti-1] for x in rows]; b=[x[1][ti] for x in rows]
print("corr norm_end vs tk2f_end", st.correlation(a,b), "sd", st.pstdev(a), st.pstdev(b))
