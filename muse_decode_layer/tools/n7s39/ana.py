import sqlite3, sys, statistics as S
for f in sys.argv[1:]:
    c = sqlite3.connect(f)
    rows = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
    # split into replays: first kernel name
    first = rows[0][2]; reps=[]; cur=[]
    for r in rows:
        if r[2]==first and cur: reps.append(cur); cur=[]
        cur.append(r)
    reps.append(cur); reps=reps[5:]
    n=len(reps[0]); reps=[r for r in reps if len(r)==n]
    print(f, len(reps), "reps", n, "kernels; total", S.median([(r[-1][1]-r[0][0])/1e3 for r in reps]))
    for i in range(n):
        st=S.median([(r[i][0]-r[0][0])/1e3 for r in reps]); en=S.median([(r[i][1]-r[0][0])/1e3 for r in reps])
        print(f"  {i:2d} {reps[0][i][2][:28]:28s} {st:7.1f} {en:7.1f} {en-st:6.1f}")
