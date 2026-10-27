import csv, sys, collections
r=list(csv.reader(open(sys.argv[1]))); h=r[1]; rows=r[2:]; ix={c:i for i,c in enumerate(h)}
S=[c for c in h if c.startswith('stall_') and 'Not' not in c]
f=lambda x: float(x.replace(',','')) if x.replace(',','').replace('.','').isdigit() else 0.
EX=int(sys.argv[2])
idx=[i for i,row in enumerate(rows) if f(row[ix['Instructions Executed']])==EX]
a,b=min(idx),max(idx); print('loop rows',a,b,'n',b-a+1)
tot=collections.Counter(); n=0
for row in rows[a:b+1]:
    for c in S: tot[c]+=f(row[ix[c]])
    n+=f(row[ix['Warp Stall Sampling (All Samples)']])
allS=sum(f(row[ix['Warp Stall Sampling (All Samples)']]) for row in rows)
print(f'loop samples {n} of {allS}'); print(', '.join(f"{c[6:]}:{v/n*100:.1f}%" for c,v in tot.most_common(10)))
for row in rows[a:b+1]:
    s=f(row[ix['Warp Stall Sampling (All Samples)']])
    if s/n>0.004 or 'SYNCS' in row[1] or 'LDS' in row[1] or 'WARPGROUP' in row[1] or 'BAR' in row[1]:
        rs=sorted(((f(row[ix[c]]),c[6:]) for c in S),reverse=True)[:3]
        print(f"{row[0][-5:]} {s/n*100:5.1f}% ex={row[ix['Instructions Executed']]:>6} {row[1][:66]:66} "+" ".join(f"{c}:{v/max(s,1)*100:.0f}" for v,c in rs if v))
