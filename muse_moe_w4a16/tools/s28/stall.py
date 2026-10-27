import csv, sys, collections
r=list(csv.reader(open(sys.argv[1])))
h=r[1]; rows=r[2:]
ix={c:i for i,c in enumerate(h)}
S=[c for c in h if c.startswith('stall_') and 'Not' not in c]
def f(x):
    try: return float(x.replace(',',''))
    except: return 0.
tot=collections.Counter(); allS=0
for row in rows:
    for c in S: tot[c]+=f(row[ix[c]])
    allS+=f(row[ix['Warp Stall Sampling (All Samples)']])
print('total samples',allS); print(', '.join(f"{c[6:]}:{v/allS*100:.1f}%" for c,v in tot.most_common(10)))
# top instructions
top=sorted(rows,key=lambda x:-f(x[ix['Warp Stall Sampling (All Samples)']]))[:int(sys.argv[2]) if len(sys.argv)>2 else 30]
for row in top:
    s=f(row[ix['Warp Stall Sampling (All Samples)']])
    rs=sorted(((f(row[ix[c]]),c[6:]) for c in S),reverse=True)[:3]
    print(f"{row[0]} {s/allS*100:5.1f}% ex={row[ix['Instructions Executed']]:>8} {row[1][:70]:70} "+" ".join(f"{c}:{v/max(s,1)*100:.0f}" for v,c in rs if v))
