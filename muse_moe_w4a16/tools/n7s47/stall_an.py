import csv, sys
from collections import defaultdict
for fn in sys.argv[1:]:
    rows=list(csv.reader(open(fn))); h=rows[1]; R=rows[2:]; ix={k:i for i,k in enumerate(h)}
    cols=[c for c in h if c.startswith("stall_") and "Not Issued" not in c]
    tot={c[6:]:sum(int(r[ix[c]] or 0) for r in R) for c in cols}; S=sum(int(r[ix['# Samples']] or 0) for r in R)
    print(fn, "samples", S, {k:round(v/S,3) for k,v in tot.items() if v/S>0.01})
    agg=defaultdict(lambda: defaultdict(int))
    for r in R:
        t=r[1].split()
        if not t: continue
        op=t[1] if t[0].startswith('@') else t[0]; op=op.split('.')[0]
        for c in ("stall_dispatch","stall_math","stall_wait","stall_selected","stall_not_selected","stall_short_sb"): agg[op][c[6:]]+=int(r[ix[c]] or 0)
        agg[op]['n']+=int(r[ix['# Samples']] or 0)
    for op,d in sorted(agg.items(), key=lambda x:-x[1]['n'])[:8]: print("   ", op, dict(d))
