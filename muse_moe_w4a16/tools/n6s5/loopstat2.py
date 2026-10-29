import csv, sys, re
rows = list(csv.reader(open(sys.argv[1])))
h = rows[1]; ia = h.index('Address'); iS = h.index('Warp Stall Sampling (All Samples)'); iE = h.index('Instructions Executed')
R = []
for r in rows[2:]:
    try: R.append((int(r[ia], 16), r[1].strip(), float(r[iS] or 0), float(r[iE].replace(',', '') or 0)))
    except: pass
base = R[0][0]; R = [(a - base, s, x, e) for a, s, x, e in R]
idx = {a: i for i, (a, *_ ) in enumerate(R)}
loops = []
for i, (a, s, x, e) in enumerate(R):
    m = re.search(r'BRA.*?(0x[0-9a-f]+)\s*;?$', s) or re.search(r'BRA\s+`?\(?\.?L_x_\d+', s)
    m = re.search(r'BRA[^;]*?\b(0x[0-9a-f]+)\b', s)
    if m:
        t = int(m.group(1), 16); t = t - base if t >= base else t
        if t < a and t in idx:
            body = R[idx[t]:i + 1]
            nm = sum('HGMMA' in b[1] for b in body)
            if nm: loops.append((t, a, nm, sum(b[2] for b in body), sum(b[3] for b in body)))
tot = sum(r[2] for r in R); te = sum(r[3] for r in R)
for t, a, nm, smp, ex in loops:
    print('%7x-%7x mma %3d samples %6d (%4.1f%%) inst %9.3g  smp/Minst %6.1f' % (t, a, nm, smp, 100 * smp / tot, ex, smp / max(ex, 1) * 1e6))
print('total samples', tot, 'inst %.3g' % te, 'avg smp/Minst %.1f' % (tot / te * 1e6))
