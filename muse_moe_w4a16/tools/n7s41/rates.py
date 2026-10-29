# n7s41: reads the n7s22 H200 DIAG 81391142 (stamped tk2f, layer routings): /tmp/n7s22/run/check.json + /tmp/n7s22/run/submission.py (routings); run from the dir holding routes.py
import json,re,sys
d=json.load(open(sys.argv[1]))
def walk(x):
    if isinstance(x,str):
        if 'diagtl' in x: yield x
    elif isinstance(x,dict):
        for v in x.values(): yield from walk(v)
    elif isinstance(x,list):
        for v in x: yield from walk(v)
seen=set()
W13=2*4096*4096*0.5625/1e6; W2=W13/2
for t in walk(d):
    for l in t.split('\n'):
        m=re.search(r'diagtl (\w+ \d+) s(\d+) T(\d+) E(\d+) total ([\d.]+): (.*?)  norm',l)
        if not m or l in seen: continue
        seen.add(l)
        f={k:tuple(map(float,v.split('/'))) for k,v in re.findall(r'(\w+) ([\d.]+/[\d.]+/[\d.]+)',m.group(6))}
        E=int(m.group(4))
        p1=f['ph1end'][1]-f['prod1st'][1]; p2=f['end'][1]-f['ph1end'][1]; p2s=f['end'][1]-f['ph2data'][1]
        print(f"{m.group(1):9s} s{m.group(2):3s} E{E:2d} tot {m.group(5):6s} ph1 {p1:5.1f}us {E*W13/p1:4.2f}TB/s | trans(med ph1end->ph2data) {f['ph2data'][1]-f['ph1end'][1]:4.2f} ph1spread {f['ph1end'][2]-f['ph1end'][0]:4.2f} | ph2 {p2:5.1f}us {E*W2/p2:4.2f}TB/s steady {E*W2*0.9/p2s:4.2f} | endspread {f['end'][2]-f['end'][0]:4.2f} | all {E*28.3/(f['end'][2]-f['prod1st'][1]):4.2f}")
