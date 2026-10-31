import json,sys,re
def prof(j,idx):
    d=json.load(open(f'submission.{j}.json'))
    for p in d['profiles']:
        if p['index']==idx:
            out={}
            for l in p['report'].split('\n')[1:]:
                m=re.match(r'(.{60,66}?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)',l)
                if m: out[m.group(1).strip()]=(float(m.group(2)),int(m.group(3)))
            return out
idx=int(sys.argv[1]); js=sys.argv[2:]
ps=[prof(j,idx) for j in js]
ks=[]
for p in ps:
    for k in p:
        if k not in ks: ks.append(k)
print(f'{"kernel":62} '+' '.join(f'{j[:8]:>9}' for j in js))
for k in ks:
    print(f'{k[:62]:62} '+' '.join(f'{p[k][0]:9.1f}' if k in p else '        -' for p in ps))
