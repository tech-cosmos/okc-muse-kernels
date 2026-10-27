import re,sys
from collections import Counter
def loops(fn,pat='moe_twqILi8ELb1E'):
    L=open(fn).read().split('\n'); out=[];on=False
    for l in L:
        if 'Function :' in l: on=pat in l
        if on: out.append(l)
    ins=[l for l in out if re.search(r'/\*[0-9a-f]{4,}\*/',l)]
    def addr(l): return int(re.search(r'/\*([0-9a-f]{4,})\*/',l).group(1),16)
    hg=[addr(l) for l in ins if 'HGMMA' in l]
    br=[]
    for l in ins:
        if ' BRA ' in l:
            m=re.search(r'0x([0-9a-f]+)',l.split('BRA')[1])
            if m: br.append((addr(l),int(m.group(1),16)))
    res=[]
    for a,t in sorted(set(br)):
        if t<a and any(t<=x<=a for x in hg):
            b=[re.search(r'/\*[0-9a-f]{4,}\*/\s+(.*?);',l).group(1).strip() for l in ins if t<=addr(l)<=a]
            if len(b)<400: res.append(b)
    return res
op=lambda b: Counter(re.sub(r'^@!?U?P\w+\s+','',x).split()[0] for x in b)
for fn in sys.argv[1:]:
    print('==',fn)
    for b in loops(fn):
        c=op(b); t=[k for k in c if k.startswith('HGMMA')]
        print(len(b), t, 'ARRIVE',c['WARPGROUP.ARRIVE'],'DEPBAR',c['WARPGROUP.DEPBAR.LE'])
