import re,sys,collections
lines=open(sys.argv[1]).read().split('\n')
fn=sys.argv[2]
st=[i for i,l in enumerate(lines) if 'Function :' in l and fn in l][0]
en=next(i for i in range(st+1,len(lines)) if 'Function :' in lines[i]) if any('Function :' in l for l in lines[st+1:]) else len(lines)
ins=[]
for l in lines[st:en]:
    m=re.match(r'\s+/\*([0-9a-f]{4,})\*/\s+(.*?);',l)
    if m: ins.append((int(m.group(1),16),m.group(2).strip()))
addr={a:i for i,(a,_) in enumerate(ins)}
for i,(a,t) in enumerate(ins):
    m=re.search(r'BRA\s+(?:`\(\.L_x_\d+\)\s*)?\S*?0x([0-9a-f]+)',t) or re.search(r'BRA.*?(0x[0-9a-f]+)',t)
    if 'BRA' in t:
        mm=re.search(r'0x([0-9a-f]+)',t)
        if mm:
            tg=int(mm.group(1),16)
            if tg<a and tg in addr:
                body=ins[addr[tg]:i+1]
                h=collections.Counter(x[1].split()[0].lstrip('@!P0123456789UPT ').split('.')[0] if not x[1].startswith('@') else x[1].split()[1].split('.')[0] for x in body)
                ng=h.get('HGMMA',0)
                if ng: print(hex(tg),hex(a),len(body),'HGMMA',ng,dict(h.most_common(25)))
