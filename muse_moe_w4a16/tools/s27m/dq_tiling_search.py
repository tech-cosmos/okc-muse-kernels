import itertools
P=[6,7,8,15,22,23,24,31]
N=64
cands=set()
for o in range(N): cands.add(frozenset((p+o)%N for p in P))
for sel in itertools.product(range(8),repeat=4):
    cands.add(frozenset(sel[p//8]*8+p%8 for p in P))
cands=[c for c in cands if len(c)==8]
base=frozenset(P)|frozenset(p+32 for p in P)
cands=[c for c in cands if not c&base]
print(len(cands))
by={}
for c in cands:
    for b in c: by.setdefault(b,[]).append(c)
sols=[]
def rec(cov,ch):
    if len(cov)==N: sols.append(ch); return True
    t=min(set(range(N))-cov)
    for c in by.get(t,[]):
        if not c&cov:
            if rec(cov|c,ch+[c]): return True
    return False
rec(base,[])
print(sols[:1])
