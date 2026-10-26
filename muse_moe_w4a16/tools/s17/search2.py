import itertools
T = [6, 7, 8, 15, 22, 23, 24, 31]; TS = set(T)
def mul_map(R, C):
    # R: source bits, C: list of shift terms. no-carry rule: all b+c distinct (copies >31 dropped but still distinct-checked below 32)
    land = [b + c for b in R for c in C if b + c < 32]
    if len(land) != len(set(land)): return None
    hit = {}
    for b in R:
        h = [b + c for c in C if b + c in TS]
        if len(h) != 1: return None
        hit[b] = h[0]
    if set(hit.values()) != TS: return None
    return hit
sols = []
for a in range(1, 7):
    used = set(T) | {t - a for t in T}
    if min(used) < 0 or len(used) != 16: continue
    rem = [b for b in range(32) if b not in used]
    # option A: rot r for 8 of rem, mul for the rest
    for r in range(1, 32):
        sr = {(t - r) % 32 for t in T}
        if not sr <= set(rem): continue
        R = [b for b in rem if b not in sr]
        for k in (1, 2, 3, 4):
            for C in itertools.combinations(range(0, 32), k):
                if mul_map(R, C): sols.append(('shift', a, 'rot', r, 'mul', R, C))
    # option B: two muls
    for R1 in itertools.combinations(rem, 8):
        if rem[0] not in R1: continue
        R2 = [b for b in rem if b not in R1]
        for k1 in (1, 2, 3):
            c1 = [C for C in itertools.combinations(range(32), k1) if mul_map(R1, C)]
            if not c1: continue
            for k2 in (1, 2, 3):
                c2 = [C for C in itertools.combinations(range(32), k2) if mul_map(R2, C)]
                if c2: sols.append(('shift', a, 'mul', R1, c1[0], 'mul', R2, c2[0]))
print(len(sols))
for s in sols[:20]: print(s)
