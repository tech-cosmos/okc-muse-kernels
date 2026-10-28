# n5s39 exhaustive family search for a 7-int-op dq(): 3 intermediates (each shl/shr/rot of q, i.e. SHF with q or RZ)
# + 4 final LOP3(x, y, IMM) outputs, x,y in {q, I1, I2, I3}, per-bit choice of the IMM bit (any lut).
# Per output bit the lop3 picks one of 2 functions g0/g1 of (x_p, y_p): non-P bits need one option == 0, P bits need
# one option == a single q wire. We enumerate all 256 luts per (x,y) pair and every 3-subset of shift sources.
import itertools, sys
import os
P = [int(x) for x in os.environ.get('PSET', '6,7,8,15,22,23,24,31').split(',')]; PS = set(P)
Z = -1  # entry: -1 = constant 0, 0..31 = q wire
def shl(a): return tuple(Z if p - a < 0 else p - a for p in range(32))
def shr(b): return tuple(Z if p + b > 31 else p + b for p in range(32))
def rot(r): return tuple((p - r) % 32 for p in range(32))
q = tuple(range(32))
srcs = {}
for k in range(1, 32):
    srcs[f"shl{k}"] = shl(k); srcs[f"shr{k}"] = shr(k); srcs[f"rot{k}"] = rot(k)
names = list(srcs)
# a 2-input boolean fn g on bits (x,y) given entries ex, ey: returns resulting entry or None (not single-wire/const)
def ev(g, ex, ey):
    # g: 4-bit truth table index (x<<1|y); evaluate symbolically over the 1-2 wires involved
    wires = sorted({e for e in (ex, ey) if e != Z})
    res = []
    for assign in itertools.product([0, 1], repeat=len(wires)):
        val = dict(zip(wires, assign))
        xv = 0 if ex == Z else val[ex]; yv = 0 if ey == Z else val[ey]
        res.append((g >> (xv * 2 + yv)) & 1)
    if all(r == 0 for r in res): return Z
    if all(r == 1 for r in res): return 'one'
    for w in wires:
        if all(r == a[wires.index(w)] for r, a in zip(res, itertools.product([0, 1], repeat=len(wires)))): return w
    return None
def options(x, y):
    """all (per-output) possible P-wirings: for each lut (g0,g1 pairs), check non-P bits can be 0;
    return list of tuples of 8 sets of wires possible at P positions (choice per bit independent)."""
    outs = set()
    for lut in range(256):
        g0 = sum(((lut >> (m * 2)) & 1) << m for m in range(4))  # imm bit = 0 -> minterms (x,y,0)
        g1 = sum(((lut >> (m * 2 + 1)) & 1) << m for m in range(4))
        ok = True; pw = []
        for p in range(32):
            r = {ev(g0, x[p], y[p]), ev(g1, x[p], y[p])}
            if p in PS:
                ws = frozenset(e for e in r if isinstance(e, int) and e >= 0)
                if not ws: ok = False; break
                pw.append(ws)
            elif Z not in r: ok = False; break
        if ok: outs.add(tuple(pw))
    return outs
cache = {}
def opts(a, b):
    key = (a, b) if a <= b else (b, a)
    if key not in cache:
        va = q if a == 'q' else srcs[a]; vb = q if b == 'q' else srcs[b]
        cache[key] = options(va, vb)
    return cache[key]
def solve(srcset):
    # collect all candidate outputs from pairs (incl. single = pair with itself)
    cands = set()
    L = ['q'] + list(srcset)
    for a, b in itertools.combinations_with_replacement(L, 2): cands |= opts(a, b)
    # expand per-bit choices into concrete 8-wire sets (distinct wires)
    conc = set()
    for c in cands:
        for pick in itertools.product(*c):
            if len(set(pick)) == 8: conc.add(frozenset(pick))
    conc = list(conc)
    # exact cover of 32 wires with 4 disjoint 8-sets
    by = {}
    for c in conc:
        for w in c: by.setdefault(w, []).append(c)
    def rec(cov, n, ch):
        if n == 4: return ch
        t = min(set(range(32)) - cov)
        for c in by.get(t, []):
            if not (c & cov):
                r = rec(cov | c, n + 1, ch + [c])
                if r: return r
        return None
    return rec(frozenset(), 0, [])
nsol = 0
combos = list(itertools.combinations(names, int(sys.argv[1]) if len(sys.argv) > 1 else 3))
if len(sys.argv) > 3: combos = combos[int(sys.argv[2])::int(sys.argv[3])]
print(len(combos), "source sets", flush=True)
for i, cs in enumerate(combos):
    r = solve(cs)
    if r:
        nsol += 1; print("SOLUTION", cs, [sorted(x) for x in r], flush=True)
        if nsol > 5: break
    if i % 5000 == 0: print("progress", i, flush=True)
print("done, solutions:", nsol)
