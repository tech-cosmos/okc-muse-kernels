# n5s39 family F2 (z3 SAT, Bool-level routing model; no bit-vector arithmetic):
#   intermediates (each 1 int op): w = q & C (C free 32-bit), s = w*K [+ X] (IMAD, K = up to 3 free set bits; copies of
#   w shifted by each set bit of K; a collision at bit t makes every bit >= t garbage), X / Y = shl/shr/rot of q (free).
#   outputs (4): DIRECT (an intermediate that is already exactly clean) or a final LOP3 over 1-2 sources in
#   {q, w, s, X, Y}: MASK  o = x & M;  SEL o = imm ? x : y (non-P bits need x or y zero there, P bits pick either);
#   MOR o = M & (x | y) (each P bit from one source, the other one zero there).
#   total ops = #intermediates + #LOP3 finals. Layout free (exact cover of the 32 q bits by the 32 P slots).
# usage: python3 f2sat.py <inter list e.g. w,s,X> <n_final_lop3> [addX=0|1]
import sys, itertools
from z3 import *
P = [6, 7, 8, 15, 22, 23, 24, 31]; PS = set(P); NP = [t for t in range(32) if t not in PS]
inter = sys.argv[1].split(","); NF = int(sys.argv[2]); ADDX = len(sys.argv) > 3 and sys.argv[3] == "1"
NDIR = 4 - NF
s = Solver()
C = [Bool(f"C{t}") for t in range(32)]
kd = [Bool(f"k{d}") for d in range(32)]
s.add(AtMost(*kd, 3)); s.add(AtLeast(*kd, 1))
# shift sources: one-hot over 93 options; entry(p) = q bit or None(zero)
shifts = []
for r in range(1, 32):
    shifts.append(("rot", r, lambda p, r=r: (p - r) % 32))
    shifts.append(("shl", r, lambda p, r=r: p - r if p - r >= 0 else None))
    shifts.append(("shr", r, lambda p, r=r: p + r if p + r <= 31 else None))
def shsel(name):
    v = [Bool(f"{name}_{j}") for j in range(len(shifts))]
    s.add(PbEq([(x, 1) for x in v], 1)); return v
SH = {n: shsel(n) for n in inter if n in ("X", "Y")}
# entries as: wire_src[p][i] = Bool "source bit p carries q_i"; zero_src[p] = Bool "is zero"; (else garbage)
def ent_q(p): return {p: BoolVal(True)}, BoolVal(False)
def ent_w(p): return {p: C[p]}, Not(C[p])
def ent_sh(n, p):
    wires = {}; zero = []
    for j, (_, _, f) in enumerate(shifts):
        i = f(p)
        if i is None: zero.append(SH[n][j])
        else: wires.setdefault(i, []).append(SH[n][j])
    return {i: Or(v) for i, v in wires.items()}, Or(zero) if zero else BoolVal(False)
# s = sum_d (w << d) for k_d  [+ X]
a = {(c, d): And(C[c], kd[d]) for c in range(32) for d in range(32) if c + d < 32}
contrib = {t: [(c, d) for (c, d) in a if c + d == t] for t in range(32)}
coll = []
for t in range(32):
    terms = [a[cd] for cd in contrib[t]]
    if ADDX and "X" in SH:
        xw, xz = ent_sh("X", t); terms.append(Not(xz))
    coll.append(AtLeast(*terms, 2) if len(terms) >= 2 else BoolVal(False))
garb = []; acc = BoolVal(False)
for t in range(32): acc = Or(acc, coll[t]); garb.append(acc)
def ent_s(p):
    wires = {}
    for (c, d) in contrib[p]: wires.setdefault(c, []).append(a[(c, d)])
    anyc = Or([a[cd] for cd in contrib[p]]) if contrib[p] else BoolVal(False)
    zero = And(Not(anyc), Not(garb[p]))
    if ADDX and "X" in SH:
        xw, xz = ent_sh("X", p)
        for i, b in xw.items(): wires.setdefault(i, []).append(And(b, Not(anyc)))
        zero = And(zero, xz)
    return {i: And(Or(v), Not(garb[p])) for i, v in wires.items()}, zero
srcs = ["q"] + inter
def ent(n, p):
    if n == "q": return ent_q(p)
    if n == "w": return ent_w(p)
    if n == "s": return ent_s(p)
    return ent_sh(n, p)
E = {(n, p): ent(n, p) for n in srcs for p in range(32)}
usage = {i: [] for i in range(32)}   # Bools: slot uses q bit i
outs = []
for o in range(4):
    if o < NDIR:   # direct output: one of the intermediates, all non-P zero, P slots its wires
        ch = [Bool(f"d{o}_{n}") for n in inter]; s.add(PbEq([(x, 1) for x in ch], 1))
        for n, b in zip(inter, ch):
            for t in NP: s.add(Implies(b, E[(n, t)][1]))
            for p in P:
                wires, z = E[(n, p)]
                s.add(Implies(b, Not(z)))
                s.add(Implies(b, Or([v for v in wires.values()] or [False])))
                for i, v in wires.items(): usage[i].append(And(b, v))
        continue
    # final lop3: pattern x,y,kind
    opts = []
    for x in srcs:
        opts.append(("MASK", x, x))
    for x, y in itertools.combinations(srcs, 2):
        opts.append(("SEL", x, y)); opts.append(("MOR", x, y))
    ch = [Bool(f"f{o}_{k}") for k in range(len(opts))]; s.add(PbEq([(b, 1) for b in ch], 1))
    for b, (kind, x, y) in zip(ch, opts):
        if kind == "SEL":
            for t in NP: s.add(Implies(b, Or(E[(x, t)][1], E[(y, t)][1])))
        for p in P:
            pick = Bool(f"pk{o}_{p}_{kind}_{x}_{y}")   # True: take x, False: take y
            wx, zx = E[(x, p)]; wy, zy = E[(y, p)]
            if kind == "MASK":
                s.add(Implies(b, Or(list(wx.values()) or [False])))
                for i, v in wx.items(): usage[i].append(And(b, v))
            else:
                if kind == "MOR":
                    s.add(Implies(And(b, pick), zy)); s.add(Implies(And(b, Not(pick)), zx))
                s.add(Implies(And(b, pick), Or(list(wx.values()) or [False])))
                s.add(Implies(And(b, Not(pick)), Or(list(wy.values()) or [False])))
                for i, v in wx.items(): usage[i].append(And(b, pick, v))
                for i, v in wy.items(): usage[i].append(And(b, Not(pick), v))
for i in range(32): s.add(PbEq([(u, 1) for u in usage[i]], 1))
r = s.check()
print("F2", sys.argv[1:], "ops", len(inter) + NF, "->", r, flush=True)
if r == sat:
    m = s.model()
    print(" C=0x%08x" % sum(1 << t for t in range(32) if is_true(m.eval(C[t], True))),
          " K bits", [d for d in range(32) if is_true(m.eval(kd[d], True))])
    for n in SH: print(" ", n, [shifts[j][:2] for j in range(len(shifts)) if is_true(m.eval(SH[n][j], True))])
    print(" ", [str(d) for d in m.decls() if (d.name().startswith("f") or d.name().startswith("d")) and is_true(m[d])])
