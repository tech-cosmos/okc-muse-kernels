# n5s39 family F3 = F2 generalised: intermediates given as a spec list, each one int op:
#   NAME=sh            shl/shr/rot of q (free amount)
#   NAME=mask:BASE     BASE & C_NAME (C free), BASE in q or an earlier sh/sel
#   NAME=imad:BASE[:ADD]  BASE*K_NAME (+ ADD) with K = 1..3 free set bits, BASE a mask; collision at t -> bits >= t garbage
#   NAME=sel:A:B       per-bit select of A/B by a free immediate, positions may also be forced to 0 only where A or B is 0
# outputs: NDIR direct (exactly clean intermediates), the rest final LOP3s (MASK / SEL / MOR over 2 sources, see f2sat.py).
# usage: python3 f3sat.py "X=sh,w=mask:q,s=imad:w,t=imad:w" <n_final_lop3>
import sys, itertools
from z3 import *
P = [6, 7, 8, 15, 22, 23, 24, 31]; PS = set(P); NP = [t for t in range(32) if t not in PS]
spec = [x.split("=") for x in sys.argv[1].split(",")]; NF = int(sys.argv[2]); NDIR = 4 - NF
assert 0 <= NDIR <= 4
s = Solver()
shifts = []
for r in range(1, 32):
    shifts.append(lambda p, r=r: (p - r) % 32)
    shifts.append(lambda p, r=r: p - r if p - r >= 0 else None)
    shifts.append(lambda p, r=r: p + r if p + r <= 31 else None)
F = BoolVal(False); T = BoolVal(True)
E = {}   # (name, p) -> (wires dict i->Bool, zero Bool); garbage = neither
for p in range(32): E[("q", p)] = ({p: T}, F)
def OR(l): return Or(l) if l else F
for name, d in spec:
    kind = d.split(":")
    if kind[0] == "sh":
        v = [Bool(f"{name}_{j}") for j in range(len(shifts))]; s.add(PbEq([(x, 1) for x in v], 1))
        for p in range(32):
            wires = {}; zero = []
            for j, f in enumerate(shifts):
                i = f(p)
                (zero.append(v[j]) if i is None else wires.setdefault(i, []).append(v[j]))
            E[(name, p)] = ({i: OR(l) for i, l in wires.items()}, OR(zero))
    elif kind[0] == "mask":
        Cb = [Bool(f"{name}_C{t}") for t in range(32)]
        for p in range(32):
            w, z = E[(kind[1], p)]
            E[(name, p)] = ({i: And(Cb[p], b) for i, b in w.items()}, Or(Not(Cb[p]), z))
    elif kind[0] == "sel":
        A, B = kind[1], kind[2]
        for p in range(32):
            c = Bool(f"{name}_c{p}")
            wa, za = E[(A, p)]; wb, zb = E[(B, p)]
            wires = {}
            for i, b in wa.items(): wires.setdefault(i, []).append(And(c, b))
            for i, b in wb.items(): wires.setdefault(i, []).append(And(Not(c), b))
            E[(name, p)] = ({i: OR(l) for i, l in wires.items()}, Or(And(c, za), And(Not(c), zb)))
    elif kind[0] == "fun":   # SHF.L funnel: hi word of (B:A) << k, k free 1..31 (A or B may be 'z' = RZ)
        A, B = kind[1], kind[2]
        v = [Bool(f"{name}_k{k}") for k in range(1, 32)]; s.add(PbEq([(x, 1) for x in v], 1))
        def src(nm, p):
            return ({}, T) if nm == "z" else E[(nm, p)]
        for p in range(32):
            wires = {}; zero = []
            for k, b in zip(range(1, 32), v):
                w, z = src(B, p - k) if p - k >= 0 else src(A, p - k + 32)
                for i, wb in w.items(): wires.setdefault(i, []).append(And(b, wb))
                zero.append(And(b, z))
            E[(name, p)] = ({i: OR(l) for i, l in wires.items()}, OR(zero))
    elif kind[0] == "pr":    # PRMT: each output byte = any byte of A or B, or zero (B = 'z' -> RZ); sign mode ignored
        A, B = kind[1], kind[2]
        bsrc = [(A, m) for m in range(4)] + [(B, m) for m in range(4)]
        for j in range(4):
            v = [Bool(f"{name}_b{j}_{m}") for m in range(8)]; s.add(PbEq([(x, 1) for x in v], 1))
            for bit in range(8):
                p = 8 * j + bit; wires = {}; zero = []
                for b, (nm, m) in zip(v, bsrc):
                    w, z = ({}, T) if nm == "z" else E[(nm, 8 * m + bit)]
                    for i, wb in w.items(): wires.setdefault(i, []).append(And(b, wb))
                    zero.append(And(b, z))
                E[(name, p)] = ({i: OR(l) for i, l in wires.items()}, OR(zero))
    elif kind[0] == "imad":
        base = kind[1]; add = kind[2] if len(kind) > 2 else None
        kd = [Bool(f"{name}_k{d}") for d in range(32)]; s.add(AtMost(*kd, 3)); s.add(AtLeast(*kd, 1))
        nz = {}; contrib = {t: [] for t in range(32)}
        for c in range(32):
            w, z = E[(base, c)]
            for d in range(32 - c):
                act = And(Not(z), kd[d]); contrib[c + d].append((act, w, kd[d]))
        garb = []; acc = F
        for t in range(32):
            terms = [x[0] for x in contrib[t]]
            if add: terms.append(Not(E[(add, t)][1]))
            acc = Or(acc, AtLeast(*terms, 2) if len(terms) >= 2 else F); garb.append(acc)
        for p in range(32):
            wires = {}
            anyc = OR([x[0] for x in contrib[p]])
            for act, w, kb in contrib[p]:
                for i, b in w.items(): wires.setdefault(i, []).append(And(kb, b))
            zero = Not(anyc)
            if add:
                wa, za = E[(add, p)]
                for i, b in wa.items(): wires.setdefault(i, []).append(And(Not(anyc), b))
                zero = And(zero, za)
            # garbage base bits (neither wire nor zero) make the copy garbage: require base clean where active
            E[(name, p)] = ({i: And(OR(l), Not(garb[p])) for i, l in wires.items()}, And(zero, Not(garb[p])))
        # an active copy of a garbage base bit is garbage: forbid (keeps the model simple and sound for routing)
        for c in range(32):
            w, z = E[(base, c)]
            s.add(Implies(Not(z), OR(list(w.values()))))
inter = [n for n, _ in spec]; srcs = ["q"] + inter
usage = {i: [] for i in range(32)}
def wire_or(w): return OR(list(w.values()))
for o in range(4):
    if o < NDIR:
        ch = [Bool(f"d{o}_{n}") for n in inter]; s.add(PbEq([(x, 1) for x in ch], 1))
        for n, b in zip(inter, ch):
            for t in NP: s.add(Implies(b, E[(n, t)][1]))
            for p in P:
                w, z = E[(n, p)]; s.add(Implies(b, wire_or(w)))
                for i, v in w.items(): usage[i].append(And(b, v))
        continue
    opts = [("MASK", x, x) for x in srcs] + [(k, x, y) for x, y in itertools.combinations(srcs, 2) for k in ("SEL", "MOR")]
    ch = [Bool(f"f{o}_{k}") for k in range(len(opts))]; s.add(PbEq([(b, 1) for b in ch], 1))
    for b, (kind, x, y) in zip(ch, opts):
        if kind == "SEL":
            for t in NP: s.add(Implies(b, Or(E[(x, t)][1], E[(y, t)][1])))
        for p in P:
            wx, zx = E[(x, p)]; wy, zy = E[(y, p)]
            if kind == "MASK":
                s.add(Implies(b, wire_or(wx)))
                for i, v in wx.items(): usage[i].append(And(b, v))
                continue
            pick = Bool(f"pk{o}_{p}_{kind}_{x}_{y}")
            if kind == "MOR":
                s.add(Implies(And(b, pick), zy)); s.add(Implies(And(b, Not(pick)), zx))
            s.add(Implies(And(b, pick), wire_or(wx))); s.add(Implies(And(b, Not(pick)), wire_or(wy)))
            for i, v in wx.items(): usage[i].append(And(b, pick, v))
            for i, v in wy.items(): usage[i].append(And(b, Not(pick), v))
for i in range(32): s.add(PbEq([(u, 1) for u in usage[i]], 1))
r = s.check()
print("F3", sys.argv[1], "finals", NF, "ops", len(inter) + NF, "->", r, flush=True)
if r == sat:
    m = s.model()
    print("  ", sorted(str(d) + "=" + str(m[d]) for d in m.decls() if is_true(m[d]) and not d.name().startswith("pk")))
