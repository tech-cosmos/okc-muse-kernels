# n5s39 superoptimizer v2 (CEGIS, z3; leaner encoding than supopt.py).
# Is there an N-instruction program (32-bit SASS-like ops) mapping a packed u32 q of 8 e2m1 codes (bit layout FREE =
# chosen by prepare, encoded as a one-hot permutation matrix) to 4 bf16x2 words o_k = sum_j q[pi(k,j)] << P[j]
# (P = sign/e1/e0/m slots of both bf16 halves; every other bit 0)?  Today's dq(): N = 8.
# ops: lop3(a,b,c,lut) [any 3-input boolean fn], shfl/shfr = SHF.L/R funnel of (b:a) by const k (covers shl/shr/rot),
#      prmt(a,b,sel=c) [incl. sign-replicate], imad = a*K + c with K any constant of <= 3 set bits (a<<k1 + a<<k2 + a<<k3),
#      iadd = a + b + c.  operands: q, 0, IMM_i (one free 32-bit immediate per instruction), earlier values.
# usage: python3 supopt2.py N ops [timeout_s]
import sys, random, time
from z3 import *
P = [6, 7, 8, 15, 22, 23, 24, 31]
N = int(sys.argv[1]); OPS = sys.argv[2].split(","); TMO = int(sys.argv[3]) if len(sys.argv) > 3 else 36000
W = 32
s = SolverFor("QF_BV")
typ = [BitVec(f"t{i}", 3) for i in range(N)]
opa = [[BitVec(f"o{i}_{j}", 4) for j in range(3)] for i in range(N)]
imm = [BitVec(f"imm{i}", W) for i in range(N)]
kk = [[BitVec(f"k{i}_{j}", W) for j in range(3)] for i in range(N)]
ke = [[Bool(f"ke{i}_{j}") for j in range(3)] for i in range(N)]
lut = [BitVec(f"lut{i}", 8) for i in range(N)]
outsel = [BitVec(f"out{k}", 4) for k in range(4)]
IX = [[BitVec(f"x{k}_{j}", 5) for j in range(8)] for k in range(4)]
for i in range(N):
    s.add(ULT(typ[i], len(OPS)))
    for j in range(3): s.add(ULT(opa[i][j], 3 + i)); s.add(ULT(kk[i][j], 32))
for k in range(4): s.add(ULT(outsel[k], N))
import os
if os.environ.get('TYPES'):
    for i, nm in enumerate(os.environ['TYPES'].split(',')): s.add(typ[i] == OPS.index(nm))
if os.environ.get('OUTS'):
    for k, v in enumerate(os.environ['OUTS'].split(',')): s.add(outsel[k] == int(v))
s.add(ULT(outsel[0], outsel[1]), ULT(outsel[1], outsel[2]), ULT(outsel[2], outsel[3]), outsel[3] == N - 1)
slots = [(k, j) for k in range(4) for j in range(8)]
s.add(Distinct(*[IX[k][j] for (k, j) in slots]))

def bperm(a, b, sel):
    by = [Extract(8 * j + 7, 8 * j, a) for j in range(4)] + [Extract(8 * j + 7, 8 * j, b) for j in range(4)]
    out = []
    for j in range(4):
        nib = Extract(4 * j + 3, 4 * j, sel)
        v = by[7]
        for m in range(6, -1, -1): v = If(Extract(2, 0, nib) == m, by[m], v)
        out.append(If(Extract(3, 3, nib) == 1, If(Extract(7, 7, v) == 1, BitVecVal(0xFF, 8), BitVecVal(0, 8)), v))
    return Concat(out[3], out[2], out[1], out[0])

def lop3(a, b, c, L):
    r = BitVecVal(0, W)
    for m in range(8):
        t = (a if m & 4 else ~a) & (b if m & 2 else ~b) & (c if m & 1 else ~c)
        r = r | If(Extract(m, m, L) == 1, t, BitVecVal(0, W))
    return r

def run(q):
    vals = []
    for i in range(N):
        def opnd(j):
            v = If(opa[i][j] == 0, q, If(opa[i][j] == 1, BitVecVal(0, W), imm[i]))
            for m in range(i): v = If(opa[i][j] == 3 + m, vals[m], v)
            return v
        a, b, c = opnd(0), opnd(1), opnd(2)
        cand = []
        for o in OPS:
            if o == "lop3": cand.append(lop3(a, b, c, lut[i]))
            elif o == "shfl": cand.append(Extract(63, 32, Concat(b, a) << ZeroExt(32, kk[i][0])))
            elif o == "shfr": cand.append(Extract(31, 0, LShR(Concat(b, a), ZeroExt(32, kk[i][0]))))
            elif o == "prmt": cand.append(bperm(a, b, c))
            elif o == "imad":
                acc = c
                for j in range(3): acc = acc + If(ke[i][j], a << kk[i][j], BitVecVal(0, W))
                cand.append(acc)
            elif o == "iadd": cand.append(a + b + c)
        v = cand[-1]
        for t in range(len(OPS) - 2, -1, -1): v = If(typ[i] == t, cand[t], v)
        vals.append(v)
    return vals

def spec_conc(qv, vals):  # concrete q: wanted bit = OR of X over set q bits
    cs = []
    for k in range(4):
        o = vals[0]
        for m in range(1, N): o = If(outsel[k] == m, vals[m], o)
        for b in range(32):
            ob = Extract(b, b, o) == 1
            if b in P:
                j = P.index(b)
                cs.append(ob == (Extract(0, 0, LShR(BitVecVal(qv, 32), ZeroExt(27, IX[k][j]))) == 1))
            else:
                cs.append(Not(ob))
    return And(cs)

tests = [0, 0xFFFFFFFF] + [random.getrandbits(32) for _ in range(6)]
for t in tests: s.add(spec_conc(t, run(BitVecVal(t, W))))
s.set("timeout", TMO * 1000)
t0 = time.time(); it = 0
while True:
    it += 1
    r = s.check()
    print(f"iter {it} tests {len(tests)} -> {r} ({time.time()-t0:.0f}s)", flush=True)
    if r != sat: print("RESULT", N, OPS, r, f"{time.time()-t0:.0f}s"); break
    mdl = s.model()
    perm = {(k, j): mdl.eval(IX[k][j], True).as_long() for (k, j) in slots}
    # concrete verification by bit-blasting a symbolic q against the model program
    v = Solver(); qs = BitVec("q", W)
    sub = []
    for i in range(N):
        sub += [(typ[i], mdl.eval(typ[i], True)), (imm[i], mdl.eval(imm[i], True)), (lut[i], mdl.eval(lut[i], True))]
        sub += [(opa[i][j], mdl.eval(opa[i][j], True)) for j in range(3)]
        sub += [(kk[i][j], mdl.eval(kk[i][j], True)) for j in range(3)] + [(ke[i][j], mdl.eval(ke[i][j], True)) for j in range(3)]
    sub += [(outsel[k], mdl.eval(outsel[k], True)) for k in range(4)]
    vals = [substitute(x, *sub) for x in run(qs)]
    bad = []
    for k in range(4):
        o = simplify(substitute(vals[0] if True else 0, *[]))
        o = vals[0]
        for m in range(1, N): o = If(mdl.eval(outsel[k]).as_long() == m, vals[m], o)
        want = BitVecVal(0, W)
        for j in range(8): want = want | (ZeroExt(31, Extract(perm[(k, j)], perm[(k, j)], qs)) << P[j])
        bad.append(o != want)
    v.add(Or(bad))
    if v.check() == unsat:
        print("FOUND program:")
        nm = ["q", "0", "imm"]
        for i in range(N):
            t = OPS[mdl.eval(typ[i]).as_long()]
            on = [(nm[o] if o < 3 else f"v{o-3}") for o in (mdl.eval(opa[i][j]).as_long() for j in range(3))]
            ks = [(mdl.eval(kk[i][j], True).as_long(), is_true(mdl.eval(ke[i][j], True))) for j in range(3)]
            print(f"  v{i} = {t}({', '.join(on)}) imm=0x{mdl.eval(imm[i], True).as_long():08x} k={ks} lut=0x{mdl.eval(lut[i], True).as_long():02x}")
        print("  outputs:", [f"v{mdl.eval(outsel[k]).as_long()}" for k in range(4)])
        print("  layout (q bit per P slot):", [[perm[(k, j)] for j in range(8)] for k in range(4)])
        print("RESULT", N, OPS, "sat", f"{time.time()-t0:.0f}s"); break
    cq = v.model().eval(qs, True).as_long()
    tests.append(cq); s.add(spec_conc(cq, run(BitVecVal(cq, W))))
