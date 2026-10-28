# n5s39 superoptimizer (CEGIS, z3) for the twm/tk2f dq() integer part.
# Question: is there an N-instruction program over SASS-like 32-bit ops that maps a packed u32 q of 8 e2m1 codes
# (bit layout = free, chosen by prepare) to 4 bf16x2 words o_k with o_k = sum_j q[idx(k,j)] << P[j]
# (P = sign/e1/e0/m slots of both halves; all other bits 0)?  Today: N = 8 (o0 = q&M; rotl10&M; rotl13&M; ((q&C)*K)&M).
# ISA per instruction (one 32-bit immediate IMM_i + shift amount K_i each): LOP3(a,b,c,lut), SHF.L/R funnel (b:a),
# PRMT(a,b,sel incl. sign-replicate), IMAD(a*b+c), IADD3(a+b+c) [with negation via LOP3 already], operands any of
# q, 0, IMM_i, earlier values.  Usage: /usr/local/bin/python3 supopt.py N [ops=lop3,shf,prmt,imad,iadd] [timeout_s]
import sys, random, time
from z3 import *
P = [6, 7, 8, 15, 22, 23, 24, 31]
N = int(sys.argv[1])
OPS = (sys.argv[2] if len(sys.argv) > 2 else "lop3,shfl,shfr,prmt,imad,iadd").split(",")
TMO = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
W = 32
s = SolverFor("QF_BV") if False else Solver()
typ = [Int(f"t{i}") for i in range(N)]
opa = [[Int(f"o{i}_{j}") for j in range(3)] for i in range(N)]
imm = [BitVec(f"imm{i}", W) for i in range(N)]
kk = [BitVec(f"k{i}", W) for i in range(N)]
lut = [BitVec(f"lut{i}", 8) for i in range(N)]
outsel = [Int(f"out{k}") for k in range(4)]
idx = [[BitVec(f"x{k}_{j}", W) for j in range(8)] for k in range(4)]
for i in range(N):
    s.add(typ[i] >= 0, typ[i] < len(OPS))
    for j in range(3): s.add(opa[i][j] >= 0, opa[i][j] < 3 + i)   # 0=q 1=zero 2=imm 3+m = v_m
    s.add(ULT(kk[i], 32))
for k in range(4): s.add(outsel[k] >= 0, outsel[k] < N)
s.add(outsel[0] < outsel[1], outsel[1] < outsel[2], outsel[2] < outsel[3])
allidx = [x for r in idx for x in r]
for x in allidx: s.add(ULT(x, 32))
s.add(Distinct(*allidx))
# symmetry/no-dead-code: output 3 is the last instruction
s.add(outsel[3] == N - 1)

def bperm(a, b, sel):
    bytes8 = [Extract(8 * j + 7, 8 * j, a) for j in range(4)] + [Extract(8 * j + 7, 8 * j, b) for j in range(4)]
    out = []
    for j in range(4):
        nib = Extract(4 * j + 3, 4 * j, sel)
        v = bytes8[7]
        for m in range(6, -1, -1):
            v = If(Extract(2, 0, nib) == m, bytes8[m], v)
        sgn = If(Extract(7, 7, v) == 1, BitVecVal(0xFF, 8), BitVecVal(0, 8))
        out.append(If(Extract(3, 3, nib) == 1, sgn, v))
    return Concat(out[3], out[2], out[1], out[0])

def lop3(a, b, c, L):
    r = BitVecVal(0, W)
    for m in range(8):
        ta = a if m & 4 else ~a
        tb = b if m & 2 else ~b
        tc = c if m & 1 else ~c
        r = r | If(Extract(m, m, L) == 1, ta & tb & tc, BitVecVal(0, W))
    return r

def run(q, consts):
    """consts: dict of concrete values or None (symbolic). returns list of values"""
    vals = []
    for i in range(N):
        def opnd(j):
            v = If(opa[i][j] == 0, q, If(opa[i][j] == 1, BitVecVal(0, W), imm[i]))
            for m in range(i):
                v = If(opa[i][j] == 3 + m, vals[m], v)
            return v
        a, b, c = opnd(0), opnd(1), opnd(2)
        cand = []
        for o in OPS:
            if o == "lop3": cand.append(lop3(a, b, c, lut[i]))
            elif o == "shfl":  # hi word of (b:a) << k
                cand.append(Extract(63, 32, Concat(b, a) << ZeroExt(32, kk[i])))
            elif o == "shfr":  # lo word of (b:a) >> k
                cand.append(Extract(31, 0, LShR(Concat(b, a), ZeroExt(32, kk[i]))))
            elif o == "prmt": cand.append(bperm(a, b, c))
            elif o == "imad": cand.append(a * b + c)
            elif o == "iadd": cand.append(a + b + c)
        v = cand[-1]
        for t in range(len(OPS) - 2, -1, -1):
            v = If(typ[i] == t, cand[t], v)
        vals.append(v)
    return vals

def spec_ok(q, vals):
    cs = []
    for k in range(4):
        o = vals[0]
        for m in range(1, N): o = If(outsel[k] == m, vals[m], o)
        want = BitVecVal(0, W)
        for j in range(8):
            want = want | ((LShR(q, idx[k][j]) & 1) << P[j])
        cs.append(o == want)
    return And(cs)

tests = [0, 0xFFFFFFFF] + [1 << i for i in range(32)] + [random.getrandbits(32) for _ in range(8)]
for t in tests:
    qv = BitVecVal(t, W)
    s.add(spec_ok(qv, run(qv, None)))
s.set("timeout", TMO * 1000)
t0 = time.time()
it = 0
while True:
    it += 1
    r = s.check()
    print(f"iter {it} tests {len(tests)} -> {r} ({time.time()-t0:.0f}s)", flush=True)
    if r != sat:
        print("RESULT", N, OPS, r); break
    mdl = s.model()
    # verify with a fresh solver: exists q with mismatch?
    v = Solver()
    qs = BitVec("q", W)
    vals = run(qs, None)
    sub = []
    for i in range(N):
        sub += [(typ[i], mdl.eval(typ[i], True)), (imm[i], mdl.eval(imm[i], True)), (kk[i], mdl.eval(kk[i], True)),
                (lut[i], mdl.eval(lut[i], True))] + [(opa[i][j], mdl.eval(opa[i][j], True)) for j in range(3)]
    sub += [(outsel[k], mdl.eval(outsel[k], True)) for k in range(4)]
    sub += [(x, mdl.eval(x, True)) for x in allidx]
    v.add(Not(substitute(spec_ok(qs, vals), *sub)))
    if v.check() == unsat:
        print("FOUND program:")
        names = ["q", "0", "imm"]
        for i in range(N):
            t = OPS[mdl.eval(typ[i]).as_long()]
            ops = [mdl.eval(opa[i][j]).as_long() for j in range(3)]
            on = [names[o] if o < 3 else f"v{o-3}" for o in ops]
            print(f"  v{i} = {t}({', '.join(on)}) imm=0x{mdl.eval(imm[i], True).as_long():08x} k={mdl.eval(kk[i], True).as_long()} lut=0x{mdl.eval(lut[i], True).as_long():02x}")
        print("  outputs:", [f"v{mdl.eval(outsel[k]).as_long()}" for k in range(4)])
        print("  layout:", [[mdl.eval(x).as_long() for x in r] for r in idx])
        print("RESULT", N, OPS, "sat"); break
    cq = v.model().eval(qs, True).as_long()
    tests.append(cq)
    qv = BitVecVal(cq, W)
    s.add(spec_ok(qv, run(qv, None)))
