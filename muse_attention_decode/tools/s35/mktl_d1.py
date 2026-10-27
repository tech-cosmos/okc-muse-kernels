"""s35: LOCAL timeline copy (NEVER submit) of submission.py for decode 1 full (_gd_full RED path + _red_fin).
Usage: python tools/s35/mktl_d1.py submission.py /tmp/s35/submission.py ; then tools/s35/run_d1.py.
Stamps (int64 per program, 32 slots at ts[pid*32]): 0 globaltimer start, 1 clock start, 2 clock q ready,
3 clock after loop, 4 clock after partial stores, 5 clock end (after red add), 6 globaltimer end, 7 ntiles,
10+i clock after tile i (i < 16). _red_fin: ts[65536 + 0..3] globaltimer start / after wait / loads done / end."""
import sys

src = open(sys.argv[1]).read()


def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a[:80], src.count(a))
    src = src.replace(a, b)


rep("NR: gl.constexpr = 4, epi=True, RCW: gl.constexpr = 0, QF: gl.constexpr = False):\n        h = gl.program_id(2) * 2\n",
    "NR: gl.constexpr = 4, epi=True, RCW: gl.constexpr = 0, QF: gl.constexpr = False, ts_ptr=None):\n        h = gl.program_id(2) * 2\n"
    "        tsb = ts_ptr + gl.program_id(0) * 32\n"
    "        gl.store(tsb, _gt35(gl.program_id(0)))\n        gl.store(tsb + 1, _ck35(gl.program_id(0)))\n")
rep("""        k_scale = gl.load(ks_ptr)
        scale = qk_scale * k_scale

        m_i = gl.full([M], -1.0e30""", """        k_scale = gl.load(ks_ptr)
        scale = qk_scale * k_scale
        gl.store(tsb + 2, _ck35(gl.max(gl.max(q.to(gl.float32), 1), 0)))

        m_i = gl.full([M], -1.0e30""")
rep("""                acc = mma_v2(gl.where(row0, ph, 0.0), v0, acc)
                acc = mma_v2(gl.where(row0, 0.0, ph), v1, acc)
""", """                acc = mma_v2(gl.where(row0, ph, 0.0), v0, acc)
                acc = mma_v2(gl.where(row0, 0.0, ph), v1, acc)
                gl.store(tsb + 10 + gl.minimum(i, 15), _ck35(gl.max(gl.max(acc, 1), 0)))
""")
rep("""        cp.wait_group(0)

        # Partials [B, HKV, S + 1, G, D]""", """        cp.wait_group(0)
        gl.store(tsb + 3, _ck35(gl.max(l_i, 0)))
        gl.store(tsb + 7, ntiles.to(gl.int64))

        # Partials [B, HKV, S + 1, G, D]""")
rep("""        if RED:
            if epi:
                _gd_red_add(acc, m_i, l_i, flg, split, kid, red_ptr, rcnt_ptr, M, D, NR)
""", """        gl.store(tsb + 4, _ck35(gl.max(l_i, 0)))
        if RED:
            if epi:
                _gd_red_add(acc, m_i, l_i, flg, split, kid, red_ptr, rcnt_ptr, M, D, NR)
        gl.store(tsb + 5, _ck35(gl.max(l_i, 0)))
        gl.store(tsb + 6, _gt35(gl.max(l_i, 0)))
""")
# thread ts_ptr through _gd_full -> body (3 calls)
rep("rcnt_ptr=None, out_ptr=None, NR: gl.constexpr = 4, RCW: gl.constexpr = 0, QF: gl.constexpr = False):\n        # SPEC > 0",
    "rcnt_ptr=None, out_ptr=None, NR: gl.constexpr = 4, RCW: gl.constexpr = 0, QF: gl.constexpr = False, ts_ptr=None):\n        # SPEC > 0")
rep("red_ptr, rcnt_ptr, out_ptr, NR, real == SPEC, RCW, QF)", "red_ptr, rcnt_ptr, out_ptr, NR, real == SPEC, RCW, QF, ts_ptr)")
rep("red_ptr, rcnt_ptr, out_ptr, NR, True, RCW, QF)", "red_ptr, rcnt_ptr, out_ptr, NR, True, RCW, QF, ts_ptr)", 2)
# _red_fin stamps
rep("""             SL: tl.constexpr = 0, RCW: tl.constexpr = 0, BAL: tl.constexpr = False):
""", """             SL: tl.constexpr = 0, RCW: tl.constexpr = 0, BAL: tl.constexpr = False, ts_ptr=None):
""")
rep("""    vs = tl.load(vs_ptr)
    gdc_wait()
    a = tl.zeros([M, D], tl.float32)""", """    vs = tl.load(vs_ptr)
    tl.store(ts_ptr + 65536, _tgt35(b))
    gdc_wait()
    tl.store(ts_ptr + 65537, _tgt35(b))
    a = tl.zeros([M, D], tl.float32)""")
rep("""    m = tl.where(l > 0.0, 0.0, -1.0e30)
    for i in range(nf):""", """    m = tl.where(l > 0.0, 0.0, -1.0e30)
    tl.store(ts_ptr + 65538, _tgt35(tl.sum(l).to(tl.int32)))
    for i in range(nf):""")
rep("""    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))""", """    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))
    tl.store(ts_ptr + 65539, _tgt35(tl.sum(o).to(tl.int32)))""")
rep("""                       S=splits, NR=4 if batch == 1 else BNR,""", """                       S=splits, ts_ptr=_TS35[0], NR=4 if batch == 1 else BNR,""")
rep("""        QF=plan.get("qf", False), **(_red_args(batch, group, d, s, device, out) if red else {}), num_warps=1)
    if red:
        _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, s, False)""",
    """        QF=plan.get("qf", False), **(_red_args(batch, group, d, s, device, out) if red else {}), ts_ptr=_TS35[0], num_warps=1)
    if red:
        _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, s, False)""")
# helpers
rep("from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait\n",
    """from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

_TS35 = [None]


@triton.jit
def _tgt35(x):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x.to(tl.int32)],
                                     dtype=tl.int64, is_pure=False, pack=1)
""")
rep("    _gd_full_redo = gluon.jit(noinline=True)(_gd_full_body.fn)\n", "    _gd_full_redo = gluon.jit(noinline=True)(_gd_full_body.fn)\n", 1)
open(sys.argv[2], "w").write(src)
print("wrote", sys.argv[2])
src = open(sys.argv[2]).read()
rep("if HAVE_GLUON:\n    @gluon.jit\n", """if HAVE_GLUON:
    @gluon.jit
    def _gt35(x):
        return gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x.to(gl.int32)], dtype=gl.int64,
                                         is_pure=False, pack=1)

    @gluon.jit
    def _ck35(x):
        return gl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,r", [x.to(gl.int32)], dtype=gl.int64,
                                         is_pure=False, pack=1)

    @gluon.jit
""")
open(sys.argv[2], "w").write(src)
