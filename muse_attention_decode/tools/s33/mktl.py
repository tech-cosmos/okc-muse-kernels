"""s33: H200 timeline DIAG copy (NEVER submit) of submission.py for verify 1 full (_attn_split + _combine).
Usage: python tools/s33/mktl.py submission.py /tmp/s33d/submission.py
On the first eager kernel_fn call of verify 1 full, graph-captures main's call with stamps (TS 1: phase stamps, TS 2:
+ per-tile stamps) and without (TS 0), replays each after a 512 MB read (cold L2), and prints 'tl33 ...' lines:
per-phase clock64 deltas per _attn_split program (us, calibrated on globaltimer), program start/end spread, and the
_combine programs' start / post-wait / end vs the last _attn_split end."""
import sys

src = open(sys.argv[1]).read()


def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a, src.count(a))
    src = src.replace(a, b)


# _attn_body / _attn_split: extra args.
rep("PREQ: tl.constexpr = 0, FIX: tl.constexpr = 0,\n                FDELTA: tl.constexpr = 14.0):\n    \"\"\"One program",
    "PREQ: tl.constexpr = 0, FIX: tl.constexpr = 0,\n                FDELTA: tl.constexpr = 14.0, TS: tl.constexpr = 0, ts_ptr=None):\n    \"\"\"One program")
rep("FIX: tl.constexpr = 0, FDELTA: tl.constexpr = 14.0):\n    \"\"\"Kernel wrapper",
    "FIX: tl.constexpr = 0, FDELTA: tl.constexpr = 14.0, TS: tl.constexpr = 0, ts_ptr=None):\n    \"\"\"Kernel wrapper")
rep("PREQ, FIX, FDELTA)\n", "PREQ, FIX, FDELTA, TS, ts_ptr)\n", 2)
# stamps in _attn_body
rep("""    split = tl.program_id(0)
    b = tl.program_id(1)
    # DUAL (with HPP 1)""", """    split = tl.program_id(0)
    b = tl.program_id(1)
    if TS > 0:
        tsb = ts_ptr + ((tl.program_id(2) * tl.num_programs(1) + b) * tl.num_programs(0) + split) * 32
        tl.store(tsb, _gt33(split))
        tl.store(tsb + 1, _ck33(split))
    # DUAL (with HPP 1)""")
rep("""    k_scale = tl.load(ks_ptr)
    scale = qk_scale * k_scale  # log2 units, applied to q . k_fp8
""", """    k_scale = tl.load(ks_ptr)
    scale = qk_scale * k_scale  # log2 units, applied to q . k_fp8
    if TS > 0:
        tl.store(tsb + 2, _ck33(tl.max(q.to(tl.float32))))
""")
rep("""    if PRE == 0 and PREQ == 0:
        # (Computed here""", """    if TS > 0:
        tl.store(tsb + 3, _ck33(tl.max(m_i)))
    if PRE == 0 and PREQ == 0:
        # (Computed here""")
rep("""    FXD: tl.constexpr = FIX > 0
""", """    FXD: tl.constexpr = FIX > 0
    if TS > 0:
        tl.store(tsb + 4, _ck33(tl.max(m_i)))
""")
rep("""        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              False, PAGE, ROW_W, D, WIN, QK8, DOT, FIXED=FXD, CL=SEQ_IN)
""", """        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              False, PAGE, ROW_W, D, WIN, QK8, DOT, FIXED=FXD, CL=SEQ_IN)
        if TS > 1:
            tl.store(tsb + 10 + tl.minimum((start - lo) // BLOCK_N, 20), _ck33(tl.sum(l_i)))
""")
rep("""                              True, PAGE, ROW_W, D, WIN, QK8, DOT, MIF, need, FIXED=FXD, CL=SEQ_IN)
""", """                              True, PAGE, ROW_W, D, WIN, QK8, DOT, MIF, need, FIXED=FXD, CL=SEQ_IN)
        if TS > 1:
            tl.store(tsb + 10 + tl.minimum((start - lo) // BLOCK_N, 20), _ck33(tl.sum(l_i)))
""")
rep("""    if PF:
        m_i += tl.where(pf == -7, 1.0, 0.0) * 0.0
""", """    if TS > 0:
        tl.store(tsb + 5, _ck33(tl.sum(l_i)))
        tl.store(tsb + 9, ((hi - lo) // BLOCK_N).to(tl.int64))
    if PF:
        m_i += tl.where(pf == -7, 1.0, 0.0) * 0.0
""")
rep("""        _finish(split, b, j, m1, l1, acc1, hp0 + 1, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S, SEP)

    if FUSE:""", """        _finish(split, b, j, m1, l1, acc1, hp0 + 1, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S, SEP)
    if TS > 0:
        tl.store(tsb + 6, _ck33(split))
        tl.store(tsb + 7, _gt33(split))

    if FUSE:""")
# _combine
rep("""             MAX_PAGES: tl.constexpr = 1, PAGE: tl.constexpr = 1, CHUNK: tl.constexpr = 1):
    tok = tl.program_id(0)
    head = tl.program_id(1)
""", """             MAX_PAGES: tl.constexpr = 1, PAGE: tl.constexpr = 1, CHUNK: tl.constexpr = 1, TS: tl.constexpr = 0,
             ts_ptr=None):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    if TS > 0:
        tsc = ts_ptr + 65536 + (head * tl.num_programs(0) + tok) * 4
        tl.store(tsc, _gt33(tok))
""")
rep("""    if PDL:
        # Launched early (programmatic dependent launch): wait here for _attn_split's partials.
        gdc_wait()
""", """    if PDL:
        # Launched early (programmatic dependent launch): wait here for _attn_split's partials.
        gdc_wait()
    if TS > 0:
        tl.store(tsc + 1, _gt33(tok))
""")
rep("""    o = acc / denom * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    tl.store(out_ptr + tok * HQ * D + head * D + d, (o * tl.sigmoid(gate)).to(tl.bfloat16))
""", """    o = acc / denom * tl.load(vs_ptr)
    if TS > 0:
        tl.store(tsc + 2, _gt33(tl.sum(acc)))
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    tl.store(out_ptr + tok * HQ * D + head * D + d, (o * tl.sigmoid(gate)).to(tl.bfloat16))
    if TS > 0:
        tl.store(tsc + 3, _gt33(tok))
""")
# launches
rep("""        FIX=fix, FDELTA=plan.get("fdelta", 14.0),
""", """        FIX=fix, FDELTA=plan.get("fdelta", 14.0), TS=_TS33, ts_ptr=_TSB33.get(device),
""")
rep("""        PAGE=cfg["page_size"], CHUNK=chunk, launch_pdl=USE_PDL)
    return out
""", """        PAGE=cfg["page_size"], CHUNK=chunk, launch_pdl=USE_PDL, TS=_TS33, ts_ptr=_TSB33.get(device))
    return out
""")
# helpers before _attn_body's first use: put them after the gdc import
rep("from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait\n",
    """from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@triton.jit
def _ck33(x):
    return tl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,r", [x.to(tl.int32, bitcast=False)],
                                     dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _gt33(x):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x.to(tl.int32, bitcast=False)],
                                     dtype=tl.int64, is_pure=False, pack=1)


_TS33 = 0
_TSB33 = {}
""")
kf = "\ndef kernel_fn(data: Inputs) -> torch.Tensor:\n"
rep(kf, "\ndef _kfn(data: Inputs) -> torch.Tensor:\n")
src += open(__file__.replace("mktl.py", "tl_tail.py")).read()
open(sys.argv[2], "w").write(src)
print("wrote", sys.argv[2])
