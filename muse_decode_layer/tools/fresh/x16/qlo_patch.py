"""Patch a submission/attn_src file: two-term E4M3 queries (q = e4m3(q) + e4m3(q - e4m3(q))) in the nvs kernels.
python tools/fresh/x16/qlo_patch.py <in> <out> [srcs=NVSCL,NVS]  (QK^T gets 2 more k32 wgmma per tile on the lo plane)"""
import sys
src, dst = sys.argv[1], sys.argv[2]
which = (sys.argv[3] if len(sys.argv) > 3 else "NVSCL,NVS").split(",")
t = open(src).read()

def patch_block(b):
    n0 = len(b)
    a = "    alignas(1024) uint8_t q[2][MR * HD];       // per head FP8 queries [row][d], SW64\n"
    assert b.count(a) == 1
    b = b.replace(a, "    alignas(1024) uint8_t q[2][2 * MR * HD];   // per head FP8 queries [row][d], SW64; hi plane, then lo\n")
    a = """    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();"""
    assert b.count(a) == 1
    b = b.replace(a, """    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    qk_k32(s, mdesc(qa + MR * HD, 512, 2), mdesc(ka, 1024, 1), 1);
    qk_k32(s, mdesc(qa + MR * HD + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();""")
    a = """        uint8_t* qd = &sm.q[h][0];
        *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32 + 16)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
"""
    assert b.count(a) == 1, b.count(a)
    b = b.replace(a, a + """        uint32_t pl[8];  // the lo plane: e4m3(x - e4m3(x))
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t f01, f23;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f01) : "h"((unsigned short)(pk[j] & 0xffff)));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f23) : "h"((unsigned short)(pk[j] >> 16)));
            const float2 h01 = __half22float2(*reinterpret_cast<const __half2*>(&f01));
            const float2 h23 = __half22float2(*reinterpret_cast<const __half2*>(&f23));
            pl[j] = pack_e4(x[4 * j] - h01.x, x[4 * j + 1] - h01.y, x[4 * j + 2] - h23.x, x[4 * j + 3] - h23.y);
        }
        *reinterpret_cast<uint4*>(qd + MR * HD + swz64(r * 64 + half * 32)) = make_uint4(pl[0], pl[1], pl[2], pl[3]);
        *reinterpret_cast<uint4*>(qd + MR * HD + swz64(r * 64 + half * 32 + 16)) = make_uint4(pl[4], pl[5], pl[6], pl[7]);
""")
    if "#include <cuda_fp16.h>" not in b:
        b = "#include <cuda_fp16.h>\n" + b
    return b

for name in [w for w in which if w not in ("GW", "GWBAL")]:
    key = f"_{name}_SRC = r'''"
    i = t.index(key) + len(key)
    j = t.index("'''", i)
    t = t[:i] + patch_block(t[i:j]) + t[j:]
open(dst, "w").write(t)
print("patched", which, len(t))

# ---- Gluon _gw_verify BAL path (v16/v32 full): the lo plane as a second [M, D] E4M3 buffer per consumer ----
if "GW" in which or "GWBAL" in which:
    t = open(dst).read()
    i0 = t.index("    def _gw_consumer_bal(")
    i1 = t.index("    @gluon.jit", i0)
    f = t[i0:i1]
    f = f.replace("def _gw_consumer_bal(qkv_ptr, q_s, k_ring,", "def _gw_consumer_bal(qkv_ptr, q_s, ql_s, k_ring,")
    a = "            q_s.store((x * gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv))\n"
    assert f.count(a) == 1
    f = f.replace(a, """            xn = x * gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]
            x8 = xn.to(gl.float8e4nv)
            q_s.store(x8)
            ql_s.store((xn - x8.to(gl.float32)).to(gl.float8e4nv))
""")
    a = """s_tok = warpgroup_mma(q_s, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
"""
    parts = f.split(a)
    assert len(parts) == 4, len(parts)
    out = parts[0]
    for p in parts[1:]:
        ind = out[out.rindex("\n") + 1:]
        line2, rest = p.split("\n", 1)
        out += a + line2 + "\n" + ind + \
            "s_tok = warpgroup_mma(ql_s, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)), s_tok, is_async=True)\n" + rest
    f = out
    t = t[:i0] + f + t[i1:]
    a = """            q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
            q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
"""
    assert t.count(a) == 1
    t = t.replace(a, a + """            q0l = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
            q1l = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
""")
    for k in ("0", "1"):
        a = f"(_gw_consumer_bal, (qkv_ptr, q{k}, k_ring,"
        assert t.count(a) == 1, a
        t = t.replace(a, f"(_gw_consumer_bal, (qkv_ptr, q{k}, q{k}l, k_ring,")
    open(dst, "w").write(t)
    print("patched GW BAL", len(t))
