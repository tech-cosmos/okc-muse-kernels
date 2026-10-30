"""n8s2: nvscl -> cluster size 2 or 4 (S=4 at verify 16: 4 splits merged over DSMEM, no _combine_hb) + O_PF port.
Edits attn_src.py's _NVSCL_SRC in place (run once)."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "attn_src.py"
s = open(p).read()
a = s.index("_NVSCL_SRC = r'''")
b = s.index("'''", a + 20)
src = s[a:b]


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (old[:80], src.count(old))
    src = src.replace(old, new)


rep("    uint64_t clbar;                            // cl: rank 1's 128 remote arrivals after that push\n",
    "    uint64_t clbar;                            // cl: rank 1's 128 remote arrivals after that push\n"
    "    alignas(128) float xo3[96 * 36];           // cl 4: the 3 peers' o / m / l for this rank's warp (slot = peer)\n")
# owner predicate
rep("        if ((w >> 1) == (int)rank) {  // rows this rank finishes",
    "        if ((cl == 4 ? w : (w >> 1)) == (int)rank) {  // rows this rank finishes")
rep("""    if (cl) {
        // 2-CTA cluster = the 2 splits of (request, H).""",
    """    if (cl == 4) {
        // n8s2: 4-CTA cluster = the 4 splits of (request, H). Rank r finishes warp r's 16 rows; each other warp pushes
        // its merged o / m / l (4.6 KB) into rank w's xo3 slot (one bulk DSMEM copy per warp, on rank w's clbar).
        if (w != (int)rank) {
            float4* xl = reinterpret_cast<float4*>(&sm.xo2[tid * 36]);
#pragma unroll
            for (int j = 0; j < 8; ++j) xl[j] = make_float4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
            xl[8] = make_float4(m[0], m[1], l[0], l[1]);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // the peers' clbar inits are visible
            __syncwarp();
            if (lane == 0) {
                const uint32_t src = smem_u32(&sm.xo2[tid * 36]);
                const int slot = (int)rank < w ? (int)rank : (int)rank - 1;
                uint32_t dst = smem_u32(&sm.xo3[slot * 32 * 36]), bar = smem_u32(&sm.clbar);
                asm volatile("mapa.shared::cluster.u32 %0, %0, %1;" : "+r"(dst) : "r"(w));
                asm volatile("mapa.shared::cluster.u32 %0, %0, %1;" : "+r"(bar) : "r"(w));
                asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             ::"r"(dst), "r"(src), "n"(32 * 36 * 4), "r"(bar) : "memory");
                asm volatile("cp.async.bulk.commit_group;\\ncp.async.bulk.wait_group.read 0;" ::: "memory");
            }
            return;
        }
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 g = __bfloat1622float2(gr[t][j]);
                gt[t][j] = make_float2(__fdividef(1.f, 1.f + __expf(-g.x)), __fdividef(1.f, 1.f + __expf(-g.y)));
            }
        fence_regs(*reinterpret_cast<float(*)[32]>(&gt[0][0]));  // keep the sigmoids before the wait
        asm volatile("{\\n.reg .pred p;\\nW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\\n@!p bra W;\\n}\\n"
                     ::"r"(smem_u32(&sm.clbar)) : "memory");
#pragma unroll
        for (int k = 0; k < 3; ++k) {  // FP32 online-softmax merge of the 3 peers' partials
            const float4* xo = reinterpret_cast<const float4*>(&sm.xo3[(k * 32 + lane) * 36]);
            const float4 x = xo[8];
            const float mn0 = fmaxf(m[0], x.x), mn1 = fmaxf(m[1], x.y);
            const float a0 = ex2(m[0] - mn0), b0 = ex2(x.x - mn0), a1 = ex2(m[1] - mn1), b1 = ex2(x.y - mn1);
            m[0] = mn0; m[1] = mn1;
            l[0] = l[0] * a0 + x.z * b0; l[1] = l[1] * a1 + x.w * b1;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 y = xo[j];
                o[4 * j + 0] = o[4 * j + 0] * a0 + y.x * b0; o[4 * j + 1] = o[4 * j + 1] * a0 + y.y * b0;
                o[4 * j + 2] = o[4 * j + 2] * a1 + y.z * b1; o[4 * j + 3] = o[4 * j + 3] * a1 + y.w * b1;
            }
        }
        const float f0 = __fdividef(vsv, l[0]), f1 = __fdividef(vsv, l[1]);
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[4 * j + 0] *= f0; o[4 * j + 1] *= f0; o[4 * j + 2] *= f1; o[4 * j + 3] *= f1;
            s0 += o[4 * j] * o[4 * j] + o[4 * j + 1] * o[4 * j + 1];
            s1 += o[4 * j + 2] * o[4 * j + 2] + o[4 * j + 3] * o[4 * j + 3];
        }
        s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
        s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
        const float r0 = rsqrtf(s0 / HD + 1e-5f), r1 = rsqrtf(s1 / HD + 1e-5f);
        const int ra = w * 16 + (lane >> 2);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            __nv_bfloat16* dsto = out + (long long)(b * 8 + (ra >> 3) + t) * (16 * HD) + (H * 8 + (ra & 7)) * HD +
                                  (lane & 3) * 2;
            const float rr = t ? r1 : r0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 g = gt[t][j];
                const __nv_bfloat162 v = __floats2bfloat162_rn(o[4 * j + 2 * t] * rr * g.x, o[4 * j + 2 * t + 1] * rr * g.y);
                *reinterpret_cast<__nv_bfloat162*>(dsto + j * 8) = v;
            }
        }
        return;
    }
    if (cl) {
        // 2-CTA cluster = the 2 splits of (request, H).""")
# clbar init: expected bytes depend on the cluster size
rep('''            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&sm.clbar)), "n"(64 * 36 * 4) : "memory");''',
    '''            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&sm.clbar)), "r"(cl == 4 ? 3 * 32 * 36 * 4 : 64 * 36 * 4) : "memory");''')
# O_PF port
rep("int qkv_w, int win, int cl, __nv_bfloat16* __restrict__ out) {",
    "int qkv_w, int win, int cl, __nv_bfloat16* __restrict__ out,\n"
    "                                              const uint8_t* __restrict__ pf, long long pfb) {")
rep("""        producer<NS, LAG>(sm, kc, vc, pt, r0, r1, H, maxp);
""", """        producer<NS, LAG>(sm, kc, vc, pt, r0, r1, H, maxp);
        if (pfb > 0 && tid < 32) {  // layer s34 (O_PF): after this CTA's last KV load is issued, L2-prefetch its share of the next GEMM's weight
            const long long per = ((pfb / gridDim.x) + 4095) & ~4095LL, end = min(pfb, per * (blockIdx.x + 1));
            for (long long o = per * blockIdx.x + tid * 16384LL; o < end; o += 32 * 16384LL)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pf + o), "r"((int)min(16384LL, end - o)) : "memory");
        }
""")
rep("         torch::Tensor out, int64_t cl) {", "         torch::Tensor out, int64_t cl, int64_t pf_ptr, int64_t pf_bytes) {")
rep("at[0].val.clusterDim.x = cl ? 2 : 1;", "at[0].val.clusterDim.x = cl ? (unsigned)cl : 1;")
rep("(int)pt.size(1), (int)qkv.size(-1), (int)win, (int)cl, (__nv_bfloat16*)out.data_ptr());",
    "(int)pt.size(1), (int)qkv.size(-1), (int)win, (int)cl, (__nv_bfloat16*)out.data_ptr(),\n"
    "        (const uint8_t*)pf_ptr, (long long)pf_bytes);")
s = s[:a] + src + s[b:]
old = '''"int64_t, torch::Tensor, int64_t);",
        cuda_sources=_NVSCL_SRC'''
assert s.count(old) == 1
s = s.replace(old, '''"int64_t, torch::Tensor, int64_t, int64_t, int64_t);",
        cuda_sources=_NVSCL_SRC''')
open(p, "w").write(s)
print("ok")
