# proj s33: g13p = gemv13 for the T=1 o with W loaded AND decoded (to packed BF16 pairs) before griddepcontrol.wait.
import sys
p = sys.argv[1]
s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)
SRC = r'''
// s33 (proj): g13p = gemv13 for k = 1024 as a PDL secondary: every chunk of the block (PF of them) is loaded AND decoded
// into packed BF16 pairs before griddepcontrol.wait, so after it only x, the FMAs, the reduction and the escapes remain
// (gemv13 after the wait was issue-bound on decode: ~2.3 us). Same sums in the same order as gemv13 (bitwise equal).
namespace g13p {
template <int P0>
__device__ __forceinline__ void pk4(unsigned sw, unsigned nw, unsigned* q) {
    q[0] = (g13::prmt(sw, 0x9180u) & 0x807f807fu) | g13::expo<P0>(nw);
    q[1] = (g13::prmt(sw, 0xB3A2u) & 0x807f807fu) | g13::expo<P0 + 1>(nw);
}
template <int U, int PF, int PQ>
__global__ void __launch_bounds__(256, 4) gemv(const uint4* __restrict__ S, const uint2* __restrict__ N,
                                               const float2* __restrict__ H, const __nv_bfloat16* __restrict__ x,
                                               __nv_bfloat16* __restrict__ out, int n, float scale) {
    constexpr int KK = 1024, CE = U * 4096, RPC = CE / KK, ESC = g13::ESC;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    const int nch = n / RPC, G = gridDim.x;
    __shared__ float red[PF + PQ][U * 8];
    __shared__ __align__(16) float xs[KK / 16 * 20];
    __shared__ float2 hs[(PF + PQ) * RPC * ESC];
    uint4 s[PF][U];
    uint2 nb[PF][U];
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        const int ch = blockIdx.x + p * G;
        if (ch < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const long i = (long)ch * (CE / 16) + u * 256 + t;
                s[p][u] = g13::ld4(S + i);
                nb[p][u] = g13::ld2(N + i);
            }
        }
    }
    for (int i = t; i < (PF + PQ) * RPC * ESC; i += 256) {
        const int ch = blockIdx.x + (i / (RPC * ESC)) * G;
        if (ch < nch) hs[i] = __ldg(H + (long)ch * RPC * ESC + i % (RPC * ESC));
    }
    unsigned q[PF][U][8];
#pragma unroll
    for (int p = 0; p < PF; ++p)
#pragma unroll
        for (int u = 0; u < U; ++u) {
            pk4<0>(s[p][u].x, nb[p][u].x, q[p][u]); pk4<2>(s[p][u].y, nb[p][u].x, q[p][u] + 2);
            pk4<0>(s[p][u].z, nb[p][u].y, q[p][u] + 4); pk4<2>(s[p][u].w, nb[p][u].y, q[p][u] + 6);
        }
    // ptxas sinks the decode below griddepcontrol.wait (it keeps the 24 raw registers live instead of 32 decoded):
    // a branch on all decoded words keeps it above (the store is harmless: red is written before it is read)
    unsigned xr = 0;
#pragma unroll
    for (int p = 0; p < PF; ++p)
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int w = 0; w < 8; ++w) xr ^= q[p][u][w];
    if (xr == 0x9e3779b9u + (unsigned)n) red[0][0] = 0.f;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // PQ more chunks per block: loaded right after the wait (beside x), decoded after it
    uint4 s2[PQ > 0 ? PQ : 1][U];
    uint2 nb2[PQ > 0 ? PQ : 1][U];
#pragma unroll
    for (int p = 0; p < PQ; ++p) {
        const int ch = blockIdx.x + (PF + p) * G;
        if (ch < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const long i = (long)ch * (CE / 16) + u * 256 + t;
                s2[p][u] = g13::ld4(S + i);
                nb2[p][u] = g13::ld2(N + i);
            }
        }
    }
    if (t < KK / 8) {
        const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(x) + t);
        const unsigned v[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) { const int kk = XI(8 * t + 2 * j); xs[kk] = __uint_as_float(v[j] << 16); xs[kk + 1] = __uint_as_float(v[j] & 0xffff0000u); }
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        if (blockIdx.x + p * G < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const float* xp = xs + XI((u * 4096 + t * 16) % KK);
                float xr[16];
#pragma unroll
                for (int i = 0; i < 16; i += 4) *reinterpret_cast<float4*>(xr + i) = *reinterpret_cast<const float4*>(xp + i);
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    a0 = fmaf(__uint_as_float(q[p][u][w] << 16), xr[2 * w], a0);
                    a1 = fmaf(__uint_as_float(q[p][u][w] & 0xffff0000u), xr[2 * w + 1], a1);
                }
                float v = a0 + a1;
#pragma unroll
                for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) red[p][u * 8 + wp] = v;
            }
        }
    }
#pragma unroll
    for (int p = 0; p < PQ; ++p) {
        if (blockIdx.x + (PF + p) * G < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                float v = g13::dot16(s2[p][u], nb2[p][u], xs + XI((u * 4096 + t * 16) % KK));
#pragma unroll
                for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) red[PF + p][u * 8 + wp] = v;
            }
        }
    }
    __syncthreads();
    if (t < (PF + PQ) * RPC) {
        const int p = t / RPC, r = t % RPC, ch = blockIdx.x + p * G;
        if (ch < nch) {
            float sum = 0.f;
#pragma unroll
            for (int u = 0; u < U; ++u)
#pragma unroll
                for (int w = 0; w < 8; ++w)
                    if ((u * 4096 + w * 512) / KK == r) sum += red[p][u * 8 + w];
            const float2* he = hs + (p * RPC + r) * ESC;
            for (int e = 0; e < ESC; ++e) {
                const float2 h = he[e];
                if (h.y == 0x1p-15f) break;
                const float xe = xs[XI(__float_as_int(h.x))];
                sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
            }
            out[(long)ch * RPC + r] = __float2bfloat16(sum * scale);
        }
    }
}
}  // namespace g13p

void g13p_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
               int64_t u, int64_t pf, int64_t pq, int64_t blocks, int64_t pdl) {
    const int k = x.size(1), n = out.size(1);
    TORCH_CHECK(x.size(0) == 1 && k == 1024 && (long)n * k <= (long)blocks * (pf + pq) * u * 4096, "g13p: shape");
    auto st = c10::cuda::getCurrentCUDAStream();
    cudaLaunchConfig_t lc = {}; lc.gridDim = blocks; lc.blockDim = 256; lc.dynamicSmemBytes = 0; lc.stream = st;
    cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0;
#define G13P_CASE(U_, P_, Q_) \
    if (u == U_ && pf == P_ && pq == Q_) { cudaLaunchKernelEx(&lc, g13p::gemv<U_, P_, Q_>, (const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), \
        (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }
    G13P_CASE(2, 2, 0) G13P_CASE(2, 1, 1) G13P_CASE(1, 2, 2) G13P_CASE(1, 1, 3) G13P_CASE(1, 3, 1)
    TORCH_CHECK(false, "g13p: no config");
}
'''
rep('G13_K = {"o": 1024}',
    'G13_K = {"o": 1024}\n'
    '# s33: g13p (W decoded before the wait) for that o: {sliding: (u, pf = chunks loaded+decoded before the wait, pq = after, blocks)}; {} = gemv13 above\n'
    'G13P = {True: (2, 1, 1, 512), False: (2, 1, 1, 512)}  # (u, pf, pq, blocks)\n'
    '_G13P_SRC = r\'\'\'' + SRC + '\'\'\'')
rep('            src = src + "\\n" + src13 + "\\n" + _R3_SRC',
    '            src = src + "\\n" + src13 + "\\n" + _G13P_SRC + "\\n" + _R3_SRC')
rep('int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\\n"\n',
    'int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\\n"\n'
    '                            "void g13p_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t, int64_t, int64_t);\\n"\n')
rep('            if x.shape[0] == 1:\n                pf = kv_pf',
    '            if o13 and sliding in G13P:\n'
    '                lib.g13p_gemv(pk[0], pk[1], pk[2], x.contiguous(), out, pk[3], *G13P[sliding],\n'
    '                              int(UNPOOL_PDL and key in GEMV_PDL_SECONDARY.get((1, sliding), ())))\n'
    '            elif x.shape[0] == 1:\n                pf = kv_pf')
open(p, "w").write(s)
print("ok")
s = open(p).read()
rep('functions=["g10_gemv", "g12_gemm", "g13_gemv", ', 'functions=["g10_gemv", "g12_gemm", "g13_gemv", "g13p_gemv", ')
open(p, "w").write(s)
print("ok2")
