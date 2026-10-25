// gemv15: gemv14 (compressed-weight tensor-core streaming GEMM) with the split-K reduction over a thread-block
// cluster: the S = k / KC blocks of a row run (one per k-chunk) form a cluster, each block writes its outputs' FP32
// partials (WG warp slots + escape fixups, scaled domain) to smem, cluster.sync, block ks sums slice ks of the outputs
// over the S blocks through distributed shared memory and writes BF16. No global partials, counters or workspace.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace g15 {
constexpr int ESC = 8;
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld2(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned ld1(const unsigned* p) {
    unsigned v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    return (s & 0x07800780u) | 0x38003800u;
}
__device__ __forceinline__ uint4 dec8(uint2 s, unsigned nw) {
    uint4 r;
    r.x = (prmt(s.x, 0x9180u) & 0x807f807fu) | expo<0>(nw);
    r.y = (prmt(s.x, 0xB3A2u) & 0x807f807fu) | expo<1>(nw);
    r.z = (prmt(s.y, 0x9180u) & 0x807f807fu) | expo<2>(nw);
    r.w = (prmt(s.y, 0xB3A2u) & 0x807f807fu) | expo<3>(nw);
    return r;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int M, int KCU, int WG, int PF, int XP, int XR>
__global__ void __launch_bounds__(1024, 1) gemm(const uint2* __restrict__ WS, const unsigned* __restrict__ WN,
                                                const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, int n, int nrg_all, int k) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16;
    constexpr int NT = M < 16 ? M : 16, NA = KC / 32 / WG;               // atoms per warp
    static_assert(NA % (PF * XP) == 0 && XR == XP && XP <= 4, "x pieces: fully staged");
    constexpr int PA = NA / XP, PB = KC / XP * 2;  // atoms per x piece (per warp), bytes per x piece row
    constexpr int XS = XR * PB + 64;                // x ring row stride in smem (bytes), padded: XR pieces of PB
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x ring [M][XR pieces] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [warps][TT][32 lanes] C fragments
    float2* hs = (float2*)(smem + M * XS + blockDim.x / 32 * TT * 512);  // [NG * 8 rows][ESC] escape headers
    float* part = (float*)(hs + blockDim.x / (32 * WG) * 8 * ESC);                           // [NG * M * 8] this block's partials
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x % S, rg0 = (blockIdx.x / S) * NG, rg = rg0 + gi;
    // x[:, chunk] in XP pieces along k through a ring of XR piece buffers (cp.async, one commit group per piece),
    // the first XR issued before any W load: the block starts on piece 0 while the rest lands.
    auto stage = [&](int p) {
        for (int i = t; i < M * (PB / 16); i += blockDim.x) {
            const int r = i / (PB / 16), q = i % (PB / 16);
            const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + (p % XR) * PB + q * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa),
                         "l"((const uint4*)(x + (long)r * k + (long)ks * KC + p * (KC / XP)) + q));
        }
    };
    // escape headers ride in x piece 0's commit group
    for (int i = t; i < NG * 8 * ESC / 2; i += blockDim.x) {
        const unsigned sa = (unsigned)__cvta_generic_to_shared(hs + 2 * i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(H + (long)rg0 * 8 * ESC + 2 * i));
    }
#pragma unroll
    for (int p = 0; p < XR; ++p) { stage(p); asm volatile("cp.async.commit_group;"); }
    const long wo = (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wo + j*WG*32
    const uint2* wbs = WS + wo;
    const unsigned* wbn = WN + wo;
    uint2 ws_[PF];
    unsigned wn_[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) { ws_[p] = ld2(wbs + p * WG * 32); wn_[p] = ld1(wbn + p * WG * 32); }
    float acc[TT][4];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
    for (int a = 0; a < NA; a += PF) {
        if (a % PA == 0) {  // piece pc = a / PA must have landed (groups complete in order)
            const int pc = a / PA;
            if (XR == XP) {     // everything staged up front
                const int left = XP - 1 - pc;
                if (left >= 3) asm volatile("cp.async.wait_group 3;");
                else if (left == 2) asm volatile("cp.async.wait_group 2;");
                else if (left == 1) asm volatile("cp.async.wait_group 1;");
                else asm volatile("cp.async.wait_group 0;");
                __syncthreads();
            } else {            // ring: committed XR + pc - 1 groups (one per boundary), piece pc is group pc
                if (pc == 0) {
                    if (XR == 4) asm volatile("cp.async.wait_group 3;");
                    else if (XR == 3) asm volatile("cp.async.wait_group 2;");
                    else asm volatile("cp.async.wait_group 1;");
                } else {
                    if (XR == 4) asm volatile("cp.async.wait_group 2;");
                    else if (XR == 3) asm volatile("cp.async.wait_group 1;");
                    else asm volatile("cp.async.wait_group 0;");
                }
                __syncthreads();  // piece pc visible to all; everyone is done with piece pc - 1
                if (pc > 0 && pc - 1 + XR < XP) stage(pc - 1 + XR);
                if (pc > 0) asm volatile("cp.async.commit_group;");
            }
        }
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint4 wv = dec8(ws_[p], wn_[p]);
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                const int kx = (kb - (a / PA) * (KC / XP) + ((a / PA) % XR) * (KC / XP)) * 2;
                uint4 xa = *(const uint4*)(xs + (tt * 16 + g) * XS + kx);
                uint4 xb = make_uint4(0, 0, 0, 0);
                if (NT > 8) xb = *(const uint4*)(xs + (tt * 16 + g + 8) * XS + kx);
                mma(acc[tt], xa.x, xb.x, xa.y, xb.y, wv.x, wv.y);
                mma(acc[tt], xa.z, xb.z, xa.w, xb.w, wv.z, wv.w);
            }
            if (a + p + PF < NA) { ws_[p] = ld2(wbs + (a + p + PF) * WG * 32); wn_[p] = ld1(wbn + (a + p + PF) * WG * 32); }
        }
    }
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) accp[(wp * TT + tt) * 32 + lane] = make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]);
    __syncthreads();
    // outputs i = ((rgl * M + tok) * 8 + row8): sum the WG warp slots of group rgl
    const int ne = NG * M * 8;
    auto total = [&](int i) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        const int tt = tok >> 4, h = (tok >> 3) & 1, l = (tok & 7) * 4 + (row8 >> 1), e = h * 2 + (row8 & 1);
        const float* base = (const float*)accp + ((rgl * WG) * TT + tt) * 128 + l * 4 + e;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[q * TT * 128];
        const float2* he = hs + (rgl * 8 + row8) * ESC;
#pragma unroll
        for (int e = 0; e < ESC; ++e) {
            const int col = __float_as_int(he[e].x) - ks * KC;
            if (col >= 0 && col < KC) s = fmaf(__bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2)), he[e].y, s);
        }
        return s;
    };
    for (int i = t; i < ne; i += blockDim.x) part[i] = total(i);
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    const int e0 = ks * ne / S, e1 = (ks + 1) * ne / S;
    for (int i = e0 + t; i < e1; i += blockDim.x) {
        float v = 0.f;
        for (int q = 0; q < S; ++q) v += cl.map_shared_rank(part, (q + ks) % S)[i];
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(v * scale);
    }
    cl.sync();  // peers may still read this block's partials
}
}  // namespace g15
// nrg_all: 8-row groups of the packed weight; bk: row runs (clusters); each run = n/8/bk rgs with WG warps each, one
// block per k-chunk (cluster of S = k / KC blocks).
void g15_gemm(torch::Tensor ws_, torch::Tensor wn, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              int64_t nrg_all, int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int S = k / (1024 * kcu), ng = n / 8 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 64) + threads / 32 * ((m + 15) / 16) * 512 + ng * 8 * g15::ESC * 8 + ng * m * 8 * 4;
    TORCH_CHECK(n / 8 % bk == 0 && threads <= 1024 && sm <= 227 * 1024 && S <= 16, "g15: shape");
#define G15_CASE(M_, U_, G_, P_, X_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_ && xp == X_) { \
        auto kern = g15::gemm<M_, U_, G_, P_, X_, X_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        if (S > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(S * bk); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = sm; \
        cfg.stream = c10::cuda::getCurrentCUDAStream(); \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeClusterDimension; \
        attr[0].val.clusterDim.x = S; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const uint2*)ws_.data_ptr(), (const unsigned*)wn.data_ptr(), \
            (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), n, nrg_all, k); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G15_CONFIGS
    TORCH_CHECK(false, "g15: no config");
}
