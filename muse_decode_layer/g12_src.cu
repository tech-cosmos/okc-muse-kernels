// r-proj e71d015 muse_bf16_proj/dev/gemv12.cu (proj agent's gemv12, m=8 tensor-core streaming GEMM), copied verbatim
// gemv12: tensor-core streaming GEMM for m = 8..32 (out[m, n] = x[m, k] @ W[n, k]^T, BF16 in, FP32 accumulate).
// mma.sync m16n8k16 with x as A (tokens) and W as B. W is repacked in prepare() into 512-B atoms = 8 rows x 32 k,
// lane l holding row l/4, k (l%4)*8..+8 (one uint4 = the B fragments of two k16 mmas; x's A fragments use the same
// k permutation); layout [ks][rg][KC/32 atoms][lane] (ks = k-chunk of KC = 1024*KCU, rg = 8-row group).
// Block b = (ks = b / bk, a run of NG rgs); it is NG groups of WG warps, group gi streams rg0+gi's KC/32 atoms
// (warp wi takes atoms wi, wi+WG, ...; PF in flight in registers). x[:, chunk] is staged in smem first (cp.async;
// it must not queue behind the W stream). Each warp's FP32 C fragments go to its own smem slot; WG slots summed.
// One k-chunk: BF16 written directly. Else FP32 partials [S][m][n] in a workspace, a per-rg counter, and the last
// of the S blocks of a rg sums them in FP32 and writes BF16 (counters re-zeroed).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g12 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int M, int KCU, int WG, int PF>
__global__ void __launch_bounds__(1024, 1) gemm(const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                int* __restrict__ cnt, int n, int nrg_all, int k, int bk) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16, XS = KC * 2 + 64;  // x row stride in smem (bytes), padded
    constexpr int NT = M < 16 ? M : 16, NA = KC / 32 / WG;               // atoms per warp
    static_assert(NA % PF == 0, "PF must divide the atoms per warp");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x[M][KC] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [warps][TT][32 lanes] C fragments
    __shared__ int flag[64];
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x / bk, rg0 = (blockIdx.x % bk) * NG, rg = rg0 + gi;
    for (int i = t; i < M * (KC / 8); i += blockDim.x) {
        const int r = i / (KC / 8), q = i % (KC / 8);
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + q * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"((const uint4*)(x + (long)r * k + (long)ks * KC) + q));
    }
    asm volatile("cp.async.commit_group;");
    const uint4* wb = W + (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wb + j*WG*32
    uint4 w[PF];
#ifndef G12_LATE_W
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
#endif
    asm volatile("cp.async.wait_all;");
    __syncthreads();
#ifdef G12_LATE_W
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
#endif
    float acc[TT][4];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
    for (int a = 0; a < NA; a += PF) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                uint4 xa = *(const uint4*)(xs + (tt * 16 + g) * XS + kb * 2);
                uint4 xb = make_uint4(0, 0, 0, 0);
                if (NT > 8) xb = *(const uint4*)(xs + (tt * 16 + g + 8) * XS + kb * 2);
                mma(acc[tt], xa.x, xb.x, xa.y, xb.y, w[p].x, w[p].y);
                mma(acc[tt], xa.z, xb.z, xa.w, xb.w, w[p].z, w[p].w);
            }
            if (a + p + PF < NA) w[p] = ldw(wb + (a + p + PF) * WG * 32);
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
        return s;
    };
    if (S == 1) {
        for (int i = t; i < ne; i += blockDim.x) {
            const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(total(i));
        }
        return;
    }
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        __stcg(ws + ((long)ks * M + tok) * n + (rg0 + rgl) * 8 + row8, total(i));
    }
    __threadfence();
    __syncthreads();
    if (t < NG) {
        const int last = atomicAdd(cnt + rg0 + t, 1) == S - 1;
        flag[t] = last;
        if (last) cnt[rg0 + t] = 0;
    }
    __syncthreads();
    __threadfence();
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        if (flag[rgl]) {
            float s = 0.f;
            for (int q = 0; q < S; ++q) s += __ldcg(ws + ((long)q * M + tok) * n + (rg0 + rgl) * 8 + row8);
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(s);
        }
    }
}
}  // namespace g12
// nrg_all: 8-row groups of the packed weight; bk: blocks per k-chunk; each block takes n/8/bk rgs with WG warps each.
void g12_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t nrg_all,
              int64_t kcu, int64_t wg, int64_t pf, int64_t bk) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int S = k / (1024 * kcu), ng = n / 8 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 64) + threads / 32 * ((m + 15) / 16) * 512;
    TORCH_CHECK(n / 8 % bk == 0 && threads <= 1024 && ng <= 64 && sm <= 227 * 1024 && (S == 1 || m * n * S <= 8 * 32 * 8192), "g12: shape");
#define G12_CASE(M_, U_, G_, P_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_) { \
        cudaFuncSetAttribute(g12::gemm<M_, U_, G_, P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g12::gemm<M_, U_, G_, P_><<<S * bk, threads, sm, s>>>((const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk); return; }
    G12_CONFIGS
    TORCH_CHECK(false, "g12: no config");
}
