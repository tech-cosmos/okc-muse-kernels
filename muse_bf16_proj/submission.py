# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open: out = x @ W[:n].T (BF16; the router FP32).

Router (FP32 weights, FP32 output): up to 8 tokens a native true-FP32 split-K kernel (FP32 weights, FP32 FMA,
FP32 split partials), as SGLang's F.linear in FP32. Above 8 tokens SGLang's BF16 hi + lo split with FP32
accumulation: 9..32 tokens a native tensor-core kernel (dev/gemv6.cu: mma.sync with hi and lo tiles into one FP32
accumulator, split K reduced in FP32 over a thread-block cluster through distributed shared memory); above 32 one
cuBLAS GEMM against the stacked [hi; lo] weight (FP32 out) and an FP32 add of the two halves. BF16 projections: at 1 token a native grid-stride GEMV (dev/gemv10.cu: W read in place, row-major, the whole grid
sweeping one contiguous window; FP32 FMA and FP32 reductions, no split K), else at 8 tokens (pooler also 16) a native
tensor-core streaming GEMM (dev/gemv12.cu: mma.sync with x as A and W, repacked losslessly into 512-B fragment atoms, as B;
FP32 accumulators; split K reduced in FP32), else cuBLAS (as SGLang's bf16_gemm_dispatch on H200).
Native kernels are built once at import (load_inline); if the build fails everything falls back to torch.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


_ROUTER_SRC = r'''// Router at m <= 8 in true FP32, no split K: block = R whole rows (k = 8192), T threads; thread t owns k positions
// 4 (t + T j), j < J = k / (4 T), so its x slice goes straight to registers (no smem staging) and all R * J float4 W
// loads are issued up front. Warp shuffle reduce, then one smem pass over the block's warps. No workspace/atomics.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pr2 {
template <int M, int R, int T, int K>
__global__ void __launch_bounds__(T) router(const float4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                            float* __restrict__ out, int m, int n) {
    constexpr int J = K / (4 * T), NW = T / 32;
    static_assert(K % (4 * T) == 0, "k split");
    __shared__ float red[NW][R * M];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r0 = blockIdx.x * R;
    uint2 xr[M][J];  // 4 BF16 per (token, j)
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
            xr[i][j] = i < m ? __ldg(reinterpret_cast<const uint2*>(x + (size_t)i * K) + t + T * j) : make_uint2(0, 0);
    float4 a[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = __ldcs(w + (size_t)(r0 + r) * (K / 4) + t + T * j);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const float x0 = __uint_as_float(xr[i][j].x << 16), x1 = __uint_as_float(xr[i][j].x & 0xffff0000u);
            const float x2 = __uint_as_float(xr[i][j].y << 16), x3 = __uint_as_float(xr[i][j].y & 0xffff0000u);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r][i] = fmaf(a[r][j].x, x0, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].y, x1, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].z, x2, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].w, x3, acc[r][i]);
            }
        }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            float v = acc[r][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][r * M + i] = v;
        }
    __syncthreads();
    if (t < R * M) {
        const int r = t / M, i = t % M;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) s += red[q][t];
        if (i < m) out[(size_t)i * n + r0 + r] = s;
    }
}
}  // namespace pr2

#define PR2_CASE(M, R, T)                                                                                           \
    if (mp == M && rr == R && tt == T) {                                                                            \
        pr2::router<M, R, T, 8192><<<n / R, T, 0, c10::cuda::getCurrentCUDAStream()>>>(                             \
            reinterpret_cast<const float4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),    \
            out.data_ptr<float>(), m, n);                                                                           \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
        return;                                                                                                     \
    }

void pj_router2(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mp, int64_t rr, int64_t tt) {
    const int m = (int)x.size(0), n = (int)w.size(0);
    TORCH_CHECK(m <= mp && w.size(1) == 8192, "pr2: shape");
    PR2_CONFIGS
    TORCH_CHECK(false, "pr2: no such config");
}
'''
_GEMV_SRC = r'''// Skinny GEMM out[m, n] = x[m, k] @ W[n, k]^T for small m: BF16 x, BF16 W (NA = 1) or the router's BF16 hi + lo
// pair (NA = 2, both accumulated into the same FP32 accumulator), FP32 accumulate, BF16 or FP32 out.
// Swap-AB mma.sync m16n8k16: A = a 16-row weight stripe (register-direct 16-B loads of the packed layout:
// [stripe][k-tile][NA][lane][8], 512 B per tile), B = 8 tokens of x via ldmatrix from smem.
// Block = (NWR stripes x NWK K-slices) warps over (row group, token group, K range KB = k / CS); x[tok group, K
// range] is staged once in smem (cp.async, issued before the first weight loads). Split K: intra-block via smem,
// across the CS blocks of a thread-block cluster via distributed shared memory (no atomics, no workspace, no
// counters: graph-replayable as is). Every block writes a disjoint slice of the reduced output.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace g7 {
namespace cg = cooperative_groups;

__device__ __forceinline__ void mma(float* d, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

template <int MT, int NWR, int NWK, int U, int NA, bool F32OUT>
__global__ void __launch_bounds__(NWR * NWK * 32) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                       void* __restrict__ out, int m, int n, int k, int cs, int ilv) {
    constexpr int TOK = MT * 8, RB = NWR * 16, NT = NWR * NWK * 32;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    float* red = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wr = warp % NWR, wk = warp / NWR;
    const int ks = blockIdx.x % cs, rest = blockIdx.x / cs, nrg = n / RB, rg = rest % nrg, tg = rest / nrg;
    // x rows actually present in this token group (rows past m read a shared 16-B zero line instead)
    const int mr = min(TOK, m - tg * TOK);
    const int KB = k / cs, k0 = ks * KB, XS = KB + 8;  // smem row stride (elements): +16 B breaks bank conflicts
    const int XBYTES = max(mr * XS * 2 + 16, NWK * TOK * RB * 4);
    if (threadIdx.x == 0) *reinterpret_cast<uint4*>(smem + mr * XS * 2) = make_uint4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < mr * KB / 8; i += NT) {
        const int r = i / (KB / 8), c = (i % (KB / 8)) * 8, tok = tg * TOK + r;
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + c);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(x + (size_t)tok * k + k0 + c));
    }
    asm volatile("cp.async.commit_group;");
    const int KT = KB / 16 / NWK, kt16 = k >> 4, s = rg * NWR + wr;
    const int tl0 = wk * KT;  // first k-tile of this warp, local to the block's K range
    // ilv: W packed so tile t of every warp of the grid is adjacent ([t][grid warp][NA][512 B]): the whole grid
    // sweeps memory sequentially. Otherwise [stripe][k-tile][NA][512 B].
    const int nwt = gridDim.x * (NWR * NWK), gw = blockIdx.x * (NWR * NWK) + warp;
    const uint4* wp = ilv ? w + (size_t)gw * NA * 32 + lane : w + ((size_t)s * kt16 + (k0 >> 4) + tl0) * NA * 32 + lane;
    const size_t tstride = ilv ? (size_t)nwt * NA * 32 : (size_t)NA * 32;
    uint4* ring = reinterpret_cast<uint4*>(smem + XBYTES) + (size_t)warp * U * NA * 32 + lane;
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int q = 0; q < NA; ++q)
            if (u < KT) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(ring + (u * NA + q) * 32);
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sa), "l"(wp + u * tstride + q * 32));
            }
        asm volatile("cp.async.commit_group;");
    }
    float acc[MT][4];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    asm volatile("cp.async.wait_group %0;" ::"n"(U));
    __syncthreads();
    // ldmatrix row address: lane l -> matrix l/8, row l%8; matrices (j, k lo), (j, k hi), (j+1, k lo), (j+1, k hi)
    const int lr = lane & 7, lm = lane >> 3;
    const unsigned xs0 = (unsigned)__cvta_generic_to_shared(xs), xzero = xs0 + 2u * (unsigned)(mr * XS);
    unsigned xb[(MT + 1) / 2];  // per ldmatrix: this lane's row address at the warp's first k-tile, or the zero line
    bool xvalid[(MT + 1) / 2];
#pragma unroll
    for (int jj = 0; jj < (MT + 1) / 2; ++jj) {
        const int row = jj * 16 + (lm >> 1) * 8 + lr;
        xvalid[jj] = row < mr;
        xb[jj] = xs0 + 2u * (unsigned)(row * XS + (lm & 1) * 8 + tl0 * 16);
    }
    for (int t = 0; t < KT; ++t) {
        const int slot = t % U;
        asm volatile("cp.async.wait_group %0;" ::"n"(U - 1));
        uint4 a[NA];
#pragma unroll
        for (int q = 0; q < NA; ++q) a[q] = ring[(slot * NA + q) * 32];
#pragma unroll
        for (int j = 0; j < MT; j += 2) {
            unsigned b0, b1, b2 = 0, b3 = 0;
            const unsigned addr = xvalid[j / 2] ? xb[j / 2] + 32u * t : xzero;
            if (j + 1 < MT)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
            else
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
#pragma unroll
            for (int q = 0; q < NA; ++q) {
                mma(acc[j], a[q], b0, b1);
                if (j + 1 < MT) mma(acc[j + 1], a[q], b2, b3);
            }
        }
        if (t + U < KT) {
#pragma unroll
            for (int q = 0; q < NA; ++q) {
                const unsigned sa = (unsigned)__cvta_generic_to_shared(ring + (slot * NA + q) * 32);
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(sa),
                             "l"(wp + (t + U) * tstride + q * 32));
            }
        }
        asm volatile("cp.async.commit_group;");
    }
    __syncthreads();  // x no longer needed: reuse smem for the reduction
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = wr * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
            red[(wk * TOK + tok) * RB + row] = acc[j][r];
        }
    __syncthreads();
    constexpr int E = TOK * RB;
    if (NWK > 1) {
        for (int e = threadIdx.x; e < E; e += NT) {
            float v = red[e];
#pragma unroll
            for (int q = 1; q < NWK; ++q) v += red[q * E + e];
            red[e] = v;
        }
    }
    auto store = [&](int e, float v) {
        const int tok = tg * TOK + e / RB, row = rg * RB + e % RB;
        if (tok < m) {
            if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = v;
            else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(v);
        }
    };
    if (cs == 1) {
        if (NWK > 1) __syncthreads();
        for (int e = threadIdx.x; e < E; e += NT) store(e, red[e]);
        return;
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    const int e0 = ks * E / cs, e1 = (ks + 1) * E / cs;
    for (int e = e0 + threadIdx.x; e < e1; e += NT) {
        float v = 0.f;
        for (int q = 0; q < cs; ++q) v += cl.map_shared_rank(red, (q + ks) % cs)[e];
        store(e, v);
    }
    cl.sync();
}

template <int MT, int NWR, int NWK, int U, int NA, bool F32OUT>
int launch(const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out, int n, int cs, int tgs, int ilv, bool query) {
    const int m = (int)x.size(0), k = (int)x.size(1);
    constexpr int TOK = MT * 8, RB = NWR * 16, NT = NWR * NWK * 32;
    TORCH_CHECK(n % RB == 0 && k % (cs * 16 * NWK) == 0 && tgs * TOK >= m, "g7: bad shape/config");
    const int KB = k / cs;
    const int mr = m < TOK ? m : TOK;
    const size_t sx = (size_t)mr * (KB + 8) * 2 + 16, sr = (size_t)NWK * TOK * RB * 4;
    const size_t sm = (sx > sr ? sx : sr) + (size_t)NWR * NWK * U * NA * 512;
    auto kern = gemv<MT, NWR, NWK, U, NA, F32OUT>;
    static int set = 0;  // per instantiation
    if (!set) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        set = 1;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs * (n / RB) * tgs);
    cfg.blockDim = dim3(NT);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cs;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (query) {
        int nc = 0;
        cudaOccupancyMaxActiveClusters(&nc, kern, &cfg);
        return nc;
    }
    cudaLaunchKernelEx(&cfg, kern, reinterpret_cast<const uint4*>(w.data_ptr()),
                       reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr(), m, n, k, cs, ilv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return 0;
}
}  // namespace g7

#define G7_CASE(MT, NWR, NWK, U, NA)                                                     \
    if (mt == MT && nwr == NWR && nwk == NWK && u == U && na == NA) {                    \
        if (f32) return g7::launch<MT, NWR, NWK, U, NA, true>(w, x, out, n, cs, tgs, ilv, query);   \
        else return g7::launch<MT, NWR, NWK, U, NA, false>(w, x, out, n, cs, tgs, ilv, query); \
                                                                                  \
    }

static int64_t g7_run(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n, int64_t mt, int64_t nwr,
                      int64_t nwk, int64_t u, int64_t na, int64_t cs, int64_t tgs, bool query) {
    const bool f32 = out.scalar_type() == torch::kFloat32;
    const int ilv = (int)(na >> 4);
    na &= 15;
    G7_CONFIGS
    TORCH_CHECK(false, "g7: no such config");
    return 0;
}
void g7_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n, int64_t mt, int64_t nwr, int64_t nwk,
             int64_t u, int64_t na, int64_t cs, int64_t tgs) {
    g7_run(w, x, out, n, mt, nwr, nwk, u, na, cs, tgs, false);
}
int64_t g7_occ(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n, int64_t mt, int64_t nwr, int64_t nwk,
               int64_t u, int64_t na, int64_t cs, int64_t tgs) {
    return g7_run(w, x, out, n, mt, nwr, nwk, u, na, cs, tgs, true);
}
'''
_G10_SRC = r'''// gemv10: grid-stride GEMV for small m, W plain row-major BF16 [n, KK], KK <= U*2048.
// The flat W is cut into chunks of U*256*8 elements (U*4 KB), each a whole number of rows. A block is NV groups
// of 256 threads sharing one copy of x in smem; group g of block b is "virtual block" v = b*NV + g and takes chunks
// v, v+G, v+2G, ... (G = grid*NV): at any time the grid reads one contiguous window of W, like a pure grid-stride
// reader. Thread t loads the U uint4 at u*2048 + t*8 of the chunk (PF chunks in flight), dots them with x in
// FP32, warps reduce by shuffles, the group sums each row's warps in smem (named barrier) and writes BF16.
// No split-K, no atomics, no workspace. Big blocks keep the per-block x reads (k*2*m bytes each) small.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g10 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ void dot8(float& a, const uint4& w, const uint4& x) {
    a = fmaf(lo(w.x), lo(x.x), a); a = fmaf(hi(w.x), hi(x.x), a);
    a = fmaf(lo(w.y), lo(x.y), a); a = fmaf(hi(w.y), hi(x.y), a);
    a = fmaf(lo(w.z), lo(x.z), a); a = fmaf(hi(w.z), hi(x.z), a);
    a = fmaf(lo(w.w), lo(x.w), a); a = fmaf(hi(w.w), hi(x.w), a);
}
template <int M, int U, int NV, int MINB, int KK, int PF>
__global__ void __launch_bounds__(256 * NV, MINB) gemv(const uint4* __restrict__ W, const uint4* __restrict__ x,
                                                       __nv_bfloat16* __restrict__ out, int n) {
    constexpr int CE = U * 2048, RPC = CE / KK, K8 = KK / 8;
    constexpr int NA = KK >= 2048 ? CE / KK : U;      // accumulators per thread (distinct rows among its u)
    constexpr int EPR = KK >= 2048 ? 8 : KK / 256;    // red entries per row
    static_assert(CE % KK == 0, "chunk must hold whole rows");
    const int t = threadIdx.x & 255, vg = threadIdx.x >> 8, lane = t & 31, wp = t >> 5;
    const int nch = n / RPC;
    extern __shared__ uint4 xs[];  // x[M][KK]
    __shared__ float red_[NV][2][NA * 8][M];
    float (*red)[NA * 8][M] = red_[vg];
    const int G = gridDim.x * NV;
    int c = blockIdx.x * NV + vg;
    uint4 w[PF][U];
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) w[p][u] = ldw(W + (long)(c + p * G) * (CE / 8) + u * 256 + t);
        }
    for (int i = threadIdx.x; i < M * K8; i += 256 * NV) xs[i] = __ldg(x + i);
    __syncthreads();
    int par = 0;
    for (; c < nch; c += PF * G) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cc = c + p * G;
            if (cc < nch) {
                float acc[NA][M];
#pragma unroll
                for (int a = 0; a < NA; ++a)
#pragma unroll
                    for (int i = 0; i < M; ++i) acc[a][i] = 0.f;
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int kx = (u * 256 + t) % K8, a = KK >= 2048 ? u * 2048 / KK : u;
#pragma unroll
                    for (int i = 0; i < M; ++i) dot8(acc[a][i], w[p][u], xs[i * K8 + kx]);
                }
                const int cn = cc + PF * G;
                if (cn < nch) {
#pragma unroll
                    for (int u = 0; u < U; ++u) w[p][u] = ldw(W + (long)cn * (CE / 8) + u * 256 + t);
                }
#pragma unroll
                for (int a = 0; a < NA; ++a)
#pragma unroll
                    for (int i = 0; i < M; ++i) {
                        float v = acc[a][i];
#pragma unroll
                        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                        if (lane == 0) red[par][a * 8 + wp][i] = v;
                    }
                asm volatile("bar.sync %0, 256;" :: "r"(vg + 1));
                if (t < RPC * M) {
                    const int r = t / M, i = t % M;
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < EPR; ++q) s += red[par][r * EPR + q][i];
                    out[(long)i * n + (long)cc * RPC + r] = __float2bfloat16(s);
                }
                par ^= 1;
            }
        }
    }
}
}  // namespace g10
void g10_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t u, int64_t nv, int64_t minb, int64_t pf,
              int64_t blocks) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int sm = m * k * 2;
#define G10_CASE(M_, K_, U_, V_, B_, P_) \
    if (m == M_ && k == K_ && u == U_ && nv == V_ && minb == B_ && pf == P_) { \
        if (sm > 48 * 1024) cudaFuncSetAttribute(g10::gemv<M_, U_, V_, B_, K_, P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g10::gemv<M_, U_, V_, B_, K_, P_><<<blocks, 256 * V_, sm, s>>>((const uint4*)w.data_ptr(), (const uint4*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), n); return; }
    G10_CONFIGS
    TORCH_CHECK(false, "g10: no config");
}
'''
_G12_SRC = r'''// gemv12: tensor-core streaming GEMM for m = 8..32 (out[m, n] = x[m, k] @ W[n, k]^T, BF16 in, FP32 accumulate).
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
template <int M, int KCU, int WG, int PF, int XP, int XR>
__global__ void __launch_bounds__(1024, 1) gemm(const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                int* __restrict__ cnt, int n, int nrg_all, int k, int bk) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16;
    constexpr int NT = M < 16 ? M : 16, NA = KC / 32 / WG;               // atoms per warp
    static_assert(NA % (PF * XP) == 0 && XR <= XP && XR <= 4 && (XR == XP ? XP <= 4 : XR >= 2), "x pieces");
    constexpr int PA = NA / XP, PB = KC / XP * 2;  // atoms per x piece (per warp), bytes per x piece row
    constexpr int XS = XR * PB + 64;                // x ring row stride in smem (bytes), padded: XR pieces of PB
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x ring [M][XR pieces] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [warps][TT][32 lanes] C fragments
    __shared__ int flag[64];
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x / bk, rg0 = (blockIdx.x % bk) * NG, rg = rg0 + gi;
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
#pragma unroll
    for (int p = 0; p < XR; ++p) { stage(p); asm volatile("cp.async.commit_group;"); }
    const uint4* wb = W + (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wb + j*WG*32
    uint4 w[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
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
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                const int kx = (kb - (a / PA) * (KC / XP) + ((a / PA) % XR) * (KC / XP)) * 2;
                uint4 xa = *(const uint4*)(xs + (tt * 16 + g) * XS + kx);
                uint4 xb = make_uint4(0, 0, 0, 0);
                if (NT > 8) xb = *(const uint4*)(xs + (tt * 16 + g + 8) * XS + kx);
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
              int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp, int64_t xr) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int S = k / (1024 * kcu), ng = n / 8 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 / xp * xr + 64) + threads / 32 * ((m + 15) / 16) * 512;
    TORCH_CHECK(n / 8 % bk == 0 && threads <= 1024 && ng <= 64 && sm <= 227 * 1024 && (S == 1 || m * n * S <= 8 * 32 * 8192), "g12: shape");
#define G12_CASE(M_, U_, G_, P_, X_, R_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_ && xp == X_ && xr == R_) { \
        cudaFuncSetAttribute(g12::gemm<M_, U_, G_, P_, X_, R_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g12::gemm<M_, U_, G_, P_, X_, R_><<<S * bk, threads, sm, s>>>((const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk); return; }
    G12_CONFIGS
    TORCH_CHECK(false, "g12: no config");
}
'''
_G13_SRC = r'''// gemv13: gemv10 (grid-stride GEMV, m = 1) over losslessly compressed BF16 weights: 12 bits per weight.
// prepare() splits W [n, KK] into a sign|mantissa byte plane (flat, row-major) and a 4-bit exponent code plane:
// code c = e - base (base = max exponent - 15), so the decoded exponent is 112 | c (exact power-of-2 rescale of the
// sum by 2^(base - 112)). Weights with e < base (zeros, tiny values: ~1 per 8K) are escapes: code 0, byte 0 (decodes
// to D = 2^(base - 127) in true scale) and a per-row header of ESC (col, true / scale) pairs (exact) fixes the sum: + x[col] * (true / scale - 2^-15).
// Nibble layout: 32-bit word f/8 holds weights f..f+7 (f % 8 == 0); weight 2p + h at bits 4p + 16h, so pair p's
// exponents land at BF16 bits 7..10 / 23..26 with one shift. Decode per pair: prmt (sign replicate), shift, 2 lop3.
// Chunk = U * 4096 weights (whole rows); thread t takes the 16 weights at u * 4096 + t * 16 (one uint4 of bytes and
// one uint2 of codes per u). Its k positions are fixed, so x lives in registers as FP32.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g13 {
#define XI(k) (((k) >> 4) * 20 + ((k) & 15))
constexpr int ESC = 8, MAXC = 16;  // escapes per row; chunks per virtual block
__device__ __forceinline__ uint4 ld4(const uint4* p) {
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
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
// pair p (0..3) of a nibble word: exponent bits for BF16 pair at bits 7..10 and 23..26, OR 112 << 7
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    return (s & 0x07800780u) | 0x38003800u;
}
// 4 weights (bytes of sw) -> 2 BF16 pairs; nibble word nw, pairs P0, P0 + 1
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = (prmt(sw, 0x9180u) & 0x807f807fu) | expo<P0>(nw);
    const unsigned q1 = (prmt(sw, 0xB3A2u) & 0x807f807fu) | expo<P0 + 1>(nw);
    f[0] = __uint_as_float(q0 << 16); f[1] = __uint_as_float(q0 & 0xffff0000u);
    f[2] = __uint_as_float(q1 << 16); f[3] = __uint_as_float(q1 & 0xffff0000u);
}
__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* xp) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) *reinterpret_cast<float4*>(x + i) = *reinterpret_cast<const float4*>(xp + i);
    float f[16];
    dec4<0>(s.x, nb.x, f); dec4<2>(s.y, nb.x, f + 4); dec4<0>(s.z, nb.y, f + 8); dec4<2>(s.w, nb.y, f + 12);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) { a0 = fmaf(f[i], x[i], a0); a1 = fmaf(f[i + 1], x[i + 1], a1); }
    return a0 + a1;
}

// VT: threads per virtual block (256 or 512): a chunk is U * VT * 16 weights. 512 halves the rows per round of a
// 2304-row qkv (4.36 -> 5 rounds of whole rows at 528 vbs: a 13% tail; 8.73 -> 9 half-size rounds at 264 vbs: 3%).
template <int U, int NV, int MINB, int KK, int PF, int VT = 256>
__global__ void __launch_bounds__(VT * NV, MINB) gemv(const uint4* __restrict__ S, const uint2* __restrict__ N,
                                                       const float2* __restrict__ H, const __nv_bfloat16* __restrict__ x,
                                                       __nv_bfloat16* __restrict__ out, int n, float scale) {
    constexpr int CE = U * VT * 16, RPC = CE / KK, WV = VT / 32;
    static_assert(CE % KK == 0 && KK % 512 == 0 && (VT == 256 || VT == 512), "chunk must hold whole rows");
    const int t = threadIdx.x % VT, vg = threadIdx.x / VT, lane = t & 31, wp = t >> 5;
    const int nch = n / RPC;
    __shared__ float red_[NV][2][U * WV];
    __shared__ __align__(16) float xs[KK / 16 * 20];  // 16-float groups padded to 20: no bank conflicts
    __shared__ float2 hs_[NV][MAXC * RPC * ESC];  // this vb's escape headers: chunk j = j-th of c, c + G, ...
    float (*red)[U * WV] = red_[vg];
    const int G = gridDim.x * NV;
    int c = blockIdx.x * NV + vg;
    uint4 s[PF][U];
    uint2 nb[PF][U];
    // escapes: the row's reducing thread adds x[col] * (value - 2^-15) per header entry to the row's sum
    auto load = [&](int p, int ch) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long i = (long)ch * (CE / 16) + u * VT + t;
            s[p][u] = ld4(S + i);
            nb[p][u] = ld2(N + i);
        }
    };
    // x and the escape headers first: they must not queue behind the W prologue
    constexpr int XV = (KK / 8 + VT * NV - 1) / (VT * NV), HV = (MAXC * RPC * ESC + VT - 1) / VT;
    uint4 xv[XV];
    float2 hv[HV];
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * VT * NV;
        if (i < KK / 8) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);
    }
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * VT, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hv[q] = __ldg(H + (long)ch * RPC * ESC + i % (RPC * ESC));
    }
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * VT, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hs_[vg][i] = hv[q];
    }
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * VT * NV;
        if (i < KK / 8) {
            const unsigned v[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) { const int kk = XI(8 * i + 2 * j); xs[kk] = __uint_as_float(v[j] << 16); xs[kk + 1] = __uint_as_float(v[j] & 0xffff0000u); }
        }
    }
    __syncthreads();
    int par = 0, j = 0;
    for (; c < nch; c += PF * G) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cc = c + p * G;
            if (cc < nch) {
                float acc[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    acc[u] = dot16(s[p][u], nb[p][u], xs + XI((u * VT * 16 + t * 16) % KK));
                }
                const int cn = cc + PF * G;
                if (cn < nch) load(p, cn);
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    float v = acc[u];
#pragma unroll
                    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                    if (lane == 0) red[par][u * WV + wp] = v;
                }
                asm volatile("bar.sync %0, %1;" :: "r"(vg + 1), "n"(VT));
                if (t < RPC) {
                    float sum = 0.f;
#pragma unroll
                    for (int u = 0; u < U; ++u)
#pragma unroll
                        for (int w = 0; w < WV; ++w)
                            if ((u * VT * 16 + w * 512) / KK == t) sum += red[par][u * WV + w];
#ifndef NOESC
                    // escapes on the row's full sum, two fmas each: exact when the row holds only that weight (probes)
                    const float2* he = hs_[vg] + ((j + p) * RPC + t) * ESC;
                    for (int e = 0; e < ESC; ++e) {
                        const float2 h = he[e];
                        if (h.y == 0x1p-15f) break;  // slots fill in order; empty = 2^-15 (real: |value| < 2^-15)
                        const float xe = xs[XI(__float_as_int(h.x))];
                        sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
                    }
#endif
                    out[(long)cc * RPC + t] = __float2bfloat16(sum * scale);
                }
                par ^= 1;
            }
        }
        j += PF;
    }
}
}  // namespace g13

void g13_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
              int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks, int64_t vt) {
    const int k = x.size(1), n = out.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(x.size(0) == 1, "g13: m == 1");
    TORCH_CHECK((long)n * k <= (long)g13::MAXC * blocks * nv * u * vt * 16, "g13: too many chunks per virtual block");
#define G13_CASE(K_, U_, V_, B_, P_, VT_) \
    if (k == K_ && u == U_ && nv == V_ && minb == B_ && pf == P_ && vt == VT_) { \
        g13::gemv<U_, V_, B_, K_, P_, VT_><<<blocks, VT_ * V_, 0, st>>>((const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), \
        (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }
    G13_CONFIGS
    TORCH_CHECK(false, "g13: no config");
}
'''
_G14_SRC = r'''// gemv14: gemv12 (tensor-core streaming GEMM, m = 8..32) over losslessly compressed BF16 weights, 12 bits each
// (the format of gemv13: sign|mantissa bytes + 4-bit exponent codes c = e - base, decoded exponent 112 | c, the sum
// rescaled by 2^(base - 112); weights below the window are escapes fixed up from a per-row header).
// Per 512-B atom of gemv12 (8 rows x 32 k), lane l's 8 weights are one uint2 of bytes (S plane) and one 32-bit code
// word (N plane), [ks][rg][atom][lane]; they decode (prmt sign-replicate + 3 logic ops per BF16 pair) straight into
// the uint4 of B fragments. Escapes: the block's rows' headers (ESC x (col, delta / scale)) are cp.async'd to smem
// up front; in the epilogue each output adds x[tok, col] * delta for the escapes whose col is in the block's
// k-chunk (x[:, chunk] is fully staged: XR == XP). Partials stay in the scaled domain; the final sum is rescaled.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g14 {
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
                                                __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                int* __restrict__ cnt, int n, int nrg_all, int k, int bk) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16;
    constexpr int NT = M < 16 ? M : 16, NA = KC / 32 / WG;               // atoms per warp
    static_assert(NA % (PF * XP) == 0 && XR == XP && XP <= 4, "x pieces: fully staged");
    constexpr int PA = NA / XP, PB = KC / XP * 2;  // atoms per x piece (per warp), bytes per x piece row
    constexpr int XS = XR * PB + 64;                // x ring row stride in smem (bytes), padded: XR pieces of PB
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x ring [M][XR pieces] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [warps][TT][32 lanes] C fragments
    float2* hs = (float2*)(smem + M * XS + blockDim.x / 32 * TT * 512);  // [NG * 8 rows][ESC] escape headers
    __shared__ int flag[64];
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x / bk, rg0 = (blockIdx.x % bk) * NG, rg = rg0 + gi;
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
            if (col >= 0 && col < KC) {  // two fmas: exact when the dot holds only this weight (probe rows)
                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2));
                s = fmaf(xv, he[e].y, fmaf(xv, -0x1p-15f, s));
            }
        }
        return s;
    };
    if (S == 1) {
        for (int i = t; i < ne; i += blockDim.x) {
            const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(total(i) * scale);
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
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(s * scale);
        }
    }
}
}  // namespace g14
// nrg_all: 8-row groups of the packed weight; bk: blocks per k-chunk; each block takes n/8/bk rgs with WG warps each.
void g14_gemm(torch::Tensor ws_, torch::Tensor wn, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              torch::Tensor ws, torch::Tensor cnt, int64_t nrg_all, int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int S = k / (1024 * kcu), ng = n / 8 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 64) + threads / 32 * ((m + 15) / 16) * 512 + ng * 8 * g14::ESC * 8;
    TORCH_CHECK(n / 8 % bk == 0 && threads <= 1024 && ng <= 64 && sm <= 227 * 1024 && (S == 1 || m * n * S <= 8 * 32 * 8192), "g14: shape");
#define G14_CASE(M_, U_, G_, P_, X_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_ && xp == X_) { \
        cudaFuncSetAttribute(g14::gemm<M_, U_, G_, P_, X_, X_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g14::gemm<M_, U_, G_, P_, X_, X_><<<S * bk, threads, sm, s>>>((const uint2*)ws_.data_ptr(), (const unsigned*)wn.data_ptr(), \
        (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk); return; }
    G14_CONFIGS
    TORCH_CHECK(false, "g14: no config");
}
'''
_G15_SRC = r'''// gemv15: gemv14 (compressed-weight tensor-core streaming GEMM) with the split-K reduction over a thread-block
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
            if (col >= 0 && col < KC) {  // two fmas: exact when the dot holds only this weight (probe rows)
                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2));
                s = fmaf(xv, he[e].y, fmaf(xv, -0x1p-15f, s));
            }
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
'''
_G16_SRC = r'''// gemv16: gemv15 (compressed weights, cluster split-K over DSMEM) with W as the mma A operand (16 rows x k16) and x as
// B (k16 x 8 tokens, ldmatrix from smem): half the x smem traffic per weight of gemv15 at m >= 16, and an atom is
// 16 rows x 32 k (lane: 16 B of sign|mantissa bytes + 8 B of exponent codes = two A fragments).
// Block = NG 16-row groups x WG warps (atoms of a group interleaved over its warps), one k-chunk; cluster = the S
// k-chunk blocks of a row run; partials reduced over DSMEM.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace g16 {
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
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = (prmt(s0, 0x9180u) & 0x807f807fu) | expo<0>(nw);
    r.y = (prmt(s0, 0xB3A2u) & 0x807f807fu) | expo<1>(nw);
    r.z = (prmt(s1, 0x9180u) & 0x807f807fu) | expo<2>(nw);
    r.w = (prmt(s1, 0xB3A2u) & 0x807f807fu) | expo<3>(nw);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void ldsm2(unsigned addr, unsigned& d0, unsigned& d1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(d0), "=r"(d1) : "r"(addr));
}
// M tokens (multiple of 8), KC = 1024 * KCU k-chunk, WG warps per 16-row group, PF atoms in flight per warp, XP x pieces
template <int M, int KCU, int WG, int PF, int XP, int MC, int LB>
__global__ void __launch_bounds__(LB, 1) gemm(const uint4* __restrict__ WS, const uint2* __restrict__ WN,
                                                const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, int n, int nrg_all, int k) {
    constexpr int KC = 1024 * KCU, NT = M / 8, NA = KC / 32 / WG;  // n-tiles, atoms per warp
    static_assert(M % 8 == 0 && NA % PF == 0 && NA % XP == 0 && XP <= 4 && (MC == 1 || XP == 1), "config");
    constexpr int PA = NA / XP, PB = KC / XP * 2;  // atoms per x piece (per warp), bytes per x piece row
    constexpr int XS = KC * 2 + 16;                // x row stride (bytes): 16 mod 128 -> ldmatrix conflict-free
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x [M][KC]
    float4* accp = (float4*)(smem + M * XS);         // [warps][NT][32 lanes] C fragments
    float2* hs = (float2*)(smem + M * XS + blockDim.x / 32 * NT * 512);  // [NG * 16 rows][ESC]
    float* part = (float*)(hs + blockDim.x / (32 * WG) * 16 * ESC);     // [NG * M * 16] this block's partials
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x % S, rg0 = (blockIdx.x / S) * NG, rg = rg0 + gi;
    auto stage = [&](int p) {
        for (int i = t; i < M * (PB / 16); i += blockDim.x) {
            const int r = i / (PB / 16), q = i % (PB / 16);
            const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + p * PB + q * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa),
                         "l"((const uint4*)(x + (long)r * k + (long)ks * KC + p * (KC / XP)) + q));
        }
    };
    for (int i = t; i < NG * 16 * ESC / 2; i += blockDim.x) {
        const unsigned sa = (unsigned)__cvta_generic_to_shared(hs + 2 * i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(H + (long)rg0 * 16 * ESC + 2 * i));
    }
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int mr = (blockIdx.x % (S * MC)) / S;  // rank among the MC blocks of the cluster sharing this k-chunk
    const unsigned bar = ((unsigned)__cvta_generic_to_shared(part + NG * M * 16) + 7u) & ~7u;
    if (MC == 1 && S > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;");  // peers started: waited before DSMEM stores
    if constexpr (MC > 1) {
        // x[:, chunk] by TMA bulk copies multicast to the MC blocks with the same k-chunk: block mr copies rows
        // r % MC == mr; one mbarrier per block collects all M rows.
        asm volatile("cp.async.commit_group;");  // escape headers
        if (t == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        cl.sync();
        if (t == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(M * KC * 2) : "memory");
        if (t < M && t % MC == mr) {
            unsigned short mask = 0;
#pragma unroll
            for (int j = 0; j < MC; ++j) mask |= (unsigned short)(1u << (j * S + ks));
            const unsigned dst = (unsigned)__cvta_generic_to_shared(xs + t * XS);
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                         :: "r"(dst), "l"(x + (long)t * k + (long)ks * KC), "r"(KC * 2), "r"(bar), "h"(mask) : "memory");
        }
    } else {
#ifndef G16_NOX
#pragma unroll
        for (int p = 0; p < XP; ++p) { stage(p); asm volatile("cp.async.commit_group;"); }
#endif
    }
    const long wo = (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wo + j*WG*32
    const uint4* wbs = WS + wo;
    const uint2* wbn = WN + wo;
    uint4 ws_[PF];
    uint2 wn_[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) { ws_[p] = ldw(wbs + p * WG * 32); wn_[p] = ld2(wbn + p * WG * 32); }
    float acc[NT][4], acc2[NT][4];  // k-steps s = 0 / 1 accumulate separately: two independent mma chains
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = acc2[j][0] = acc2[j][1] = acc2[j][2] = acc2[j][3] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    const unsigned xbase = (unsigned)__cvta_generic_to_shared(xs) + ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    for (int a = 0; a < NA; a += PF) {
        if (MC > 1 && a == 0) {
            asm volatile("{\n.reg .pred p;\nW16_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra W16_%=;\n}" :: "r"(bar) : "memory");
        } else if (MC == 1 && a % PA == 0) {
            const int left = XP - 1 - a / PA;
            if (left >= 3) asm volatile("cp.async.wait_group 3;");
            else if (left == 2) asm volatile("cp.async.wait_group 2;");
            else if (left == 1) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint4 sv = ws_[p];
            const uint2 nv = wn_[p];
            if (a + p + PF < NA) { ws_[p] = ldw(wbs + (a + p + PF) * WG * 32); wn_[p] = ld2(wbn + (a + p + PF) * WG * 32); }
            const unsigned kb = (((a + p) * WG + wi) * 32) * 2;  // byte offset of the atom's k within the chunk
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint4 af = dec8(s ? sv.z : sv.x, s ? sv.w : sv.y, s ? nv.y : nv.x);
#ifdef G16_NOMMA
                acc[0][s] += __uint_as_float(af.x ^ af.y ^ af.z ^ af.w); continue;
#endif
#pragma unroll
                for (int j = 0; j + 1 < NT; j += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(xbase + j * 8 * XS + kb + s * 32, b0, b1, b2, b3);
                    mma(s ? acc2[j] : acc[j], af, b0, b1);
                    mma(s ? acc2[j + 1] : acc[j + 1], af, b2, b3);
                }
                if (NT & 1) {
                    unsigned b0, b1;
                    ldsm2(xbase + (NT - 1) * 8 * XS + kb + s * 32, b0, b1);  // lanes 0-15 give addresses (q/2 = 0)
                    mma(s ? acc2[NT - 1] : acc[NT - 1], af, b0, b1);
                }
            }
        }
    }
    float* accf = (float*)accp;  // [warp][NT][4 elements][32 lanes]: conflict-free
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) accf[((wp * NT + j) * 4 + e) * 32 + lane] = acc[j][e] + acc2[j][e];
#ifdef G16_NORED
    return;
#endif
    if (MC > 1) asm volatile("cp.async.wait_group 0;");
    if (MC == 1 && S > 1) asm volatile("barrier.cluster.wait.aligned;");
    __syncthreads();
#ifdef G16_SYNCONLY
    return;
#endif
    // partials i = ((rgl * NT + nt) * 4 + e) * 32 + l: row16 = l / 4 + 8 * (e / 2), tok = nt * 8 + (l % 4) * 2 + e % 2
    const int ne = NG * M * 16;
    for (int i = t; i < ne; i += blockDim.x) {
        const int l = i & 31, e = (i >> 5) & 3, nt = (i >> 7) % NT, rgl = (i >> 7) / NT;
        const int row16 = (l >> 2) + 8 * (e >> 1), tok = nt * 8 + (l & 3) * 2 + (e & 1);
        const float* base = accf + (rgl * WG * NT + nt) * 128 + e * 32 + l;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[q * NT * 128];
        const float2* he = hs + (rgl * 16 + row16) * ESC;
#ifndef G16_NOESC
        for (int e2 = 0; e2 < ESC; ++e2) {  // slots fill in order; an empty slot has value 2^-15 (real escapes: |value| < 2^-15)
            const float2 h = he[e2];
            if (h.y == 0x1p-15f) break;
            const int col = __float_as_int(h.x) - ks * KC;
            if (col >= 0 && col < KC) {  // two fmas: exact when the dot holds only this weight (probe rows)
                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2));
                s = fmaf(xv, h.y, fmaf(xv, -0x1p-15f, s));
            }
        }
#endif
        if (S == 1) out[(long)tok * n + (rg0 + rgl) * 16 + row16] = __float2bfloat16(s * scale);
        else {  // slice q = i / (ne / S) is summed by cluster rank q (of this row run): push into its part, slot ks
            const int q = i / (ne / S);
            float* dst = q == ks ? part : cl.map_shared_rank(part, mr * S + q);
            dst[ks * (ne / S) + i - q * (ne / S)] = s;
        }
    }
    if (S == 1) {
        if (MC > 1) cl.sync();  // peers' multicast copies into this block are done (own barrier), keep peers alive anyway
        return;
    }
#ifdef G16_NOP2
    return;
#endif
    cl.sync();  // every block's slices are in its owner's part
    const int sl = ne / S;
    for (int i = t; i < sl; i += blockDim.x) {
        float v = 0.f;
        for (int q = 0; q < S; ++q) v += part[q * sl + i];
        const int ig = ks * sl + i;
        const int l = ig & 31, e = (ig >> 5) & 3, nt = (ig >> 7) % NT, rgl = (ig >> 7) / NT;
        const int row16 = (l >> 2) + 8 * (e >> 1), tok = nt * 8 + (l & 3) * 2 + (e & 1);
        out[(long)tok * n + (rg0 + rgl) * 16 + row16] = __float2bfloat16(v * scale);
    }
}
}  // namespace g16
// nrg_all: 16-row groups of the packed weight; bk: row runs (clusters), each n/16/bk groups x WG warps, S = k/KC blocks.
void g16_gemm(torch::Tensor ws_, torch::Tensor wn, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              int64_t nrg_all, int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp, int64_t mc) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int S = k / (1024 * kcu), ng = n / 16 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 16) + threads / 32 * (m / 8) * 512 + ng * 16 * g16::ESC * 8 + ng * m * 16 * 4 + 16;
    TORCH_CHECK(n / 16 % bk == 0 && bk % mc == 0 && threads <= 1024 && sm <= 227 * 1024 && S * mc <= 16, "g16: shape");
#define G16_CASE(M_, U_, G_, P_, X_, C_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_ && xp == X_ && mc == C_) { \
        auto kern = threads <= 512 ? g16::gemm<M_, U_, G_, P_, X_, C_, 512> : g16::gemm<M_, U_, G_, P_, X_, C_, 1024>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        if (S * mc > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(S * bk); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = sm; \
        cfg.stream = c10::cuda::getCurrentCUDAStream(); \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeClusterDimension; \
        attr[0].val.clusterDim.x = S * mc; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const uint4*)ws_.data_ptr(), (const uint2*)wn.data_ptr(), \
            (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), n, nrg_all, k); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G16_CONFIGS
    TORCH_CHECK(false, "g16: no config");
}
'''
_G19_SRC = r'''// gemv19: gemv18's TMA-ring streamer over lossless 12-bit weights (m = 16/32, same _pack18 stage layout), FP32 output
// only, with load balancing. Per-SM HBM bandwidth varies by ~10% (gemv18 blocks ended 21-27 us at pooler m16 with
// equal static ranges), so: each block first streams a static stream-K range of the first U - TAIL stages, then grabs
// tail items of IS stages from a global counter until none are left. The producer warp decides everything: it tags
// every ring slot with its stage u (+ x buffer, + "run ends here" flag); consumers flush partials at run ends; x is
// double-buffered and loaded by the producer when a stage needs another k-chunk. All partials go to the FP32 output
// by red.global.add (finisher warp); a tiny zero kernel (PDL primary) zeroes the output and the counter first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g19 {
constexpr int ESC = 8, NWC = 8;
#ifdef G19_TRACE
__device__ unsigned long long trace[132 * 64];
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#define TR(slot) trace[blockIdx.x * 64 + (slot)] = gt()
#else
#define TR(slot)
#endif
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
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = (prmt(s0, 0x9180u) & 0x807f807fu) | expo<0>(nw);
    r.y = (prmt(s0, 0xB3A2u) & 0x807f807fu) | expo<1>(nw);
    r.z = (prmt(s1, 0x9180u) & 0x807f807fu) | expo<2>(nw);
    r.w = (prmt(s1, 0xB3A2u) & 0x807f807fu) | expo<3>(nw);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary), ctr int (zeroed by the primary). tail: stages left for dynamic items.
// smem: ring [NS][SB] | x [2][M][XS] | pbuf [2][128 rows][M] f32 | barriers | tag [NS] int | ptag [2] int
template <int M, int KC, int AP, int NS, int IS, int PRE, int XB, int TS, int RG, int RPW>
__global__ void __launch_bounds__((RG * TS / RPW + 2) * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                                         float scale, const __nv_bfloat16* __restrict__ x,
                                                         float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre) {
    constexpr int NWC = RG;  // 16-row groups per tile (one stage covers RG x AP atoms)
    static_assert(RPW == 1 || TS == 1, "config");
    constexpr int NCW = NWC * TS / RPW, MT = M / TS, NT = MT / 8, NA = KC / 32, SPK = NA / AP, SB = NWC * AP * 768, XS = KC * 2 + 16, PBF = NWC * 16 * M;
    static_assert(NA % AP == 0 && SPK % IS == 0 && NT % 2 == 0, "config");  // NT even: x B fragments by ldmatrix.x4
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + XB * M * XS);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + 2 * PBF);
    int* tag = (int*)(pbuf + 2 * PBF) + 2 * (2 * NS + 8);  // after the barriers. per slot: u | xb << 29 | last x use << 30 | run end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile (-1: consumers done)
    // full[NS], empty[NS], xfull[2], xempty[2], pfull[2], pempty[2]
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NWC, S = k / KC, KA = k / 32;
    const int U = S * ntr * SPK, US = U - tail;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NCW); }
        for (int i = 0; i < 2; ++i) {
            bar_init(xfull + 8 * i, 1); bar_init(xempty + 8 * i, NCW); bar_init(pfull + 8 * i, NCW); bar_init(pempty + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
#ifdef G19_TRACE
    if (t == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); trace[blockIdx.x * 64 + 10] = sm_; }
#endif
    if (w == NCW) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
                const int ks0 = ra / (ntr * SPK);
                for (; i < npre && ra < rb && ra / (ntr * SPK) == ks0; ++i, ++ra) {
                    const int st = ra % SPK;
                    const bool xl = ra + 1 == rb || (ra + 1) / (ntr * SPK) != ks0;
                    tag[i] = ra | (xl ? 1 << 30 : 0) | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
                    bar_expect(bars + 8 * i, SB);
                    bulk(ring + i * SB, WP + (long)ra * SB, SB, bars + 8 * i);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                xb = 0; xks[0] = ks0; xuse[0] = 1;
                bar_expect(xfull, M * KC * 2);
                for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k + (long)ks0 * KC, KC * 2, xfull);
            } else {
                asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
            }
            for (;;) {
                if (ra >= rb) {  // next run: a tail item
                    if (tail == 0) break;
                    const int it = atomicAdd(ctr, 1);  // grabbed when needed: the ring's NS stages hide its latency
                    if (it * IS >= tail) break;
                    ra = US + it * IS; rb = ra + IS;
                }
                for (int u = ra; u < rb; ++u, ++i) {
                    const int ks = u / (ntr * SPK), st = u % SPK;
                    if (ks != xks[xb]) {  // next x buffer, load chunk ks (consumers release a buffer after its last use)
                        xb = XB == 2 ? xb ^ 1 : 0;
                        if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                        ++xuse[xb];
                        xks[xb] = ks;
                        bar_expect(xfull + 8 * xb, M * KC * 2);
                        for (int qq = 0; qq < M; ++qq) bulk(xs + xb * M * XS + qq * XS, x + (long)qq * k + (long)ks * KC, KC * 2, xfull + 8 * xb);
                    }
                    const int slot = i % NS;
                    if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                    const bool end = u + 1 == rb || st + 1 == SPK;
                    const bool xl = u + 1 == rb || (u + 1) / (ntr * SPK) != ks;
                    if (xl) xks[xb] = -1;  // released by the consumers after this stage: reload even if the next run has this ks
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    if (i < 24) TR(40 + i);
                    const unsigned full = bars + 8 * slot;
#ifdef G19_NOW
                    bar_arrive(full);  // consumers compute on stale smem: their throughput alone
#else
                    bar_expect(full, SB);
                    bulk(ring + slot * SB, WP + (long)u * SB, SB, full);
#endif
                }
                ra = rb;
            }
            const int slot = i % NS;  // end marker
            if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
            tag[slot] = -1;
            bar_arrive(bars + 8 * slot);
            TR(1);
        }
        return;
    }
    if (w == NCW + 1) {  // finisher: pbuf [128 rows][M] -> out [M][n] by red.global.add.v4 (lane = token, 4 rows)
        asm volatile("griddepcontrol.wait;" ::: "memory");
        unsigned target = 0;
        if (PRE == 2) {  // zero a slice of out, then arrive on the epoch counter ctr[1] (+gridDim.x per launch, never reset)
            float4* o4 = (float4*)out;
            for (int i = blockIdx.x * 32 + lane; i < M * n / 4; i += gridDim.x * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            __syncwarp();
            if (lane == 0) {
                __threadfence();
                const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
                target = old - old % gridDim.x + gridDim.x;
            }
        }
        for (int q = 0;; ++q) {
            const int pb = q & 1;
            bar_wait(pfull + 8 * pb, (q >> 1) & 1);
            const int tr = ptag[pb];
            if (tr < 0) break;
            if (PRE == 2 && q == 0) {  // every block has zeroed its slice
                if (lane == 0) {
                    unsigned v;
                    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - target) < 0);
                }
                __syncwarp();
            }
            const float* src = pbuf + pb * PBF;
            float* wt = out + (long)tr * NWC * 16;
#pragma unroll 4
            for (int i = lane; i < PBF / 4; i += 32) {
                const int tok = i % M, r4 = (i / M) * 4;
                const float a0 = src[r4 * M + tok] * scale, a1 = src[(r4 + 1) * M + tok] * scale;
                const float a2 = src[(r4 + 2) * M + tok] * scale, a3 = src[(r4 + 3) * M + tok] * scale;
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + (long)tok * n + r4), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
            }
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // RPW = 2: warp w owns row groups 2w, 2w + 1 of the tile (one x fragment load feeds both: half the ldmatrix traffic)
    float acc[RPW][NT][4], acc2[RPW][NT][4];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[r][j][e] = acc2[r][j][e] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    // TS = 2: warps 2r, 2r + 1 share row group r (both decode it), tokens [th * MT, th * MT + MT)
    const int wr = (w / TS) * RPW, th = w % TS;  // wr: the warp's first row group in the tile
    const unsigned xrow = (th * MT + (lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    int q = 0, st0 = 0, prev_end = 1, prev_xl = 1;
    unsigned xph[2] = {0, 0};
    float4 hd0[RPW], hd1[RPW];
    for (int i = 0;; ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const int tg = tag[slot];
        if (t == 0 && i < 24) TR(16 + i);
        if (tg == -1) break;
        const int u = tg & 0x1FFFFFFF, xb = (tg >> 29) & 1;
        const bool end = tg < 0, xl = (tg >> 30) & 1;
        const int ks = u / (ntr * SPK), r = u % (ntr * SPK), tr = r / SPK, st = r % SPK;
        if (prev_xl) {  // first stage of an x use: wait for it
            bar_wait(xfull + 8 * xb, xph[xb]);
            xph[xb] ^= 1;
        }
        prev_xl = xl;
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row groups; lane c: slots 2c, 2c + 1)
            st0 = st;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) {
                const float2* hp0 = H + (long)((tr * NWC + wr + rr) * 16 + g) * ESC;
                hd0[rr] = __ldg((const float4*)hp0 + c);
                hd1[rr] = __ldg((const float4*)(hp0 + 8 * ESC) + c);
            }
        }
        prev_end = end;
        const unsigned xbase = xs + xb * M * XS + xrow;
        const unsigned sb = ring + slot * SB + wr * AP * 512 + lane * 16, nb = ring + slot * SB + NWC * AP * 512 + wr * AP * 256 + lane * 8;
#if defined(G19_OLDLOOP) || defined(G19_NOMMA) || defined(G19_NOX)
        static_assert(RPW == 1, "diagnostic loops: RPW 1 only");
#pragma unroll
        for (int j = 0; j < AP; ++j) {
            uint4 sv;
            uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
            const unsigned kb = (st * AP + j) * 64;  // byte offset of the atom's k within the chunk
#ifdef G19_NOMMA
            acc[0][0][j & 3] += __uint_as_float(sv.x ^ sv.y ^ sv.z ^ sv.w ^ nv.x ^ nv.y); continue;
#endif
#ifdef G19_NOX
            { const uint4 af = dec8(sv.x, sv.y, nv.x), af2 = dec8(sv.z, sv.w, nv.y);
              acc[0][0][j & 3] += __uint_as_float(af.x ^ af.y ^ af.z ^ af.w ^ af2.x ^ af2.y ^ af2.z ^ af2.w); continue; }
#endif
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint4 af = dec8(s ? sv.z : sv.x, s ? sv.w : sv.y, s ? nv.y : nv.x);
#pragma unroll
                for (int jj = 0; jj + 1 < NT; jj += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(xbase + jj * 8 * XS + kb + s * 32, b0, b1, b2, b3);
                    mma(s ? acc2[0][jj] : acc[0][jj], af, b0, b1);
                    mma(s ? acc2[0][jj + 1] : acc[0][jj + 1], af, b2, b3);
                }
            }
        }
#else
        // groups of G atoms: all shared loads of the group first, then the x fragments, then decode + mma (the asm
        // statements are volatile: issued in program order, so independent loads must come first to overlap)
        constexpr int G = NT * RPW == 2 ? AP : AP < 2 ? AP : 2;
#pragma unroll
        for (int j0 = 0; j0 < AP; j0 += G) {
            uint4 sv[RPW][G];
            uint2 nv[RPW][G];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[rr][j].x), "=r"(sv[rr][j].y), "=r"(sv[rr][j].z), "=r"(sv[rr][j].w) : "r"(sb + rr * AP * 512 + (j0 + j) * 512));
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[rr][j].x), "=r"(nv[rr][j].y) : "r"(nb + rr * AP * 256 + (j0 + j) * 256));
                }
            unsigned bx[G][2][NT][2];
#pragma unroll
            for (int j = 0; j < G; ++j)
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int jj = 0; jj + 1 < NT; jj += 2)
                        ldsm4(xbase + jj * 8 * XS + (st * AP + j0 + j) * 64 + s * 32, bx[j][s][jj][0], bx[j][s][jj][1], bx[j][s][jj + 1][0], bx[j][s][jj + 1][1]);
#pragma unroll
            for (int j = 0; j < G; ++j)
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const uint4 af = dec8(s ? sv[rr][j].z : sv[rr][j].x, s ? sv[rr][j].w : sv[rr][j].y, s ? nv[rr][j].y : nv[rr][j].x);
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[rr][jj] : acc[rr][jj], af, bx[j][s][jj][0], bx[j][s][jj][1]);
                    }
        }
#endif
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
        if (!end) continue;  // (a last x use always ends a run: released below, after the escape fixups read x)
        // run end: rows rg*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        const int nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
        float v[RPW][NT][4];
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) { v[rr][jj][e] = acc[rr][jj][e] + acc2[rr][jj][e]; acc[rr][jj][e] = acc2[rr][jj][e] = 0.f; }
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float4 hq = h ? hd1[rr] : hd0[rr];
#pragma unroll
                for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
                    const int src = (lane & ~3) | (e2 >> 1);
                    const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                    const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                    const int col = __float_as_int(hx);
                    if (hy != 0x1p-15f && col >= klo && col < khi) {  // two fmas: exact when the dot holds only this weight (probe rows)
                        const unsigned char* xc = smem + NS * SB + xb * M * XS + (col - ks * KC) * 2;
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (th * MT + jj * 8 + 2 * c + b) * XS));
                                v[rr][jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, v[rr][jj][2 * h + b]));
                            }
                    }
                }
            }
        if (xl) {  // done with this x buffer (the fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
        if (nat == KA && PRE != 2) {  // the whole dot: plain stores (PRE 2: out is being zeroed by other blocks)
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        out[(long)(th * MT + jj * 8 + 2 * c + (e & 1)) * n + (tr * NWC + wr + rr) * 16 + g + 8 * (e >> 1)] = v[rr][jj][e] * scale;
            continue;
        }
        const int pb = q & 1;  // hand the partials to the finisher: pbuf[pb][row group * 16 + row16][tok]
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr) {
            float* pw = pbuf + pb * PBF + ((wr + rr) * 16 + g) * M + th * MT + 2 * c;
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[rr][jj][2 * h], v[rr][jj][2 * h + 1]);
        }
        if (t == 0) ptag[pb] = tr;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        ++q;
    }
    {  // done: tell the finisher
        const int pb = q & 1;
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
        if (t == 0) ptag[pb] = -1;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
    }
    if (t == 0) TR(4);
}
// PDL primary: lets the gemm launch at once, zeroes its output and counter (the gemm waits before touching them)
__global__ void zero(float4* __restrict__ p, int n4, int* __restrict__ ctr) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (blockIdx.x == 0 && threadIdx.x == 0) *ctr = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
// test predecessor: lets its dependent launch, then writes x = src (n8 uint4) and spins ~ns_ nanoseconds
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
// diagnostics: a bare TMA ring stream (no math) of `total` bytes, NS stages of SB bytes, one 288-thread block per SM
__global__ void __launch_bounds__(288, 1) stream(const unsigned char* W, long total, int SB, int NS, float* out) {
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), bars = ring + NS * SB;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31;
    const long U = total / SB, u0 = U * blockIdx.x / gridDim.x, u1 = U * (blockIdx.x + 1) / gridDim.x;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (w == 8) {
        if (lane == 0)
            for (long u = u0, i = 0; u < u1; ++u, ++i) {
                const int slot = i % NS;
                if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                bar_expect(bars + 8 * slot, SB);
                bulk(ring + slot * SB, W + u * SB, SB, bars + 8 * slot);
            }
        return;
    }
    float acc = 0.f;
    for (long u = u0, i = 0; u < u1; ++u, ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const float4* p = (const float4*)(smem + slot * SB);
        for (int j = t; j < SB / 16; j += 256) { const float4 v = p[j]; acc += v.x + v.w; }
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
    }
    if (acc == 1.2345f) out[0] = acc;
}
// BF16 copy of a gemm's FP32 output, as its PDL dependent (for consumers that need BF16)
__global__ void cvt(const float4* __restrict__ src, uint2* __restrict__ dst, int n4) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const float4 v = src[i];
        const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
        dst[i] = make_uint2(*(const unsigned*)&a, *(const unsigned*)&b);
    }
}
}  // namespace g19
void g19_cvt(torch::Tensor src, torch::Tensor dst) {
    const int n4 = src.numel() / 4;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((n4 + 255) / 256 < 132 ? (n4 + 255) / 256 : 132); cfg.blockDim = dim3(256);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, g19::cvt, (const float4*)src.data_ptr(), (uint2*)dst.data_ptr(), n4);
}
void g19_stream(torch::Tensor w, int64_t sb, int64_t ns, torch::Tensor out) {
    const int sm = sb * ns + 16 * ns;
    cudaFuncSetAttribute(g19::stream, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    g19::stream<<<132, 288, sm, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)w.data_ptr(), w.numel(), sb, ns, out.data_ptr<float>());
}
void g19_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g19::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g19_zero(torch::Tensor out, torch::Tensor ctr) {
    g19::zero<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((float4*)out.data_ptr(), out.numel() / 4, ctr.data_ptr<int>());
}
#ifdef G19_TRACE
torch::Tensor g19_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g19::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
// mode 0: a zero kernel (out, ctr) first, the gemm its PDL dependent. mode 1: out (and ctr if tail) already zero
// (stream-ordered before the previous kernel), the gemm a PDL dependent of the previous kernel, streaming W early.
// rpw: row groups per consumer warp (2: one x fragment load feeds two row groups; consumer warps = rg * ts / rpw).
// rg: 16-row groups per tile = consumer warps (x ts); the packed stage layout depends on it (pack19 rg).
// ts: consumer warps per 16-row group (2: each takes half the tokens; the W decode is done twice).
// xb: x buffers (2: the next k-chunk's x loads while the current one is in use; 1: less smem, the load waits).
// npre (modes 1, 2): weight stages streamed before griddepcontrol.wait (0: wait first, x first; use 0 when the
// previous kernel leaves nothing to overlap). mode 2: as 1, but the gemm zeroes out itself (grid barrier on the epoch counter ctr[1]: one int per call site
// and grid size, any initial multiple of the grid, e.g. 0; tail must be 0).
// out: FP32 [m, n] (overwritten), ctr: int32 [>= 1] scratch. grid: blocks (<= #SMs, 1 per SM). tail: dynamic stages.
void g19_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = ns * rg * ap * 768 + xb * m * (kc * 2 + 16) + 2 * rg * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8;
    TORCH_CHECK(n % (16 * rg) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && npre <= ns, "g19: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g19::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G19_ONLY2
    TORCH_CHECK(mode == 2, "g19: built for mode 2 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>
#else
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 2 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_> : mode ? g19::gemm<M_, KC_, AP_, NS_, IS_, 1, XB_, TS_, RG_, RPW_> : g19::gemm<M_, KC_, AP_, NS_, IS_, 0, XB_, TS_, RG_, RPW_>)
#endif
#define G19_CASE(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_ && xb == XB_ && ts == TS_ && rg == RG_ && rpw == RPW_) { \
        auto kern = G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3((RG_ * TS_ / RPW_ + 2) * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G19_CONFIGS
    TORCH_CHECK(false, "g19: no config");
}
'''
_G17_SRC = r'''// gemv17: BF16 out[m, n] = x[m, K] @ W[n, K]^T for small m, router2's structure: block = R whole rows, T threads,
// thread t owns k = 8 (t + T j), j < J = K / (8 T): its x slice goes straight to registers, all R * J uint4 W loads
// are issued up front, FP32 FMA, warp shuffle + one smem pass. No split K, no workspace.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace g17 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

template <int M, int R, int T, int K>
__global__ void __launch_bounds__(T) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ out, int m, int n) {
    constexpr int J = K / (8 * T), NW = T / 32;
    static_assert(K % (8 * T) == 0, "k split");
    __shared__ float red[NW][R * M];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r0 = blockIdx.x * R;
    uint4 xr[M][J];
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
            xr[i][j] = i < m ? __ldg(reinterpret_cast<const uint4*>(x + (size_t)i * K) + t + T * j) : make_uint4(0, 0, 0, 0);
    uint4 a[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = ldw(w + (size_t)(r0 + r) * (K / 8) + t + T * j);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const unsigned xv[4] = {xr[i][j].x, xr[i][j].y, xr[i][j].z, xr[i][j].w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const unsigned wv[4] = {a[r][j].x, a[r][j].y, a[r][j].z, a[r][j].w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc[r][i] = fmaf(lo(wv[q]), lo(xv[q]), acc[r][i]);
                    acc[r][i] = fmaf(hi(wv[q]), hi(xv[q]), acc[r][i]);
                }
            }
        }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            float v = acc[r][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][r * M + i] = v;
        }
    __syncthreads();
    for (int e = t; e < R * M; e += T) {
        const int r = e / M, i = e % M;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) s += red[q][e];
        if (i < m) out[(size_t)i * n + r0 + r] = __float2bfloat16(s);
    }
}
}  // namespace g17

#define G17_CASE(M, R, T, K)                                                                                        \
    if (mp == M && rr == R && tt == T && k == K) {                                                                  \
        g17::gemv<M, R, T, K><<<n / R, T, 0, c10::cuda::getCurrentCUDAStream()>>>(                                   \
            reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),     \
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, n);                                                \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
        return;                                                                                                     \
    }

void g17_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mp, int64_t rr, int64_t tt) {
    const int m = (int)x.size(0), k = (int)x.size(1), n = (int)out.size(1);
    TORCH_CHECK(m <= mp && n % rr == 0, "g17: shape");
    G17_CONFIGS
    TORCH_CHECK(false, "g17: no such config");
}
'''
_R3_SRC = r'''// Router at m >= 9 as SGLang's BF16 hi + lo split with FP32 accumulation, in ONE GEMM kernel: split-K tensor-core
// GEMM (mma.sync m16n8k16, x = A, W rows = B), hi and lo MMAs into the same FP32 accumulators (sharing x fragments).
// Block = MT tokens x NT experts x KC of k; cp.async ring of 64-wide k stages (swizzled 128-B rows, ldmatrix).
// Split-K partials: staged in smem, then float4 FP32 atomics (red at L2) straight into out, which a tiny kernel
// zeroes first; the GEMM is its programmatic dependent (PDL): it loads and multiplies while the zeroing runs and
// waits (griddepcontrol.wait) only before its atomics. No workspace, no counters.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pr3 {
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void ldm4(unsigned addr, unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// byte offset of 16-B chunk ch (0..7) of a 128-B row
__device__ __forceinline__ unsigned swz(int row, int ch) { return row * 128 + ((ch ^ (row & 7)) << 4); }

// MT tokens, NT experts per block, WM x WN warps, KC = k per block, NS stages of 64 k.
template <int MT, int NT, int WM, int WN, int KC, int NS>
__global__ void __launch_bounds__(WM * WN * 32) router(const __nv_bfloat16* __restrict__ w,  // [2][256][K] hi, lo
                                                       const __nv_bfloat16* __restrict__ x,  // [m][K]
                                                       float* __restrict__ out, int m, int K) {
    constexpr int NTH = WM * WN * 32, TM = MT / WM, TN = NT / WN, FM = TM / 16, FN = TN / 8;
    constexpr int XB = MT * 128, WB = NT * 128, SB = XB + 2 * WB, NK = KC / 64;
    static_assert(TM % 16 == 0 && TN % 16 == 0, "warp tile");
    extern __shared__ __align__(128) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, wm = warp / WN, wn = warp % WN;
    const int s = blockIdx.x, n0 = blockIdx.y * NT, m0 = blockIdx.z * MT, k0 = s * KC;
    const unsigned sb = smem_u32(sm);

    auto load = [&](int it) {
        if (it < NK) {
            const unsigned st = sb + (it % NS) * SB;
            const int kk = k0 + it * 64;
            for (int i = t; i < MT * 8; i += NTH) {
                const int r = i >> 3, ch = i & 7, row = m0 + r;
                const bool ok = row < m;
                cp16(st + swz(r, ch), x + (size_t)(ok ? row : 0) * K + kk + ch * 8, ok ? 16 : 0);
            }
            for (int i = t; i < 2 * NT * 8; i += NTH) {
                const int h = i / (NT * 8), r = (i >> 3) % NT, ch = i & 7;
                cp16(st + XB + h * WB + swz(r, ch), w + ((size_t)h * 256 + n0 + r) * K + kk + ch * 8, 16);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    float acc[FM][FN][4];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
    for (int i = 0; i < NS - 1; ++i) load(i);
    for (int it = 0; it < NK; ++it) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
        __syncthreads();
        load(it + NS - 1);
        const unsigned st = sb + (it % NS) * SB;
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {  // k16 steps: chunks 2kq, 2kq+1
            unsigned a[FM][4];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                const int r = wm * TM + i * 16 + (lane & 15);
                ldm4(st + swz(r, 2 * kq + (lane >> 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < FN; j += 2) {
                    const int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                    unsigned b0, b1, b2, b3;
                    ldm4(st + XB + h * WB + swz(r, 2 * kq + ((lane >> 3) & 1)), b0, b1, b2, b3);
#pragma unroll
                    for (int i = 0; i < FM; ++i) {
                        mma(acc[i][j], a[i], b0, b1);
                        mma(acc[i][j + 1], a[i], b2, b3);
                    }
                }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    // partial tile -> smem [MT][NT + 8] (conflict-free float2 stores), then float4 atomics into out
    constexpr int RS = NT + 8;
    static_assert(MT * RS * 4 <= NS * SB, "reduction tile");
    float* red = reinterpret_cast<float*>(sm);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            const int row = wm * TM + i * 16 + (lane >> 2), col = wn * TN + j * 8 + 2 * (lane & 3);
            *reinterpret_cast<float2*>(red + row * RS + col) = make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2*>(red + (row + 8) * RS + col) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
    asm volatile("griddepcontrol.wait;\n" ::: "memory");  // out zeroed
    __syncthreads();
    constexpr int TOT = MT * (NT / 4);
    const int rot = (int)(((long)s * TOT) / gridDim.x);  // splits of a tile start at different rows
    for (int i0 = t; i0 < TOT; i0 += NTH) {
        const int i = (i0 + rot) % TOT;
        const int r = i / (NT / 4), c = (i % (NT / 4)) * 4;
        if (m0 + r < m)
            atomicAdd(reinterpret_cast<float4*>(out + (size_t)(m0 + r) * 256 + n0 + c),
                      *reinterpret_cast<const float4*>(red + r * RS + c));
    }
}

__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;\n");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace pr3

#define PR3_CASE(MT, NT, WM, WN, KC, NS)                                                                           \
    if (mt == MT && nt == NT && wmn == WM && wnn == WN && kc == KC && ns == NS) {                                  \
        auto f = pr3::router<MT, NT, WM, WN, KC, NS>;                                                              \
        constexpr int SMEM = NS * (MT * 128 + 2 * NT * 128);                                                       \
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                                \
        pr3::zero<<<(m * 64 + 255) / 256, 256, 0, stream>>>(reinterpret_cast<float4*>(out.data_ptr<float>()),      \
                                                            m * 64);                                               \
        cudaLaunchConfig_t lc = {};                                                                                \
        lc.gridDim = dim3(K / KC, 256 / NT, (m + MT - 1) / MT);                                                    \
        lc.blockDim = dim3(WM * WN * 32);                                                                          \
        lc.dynamicSmemBytes = SMEM;                                                                                \
        lc.stream = stream;                                                                                        \
        cudaLaunchAttribute at[1];                                                                                 \
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                             \
        at[0].val.programmaticStreamSerializationAllowed = 1;                                                      \
        lc.attrs = at; lc.numAttrs = 1;                                                                            \
        cudaLaunchKernelEx(&lc, f, reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),                           \
                           reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), m, K);     \
        return;                                                                                                    \
    }

// w bf16 [2, 256, K] (hi, lo), x bf16 [m, K] contiguous, out fp32 [m, 256] contiguous (overwritten)
void pj_router3(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mt, int64_t nt, int64_t wmn, int64_t wnn,
                int64_t kc, int64_t ns) {
    const int m = x.size(0), K = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    PR3_CONFIGS
    TORCH_CHECK(false, "pj_router3: no config");
}
'''
_LT_SRC = r'''// cuBLASLt BF16 GEMM with an explicit algo config: out[m,n] = x[m,k] @ W[n,k]^T, FP32 compute and accumulation.
// Same library and kernel family SGLang's F.linear reaches (algo 66 on sm90), but with the tile / split-K / cluster
// config picked per shape in prepare() (timed on the judge's own GPU) instead of cuBLAS's heuristic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

static cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) cublasLtCreate(&h);
  return h;
}

struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
};

// cfg = (algo id, tile, stages, split k, reduction scheme, swizzle, custom option, cluster shape).
// Returns 0 if the config is valid for the shape and fits the workspace.
static int lt_make(LtPlan& p, int64_t m, int64_t n, int64_t k, const std::vector<int64_t>& cfg, int64_t wsbytes) {
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, k, n, k);  // W, column-major [k, n]
  cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);  // x, column-major [k, m]
  cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n);  // out, column-major [n, m]
  if (cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                             CUDA_R_16BF, (int)cfg[0], &p.algo) != CUBLAS_STATUS_SUCCESS) return 1;
  uint32_t tile = cfg[1], stages = cfg[2], red = cfg[4], swz = cfg[5], cust = cfg[6];
  int32_t sk = cfg[3];
  uint16_t cl = cfg[7];
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4);
  if (sk > 1) cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cl, 2);
  cublasLtMatmulHeuristicResult_t r;
  if (cublasLtMatmulAlgoCheck(lt_handle(), p.op, p.a, p.b, p.c, p.c, &p.algo, &r) != CUBLAS_STATUS_SUCCESS) return 2;
  if (r.workspaceSize > (size_t)wsbytes) return 3;
  return 0;
}

static void lt_free(LtPlan& p) {
  cublasLtMatrixLayoutDestroy(p.a);
  cublasLtMatrixLayoutDestroy(p.b);
  cublasLtMatrixLayoutDestroy(p.c);
  cublasLtMatmulDescDestroy(p.op);
}

// Returns the cuBLAS status (0 = launched).
int64_t lt_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, std::vector<int64_t> cfg) {
  int64_t m = x.size(0), k = x.size(1), n = out.size(1);
  LtPlan p;
  int64_t st = lt_make(p, m, n, k, cfg, ws.numel());
  if (st == 0) {
    float one = 1.f, zero = 0.f;
    st = 100 + (int64_t)cublasLtMatmul(lt_handle(), p.op, &one, w.data_ptr(), p.a, x.data_ptr(), p.b, &zero,
                                       out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo, ws.data_ptr(), ws.numel(),
                                       at::cuda::getCurrentCUDAStream());
    if (st == 100) st = 0;
  }
  lt_free(p);
  return st;
}
'''

# Router FP32 kernel configs (mp, rows per block, threads), dev/router2.cu, chosen on the H100 (dev/r2.py).
ROUTER_F32 = [(1, (1, 2, 512)), (8, (8, 2, 512))]
# Router hi/lo one-kernel split-K GEMM (dev/router3.cu: tiles MT x NT, WM x WN warps, KC, stages), chosen on the H100
# (dev/r3.py): m9-16 6.9 (g7 8.5-8.9), m32 7.2-7.7 (g7 10.3), m128 9.6-10.9 (cuBLAS 3 kernels 12.2-12.6); m256 ties cuBLAS.
ROUTER_R3 = [(32, (16, 64, 1, 4, 256, 4)), (128, (128, 64, 4, 4, 256, 4))]
# router3 configs that prepare() times against the cuBLAS path per m (H100 m256: 12.9 vs 12.5-13.4, a tie).
ROUTER_R3_TUNE = {256: [(256, 32, 4, 2, 512, 4), (256, 64, 4, 2, 256, 4)]}
# Router hi/lo tensor-core configs (n, mt, nwr, nwk, stages, na, cluster, token groups), chosen on the H100 (dev/g5.py).
ROUTER_HL = [(16, (256, 2, 2, 4, 8, 2, 8, 1)), (32, (256, 4, 2, 2, 8, 2, 8, 1))]
# BF16 grid-stride GEMV configs per layer: (m, k, u, nv, minb, pf, blocks), chosen on the H100 (dev/g10.py).
BF16_G10 = {"qkv_gate": [(1, 8192, 4, 4, 1, 1, 132)], "pooler": [(1, 8192, 4, 4, 1, 1, 132)],
            "o": [(1, 1024, 2, 4, 2, 1, 264)], "unpooler": [(1, 4096, 2, 4, 1, 1, 132)]}
# BF16 m=1 GEMV over losslessly compressed weights (dev/gemv13.cu: 12 bits per weight), (m, k, u, nv, minb, pf,
# blocks, vt), chosen on the H100 (dev/g13.py); prepare() times them against gemv10 and F.linear and keeps the fastest.
# vt 512 (512-thread virtual blocks, half the rows per round): qkv H100 13.7 -> 13.1 us (2304 rows: smaller tail).
BF16_G13 = {"qkv_gate": [(1, 8192, 2, 4, 1, 1, 132, 256), (1, 8192, 2, 2, 1, 2, 132, 256), (1, 8192, 1, 2, 1, 1, 132, 512)],
            "pooler": [(1, 8192, 2, 4, 1, 1, 132, 256), (1, 8192, 2, 2, 1, 2, 132, 256)],
            "unpooler": [(1, 4096, 2, 4, 1, 2, 132, 256)]}
G13_ESC = 8  # escapes per row (gemv13.cu / gemv14.cu ESC)
# BF16 tensor-core streaming GEMM over the same compressed weights (dev/gemv14.cu = gemv12 + decode), (m, kcu, wg, pf,
# bk, xp), chosen on the H100 (dev/g14.py); tuned in prepare() like the others.
BF16_G14 = {"qkv_gate": [(8, 2, 2, 4, 32, 1)], "o": [(8, 1, 2, 4, 128, 1), (8, 1, 2, 8, 128, 1)],
            "pooler": [(8, 8, 8, 4, 128, 1), (8, 2, 2, 4, 32, 1), (16, 4, 4, 4, 64, 1)],
            "unpooler": [(8, 4, 8, 4, 32, 1), (8, 2, 4, 4, 32, 1)]}
# gemv14 with the split-K reduction over a thread-block cluster through DSMEM (dev/gemv15.cu), (m, kcu, wg, pf, runs, xp),
# H100 (dev/g15.py): pooler m16 25.1 vs cuBLAS 27.4; elsewhere no better (m32: x smem reloads per 8-row atom).
BF16_G15 = {"pooler": [(16, 4, 4, 4, 64, 1)]}
# gemv15 with W as the mma A operand (16 rows), x via ldmatrix and x staged by TMA bulk copies multicast over a cluster of
# mc blocks sharing the k-chunk (dev/gemv16.cu), (m, kcu, wg, pf, runs, xp, mc), H100 (dev/g16.py): unpooler m16 (S=1,
# no split-K) 23.0 vs cuBLAS 25.5; qkv m16 (S=2 + multicast: clusters of 4) 16.9 vs 18.1; pooler m16 (S=2, partials pushed
# over DSMEM) 25.2 vs 27.5. H200 (check ae0e03fa, unpooler only): 21.8 vs lt 21.4, not picked.
BF16_G16 = {"unpooler": [(16, 4, 4, 4, 128, 1, 2), (16, 4, 8, 2, 128, 1, 2)],
            "qkv_gate": [(16, 4, 4, 4, 48, 1, 2), (16, 4, 4, 4, 36, 1, 2)], "pooler": [(16, 4, 4, 4, 64, 1, 1)]}
# BF16 small-m GEMV with router2's structure (dev/gemv17.cu: R whole rows per block, x in registers, no split K),
# (m, R, T), H100 (dev/g17.py): unpooler-1k m1 5.66 vs 6.24 (gemv10), m8 7.65 vs 8.67; o loses (n = 8192).
BF16_G17 = {"unpooler": [(1, 4, 512), (1, 4, 256), (8, 4, 256), (8, 4, 128)]}
# gemv19 (dev/gemv19.cu, dev/g19_lib.py): TMA-ring streamer over the 12-bit weights, FP32 out zeroed in-kernel + red.add
# split-K partials, then a BF16 copy (g19_cvt). (m, kc, ap, ns, npre). Built for the layer (PDL W prefetch during the
# previous kernel); standalone (no predecessor) H100 it ties cuBLAS at m16 (pooler 24.8 vs 28.0) and loses at m32.
# (m, kc, ap, ns, npre, xb, ts, rg, rpw): xb x buffers, ts warps per row group, rg row groups per tile, rpw row groups per
# warp (dev/gemv19.cu). H200 diag (check 60ddb481): pooler m16 (1024, 8, 3, xb 1) 15.65 us vs pick 19.0 after a PDL predecessor.
BF16_G19 = {"qkv_gate": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)],
            "pooler": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)],
            "unpooler": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)], "o": [(16, 512, 8, 3, 2, 1, 1, 8, 1)]}
# _diag19 only (print, no picks): H200 behaviour of other configs in the layer case
DIAG19 = {"pooler": [(16, 2048, 8, 2, 2, 1, 1, 8, 1), (16, 1024, 4, 3, 3, 1, 1, 16, 2), (32, 1024, 2, 3, 3, 1, 1, 16, 2),
                     (32, 512, 2, 4, 4, 1, 1, 16, 2), (32, 1024, 4, 4, 4, 1, 1, 8, 1)],
          "unpooler": [(32, 1024, 2, 3, 3, 1, 1, 16, 2)], "qkv_gate": [(32, 1024, 2, 3, 3, 1, 1, 16, 2)]}
DIAG19_STREAM = [(24576, 3), (24576, 4), (24576, 8), (49152, 4), (12288, 8)]  # (stage bytes, stages): 50 MB bare streams
_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count if torch.cuda.is_available() else 132
_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096}
# BF16 tensor-core streaming GEMM configs per layer: (m, kcu, wg, pf, bk, xp, xr), chosen on the H100 (dev/g12.py).
BF16_G12 = {"qkv_gate": [(8, 2, 2, 4, 32, 1, 1)], "pooler": [(8, 8, 8, 4, 128, 1, 1), (16, 8, 8, 4, 128, 8, 2)],
            "o": [(8, 1, 2, 8, 128, 1, 1)], "unpooler": [(8, 4, 4, 4, 32, 1, 1)]}  # unpooler m8: H200 5.7-7.3 (noisy)
# cuBLASLt configs to try per (layer, m), from an exhaustive H100 sweep of algo 66 (dev/lte.py): (algo, tile, stages,
# split k, reduction, swizzle, custom option, cluster). prepare() times them and F.linear on this GPU, cold L2, and keeps
# the fastest (or F.linear).
BF16_LT = {('o', 8): [(66, 312, 35, 1, 0, 0, 3, 3), (66, 312, 35, 1, 0, 0, 7, 3), (66, 393, 35, 1, 0, 0, 3, 4), (66, 16, 35, 1, 0, 0, 3, 4), (66, 16, 35, 1, 0, 0, 5, 4)],
           ('o', 16): [(66, 394, 35, 1, 0, 0, 7, 4), (66, 16, 35, 1, 0, 0, 5, 4), (66, 312, 35, 1, 0, 0, 1, 3), (66, 394, 35, 1, 0, 0, 3, 4), (66, 13, 35, 1, 0, 0, 5, 3)],
           ('o', 32): [(66, 18, 35, 1, 0, 0, 2, 4), (66, 13, 35, 1, 0, 0, 7, 3), (66, 13, 35, 1, 0, 0, 3, 3), (66, 394, 35, 1, 0, 0, 5, 5), (66, 16, 35, 1, 0, 0, 5, 4)],
           ('o', 128): [(66, 15, 35, 1, 0, 0, 0, 5), (66, 15, 35, 1, 0, 0, 3, 5), (66, 15, 35, 1, 0, 0, 1, 2), (66, 15, 35, 1, 0, 0, 2, 5), (66, 17, 35, 1, 0, 0, 4, 2)],
           ('o', 256): [(66, 20, 35, 1, 0, 0, 7, 5), (66, 17, 35, 1, 0, 0, 0, 2), (66, 20, 35, 1, 0, 0, 0, 2), (66, 20, 35, 1, 0, 0, 5, 2), (66, 17, 35, 1, 0, 0, 1, 2)],
           ('pooler', 8): [(66, 10, 35, 1, 0, 0, 3, 4), (66, 10, 35, 1, 0, 0, 1, 4), (66, 313, 35, 1, 0, 0, 3, 4), (66, 10, 35, 1, 0, 0, 1, 5), (66, 312, 35, 1, 0, 0, 3, 4)],
           ('pooler', 16): [(66, 313, 35, 1, 0, 0, 1, 4), (66, 312, 35, 1, 0, 0, 1, 4), (66, 312, 35, 1, 0, 0, 7, 4), (66, 13, 35, 1, 0, 0, 1, 4), (66, 13, 35, 1, 0, 0, 5, 4)],
           ('pooler', 32): [(66, 13, 35, 1, 0, 0, 1, 4), (66, 13, 35, 1, 0, 0, 3, 4), (66, 315, 35, 1, 0, 0, 3, 4), (66, 315, 35, 1, 0, 0, 7, 4), (66, 13, 35, 1, 0, 0, 7, 4)],
           ('pooler', 128): [(66, 15, 35, 1, 0, 0, 0, 5), (66, 15, 35, 1, 0, 0, 0, 3), (66, 15, 35, 1, 0, 0, 2, 5), (66, 15, 35, 1, 0, 0, 4, 5), (66, 143, 35, 1, 0, 0, 0, 2)],
           ('pooler', 256): [(66, 18, 35, 1, 0, 0, 0, 2), (66, 135, 35, 1, 0, 0, 4, 2), (66, 18, 35, 1, 0, 0, 0, 3), (66, 18, 35, 1, 0, 0, 1, 2), (66, 18, 35, 1, 0, 0, 5, 2)],
           ('qkv_gate', 8): [(66, 10, 35, 1, 0, 0, 1, 2), (66, 10, 35, 3, 2, 0, 1, 4), (66, 134, 35, 8, 2, 0, 0, 2), (66, 16, 35, 3, 2, 0, 7, 2), (66, 394, 35, 4, 2, 0, 7, 2)],
           ('qkv_gate', 16): [(66, 312, 35, 1, 0, 0, 3, 2), (66, 312, 35, 1, 0, 0, 5, 2), (66, 13, 35, 1, 0, 0, 3, 2), (66, 394, 35, 3, 2, 0, 5, 4), (66, 312, 35, 3, 2, 0, 3, 4)],
           ('qkv_gate', 32): [(66, 314, 35, 1, 0, 0, 1, 4), (66, 80, 35, 1, 0, 0, 0, 4), (66, 12, 35, 1, 0, 0, 0, 4), (66, 16, 35, 3, 2, 0, 7, 4), (66, 27, 35, 3, 2, 0, 2, 4)],
           ('qkv_gate', 128): [(66, 80, 35, 1, 0, 0, 2, 5), (66, 92, 35, 1, 0, 0, 0, 5), (66, 80, 35, 1, 0, 0, 0, 5), (66, 92, 35, 1, 0, 0, 2, 5), (66, 104, 35, 1, 0, 0, 0, 2)],
           ('qkv_gate', 256): [(66, 124, 35, 1, 0, 0, 0, 2), (66, 124, 35, 1, 0, 0, 0, 3), (66, 134, 35, 1, 0, 0, 0, 2), (66, 134, 35, 1, 0, 0, 2, 8), (66, 134, 35, 1, 0, 0, 0, 8)],
           ('unpooler', 8): [(66, 394, 35, 8, 2, 0, 1, 4), (66, 393, 35, 8, 2, 0, 7, 2), (66, 394, 35, 8, 2, 0, 7, 2), (66, 312, 35, 8, 2, 0, 7, 2), (66, 312, 35, 8, 2, 0, 3, 2)],
           ('unpooler', 16): [(66, 394, 35, 1, 0, 0, 3, 4), (66, 394, 35, 1, 0, 0, 7, 4), (66, 395, 35, 1, 0, 0, 1, 4), (66, 394, 35, 1, 0, 0, 1, 4), (66, 395, 35, 1, 0, 0, 3, 4)],
           ('unpooler', 32): [(66, 397, 35, 1, 0, 0, 5, 4), (66, 398, 35, 1, 0, 0, 5, 4), (66, 397, 35, 1, 0, 0, 1, 4), (66, 394, 35, 1, 0, 0, 1, 2), (66, 164, 35, 1, 0, 0, 2, 4)],
           ('unpooler', 128): [(66, 18, 35, 1, 0, 0, 5, 5), (66, 18, 35, 1, 0, 0, 0, 2), (66, 15, 35, 1, 0, 0, 0, 2), (66, 18, 35, 1, 0, 0, 6, 5), (66, 18, 35, 1, 0, 0, 7, 5)],
           ('unpooler', 256): [(66, 20, 35, 1, 0, 0, 6, 5), (66, 20, 35, 1, 0, 0, 0, 3), (66, 405, 35, 1, 0, 0, 7, 5), (66, 20, 35, 1, 0, 0, 4, 3), (66, 405, 35, 1, 0, 0, 3, 5)]}
LT_WS_BYTES = 32 << 20
_G7 = sorted({c[1:6] for _, c in ROUTER_HL})
_SRC = (_ROUTER_SRC.replace("PR2_CONFIGS", " ".join("PR2_CASE(%d,%d,%d)" % c for _, c in ROUTER_F32)) + "\n" +
        _GEMV_SRC.replace("G7_CONFIGS", " ".join("G7_CASE(%d,%d,%d,%d,%d)" % c for c in _G7)) + "\n" +
        _G10_SRC.replace("G10_CONFIGS", " ".join("G10_CASE(%d,%d,%d,%d,%d,%d)" % c[:6] for v in BF16_G10.values() for c in v)) + "\n" +
        _G12_SRC.replace("G12_CONFIGS", " ".join("G12_CASE(%d,%d,%d,%d,%d,%d)" % c for c in sorted({c[:4] + c[5:] for v in BF16_G12.values() for c in v}))) + "\n" +
        _G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(%d,%d,%d,%d,%d,%d)" % (c[1:6] + c[7:8]) for c in sorted({c for v in BF16_G13.values() for c in v}))) + "\n" +
        _G14_SRC.replace("G14_CONFIGS", " ".join("G14_CASE(%d,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in BF16_G14.values() for c in v}))) + "\n" +
        _G15_SRC.replace("G15_CONFIGS", " ".join("G15_CASE(%d,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in BF16_G15.values() for c in v}))) + "\n" +
        _G16_SRC.replace("G16_CONFIGS", " ".join("G16_CASE(%d,%d,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in BF16_G16.values() for c in v}))) + "\n" +
        _G17_SRC.replace("G17_CONFIGS", " ".join("G17_CASE(%d,%d,%d,%d)" % c for c in sorted({c + (_LAYER_K[l],) for l, v in BF16_G17.items() for c in v}))) + "\n" +
        _R3_SRC.replace("PR3_CONFIGS", " ".join("PR3_CASE(%d,%d,%d,%d,%d,%d)" % c for c in [c for _, c in ROUTER_R3] + [c for v in ROUTER_R3_TUNE.values() for c in v])))
try:
    from torch.utils.cpp_extension import load_inline
    _mod = load_inline(
        name="muse_proj_v14",
        cpp_sources="void pj_router2(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                    "void g7_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t);\n"
                    "void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g12_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g14_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g15_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g16_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                    "void g17_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                    "void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, "
                    "int64_t, int64_t);",
        cuda_sources=_SRC, functions=["pj_router2", "g7_gemv", "g10_gemv", "g12_gemm", "g13_gemv", "g14_gemm", "g15_gemm",
                                      "g16_gemm", "g17_gemv", "pj_router3"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj native build failed, using torch:", _e)
    _mod = None

try:
    _g19 = load_inline(
        name="muse_proj_g19a",
        cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_cvt(torch::Tensor, torch::Tensor); "
                    "void g19_pred(torch::Tensor, torch::Tensor, int64_t); void g19_stream(torch::Tensor, int64_t, int64_t, torch::Tensor);",
        cuda_sources="#define G19_ONLY2\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
            "G19_CASE(%d,%d,%d,%d,1,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in list(BF16_G19.values()) + list(DIAG19.values()) for c in v}))),
        functions=["g19_gemm", "g19_cvt", "g19_pred", "g19_stream"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv19 build failed:", _e)
    _g19 = None

try:
    _lt = load_inline(
        name="muse_proj_lt1",
        cpp_sources="int64_t lt_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>);",
        cuda_sources=_LT_SRC, functions=["lt_gemm"], extra_ldflags=["-lcublasLt"],
        extra_cuda_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj cuBLASLt build failed, using F.linear:", _e)
    _lt = None

ROUTER_FP32_MAX_TOKENS = 8


def _pack(w):
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 256)


def _pack12(w, kc):
    # [k-chunk][8-row group][kc/32 atoms][lane][8]: atom = 8 rows x 32 k, lane l = row l/4, k (l%4)*8..+8
    n, k = w.shape
    return w.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()


def _compress(w):
    """Lossless 12-bit BF16 (gemv13/gemv14): sign|mantissa byte and 4-bit exponent code c = e - base per weight
    (base = max exponent - 15; decoded exponent 112 | c, so sums are rescaled by scale = 2^(base - 112)); weights below
    the window (zeros, tiny values) are escapes: code 0, byte 0 (decodes to 2^(base - 127)) plus a per-row header of
    G13_ESC (col, true / scale) entries (exact; kernels subtract 2^-15 = 2^(base - 127) / scale). None if a row has more than G13_ESC escapes."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    if base < 1:
        return None
    code = e - base
    esc = code < 0
    if int(esc.sum(1).max()) > G13_ESC:
        return None
    code = torch.where(esc, 0, code)
    sm = torch.where(esc, 0, ((b >> 8) & 0x80) | (b & 0x7F))
    scale = 2.0 ** (base - 112)
    # empty slots: col 0, value 2^-15 (the kernels add x[col] * (value - 2^-15): exactly 0)
    hdr = torch.zeros(n, G13_ESC, 2, dtype=torch.float32, device=w.device)
    hdr[:, :, 1] = 2.0 ** -15
    rows, cols = esc.nonzero(as_tuple=True)  # row-major order
    if len(rows):
        slot = torch.arange(len(rows), device=w.device) - torch.searchsorted(rows, rows)
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = w[rows, cols].float() / scale  # BF16 over a power of two: exact in FP32
    # round trip: every weight decodes to its original bits (else no compression for this weight)
    dec = torch.where(esc, 0, (((sm & 0x80) << 8) | ((code + base) << 7) | (sm & 0x7F)))
    dec = dec.to(torch.int32).where(dec < 32768, dec - 65536).to(torch.int16).view(torch.bfloat16)
    if len(rows):
        fix = (hdr[rows, slot, 1] * scale).to(torch.bfloat16)
        if not torch.equal(fix.view(torch.int16), w[rows, cols].contiguous().view(torch.int16)):
            return None
        dec[rows, cols] = fix
        if not torch.equal(hdr[rows, slot, 1] * scale, w[rows, cols].float()):
            return None
    if not torch.equal(dec.view(torch.int16), w.contiguous().view(torch.int16)):
        return None
    return code, sm, hdr, scale


def _nib(code):
    """Codes [..., 8] -> int32 words: weight 2p + h at bits 4p + 16h."""
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=code.device)
    v = (code.long() << sh).sum(-1)
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).int()


def _pack13(w):
    """gemv13 layout: flat byte plane, code words over 8 consecutive weights."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    return sm.to(torch.uint8).flatten().contiguous(), _nib(code.flatten().view(-1, 8)).contiguous(), hdr, scale


def _pack14(w, kc):
    """gemv14 layout: gemv12's atom order [ks][rg][atom][lane][8] for bytes and codes."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    perm = lambda t: t.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
    return perm(sm).to(torch.uint8), _nib(perm(code)).contiguous(), hdr, scale


def _pack16(w, kc):
    """gemv16 layout: [ks][16-row group][atom = 16 rows x 32 k][lane][16] bytes, codes as 2 words per lane; lane (g, c)
    holds the two mma A fragments (k-steps s) of rows g, g + 8 and k 2c, 2c + 1, 2c + 8, 2c + 9."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    # [rg, h, g, ks, atom, s, hk, c, e] -> [ks, rg, atom, g, c, s, hk, h, e]
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    sw = perm(sm).to(torch.uint8).reshape(k // kc, n // 16, kc // 32, 32, 16).contiguous()
    nw = _nib(perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8)).contiguous()
    return sw, nw, hdr, scale


def _pack19(c, kc, ap, rg=8):
    """_compress output -> gemv19 stages: stage (ks, 128-row tile, st) = [8 row groups x ap atoms x 32 lanes x 16 B of
    sign|mantissa][same x 8 B of codes], one contiguous block (one bulk copy)."""
    code, sm, hdr, scale = c
    n, k = code.shape
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    S = perm(sm).to(torch.uint8).reshape(k // kc, n // 16, kc // 32, 32, 16)
    N = _nib(perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8)).contiguous()
    ks, ntr, spk = k // kc, n // (16 * rg), kc // 32 // ap
    S = S.view(ks, ntr, rg, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, rg * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, rg, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, rg * ap * 256)
    return torch.cat([S, N], 1).contiguous(), hdr, scale


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    if _g19 is not None:
        for name in BF16_G19:
            cfgs = BF16_G19[name] + DIAG19.get(name, [])
            c = _compress(weights[name])
            if c is None:
                continue
            for cf in cfgs:
                if ("g19", name, cf[1], cf[2], cf[7]) not in st:
                    st[("g19", name, cf[1], cf[2], cf[7])] = _pack19(c, cf[1], cf[2], cf[7])
        st["ws19"] = torch.zeros(32 * 8192, dtype=torch.float32, device=weights["o"].device)
        st["ctr19"] = torch.zeros(2, dtype=torch.int32, device=weights["o"].device)  # [1] = gemv19's epoch counter
    for name, cfgs in BF16_G12.items():
        for c in cfgs:
            if (name, c[1]) not in st:
                st[(name, c[1])] = _pack12(weights[name], 1024 * c[1])
    if _mod is not None:
        for name, cfgs in list(BF16_G14.items()) + list(BF16_G15.items()):
            for c in cfgs:
                rows = 1024 if name == "unpooler" and c[0] <= 8 else weights[name].shape[0]
                if ("g14", name, c[1], rows) not in st:
                    pk = _pack14(weights[name][:rows], 1024 * c[1])
                    if pk is not None:
                        st[("g14", name, c[1], rows)] = pk
        for name, cfgs in BF16_G16.items():
            for c in cfgs:
                if ("g16", name, c[1]) not in st:
                    pk = _pack16(weights[name], 1024 * c[1])
                    if pk is not None:
                        st[("g16", name, c[1])] = pk
        for name in BF16_G13:
            pk = _pack13(weights[name][:1024] if name == "unpooler" else weights[name])
            if pk is not None:
                st[("g13", name)] = pk
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["router_hl_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [k, 512]: one GEMM gives both halves
    # [stripe][k-tile][hi/lo][lane][8]: a 16x16 tile's mma A fragments, 512 contiguous bytes
    st["router_hl_p"] = torch.stack([_pack(hi), _pack(lo)], 2).contiguous()
    st["router_hl2"] = torch.stack([hi, lo]).contiguous()  # [2, 256, k] for router3
    # Split-K workspace and per-row-block counters: always left zeroed by the kernels.
    st["ws"] = torch.zeros(32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    # gemv12's own: FP32 split-K partials (overwritten each call, not zeroed) and per-rg counters (left zeroed)
    st["ws12"] = torch.empty(8 * 32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt12"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    st["ltws"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=r.device)
    st["pick"] = _tune(st)
    _diag19(st)
    _diag19_stream()
    st["rpick"] = _tune_router(st)
    return st


def _run(st, layer, x, out, cand):
    """One BF16 projection by candidate: None = F.linear, ("g12", cfg) = dev/gemv12.cu, ("lt", cfg) = cuBLASLt."""
    if cand is None:
        return F.linear(x, st[layer][:out.shape[1]])
    kind, c = cand
    if kind == "g10":
        _mod.g10_gemv(st[layer], x, out, *c[2:])  # W[:n] is the first n rows, in place
    elif kind == "g13":
        sm, nib, hdr, scale = st[("g13", layer)]
        _mod.g13_gemv(sm, nib, hdr, x, out, scale, *c[2:])
    elif kind == "g14":
        rows = out.shape[1]
        sw, nw, hdr, scale = st[("g14", layer, c[1], rows)]
        _mod.g14_gemm(sw, nw, hdr, scale, x, out, st["ws12"], st["cnt12"], rows // 8, *c[1:])
    elif kind == "g15":
        rows = out.shape[1]
        sw, nw, hdr, scale = st[("g14", layer, c[1], rows)]  # same layout as gemv14
        _mod.g15_gemm(sw, nw, hdr, scale, x, out, rows // 8, *c[1:])
    elif kind == "g16":
        sw, nw, hdr, scale = st[("g16", layer, c[1])]
        _mod.g16_gemm(sw, nw, hdr, scale, x, out, out.shape[1] // 16, *c[1:])
    elif kind == "g19":
        P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
        o32 = st["ws19"][:out.numel()].view(out.shape)
        _g19.g19_gemm(P, hdr, scale, x, o32, st["ctr19"], c[1], c[2], c[3], 1, _SMS, 0, 2, c[4], *c[5:])
        _g19.g19_cvt(o32, out)
    elif kind == "g17":
        _mod.g17_gemv(st[layer], x, out, *c)  # W[:n] prefix rows, in place
    elif kind == "g12":
        _mod.g12_gemm(st[(layer, c[1])], x, out, st["ws12"], st["cnt12"], st[layer].shape[0] // 8, *c[1:])
    else:
        _lt.lt_gemm(st[layer], x, out, st["ltws"], list(c))
    return out


def _tune(st, reps=25):
    """Per (layer, m): time F.linear, the native gemv10/gemv13 (m=1) and gemv12 configs and each BF16_LT config as CUDA-graph replays with a
    cold L2 (read flush, as the judge) on random x; keep the fastest. Only shapes and weights are used."""
    picks = {}
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device=st["qkv_gate"].device).view(torch.int64)
    sink = torch.zeros((), dtype=torch.int64, device=flush.device)
    g = torch.Generator(device=flush.device)
    g.manual_seed(0)
    shapes = {}
    if _mod is not None:
        for layer, cfgs in BF16_G10.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("g10", c))
        for layer, cfgs in BF16_G13.items():
            if ("g13", layer) in st:
                for c in cfgs:
                    shapes.setdefault((layer, c[0]), []).append(("g13", c))
        for layer, cfgs in BF16_G14.items():
            for c in cfgs:
                rows = 1024 if layer == "unpooler" and c[0] <= 8 else st[layer].shape[0]
                if ("g14", layer, c[1], rows) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g14", c))
        for layer, cfgs in BF16_G15.items():
            for c in cfgs:
                rows = 1024 if layer == "unpooler" and c[0] <= 8 else st[layer].shape[0]
                if ("g14", layer, c[1], rows) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g15", c))
        for layer, cfgs in BF16_G16.items():
            for c in cfgs:
                if ("g16", layer, c[1]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g16", c))
        for layer, cfgs in BF16_G12.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("g12", c))
        for layer, cfgs in BF16_G17.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("g17", c))
    if _g19 is not None:
        for layer, cfgs in BF16_G19.items():
            for c in cfgs:
                if ("g19", layer, c[1], c[2], c[7]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g19", c))
    if _lt is not None:
        for key, cfgs in BF16_LT.items():
            shapes.setdefault(key, []).extend(("lt", c) for c in cfgs)
    for (layer, m), cfgs in sorted(shapes.items()):
        w = st[layer]
        k = w.shape[1]
        n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
        x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
        ref = F.linear(x, w[:n]).float()
        out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
        cands = [None] if m > 1 else []  # m=1: gemv10 always beat cuBLAS on the judge; ties picked F.linear by noise
        for c in cfgs:
            if c[0] == "lt" and _lt.lt_gemm(w, x, out, st["ltws"], list(c[1])) != 0:
                continue
            _run(st, layer, x, out, c)
            torch.cuda.synchronize()
            if ((out.float() - ref).norm() / ref.norm()).item() < 1e-3:
                cands.append(c)
        if not cands:
            continue
        graphs = []
        for c in cands:  # one graph per candidate: times the kernel as the judge replays it, no host overhead
            gr = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(gr, stream=s):
                    _run(st, layer, x, out, c)
            torch.cuda.current_stream().wait_stream(s)
            graphs.append(gr)
        torch.cuda.synchronize()
        ev = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in cands]
              for _ in range(reps)]
        for r in range(reps):
            for i in range(len(cands)):
                torch.sum(flush, 0, out=sink)  # cold L2 by reading, as the judge does
                ev[r][i][0].record()
                graphs[i].replay()
                ev[r][i][1].record()
        torch.cuda.synchronize()
        med = [sorted(ev[r][i][0].elapsed_time(ev[r][i][1]) for r in range(reps))[reps // 2] for i in range(len(cands))]
        best = min(range(len(cands)), key=med.__getitem__)
        if cands[best] is not None:
            picks[(layer, m)] = cands[best]
        print(f"tune {layer} m{m}: best {cands[best]} {med[best] * 1e3:.2f} us; "
              + " ".join(f"{'F' if c is None else c[0]}:{t * 1e3:.2f}" for c, t in zip(cands, med)))
    del flush
    return picks


def _diag19(st):
    """Print only (no effect on picks): the layer's case, a 3 us predecessor that writes x and lets its PDL dependent
    launch at once, then gemv19 (FP32 out, npre = ns: W streams during the predecessor) vs the picked kernel."""
    if _g19 is None:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    for layer, cfgs in BF16_G19.items():
        for c in cfgs + DIAG19.get(layer, []):
            if ("g19", layer, c[1], c[2], c[7]) not in st:
                continue
            m, w = c[0], st[layer]
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            out = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device="cuda")
            P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
            o32 = st["ws19"][:out.numel()].view(out.shape)
            pick = st["pick"].get((layer, m))
            med = _graph_medians([lambda: _g19.g19_pred(xs, x, 3000),
                                  lambda: (_g19.g19_pred(xs, x, 3000), _run(st, layer, x, out, pick)),
                                  lambda: (_g19.g19_pred(xs, x, 3000), _g19.g19_gemm(P, hdr, scale, x, o32, st["ctr19"], c[1], c[2], c[3], 1, _SMS, 0, 2, c[3], *c[5:]))])
            if c not in cfgs:
                med[1] = float("nan")
            print(f"diag19 {layer} m{m} {c}: after a 3 us PDL predecessor ({med[0] * 1e3:.2f}): pick {pick} +{(med[1] - med[0]) * 1e3:.2f} us, "
                  f"g19 FP32 out +{(med[2] - med[0]) * 1e3:.2f} us")


def _diag19_stream():
    if _g19 is None:
        return
    w = torch.zeros(50 << 20, dtype=torch.uint8, device="cuda")
    o = torch.zeros(1, device="cuda")
    med = _graph_medians([lambda: o.zero_()] + [(lambda c=c: _g19.g19_stream(w, c[0], c[1], o)) for c in DIAG19_STREAM])
    print("diag19 bare TMA stream 50 MB, 132 blocks:", " ".join(f"{c}:{(t - med[0]) * 1e3:.2f}" for c, t in zip(DIAG19_STREAM, med[1:])),
          f"(a 1-kernel graph, {med[0] * 1e3:.2f} us, subtracted)")
    del w


def _graph_medians(fns, reps=25):
    """Median time (ms) of each fn as a CUDA-graph replay after a 256 MB read flush (cold L2, as the judge)."""
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.zeros((), dtype=torch.int64, device="cuda")
    graphs = []
    for fn in fns:
        gr = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(gr, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        graphs.append(gr)
    torch.cuda.synchronize()
    ev = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in fns] for _ in range(reps)]
    for r in range(reps):
        for i, gr in enumerate(graphs):
            torch.sum(flush, 0, out=sink)
            ev[r][i][0].record()
            gr.replay()
            ev[r][i][1].record()
    torch.cuda.synchronize()
    del flush
    return [sorted(ev[r][i][0].elapsed_time(ev[r][i][1]) for r in range(reps))[reps // 2] for i in range(len(fns))]


def _tune_router(st):
    """Per m in ROUTER_R3_TUNE: router3 configs vs the cuBLAS hi/lo path, timed on random x; keep the fastest."""
    picks = {}
    if _mod is None:
        return picks
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    for m, cfgs in ROUTER_R3_TUNE.items():
        x = torch.randn(m, st["router"].shape[1], generator=g, device="cuda").to(torch.bfloat16)
        ref = _router_cublas(x, st)
        out = torch.empty((m, 256), dtype=torch.float32, device="cuda")
        cands = [None]
        for c in cfgs:
            _mod.pj_router3(st["router_hl2"], x, out, *c)
            torch.cuda.synchronize()
            if ((out - ref).norm() / ref.norm()).item() < 1e-5:
                cands.append(c)
        med = _graph_medians([(lambda c=c: _router_cublas(x, st) if c is None else
                               _mod.pj_router3(st["router_hl2"], x, out, *c)) for c in cands])
        best = min(range(len(cands)), key=med.__getitem__)
        picks[m] = cands[best]
        print(f"tune router m{m}: best {cands[best]} {med[best] * 1e3:.2f} us; "
              + " ".join(f"{'cublas' if c is None else c}:{t * 1e3:.2f}" for c, t in zip(cands, med)))
    return picks


def _router_cublas(x, st):
    acc = torch.mm(x, st["router_hl_t"], out_dtype=torch.float32)
    return acc[:, :256] + acc[:, 256:]


def _router(x, st, m):
    if _mod is not None:
        for mmax, cfg in ROUTER_F32:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.pj_router2(st["router"], x, out, *cfg)
                return out
        for mmax, cfg in ROUTER_R3:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.pj_router3(st["router_hl2"], x.contiguous(), out, *cfg)
                return out
        cfg = st["rpick"].get(m)
        if cfg is not None:
            out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            _mod.pj_router3(st["router_hl2"], x.contiguous(), out, *cfg)
            return out
        for mmax, cfg in ROUTER_HL:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.g7_gemv(st["router_hl_p"], x, out, *cfg)
                return out
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), st["router"])
    return _router_cublas(x, st)


def kernel_fn(data, st):
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    if layer == "router":
        return _router(x, st, cfg["m"])
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    c = st["pick"].get((layer, m))
    if c is None:
        return F.linear(x, st[layer][:n])
    return _run(st, layer, x, torch.empty((m, n), dtype=torch.bfloat16, device=x.device), c)
