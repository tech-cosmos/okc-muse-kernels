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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
// 4 weights (bytes of sw) -> 2 BF16 pairs; nibble word nw, pairs P0, P0 + 1
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = decw<P0>(sw, nw, 0x9180u, 0x4140u);
    const unsigned q1 = decw<P0 + 1>(sw, nw, 0xB3A2u, 0x4342u);
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
#ifdef G13_CPA
    // cp.async ring: PF chunks in flight in smem (each thread reads back only its own 16 + 8 B per u: no sync)
    extern __shared__ __align__(16) unsigned char dsm[];
    uint4* sS = reinterpret_cast<uint4*>(dsm);                  // [PF][U][VT * NV]
    uint2* sN = reinterpret_cast<uint2*>(sS + PF * U * VT * NV);  // [PF][U][VT * NV]
    auto load = [&](int p, int ch) {
        if (ch < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const long i = (long)ch * (CE / 16) + u * VT + t;
                const unsigned ds = (unsigned)__cvta_generic_to_shared(sS + (p * U + u) * VT * NV + threadIdx.x);
                const unsigned dn = (unsigned)__cvta_generic_to_shared(sN + (p * U + u) * VT * NV + threadIdx.x);
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" :: "r"(ds), "l"(S + i) : "memory");
                asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 8;" :: "r"(dn), "l"(N + i) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
#else
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
#endif
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
#ifdef G13_CPA
    for (int p = 0; p < PF - 1; ++p) load(p, c + p * G);
#else
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
#endif
#ifdef G13_L2PF
    // L2 bulk prefetch of this vb's later chunks (no registers): HBM streams at full rate from the start
    if (t == 0)
        for (int cc = c + PF * G, d = 0; cc < nch && d < G13_L2PF; cc += G, ++d) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(S + (long)cc * (CE / 16)), "r"(CE) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(N + (long)cc * (CE / 16)), "r"(CE / 2) : "memory");
        }
#endif
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
#ifdef G13_CPA
    for (int i = 0; c < nch; c += G, ++i) {
        load((i + PF - 1) % PF, c + (PF - 1) * G);  // the slot chunk i - 1 used
        asm volatile("cp.async.wait_group %0;" :: "n"(PF - 1) : "memory");
        const int p = i % PF, cc = c;
        {
            {
                float acc[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const uint4 sv = sS[(p * U + u) * VT * NV + threadIdx.x];
                    const uint2 nv = sN[(p * U + u) * VT * NV + threadIdx.x];
                    acc[u] = dot16(sv, nv, xs + XI((u * VT * 16 + t * 16) % KK));
                }
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
                    const float2* he = hs_[vg] + (i * RPC + t) * ESC;
                    for (int e = 0; e < ESC; ++e) {
                        const float2 h = he[e];
                        if (h.y == 0x1p-15f) break;
                        const float xe = xs[XI(__float_as_int(h.x))];
                        sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
                    }
                    out[(long)cc * RPC + t] = __float2bfloat16(sum * scale);
                }
                par ^= 1;
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
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
#endif
}
}  // namespace g13

#ifdef G13_CPA
#define G13_DSM(U_, V_, P_, VT_) ((P_) * (U_) * (VT_) * (V_) * 24)
#else
#define G13_DSM(U_, V_, P_, VT_) 0
#endif
void g13_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
              int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks, int64_t vt) {
    const int k = x.size(1), n = out.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(x.size(0) == 1, "g13: m == 1");
    TORCH_CHECK((long)n * k <= (long)g13::MAXC * blocks * nv * u * vt * 16, "g13: too many chunks per virtual block");
#define G13_CASE(K_, U_, V_, B_, P_, VT_) \
    if (k == K_ && u == U_ && nv == V_ && minb == B_ && pf == P_ && vt == VT_) { \
        const int dsm_ = G13_DSM(U_, V_, P_, VT_); \
        if (dsm_) cudaFuncSetAttribute(g13::gemv<U_, V_, B_, K_, P_, VT_>, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm_); \
        g13::gemv<U_, V_, B_, K_, P_, VT_><<<blocks, VT_ * V_, dsm_, st>>>((const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), \
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(uint2 s, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s.x, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s.x, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s.y, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s.y, nw, 0xB3A2u, 0x4342u);
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
#ifdef G14_BPF  // s36 (as pr8 npf 2xx): bulk L2 prefetch of atoms [PF, PF + G14_BPF) of each row group (WS, WN)
    if (t < 2 * NG) {
        const int nb = min(G14_BPF, NA - PF);
        const long o0 = (((long)ks * nrg_all + rg0 + (t >> 1)) * (KC / 32) + PF * WG) * 32;
        const char* src = (t & 1) ? (const char*)(WN + o0) : (const char*)(WS + o0);
        if (nb > 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"((unsigned)nb * WG * 32 * ((t & 1) ? 4 : 8)) : "memory");
    }
#endif
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(uint2 s, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s.x, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s.x, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s.y, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s.y, nw, 0xB3A2u, 0x4342u);
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
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
#ifdef G19_RAW  // s24: plain BF16 weights (stages = mma A fragments, [atom][s][lane][16 B]): no decode, no escapes, scale 1
constexpr int WB = 1024;
#else
constexpr int WB = 768;  // bytes per 16-row x 32-k atom: 512 sign|mantissa + 256 codes
#endif
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
#ifdef G19_MMANV  // not volatile: a pure register op, free to be scheduled among the decode / ldmatrix work
    asm(
#else
    asm volatile(
#endif
                 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
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
#ifdef G19_SPIN  // diag: non-suspending test_wait spin instead of try_wait
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary), ctr int (zeroed by the primary). tail: stages left for dynamic items.
// smem: ring [NS][SB] | x [2][M][XS] | pbuf [2][128 rows][M] f32 | barriers | tag [NS] int | ptag [2] int
#ifdef G19_NOFIXB  // diag: headers loaded and tested, but no slot is in range (opaque): the fixups' own share
#define KHI_B (khi - (int)(gridDim.z << 30))
#else
#define KHI_B khi
#endif
#ifndef G19_NPB
#define G19_NPB 2
#endif
constexpr int NPB = G19_NPB;  // partial buffers (1: less smem, for 2 blocks per SM)
#ifndef G19_MINB
#define G19_MINB 1
#endif
template <int M, int KC, int AP, int NS, int IS, int PRE, int XB, int TS, int RG, int RPW>
__global__ void __launch_bounds__((RG * TS / RPW + 2) * 32, G19_MINB) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                                         float scale, const __nv_bfloat16* __restrict__ x,
                                                         float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre,
                                                         __nv_bfloat16* __restrict__ ob, int tm) {
    // PRE 3: BF16 output ob [M, n]; out = FP32 workspace that is ZERO on entry and left zero (the finisher that completes
    // a tile's K reads it back, writes BF16, re-zeroes it); ctr[2 + tile] = atoms done per tile (+KA per call, never reset)
    constexpr int NWC = RG;  // 16-row groups per tile (one stage covers RG x AP atoms)
    static_assert(RPW == 1 || TS == 1, "config");
    constexpr int NCW = NWC * TS / RPW, MT = M / TS, NT = MT / 8, NA = KC / 32, SPK = NA / AP, SB = NWC * AP * WB, XS = KC * 2 + 16, PBF = NWC * 16 * M;
    static_assert(NA % AP == 0 && SPK % IS == 0 && (NT % 2 == 0 || NT == 1) && M <= 32, "config");  // NT even: x B fragments by ldmatrix.x4 (NT 1: one x4 = both k16 halves)
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + XB * M * XS);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + NPB * PBF);
#ifdef G19_XQ  // x chunk in 4 quarters, one mbarrier each (xq[buffer][quarter], after the other barriers): stages start on their quarter
#ifndef G19_XPAR
#error "G19_XQ needs G19_XPAR"
#endif
    constexpr int XQB = 16;
#else
    constexpr int XQB = 0;
#endif
    int* tag = (int*)(pbuf + NPB * PBF) + 2 * (2 * NS + 8) + XQB;  // after the barriers. per slot: u | xb << 29 | last x use << 30 | run end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile | atoms << 12 (-1: consumers done)
    // PRE 3: tile read-back buffer [M][NWC * 16] f32 (bulk copy) + its mbarrier
    const unsigned cbuf = ((unsigned)__cvta_generic_to_shared(ptag + 2) + 127u) & ~127u, cbar = cbuf + M * NWC * 64;
    // full[NS], empty[NS], xfull[2], xempty[2], pfull[2], pempty[2]
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48, xq = xfull + 64;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NWC, S = k / KC, KA = k / 32;
    const int U = S * ntr * SPK, US = U - tail;
    // tm: tile-major stage order (pack19 tm): stage u = (tile, k-chunk, st): tiles complete throughout the kernel
    // (mode 3: their BF16 conversions overlap the stream instead of all landing in the tail)
    auto KSof = [&](int u_) { return tm ? (u_ / SPK) % S : u_ / (ntr * SPK); };
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NCW); }
        for (int i = 0; i < 2; ++i) {
            bar_init(xfull + 8 * i, 1); bar_init(xempty + 8 * i, NCW); bar_init(pfull + 8 * i, NCW); bar_init(pempty + 8 * i, 1);
        }
        if (XQB) for (int i = 0; i < 8; ++i) bar_init(xq + 8 * i, 1);
        if (PRE == 3) bar_init(cbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
#ifdef G19_TRACE
    if (t == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); trace[blockIdx.x * 64 + 10] = sm_; }
#endif
#ifdef G19_XPAR
    if (w == NCW) {  // producer warp, all lanes in lockstep: lane 0 issues W / tags / barriers, x rows go one per lane
        int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
        int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
        auto xload = [&](int b_, int ks_) {
#ifdef G19_XQ
            if (lane == 0) for (int q4 = 0; q4 < 4; ++q4) bar_expect(xq + 8 * (b_ * 4 + q4), M * KC / 2);
            __syncwarp();
            for (int i4 = lane; i4 < 4 * M; i4 += 32) {  // quarter-major: quarter 0 of every row first
                const int q4 = i4 / M, qq = i4 % M;
                bulk(xs + b_ * M * XS + qq * XS + q4 * (KC / 2), x + (long)qq * k + (long)ks_ * KC + q4 * (KC / 4), KC / 2, xq + 8 * (b_ * 4 + q4));
            }
#else
            if (lane == 0) bar_expect(xfull + 8 * b_, M * KC * 2);
            __syncwarp();
            for (int qq = lane; qq < M; qq += 32) bulk(xs + b_ * M * XS + qq * XS, x + (long)qq * k + (long)ks_ * KC, KC * 2, xfull + 8 * b_);
#endif
        };
        if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
            const int ks0 = KSof(ra);
            for (; i < npre && ra < rb && KSof(ra) == ks0; ++i, ++ra) {
                const int st = ra % SPK;
                const bool xl = ra + 1 == rb || KSof(ra + 1) != ks0;
                if (lane == 0) {
                    tag[i] = ra | (xl ? 1 << 30 : 0) | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
                    bar_expect(bars + 8 * i, SB);
                    bulk(ring + i * SB, WP + (long)ra * SB, SB, bars + 8 * i);
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            xb = 0; xks[0] = ks0; xuse[0] = 1;
            xload(0, ks0);
        } else {
            asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
        }
        for (;;) {
            if (ra >= rb) {  // next run: a tail item
                if (tail == 0) break;
                int it = 0;
                if (lane == 0) it = atomicAdd(ctr, 1);
                it = __shfl_sync(0xffffffffu, it, 0);
                if (it * IS >= tail) break;
                ra = US + it * IS; rb = ra + IS;
            }
            for (int u = ra; u < rb; ++u, ++i) {
                const int ks = KSof(u), st = u % SPK;
                if (ks != xks[xb]) {  // next x buffer, load chunk ks (consumers release a buffer after its last use)
                    xb = XB == 2 ? xb ^ 1 : 0;
                    if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                    ++xuse[xb];
                    xks[xb] = ks;
                    xload(xb, ks);
                }
                const int slot = i % NS;
                if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                const bool end = u + 1 == rb || st + 1 == SPK;
                const bool xl = u + 1 == rb || KSof(u + 1) != ks;
                if (xl) xks[xb] = -1;  // released by the consumers after this stage: reload even if the next run has this ks
                if (lane == 0) {
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    if (i < 24) TR(40 + i);
                    const unsigned full = bars + 8 * slot;
#ifdef G19_NOW
                    bar_arrive(full);
#else
                    bar_expect(full, SB);
                    bulk(ring + slot * SB, WP + (long)u * SB, SB, full);
#endif
                }
            }
            ra = rb;
        }
        const int slot = i % NS;  // end marker
        if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
        if (lane == 0) {
            tag[slot] = -1;
            bar_arrive(bars + 8 * slot);
            TR(1);
        }
        return;
    }
#else
    if (w == NCW) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
                const int ks0 = KSof(ra);
                for (; i < npre && ra < rb && KSof(ra) == ks0; ++i, ++ra) {
                    const int st = ra % SPK;
                    const bool xl = ra + 1 == rb || KSof(ra + 1) != ks0;
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
                    const int ks = KSof(u), st = u % SPK;
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
                    const bool xl = u + 1 == rb || KSof(u + 1) != ks;
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
#endif
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
        int ncv = 0;
        for (int q = 0;; ++q) {
            const int pb = q % NPB;
            bar_wait(pfull + 8 * pb, (q / NPB) & 1);
            const int ptg = ptag[pb];
            if (ptg < 0) break;
            const int tr = ptg & 0xFFF;
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
            if (PRE == 3) __threadfence();  // this lane's reds before the tile ticket
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
            if (lane == 0 && q < 2) TR(2 + q);
            if (PRE == 3) {
                int last = 0;
                if (lane == 0) {
                    const int na = ptg >> 12, old = atomicAdd(ctr + 2 + tr, na);
                    last = (old + na) % KA == 0;
                }
                if (__shfl_sync(0xffffffffu, last, 0)) {  // all of the tile's K is in: FP32 -> BF16, re-zero
                    // one bulk copy per token row (512 B for RG 8): a single round trip instead of a chain of loads
                    if (lane == 0) {
                        __threadfence();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        bar_expect(cbar, M * NWC * 64);
                        for (int tok = 0; tok < M; ++tok) bulk(cbuf + tok * NWC * 64, wt + (long)tok * n, NWC * 64, cbar);
                    }
                    bar_wait(cbar, ncv & 1);
                    ++ncv;
                    const float4* cb = (const float4*)(smem + (cbuf - (unsigned)__cvta_generic_to_shared(smem)));
#pragma unroll 4
                    for (int i = lane; i < M * NWC * 4; i += 32) {
                        const int tok = i / (NWC * 4), r4 = (i % (NWC * 4)) * 4;
                        const float4 a = cb[i];
                        __stcg((float4*)(wt + (long)tok * n + r4), make_float4(0.f, 0.f, 0.f, 0.f));
                        const __nv_bfloat162 b0 = __floats2bfloat162_rn(a.x, a.y), b1 = __floats2bfloat162_rn(a.z, a.w);
                        *(uint2*)(ob + (long)tok * n + (long)tr * NWC * 16 + r4) = make_uint2(*(const unsigned*)&b0, *(const unsigned*)&b1);
                    }
                    __syncwarp();  // cbuf reads done before the next bulk copy
                }
            }
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (t == 0) TR(9);
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
    // NT 1 (M 8): matrix q = (k16 step s = q / 2, k half q % 2) of the single n-tile
    const unsigned xrow = NT == 1 ? (th * MT + (lane & 7)) * XS + ((lane >> 3) & 1) * 16 + (lane >> 4) * 32
                                  : (th * MT + (lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    int q = 0, st0 = 0, prev_end = 1, prev_xl = 1;
    unsigned xph[2] = {0, 0}, xpc[2] = {0, 0};
    float4 hd0[RPW], hd1[RPW];
#ifdef G19_HPF  // s48: the next run's headers (predicted: the static order's next run) are loaded one run ahead
    float4 hn0[RPW], hn1[RPW];
    int htr = -1;
#endif
    for (int i = 0;; ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const int tg = tag[slot];
        if (t == 0 && i < 24) TR(16 + i);
        if (tg == -1) break;
        const int u = tg & 0x1FFFFFFF, xb = (tg >> 29) & 1;
        const bool end = tg < 0, xl = (tg >> 30) & 1;
        const int ks = KSof(u), tr = tm ? u / (S * SPK) : (u % (ntr * SPK)) / SPK, st = u % SPK;
#ifdef G19_XQ
        if (prev_xl) { xpc[xb] = xph[xb]; xph[xb] ^= 1; }  // a new x use of buffer xb: its quarters' phase parity
        prev_xl = xl;
        for (int q4 = (st * AP * 32) / (KC / 4); q4 <= ((st + 1) * AP * 32 - 1) / (KC / 4); ++q4) bar_wait(xq + 8 * (xb * 4 + q4), xpc[xb]);
        if (false) {
#else
        if (prev_xl) {  // first stage of an x use: wait for it
#endif
#ifndef G19_NOXW  // diag: never wait for x (garbage results): the x arrival's share of the time
            bar_wait(xfull + 8 * xb, xph[xb]);
#endif
            if (t == 0 && i == 0) TR(5);
            xph[xb] ^= 1;
        }
        prev_xl = xl;
#ifndef G19_RAW
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row groups; lane c: slots 2c, 2c + 1)
            st0 = st;
#ifdef G19_HPF
            const bool hit = tr == htr;
            const int un = (u / SPK + 1) * SPK, trn = tm ? un / (S * SPK) : (un % (ntr * SPK)) / SPK;
            htr = un < U ? trn : -1;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) {
                if (hit) { hd0[rr] = hn0[rr]; hd1[rr] = hn1[rr]; }
                else {
                    const float2* hp0 = H + (long)((tr * NWC + wr + rr) * 16 + g) * ESC;
                    hd0[rr] = __ldg((const float4*)hp0 + c);
                    hd1[rr] = __ldg((const float4*)(hp0 + 8 * ESC) + c);
                }
                if (htr >= 0) {
                    const float2* hq0 = H + (long)((htr * NWC + wr + rr) * 16 + g) * ESC;
                    hn0[rr] = __ldg((const float4*)hq0 + c);
                    hn1[rr] = __ldg((const float4*)(hq0 + 8 * ESC) + c);
                }
            }
            if (false)
#endif
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) {
                const float2* hp0 = H + (long)((tr * NWC + wr + rr) * 16 + g) * ESC;
#ifdef G19_NOHDR  // diag: no header loads (empty slots: escapes unfixed): the header loads' share of the time
                hd0[rr] = hd1[rr] = make_float4(0.f, 0x1p-15f, 0.f, 0x1p-15f); (void)hp0;
#else
                hd0[rr] = __ldg((const float4*)hp0 + c);
                hd1[rr] = __ldg((const float4*)(hp0 + 8 * ESC) + c);
#endif
            }
        }
#endif
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
#ifdef G19_RAW
            uint4 fa[RPW][G][2];
            const unsigned rb_ = ring + slot * SB + wr * AP * 1024 + lane * 16;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int j = 0; j < G; ++j)
#pragma unroll
                    for (int s = 0; s < 2; ++s)
                        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(fa[rr][j][s].x), "=r"(fa[rr][j][s].y), "=r"(fa[rr][j][s].z), "=r"(fa[rr][j][s].w) : "r"(rb_ + rr * AP * 1024 + (j0 + j) * 1024 + s * 512));
#else
            uint4 sv[RPW][G];
            uint2 nv[RPW][G];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int j = 0; j < G; ++j) {
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[rr][j].x), "=r"(sv[rr][j].y), "=r"(sv[rr][j].z), "=r"(sv[rr][j].w) : "r"(sb + rr * AP * 512 + (j0 + j) * 512));
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[rr][j].x), "=r"(nv[rr][j].y) : "r"(nb + rr * AP * 256 + (j0 + j) * 256));
                }
#endif
            unsigned bx[G][2][NT][2];
#pragma unroll
            for (int j = 0; j < G; ++j)
                if (NT == 1) ldsm4(xbase + (st * AP + j0 + j) * 64, bx[j][0][0][0], bx[j][0][0][1], bx[j][1][0][0], bx[j][1][0][1]);
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
#ifdef G19_RAW
                        const uint4 af = fa[rr][j][s];
#else
                        const uint4 af = dec8(s ? sv[rr][j].z : sv[rr][j].x, s ? sv[rr][j].w : sv[rr][j].y, s ? nv[rr][j].y : nv[rr][j].x);
#endif
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[rr][jj] : acc[rr][jj], af, bx[j][s][jj][0], bx[j][s][jj][1]);
                    }
        }
#endif
#ifdef G19_TRACE
        if (t == 0 && i == 0) { TR(6); volatile float z_ = hd0[0].x + hd1[0].y; TR(7); }
        if (t == 0 && i == 1) TR(8);
#endif
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
        if (!end) continue;  // (a last x use always ends a run: released below, after the escape fixups read x)
        // run end: rows rg*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        if (t == 0 && q == 0) TR(11);
        const int nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
#if defined(G19_OLDFIX) || defined(G19_RFIX)
        constexpr bool sfix = false;
#else
        const bool sfix = !(nat == KA && PRE != 2);  // pbuf path: escape fixups applied in smem after the store (below)
#endif
        float v[RPW][NT][4];
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) { v[rr][jj][e] = acc[rr][jj][e] + acc2[rr][jj][e]; acc[rr][jj][e] = acc2[rr][jj][e] = 0.f; }
#pragma unroll
        for (int rr = 0; rr < (WB == 1024 ? 0 : RPW); ++rr)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (sfix) break;
                const float4 hq = h ? hd1[rr] : hd0[rr];
#ifndef G19_OLDFIX
                // which slots any lane of the warp needs (owner lanes test their own 2 slots; OR over the 8 row lanes g):
                // the loop below then skips empty slots by warp-uniform branches instead of a serialized shfl chain
                unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
                unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
                f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
                f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#endif
#pragma unroll
                for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
#ifndef G19_OLDFIX
                    if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
#endif
#ifdef G19_NOFIX
                    continue;
#endif
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
        if (t == 0 && q == 0) TR(12);
        if (xl && !sfix) {  // done with this x buffer (the fixups above were its last reads)
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
                    {
                        const long o_ = (long)(th * MT + jj * 8 + 2 * c + (e & 1)) * n + (tr * NWC + wr + rr) * 16 + g + 8 * (e >> 1);
                        if (PRE == 3) ob[o_] = __float2bfloat16_rn(v[rr][jj][e] * scale);
                        else out[o_] = v[rr][jj][e] * scale;
                    }
            continue;
        }
        const int pb = q % NPB;  // hand the partials to the finisher: pbuf[pb][row group * 16 + row16][tok]
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
        if (t == 0 && q == 1) TR(13);
#pragma unroll
        for (int rr = 0; rr < RPW; ++rr) {
            float* pw = pbuf + pb * PBF + ((wr + rr) * 16 + g) * M + th * MT + 2 * c;
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[rr][jj][2 * h], v[rr][jj][2 * h + 1]);
        }
#if !defined(G19_OLDFIX) && !defined(G19_RFIX)
        // escape fixups on the stored partials, one escape at a time over the warp (lane = token): the in-range slots
        // (lane c holds slots 2c, 2c + 1 of rows g, g + 8) are found by ballots and walked in lane order, warp-uniformly
        if (WB != 1024) {
            __syncwarp();
            const unsigned char* xt = smem + NS * SB + xb * M * XS + (th * MT + lane) * XS - ks * KC * 2;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int pr = 0; pr < 2; ++pr) {
                        const float4 hq = h ? hd1[rr] : hd0[rr];
                        const float hx = pr ? hq.z : hq.x, hy = pr ? hq.w : hq.y;
                        unsigned b = __ballot_sync(0xffffffffu, hy != 0x1p-15f && __float_as_int(hx) >= klo && __float_as_int(hx) < KHI_B);
#ifdef G19_FIX2  // s48: two escapes per round (loads of both issued first; same per-address order: bit-identical)
                        while (b) {
                            const int s1 = __ffs(b) - 1;
                            b &= b - 1;
                            const bool two = b != 0;
                            const int s2 = two ? __ffs(b) - 1 : s1;
                            b &= b - 1;
                            const int col1 = __float_as_int(__shfl_sync(0xffffffffu, hx, s1)), col2 = __float_as_int(__shfl_sync(0xffffffffu, hx, s2));
                            const float hv1 = __shfl_sync(0xffffffffu, hy, s1), hv2 = __shfl_sync(0xffffffffu, hy, s2);
                            if (lane < MT) {
                                float* pp1 = pbuf + pb * PBF + ((wr + rr) * 16 + (s1 >> 2) + 8 * h) * M + th * MT + lane;
                                float* pp2 = pbuf + pb * PBF + ((wr + rr) * 16 + (s2 >> 2) + 8 * h) * M + th * MT + lane;
                                const float xv1 = __bfloat162float(*(const __nv_bfloat16*)(xt + col1 * 2));
                                const float xv2 = __bfloat162float(*(const __nv_bfloat16*)(xt + col2 * 2));
                                const float p1 = *pp1, p2 = *pp2;
                                const float r1 = fmaf(xv1, hv1, fmaf(xv1, -0x1p-15f, p1));
                                if (!two) *pp1 = r1;
                                else if ((s1 >> 2) == (s2 >> 2)) *pp1 = fmaf(xv2, hv2, fmaf(xv2, -0x1p-15f, r1));
                                else { *pp1 = r1; *pp2 = fmaf(xv2, hv2, fmaf(xv2, -0x1p-15f, p2)); }
                            }
                        }
#else
                        while (b) {
                            const int src = __ffs(b) - 1;
                            b &= b - 1;
                            const int col = __float_as_int(__shfl_sync(0xffffffffu, hx, src));
                            const float hv = __shfl_sync(0xffffffffu, hy, src);
                            if (lane < MT) {
                                float* pp = pbuf + pb * PBF + ((wr + rr) * 16 + (src >> 2) + 8 * h) * M + th * MT + lane;
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xt + col * 2));
                                *pp = fmaf(xv, hv, fmaf(xv, -0x1p-15f, *pp));
                            }
                        }
#endif
                    }
            __syncwarp();
        }
        if (xl) {  // done with this x buffer (the fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
#endif
        if (t == 0) ptag[pb] = tr | nat << 12;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        if (t == 0 && q == 0) TR(14);
        if (t == 0 && q == 1) TR(15);
        ++q;
    }
    {  // done: tell the finisher
        const int pb = q % NPB;
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
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
// ns_ < 0 (s40 probe): instead of spinning, -ns_ dependent __ldcg loads per thread through a 256 MB random index table
// (a latency-bound, memory-touching predecessor like the layer's glue)
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_, const int* chase) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    if (ns_ < 0) {
        int idx = (int)((blockIdx.x * blockDim.x + threadIdx.x) * 7919u & ((64u << 20) - 1));
        for (int i = 0; i < -ns_; ++i) idx = __ldcg(chase + idx);
        if (idx == -1) dst[0] = src[0];
        return;
    }
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
    static torch::Tensor chase;
    if (ns_ < 0 && !chase.defined()) chase = torch::randint(0, 64 << 20, {64 << 20}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    g19::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_,
                                                                 ns_ < 0 ? chase.data_ptr<int>() : nullptr);
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
// mode 3: BF16 out ob; out = FP32 workspace [m, n], zero before the first call and left zero; ctr int32 [>= 2 + n / (16 rg)]
// zero before the first call (one per call site; tail must be 0). No predecessor needed (x read after griddepcontrol.wait).
void g19_gemm_ob(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw,
              __nv_bfloat16* ob, int tm);
void g19_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw) {
    TORCH_CHECK(mode < 3, "g19: mode 3 = g19_gemm_bf16");
    g19_gemm_ob(wp, h, scale, x, out, ctr, kc, ap, ns, is, grid, tail, mode, npre, xb, ts, rg, rpw, nullptr, 0);
}
void g19_gemm_bf16(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor ws, torch::Tensor ob, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t grid, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw, int64_t tm) {
    TORCH_CHECK(ob.scalar_type() == at::kBFloat16 && ob.sizes() == ws.sizes() && ctr.numel() >= 2 + ws.size(1) / (16 * rg), "g19_gemm_bf16: args");
    g19_gemm_ob(wp, h, scale, x, ws, ctr, kc, ap, ns, 1, grid, 0, 3, npre, xb, ts, rg, rpw, (__nv_bfloat16*)ob.data_ptr(), (int)tm);
}
void g19_gemm_ob(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw,
              __nv_bfloat16* ob, int tm) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = ns * rg * ap * g19::WB + xb * m * (kc * 2 + 16) + g19::NPB * rg * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8
#ifdef G19_XQ
                 + 64
#endif

                 + (mode == 3 ? 128 + m * rg * 64 + 16 : 0);
    TORCH_CHECK(n % (16 * rg) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && npre <= ns && n / (16 * rg) <= 4096, "g19: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g19::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G19_ONLY2
    TORCH_CHECK(mode == 2, "g19: built for mode 2 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>
#elif defined(G19_ONLY23)
    TORCH_CHECK(mode >= 2, "g19: built for modes 2, 3 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 3 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 3, XB_, TS_, RG_, RPW_> : g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>)
#else
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 3 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 3, XB_, TS_, RG_, RPW_> : mode == 2 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_> : mode ? g19::gemm<M_, KC_, AP_, NS_, IS_, 1, XB_, TS_, RG_, RPW_> : g19::gemm<M_, KC_, AP_, NS_, IS_, 0, XB_, TS_, RG_, RPW_>)
#endif
#define G19_CASE(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_ && xb == XB_ && ts == TS_ && rg == RG_ && rpw == RPW_) { \
        auto kern = G19_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        if (G19_MINB > 1) { cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
          int nb_ = 0, nsm_ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_, kern, (RG_ * TS_ / RPW_ + 2) * 32, sm); \
          cudaDeviceGetAttribute(&nsm_, cudaDevAttrMultiProcessorCount, 0); \
          TORCH_CHECK(mode < 2 || nb_ * nsm_ >= grid, "g19: grid not co-resident (grid barrier would hang): ", nb_, " blocks/SM"); } \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3((RG_ * TS_ / RPW_ + 2) * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre, ob, tm); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G19_CONFIGS
    TORCH_CHECK(false, "g19: no config");
}
'''
_G20_SRC = r'''// gemv19: gemv18's TMA-ring streamer over lossless 12-bit weights (m = 16/32, same _pack18 stage layout), FP32 output
// only, with load balancing. Per-SM HBM bandwidth varies by ~10% (gemv18 blocks ended 21-27 us at pooler m16 with
// equal static ranges), so: each block first streams a static stream-K range of the first U - TAIL stages, then grabs
// tail items of IS stages from a global counter until none are left. The producer warp decides everything: it tags
// every ring slot with its stage u (+ x buffer, + "run ends here" flag); consumers flush partials at run ends; x is
// double-buffered and loaded by the producer when a stage needs another k-chunk. All partials go to the FP32 output
// by red.global.add (finisher warp); a tiny zero kernel (PDL primary) zeroes the output and the counter first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
namespace g20 {
constexpr int ESC = 8, NWC = 8;
#ifdef G20_TRACE
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
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
    asm volatile("{\n.reg .pred p;\nW20_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W20_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
#ifdef G20_NOWG
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
#else
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
#endif
__device__ __forceinline__ void fence_r(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// K-major SW128 B descriptor (8 token rows x 128 B core groups, 1024 B apart)
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int M> __device__ __forceinline__ void wgmma(float* d, const unsigned* a, unsigned long long desc);
template <> __device__ __forceinline__ void wgmma<16>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<32>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// one ring stage of a consumer warp: AP atoms of its 16-row group, decoded into af (one A fragment per k16), then
// 2 * AP wgmma m64nMk16 of its warpgroup (4 row groups = 64 rows) over the x chunk (SW128, 64 k per M * 128 B block)
template <int M, int AP>
__device__ __forceinline__ void stage(float* acc, unsigned (&af)[2 * AP][4], unsigned sb, unsigned nb, unsigned xbuf, int kk0) {
#pragma unroll
    for (int j = 0; j < AP; ++j) {
        uint4 sv;
        uint2 nv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
#ifdef G20_NODEC
        const uint4 a0 = make_uint4(sv.x, sv.y, nv.x, nv.y), a1 = make_uint4(sv.z, sv.w, nv.y ^ sv.x, nv.x ^ sv.y);
#else
        const uint4 a0 = dec8(sv.x, sv.y, nv.x), a1 = dec8(sv.z, sv.w, nv.y);
#endif
        af[2 * j][0] = a0.x; af[2 * j][1] = a0.y; af[2 * j][2] = a0.z; af[2 * j][3] = a0.w;
        af[2 * j + 1][0] = a1.x; af[2 * j + 1][1] = a1.y; af[2 * j + 1][2] = a1.z; af[2 * j + 1][3] = a1.w;
    }
    wg_fence();
#pragma unroll
    for (int j = 0; j < 2 * AP; ++j) {
        const int kk = kk0 + j * 16;
#ifdef G20_NOMMA
        acc[j & 7] += __uint_as_float(af[j][0] ^ af[j][1] ^ af[j][2] ^ af[j][3] ^ (unsigned)kk ^ xbuf);
#else
        wgmma<M>(acc, af[j], sw128_desc(xbuf + (kk >> 6) * M * 128 + (kk & 63) * 2));
#endif
    }
    wg_commit();
}
// M tokens (16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary or in-kernel), ctr int. tail: stages left for dynamic items.
// smem (1024-aligned): ring [NS][SB] | x [XB][KC/64][M][128 B] SW128 | pbuf [2][128 rows][M] f32 | barriers | tags
// consumers: 8 warps = 2 warpgroups; warp w = 16-row group w of the 128-row tile (same packing as gemv19, rg 8)
#ifndef G20_NPB
#define G20_NPB 2
#endif
constexpr int NPB = G20_NPB;  // partial buffers (1: less smem, for 2 blocks per SM)
template <int M, int KC, int AP, int NS, int IS, int PRE, int XB, int TS, int RG, int RPW>
#ifndef G20_MINB
#define G20_MINB 1
#endif
__global__ void __launch_bounds__(320, G20_MINB) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                               float scale, const __grid_constant__ CUtensorMap xmap,
                                               const __nv_bfloat16* __restrict__ x,
                                               float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre) {
    static_assert(TS == 1 && RG == 8 && RPW == 1, "g20: rg 8, ts 1, rpw 1 only");
    constexpr int NCW = 8, NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = NCW * AP * 768, XBF = KC * 2 * M, PBF = NCW * 16 * M;
    static_assert(NA % AP == 0 && SPK % IS == 0 && KC % 64 == 0 && SB % 1024 == 0, "config");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + XB * XBF);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + NPB * PBF);
    int* tag = (int*)(pbuf + NPB * PBF) + 2 * (2 * NS + 8);  // per slot: u | xb << 29 | last x use << 30 | run end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile (-1: consumers done)
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NCW, S = k / KC, KA = k / 32;
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
    if (w == NCW) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // stream the first stages' W while the previous kernel runs
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
                bar_expect(xfull, XBF);
                for (int b = 0; b < KC / 64; ++b) tma2d(xs + b * M * 128, &xmap, ks0 * KC + b * 64, 0, xfull);
            } else {
                asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
            }
            for (;;) {
                if (ra >= rb) {  // next run: a tail item
                    if (tail == 0) break;
                    const int it = atomicAdd(ctr, 1);
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
                        bar_expect(xfull + 8 * xb, XBF);
                        for (int b = 0; b < KC / 64; ++b) tma2d(xs + xb * XBF + b * M * 128, &xmap, ks * KC + b * 64, 0, xfull + 8 * xb);
                    }
                    const int slot = i % NS;
                    if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                    const bool end = u + 1 == rb || st + 1 == SPK;
                    const bool xl = u + 1 == rb || (u + 1) / (ntr * SPK) != ks;
                    if (xl) xks[xb] = -1;
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    if (i < 24) TR(40 + i);
                    const unsigned full = bars + 8 * slot;
#ifdef G20_NOW
                    bar_arrive(full);
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
            const int pb = q % NPB;
            bar_wait(pfull + 8 * pb, (q / NPB) & 1);
            const int tr = ptag[pb];
            if (tr < 0) break;
            if (PRE == 2 && q == 0) {
                if (lane == 0) {
                    unsigned v;
                    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - target) < 0);
                }
                __syncwarp();
            }
            const float* src = pbuf + pb * PBF;
            float* wt = out + (long)tr * NCW * 16;
#pragma unroll 4
            for (int i = lane; i < PBF / 4; i += 32) {
                const int tok = i % M, r4 = (i / M) * 4;
                const float a0 = src[r4 * M + tok] * scale, a1 = src[(r4 + 1) * M + tok] * scale;
                const float a2 = src[(r4 + 2) * M + tok] * scale, a3 = src[(r4 + 3) * M + tok] * scale;
#ifndef G20_NORED
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + (long)tok * n + r4), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
#else
                if (a0 == 1.2345f) wt[0] = a1 + a2 + a3;
#endif
            }
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT * 4];  // wgmma m64nM accumulator: per n8 tile jj, [4] = mma.sync C layout (rows g, g + 8; tokens 2c, 2c + 1)
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
    unsigned afA[2 * AP][4], afB[2 * AP][4];
    const int wr = w;  // the warp's row group in the tile
    int q = 0, st0 = 0, prev_end = 1, prev_xl = 1;
    unsigned xph[2] = {0, 0};
    float4 hd0, hd1;
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
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row group; lane c: slots 2c, 2c + 1)
            st0 = st;
            const float2* hp0 = H + (long)((tr * NCW + wr) * 16 + g) * ESC;
            hd0 = __ldg((const float4*)hp0 + c);
            hd1 = __ldg((const float4*)(hp0 + 8 * ESC) + c);
        }
        prev_end = end;
        const unsigned xbuf = xs + xb * XBF;
        const unsigned sb = ring + slot * SB + wr * AP * 512 + lane * 16, nb = ring + slot * SB + NCW * AP * 512 + wr * AP * 256 + lane * 8;
        // double-buffered A fragments: this stage's group is in flight while the next stage decodes into the other set
#ifdef G20_SB
        stage<M, AP>(acc, afA, sb, nb, xbuf, st * AP * 32);
#else
        if (i & 1) stage<M, AP>(acc, afB, sb, nb, xbuf, st * AP * 32);
        else stage<M, AP>(acc, afA, sb, nb, xbuf, st * AP * 32);
#endif
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));  // W of this slot is in registers (decoded)
        if (!end) {
#ifdef G20_SB
            wg_wait<0>();
#pragma unroll
            for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) fence_r(afA[j][e]);
            continue;
#endif
            wg_wait<1>();  // the previous stage's group is done: its A set may be overwritten next
#pragma unroll
            for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) { if (i & 1) fence_r(afA[j][e]); else fence_r(afB[j][e]); }
            continue;
        }
        wg_wait<0>();
#pragma unroll
        for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { fence_r(afA[j][e]); fence_r(afB[j][e]); }
#pragma unroll
        for (int e = 0; e < NT * 4; ++e) fence_f(acc[e]);
        // run end: rows wr*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        const int nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
        float v[NT][4];
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) { v[jj][e] = acc[jj * 4 + e]; acc[jj * 4 + e] = 0.f; }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = h ? hd1 : hd0;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
#ifdef G20_NOFIX
                if (hy == 1.2345f)
#else
                if (hy != 0x1p-15f && col >= klo && col < khi) 
#endif
                {  // two fmas: exact when the dot holds only this weight (probe rows)
                    const int kk = col - ks * KC;
                    const unsigned char* xc = smem + NS * SB + xb * XBF + (kk >> 6) * M * 128 + (kk & 7) * 2;
                    const int ch = (kk & 63) >> 3;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const int tok = jj * 8 + 2 * c + b;
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + tok * 128 + ((ch ^ (tok & 7)) << 4)));
                            v[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, v[jj][2 * h + b]));
                        }
                }
            }
        }
        if (xl) {  // done with this x buffer (wgmma groups done, fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
        if (nat == KA && PRE != 2) {  // the whole dot: plain stores
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    out[(long)(jj * 8 + 2 * c + (e & 1)) * n + (tr * NCW + wr) * 16 + g + 8 * (e >> 1)] = v[jj][e] * scale;
            continue;
        }
        const int pb = q % NPB;  // hand the partials to the finisher: pbuf[pb][row group * 16 + row16][tok]
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
        {
            float* pw = pbuf + pb * PBF + (wr * 16 + g) * M + 2 * c;
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[jj][2 * h], v[jj][2 * h + 1]);
        }
        if (t == 0) ptag[pb] = tr;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        ++q;
    }
    wg_wait<0>();
    {  // done: tell the finisher
        const int pb = q % NPB;
        if (q >= NPB) bar_wait(pempty + 8 * pb, ((q / NPB) - 1) & 1);
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
}  // namespace g20
void g20_cvt(torch::Tensor src, torch::Tensor dst) {
    const int n4 = src.numel() / 4;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((n4 + 255) / 256 < 132 ? (n4 + 255) / 256 : 132); cfg.blockDim = dim3(256);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, g20::cvt, (const float4*)src.data_ptr(), (uint2*)dst.data_ptr(), n4);
}
void g20_stream(torch::Tensor w, int64_t sb, int64_t ns, torch::Tensor out) {
    const int sm = sb * ns + 16 * ns;
    cudaFuncSetAttribute(g20::stream, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    g20::stream<<<132, 288, sm, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)w.data_ptr(), w.numel(), sb, ns, out.data_ptr<float>());
}
void g20_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g20::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g20_zero(torch::Tensor out, torch::Tensor ctr) {
    g20::zero<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((float4*)out.data_ptr(), out.numel() / 4, ctr.data_ptr<int>());
}
#ifdef G20_TRACE
torch::Tensor g20_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g20::trace, sizeof(unsigned long long) * 132 * 64);
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
static PFN_cuTensorMapEncodeTiled_v12000 g20_encode = nullptr;
// x tensor map: bf16 [m, k], box {64 k, m tokens}, SW128 (encoded at every launch from the x actually passed: under
// graph capture it is baked into the kernel node with that buffer)
static void g20_xmap(CUtensorMap& map, void* ptr, int m, int k) {
    if (!g20_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g20_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g20_encode, "g20: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)m};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g20_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g20: cuTensorMapEncodeTiled failed ", (int)r);
}
void g20_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb, int64_t ts, int64_t rg, int64_t rpw) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = 1024 + ns * 8 * ap * 768 + xb * m * kc * 2 + g20::NPB * 8 * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8;
    CUtensorMap xmap;
    g20_xmap(xmap, x.data_ptr(), m, k);
    TORCH_CHECK(n % (16 * rg) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && npre <= ns, "g20: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g20::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G20_ONLY2
    TORCH_CHECK(mode == 2, "g20: built for mode 2 only");
#define G20_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) g20::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_>
#else
#define G20_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) (mode == 2 ? g20::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_, TS_, RG_, RPW_> : mode ? g20::gemm<M_, KC_, AP_, NS_, IS_, 1, XB_, TS_, RG_, RPW_> : g20::gemm<M_, KC_, AP_, NS_, IS_, 0, XB_, TS_, RG_, RPW_>)
#endif
#define G20_CASE(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_ && xb == XB_ && ts == TS_ && rg == RG_ && rpw == RPW_) { \
        auto kern = G20_KERN(M_, KC_, AP_, NS_, IS_, XB_, TS_, RG_, RPW_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
        { int nb_ = 0, nsm_ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb_, kern, 320, sm); \
          cudaDeviceGetAttribute(&nsm_, cudaDevAttrMultiProcessorCount, 0); \
          TORCH_CHECK(mode < 2 || nb_ * nsm_ >= grid, "g20: grid not co-resident (grid barrier would hang): ", nb_, " blocks/SM"); } \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, xmap, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G20_CONFIGS
    TORCH_CHECK(false, "g20: no config");
}
'''
_G22_SRC = r'''// gemv22: GEMM for m = 128 / 256 tokens (the layer's T=128/256: verify 16 / 32) over lossless 12-bit BF16 weights
// (gemv19's packed atoms, 4-bit exponent codes + per-row escape headers), FP32 output [m, n].
// Swap-AB wgmma: W rows = the wgmma M (64 per warpgroup, A decoded into registers), tokens = N (128 or 256, B = x by a
// SW128 tensor map). Tile-major stream-K: stage u = (128-row tile, 64-k step), each block streams a contiguous range of
// stages; W (12 KB) and the stage's x box (M x 64 k) share one ring slot. Consumers (8 warps = 2 warpgroups) flush their
// partials at run ends (tile change or range end) by red.global.add into out, which a zero kernel cleared first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#ifdef G22_NOPEER
#define G22_NOPEERV 1
#else
#define G22_NOPEERV 0
#endif
#if defined(G22_NOFIX) || defined(G22_NOW)
#define G22_NOFIXV 1
#else
#define G22_NOFIXV 0
#endif
namespace g22 {
constexpr int ESC = 8;
#ifdef G22_TRACE
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
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
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
// arrive on the mbarrier at the same smem offset in CTA `peer` of the cluster
__device__ __forceinline__ void bar_arrive_peer(unsigned b, unsigned peer) {
#ifdef G22_NOPEER
    return;
#endif
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(b), "r"(peer));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
__device__ __forceinline__ void tma2d_mc(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar, unsigned short mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW22_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W22_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
#ifdef G22_NOWG
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
#else
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
#endif
__device__ __forceinline__ void fence_r(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// K-major SW128 B descriptor (8 token rows x 128 B core groups, 1024 B apart)
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int M> __device__ __forceinline__ void wgmma(float* d, const unsigned* a, unsigned long long desc);
template <> __device__ __forceinline__ void wgmma<16>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<32>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<128>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<256>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

// one ring stage of a consumer warp: AP atoms (32 k) of its 16-row group decoded into af (one A fragment per k16),
// then 2 * AP wgmma m64nMk16 of its warpgroup over the slot's x box(es) ([M tokens][64 k] SW128 per 64 k)
template <int M, int AP>
__device__ __forceinline__ void stage(float* acc, unsigned (&af)[2 * AP][4], unsigned sb, unsigned nb, unsigned xbuf) {
#pragma unroll
    for (int j = 0; j < AP; ++j) {
        uint4 sv;
        uint2 nv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
        const uint4 a0 = dec8(sv.x, sv.y, nv.x), a1 = dec8(sv.z, sv.w, nv.y);
        af[2 * j][0] = a0.x; af[2 * j][1] = a0.y; af[2 * j][2] = a0.z; af[2 * j][3] = a0.w;
        af[2 * j + 1][0] = a1.x; af[2 * j + 1][1] = a1.y; af[2 * j + 1][2] = a1.z; af[2 * j + 1][3] = a1.w;
    }
    wg_fence();
#pragma unroll
    for (int j = 0; j < 2 * AP; ++j) {
        const int kk = j * 16;
#ifdef G22_NOMMA
        acc[j & 7] += __uint_as_float(af[j][0] ^ af[j][1] ^ af[j][2] ^ af[j][3] ^ (unsigned)kk ^ xbuf);
#else
        wgmma<M>(acc, af[j], sw128_desc(xbuf + (kk >> 6) * M * 128 + (kk & 63) * 2));
#endif
    }
    wg_commit();
}
// M tokens (128/256), AP atoms (32 k) per warp per stage (AP * 32 = k per stage, a multiple of 64), NS ring stages.
// WP: packed stages, tile-major (u = tile * S + ks, S = k / (32 AP)). out FP32 [M, n], zeroed before (PDL primary).
// packed stage (PB bytes): W atoms (SB) | E escapes (row in tile | k in stage << 8, true / scale; row -1: none)
// smem (1024-aligned): ring [NS][stage PB | pad to XO | XS x] | barriers | flush scratch
template <int M, int AP, int NS, int E, int CL>
__global__ void __launch_bounds__(384, 1) gemm(const unsigned char* __restrict__ WP,                                                float scale, const __grid_constant__ CUtensorMap xmap,
                                               const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int n, int k, int npre) {
    constexpr int NCW = 8, NT = M / 8, SB = NCW * AP * 768, XS = AP / 2 * M * 128, PB = SB + 8 * E, XO = SB + 1024, SL = XO + XS;
    static_assert(AP % 2 == 0 && SB % 1024 == 0 && E % 32 == 0 && E <= 128, "config");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned bars = ring + NS * SL;
    const int t = threadIdx.x, lane = t & 31, w = __shfl_sync(0xffffffffu, t >> 5, 0), g = lane >> 2, c = lane & 3;
    const int S = k / (32 * AP), ntr = n / (16 * NCW);
    // tile-aligned split: tile t gets nb = G / ntr (+1 for the first G % ntr tiles) blocks: one run per block
    // CL 2: clusters of 2 blocks on tiles (2p, 2p + 1) with the same k range; each loads half the x box, multicast to both
    const int cr = CL == 2 ? blockIdx.x & 1 : 0;
    int ra0, rb0;
    {
        const int G = gridDim.x / CL, b = blockIdx.x / CL, base = G / (ntr / CL), ex = G % (ntr / CL);
        int tt, j, nb;
        if (b < ex * (base + 1)) { tt = b / (base + 1); j = b % (base + 1); nb = base + 1; }
        else { const int b2 = b - ex * (base + 1); tt = ex + b2 / base; j = b2 % base; nb = base; }
        tt = tt * CL + cr;
        ra0 = tt * S + S * j / nb; rb0 = tt * S + S * (j + 1) / nb;
    }
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), (NCW + (G22_NOFIXV ? 0 : 1)) * (G22_NOPEERV ? 1 : CL)); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (CL == 1) __syncthreads();
    else cluster_sync();  // the peer's barriers are initialized before any multicast / remote arrive
    if (t == 0) TR(0);
    if (w >= NCW) {  // producer warpgroup: one lane streams the block's stages
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (w == NCW && lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
            int i = 0;
            // W of the first stages may stream before the previous kernel ends (x must wait)
            for (; i < npre && i < NS && ra0 + i < rb0; ++i) {
                bar_expect(bars + 8 * i, PB + XS);
                bulk(ring + i * SL, WP + (long)(ra0 + i) * PB, PB, bars + 8 * i);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            TR(1);
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS, ks = u % S;
                if (j >= i) {
                    if (j >= NS) bar_wait(bars + 8 * (NS + slot), ((j / NS) - 1) & 1);
#ifdef G22_NOX
                    bar_expect(bars + 8 * slot, u == ra0 ? PB + XS : PB);
#elif defined(G22_NOW)
                    bar_expect(bars + 8 * slot, XS);  // diag: no W loads (consumers decode stale smem)
#else
                    bar_expect(bars + 8 * slot, PB + XS);
#endif
#ifndef G22_NOW
                    bulk(ring + slot * SL, WP + (long)u * PB, PB, bars + 8 * slot);
#endif
                }
#ifdef G22_NOX
                if (u == ra0) {
#else
                {
#endif
#pragma unroll
                for (int b = 0; b < AP / 2; ++b) {
#ifdef G22_NOMC
                    if (true) { tma2d(ring + slot * SL + XO + b * M * 128, &xmap, ks * AP * 32 + b * 64, 0, bars + 8 * slot);
                        tma2d(ring + slot * SL + XO + b * M * 128 + (M / 2) * 128, &xmap, ks * AP * 32 + b * 64, M / 2, bars + 8 * slot); }
#else
                    if (CL == 1) tma2d(ring + slot * SL + XO + b * M * 128, &xmap, ks * AP * 32 + b * 64, 0, bars + 8 * slot);
#endif
                    else tma2d_mc(ring + slot * SL + XO + b * M * 128 + cr * (M / 2) * 128, &xmap, ks * AP * 32 + b * 64, cr * (M / 2), bars + 8 * slot, 3);
                }
                }
            }
        } else if (w == NCW + 1 && !G22_NOFIXV) {  // fixup warp: escapes, x from the ring slot, red.add into out
            // escapes of stage u: EV[EST[u] .. EST[u + 1]) = (row | (k % 32AP) << 16, true / scale); the block's are contiguous
            asm volatile("griddepcontrol.wait;" ::: "memory");
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS;
                bar_wait(bars + 8 * slot, (j / NS) & 1);
                const unsigned xb = ring + slot * SL + XO;
                const int row0 = (u / S) * 128;
#pragma unroll 1
                for (int e0 = 0; e0 < E; e0 += 32) {
                    int2 ev;
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(ev.x), "=r"(ev.y) : "r"(ring + slot * SL + SB + (e0 + lane) * 8));
                    const unsigned full = __ballot_sync(0xffffffffu, ev.x >= 0);
                    unsigned bal = full;
                    while (bal) {
                        const int src = __ffs(bal) - 1;
                        bal &= bal - 1;
                        const int ex = __shfl_sync(0xffffffffu, ev.x, src), row = row0 + (ex & 0xFF), kk = ex >> 8;
                        const float hy = __int_as_float(__shfl_sync(0xffffffffu, ev.y, src));
#pragma unroll 4
                        for (int tok = lane; tok < M; tok += 32) {
                            const unsigned ad = xb + (kk >> 6) * M * 128 + tok * 128 + ((((kk & 63) >> 3) ^ (tok & 7)) << 4) + (kk & 7) * 2;
                            unsigned short xr;
                            asm volatile("ld.shared.u16 %0, [%1];" : "=h"(xr) : "r"(ad));
                            const float xv = __uint_as_float((unsigned)xr << 16);
                            asm volatile("red.global.add.f32 [%0], %1;" :: "l"(out + (long)tok * n + row), "f"(fmaf(xv, hy, -xv * 0x1p-15f) * scale));
                        }
                    }
                    if (full != 0xffffffffu) break;
                }
                __syncwarp();
                if (lane == 0) { bar_arrive(bars + 8 * (NS + slot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + slot), cr ^ 1); }
            }
            if (lane == 0) TR(7);
        }
        __syncwarp();
        if (CL == 2) cluster_sync();  // no block exits while its peer may still multicast into it / arrive on it
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT * 4];  // wgmma m64nM accumulator: per n8 tile jj, [4] = mma.sync C layout (rows g, g + 8; tokens 2c, 2c + 1)
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
    unsigned afA[2 * AP][4], afB[2 * AP][4];
    int i = 0, pslot = -1, nrun = 0;
    // one stage with A set af (ao: the previous stage's set, maybe in flight)
#define G22_STEP(af, ao) { \
        const int slot = i % NS; \
        bar_wait(bars + 8 * slot, (i / NS) & 1); \
        stage<M, AP>(acc, af, ring + slot * SL + w * AP * 512 + lane * 16, ring + slot * SL + NCW * AP * 512 + w * AP * 256 + lane * 8, ring + slot * SL + XO); \
        wg_wait<1>(); \
        _Pragma("unroll") for (int j = 0; j < 2 * AP; ++j) _Pragma("unroll") for (int e = 0; e < 4; ++e) fence_r(ao[j][e]); \
        __syncwarp(); \
        if (lane == 0 && pslot >= 0) { bar_arrive(bars + 8 * (NS + pslot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + pslot), cr ^ 1); } \
        pslot = slot; ++i; }
#pragma unroll 1
    for (int ra = ra0; ra < rb0;) {  // runs: the block's range split at tile boundaries
        const int tr = ra / S, rb = min(rb0, (tr + 1) * S), nst = rb - ra;
        int s = 0;
        if (t == 0 && nrun == 0) { bar_wait(bars, 0); TR(2); }
#pragma unroll 1
        for (; s + 1 < nst; s += 2) {
            G22_STEP(afA, afB)
            G22_STEP(afB, afA)
        }
        if (s < nst) G22_STEP(afA, afB)
        wg_wait<0>();
        if (t == 0) TR(3 + 2 * nrun);
#pragma unroll
        for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { fence_r(afA[j][e]); fence_r(afB[j][e]); }
#pragma unroll
        for (int e = 0; e < NT * 4; ++e) fence_f(acc[e]);
        __syncwarp();
        if (lane == 0) { bar_arrive(bars + 8 * (NS + pslot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + pslot), cr ^ 1); }
        pslot = -1;
        constexpr int G4 = NT < 4 ? NT : 4;
        // flush: 32-token groups through smem ([tok][16 rows], stride 20 floats), red.add.v4 along rows
        {
            float* sc = (float*)(smem + NS * SL + 16 * NS + 64) + w * 32 * 20;
            float* ob = out + (tr * NCW + w) * 16;
#pragma unroll
            for (int t0 = 0; t0 < NT; t0 += G4) {
#pragma unroll
                for (int jj = t0; jj < t0 + G4; ++jj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) sc[((jj - t0) * 8 + 2 * c + (e & 1)) * 20 + g + 8 * (e >> 1)] = acc[jj * 4 + e] * scale;
                __syncwarp();
#pragma unroll
                for (int i2 = 0; i2 < G4; ++i2) {
                    const int f = lane + 32 * i2, tl = f >> 2, q4 = f & 3;
                    const float4 v = *(const float4*)(sc + tl * 20 + 4 * q4);
#ifdef G22_NORED
                    if (v.x == 1.2345f) ob[0] = v.y;
#else
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ob + (long)(t0 * 8 + tl) * n + 4 * q4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#endif
                }
                __syncwarp();
            }
#pragma unroll
            for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
        }
        if (t == 0) TR(4 + 2 * nrun);
        ++nrun;
        ra = rb;
    }
    if (t == 0) TR(9);
    if (CL == 2) cluster_sync();
#undef G22_STEP
}
// PDL primary: lets the gemm launch at once and zeroes its output (the gemm waits before touching it)
__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace g22
#ifdef G22_TRACE
torch::Tensor g22_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g22::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
static PFN_cuTensorMapEncodeTiled_v12000 g22_encode = nullptr;
static void g22_xmap(CUtensorMap& map, void* ptr, int m, int k, int rows) {
    if (!g22_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g22_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g22_encode, "g22: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)rows};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g22_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g22: cuTensorMapEncodeTiled failed ", (int)r);
}
// mode 0: zero kernel then the gemm (PDL dependent). mode 1: out already zero, the gemm a PDL dependent of the
// previous kernel (npre W stages stream before griddepcontrol.wait).
void g22_gemm(torch::Tensor wp, double scale, torch::Tensor x, torch::Tensor out,
              int64_t ap, int64_t ns, int64_t e, int64_t cl, int64_t grid, int64_t mode, int64_t npre) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = 1024 + ns * (8 * ap * 768 + 1024 + ap / 2 * m * 128) + 16 * ns + 64 + 8 * 32 * 20 * 4;
    CUtensorMap xmap;
    g22_xmap(xmap, x.data_ptr(), m, k, m / cl);
    TORCH_CHECK(n % 128 == 0 && k % (32 * ap) == 0 && sm <= 227 * 1024 && n / 128 <= grid && grid <= (n / 128) * (k / (32 * ap))
                && grid % cl == 0 && (n / 128) % cl == 0, "g22: shape/smem/grid ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g22::zero<<<132, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4);
#define G22_CASE(M_, AP_, NS_, E_, CL_) \
    if (m == M_ && ap == AP_ && ns == NS_ && e == E_ && cl == CL_) { \
        auto kern = g22::gemm<M_, AP_, NS_, E_, CL_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[2]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        attr[1].id = cudaLaunchAttributeClusterDimension; attr[1].val.clusterDim.x = CL_; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1; \
        cfg.attrs = attr; cfg.numAttrs = 2; \
        if (CL_ > 1) { int nc_ = 0; cudaOccupancyMaxActiveClusters(&nc_, kern, &cfg); \
            TORCH_CHECK(nc_ * CL_ >= grid, "g22: grid not co-resident as clusters: ", nc_, " clusters"); } \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (float)scale, xmap, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), n, k, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G22_CONFIGS
    TORCH_CHECK(false, "g22: no config");
}
'''
_TG2_SRC = r'''/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from TensorRT-LLM tinygemm2 kernel for FlashInfer integration.
 * Original: TensorRT-LLM/cpp/tensorrt_llm/kernels/tinygemm2/
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda/barrier>
#include <cuda/std/utility>

#include "cuda_bf16.h"
#include "cuda_pipeline.h"
#include <iostream>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#define TVM_FFI_ICHECK(c) if (c) {} else std::cerr

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace ptx = cuda::ptx;

// ============================================================================
// Device helpers (from tinygemm2_kernel.cuh)
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ void ldmatrix(__nv_bfloat16 rv[2], uint32_t smem_ptr) {
  int dst;
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(dst)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = dst;
}

__device__ void ldmatrix2(__nv_bfloat16 rv[4], uint32_t smem_ptr) {
  int x, y;
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(x), "=r"(y)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = x;
  rvi[1] = y;
}

__device__ void ldmatrix4(__nv_bfloat16 rv[8], uint32_t smem_ptr) {
  int x, y, z, w;
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "r"(smem_ptr));
  int* rvi = reinterpret_cast<int*>(&rv[0]);
  rvi[0] = x;
  rvi[1] = y;
  rvi[2] = z;
  rvi[3] = w;
}

__device__ void HMMA_1688(float d[4], __nv_bfloat16 a[4], __nv_bfloat16 b[2], float c[4]) {
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
  float const* C = reinterpret_cast<float const*>(&c[0]);
  float* D = reinterpret_cast<float*>(&d[0]);
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ void HMMA_16816(float d[4], __nv_bfloat16 a[8], __nv_bfloat16 b[4], float c[4]) {
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[0]);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[0]);
  float const* C = reinterpret_cast<float const*>(&c[0]);
  float* D = reinterpret_cast<float*>(&d[0]);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
        "f"(C[2]), "f"(C[3]));
}

__device__ void bar_wait(uint32_t bar_ptr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar_ptr),
      "r"(phase));
}

__device__ bool bar_try_wait(uint32_t bar_ptr, int phase) {
  uint32_t success;
  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(success)
      : "r"(bar_ptr), "r"(phase));
  return success;
}

__device__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}
#endif  // __CUDA_ARCH__ >= 900

// ============================================================================
// Kernel template (from tinygemm2_kernel.cuh)
// ============================================================================

template <int WARP_TILE_M, int TILE_M, int TILE_N, int TILE_K, int STAGES, int STAGE_UNROLL,
          bool USE_PDL = false, bool HAS_BIAS = true>
__global__ __launch_bounds__(384, 1) void tinygemm_kernel(
    __nv_bfloat16* output, __nv_bfloat16* weights, __nv_bfloat16* activations, __nv_bfloat16* bias,
    int M, int N, int K, const __grid_constant__ CUtensorMap weight_map,
    const __grid_constant__ CUtensorMap activation_map) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

  extern __shared__ __align__(128) char smem[];

  __nv_bfloat16* sh_weights = (__nv_bfloat16*)&smem[0];
  __nv_bfloat16* sh_activations =
      (__nv_bfloat16*)&smem[STAGES * STAGE_UNROLL * TILE_M * TILE_K * sizeof(__nv_bfloat16)];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar_wt_ready[STAGES];
  __shared__ barrier bar_act_ready[STAGES];
  __shared__ barrier bar_data_consumed[STAGES];

  __shared__ float4 reduction_buffer[128];
  __shared__ nv_bfloat16 sh_bias[HAS_BIAS ? TILE_M : 1];

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      init(&bar_wt_ready[i], 1);
      init(&bar_act_ready[i], 1);
      init(&bar_data_consumed[i], 32);
    }
    ptx::fence_proxy_async(ptx::space_shared);
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&weight_map))
                 : "memory");
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(reinterpret_cast<uint64_t>(&activation_map))
                 : "memory");
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int phase = 0;

  int mib = blockIdx.x * TILE_M;
  int ni = blockIdx.y * TILE_N;

  float accum[4];
  for (int i = 0; i < 4; i++) accum[i] = 0.f;

  int const K_LOOPS_DMA = (K + 4 * TILE_K * STAGE_UNROLL - 1) / (4 * (TILE_K * STAGE_UNROLL));
  int const K_LOOPS_COMPUTE = K_LOOPS_DMA;

  if (warp_id >= 4) {
    // Data loading threads: one elected thread per warp issues TMA operations.
    // Non-elected threads in warps 4-11 idle but must still participate in the
    // block-wide __syncthreads() at the end of the kernel.
    if (elect_one_sync()) {
      int stage = warp_id % 4;

      bool weight_warp = warp_id < 8;
      if constexpr (USE_PDL) {
        if (!weight_warp) {
          cudaGridDependencySynchronize();
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }

      for (int ki = 0; ki < K_LOOPS_DMA; ki++) {
        int k = (ki * 4 + (warp_id % 4)) * TILE_K * STAGE_UNROLL;

        uint64_t desc_ptr_wt = reinterpret_cast<uint64_t>(&weight_map);
        uint64_t desc_ptr_act = reinterpret_cast<uint64_t>(&activation_map);

        uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
        uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);
        int bytes_wt = TILE_M * TILE_K * sizeof(__nv_bfloat16);
        int bytes_act = TILE_N * TILE_K * sizeof(__nv_bfloat16);

        bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);

        if (weight_warp)
          asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                       :
                       : "r"(bar_ptr_wt), "r"(STAGE_UNROLL * bytes_wt));
        if (!weight_warp)
          asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                       :
                       : "r"(bar_ptr_act), "r"(STAGE_UNROLL * bytes_act));

        for (int i = 0; i < STAGE_UNROLL; i++) {
          uint32_t smem_ptr_wt =
              __cvta_generic_to_shared(&sh_weights[(stage * STAGE_UNROLL + i) * TILE_M * TILE_K]);
          uint32_t crd0 = k + i * TILE_K;
          uint32_t crd1 = mib;
          if (weight_warp)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
                "[%1, "
                "{%3,%4}], [%2];"
                :
                : "r"(smem_ptr_wt), "l"(desc_ptr_wt), "r"(bar_ptr_wt), "r"(crd0), "r"(crd1)
                : "memory");

          uint32_t smem_ptr_act = __cvta_generic_to_shared(
              &sh_activations[(stage * STAGE_UNROLL + i) * TILE_N * TILE_K]);
          crd0 = k + i * TILE_K;
          crd1 = ni;
          if (!weight_warp)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
                "[%1, "
                "{%3,%4}], [%2];"
                :
                : "r"(smem_ptr_act), "l"(desc_ptr_act), "r"(bar_ptr_act), "r"(crd0), "r"(crd1)
                : "memory");
        }

        stage += 4;
        if (stage >= STAGES) {
          stage = warp_id % 4;
          phase ^= 1;
        }
      }
      // Wait for pending loads to be consumed before exiting, to avoid race
      for (int i = 0; i < (STAGES / 4) - 1; i++) {
        bar_wait(__cvta_generic_to_shared(&bar_data_consumed[stage]), phase ^ 1);
        stage += 4;
        if (stage >= STAGES) {
          stage = warp_id % 4;
          phase ^= 1;
        }
      }
    }
  }
  // Compute threads
  else {
    // Sneak the bias load into the compute warps since they're just waiting
    if constexpr (HAS_BIAS) {
      if (threadIdx.x < TILE_M) sh_bias[threadIdx.x] = bias[mib + threadIdx.x];
    }

    int stage = warp_id;

    int phase = 0;
    int lane_id_div8 = lane_id / 8;
    int lane_id_mod8 = lane_id % 8;

    int lane_row_offset_wt = (lane_id_div8 % 2) ? 8 : 0;
    int lane_col_offset_wt = (lane_id_div8 / 2) ? 1 : 0;

    int row_wt = lane_id_mod8 + lane_row_offset_wt;
    int row_act = lane_id_mod8;

    int row_offset_wt = (reinterpret_cast<uintptr_t>(sh_weights) / 128) % 8;
    int row_offset_act = row_offset_wt;

    uint32_t bar_ptr_wt = __cvta_generic_to_shared(&bar_wt_ready[stage]);
    uint32_t bar_ptr_act = __cvta_generic_to_shared(&bar_act_ready[stage]);

    bool weight_ready = bar_try_wait(bar_ptr_wt, phase);
    bool act_ready = bar_try_wait(bar_ptr_act, phase);

#pragma unroll 2
    for (int ki = 0; ki < K_LOOPS_COMPUTE; ki++) {
      int next_stage = stage + 4;
      int next_phase = phase;
      if (next_stage >= STAGES) {
        next_stage = warp_id;
        next_phase ^= 1;
      }

      while (!weight_ready || !act_ready) {
        weight_ready = bar_try_wait(__cvta_generic_to_shared(&bar_wt_ready[stage]), phase);
        act_ready = bar_try_wait(__cvta_generic_to_shared(&bar_act_ready[stage]), phase);
      }

      if (ki + 1 < K_LOOPS_COMPUTE) {
        weight_ready =
            bar_try_wait(__cvta_generic_to_shared(&bar_wt_ready[next_stage]), next_phase);
        act_ready = bar_try_wait(__cvta_generic_to_shared(&bar_act_ready[next_stage]), next_phase);
      }

#pragma unroll
      for (int su = 0; su < STAGE_UNROLL; su++) {
        __nv_bfloat16* ptr_weights = &sh_weights[(stage * STAGE_UNROLL + su) * TILE_M * TILE_K];
        __nv_bfloat16* ptr_act = &sh_activations[(stage * STAGE_UNROLL + su) * TILE_N * TILE_K];

#pragma unroll
        for (int kii = 0; kii < TILE_K / 16; kii++) {
          __nv_bfloat16 a[8];
          __nv_bfloat16 b[4];

          int col = 2 * kii + lane_col_offset_wt;
          int col_sw = ((row_wt + row_offset_wt) % 8) ^ col;

          ldmatrix4(a, __cvta_generic_to_shared(&ptr_weights[row_wt * TILE_K + col_sw * 8]));

          col = 2 * kii + lane_id_div8;
          col_sw = ((row_act + row_offset_act) % 8) ^ col;

          ldmatrix2(b, __cvta_generic_to_shared(&ptr_act[row_act * TILE_K + 8 * col_sw]));

          HMMA_16816(accum, a, b, accum);
        }
      }

      uint32_t bar_c = __cvta_generic_to_shared(&bar_data_consumed[stage]);
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(bar_c));

      stage = next_stage;
      phase = next_phase;
    }

    float4 accum4;
    accum4.x = accum[0];
    accum4.y = accum[1];
    accum4.z = accum[2];
    accum4.w = accum[3];
    reduction_buffer[threadIdx.x] = accum4;
  }

  // All threads (warps 0-11) must participate in __syncthreads().
  // This ensures warp 0 sees reduction_buffer writes from warps 1-3.
  __syncthreads();

  if (warp_id == 0) {
    int mi = mib + warp_id * WARP_TILE_M;
    int tm = mi + lane_id / 4;
    int tn = ni + 2 * (lane_id % 4);

    float4 accum1 = reduction_buffer[32 + threadIdx.x];
    float4 accum2 = reduction_buffer[64 + threadIdx.x];
    float4 accum3 = reduction_buffer[96 + threadIdx.x];

    accum[0] = accum[0] + accum1.x + accum2.x + accum3.x;
    accum[1] = accum[1] + accum1.y + accum2.y + accum3.y;
    accum[2] = accum[2] + accum1.z + accum2.z + accum3.z;
    accum[3] = accum[3] + accum1.w + accum2.w + accum3.w;

    if constexpr (HAS_BIAS) {
      float bias_lo = __bfloat162float(sh_bias[tm - mib]);
      float bias_hi = __bfloat162float(sh_bias[tm + 8 - mib]);

      if (tn < N && tm < M) output[tn * M + tm] = __float2bfloat16(accum[0] + bias_lo);
      if (tn + 1 < N && tm < M) output[(tn + 1) * M + tm] = __float2bfloat16(accum[1] + bias_lo);
      if (tn < N && tm + 8 < M) output[tn * M + tm + 8] = __float2bfloat16(accum[2] + bias_hi);
      if (tn + 1 < N && tm + 8 < M)
        output[(tn + 1) * M + tm + 8] = __float2bfloat16(accum[3] + bias_hi);
    } else {
      if (tn < N && tm < M) output[tn * M + tm] = __float2bfloat16(accum[0]);
      if (tn + 1 < N && tm < M) output[(tn + 1) * M + tm] = __float2bfloat16(accum[1]);
      if (tn < N && tm + 8 < M) output[tn * M + tm + 8] = __float2bfloat16(accum[2]);
      if (tn + 1 < N && tm + 8 < M) output[(tn + 1) * M + tm + 8] = __float2bfloat16(accum[3]);
    }
  }
#endif  // __CUDA_ARCH__ >= 900
}

// ============================================================================
// Host launcher (adapted from tinygemm2_cuda.cu)
// ============================================================================

namespace flashinfer {
namespace tinygemm2 {

// Kernel mapping for the linear operation output = input @ weight.T + bias:
//   input (batch_size, input_features) -> activations (gA), indexed as (N, K)
//   weight (output_features, input_features) -> weights (gB), indexed as (M, K)
//   output (batch_size, output_features) -> gC, stored column-major as (M, N)
//   bias (output_features,) -> bias, indexed as (M,)
//
// The kernel internally computes C[n,m] = sum_k A[n,k] * B[m,k] + bias[m]
// and writes output in column-major: output[n * M + m].
// The Python wrapper transposes this back to row-major (batch_size, output_features).

// Query the max dynamic shared memory for the current device (cached per device).
static int get_max_dynamic_smem() {
  static int cached = -1;
  if (cached >= 0) return cached;
  int device;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&cached, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return cached;
}

template <int STAGES, bool HAS_BIAS>
void launch_tinygemm2_impl(__nv_bfloat16* gA, __nv_bfloat16* gB, __nv_bfloat16* gC,
                           __nv_bfloat16* bias, int batch_size, int output_features,
                           int input_features, cudaStream_t stream, bool use_pdl) {
  static int const WARP_TILE_M = 16;
  static int const TILE_M = WARP_TILE_M;
  static int const TILE_N = 8;
  static int const TILE_K = 64;
  static int const STAGE_UNROLL = 4;

  CUtensorMap weight_map{};
  CUtensorMap activation_map{};

  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {(uint64_t)input_features, (uint64_t)output_features};
  uint64_t stride[rank - 1] = {input_features * sizeof(__nv_bfloat16)};
  uint32_t box_size[rank] = {TILE_K, TILE_M};
  uint32_t elem_stride[rank] = {1, 1};

  CUresult res = cuTensorMapEncodeTiled(
      &weight_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gB, size, stride,
      box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for weight_map with error code " << res;

  size[1] = batch_size;
  box_size[1] = TILE_N;

  res = cuTensorMapEncodeTiled(
      &activation_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gA, size,
      stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for activation_map with error code " << res;

  int smem_size =
      STAGES * STAGE_UNROLL *
      (TILE_M * TILE_K * sizeof(__nv_bfloat16) + TILE_N * TILE_K * sizeof(__nv_bfloat16));

  int tiles_m = (output_features + TILE_M - 1) / TILE_M;
  int tiles_n = (batch_size + TILE_N - 1) / TILE_N;

  dim3 grid(tiles_m, tiles_n);
  dim3 block(384);

  if (use_pdl) {
    // PDL path: use cudaLaunchKernelEx with programmatic stream serialization.
    auto status = cudaFuncSetAttribute(
        tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, true, HAS_BIAS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    status = cudaLaunchKernelEx(
        &config,
        &tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, true, HAS_BIAS>,
        gC, gA, gB, bias, output_features, batch_size, input_features, weight_map, activation_map);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaLaunchKernelEx failed: " << cudaGetErrorString(status);
  } else {
    // Standard path: no PDL intrinsics compiled into the kernel, plain <<<>>> launch.
    auto status = cudaFuncSetAttribute(
        tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, false, HAS_BIAS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);

    tinygemm_kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, false, HAS_BIAS>
        <<<grid, block, smem_size, stream>>>(gC, gA, gB, bias, output_features, batch_size,
                                             input_features, weight_map, activation_map);
    status = cudaGetLastError();
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "tinygemm_kernel launch failed: " << cudaGetErrorString(status);
  }
}

template <bool HAS_BIAS>
void launch_tinygemm2(__nv_bfloat16* gA, __nv_bfloat16* gB, __nv_bfloat16* gC, __nv_bfloat16* bias,
                      int batch_size, int output_features, int input_features, cudaStream_t stream,
                      bool use_pdl) {
  // Dynamic shared memory per pipeline stage (STAGE_UNROLL=4 stages per group):
  //   STAGE_UNROLL * (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(bf16)
  //   = 4 * (16*64 + 8*64) * 2 = 12288 bytes per STAGES increment
  static constexpr int SMEM_PER_STAGE_GROUP = 4 * (16 * 64 + 8 * 64) * 2;  // 12288

  int max_smem = get_max_dynamic_smem();
  int max_stages = max_smem / SMEM_PER_STAGE_GROUP;

  // Dispatch to the largest STAGES that fits. SM90/100f typically allows 16, SM120 allows 4.
  if (max_stages >= 16) {
    launch_tinygemm2_impl<16, HAS_BIAS>(gA, gB, gC, bias, batch_size, output_features,
                                        input_features, stream, use_pdl);
  } else if (max_stages >= 4) {
    // We must keep STAGES as a multiple of 4 (one slot per compute warp).
    launch_tinygemm2_impl<4, HAS_BIAS>(gA, gB, gC, bias, batch_size, output_features,
                                       input_features, stream, use_pdl);
  } else {
    TVM_FFI_ICHECK(false) << "Device has insufficient shared memory for tinygemm2 kernel ("
                          << max_smem << " bytes available, minimum " << 4 * SMEM_PER_STAGE_GROUP
                          << " bytes required)";
  }
}

}  // namespace tinygemm2
}  // namespace flashinfer

// proj port (s26): flashinfer 0.7.0 csrc/tinygemm2.cu (Apache-2.0), torch binding for H100 A/B only.
#define TG2_ARGS reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(w.data_ptr()), \
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nullptr, (int)x.size(0), (int)w.size(0), (int)x.size(1), stream, use_pdl
void tg2(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  flashinfer::tinygemm2::launch_tinygemm2<false>(TG2_ARGS);
}
#ifdef TG2_STAGES
// s26 sweep: explicit ring depth (STAGES x 12 KB smem; 16 = the stock choice on sm90)
void tg2s(torch::Tensor x, torch::Tensor w, torch::Tensor out, bool use_pdl, int64_t stages) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (stages == 4) flashinfer::tinygemm2::launch_tinygemm2_impl<4, false>(TG2_ARGS);
  else if (stages == 8) flashinfer::tinygemm2::launch_tinygemm2_impl<8, false>(TG2_ARGS);
  else flashinfer::tinygemm2::launch_tinygemm2_impl<16, false>(TG2_ARGS);
}
#endif
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
_G21_SRC = r'''// gemv21: m = 1 GEMV over gemv13's lossless 12-bit weights (same packing: byte plane, nibble words, escape headers)
// with the weight stream in a shared-memory TMA ring instead of registers: one bulk copy per plane per row, so the
// bytes in flight per SM (NS rows x 12 KB) no longer cost registers. Few registers per thread leave room for another
// kernel's CTAs on the same SM (the layer's csl attention beside the qkv GEMV, QG).
// Block = NG consumer groups of WG warps + 1 producer warp. Stage = one row (bytes, nibbles, header). Rows by
// global ticket (dynamic: no slow-SM tail); this CTA's it-th row goes to stage it % NS, group it % NG. A consumer thread owns fixed 16-weight groups of
// the row (x in registers as FP32), warp sums -> red[s][warp]; the group's last warp (smem ticket) adds the WG
// partials in warp order (deterministic); the producer warp, before refilling a stage, adds that row's WG partials,
// applies the escapes (true / scale - 2^-15 per entry, as gemv13) and stores.
// qg != nullptr: after a row's store, red.release +1 on qg[row >= nq] (QG: a PDL secondary spins on these).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g21 {
constexpr int ESC = 8;
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = decw<P0>(sw, nw, 0x9180u, 0x4140u);
    const unsigned q1 = decw<P0 + 1>(sw, nw, 0xB3A2u, 0x4342u);
    f[0] = __uint_as_float(q0 << 16); f[1] = __uint_as_float(q0 & 0xffff0000u);
    f[2] = __uint_as_float(q1 << 16); f[3] = __uint_as_float(q1 & 0xffff0000u);
}
__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* x) {
    float f[16];
    dec4<0>(s.x, nb.x, f); dec4<2>(s.y, nb.x, f + 4); dec4<0>(s.z, nb.y, f + 8); dec4<2>(s.w, nb.y, f + 12);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) { a0 = fmaf(f[i], x[i], a0); a1 = fmaf(f[i + 1], x[i + 1], a1); }
    return a0 + a1;
}
__device__ __forceinline__ void mbar_init(unsigned a, unsigned cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory");
}
#ifndef G21_SUSP
#define G21_SUSP 0
#endif
__device__ __forceinline__ void mbar_wait(unsigned a, unsigned par) {
    if (G21_SUSP)  // suspend-time hint: the warp sleeps until the phase completes instead of spinning on issue slots
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t@!p bra W;\n\t}"
                     ::"r"(a), "r"(par), "r"(G21_SUSP) : "memory");
    else
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}"
                     ::"r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned a) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned a, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, unsigned bytes, unsigned mb) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory");
}

template <int KK, int WG, int NG, int NS>
__global__ void __launch_bounds__((NG * WG + 1) * 32, 1) gemv(const unsigned char* __restrict__ S,
                                                              const unsigned char* __restrict__ N,
                                                              const float2* __restrict__ H,
                                                              const __nv_bfloat16* __restrict__ x,
                                                              __nv_bfloat16* __restrict__ out, int n, float scale,
                                                              unsigned* qg, int nq, unsigned long long* tk) {
    constexpr int GT = WG * 32, NJ = KK / 16 / GT, RB = KK + KK / 2 + ESC * 8, TQ = 4, DT = 3;
    static_assert(KK % (16 * GT) == 0 && NS % NG == 0, "shape");
    extern __shared__ __align__(128) unsigned char sm[];
    unsigned char* ring = sm;                                                  // [NS][RB]
    __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(sm + NS * RB);       // [KK] (escapes)
    float* red = reinterpret_cast<float*>(sm + NS * RB + KK * 2);             // [NS][WG]
    int* tick = reinterpret_cast<int*>(red + NS * WG);                        // [NS]
    int* rowid = tick + NS;                                                    // [NS]
    unsigned long long* bars = reinterpret_cast<unsigned long long*>(sm + ((NS * RB + KK * 2 + NS * WG * 4 + NS * 8 + 7) & ~7));
    const unsigned full0 = (unsigned)__cvta_generic_to_shared(bars), empty0 = full0 + NS * 8;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = warp / WG, wi = warp % WG, gt = wi * 32 + lane;
    // Rows by ticket: the producer keeps TQ tickets in flight (one thread's atomics on one address return increasing
    // values); it stops at its first ticket >= n, holding TQ - 1 more (all >= n): one call takes exactly
    // n + TQ * gridDim.x tickets, so tk is never reset and ticket % (n + TQ * gridDim.x) is the row (or a stop).
    // Static rows first (row = blockIdx.x + it * G for it < r0), tickets only for the dynamic tail (the last ~DT
    // rows per CTA: one counter cannot hand out every row, same-address atomics serialize at ~0.5 us per round).
    const int G = gridDim.x, r0 = max(0, n / G - DT), nd = n - r0 * G;
    const unsigned long long NT = (unsigned long long)nd + TQ * G;
    unsigned long long tq[TQ];
    // lane 1 of the producer warp owns the tickets: lane 0's release arrives (TMA issue) and lane 2's release counter
    // bumps (finalize) must not wait for a pending ticket atomic. Slot u % TQ serves iteration u >= r0 and is fetched
    // at iteration u - TQ (before the loop if that is < 0).
    if (grp == NG && lane == 1) {
#pragma unroll
        for (int j = 0; j < TQ; ++j)
            if (r0 + j < TQ) tq[(r0 + j) % TQ] = atomicAdd(tk, 1ull);
    }
    // x first (consumers: their fixed 16-weight groups, to registers; everyone: the bf16 copy for the escapes)
    uint4 xv[NJ][2];
    if (grp < NG) {
#pragma unroll
        for (int q = 0; q < NJ; ++q) {
            const uint4* p = reinterpret_cast<const uint4*>(x + 16 * (gt + q * GT));
            xv[q][0] = __ldg(p); xv[q][1] = __ldg(p + 1);
        }
    }
    for (int i = threadIdx.x; i < KK / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(xb)[i] = __ldg(reinterpret_cast<const uint4*>(x) + i);
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, WG); tick[s] = 0; }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (grp == NG) {  // producer warp: refills stages; finalizes each row (sum of the WG partials, escapes, store)
        // just before its stage is refilled (consumers only compute and release)
        auto finalize = [&](int f) {
            const int s = f % NS;
            mbar_wait(empty0 + 8 * s, (f / NS) & 1);
#ifdef G21_TRACE
            if (lane == 0 && f < 32 && blockIdx.x < 4) ((long long*)qg)[(blockIdx.x * 32 + f + NS) * 4 + 2] = clock64();
#endif
            const int row = rowid[s];
            float r = lane < WG ? red[s * WG + lane] : 0.f;
#pragma unroll
            for (int o = 16; o; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
            if (lane == 2) {
                const float2* he = reinterpret_cast<const float2*>(ring + s * RB + KK + KK / 2);
                for (int e = 0; e < ESC; ++e) {
                    const float2 h = he[e];
                    if (h.y == 0x1p-15f) break;  // slots fill in order; empty = 2^-15
                    const float xe = __bfloat162float(xb[__float_as_int(h.x)]);
                    r = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, r));
                }
                out[row] = __float2bfloat16(r * scale);
#ifndef G21_TRACE
                if (qg) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(qg + (row >= nq)) : "memory");
#endif
            }
            __syncwarp();
        };
        // ticket slot j of the queue is used at iterations it = j (mod TQ): unrolled so each slot keeps its own
        // register (a shifted queue would wait for every pending atomic). All of a call's tickets lie in one window
        // of NT: its base comes from the first ticket (one 64-bit division per CTA).
        unsigned long long tbase = ~0ull;
        int it = 0;
        bool done = false;
        while (!done) {
#pragma unroll
            for (int j = 0; j < TQ; ++j, ++it) {
                const int s = it % NS;
#ifdef G21_TRACE
                long long c0_ = clock64();
#endif
                if (it >= NS) finalize(it - NS);
#ifdef G21_TRACE
                long long c1_ = clock64();
                if (lane == 0 && it < 32 && blockIdx.x < 4) { ((long long*)qg)[(blockIdx.x * 32 + it) * 4] = c0_; ((long long*)qg)[(blockIdx.x * 32 + it) * 4 + 1] = c1_; }
#endif
                int row = blockIdx.x + it * G;
                if (lane == 1 && it >= r0) {
                    if (tbase == ~0ull) tbase = tq[j] - tq[j] % NT;
                    const int d = (int)(tq[j] - tbase);
                    row = d < nd ? r0 * G + d : n;
                }
                if (lane == 1 && row < n && it + TQ >= r0) tq[j] = atomicAdd(tk, 1ull);
                row = __shfl_sync(0xffffffffu, row, 1);
                if (row >= n) { done = true; break; }
                if (lane == 0) {
                    const unsigned d = (unsigned)__cvta_generic_to_shared(ring + s * RB), mb = full0 + 8 * s;
                    rowid[s] = row;
                    mbar_expect(mb, RB);
                    bulk(d, S + (long)row * KK, KK, mb);
                    bulk(d + KK, N + (long)row * (KK / 2), KK / 2, mb);
                    bulk(d + KK + KK / 2, H + (long)row * ESC, ESC * 8, mb);
                }
                __syncwarp();
#ifdef G21_TRACE
                if (lane == 0 && it < 32 && blockIdx.x < 4) ((long long*)qg)[(blockIdx.x * 32 + it) * 4 + 3] = clock64();
#endif
            }
        }
        for (int f = it >= NS ? it - NS + 1 : 0; f < it; ++f) finalize(f);  // (it - NS was finalized in the loop)
        if (lane == 0)
            for (int j = 0; j < NG; ++j) {  // one stop per group (those stages' last rows are finalized)
                const int s = (it + j) % NS;
                rowid[s] = -1;
                mbar_arrive(full0 + 8 * s);
            }
        return;
    }
    float xr[NJ][16];
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const unsigned v[8] = {xv[q][0].x, xv[q][0].y, xv[q][0].z, xv[q][0].w, xv[q][1].x, xv[q][1].y, xv[q][1].z, xv[q][1].w};
#pragma unroll
        for (int j = 0; j < 8; ++j) { xr[q][2 * j] = __uint_as_float(v[j] << 16); xr[q][2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u); }
    }
    for (int it = grp;; it += NG) {
        const int s = it % NS;
        mbar_wait(full0 + 8 * s, (it / NS) & 1);
        if (rowid[s] < 0) break;
        const unsigned char* st = ring + s * RB;
        float v = 0.f;
#ifndef G21_NOMATH
#pragma unroll
        for (int q = 0; q < NJ; ++q) {
            const int j = gt + q * GT;
            const uint4 b = *reinterpret_cast<const uint4*>(st + 16 * j);
            const uint2 c = *reinterpret_cast<const uint2*>(st + KK + 8 * j);
            v += dot16(b, c, xr[q]);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
        if (lane == 0) {
            red[s * WG + wi] = v;
            mbar_arrive(empty0 + 8 * s);  // release: orders the partial before the producer's finalize
        }
        __syncwarp();
    }
}
}  // namespace g21

void g21_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
              int64_t wg, int64_t ng, int64_t ns, int64_t blocks, torch::Tensor qg, int64_t nq, torch::Tensor tk) {
    const int k = x.size(1), n = out.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(x.size(0) == 1, "g21: m == 1");
    unsigned* q = qg.numel() ? (unsigned*)qg.data_ptr() : nullptr;
#define G21_CASE(K_, W_, G_, S_) \
    if (k == K_ && wg == W_ && ng == G_ && ns == S_) { \
        constexpr int RB = K_ + K_ / 2 + g21::ESC * 8; \
        const int smem = ((S_ * RB + K_ * 2 + S_ * W_ * 4 + S_ * 8 + 7) & ~7) + 2 * S_ * 8; \
        static bool set = false; \
        if (!set) { \
            cudaFuncSetAttribute(g21::gemv<K_, W_, G_, S_>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            cudaFuncSetAttribute(g21::gemv<K_, W_, G_, S_>, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
            set = true; } \
        g21::gemv<K_, W_, G_, S_><<<blocks, (G_ * W_ + 1) * 32, smem, st>>>((const unsigned char*)s.data_ptr(), \
            (const unsigned char*)nb.data_ptr(), (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), n, (float)scale, q, (int)nq, (unsigned long long*)tk.data_ptr()); return; }
    G21_CONFIGS
    TORCH_CHECK(false, "g21: no config");
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
# s16: gemv13 for o (k 1024) U2 V4 loses on the H100 tuner (11.3 vs gemv10 11.0: ~2 rounds per vb, latency-bound);
# s23: U2 V2 B2 on 264 blocks (512-thread blocks, 2/SM) wins on H100 g13.py (8.16-8.38 vs gemv10 8.53-8.58 us) but LOSES
# on the H200 (check 2722e8c4 tuner: g13 10.05 / U2V1B4 g528 10.11 vs gemv10 9.70): o m1 is latency-bound there. Dead.
# gemv13 built with G13_CPA: the weight chunks go through a cp.async smem ring (PF chunks in flight, no registers).
# H100 qkv m1 (2,2,1,3) 13.4 vs gemv13 14.4 us, pooler tie, unp1k 7.6 vs 7.7 (dev/g13.py XF=-DG13_CPA).
BF16_G13C_S16 = {"qkv_gate": [(1, 8192, 2, 2, 1, 3, 132, 256), (1, 8192, 2, 2, 1, 4, 132, 256)],
             "pooler": [(1, 8192, 2, 2, 1, 3, 132, 256), (1, 8192, 2, 2, 1, 4, 132, 256)],
             "unpooler": [(1, 4096, 2, 2, 1, 3, 132, 256), (1, 4096, 2, 2, 1, 4, 132, 256)]}
# s17: H200 diag (bytes in flight: H200 needs ~40% more than H100 at the same latency); H100 ties/loses.
BF16_G13C_S17 = {l: [(1, 8192, 2, 2, 1, 3, 132, 256), (1, 8192, 2, 2, 1, 4, 132, 256), (1, 8192, 2, 4, 1, 2, 132, 256),
                 (1, 8192, 1, 2, 1, 4, 132, 512)] for l in ("qkv_gate", "pooler")}
BF16_G13C = {}  # H200 check 29a2b16b: g13c loses every config (qkv 14.30+ vs g13 13.98, pooler 18.69+ vs 18.53)
G13_ESC = 8  # escapes per row (gemv13.cu / gemv14.cu ESC)
# BF16 tensor-core streaming GEMM over the same compressed weights (dev/gemv14.cu = gemv12 + decode), (m, kcu, wg, pf,
# bk, xp), chosen on the H100 (dev/g14.py); tuned in prepare() like the others.
# s16: qkv m8 without split-K (kcu 8, 96 blocks of 3 row groups): H100 15.2-15.6 vs (8, 2, 2, 4, 32, 1) 16.8 us
BF16_G14 = {"qkv_gate": [(8, 2, 2, 4, 32, 1), (8, 8, 8, 4, 96, 1), (8, 8, 8, 4, 96, 2), (8, 8, 8, 8, 96, 1)], "o": [(8, 1, 2, 4, 128, 1), (8, 1, 2, 8, 128, 1)],
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
# gemv21 (dev/gemv21.cu): m=1 over gemv13's 12-bit packing with the weight stream in a TMA ring (one row per stage),
# NG consumer groups of WG warps + a producer warp, rows by global ticket. (m, k, wg, ng, ns). H100 (dev/g21.py): slower
# than gemv13 (qkv 15.1 vs 13.7: HBM-bound there, per-row consumer latency ~0.8 us); built for the layer's QG overlap
# (few registers: csl fits beside it) and the H200, where gemv13 runs at ~2.6 TB/s.
# H200 (check 96165c87, first version: consumer-side finalize, static rows): qkv m1 17.7-18.3 vs gemv13 14.0, pooler 26.2
# vs 18.7, unp1k 11.2 vs g17 7.3 (tuner us): DEAD, not tuned any more (s13 NOTES).
BF16_G21 = {}
BF16_G17 = {"unpooler": [(1, 4, 512), (1, 4, 256), (8, 4, 256), (8, 4, 128)]}
# flashinfer 0.7.0 tinygemm_bf16 (TRT-LLM tinygemm2) ported to dev/tinygemm2.cu: TMA, 16 deep stages, n/16 blocks.
# H100 (dev/tg2.py, s26): unpooler-1k m8 6.53 vs 7.26 us (g17); loses elsewhere (m1 all, o/qkv/pooler m8, m>=16).
BF16_TG2 = {"unpooler": [(8,)]}
# gemv19 (dev/gemv19.cu, dev/g19_lib.py): TMA-ring streamer over the 12-bit weights, FP32 out zeroed in-kernel + red.add
# split-K partials, then a BF16 copy (g19_cvt). (m, kc, ap, ns, npre). Built for the layer (PDL W prefetch during the
# previous kernel); standalone (no predecessor) H100 it ties cuBLAS at m16 (pooler 24.8 vs 28.0) and loses at m32.
# (m, kc, ap, ns, npre, xb, ts, rg, rpw): xb x buffers, ts warps per row group, rg row groups per tile, rpw row groups per
# warp (dev/gemv19.cu). H200 diag (check 60ddb481): pooler m16 (1024, 8, 3, xb 1) 15.65 us vs pick 19.0 after a PDL predecessor.
BF16_G19 = {"qkv_gate": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)],
            "pooler": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)],
            "unpooler": [(16, 1024, 8, 3, 2, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1)], "o": [(16, 512, 8, 3, 2, 1, 1, 8, 1)]}
# gemv20 (dev/gemv20.cu) = gemv19 with wgmma consumers (x by an SW128 tensor map). _diag20 only (print, no picks):
# variant a = 1 block/SM (grid #SMs), b = 2 blocks/SM (grid 2 x #SMs, single-buffered A fragments, 1 partial buffer).
# (m, kc, ap, ns) per layer; npre = ns, xb 1.
DIAG20_S12 = {"a": [(16, 1024, 4, 4), (16, 1024, 8, 3), (32, 1024, 4, 4), (32, 1024, 8, 2)],
          "b": [(16, 1024, 4, 2), (16, 512, 4, 2), (32, 512, 4, 2), (32, 512, 2, 3)],
          "c": [(16, 1024, 4, 2), (16, 512, 4, 2), (32, 512, 4, 2), (32, 512, 2, 3)]}  # c = gemv19 (mma.sync), 2 blocks/SM
DIAG20 = {}  # s12 diag done (DIAG20_S12: all lose on the H200); nothing built
# _diag19 only (print, no picks): H200 behaviour of other configs in the layer case
DIAG19 = {}  # s10 diag configs measured (checks 9b09c536/60ddb481/f1319777): see NOTES-proj
# s15: after the escape-fixup rework (gemv19 fixups on the stored partials): the layer's unpool m16 config and AP4 m32
DIAG19_S15 = {"unpooler": [(16, 1024, 4, 4, 4, 2, 1, 8, 1), (32, 1024, 4, 4, 4, 1, 1, 8, 1)],
              "pooler": [(32, 1024, 4, 4, 4, 1, 1, 8, 1)], "qkv_gate": [(16, 1024, 4, 4, 4, 2, 1, 8, 1), (32, 1024, 4, 4, 4, 1, 1, 8, 1)]}
DIAG19 = {}  # s15 diag done (check ce84da07, NOTES-proj s15)
# s15: _diag19q only (print): qkv with BF16 out written by gemv19 itself (mode 3: tile tickets, self-cleaning FP32
# workspace, tile-major stages), standalone as the layer's first op (no predecessor), vs the picked kernel
# H200 (check 4e5fad5b): m16 19.6 vs lt 16.9, m32 23.3 vs 17.5 us: DEAD (no predecessor to hide g19's startup)
DIAG19Q_S15 = {"qkv_gate": [(16, 1024, 4, 4, 4, 2, 1, 8, 1), (16, 1024, 8, 3, 3, 1, 1, 8, 1), (32, 1024, 8, 2, 2, 1, 1, 8, 1),
                            (32, 1024, 4, 4, 4, 1, 1, 8, 1)]}
DIAG19Q = {}
DIAG_SLAB = {}  # s22 diag done (check ccf41026: qkv m32 bmm2 16.03 vs lt 17.50 H200 us; NOTES s22); was {"qkv_gate": (16, 32), "o": (32,), "pooler": (32,)}
DIAG_EXT = {}  # s25 diag done (check 5abb71f1: DeepGEMM bf16_gemm_nt ties/loses vs lt everywhere); was {"qkv_gate": (16, 32, 128, 256), "o": (128, 256), "pooler": (32, 128, 256), "unpooler": (128,)}
# s50: _diag19d only (print): H200 decomposition of gemv19 at m32/m16 in the layer's case (3 us PDL predecessor, npre = ns),
# each config rebuilt with diag defines: "" (= main), G19_NOW (no W loads: consumer-only), +G19_NOMMA (no decode/mma),
# +G19_NOX (decode, no x/mma), G19_NOFIX (no escape fixups); grids 132 and 108 (the layer's pooler grid). H100 proxy
# (NOTES s16/s34): skeleton 13.8, decode +5.9, mma+ldsm +7.5 proxy-us at pool m32 G72; the H200 itself never measured.
DIAG19D = {"pooler": [(32, 1024, 8, 2, 2, 1, 1, 8, 1), (16, 1024, 8, 3, 2, 1, 1, 8, 1)], "unpooler": [(32, 1024, 8, 2, 2, 1, 1, 8, 1)]}
DIAG19D_DEFS = ["", "G19_NOW", "G19_NOW G19_NOMMA", "G19_NOW G19_NOX", "G19_NOFIX"]
DIAG19D_GRIDS = (132, 108)
DIAG19_STREAM = []  # s17 H200: 50 MB 13.89, 28 MB 8.51 us at (24576, 4) (1-kernel graph 4.6 us subtracted); was [(24576, 4), (12288, 8)]  # (stage bytes, stages): 50 MB and 28 MB bare streams
# gemv22 (dev/gemv22.cu, dev/g22.py): m = 128/256 swap-AB wgmma GEMM over the 12-bit weights, tile-major stream-K (one
# run per block), escapes packed per stage (E entries), FP32 out (zero kernel + red.add), then a BF16 copy.
# (m, ap, ns, E). H100: qkv m128 20.8 vs cuBLAS 20.6, others lose (H100 TMA stream ~2.9 TB/s); built for the H200.
# H200 (check fb2fe4ee): loses every m128/m256 shape (qkv m128 22.1 vs lt 18.8, pooler 29.6 vs 24.4; m256 +10..19 us):
# per-SM smem ingest (W + x box per stage, x/W = 1.33 at m128) caps it, not HBM. DEAD; candidates were:
BF16_G22_S16 = {l: [(128, 2, 6, 32), (128, 4, 3, 32), (256, 2, 4, 32), (256, 2, 3, 32)] for l in ("qkv_gate", "o", "pooler", "unpooler")}
BF16_G22 = {}
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
        _G21_SRC.replace("G21_CONFIGS", " ".join("G21_CASE(%d,%d,%d,%d)" % c[1:] for c in sorted({c for v in BF16_G21.values() for c in v}))) + "\n" +
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
                    "void g21_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                    "int64_t, int64_t, int64_t, torch::Tensor, int64_t, torch::Tensor);\n"
                    "void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, "
                    "int64_t, int64_t);",
        cuda_sources=_SRC, functions=["pj_router2", "g7_gemv", "g10_gemv", "g12_gemm", "g13_gemv", "g14_gemm", "g15_gemm",
                                      "g16_gemm", "g17_gemv", "g21_gemv", "pj_router3"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj native build failed, using torch:", _e)
    _mod = None

try:
    _g19 = load_inline(
        name="muse_proj_g19a",
        cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); void g19_cvt(torch::Tensor, torch::Tensor); "
                    "void g19_gemm_bf16(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t); "
                    "void g19_pred(torch::Tensor, torch::Tensor, int64_t); void g19_stream(torch::Tensor, int64_t, int64_t, torch::Tensor);",
        cuda_sources=("#define G19_ONLY23\n" if DIAG19Q else "#define G19_ONLY2\n") + _G19_SRC.replace("G19_CONFIGS", " ".join(
            "G19_CASE(%d,%d,%d,%d,1,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in list(BF16_G19.values()) + list(DIAG19.values()) + list(DIAG19Q.values()) for c in v}))),
        functions=["g19_gemm", "g19_cvt", "g19_pred", "g19_stream", "g19_gemm_bf16"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv19 build failed:", _e)
    _g19 = None

try:
    _g13c = load_inline(
        name="muse_proj_g13c",
        cpp_sources="void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources="#define G13_CPA\n" + _G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(%d,%d,%d,%d,%d,%d)" % (c[1:6] + c[7:8]) for c in sorted({c for v in BF16_G13C.values() for c in v}))),
        functions=["g13_gemv"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) if BF16_G13C else None
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv13 cp.async build failed:", _e)
    _g13c = None

try:
    _g22 = load_inline(
        name="muse_proj_g22a",
        cpp_sources="void g22_gemm(torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_G22_SRC.replace("G22_CONFIGS", " ".join("G22_CASE(%d,%d,%d,%d)" % c for c in sorted({c for v in BF16_G22.values() for c in v}))),
        functions=["g22_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) if BF16_G22 else None
except Exception as _e:  # noqa: BLE001
    print("muse_proj gemv22 build failed:", _e)
    _g22 = None

_g20 = {}
for _v, _defs in [v for v in (("a", ""), ("b", "#define G20_MINB 2\n#define G20_SB\n#define G20_NPB 1\n"), ("c", "")) if v[0] in DIAG20]:
    try:
        if _v == "c":  # gemv19 with 2 blocks per SM, exposed as g20_gemm
            _g20[_v] = load_inline(
                name="muse_proj_g19c",
                cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources="#define G19_ONLY2\n#define G19_MINB 2\n#define G19_NPB 1\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
                    "G19_CASE(%d,%d,%d,%d,1,1,1,8,1)" % c for c in sorted(set(DIAG20[_v])))),
                functions=["g19_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
            _g20[_v].g20_gemm = _g20[_v].g19_gemm
            continue
        _g20[_v] = load_inline(
            name="muse_proj_g20" + _v,
            cpp_sources="void g20_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                        "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
            cuda_sources="#define G20_ONLY2\n" + _defs + _G20_SRC.replace("G20_CONFIGS", " ".join(
                "G20_CASE(%d,%d,%d,%d,1,1,1,8,1)" % c for c in sorted(set(DIAG20[_v])))),
            functions=["g20_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
    except Exception as _e:  # noqa: BLE001
        print("muse_proj gemv20", _v, "build failed:", _e)

try:
    _tg2 = load_inline(
        name="muse_proj_tg2a", cpp_sources="void tg2(torch::Tensor, torch::Tensor, torch::Tensor, bool);",
        cuda_sources=_TG2_SRC, functions=["tg2"], extra_ldflags=["-lcuda"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]) if BF16_TG2 else None
except Exception as _e:  # noqa: BLE001
    print("muse_proj tinygemm2 build failed:", _e)
    _tg2 = None

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


def _dec3(sm, code):
    """s35 DEC3 layout over groups of 8 weights (last dim = the kernels' dec8 unit: 8 bytes + 1 word): byte = c0 << 7 |
    mantissa; word: weight 2P + h's sign, c1, c2, c3 at bits (15, 8, 9, 10) + 16h + (0, -4, -8, 4)[P] (mod 32). Kernels
    (decw in dev/gemv*.cu): prmt with 0x38 for the exponent's top bits + one lop3 per BF16 pair. Checked lossless here."""
    sm, code = sm.long(), code.long()
    i = torch.arange(8, device=sm.device)
    off = 16 * (i % 2) + torch.tensor([0, -4, -8, 4], device=sm.device)[i // 2]
    pos = [(b + off) % 32 for b in (15, 8, 9, 10)]
    bits = [(sm >> 7) & 1, (code >> 1) & 1, (code >> 2) & 1, (code >> 3) & 1]
    v = sum(b << q for b, q in zip(bits, pos)).sum(-1)
    b3 = ((code & 1) << 7) | (sm & 0x7F)
    back = [(v.unsqueeze(-1) >> q) & 1 for q in pos]  # round trip: every bit lands in its own place
    if not (torch.equal(back[0], bits[0]) and torch.equal(back[1] | back[2] << 1 | back[3] << 2, code >> 1)
            and torch.equal(b3 >> 7, code & 1) and torch.equal(b3 & 0x7F, sm & 0x7F)):
        raise RuntimeError("DEC3 packing round trip failed")
    return b3.to(torch.uint8), torch.where(v >= 2 ** 31, v - 2 ** 32, v).int()


def _pack13(w):
    """gemv13 layout: flat byte plane, code words over 8 consecutive weights."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    s3, n3 = _dec3(sm.flatten().view(-1, 8), code.flatten().view(-1, 8))
    return s3.flatten().contiguous(), n3.contiguous(), hdr, scale


def _pack14(w, kc):
    """gemv14 layout: gemv12's atom order [ks][rg][atom][lane][8] for bytes and codes."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    perm = lambda t: t.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
    s3, n3 = _dec3(perm(sm), perm(code))
    return s3, n3.contiguous(), hdr, scale


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
    s3, nw = _dec3(perm(sm).reshape(k // kc, n // 16, kc // 32, 32, 2, 8), perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8))
    sw = s3.reshape(k // kc, n // 16, kc // 32, 32, 16).contiguous()
    nw = nw.contiguous()
    return sw, nw, hdr, scale


def _pack19(c, kc, ap, rg=8, tm=0):
    """_compress output -> gemv19 stages: stage (ks, 128-row tile, st) = [8 row groups x ap atoms x 32 lanes x 16 B of
    sign|mantissa][same x 8 B of codes], one contiguous block (one bulk copy)."""
    code, sm, hdr, scale = c
    n, k = code.shape
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()
    S, N = _dec3(perm(sm).reshape(k // kc, n // 16, kc // 32, 32, 2, 8), perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8))
    S = S.reshape(k // kc, n // 16, kc // 32, 32, 16)
    N = N.contiguous()
    ks, ntr, spk = k // kc, n // (16 * rg), kc // 32 // ap
    pm = (1, 0, 3, 2, 4) if tm else (0, 1, 3, 2, 4)  # tm: tile-major stages (tile, ks, st) for mode 3
    S = S.view(ks, ntr, rg, spk, ap * 512).permute(*pm).reshape(ks * ntr * spk, rg * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, rg, spk, ap * 256).permute(*pm).reshape(ks * ntr * spk, rg * ap * 256)
    return torch.cat([S, N], 1).contiguous(), hdr, scale


def _pack22(c, ap, E):
    """_compress output -> gemv22 stages, tile-major (u = 128-row tile * S + ks, S = k / 32ap): [8 row groups x ap atoms x
    32 lanes x 16 B sign|mantissa][same x 8 B codes][E escapes: int32 (row in tile | k in stage << 8, true / scale bits),
    padding (-1, 0)]. None if a stage has more than E escapes."""
    code, sm, hdr, scale = c
    n, k = code.shape
    kc = 32 * ap
    P, _, _ = _pack19(c, kc, ap, 8, 1)  # tm: (tile, ks, st), one stage per k-chunk
    ks, ntr = k // kc, n // 128
    EV = torch.full((ks * ntr, E, 2), -1, dtype=torch.int32, device=code.device)
    EV[:, :, 1] = 0
    rows, slots = (hdr[:, :, 1] != 2.0 ** -15).nonzero(as_tuple=True)
    if len(rows):
        col = hdr[rows, slots, 0].view(torch.int32).long()
        u = (rows // 128) * ks + col // kc
        o = torch.argsort(u * (1 << 20) + rows * 8 + slots)
        rows, col, u, hy = rows[o], col[o], u[o], hdr[rows[o], slots[o], 1]
        rank = torch.arange(len(u), device=code.device) - torch.searchsorted(u, u)
        if int(rank.max()) >= E:
            return None
        EV[u, rank, 0] = ((rows % 128) | ((col % kc) << 8)).int()
        EV[u, rank, 1] = hy.view(torch.int32)
    return torch.cat([P, EV.view(torch.uint8).view(ks * ntr, -1)], 1).contiguous(), scale


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    if _g22 is not None:
        for name, cfgs in BF16_G22.items():
            c = _compress(weights[name])
            if c is None:
                continue
            for cf in cfgs:
                if ("g22", name, cf[1], cf[3]) not in st:
                    pk = _pack22(c, cf[1], cf[3])
                    if pk is not None:
                        st[("g22", name, cf[1], cf[3])] = pk
        st["ws22"] = torch.zeros(256 * 8192, dtype=torch.float32, device=weights["o"].device)
    if _g19 is not None:
        for name in BF16_G19:
            cfgs = BF16_G19[name] + DIAG19.get(name, [])
            c = _compress(weights[name])
            if c is None:
                continue
            for cf in cfgs:
                if ("g19", name, cf[1], cf[2], cf[7]) not in st:
                    st[("g19", name, cf[1], cf[2], cf[7])] = _pack19(c, cf[1], cf[2], cf[7])
            for cf in [c_ for v_ in DIAG20.values() for c_ in v_] if _g20 else []:
                if ("g19", name, cf[1], cf[2], 8) not in st and weights[name].shape[1] % cf[1] == 0:
                    st[("g19", name, cf[1], cf[2], 8)] = _pack19(c, cf[1], cf[2], 8)
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
    _diag19q(st)
    _diag20(st)
    _diag19_stream()
    _diag19d(st)
    _diag_slab(st)
    _diag_ext(st)
    st["rpick"] = _tune_router(st)
    return st


def _run(st, layer, x, out, cand):
    """One BF16 projection by candidate: None = F.linear, ("g12", cfg) = dev/gemv12.cu, ("lt", cfg) = cuBLASLt."""
    if cand is None:
        return F.linear(x, st[layer][:out.shape[1]])
    kind, c = cand
    if kind == "g10":
        _mod.g10_gemv(st[layer], x, out, *c[2:])  # W[:n] is the first n rows, in place
    elif kind == "g13c":
        sm, nib, hdr, scale = st[("g13", layer)]
        _g13c.g13_gemv(sm, nib, hdr, x, out, scale, *c[2:])
    elif kind == "g13":
        sm, nib, hdr, scale = st[("g13", layer)]
        _mod.g13_gemv(sm, nib, hdr, x, out, scale, *c[2:])
    elif kind == "g21":
        sm, nib, hdr, scale = st[("g13", layer)]
        _mod.g21_gemv(sm, nib, hdr, x, out, scale, *c[2:], _SMS, st["cnt"][:0], 0, st[("tk21", layer, c)])
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
    elif kind == "g22":
        P, scale = st[("g22", layer, c[1], c[3])]
        o32 = st["ws22"][:out.numel()].view(out.shape)
        _g22.g22_gemm(P, scale, x, o32, c[1], c[2], c[3], _SMS, 0, 0)
        _g19.g19_cvt(o32, out) if _g19 is not None else out.copy_(o32)
    elif kind == "g17":
        _mod.g17_gemv(st[layer], x, out, *c)  # W[:n] prefix rows, in place
    elif kind == "tg2":
        _tg2.tg2(x, st[layer][:out.shape[1]], out, False)  # W[:n] prefix rows, BF16 out
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
        for layer, cfgs in BF16_G13C.items() if _g13c is not None else []:
            if ("g13", layer) in st:
                for c in cfgs:
                    shapes.setdefault((layer, c[0]), []).append(("g13c", c))
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
        for layer, cfgs in BF16_G21.items():
            if ("g13", layer) in st:
                for c in cfgs:
                    st[("tk21", layer, c)] = torch.zeros(1, dtype=torch.int64, device=st[layer].device)
                    shapes.setdefault((layer, c[0]), []).append(("g21", c))
    if _g19 is not None:
        for layer, cfgs in BF16_G19.items():
            for c in cfgs:
                if ("g19", layer, c[1], c[2], c[7]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g19", c))
    if _g22 is not None:
        for layer, cfgs in BF16_G22.items():
            for c in cfgs:
                if ("g22", layer, c[1], c[3]) in st:
                    shapes.setdefault((layer, c[0]), []).append(("g22", c))
    if _tg2 is not None:
        for layer, cfgs in BF16_TG2.items():
            for c in cfgs:
                shapes.setdefault((layer, c[0]), []).append(("tg2", c))
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


def _diag19d(st):
    """Print only: per DIAG19D config and grid, us over the 3 us predecessor for each DIAG19D_DEFS build."""
    if _g19 is None or not DIAG19D:
        return
    mods = []
    for i, defs in enumerate(DIAG19D_DEFS):
        try:
            mods.append(load_inline(
                name="muse_proj_g19d%d" % i, cpp_sources="void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                "torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources="".join("#define %s\n" % d for d in defs.split()) + "#define G19_ONLY2\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
                    "G19_CASE(%d,%d,%d,%d,1,%d,%d,%d,%d)" % (c[:4] + c[5:]) for c in sorted({c for v in DIAG19D.values() for c in v}))),
                functions=["g19_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"]))
        except Exception as e:  # noqa: BLE001
            print("diag19d build failed:", defs, e)
            mods.append(None)
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs in DIAG19D.items():
        for c in cfgs:
            if ("g19", layer, c[1], c[2], c[7]) not in st:
                continue
            m, w = c[0], st[layer]
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
            o32 = st["ws19"][:m * w.shape[0]].view(m, w.shape[0])
            for grid in DIAG19D_GRIDS:
                fns, names = [lambda: _g19.g19_pred(xs, x, 3000)], []
                for defs, mod in zip(DIAG19D_DEFS, mods):
                    if mod is None:
                        continue
                    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")  # one epoch counter per (build, grid)
                    fns.append(lambda mod=mod, ctr=ctr, grid=grid: (_g19.g19_pred(xs, x, 3000), mod.g19_gemm(
                        P, hdr, scale, x, o32, ctr, c[1], c[2], c[3], 1, grid, 0, 2, c[3], *c[5:])))
                    names.append(defs.replace("G19_", "") or "main")
                try:
                    med = _graph_medians(fns)
                except Exception as e:  # noqa: BLE001
                    print("diag19d", layer, m, grid, "failed:", e)
                    continue
                print(f"diag19d {layer} m{m} {c} G{grid} (pred {med[0] * 1e3:.2f}):", " | ".join(
                    f"{nm_} +{(t - med[0]) * 1e3:.2f}" for nm_, t in zip(names, med[1:])))


def _diag19q(st):
    """Print only: qkv by gemv19 mode 3 (BF16 out, no predecessor, tm 0/1) vs the picked kernel; plus max abs diff."""
    if _g19 is None:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs in DIAG19Q.items():
        c_ = _compress(st[layer])
        if c_ is None:
            continue
        w = st[layer]
        ctr = torch.zeros(4096, dtype=torch.int32, device="cuda")
        for c in cfgs:
            m = c[0]
            x = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            out = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device="cuda")
            ob = torch.empty_like(out)
            ws = torch.zeros(m, w.shape[0], dtype=torch.float32, device="cuda")
            pick = st["pick"].get((layer, m))
            fns = [lambda: _run(st, layer, x, out, pick)]
            for tm in (0, 1):
                P, hdr, scale = _pack19(c_, c[1], c[2], c[7], tm)
                fns.append(lambda P=P, hdr=hdr, scale=scale, tm=tm: _g19.g19_gemm_bf16(P, hdr, scale, x, ws, ob, ctr, c[1], c[2], c[3], _SMS, c[4], *c[5:], tm))
            med = _graph_medians(fns)
            fns[0](); fns[2](); torch.cuda.synchronize()
            d = (ob.float() - out.float()).abs().max().item()
            print(f"diag19q {layer} m{m} {c}: standalone: pick {pick} {med[0] * 1e3:.2f} us, g19 BF16 out tm0 {med[1] * 1e3:.2f} "
                  f"tm1 {med[2] * 1e3:.2f} us (max |diff| vs pick {d:.3g}, ws clean {int(torch.count_nonzero(ws)) == 0})")
            del P


def _diag20(st):
    """Print only: gemv20 variants vs gemv19 (BF16_G19 configs) in the layer's case (3 us PDL predecessor), H200 us."""
    if _g19 is None or not _g20:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, cfgs19 in BF16_G19.items():
        for m in (16, 32):
            w = st[layer]
            xs = torch.randn(m, w.shape[1], generator=g, device="cuda").to(torch.bfloat16)
            x = xs.clone()
            o32 = st["ws19"][:m * w.shape[0]].view(m, w.shape[0])
            ref = F.linear(xs.float(), w.float())
            fns, names = [lambda: _g19.g19_pred(xs, x, 3000)], []
            for c in cfgs19:
                if c[0] == m and ("g19", layer, c[1], c[2], c[7]) in st:
                    P, hdr, scale = st[("g19", layer, c[1], c[2], c[7])]
                    fns.append(lambda P=P, hdr=hdr, scale=scale, c=c: (_g19.g19_pred(xs, x, 3000), _g19.g19_gemm(P, hdr, scale, x, o32, st["ctr19"], c[1], c[2], c[3], 1, _SMS, 0, 2, c[3], *c[5:])))
                    names.append("g19%s" % (c[1:4],))
            for v, cfgs in DIAG20.items():
                if v not in _g20:
                    continue
                grid = _SMS * (2 if v in "bc" else 1)
                ctr = st.setdefault(("ctr20", v), torch.zeros(2, dtype=torch.int32, device="cuda"))
                for c in cfgs:
                    if c[0] != m or ("g19", layer, c[1], c[2], 8) not in st:
                        continue
                    P, hdr, scale = st[("g19", layer, c[1], c[2], 8)]
                    run = lambda P=P, hdr=hdr, scale=scale, c=c, v=v, grid=grid, ctr=ctr: _g20[v].g20_gemm(P, hdr, scale, x, o32, ctr, c[1], c[2], c[3], 1, grid, 0, 2, c[3], 1, 1, 8, 1)
                    try:
                        o32.fill_(7.0)
                        torch.cuda.synchronize()
                        run()
                        torch.cuda.synchronize()
                        err = ((o32 - ref).norm() / ref.norm()).item()
                    except Exception as e:  # noqa: BLE001
                        print("diag20", layer, m, v, c, "failed:", e)
                        continue
                    fns.append(lambda run=run: (_g19.g19_pred(xs, x, 3000), run()))
                    names.append("g20%s%s err %.1e" % (v, c[1:], err))
            med = _graph_medians(fns)
            print(f"diag20 {layer} m{m} (pred {med[0] * 1e3:.2f}):", " | ".join(f"{nm_} +{(t - med[0]) * 1e3:.2f}" for nm_, t in zip(names, med[1:])))


def _diag_slab(st):
    """Print only: the layer's FP32-consumer option, x @ W^T split over k into ns FP32 slabs by one strided batched
    GEMM (the consumer sums the slabs; no split-K reduce kernel) vs this shape's pick (BF16 out)."""
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for layer, ms in DIAG_SLAB.items():
        w = st[layer]
        n, k = w.shape
        for m in ms:
            x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
            pick = st["pick"].get((layer, m))
            fns = [lambda: _run(st, layer, x, out, pick) if pick is not None else torch.mm(x, w.t(), out=out),
                   lambda: torch.mm(x, w.t(), out_dtype=torch.float32)]
            nss = [ns for ns in (2, 3, 4, 8) if k % ns == 0]
            for ns in nss:
                kc = k // ns
                fns.append(lambda ns=ns, kc=kc: torch.bmm(x.view(m, ns, kc).permute(1, 0, 2), w.view(n, ns, kc).permute(1, 2, 0),
                                                          out_dtype=torch.float32))
            med = _graph_medians(fns)
            print(f"diagslab {layer} m{m}: pick {pick} {med[0] * 1e3:.2f} mm32 {med[1] * 1e3:.2f}",
                  " ".join(f"bmm{ns}:{t * 1e3:.2f}" for ns, t in zip(nss, med[2:])))


def _diag_ext(st):
    """Print only: which GEMM entry points the judge image's libraries export, and DeepGEMM's BF16 GEMM (if present)
    vs this shape's pick, as graph replays after the cold-L2 flush (+ max abs diff to the pick's output)."""
    import importlib
    import time
    t0 = time.time()
    libs = {}
    for name in ("deep_gemm", "flashinfer", "flashinfer.gemm", "sgl_kernel"):
        try:
            mod = importlib.import_module(name)
            libs[name] = mod
            names = sorted(a for a in dir(mod) if any(s_ in a.lower() for s_ in ("bf16", "gemm", "mm", "sms")))
            print(f"diagext {name} {getattr(mod, '__version__', '?')}: {' '.join(names)[:2000]}")
        except Exception as e:
            print(f"diagext {name} import failed: {type(e).__name__}: {str(e)[:200]}")
    fn = getattr(libs.get("deep_gemm"), "bf16_gemm_nt", None)
    if fn is None or not DIAG_EXT:
        return
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    for layer, ms in DIAG_EXT.items():
        for m in ms:
            if time.time() - t0 > 240:
                print("diagext: time cap")
                return
            try:
                w = st[layer]
                n, k = w.shape
                x = torch.randn(m, k, generator=g, device=w.device).to(torch.bfloat16)
                out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
                ref = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
                pick = st["pick"].get((layer, m))
                runp = (lambda: _run(st, layer, x, ref, pick)) if pick is not None else (lambda: torch.mm(x, w.t(), out=ref))
                runp()
                torch.cuda.synchronize()
                t1 = time.time()
                fn(x, w, out)
                torch.cuda.synchronize()
                jit = time.time() - t1
                err = (out.float() - ref.float()).abs().max().item()
                med = _graph_medians([runp, lambda: fn(x, w, out)])
                print(f"diagext {layer} m{m}: pick {pick} {med[0] * 1e3:.2f} deep_gemm bf16_gemm_nt {med[1] * 1e3:.2f} us "
                      f"(first call {jit:.1f} s, maxdiff {err:.3g})")
            except Exception as e:
                print(f"diagext {layer} m{m} failed: {type(e).__name__}: {str(e)[:300]}")


def _diag19_stream():
    if _g19 is None:
        return
    for mb in (50, 28):
        w = torch.zeros(mb << 20, dtype=torch.uint8, device="cuda")
        o = torch.zeros(1, device="cuda")
        med = _graph_medians([lambda: o.zero_()] + [(lambda c=c: _g19.g19_stream(w, c[0], c[1], o)) for c in DIAG19_STREAM])
        print(f"diag19 bare TMA stream {mb} MB, 132 blocks:", " ".join(f"{c}:{(t - med[0]) * 1e3:.2f}" for c, t in zip(DIAG19_STREAM, med[1:])),
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
