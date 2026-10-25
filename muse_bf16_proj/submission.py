# okc: problem=muse_bf16_proj
"""Dense projections of Muse Spark Open: out = x @ W[:n].T (BF16; the router FP32).

Router (FP32 weights, FP32 output): up to 8 tokens a native true-FP32 split-K kernel (FP32 weights, FP32 FMA,
FP32 split partials), as SGLang's F.linear in FP32. Above 8 tokens SGLang's BF16 hi + lo split with FP32
accumulation: 9..32 tokens a native tensor-core kernel (dev/gemv6.cu: mma.sync with hi and lo tiles into one FP32
accumulator, split K reduced in FP32 over a thread-block cluster through distributed shared memory); above 32 one
cuBLAS GEMM against the stacked [hi; lo] weight (FP32 out) and an FP32 add of the two halves. BF16 projections: at 1 token a native grid-stride GEMV (dev/gemv10.cu: W read in place, row-major, the whole grid
sweeping one contiguous window; FP32 FMA and FP32 reductions, no split K), else cuBLAS (as SGLang's
bf16_gemm_dispatch on H200).
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


_ROUTER_SRC = r'''// Router at m <= 8 in true FP32 (as SGLang's F.linear): out[m, 256] = x[m, k] (BF16 -> FP32) @ W[256, k]^T (FP32).
// Block = (RB rows, K chunk of KB): x[:, chunk] staged in smem as FP32; warp w takes rows r0 + w*RW .. +RW, each
// lane FP32 float4 loads of W at k = 4 (lane + 32 j) (all KB/128 loads in flight), FMA against smem x, warp
// shuffle reduce. Split K: FP32 red.global into a workspace, the last block of a row block writes the output and
// re-zeroes workspace + counter.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pr {
template <int M, int KB, int RW, int NWARP>
__global__ void __launch_bounds__(NWARP * 32) router(const float4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                     float* __restrict__ out, float* __restrict__ ws,
                                                     int* __restrict__ cnt, int m, int n, int k) {
    constexpr int J = KB / 128, RB = NWARP * RW;
    __shared__ __align__(16) float xs[M][KB];
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = blockIdx.x * RB + warp * RW, k0 = blockIdx.y * KB;
    constexpr int XV = (M * KB / 8 + NWARP * 32 - 1) / (NWARP * 32);
    uint4 xr[XV];
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * NWARP * 32, t = i / (KB / 8), c = (i % (KB / 8)) * 8;
        xr[q] = make_uint4(0, 0, 0, 0);
        if (i < M * KB / 8 && t < m) xr[q] = __ldcs(reinterpret_cast<const uint4*>(x + (size_t)t * k + k0 + c));
    }
    float4 a[RW][J];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = __ldcs(w + ((size_t)(r0 + r) * k + k0) / 4 + lane + 32 * j);
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * NWARP * 32, t = i / (KB / 8), c = (i % (KB / 8)) * 8;
        if (i >= M * KB / 8) break;
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xr[q]);
        float v[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); v[2 * e] = f.x; v[2 * e + 1] = f.y; }
        *reinterpret_cast<float4*>(&xs[t][c]) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4*>(&xs[t][c + 4]) = make_float4(v[4], v[5], v[6], v[7]);
    }
    __syncthreads();
    float acc[RW][M];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) acc[r][t] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float4 xv = *reinterpret_cast<const float4*>(&xs[t][4 * (lane + 32 * j)]);
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc[r][t] = fmaf(a[r][j].x, xv.x, acc[r][t]);
                acc[r][t] = fmaf(a[r][j].y, xv.y, acc[r][t]);
                acc[r][t] = fmaf(a[r][j].z, xv.z, acc[r][t]);
                acc[r][t] = fmaf(a[r][j].w, xv.w, acc[r][t]);
            }
        }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t)
#pragma unroll
            for (int o = 16; o; o >>= 1) acc[r][t] += __shfl_xor_sync(0xffffffffu, acc[r][t], o);
    const bool split = gridDim.y > 1;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int t = 0; t < M; ++t)
                if (t < m) {
                    if (split) atomicAdd(ws + (size_t)t * n + r0 + r, acc[r][t]);
                    else out[(size_t)t * n + r0 + r] = acc[r][t];
                }
    }
    if (!split) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(cnt + blockIdx.x, 1);
        last = prev == (int)gridDim.y - 1;
        if (last) cnt[blockIdx.x] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < m * RB; i += NWARP * 32) {
        const int t = i / RB, row = blockIdx.x * RB + i % RB;
        float* p = ws + (size_t)t * n + row;
        out[(size_t)t * n + row] = __ldcg(p);
        __stcg(p, 0.f);
    }
}
}  // namespace pr

#define PR_CASE(M, KB, RW, NWARP)                                                                                   \
    if (mp == M && kb == KB && rw == RW && nwarp == NWARP) {                                                        \
        dim3 grid(n / (RW * NWARP), k / KB);                                                                        \
        pr::router<M, KB, RW, NWARP><<<grid, NWARP * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(                   \
            reinterpret_cast<const float4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),    \
            out.data_ptr<float>(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k);                             \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                             \
        return;                                                                                                     \
    }

void pj_router(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t mp,
               int64_t kb, int64_t rw, int64_t nwarp) {
    const int m = (int)x.size(0), n = (int)w.size(0), k = (int)w.size(1);
    TORCH_CHECK(m <= mp, "pr: m");
    PR_CONFIGS
    TORCH_CHECK(false, "pr: no such config");
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

# Router FP32 kernel configs (mp, kb, rw, nwarp), chosen on the H100 (dev/rt.py).
ROUTER_F32 = [(1, (1, 2048, 1, 8)), (8, (8, 512, 2, 8)), (16, (16, 512, 2, 8))]
ROUTER_F32 = [c for c in ROUTER_F32 if c[0] <= 8]
# Router hi/lo tensor-core configs (n, mt, nwr, nwk, stages, na, cluster, token groups), chosen on the H100 (dev/g5.py).
ROUTER_HL = [(16, (256, 2, 2, 4, 8, 2, 8, 1)), (32, (256, 4, 2, 2, 8, 2, 8, 1))]
# BF16 grid-stride GEMV configs per layer: (m, k, u, nv, minb, pf, blocks), chosen on the H100 (dev/g10.py).
BF16_G10 = {"qkv_gate": [(1, 8192, 4, 4, 1, 1, 132)], "pooler": [(1, 8192, 4, 4, 1, 1, 132)],
            "o": [(1, 1024, 2, 4, 2, 1, 264)], "unpooler": [(1, 4096, 2, 4, 1, 1, 132)]}
_G7 = sorted({c[1:6] for _, c in ROUTER_HL})
_SRC = (_ROUTER_SRC.replace("PR_CONFIGS", " ".join("PR_CASE(%d,%d,%d,%d)" % c for _, c in ROUTER_F32)) + "\n" +
        _GEMV_SRC.replace("G7_CONFIGS", " ".join("G7_CASE(%d,%d,%d,%d,%d)" % c for c in _G7)) + "\n" +
        _G10_SRC.replace("G10_CONFIGS", " ".join("G10_CASE(%d,%d,%d,%d,%d,%d)" % c[:6] for v in BF16_G10.values() for c in v)))
try:
    from torch.utils.cpp_extension import load_inline
    _mod = load_inline(
        name="muse_proj_v5",
        cpp_sources="void pj_router(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
                    "int64_t, int64_t, int64_t);\n"
                    "void g7_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t);\n"
                    "void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_SRC, functions=["pj_router", "g7_gemv", "g10_gemv"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("muse_proj native build failed, using torch:", _e)
    _mod = None

ROUTER_FP32_MAX_TOKENS = 8


def _pack(w):
    n, k = w.shape
    return w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 256)


def prepare(weights):
    st = {name: w.contiguous() for name, w in weights.items()}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["router_hl_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [k, 512]: one GEMM gives both halves
    # [stripe][k-tile][hi/lo][lane][8]: a 16x16 tile's mma A fragments, 512 contiguous bytes
    st["router_hl_p"] = torch.stack([_pack(hi), _pack(lo)], 2).contiguous()
    # Split-K workspace and per-row-block counters: always left zeroed by the kernels.
    st["ws"] = torch.zeros(32 * 8192, dtype=torch.float32, device=r.device)
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device=r.device)
    return st


def _router(x, st, m):
    if _mod is not None:
        for mmax, cfg in ROUTER_F32:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.pj_router(st["router"], x, out, st["ws"], st["cnt"], *cfg)
                return out
        for mmax, cfg in ROUTER_HL:
            if m <= mmax:
                out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
                _mod.g7_gemv(st["router_hl_p"], x, out, *cfg)
                return out
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), st["router"])
    acc = torch.mm(x, st["router_hl_t"], out_dtype=torch.float32)
    return acc[:, :256] + acc[:, 256:]


def kernel_fn(data, st):
    x, cfg = data["x"], data["config"]
    layer = cfg["layer"]
    if layer == "router":
        return _router(x, st, cfg["m"])
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    if _mod is not None:
        for c in BF16_G10.get(layer, ()):
            if m == c[0] and k == c[1]:
                out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
                _mod.g10_gemv(st[layer], x, out, *c[2:])  # W[:n] is the first n rows, in place
                return out
    return F.linear(x, st[layer][:n])
