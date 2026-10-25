# okc: problem=muse_moe_w4a16
"""Routed experts, NVFP4 W4A16, own CUDA kernels (moe agent).

Four launches per call:
  route : one CTA buckets this GPU's (token, slot) routes by local expert (counts, offsets,
          active-expert list sorted by load, sorted route ids).
  gemm13: persistent grouped GEMM over (active expert, row tile) items. Weights are NVFP4
          repacked in prepare() into mma.sync A-fragment order with the 4-bit codes
          bit-permuted so that dequant is ~1 int op per element (masks/shifts land each code
          straight in a bf16's sign/exponent/mantissa bits, a tiny bf16 x a bias-folded block
          scale gives the exact e2m1 * e4m3 value). bf16 mma m16n8k16, FP32 accumulate, 8 warps
          split K, smem reduce; epilogue applies the global scale and SiLU(gate) * up -> bf16.
  gemm2 : the same kernel on w2, writes y bf16 per route.
  norm  : per token, (1 + w) * y, RMSNorm, weighted sum over its local slots (FP32) -> bf16.
"""

import hashlib
from typing import TypedDict

import torch


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8]
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


H = 4096
I = 4096
TOP_K = 8
LOCAL = 32
POST_NORM_EPS = 1e-8

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cuda.h>
#include <cudaTypedefs.h>

#define KD 4096          // K of both GEMMs
#define NOUT 4096        // output rows of both GEMMs (gate/up each I, w2 H)

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// 8 e2m1 codes (bit-permuted in prepare) -> 4 bf16x2 A registers (rows g, g+8, g, g+8),
// each = code * 2^-126 (exact, subnormals for 0/0.5) times the block scale * 2^118.
// Output 0 = q & M, outputs 1/2 = rotl(q, 10/13) & M, output 3 = the 8 leftover bits (mask 0x001B001B)
// moved by one multiply (x 2^3 + 2^8 + 2^14: the shifted copies never overlap, so no carries) & M.
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M;
  const uint32_t o1 = __funnelshift_l(q, q, 10) & M;
  const uint32_t o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0);
  a[1] = bmul(o1, s1);
  a[2] = bmul(o2, s0);
  a[3] = bmul(o3, s1);
}

__device__ __forceinline__ uint32_t u4get(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// meta: [0] nact, [1..33) active experts (by load, descending), [33..66) offsets (33 entries).
template <int R, int G, bool GATED, int MB>
__global__ void __launch_bounds__(256, MB) moe_gemm(
    const uint4* __restrict__ Wq, const uint4* __restrict__ Ws, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
    const int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  constexpr int NR = GATED ? 2 * R : R;
  constexpr int NRT = GATED ? 2 * NOUT / 16 : NOUT / 16;  // packed row tiles per expert
  constexpr int TPE = NOUT / 16 / R;                        // items per expert
  __shared__ uint32_t slut[256];
  __shared__ float red[NR * G * 4 * 32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  slut[tid] = lut[tid];
  const int nact = meta[0];
  const int nitems = nact * TPE;
  __syncthreads();
  for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
    const int a = item / TPE, tg = item - a * TPE;
    const int e = meta[1 + a];
    const int off = meta[33 + e];
    const int n = meta[34 + e] - off;
    const uint4* wb = Wq + (size_t)e * NRT * 64 * 32;
    const uint4* sb = Ws + (size_t)e * NRT * 8 * 32;
    int rts[NR];
#pragma unroll
    for (int i = 0; i < R; ++i) {
      rts[i] = tg * R + i;
      if (GATED) rts[R + i] = NOUT / 16 + tg * R + i;
    }
    uint4 sc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) sc[i] = __ldg(sb + ((size_t)rts[i] * 8 + warp) * 32 + lane);
    const float gsc = gs[e] * 256.0f;
    for (int p0 = 0; p0 < n; p0 += 8 * G) {
      for (int i = tid; i < NR * G * 128; i += 256) red[i] = 0.f;
      float acc[NR][G][4];
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[i][gi][r] = 0.f;
      const uint4* xr[G];
      bool xv[G];
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int idx = p0 + gi * 8 + g;
        xv[gi] = idx < n;
        int row = 0;
        if (xv[gi]) row = GATED ? (sroutes[off + idx] >> 3) : (off + idx);
        xr[gi] = reinterpret_cast<const uint4*>(X + (size_t)row * KD + warp * 512 + 16 * t);
      }
      constexpr int PF = (NR * 8 <= 16) ? 8 : (NR * 4 <= 16) ? 4 : 2;  // k chunks loaded per batch
#pragma unroll
      for (int c0 = 0; c0 < 8; c0 += PF) {
        uint4 w[PF][NR];
        uint4 xb[PF][G][2];
#pragma unroll
        for (int c = 0; c < PF; ++c)
#pragma unroll
          for (int i = 0; i < NR; ++i)
            w[c][i] = ld_stream(wb + ((size_t)rts[i] * 64 + warp * 8 + c0 + c) * 32 + lane);
#pragma unroll
        for (int c = 0; c < PF; ++c)
#pragma unroll
          for (int gi = 0; gi < G; ++gi) {
            if (xv[gi]) {
              xb[c][gi][0] = __ldg(xr[gi] + (c0 + c) * 8);
              xb[c][gi][1] = __ldg(xr[gi] + (c0 + c) * 8 + 1);
            } else {
              xb[c][gi][0] = make_uint4(0, 0, 0, 0);
              xb[c][gi][1] = make_uint4(0, 0, 0, 0);
            }
          }
#pragma unroll
        for (int c = 0; c < PF; ++c) {
          const int kc = c0 + c;
#pragma unroll
          for (int i = 0; i < NR; ++i) {
            const uint32_t sw = u4get(sc[i], kc >> 1) >> ((kc & 1) * 16);
            const uint32_t s0 = slut[sw & 0xFF], s1 = slut[(sw >> 8) & 0xFF];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              uint32_t af[4];
              dq(u4get(w[c][i], j), s0, s1, af);
#pragma unroll
              for (int gi = 0; gi < G; ++gi) {
                const uint32_t b0 = u4get(xb[c][gi][j >> 1], (j & 1) * 2);
                const uint32_t b1 = u4get(xb[c][gi][j >> 1], (j & 1) * 2 + 1);
                mma16816(acc[i][gi], af, b0, b1);
              }
            }
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
#pragma unroll
          for (int r = 0; r < 4; ++r) atomicAdd(&red[((i * G + gi) * 4 + r) * 32 + lane], acc[i][gi][r]);
      __syncthreads();
      constexpr int NE = R * G * 128;
      for (int idx = tid; idx < NE; idx += 256) {
        const int ln = idx & 31, r = (idx >> 5) & 3, gi = (idx >> 7) % G, i = idx / (128 * G);
        const int rowin = (ln >> 2) + ((r & 2) ? 8 : 0);
        const int tok = p0 + gi * 8 + (ln & 3) * 2 + (r & 1);
        if (tok < n) {
          const int col = (tg * R + i) * 16 + rowin;
          const float v = red[idx] * gsc;
          if (GATED) {
            const float u = red[idx + R * G * 128] * gsc;
            const float h = __fdividef(v, 1.0f + __expf(-v)) * u;
            Out[(size_t)(off + tok) * NOUT + col] = __float2bfloat16(h);
          } else {
            Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void __launch_bounds__(1024) moe_route(const int* __restrict__ ids, int nroutes, int start, int count,
                                                  int* __restrict__ meta, int* __restrict__ sroutes) {
  __shared__ int cnt[32], base[32], fill[32];
  const int tid = threadIdx.x;
  if (tid < 32) { cnt[tid] = 0; fill[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < nroutes; i += blockDim.x) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid < 32) {
    const int c = cnt[tid];
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    base[tid] = incl - c;
    meta[33 + tid] = incl - c;
    if (tid == 31) { meta[65] = incl; meta[66] = 0; meta[67] = 0; }
    // rank among active experts: heavier first, ties by id
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(0xffffffffu, c, j);
      rank += (cj > c) || (cj == c && j < tid);
    }
    const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
    if (tid == 0) meta[0] = na;
    if (tid >= na) meta[1 + tid] = 0;
    __syncwarp();
    if (c > 0) meta[1 + rank] = tid;
  }
  __syncthreads();
  for (int i = tid; i < nroutes; i += blockDim.x) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) sroutes[base[e] + atomicAdd(&fill[e], 1)] = i;
  }
}

// One CTA (512 threads x 8 values) per token.
__global__ void __launch_bounds__(512) moe_norm(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ ids,
                                                const float* __restrict__ tw, const float* __restrict__ w1,
                                                int start, int count, float eps, __nv_bfloat16* __restrict__ out) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = ids[tok * 8 + s] - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      uint4 raw = *reinterpret_cast<const uint4*>(Y + ((size_t)(tok * 8 + s)) * NOUT + tid * 8);
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 f = __bfloat1622float2(p[k]);
        v[s][2 * k] = f.x * wv[2 * k];
        v[s][2 * k + 1] = f.y * wv[2 * k + 1];
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) ss[s] += v[s][k] * v[s][k];
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[s][k] = 0.f;
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float x = ss[s];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) ssum[warp][s] = x;
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!loc[s]) continue;
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) tot += ssum[w][s];
    const float f = tw[tok * 8 + s] * rsqrtf(tot / NOUT + eps);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += f * v[s][k];
  }
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  *reinterpret_cast<uint4*>(out + (size_t)tok * NOUT + tid * 8) = o;
}


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }


template <int NS, int NG, int NW, typename F>
__device__ __forceinline__ void big_kloop(float (&acc)[8][4], const F& issue, const uint4* __restrict__ sb, int rt,
                                          const uint4* xs, const uint4* wsm, const uint32_t* slut,
                                          int warp, int lane, int g, int t) {
  constexpr int XS = 64 * 9, WS = NW * 32;
      
#pragma unroll
  for (int gi = 0; gi < 8; ++gi)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[gi][r] = 0.f;
  uint4 sc, scn = __ldg(sb + ((size_t)rt * 8) * 32 + lane);
#pragma unroll 1
  for (int kc0 = 0; kc0 < 64; kc0 += NS) {
    if ((kc0 & 7) == 0) {
      sc = scn;
      if (kc0 < 56) scn = __ldg(sb + ((size_t)rt * 8 + (kc0 >> 3) + 1) * 32 + lane);
    } else {
      sc.x = sc.z; sc.y = sc.w;  // NS == 4, second half of the scale group
    }
#pragma unroll
    for (int u = 0; u < NS; ++u) {
      const int kc = kc0 + u;
      cp_wait<NS - 2>();
      __syncthreads();
      if (kc + NS - 1 < 64) issue(kc + NS - 1);
      cp_commit();
      const uint4 wv = wsm[u * WS + warp * 32 + lane];
      const uint32_t sw = u4get(sc, u >> 1) >> ((u & 1) * 16);
      const uint32_t s0 = slut[sw & 0xFF], s1 = slut[(sw >> 8) & 0xFF];
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        uint32_t af[2][4];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) dq(u4get(wv, jp * 2 + jj), s0, s1, af[jj]);
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
          if (gi < NG) {
            const uint4 b = xs[u * XS + (gi * 8 + g) * 9 + t * 2 + jp];
            mma16816(acc[gi], af[0], b.x, b.y);
            mma16816(acc[gi], af[1], b.z, b.w);
          }
        }
      }
    }
  }
}

// Big-M grouped GEMM. Item = (active expert, 256 packed rows); each of the 16 warps owns one 16-row tile
// over the full K (gated: warps 0-7 gate tiles, 8-15 the matching up tiles), so every weight is
// dequantized once per 64-token pass. All tokens' activations for a 64-wide k chunk are staged in smem
// (shared by the 16 warps) together with each warp's weights for that chunk, in an NS-stage cp.async
// ring. Items are taken from a global counter (heaviest experts first).
template <bool GATED, int NS, int NW, int MB>
__global__ void __launch_bounds__(NW * 32, MB) moe_big(
    const uint4* __restrict__ Wq, const uint4* __restrict__ Ws, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  constexpr int NRT = GATED ? 2 * NOUT / 16 : NOUT / 16;
  constexpr int IPE = GATED ? NOUT / (8 * NW) : NOUT / (16 * NW);  // items per expert
  constexpr int XS = 64 * 9;                             // uint4 per x stage (64 rows, 144 B stride)
  constexpr int WS = NW * 32;                            // uint4 per weight stage
  constexpr int HW = NW / 2;
  static_assert(NS == 4 || NS == 8, "NS");
  extern __shared__ uint4 smem[];
  uint4* xs = smem;
  uint4* wsm = smem + NS * XS;
  __shared__ uint32_t slut[256];
  __shared__ int s_item;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += NW * 32) slut[i] = lut[i];
  const int nitems = meta[0] * IPE;
  int* counter = meta + (GATED ? 66 : 67);
  for (;;) {
    __syncthreads();
    if (tid == 0) s_item = atomicAdd(counter, 1);
    __syncthreads();
    const int item = s_item;
    if (item >= nitems) break;
    const int a = item / IPE, rb = item - a * IPE;
    const int e = meta[1 + a];
    const int off = meta[33 + e];
    const int n = meta[34 + e] - off;
    const uint4* wb = Wq + (size_t)e * NRT * 64 * 32;
    const uint4* sb = Ws + (size_t)e * NRT * 8 * 32;
    const int rt = GATED ? ((warp >= HW) ? NOUT / 16 : 0) + rb * HW + (warp % HW) : rb * NW + warp;
    const float gsc = gs[e] * 256.0f;
    for (int p0 = 0; p0 < n; p0 += 64) {
      const int nv = min(64, n - p0);
      const int ng = (nv + 7) >> 3;
      // x chunks this thread copies: c = tid + h * NW * 32 -> row c >> 3, 16 B segment c & 7
      constexpr int XH = 512 / (NW * 32);
      const uint4* xsrc[XH];
      uint4* xdst[XH];
      int xbytes[XH];
      bool xdo[XH];
#pragma unroll
      for (int h = 0; h < XH; ++h) {
        const int c = tid + h * NW * 32, xrow = c >> 3;
        xdo[h] = xrow < ng * 8;
        xbytes[h] = xrow < nv ? 16 : 0;
        int src = 0;
        if (xrow < nv) src = GATED ? (sroutes[off + p0 + xrow] >> 3) : (off + p0 + xrow);
        xsrc[h] = reinterpret_cast<const uint4*>(X + (size_t)src * KD) + (c & 7);
        xdst[h] = xs + xrow * 9 + (c & 7);
      }
      const uint4* wsrc = wb + (size_t)rt * 64 * 32 + lane;
      uint4* wdst = wsm + warp * 32 + lane;
      auto issue = [&](int kc) {
        const int s = kc % NS;
#pragma unroll
        for (int h = 0; h < XH; ++h)
          if (xdo[h]) cp_async16(xdst[h] + s * XS, xsrc[h] + kc * 8, xbytes[h]);
        cp_async16(wdst + s * WS, wsrc + kc * 32, 16);
      };
#pragma unroll
      for (int s = 0; s < NS - 1; ++s) { issue(s); cp_commit(); }
      float acc[8][4];
      switch (ng) {
        case 1: big_kloop<NS, 1, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
        case 2: big_kloop<NS, 2, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
        case 3: case 4: big_kloop<NS, 4, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
        default: big_kloop<NS, 8, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
      }
      cp_wait<0>();
      __syncthreads();
      float* red = reinterpret_cast<float*>(smem);  // gated: up tiles hand their values to the gate warps
      if (GATED && warp >= HW) {
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
          if (gi < ng)
#pragma unroll
            for (int r = 0; r < 4; ++r) red[(((warp - HW) * 8 + gi) * 4 + r) * 32 + lane] = acc[gi][r];
      }
      if (GATED) __syncthreads();
      if (!GATED || warp < HW) {
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
          if (gi < ng) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const int col = rt * 16 + g + ((r & 2) ? 8 : 0);
              const int tok = p0 + gi * 8 + t * 2 + (r & 1);
              if (tok < n) {
                const float v = acc[gi][r] * gsc;
                if (GATED) {
                  const float u = red[((warp * 8 + gi) * 4 + r) * 32 + lane] * gsc;
                  Out[(size_t)(off + tok) * NOUT + col] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
                } else {
                  Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
                }
              }
            }
          }
        }
      }
      __syncthreads();  // red / stage buffers are reused by the next pass
    }
  }
}

template <bool GATED, int NS, int NW, int MB>
static int launch_big(const void* wq, const void* ws, const void* gs, const void* lut, const void* x,
                      const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NS * (64 * 9 + NW * 32) * 16;
  if (!grid) {
    int dev, nsm, occ;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_big<GATED, NS, NW, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_big<GATED, NS, NW, MB>, NW * 32, SMEM);
    if (occ < 1) occ = 1;
    grid = nsm * occ;
  }
  moe_big<GATED, NS, NW, MB><<<grid, NW * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint4*)ws, (const float*)gs, (const uint32_t*)lut, (const __nv_bfloat16*)x,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

template <int R, int G, bool GATED, int MB = (R * G == 1) ? 4 : 2>
static int launch_gemm(const void* wq, const void* ws, const void* gs, const void* lut, const void* x,
                       const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  if (!grid) {
    int dev, nsm, occ;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_gemm<R, G, GATED, MB>, 256, 0);
    if (occ < 1) occ = 1;
    grid = nsm * occ;
  }
  moe_gemm<R, G, GATED, MB><<<grid, 256, 0, st>>>(
      (const uint4*)wq, (const uint4*)ws, (const float*)gs, (const uint32_t*)lut, (const __nv_bfloat16*)x,
      (const int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

#define CFG(R, G) (R * 10 + G)
#define DISPATCH(GATED)                                                                     \
  switch (cfg) {                                                                            \
    case 10: return launch_gemm<1, 1, GATED, 3>(wq, ws, gs, lut, x, meta, sr, out, st);    \
    case CFG(1, 1): return launch_gemm<1, 1, GATED>(wq, ws, gs, lut, x, meta, sr, out, st); \
    case CFG(2, 2): return launch_gemm<2, 2, GATED>(wq, ws, gs, lut, x, meta, sr, out, st); \
    case 118: return launch_big<GATED, 8, 8, 2>(wq, ws, gs, lut, x, meta, sr, out, st);     \
  }                                                                                         \
  return -1;


// ---- decode path (M <= 32): stream-K over (active expert, row-tile pair, k chunk) units ----
// Acc13 [pos][8192] / Acc2 [pos][4096] FP32 are zeroed in-call (route2 / act) and receive red.add partials.

// CTA 0: the route (as moe_route) plus inv[route] = sorted position. CTAs 1..: zero Acc13's used rows.
__global__ void __launch_bounds__(256) moe_route2(const int* __restrict__ ids, int nroutes, int start, int count,
                                                  int* __restrict__ meta, int* __restrict__ sroutes,
                                                  int* __restrict__ inv, float4* __restrict__ acc13) {
  const int tid = threadIdx.x;
  if (blockIdx.x > 0) {
    int c = 0;
    for (int i = tid; i < nroutes; i += 256) { const int e = ids[i] - start; c += (e >= 0 && e < count); }
    __shared__ int s_n;
    if (tid == 0) s_n = 0;
    __syncthreads();
    if (c) atomicAdd(&s_n, c);
    __syncthreads();
    const size_t tot = (size_t)s_n * 2048;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)(blockIdx.x - 1) * 256 + tid; i < tot; i += (size_t)(gridDim.x - 1) * 256) acc13[i] = z;
    return;
  }
  __shared__ int cnt[32], base[32], fill[32];
  if (tid < 32) { cnt[tid] = 0; fill[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < nroutes; i += 256) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid < 32) {
    const int c = cnt[tid];
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    base[tid] = incl - c;
    meta[33 + tid] = incl - c;
    if (tid == 31) { meta[65] = incl; meta[66] = 0; meta[67] = 0; }
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(0xffffffffu, c, j);
      rank += (cj > c) || (cj == c && j < tid);
    }
    const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
    if (tid == 0) meta[0] = na;
    if (tid >= na) meta[1 + tid] = 0;
    __syncwarp();
    if (c > 0) meta[1 + rank] = tid;
  }
  __syncthreads();
  for (int i = tid; i < nroutes; i += 256) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) {
      const int pos = base[e] + atomicAdd(&fill[e], 1);
      sroutes[pos] = i;
      inv[i] = pos;
    }
  }
}

// One segment of a warp's unit range, all in expert e. Weights (both row tiles + their 4 scale bytes) are
// loaded D units ahead in a register ring; x fragments of NG token groups one unit ahead.
template <int NRT, int NG, int D, bool XROUTE>
__device__ __forceinline__ void sk_seg(int u, const int send, const uint4* __restrict__ Wq,
                                       const uint32_t* __restrict__ S2, const uint32_t* slut,
                                       const __nv_bfloat16* __restrict__ X, const int* __restrict__ sroutes,
                                       float* __restrict__ Acc, int e, int off, int n, int lane) {
  constexpr int UPE = NRT / 2 * 64;
  static_assert(D % 2 == 0, "D even (x double buffer)");
  const int g = lane >> 2, t = lane & 3;
  const int lu0 = u & (UPE - 1);
  const uint4* wp = Wq + ((size_t)e * NRT * 64 + lu0) * 32 + lane;
  const uint32_t* sp = S2 + ((size_t)e * UPE + lu0) * 32 + lane;
  const uint4* xr[NG];
  bool xv[NG];
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int idx = gi * 8 + g;
    xv[gi] = idx < n;
    int row = 0;
    if (xv[gi]) row = XROUTE ? (sroutes[off + idx] >> 3) : (off + idx);
    xr[gi] = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
  }
  int xk = lu0 & 63;  // k chunk of the next x load
  uint4 rw0[D], rw1[D], rx[2][NG][2];
  uint32_t rs[D];
  auto issue = [&](int s) {
    rw0[s] = ld_stream(wp);
    rw1[s] = ld_stream(wp + NRT * 32 * 32);
    rs[s] = __ldg(sp);
    wp += 32;
    sp += 32;
  };
  auto xload = [&](int b) {
#pragma unroll
    for (int gi = 0; gi < NG; ++gi) {
      if (xv[gi]) {
        rx[b][gi][0] = __ldg(xr[gi] + xk * 8);
        rx[b][gi][1] = __ldg(xr[gi] + xk * 8 + 1);
      } else {
        rx[b][gi][0] = make_uint4(0, 0, 0, 0);
        rx[b][gi][1] = make_uint4(0, 0, 0, 0);
      }
    }
    xk = (xk + 1) & 63;
  };
#pragma unroll
  for (int s = 0; s < D; ++s)
    if (u + s < send) issue(s);
  xload(0);
  float acc[2][NG][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int gi = 0; gi < NG; ++gi)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][gi][r] = 0.f;
  for (int base = u; base < send; base += D) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      const int uc = base + s;
      if (uc >= send) break;
      if (uc + 1 < send) xload((s + 1) & 1);
      const uint32_t sw = rs[s];
      const uint32_t sc[4] = {slut[sw & 0xFF], slut[(sw >> 8) & 0xFF], slut[(sw >> 16) & 0xFF], slut[sw >> 24]};
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 wv = i ? rw1[s] : rw0[s];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * i], sc[2 * i + 1], af);
#pragma unroll
          for (int gi = 0; gi < NG; ++gi)
            mma16816(acc[i][gi], af, u4get(rx[s & 1][gi][j >> 1], (j & 1) * 2),
                     u4get(rx[s & 1][gi][j >> 1], (j & 1) * 2 + 1));
        }
      }
      if (uc + D < send) issue(s);
      if (((uc + 1) & 63) == 0 || uc + 1 == send) {
        const int rp = (uc & (UPE - 1)) >> 6;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int gi = 0; gi < NG; ++gi)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const int tok = gi * 8 + 2 * t + (r & 1);
              const int col = (rp + i * (NRT / 2)) * 16 + g + ((r & 2) ? 8 : 0);
              if (tok < n) atomicAdd(Acc + (size_t)(off + tok) * (NRT * 16) + col, acc[i][gi][r]);
              acc[i][gi][r] = 0.f;
            }
      }
    }
  }
}

// Units: for each active expert a and each group of 8 of its tokens, UPE (row-tile pair, k chunk) units
// (experts with > 8 tokens re-read their weights per group: rare in decode). Each warp takes a contiguous
// range of units: perfect balance whatever the routing, partial sums flushed with red.add.
template <int NRT, int D, int MB, bool XROUTE>
__global__ void __launch_bounds__(256, MB) moe_sk(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2,
                                                  const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
                                                  const int* __restrict__ meta, const int* __restrict__ sroutes,
                                                  float* __restrict__ Acc) {
  constexpr int UPE = NRT / 2 * 64;
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_us[33];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  slut[tid] = lut[tid];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  const int nact = meta[0];
  if (tid < 32) {
    const int e = meta[1 + tid];
    s_e[tid] = e;
    const int ng = tid < nact ? (meta[34 + e] - meta[33 + e] + 7) >> 3 : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_us[tid + 1] = incl * UPE;
    if (tid == 0) s_us[0] = 0;
  }
  __syncthreads();
  const unsigned W = gridDim.x * 8, gw = blockIdx.x * 8 + warp;
  const unsigned U = (unsigned)s_us[nact];
  const int u0 = (int)(((unsigned long long)U * gw) / W), u1 = (int)(((unsigned long long)U * (gw + 1)) / W);
  int u = u0, a = 0;
  while (u < u1) {
    while (s_us[a + 1] <= u) ++a;
    const int gi = (u - s_us[a]) / UPE;
    const int send = min(u1, s_us[a] + (gi + 1) * UPE);
    const int e = s_e[a], off = s_off[e] + gi * 8, n = min(8, s_off[e + 1] - off);
    sk_seg<NRT, 1, D, XROUTE>(u, send, Wq, S2, slut, X, sroutes, Acc, e, off, n, lane);
    u = send;
  }
}

// inter[pos] = bf16(silu(gate) * up) from Acc13 (x global scale); zeroes Acc2[pos].
__global__ void __launch_bounds__(256) moe_act(const float4* __restrict__ acc13, const int* __restrict__ meta,
                                               const int* __restrict__ sroutes, const int* __restrict__ ids,
                                               const float* __restrict__ gs, int start, __nv_bfloat16* __restrict__ inter,
                                               float4* __restrict__ acc2) {
  const int pos = blockIdx.x, tid = threadIdx.x;
  if (pos >= meta[65]) return;
  const int e = ids[sroutes[pos]] - start;
  const float gsc = gs[e] * 256.0f;
  const float4* gp = acc13 + (size_t)pos * 2048 + tid * 4;
  const float4* up = gp + 1024;
  uint32_t o[8];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 a = gp[k], b = up[k];
    const float v[4] = {a.x * gsc, a.y * gsc, a.z * gsc, a.w * gsc}, w[4] = {b.x * gsc, b.y * gsc, b.z * gsc, b.w * gsc};
    float h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) h[q] = __fdividef(v[q], 1.0f + __expf(-v[q])) * w[q];
    __nv_bfloat162 p0 = __floats2bfloat162_rn(h[0], h[1]), p1 = __floats2bfloat162_rn(h[2], h[3]);
    o[2 * k] = *reinterpret_cast<uint32_t*>(&p0);
    o[2 * k + 1] = *reinterpret_cast<uint32_t*>(&p1);
  }
  uint4* dst = reinterpret_cast<uint4*>(inter + (size_t)pos * NOUT + tid * 16);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < 4; ++k) acc2[(size_t)pos * 1024 + tid * 4 + k] = z;
}

// As moe_norm, but y comes from Acc2 (FP32, sorted positions, x global scale).
__global__ void __launch_bounds__(512) moe_norm2(const float* __restrict__ Y, const int* __restrict__ inv,
                                                 const int* __restrict__ ids, const float* __restrict__ tw,
                                                 const float* __restrict__ w1, const float* __restrict__ gs, int start,
                                                 int count, float eps, __nv_bfloat16* __restrict__ out) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = ids[tok * 8 + s] - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      const float gsc = gs[e] * 256.0f;
      const float4* p = reinterpret_cast<const float4*>(Y + (size_t)inv[tok * 8 + s] * NOUT + tid * 8);
      const float4 a = p[0], b = p[1];
      const float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int k = 0; k < 8; ++k) v[s][k] = f[k] * gsc * wv[k];
#pragma unroll
      for (int k = 0; k < 8; ++k) ss[s] += v[s][k] * v[s][k];
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[s][k] = 0.f;
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float x = ss[s];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) ssum[warp][s] = x;
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!loc[s]) continue;
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) tot += ssum[w][s];
    const float f = tw[tok * 8 + s] * rsqrtf(tot / NOUT + eps);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += f * v[s][k];
  }
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  *reinterpret_cast<uint4*>(out + (size_t)tok * NOUT + tid * 8) = o;
}

template <int NRT, int D, int MB, bool XROUTE>
static int launch_sk(const void* wq, const void* s2, const void* lut, const void* x, const void* meta,
                     const void* sroutes, void* acc, cudaStream_t st) {
  static int grid = 0;
  if (!grid) {
    int dev, nsm, occ;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_sk<NRT, D, MB, XROUTE>, 256, 0);
    if (occ < 1) occ = 1;
    grid = nsm * occ;
  }
  moe_sk<NRT, D, MB, XROUTE><<<grid, 256, 0, st>>>((const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut,
                                                    (const __nv_bfloat16*)x, (const int*)meta, (const int*)sroutes,
                                                    (float*)acc);
  return (int)cudaGetLastError();
}


// ---- TMA decode GEMM: one CTA per SM, a producer lane streams each block of NC consecutive units (both row
// tiles' weights + scales: 3 contiguous bulk copies) into an NSTG-stage smem ring; consumer warp w takes unit w
// of every block, i.e. k chunks w, w + NC, ... of each row-tile pair, so its x fragments stay in registers.
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

template <int NRT, int NC, int NSTG, bool XROUTE>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2,
                                                          const uint32_t* __restrict__ lut,
                                                          const __nv_bfloat16* __restrict__ X,
                                                          const int* __restrict__ meta, const int* __restrict__ sroutes,
                                                          float* __restrict__ Acc) {
  constexpr int RPE = NRT / 2;              // row-tile pairs per (expert, token group)
  constexpr int NQ = 64 / NC;               // blocks per row-tile pair
  constexpr int SW = NC * 32;               // uint4 per weight half-stage
  constexpr int SB = NC * 1152;             // bytes per stage
  extern __shared__ __align__(128) uint4 smem[];
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_rs[33];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  const int nact = meta[0];
  if (tid < 32) {
    const int e = meta[1 + tid];
    s_e[tid] = e;
    const int ng = tid < nact ? (meta[34 + e] - meta[33 + e] + 7) >> 3 : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_rs[tid + 1] = incl * RPE;
    if (tid == 0) s_rs[0] = 0;
  }
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NC * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nrp = s_rs[nact];
  const int r0 = (int)((long long)nrp * blockIdx.x / gridDim.x), r1 = (int)((long long)nrp * (blockIdx.x + 1) / gridDim.x);
  if (warp == NC) {
    if (lane == 0) {
      int a = 0, st = 0, ph = 0, i = 0;
      for (int R = r0; R < r1; ++R) {
        while (s_rs[a + 1] <= R) ++a;
        const int e = s_e[a];
        const int lr = (R - s_rs[a]) & (RPE - 1);
        const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
        const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
        for (int q = 0; q < NQ; ++q, ++i) {
          if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
          char* sbase = reinterpret_cast<char*>(smem) + st * SB;
          mbar_expect_tx(&full[st], SB);
          bulk_g2s(sbase, src + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 512, src + NRT * 32 * 32 + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 1024, ssrc + q * NC * 32, NC * 128, &full[st]);
          if (++st == NSTG) { st = 0; ph ^= 1; }
        }
      }
    }
    return;
  }
  // consumer warp
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, st = 0, ph = 0;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {  // new (expert, token group) segment: x fragments for this warp's k chunks
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;  // segment = (expert, token group)
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      const bool v = g < n;
      int row = 0;
      if (v) row = XROUTE ? (sroutes[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
        xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
      }
    }
    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      const uint4 wv0 = sw4[0];
      const uint4 wv1 = sw4[SW];
      const uint32_t sw = reinterpret_cast<const uint32_t*>(sw4 + 2 * SW - warp * 32 - lane)[warp * 32 + lane];
      mbar_arrive(&empty[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
      const uint32_t sc[4] = {slut[sw & 0xFF], slut[(sw >> 8) & 0xFF], slut[(sw >> 16) & 0xFF], slut[sw >> 24]};
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        const uint4 wv = ii ? wv1 : wv0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
          mma16816(acc[ii], af, u4get(xf[q][j >> 1], (j & 1) * 2), u4get(xf[q][j >> 1], (j & 1) * 2 + 1));
        }
      }
    }
    const int rp = (R - seg_start) & (RPE - 1);
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int tok = 2 * t + (r & 1);
        const int col = (rp + ii * RPE) * 16 + g + ((r & 2) ? 8 : 0);
        if (tok < n) atomicAdd(Acc + (size_t)(off + tok) * (NRT * 16) + col, acc[ii][r]);
      }
  }
}

template <int NRT, int NC, int NSTG, bool XROUTE>
static int launch_tk(const void* wq, const void* s2, const void* lut, const void* x, const void* meta,
                     const void* sroutes, void* acc, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * NC * 1152;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tk<NRT, NC, NSTG, XROUTE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tk<NRT, NC, NSTG, XROUTE><<<grid, (NC + 1) * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const __nv_bfloat16*)x, (const int*)meta,
      (const int*)sroutes, (float*)acc);
  return (int)cudaGetLastError();
}


// ---- moe_tk2: as moe_tk, but each row-tile pair is reduced across the consumer warps in smem and written with
// plain stores (no atomics, no zeroing). GATED (gemm13): every CTA computes the routing itself (deterministic,
// identical in all CTAs; CTA 0 publishes meta / sroutes / inv for the later kernels), epilogue silu(gate) * up
// -> bf16 inter[pos]. !GATED (gemm2): routing from meta, epilogue writes FP32 y[pos] (x global scale later).
template <int NRT, int NC, int NSTG, bool GATED>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2(
    const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, const uint32_t* __restrict__ lut,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids, int nroutes,
    int start, int count, int* __restrict__ meta, int* __restrict__ sroutes, int* __restrict__ inv,
    void* __restrict__ Out) {
  static_assert(NC == 8, "routing uses 8 warps x 32 ids");
  constexpr int RPE = NRT / 2;
  constexpr int NQ = 64 / NC;
  constexpr int SW = NC * 32;
  constexpr int SB = NC * 1152;
  extern __shared__ __align__(128) uint4 smem[];
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_rs[33], s_srt[256], s_wc[8][32], s_nact;
  __shared__ float s_red[2][NC * 256];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NC * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (GATED) {
    // routing (M * 8 <= 256 routes): warp w takes routes [32 w, 32 w + 32)
    int e = -1, rank = 0;
    if (warp < 8) {
      s_wc[warp][lane] = 0;
      const int i = warp * 32 + lane;
      if (i < nroutes) {
        const int v = ids[i] - start;
        if (v >= 0 && v < count) e = v;
      }
      const unsigned m = __match_any_sync(0xffffffffu, e);
      rank = __popc(m & ((1u << lane) - 1));
      __syncwarp();
      if (e >= 0 && rank == 0) s_wc[warp][e] = __popc(m);
    }
    __syncthreads();
    if (tid < 32) {
      int c = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) c += s_wc[w][tid];
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      int run = incl - c;
      s_off[tid] = run;
      if (tid == 31) s_off[32] = incl;
#pragma unroll
      for (int w = 0; w < 8; ++w) { const int cw = s_wc[w][tid]; s_wc[w][tid] = run; run += cw; }
      int rk = 0;
      for (int j = 0; j < 32; ++j) {
        const int cj = __shfl_sync(0xffffffffu, c, j);
        rk += (cj > c) || (cj == c && j < tid);
      }
      const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
      if (tid == 0) s_nact = na;
      if (tid >= na) s_e[tid] = 0;
      __syncwarp();
      if (c > 0) s_e[rk] = tid;
    }
    __syncthreads();
    if (e >= 0) {
      const int i = warp * 32 + lane, pos = s_wc[warp][e] + rank;
      s_srt[pos] = i;
      if (blockIdx.x == 0) { sroutes[pos] = i; inv[i] = pos; }
    }
    if (blockIdx.x == 0 && tid < 33) {
      meta[33 + tid] = s_off[tid];
      if (tid < 32) meta[1 + tid] = s_e[tid];
      if (tid == 0) meta[0] = s_nact;
    }
  } else {
    if (tid < 33) s_off[tid] = meta[33 + tid];
    if (tid < 32) s_e[tid] = meta[1 + tid];
    if (tid == 0) s_nact = meta[0];
  }
  __syncthreads();
  const int nact = s_nact;
  if (tid < 32) {
    const int e = s_e[tid];
    const int ng = tid < nact ? (s_off[e + 1] - s_off[e] + 7) >> 3 : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_rs[tid + 1] = incl * RPE;
    if (tid == 0) s_rs[0] = 0;
  }
  __syncthreads();
  const int nrp = s_rs[nact];
  const int r0 = (int)((long long)nrp * blockIdx.x / gridDim.x), r1 = (int)((long long)nrp * (blockIdx.x + 1) / gridDim.x);
  if (warp == NC) {
    if (lane == 0) {
      int a = 0, st = 0, ph = 0, i = 0;
      for (int R = r0; R < r1; ++R) {
        while (s_rs[a + 1] <= R) ++a;
        const int e = s_e[a];
        const int lr = (R - s_rs[a]) & (RPE - 1);
        const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
        const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
        for (int q = 0; q < NQ; ++q, ++i) {
          if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
          char* sbase = reinterpret_cast<char*>(smem) + st * SB;
          mbar_expect_tx(&full[st], SB);
          bulk_g2s(sbase, src + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 512, src + NRT * 32 * 32 + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 1024, ssrc + q * NC * 32, NC * 128, &full[st]);
          if (++st == NSTG) { st = 0; ph ^= 1; }
        }
      }
    }
    return;
  }
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, st = 0, ph = 0, rb = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;  // segment = (expert, token group)
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      const bool v = g < n;
      int row = 0;
      if (v) row = GATED ? (s_srt[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
        xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
      }
    }
    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      const uint4 wv0 = sw4[0];
      const uint4 wv1 = sw4[SW];
      const uint32_t sw = reinterpret_cast<const uint32_t*>(sw4 + 2 * SW - warp * 32 - lane)[warp * 32 + lane];
      mbar_arrive(&empty[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
      const uint32_t sc[4] = {slut[sw & 0xFF], slut[(sw >> 8) & 0xFF], slut[(sw >> 16) & 0xFF], slut[sw >> 24]};
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        const uint4 wv = ii ? wv1 : wv0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
          mma16816(acc[ii], af, u4get(xf[q][j >> 1], (j & 1) * 2), u4get(xf[q][j >> 1], (j & 1) * 2 + 1));
        }
      }
    }
    // cross-warp reduce (double-buffered: one consumer barrier per pair)
    float* red = s_red[rb];
    rb ^= 1;
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[((warp * 2 + ii) * 4 + r) * 32 + lane] = acc[ii][r];
    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
    const int rp = (R - seg_start) & (RPE - 1);
    const int ct = tid, l = ct & 31, gg = l >> 2, tt = l & 3;
    if (GATED) {
      if (ct < 128) {
        const int r = ct >> 5;
        float v = 0.f, u = 0.f;
#pragma unroll
        for (int w = 0; w < NC; ++w) { v += red[((w * 2) * 4 + r) * 32 + l]; u += red[((w * 2 + 1) * 4 + r) * 32 + l]; }
        const int tok = 2 * tt + (r & 1);
        if (tok < n) {
          v *= gsc;
          u *= gsc;
          const int col = rp * 16 + gg + ((r & 2) ? 8 : 0);
          reinterpret_cast<__nv_bfloat16*>(Out)[(size_t)(off + tok) * NOUT + col] =
              __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        }
      }
    } else {
      const int ii = ct >> 7, r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED>
static int launch_tk2(const void* wq, const void* s2, const void* lut, const void* gs, const void* x, const void* ids,
                      int nroutes, int start, int count, void* meta, void* sroutes, void* inv, void* out,
                      cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * NC * 1152;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tk2<NRT, NC, NSTG, GATED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tk2<NRT, NC, NSTG, GATED><<<grid, (NC + 1) * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const float*)gs, (const __nv_bfloat16*)x,
      (const int*)ids, nroutes, start, count, (int*)meta, (int*)sroutes, (int*)inv, out);
  return (int)cudaGetLastError();
}

#define SK_DISPATCH(NRT, XR)                                                                 \
  switch (cfg) {                                                                             \
    case 242: return launch_sk<NRT, 4, 2, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 223: return launch_sk<NRT, 2, 3, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 243: return launch_sk<NRT, 4, 3, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 224: return launch_sk<NRT, 2, 4, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 262: return launch_sk<NRT, 6, 2, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 3168: return launch_tk<NRT, 16, 8, XR>(wq, s2, lut, x, meta, sr, acc, st);         \
    case 3166: return launch_tk<NRT, 16, 6, XR>(wq, s2, lut, x, meta, sr, acc, st);         \
    case 3088: return launch_tk<NRT, 8, 8, XR>(wq, s2, lut, x, meta, sr, acc, st);          \
    case 3812: return launch_tk<NRT, 8, 12, XR>(wq, s2, lut, x, meta, sr, acc, st);         \
  }                                                                                          \
  return -1;

// ---- moe_tb: big-M (M > 32) grouped GEMM on a TMA ring, 1 CTA/SM, 8 consumer warps + 1 producer warp.
// Item = (active expert, 64-token pass, row block of 8 warps x RT 16-row tiles), taken by the producer from a
// global counter (meta[66] / meta[67], zeroed by moe_route), heaviest experts first. A stage = one 64-wide k
// chunk: the item's weights (one bulk copy; prepare packs [e][rb][kc][warp][RT][lane]), their scales
// ([e][rb][kc][warp][lane][RT] u16) and the pass's activation rows (one 128 B copy per token, 144 B smem stride).
// Gated: a warp's RT tiles are RT/2 gate tiles + the matching up tiles -> silu(gate) * up in registers.
template <int RT, bool GATED>
struct TBC {
  static constexpr int WB = 8 * RT * 512;
  static constexpr int SB = 8 * RT * 64;
  static constexpr int XB = 64 * 144;
  static constexpr int STG = WB + SB + XB;
  static constexpr int NRB = (GATED ? 2 * NOUT : NOUT) / 16 / (8 * RT);
};

template <int RT, bool GATED>
__device__ __forceinline__ void tb_ldw(const char* sb, int warp, int lane, uint4 (&wv)[RT], uint32_t (&sc)[RT / 2]) {
  using C = TBC<RT, GATED>;
#pragma unroll
  for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb)[(warp * RT + r) * 32 + lane];
  if (RT == 4) {
    const uint2 v = *reinterpret_cast<const uint2*>(sb + C::WB + (warp * 32 + lane) * RT * 2);
    sc[0] = v.x; sc[RT / 2 - 1] = v.y;
  } else {
    sc[0] = *reinterpret_cast<const uint32_t*>(sb + C::WB + (warp * 32 + lane) * RT * 2);
  }
}

template <int RT, int NG>
__device__ __forceinline__ void tb_half(float (&acc)[RT][8][4], const uint4 (&wv)[RT], const uint32_t (&sc)[RT / 2],
                                        const uint4 (&b)[NG], int jp, const uint32_t* slut) {
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);
    const uint32_t s0 = slut[sw & 0xFF], s1 = slut[(sw >> 8) & 0xFF];
    uint32_t af[2][4];
    dq(u4get(wv[r], jp * 2), s0, s1, af[0]);
    dq(u4get(wv[r], jp * 2 + 1), s0, s1, af[1]);
#pragma unroll
    for (int gi = 0; gi < NG; ++gi) {
      mma16816(acc[r][gi], af[0], b[gi].x, b[gi].y);
      mma16816(acc[r][gi], af[1], b[gi].z, b[gi].w);
    }
  }
}

// The caller has waited for full[st]. Software-pipelined: the next stage's weights and scales are fetched
// before the second half of this stage's MMAs.
template <int RT, int NG, int NSTG, bool GATED>
__device__ __forceinline__ void tb_kloop(float (&acc)[RT][8][4], const char* smem, uint64_t* full, uint64_t* empty,
                                         int& st, int& ph, const uint32_t* slut, int warp, int lane, int g, int t) {
  using C = TBC<RT, GATED>;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int gi = 0; gi < 8; ++gi)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][gi][q] = 0.f;
  uint4 wv[RT];
  uint32_t sc[RT / 2];
  tb_ldw<RT, GATED>(smem + st * C::STG, warp, lane, wv, sc);
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const uint4* xs = reinterpret_cast<const uint4*>(smem + st * C::STG + C::WB + C::SB);
    {
      uint4 b[NG];
#pragma unroll
      for (int gi = 0; gi < NG; ++gi) b[gi] = xs[(gi * 8 + g) * 9 + t * 2];
      tb_half<RT, NG>(acc, wv, sc, b, 0, slut);
    }
    uint4 b[NG];
#pragma unroll
    for (int gi = 0; gi < NG; ++gi) b[gi] = xs[(gi * 8 + g) * 9 + t * 2 + 1];
    uint4 wn[RT];
    uint32_t sn[RT / 2];
    const int cst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
    if (kc < 63) {
      mbar_wait(&full[st], ph);
      tb_ldw<RT, GATED>(smem + st * C::STG, warp, lane, wn, sn);
    }
    mbar_arrive(&empty[cst]);
    tb_half<RT, NG>(acc, wv, sc, b, 1, slut);
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = wn[r];
#pragma unroll
    for (int r = 0; r < RT / 2; ++r) sc[r] = sn[r];
  }
}

template <int RT, int NSTG, bool GATED>
__global__ void __launch_bounds__(384, 1) moe_tb(
    const char* __restrict__ Wb, const char* __restrict__ Sb, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  using C = TBC<RT, GATED>;
  constexpr int NRB = C::NRB;
  extern __shared__ __align__(128) uint4 smem_tb[];
  char* smem = reinterpret_cast<char*>(smem_tb);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[33], s_item[NSTG];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int np = tid < nact ? ((s_off[e + 1] - s_off[e] + 63) >> 6) * NRB : 0;
    int incl = np;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_cum[tid + 1] = incl;
    if (tid == 0) s_cum[0] = 0;
  }
  __syncthreads();
  const int nitems = s_cum[32];
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8) return;
    int* counter = meta + (GATED ? 66 : 67);
    int st = 0, ph = 0, i = 0;
    for (;;) {
      int item = 0;
      if (lane == 0) item = atomicAdd(counter, 1);
      item = __shfl_sync(0xffffffffu, item, 0);
      if (item >= nitems) {
        if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        if (lane == 0) { s_item[st] = -1; mbar_arrive(&full[st]); }
        break;
      }
      int a = 0;
      while (s_cum[a + 1] <= item) ++a;
      const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
      const int e = s_e[a];
      const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
      const char* wsrc = Wb + ((size_t)e * NRB + rb) * 64 * C::WB;
      const char* ssrc = Sb + ((size_t)e * NRB + rb) * 64 * C::SB;
      const char* xsrc[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = lane + 32 * h;
        int src = 0;
        if (row < nv) src = GATED ? (sroutes[off + row] >> 3) : (off + row);
        xsrc[h] = reinterpret_cast<const char*>(X + (size_t)src * KD);
      }
      const int bytes = C::WB + C::SB + nv * 128;
      for (int kc = 0; kc < 64; ++kc, ++i) {
        if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * C::STG;
        if (lane == 0) {
          if (kc == 0) s_item[st] = item;
          mbar_expect_tx(&full[st], bytes);
        }
        __syncwarp();
        if (lane == 0) bulk_g2s(sb, wsrc + (size_t)kc * C::WB, C::WB, &full[st]);
        if (lane == 1) bulk_g2s(sb + C::WB, ssrc + (size_t)kc * C::SB, C::SB, &full[st]);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if (lane + 32 * h < nv) bulk_g2s(sb + C::WB + C::SB + (lane + 32 * h) * 144, xsrc[h] + kc * 128, 128, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    int a = 0;
    while (s_cum[a + 1] <= item) ++a;
    const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
    const int e = s_e[a];
    const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
    const float gsc = gs[e] * 256.0f;
    float acc[RT][8][4];
    switch ((nv + 7) >> 3) {
      case 1: tb_kloop<RT, 1, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 2: tb_kloop<RT, 2, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 3: tb_kloop<RT, 3, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 4: tb_kloop<RT, 4, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 5: tb_kloop<RT, 5, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 6: tb_kloop<RT, 6, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 7: tb_kloop<RT, 7, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      default: tb_kloop<RT, 8, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
    }
    constexpr int H = GATED ? RT / 2 : RT;
#pragma unroll
    for (int r = 0; r < H; ++r) {
      const int tile = rb * 8 * H + warp * H + r;
#pragma unroll
      for (int gi = 0; gi < 8; ++gi) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tok = gi * 8 + t * 2 + (q & 1);
          if (tok < nv) {
            const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
            const float v = acc[r][gi][q] * gsc;
            if (GATED) {
              const float u = acc[r + H][gi][q] * gsc;
              Out[(size_t)(off + tok) * NOUT + col] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
            } else {
              Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
            }
          }
        }
      }
    }
  }
}

template <int RT, int NSTG, bool GATED>
static int launch_tb(const void* wb, const void* sb, const void* gs, const void* lut, const void* x,
                     const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * TBC<RT, GATED>::STG;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tb<RT, NSTG, GATED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tb<RT, NSTG, GATED><<<grid, 384, SMEM, st>>>(
      (const char*)wb, (const char*)sb, (const float*)gs, (const uint32_t*)lut, (const __nv_bfloat16*)x,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

// ---- moe_tw: big-M (M > 32) grouped GEMM with wgmma (A = dequantized weights from registers, B = activations
// from smem via a TMA tensor map, SW128). Same items / producer / weight packing as moe_tb, but activations come
// from a gathered, K-permuted copy (moe_gather: row pos = route in expert order; within each 64-chunk logical
// k = j*16 + h*8 + 2t + b holds physical k = 16t + 4j + 2h + b, the order _pack_weight uses), and gemm13 writes
// inter in that order too. 2 consumer warpgroups (warp w: RT 16-row tiles), 1 producer warp, 1 CTA/SM.
struct TMaps { CUtensorMap m[4]; };  // box rows 8, 16, 32, 64

template <int RT, bool GATED>
struct TWC {
  static constexpr int XB = 64 * 128;
  static constexpr int WB = 8 * RT * 512;
  static constexpr int SB = 8 * RT * 64;
  static constexpr int STG = XB + WB + SB;
  static constexpr int NRB = (GATED ? 2 * NOUT : NOUT) / 16 / (8 * RT);
};

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 62);
}

//@@WGMMA@@

template <int RT, int KG, int N, int NSTG, bool GATED>
__device__ __forceinline__ void tw_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                        const uint32_t* slut, int warp, int lane, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  using C = TWC<RT, GATED>;
  constexpr int NA = N / 2;
  float acc[RT][NA];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[KG][RT][4], afB[KG][RT][4];
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const char* sb = smem + st * C::STG;
    if (kc) mbar_wait(&full[st], ph);
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + C::XB)[(warp * RT + r) * 32 + lane];
    uint32_t sc[RT / 2];
    if (RT == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(sb + C::XB + C::WB + (warp * 32 + lane) * RT * 2);
      sc[0] = v.x; sc[RT / 2 - 1] = v.y;
    } else {
      sc[0] = *reinterpret_cast<const uint32_t*>(sb + C::XB + C::WB + (warp * 32 + lane) * RT * 2);
    }
    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);
      s0[r] = slut[sw & 0xFF];
      s1[r] = slut[(sw >> 8) & 0xFF];
    }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; j0 += KG) {
      // ring slot = group parity (KG == 4: stage parity, tracked by kc)
      const bool odd = (KG == 4) ? (kc & 1) : ((j0 / KG) & 1);
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (odd) dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afB[jj][r]);
          else dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afA[jj][r]);
        }
      wg_fence();
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r) wgmma_bf16<N>(acc[r], odd ? afB[jj][r] : afA[jj][r], d0 + (j0 + jj) * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (odd) fence_r(afA[jj][r][i]);
            else fence_r(afB[jj][r][i]);
          }
      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);
    }
    pst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
#pragma unroll
  for (int r = 0; r < RT; ++r) {
#pragma unroll
    for (int jj = 0; jj < KG; ++jj)
#pragma unroll
      for (int i = 0; i < 4; ++i) { fence_r(afA[jj][r][i]); fence_r(afB[jj][r][i]); }
#pragma unroll
    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
  const int g = lane >> 2, t = lane & 3;
  constexpr int H = GATED ? RT / 2 : RT;
#pragma unroll
  for (int r = 0; r < H; ++r) {
    const int tile = rb * 8 * H + warp * H + r;
#pragma unroll
    for (int gi = 0; gi < N / 8; ++gi) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = gi * 8 + t * 2 + (q & 1);
        if (tok < nv) {
          const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
          const float v = acc[r][gi * 4 + q] * gsc;
          if (GATED) {
            const float u = acc[r + H][gi * 4 + q] * gsc;
            // logical (permuted) position of physical k = col for gemm2's B
            const int c = col & 63;
            const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
            Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
          } else {
            Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
          }
        }
      }
    }
  }
}

template <int RT, int KG, int NSTG, bool GATED>
__global__ void __launch_bounds__(384, 1) moe_tw(
    const char* __restrict__ Wb, const char* __restrict__ Sb, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  using C = TWC<RT, GATED>;
  constexpr int NRB = C::NRB;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[33], s_item[NSTG];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int np = tid < nact ? ((s_off[e + 1] - s_off[e] + 63) >> 6) * NRB : 0;
    int incl = np;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_cum[tid + 1] = incl;
    if (tid == 0) s_cum[0] = 0;
  }
  __syncthreads();
  const int nitems = s_cum[32];
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8 || lane != 0) return;
    int* counter = meta + (GATED ? 66 : 67);
    int st = 0, ph = 0, i = 0;
    for (;;) {
      const int item = atomicAdd(counter, 1);
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int a = 0;
      while (s_cum[a + 1] <= item) ++a;
      const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
      const int e = s_e[a];
      const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const char* wsrc = Wb + ((size_t)e * NRB + rb) * 64 * C::WB;
      const char* ssrc = Sb + ((size_t)e * NRB + rb) * 64 * C::SB;
      const int bytes = C::WB + C::SB + (1024 << mi);
      for (int kc = 0; kc < 64; ++kc, ++i) {
        if (kc && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * C::STG;
        if (kc == 0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, &maps.m[mi], kc * 64, off, &full[st]);
        bulk_g2s(sb + C::XB, wsrc + (size_t)kc * C::WB, C::WB, &full[st]);
        bulk_g2s(sb + C::XB + C::WB, ssrc + (size_t)kc * C::SB, C::SB, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    int a = 0;
    while (s_cum[a + 1] <= item) ++a;
    const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
    const int e = s_e[a];
    const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
    const float gsc = gs[e] * 256.0f;
#define TWI(NN) tw_item<RT, KG, NN, NSTG, GATED>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, Out)
    switch ((nv + 7) >> 3) {
      case 1: TWI(8); break;
      case 2: TWI(16); break;
      case 3: TWI(24); break;
      case 4: TWI(32); break;
      case 5: TWI(40); break;
      case 6: TWI(48); break;
      case 7: TWI(56); break;
      default: TWI(64); break;
    }
#undef TWI
  }
}

template <int RT, int KG, int NSTG, bool GATED>
static int launch_tw(const void* wb, const void* sb, const void* gs, const void* lut, const TMaps& maps,
                     const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * TWC<RT, GATED>::STG + 1024;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tw<RT, KG, NSTG, GATED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tw<RT, KG, NSTG, GATED><<<grid, 384, SMEM, st>>>(
      (const char*)wb, (const char*)sb, (const float*)gs, (const uint32_t*)lut, maps,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

// xg[pos] = x[sroutes[pos] >> 3] with each 64-chunk K-permuted (logical word (j*2+h)*4+t <- physical word 8t+2j+h).
__global__ void __launch_bounds__(256) moe_gather(const uint32_t* __restrict__ X, const int* __restrict__ meta,
                                                  const int* __restrict__ sroutes, uint32_t* __restrict__ XG) {
  const int pos = blockIdx.x;
  if (pos >= meta[65]) return;
  const uint32_t* src = X + (size_t)(sroutes[pos] >> 3) * (KD / 2);
  uint32_t* dst = XG + (size_t)pos * (KD / 2);
#pragma unroll
  for (int i = 0; i < KD / 2 / 256; ++i) {
    const int o = threadIdx.x + 256 * i, w = o & 31, t = w & 3, jh = w >> 2;
    dst[o] = src[(o & ~31) | (8 * t + jh)];
  }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define PTR(t) ((const void*)(t).data_ptr())
int64_t moe_gemm13(int64_t cfg, torch::Tensor wq_, torch::Tensor ws_, torch::Tensor gs_, torch::Tensor lut_,
                   torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *ws = PTR(ws_), *gs = PTR(gs_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_),
             *sr = PTR(sr_);
  void* out = out_.data_ptr();
  DISPATCH(true)
}
int64_t moe_gemm2(int64_t cfg, torch::Tensor wq_, torch::Tensor ws_, torch::Tensor gs_, torch::Tensor lut_,
                  torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *ws = PTR(ws_), *gs = PTR(gs_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_),
             *sr = PTR(sr_);
  void* out = out_.data_ptr();
  DISPATCH(false)
}
int64_t moe_route_launch(torch::Tensor ids, int64_t start, int64_t count, torch::Tensor meta, torch::Tensor sroutes) {
  moe_route<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const int*)ids.data_ptr(), (int)ids.numel(), (int)start,
                                                               (int)count, (int*)meta.data_ptr(), (int*)sroutes.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_norm_launch(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start,
                        int64_t count, double eps, torch::Tensor out) {
  const int M = (int)out.size(0);
  moe_norm<<<M, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)y.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (__nv_bfloat16*)out.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_tb_launch(int64_t gated, int64_t cfg, torch::Tensor wb_, torch::Tensor sb_, torch::Tensor gs_,
                      torch::Tensor lut_, torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wb = PTR(wb_), *sb = PTR(sb_), *gs = PTR(gs_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_),
             *sr = PTR(sr_);
  void* out = out_.data_ptr();
#define TB(RT, NS)                                                                       \
  if (cfg == RT * 10 + NS) {                                                             \
    if (gated) return launch_tb<RT, NS, true>(wb, sb, gs, lut, x, meta, sr, out, st);    \
    return launch_tb<RT, NS, false>(wb, sb, gs, lut, x, meta, sr, out, st);              \
  }
  TB(4, 8) TB(2, 8)
  return -1;
}
static PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
// 4 tensor maps (box rows 8/16/32/64, box cols 64, SW128) over a bf16 [rows, KD] buffer. Encoded at every launch
// from the buffer actually passed (under graph capture they are baked into the kernel node with that buffer).
static void tw_maps(TMaps& maps, void* ptr, int64_t rows) {
  if (!g_encode) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_encode, 12000, cudaEnableDefault, &q);
  }
  for (int i = 0; i < 4; ++i) {
    cuuint64_t dims[2] = {(cuuint64_t)KD, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)KD * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)(8 << i)};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g_encode(&maps.m[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  }
}
int64_t moe_gather_launch(torch::Tensor x, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor xg) {
  moe_gather<<<(int)sroutes.numel(), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint32_t*)x.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(), (uint32_t*)xg.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_tw_launch(int64_t gated, int64_t cfg, torch::Tensor wb_, torch::Tensor sb_, torch::Tensor gs_,
                      torch::Tensor lut_, torch::Tensor xb_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wb = PTR(wb_), *sb = PTR(sb_), *gs = PTR(gs_), *lut = PTR(lut_), *meta = PTR(meta_), *sr = PTR(sr_);
  void* out = out_.data_ptr();
  TMaps maps;
  tw_maps(maps, xb_.data_ptr(), xb_.size(0));
#define TW(RT, KG, NS)                                                                       \
  if (cfg == RT * 100 + KG * 10 + NS) {                                                      \
    if (gated) return launch_tw<RT, KG, NS, true>(wb, sb, gs, lut, maps, meta, sr, out, st);  \
    return launch_tw<RT, KG, NS, false>(wb, sb, gs, lut, maps, meta, sr, out, st);            \
  }
  TW(2, 4, 8) TW(2, 2, 8) TW(2, 1, 8) TW(4, 1, 8)
  return -1;
}
int64_t moe_tk2_launch(int64_t gated, int64_t nstg, torch::Tensor wq_, torch::Tensor s2_, torch::Tensor lut_,
                       torch::Tensor gs_, torch::Tensor x_, torch::Tensor ids_, int64_t start, int64_t count,
                       torch::Tensor meta_, torch::Tensor sr_, torch::Tensor inv_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *s2 = PTR(s2_), *lut = PTR(lut_), *gs = PTR(gs_), *x = PTR(x_), *ids = PTR(ids_);
  void *meta = meta_.data_ptr(), *sr = sr_.data_ptr(), *inv = inv_.data_ptr(), *out = out_.data_ptr();
  const int nr = (int)ids_.numel();
  if (nr > 256) return -2;
#define TK2(NS)                                                                                              \
  if (nstg == NS) {                                                                                          \
    if (gated) return launch_tk2<512, 8, NS, true>(wq, s2, lut, gs, x, ids, nr, (int)start, (int)count, meta, sr, inv, out, st); \
    return launch_tk2<256, 8, NS, false>(wq, s2, lut, gs, x, ids, nr, (int)start, (int)count, meta, sr, inv, out, st); \
  }
  TK2(8) TK2(10) TK2(12)
  return -1;
}
int64_t moe_sk_launch(int64_t gated, int64_t cfg, torch::Tensor wq_, torch::Tensor s2_, torch::Tensor lut_,
                      torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor acc_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *s2 = PTR(s2_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_), *sr = PTR(sr_);
  void* acc = acc_.data_ptr();
  if (gated) { SK_DISPATCH(512, true) } else { SK_DISPATCH(256, false) }
}
int64_t moe_route2_launch(torch::Tensor ids, int64_t start, int64_t count, torch::Tensor meta, torch::Tensor sroutes,
                          torch::Tensor inv, torch::Tensor acc13, int64_t nzero) {
  moe_route2<<<1 + nzero, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int*)ids.data_ptr(), (int)ids.numel(), (int)start, (int)count, (int*)meta.data_ptr(),
      (int*)sroutes.data_ptr(), (int*)inv.data_ptr(), (float4*)acc13.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_act_launch(torch::Tensor acc13, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor ids, torch::Tensor gs,
                       int64_t start, torch::Tensor inter, torch::Tensor acc2) {
  moe_act<<<(int)inter.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const float4*)acc13.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(),
      (const int*)ids.data_ptr(), (const float*)gs.data_ptr(), (int)start, (__nv_bfloat16*)inter.data_ptr(),
      (float4*)acc2.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_norm2_launch(torch::Tensor acc2, torch::Tensor inv, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1,
                         torch::Tensor gs, int64_t start, int64_t count, double eps, torch::Tensor out) {
  const int M = (int)out.size(0);
  moe_norm2<<<M, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const float*)acc2.data_ptr(), (const int*)inv.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (const float*)gs.data_ptr(), (int)start, (int)count, (float)eps,
      (__nv_bfloat16*)out.data_ptr());
  return (int64_t)cudaGetLastError();
}
"""


def _gen_wgmma():
    out = ["template <int N> __device__ __forceinline__ void wgmma_bf16(float* d, const uint32_t* a, uint64_t desc);"]
    for n in range(8, 72, 8):
        na = n // 2
        dl = ", ".join(f"%{i}" for i in range(na))
        al = ", ".join(f"%{na + i}" for i in range(4))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
        out.append(
            f"template <> __device__ __forceinline__ void wgmma_bf16<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs}\n'
            f'    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}')
    return "\n".join(out)


CUDA_SRC = CUDA_SRC.replace("//@@WGMMA@@", _gen_wgmma())

_LIB = None
_CPP = ("int64_t moe_gemm13(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gemm2(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor);"
        "int64_t moe_norm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor);"
        "int64_t moe_sk_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tk2_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tb_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gather_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tw_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route2_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t);"
        "int64_t moe_act_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
        "torch::Tensor, torch::Tensor);"
        "int64_t moe_norm2_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, double, torch::Tensor);")


def _lib():
    global _LIB
    if _LIB is None:
        from torch.utils.cpp_extension import load_inline
        h = hashlib.sha256(CUDA_SRC.encode()).hexdigest()[:12]
        _LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_CPP, cuda_sources=CUDA_SRC,
                           functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch", "moe_sk_launch", "moe_tk2_launch", "moe_tb_launch",
                                      "moe_gather_launch", "moe_tw_launch",
                                      "moe_route2_launch", "moe_act_launch", "moe_norm2_launch"],
                           extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"],
                           extra_cflags=["-O3"], verbose=False)
    return _LIB


# ---- prepare-time repacking (lossless: same bits, new order) ----

def _unpack_codes(packed):
    """uint8 [..., K/2] -> uint8 codes [..., K] (element 2i = low nibble)."""
    lo = packed & 0x0F
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# bit positions (lo element, hi element) of (sign, e1, e0, m) for the 4 outputs, see dq().
_POS = [
    ((15, 8, 7, 6), (31, 24, 23, 22)),
    ((5, 30, 29, 28), (21, 14, 13, 12)),  # rotl 10
    ((2, 27, 26, 25), (18, 11, 10, 9)),   # rotl 13
    ((1, 0, 4, 3), (17, 16, 20, 19)),     # (q & 0x001B001B) * (2^3 + 2^8 + 2^14)
]


def _pack_weight(packed):
    """One expert's uint8 [N, K/2] -> int32 [N/16, K/64, 32 lanes, 4] in A-fragment order."""
    N = packed.shape[0]
    K = packed.shape[1] * 2
    c = _unpack_codes(packed).to(torch.int32)  # [N, K]
    # row = rt*16 + h*8 + g ; k = kc*64 + t*16 + j*4 + sb*2 + lohi
    c = c.view(N // 16, 2, 8, K // 64, 4, 4, 2, 2)  # rt h g kc t j sb lohi
    c = c.permute(0, 3, 2, 4, 5, 6, 1, 7)          # rt kc g t j sb h lohi
    c = c.reshape(N // 16, K // 64, 32, 4, 8)       # slot = (sb*2 + h)*2 + lohi = r*2 + lohi
    word = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for r in range(4):
        for lohi in range(2):
            code = c[..., r * 2 + lohi].to(torch.int64)
            ps, pe1, pe0, pm = _POS[r][lohi]
            word |= ((code >> 3) & 1) << ps
            word |= ((code >> 2) & 1) << pe1
            word |= ((code >> 1) & 1) << pe0
            word |= (code & 1) << pm
    return (word - ((word >> 31) << 32)).to(torch.int32)  # same 32 bits, as int32


def _pack_scales(sf):
    """One expert's e4m3 [N, K/16] -> uint8 [N/16, K/512, 32 lanes, 16]."""
    N = sf.shape[0]
    K = sf.shape[1] * 16
    b = sf.view(torch.uint8).view(N // 16, 2, 8, K // 512, 8, 4)  # rt h g kg kcl t
    b = b.permute(0, 3, 2, 5, 4, 1)                                # rt kg g t kcl h
    return b.reshape(N // 16, K // 512, 32, 16).contiguous()


def _pair_scales(ps):
    """_pack_scales output [NRT, K/512, 32, 16] -> uint8 [NRT/2, K/64, 32 lanes, 4]: per (row-tile pair rp, k chunk)
    each lane's 4 bytes (tile rp rows g, g+8, tile rp + NRT/2 rows g, g+8)."""
    n = ps.shape[0]
    v = ps.view(n, ps.shape[1], 32, 8, 2).permute(0, 1, 3, 2, 4).reshape(n, ps.shape[1] * 8, 32, 2)
    return torch.cat((v[: n // 2], v[n // 2:]), dim=-1).contiguous()


def _tb_pack(wq, ws, gated, rt):
    """moe_tb layout: weights [E][rb][kc][warp][RT][lane] uint4, scales [E][rb][kc][warp][lane][RT] u16.
    Gated: warp w of block rb owns gate tiles rb*8*H + w*H + (0..H) and the matching up tiles (H = RT/2)."""
    E, nrt = wq.shape[0], wq.shape[1]
    nrb = nrt // (8 * rt)
    rb = torch.arange(nrb).view(-1, 1, 1)
    w = torch.arange(8).view(1, -1, 1)
    r = torch.arange(rt).view(1, 1, -1)
    if gated:
        h = rt // 2
        tile = (r >= h).long() * (nrt // 2) + rb * 8 * h + w * h + r % h
    else:
        tile = rb * 8 * rt + w * rt + r
    tile = tile.to(wq.device)
    W, S = [], []
    for e in range(E):
        we = wq[e].reshape(nrt, 64, 128)[tile]                                   # rb w r kc lane*4
        W.append(we.permute(0, 3, 1, 2, 4).contiguous())
        se = ws[e].view(torch.int16).reshape(nrt, 8, 32, 8).permute(0, 1, 3, 2).reshape(nrt, 64, 32)[tile]
        S.append(se.permute(0, 3, 1, 4, 2).contiguous())                         # rb kc w lane r
    return torch.stack(W), torch.stack(S)


def _scale_lut(device):
    """e4m3 byte -> bf16x2 of (value * 2^118), both halves."""
    v = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float() * (2.0 ** 118)
    bits = v.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    word = bits | (bits << 16)
    return word.to(device)


def prepare(weights):
    lib = _lib()
    dev = weights["w13"].device
    assert weights["w13"].shape[1:] == (2 * I, H // 2) and weights["w2"].shape[1:] == (H, I // 2)
    st = {"lib": lib, "start": 0, "count": LOCAL}
    w13q, w13s, w2q, w2s = [], [], [], []
    for e in range(LOCAL):
        w13q.append(_pack_weight(weights["w13"][e]))
        w13s.append(_pack_scales(weights["w13_sf"][e]))
        w2q.append(_pack_weight(weights["w2"][e]))
        w2s.append(_pack_scales(weights["w2_sf"][e]))
    st["w13q"] = torch.stack(w13q)
    st["w13s"] = torch.stack(w13s)
    st["w13s2"] = torch.stack([_pair_scales(x) for x in w13s]).view(torch.int32)
    st["w2s2"] = torch.stack([_pair_scales(x) for x in w2s]).view(torch.int32)
    st["w2q"] = torch.stack(w2q)
    st["w2s"] = torch.stack(w2s)
    del w13q, w13s, w2q, w2s
    if TW_CFG:
        rt = TW_CFG // 100
        if rt not in TB_PACK:
            st[f"w13b{rt}"], st[f"w13sb{rt}"] = _tb_pack(st["w13q"], st["w13s"], True, rt)
            st[f"w2b{rt}"], st[f"w2sb{rt}"] = _tb_pack(st["w2q"], st["w2s"], False, rt)
    for rt in TB_PACK:
        st[f"w13b{rt}"], st[f"w13sb{rt}"] = _tb_pack(st["w13q"], st["w13s"], True, rt)
        st[f"w2b{rt}"], st[f"w2sb{rt}"] = _tb_pack(st["w2q"], st["w2s"], False, rt)
    st["gs13"] = weights["w13_gs"][:LOCAL].float().contiguous()
    st["gs2"] = weights["w2_gs"][:LOCAL].float().contiguous()
    st["lut"] = _scale_lut(dev)
    st["w1"] = (1.0 + weights["post_norm_w"].float()).contiguous()
    torch.cuda.synchronize()
    return st


# (R, G) per batch size for gemm13 / gemm2: R row tiles (x16 rows) per item, G token groups of 8 per pass.
def _plan(M):
    if M == 1:
        return 10, 10
    if M <= 32:
        return 11, 11
    return 118, 118


PLAN_OVERRIDE = None
TB_RT = 0  # big-M moe_tb on (1) or off (0: old moe_big path)
TB_CFG = 48  # RT*10 + stages
TW_CFG = 218  # big-M wgmma kernel moe_tw: RT*100 + KG*10 + stages (0: off)
TB_PACK = ()  # RT values packed in prepare for moe_tb
_CFGS = (10, 11, 22, 118)
SK_CFG = 3088  # decode kernel cfg: 2DM = moe_sk (D, MB); 3[NC][NSTG] = moe_tk
TK2_NSTG = 8  # decode path: moe_tk2 stages (0: the 5-launch SK_CFG path)
SK_ZERO = 64  # CTAs zeroing Acc13 in route2


def kernel_fn(data, st):
    cfg = data["config"]
    assert cfg["local_start"] == st["start"] and cfg["local_count"] == st["count"]
    lib = st["lib"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].float().contiguous()
    M = x.shape[0]
    dev = x.device
    meta = torch.empty(68, dtype=torch.int32, device=dev)
    sroutes = torch.empty(M * TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((M * TOP_K, I), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M * TOP_K, H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    if M <= 32 and TK2_NSTG:
        inv = torch.empty(M * TOP_K, dtype=torch.int32, device=dev)
        y32 = torch.empty((M * TOP_K, H), dtype=torch.float32, device=dev)
        lib.moe_tk2_launch(1, TK2_NSTG, st["w13q"], st["w13s2"], st["lut"], st["gs13"], x, ids, st["start"], st["count"],
                           meta, sroutes, inv, inter)
        lib.moe_tk2_launch(0, TK2_NSTG, st["w2q"], st["w2s2"], st["lut"], st["gs2"], inter, ids, st["start"],
                           st["count"], meta, sroutes, inv, y32)
        lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], POST_NORM_EPS, out)
        return out
    if M <= 32:
        sk = PLAN_OVERRIDE(M) if PLAN_OVERRIDE else (SK_CFG, SK_CFG)
        inv = torch.empty(M * TOP_K, dtype=torch.int32, device=dev)
        acc13 = torch.empty((M * TOP_K, 2 * I), dtype=torch.float32, device=dev)
        acc2 = torch.empty((M * TOP_K, H), dtype=torch.float32, device=dev)
        lib.moe_route2_launch(ids, st["start"], st["count"], meta, sroutes, inv, acc13, SK_ZERO)
        lib.moe_sk_launch(1, sk[0], st["w13q"], st["w13s2"], st["lut"], x, meta, sroutes, acc13)
        lib.moe_act_launch(acc13, meta, sroutes, ids, st["gs13"], st["start"], inter, acc2)
        lib.moe_sk_launch(0, sk[1], st["w2q"], st["w2s2"], st["lut"], inter, meta, sroutes, acc2)
        lib.moe_norm2_launch(acc2, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], POST_NORM_EPS, out)
        return out
    p13, p2 = PLAN_OVERRIDE(M) if PLAN_OVERRIDE else _plan(M)
    assert p13 in _CFGS and p2 in _CFGS, (p13, p2)
    lib.moe_route_launch(ids, st["start"], st["count"], meta, sroutes)
    if TW_CFG:
        # inter here holds gemm2's B in the K-permuted order (see moe_tw)
        rt = TW_CFG // 100
        xg = torch.empty((M * TOP_K, H), dtype=torch.bfloat16, device=dev)
        lib.moe_gather_launch(x, meta, sroutes, xg)
        lib.moe_tw_launch(1, TW_CFG, st[f"w13b{rt}"], st[f"w13sb{rt}"], st["gs13"], st["lut"], xg, meta, sroutes, inter)
        lib.moe_tw_launch(0, TW_CFG, st[f"w2b{rt}"], st[f"w2sb{rt}"], st["gs2"], st["lut"], inter, meta, sroutes, y)
        lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out)
        return out
    if TB_RT:
        rt = TB_CFG // 10
        lib.moe_tb_launch(1, TB_CFG, st[f"w13b{rt}"], st[f"w13sb{rt}"], st["gs13"], st["lut"], x, meta, sroutes, inter)
        lib.moe_tb_launch(0, TB_CFG, st[f"w2b{rt}"], st[f"w2sb{rt}"], st["gs2"], st["lut"], inter, meta, sroutes, y)
        lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out)
        return out
    lib.moe_gemm13(p13, st["w13q"], st["w13s"], st["gs13"], st["lut"], x, meta, sroutes, inter)
    lib.moe_gemm2(p2, st["w2q"], st["w2s"], st["gs2"], st["lut"], inter, meta, sroutes, y)
    lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out)
    return out
