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
"""

_LIB = None
_CPP = ("int64_t moe_gemm13(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gemm2(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor);"
        "int64_t moe_norm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor);")


def _lib():
    global _LIB
    if _LIB is None:
        from torch.utils.cpp_extension import load_inline
        h = hashlib.sha256(CUDA_SRC.encode()).hexdigest()[:12]
        _LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_CPP, cuda_sources=CUDA_SRC,
                           functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch"],
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
    st["w2q"] = torch.stack(w2q)
    st["w2s"] = torch.stack(w2s)
    del w13q, w13s, w2q, w2s
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
_CFGS = (10, 11, 22, 118)


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
    p13, p2 = PLAN_OVERRIDE(M) if PLAN_OVERRIDE else _plan(M)
    assert p13 in _CFGS and p2 in _CFGS, (p13, p2)
    lib.moe_route_launch(ids, st["start"], st["count"], meta, sroutes)
    lib.moe_gemm13(p13, st["w13q"], st["w13s"], st["gs13"], st["lut"], x, meta, sroutes, inter)
    lib.moe_gemm2(p2, st["w2q"], st["w2s"], st["gs2"], st["lut"], inter, meta, sroutes, y)
    lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out)
    return out
