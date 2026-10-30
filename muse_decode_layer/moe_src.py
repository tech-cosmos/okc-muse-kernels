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

// launch, optionally as a PDL secondary (programmatic stream serialization: CUDA-graph capturable)
static int g_pdl = 4, g_twpf = 0, g_blk = 0;  // g_pdl bits: 1 decode gemm2, 2 decode norm2, 4 big-M kernels
template <typename... KA, typename... AA>
static int launch_k(void (*k)(KA...), int grid, int block, int smem, cudaStream_t st, bool pdl, AA... args) {
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(grid);
  lc.blockDim = dim3(block);
  lc.dynamicSmemBytes = smem;
  lc.stream = st;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = la;
  lc.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&lc, k, ((KA)args)...);
  return (int)cudaGetLastError();
}
int64_t moe_set_pdl(int64_t v, int64_t twpf) { g_pdl = (int)v; g_twpf = (int)twpf; return 0; }
int64_t moe_set_blk(int64_t v) { g_blk = (int)v; return 0; }
static int g_orx = 0;  // layer n5s25 DIAG (TK2F_ORX): tk2f oracle code of the current launch (cfg / 100000000)
static int g_sa = 0;  // s9: moe_twf uses the arithmetic scale decode (set from kernel_fn: prepare() checked the scales)
int64_t moe_set_sa(int64_t v) { g_sa = (int)v; return 0; }
static int g_hot = 0;  // s29: moe_twm skips the 65..128-token experts (moe_hot ran them)
int64_t moe_set_hot(int64_t v) { g_hot = (int)v; return 0; }

#define KD 4096          // K of both GEMMs
#define NOUT 4096        // output rows of both GEMMs (gate/up each I, w2 H)
#define TK2F_RING_SA 0   // s12: SA instantiations of the deeper tk2f rings (208/206/403/108): H200 diag 619ad6ad, all lose
#define TK2F_ORX 0       // layer n5s25: moe n5s32 oracle probes orz/orn/ornc (tuner timing only; DIAG check copies)
#define TK2F_DIAG 0      // layer s81: tk2f DIAG instantiations (no-math / KPS 4 SQ timing probes, tuner-only; set by TK2F_DIAG below)
#define TK2F_L2H 4       // s21: tk2f weight-copy L2 hints: 1 = evict_last for non-final 8-token groups of an expert, evict_first else; 2 = evict_last non-final only; 3 = evict_first final only
#define NORM_EARLY 1     // n4s1 (layer s53: set here, layer has its own _lib): twm/moe_hot flag each expert's finished gemm2 at meta[100 + e] (>= 1 << 20); moe_norme waits per token on them
#define TWM_EPI 1        // n6s3 (moe a60019d): staged, rolled tw_item / tw_epi epilogues (i-cache); bitwise
#define TWM_HOTC 0       // n6s3 (moe 0be4fa0): moe_twm's in-twm hot class compiled out (layer: twm_hot is in no TUNE list); set 1 to use twm_hot
#define TK2F_REL 2       // n6s6 (moe 5eb1e15+): tk2h gemm13 publishes by a helper warp NC + 1 (H200 DIAG 8c5f1376: tk2h d32 -0.75 us; tk2f 1-group paths +0.3: 2 = tk2h only)
#define TWM_REL 1        // n6s4 (moe 7d2a079): moe_twm gemm13+gemm2 releases by helper warp 9
#ifndef TWM_ORC
#define TWM_ORC 0        // fresh f1s2 ORACLE (timing only, wrong output, never submit): tw_item bit 1 dq ~free (wgmma kept),
                         // 2 no wgmma / 4 compiled-out epilogue: BOTH also let ptxas drop the dq (ncu: 15M vs 77M instr) = stream-only;
                         // 16/32/64 runtime-skip all / gemm13 / gemm2 epilogues, 128 epilogue without global stores
                         // f1s12: 8 no wgmma, dq kept by an XOR sink into acc; 256 flat read of all active experts' bytes
#endif
#define TWM_KSC 2        // n6s4 (moe 7d2a079): moe_twm built with + without the K-split tail code (picked per call); 0 = never split, 1 = always built in
#ifndef TK2G_PAR4
#define TK2G_PAR4 1      // n17: tk2g_bound's 4 calls on 4 warps + 32-bit p (bitwise: same bounds)
#endif
#ifndef TK2G_PAR
#define TK2G_PAR 1       // n4s5: GS16 weighted split bounds by warp 0 (tk2g_bound) instead of every thread (tk2g_split)
#endif
#define TK2_EXP 0        // s12: tk2h (TK2H/PSC) and weighted-split (TK2W) instantiations: H200 diags, no gain (off: compile time)

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

// s9 SA: e4m3 block scale byte k of w -> bf16x2 (both halves) of value * 2^118 without the LUT: exact for positive
// normal scales (prepare() checks all of them, else SA = 0): bf16 bits = (e << 7 | m << 4) + (238 << 7).
// s27 SCB (r-moe): two scales per decode: scp<P>(w) = bf16 halves (byte 2P, byte 2P + 1) in the format of sca<>; the
// broadcast of one half (blo / bhi) folds into HMUL2's .H0_H0 / .H1_H1 operand selector (free). Bitwise = sca<>.
template <int P>
__device__ __forceinline__ uint32_t scp(uint32_t w) {
  return __byte_perm(w, 0u, P ? 0x4342u : 0x4140u) * 16u + 0x77007700u;
}
__device__ __forceinline__ uint32_t blo(uint32_t p) { uint32_t r; asm("{.reg .b16 l,h; mov.b32 {l,h}, %1; mov.b32 %0, {l,l};}" : "=r"(r) : "r"(p)); return r; }
__device__ __forceinline__ uint32_t bhi(uint32_t p) { uint32_t r; asm("{.reg .b16 l,h; mov.b32 {l,h}, %1; mov.b32 %0, {h,h};}" : "=r"(r) : "r"(p)); return r; }

template <int K>
__device__ __forceinline__ uint32_t sca(uint32_t w) {
  return __byte_perm(w, 0u, 0x4040u | (K << 8) | K) * 16u + 0x77007700u;
}

// layer s73 (moe n20 S6, r-moe 4a87466): 6-bit block-scale codes (prepare _s6_pack: c = e4m3 byte - 80, all scales in
// [80, 143]). Per lane and SQ stage (KPS 2) 8 codes k = 4 h + r: n = the low nibbles (bits 4k), hw = the high 2 bits
// (bits 2k). One PRMT puts the nibble in bits 4-7 and the high bits in 8-9 of each half; | 0x7C00 = (c + 80) * 16 +
// 0x7700: the same bf16x2 as sca<r>() of the original byte, 2 instructions per scale.
__device__ __forceinline__ void s6dec(uint32_t n, uint32_t hw, int h, uint32_t* sc) {
  const uint32_t n4 = n << 4;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const uint32_t b = 2 * h + (r >> 1), c = 4 + h;
    const uint32_t x = __byte_perm((r & 1) ? n : n4, hw >> (2 * r), b | (c << 4) | (b << 8) | (c << 12));
    sc[r] = (x & 0x03F003F0u) | 0x7C007C00u;
  }
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
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
    meta[68 + tid] = 0;  // moe_twf: gemm13 items done per expert
#if NORM_EARLY
    meta[100 + tid] = 0;  // moe_twm/moe_hot: gemm2 items done per expert (+ 1 << 20 when all are)
#endif
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
                                                int start, int count, float eps, __nv_bfloat16* __restrict__ out,
                                                const int* rdy) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  __shared__ int se[8];
  __shared__ float stw[8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  if (rdy) {
    // n4s1 (NORM_EARLY): wait only for the experts of this token's routes (flagged by moe_twm / moe_hot after their
    // last gemm2 item of the expert): CTAs run on the SMs twm's drain frees. Y then read through L2 (ld.cg).
    if (tid < 8) {
      const int e = __ldcg(ids + tok * 8 + tid) - start;
      se[tid] = e;  // n5s24 (moe 725f39a): ids + tw read once, before the spin (no serial L2 trips after it)
      stw[tid] = __ldcg(tw + tok * 8 + tid);
      if (e >= 0 && e < count) {
        int fv;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(fv) : "l"(rdy + e) : "memory");
          if (fv >= (1 << 20)) break;
          __nanosleep(256);
        }
      }
    }
    __syncthreads();
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of gemm2
    if (tid < 8) {
      se[tid] = __ldcg(ids + tok * 8 + tid) - start;
      stw[tid] = __ldcg(tw + tok * 8 + tid);
    }
    __syncthreads();
  }
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = se[s];
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      uint4 raw = __ldcg(reinterpret_cast<const uint4*>(Y + ((size_t)(tok * 8 + s)) * NOUT + tid * 8));
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
    const float f = stw[s] * rsqrtf(tot / NOUT + eps);
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
// SLOT (s13, moe_tk2f PF & 4): Y row = route slot tok * 8 + s, already scaled by gs2 * 256 (no inv / gs loads)
template <bool SLOT = false>
__global__ void __launch_bounds__(512) moe_norm2(const float* __restrict__ Y, const int* __restrict__ inv,
                                                 const int* __restrict__ ids, const float* __restrict__ tw,
                                                 const float* __restrict__ w1, const float* __restrict__ gs, int start,
                                                 int count, float eps, __nv_bfloat16* __restrict__ out, int* zc) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  float v[8][8], ss[8], twv[8];
  bool loc[8];
  int eid[8];
  // n5s24 (moe 725f39a): all ids and tw of the token in one batch (tw was loaded per local slot after the barrier).
  // moe n10s1 IDW: ... and BEFORE the wait (they are complete before moe_tk2f starts: the harness's inputs, or the
  // layer's routing, joined into the stream before moe_tk2f); a branch on the loaded words keeps ptxas from hoisting
  // ACQBULK above them. Y (moe_tk2f's output) then needs .cg loads: ptxas hoists .nc loads whose address is known
  // above the wait (SASS checked). H100 layer d16/d32 sl -0.5..-1.7 us; bitwise.
#pragma unroll
  for (int s = 0; s < 8; ++s) { eid[s] = ids[tok * 8 + s]; twv[s] = tw[tok * 8 + s]; }
  {
    unsigned u = 0;
#pragma unroll
    for (int s = 0; s < 8; ++s) u ^= (unsigned)eid[s] ^ __float_as_uint(twv[s]);
    __shared__ unsigned pin;
    if (u == 0x7fbfffffu) *(volatile unsigned*)&pin = u;  // a store to its own word, never read (volatile: kept)
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of gemm2
  if (tok == 0 && tid < 128) zc[tid] = 0;  // moe_tk2f's segment counters: zero again for the next call
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = eid[s] - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      const float4* p = reinterpret_cast<const float4*>(Y + (size_t)(SLOT ? tok * 8 + s : __ldcg(inv + tok * 8 + s)) * NOUT + tid * 8);
      const float4 a = __ldcg(p), b = __ldcg(p + 1);  // .cg: see IDW above
      const float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      if (SLOT) {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[s][k] = f[k] * wv[k];
      } else {
        const float gsc = gs[e] * 256.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) v[s][k] = f[k] * gsc * wv[k];
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
    const float f = twv[s] * rsqrtf(tot / NOUT + eps);
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
__device__ __forceinline__ void prefetch_l2(const void* src, int bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(void* dst, const void* src, int bytes, uint64_t* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
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
// row R (local index in its expert's rows, ng token groups) -> row pair rp, group gi. blk = 0 or ng = 1: groups one
// after the other (rp fastest); else blocks of blk rps cycle through the groups, so a CTA re-reads the same weights
// for the next group right away (from L2)
__device__ __forceinline__ int tk2_map(int local, int ng, int blk, int rpe, int& gi) {
  if (ng == 1 || !blk) {
    gi = local / rpe;
    return local & (rpe - 1);
  }
  const int b = local / (blk * ng), w = local - b * blk * ng;
  gi = w / blk;
  return b * blk + (w - gi * blk);
}

template <int NRT, int NC, int NSTG, bool GATED>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2(
    const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, const uint32_t* __restrict__ lut,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids, int nroutes,
    int start, int count, int* __restrict__ meta, int* __restrict__ sroutes, int* __restrict__ inv,
    void* __restrict__ Out, int flags) {
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
  const int blk = flags & 0xFFFF;
  // PDL: let the next kernel's CTAs launch as soon as SMs free up (they wait for us before reading our output)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (GATED || (flags >> 16)) {
    // routing (M * 8 <= 256 routes): warp w takes routes [32 w, 32 w + 32); gemm2 recomputes it (same result)
    // instead of reading gemm13's meta, so its producer can stream w2 before gemm13 has finished
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
      if (GATED && blockIdx.x == 0) { sroutes[pos] = i; inv[i] = pos; }
    }
    if (GATED && blockIdx.x == 0 && tid < 33) {
      meta[33 + tid] = s_off[tid];
      if (tid < 32) meta[1 + tid] = s_e[tid];
      if (tid == 0) meta[0] = s_nact;
    }
  } else {  // gemm2 not a PDL secondary: gemm13's routing
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
        int gi;
        const int lr = tk2_map(R - s_rs[a], (s_rs[a + 1] - s_rs[a]) / RPE, blk, RPE, gi);
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
  if (!GATED) asm volatile("griddepcontrol.wait;" ::: "memory");  // inter is gemm13's output
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, rpb = 0, off = 0, n = 0, st = 0, ph = 0, rb = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      // segment = rows of one (expert, token group) in a row: the whole expert, or blk rps when it has > 1 group
      const int ng = (s_rs[a + 1] - s_rs[a]) / RPE;
      int gi;
      const int lr = tk2_map(R - s_rs[a], ng, blk, RPE, gi);
      if (ng == 1 || !blk) {
        rpb = 0;
        seg_start = s_rs[a] + gi * RPE;
        seg_end = seg_start + RPE;
      } else {
        rpb = lr - lr % blk;
        seg_start = R - lr % blk;
        seg_end = seg_start + blk;
      }
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
    const int rp = rpb + (R - seg_start);
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
  return launch_k(moe_tk2<NRT, NC, NSTG, GATED>, grid, (NC + 1) * 32, SMEM, st, !GATED && (g_pdl & 1),
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const float*)gs, (const __nv_bfloat16*)x,
      (const int*)ids, nroutes, start, count, (int*)meta, (int*)sroutes, (int*)inv, out, g_blk | ((!GATED && (g_pdl & 1)) << 16));
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
struct TMaps { CUtensorMap m[6]; };  // box rows 8, 16, 32, 64, 96 (s17: moe_twm hot class), 128 (s29: moe_hot)

// NW = consumer warps: 8 (RT 2, 2 consumer WGs + a producer WG, setmaxnreg) or 16 (RT 1, 4 consumer WGs + a
// producer warp; same weight/scale packing: warp 2w+r of NW 16 = (warp w, tile r) of NW 8; gated: even warps gate,
// odd warps up, joined through smem in the epilogue).
template <int RT, bool GATED, int NW = 8>
struct TWC {
  static constexpr int XB = 64 * 128;
  static constexpr int WB = NW * RT * 512;
  static constexpr int SB = NW * RT * 64;
  static constexpr int STG = XB + WB + SB;
  static constexpr int NRB = (GATED ? 2 * NOUT : NOUT) / 16 / (NW * RT);
  static constexpr int XCH = (NW == 16 && GATED) ? 8 * 32 * 32 * 4 : 0;  // gate/up exchange
  static constexpr int THREADS = NW == 16 ? 16 * 32 + 32 : 384;
  static constexpr int PS = RT == 4 ? 32 : 64;  // tokens per pass
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

template <int RT, int N, bool GATED, int NW>
__device__ __forceinline__ void tw_epi(float (&acc)[RT][N / 2], int lane, int warp, int nv, int off, int rb, float gsc,
                                       const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int H = GATED ? RT / 2 : RT;
#if TWM_EPI
  // n6s3: rolled 8-token-group loop (i-cache), accumulators shift down by 4 per group; same values/stores as below
#pragma unroll 1
  for (int gi = 0; gi < N / 8; ++gi) {
    if (gi * 8 >= nv) break;
#pragma unroll
    for (int r = 0; r < H; ++r) {
      const int tile = rb * NW * H + warp * H + r;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = gi * 8 + t * 2 + (q & 1);
        if (tok < nv) {
          const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
          const float v = acc[r][q] * gsc;
          if (GATED) {
            const float u = acc[r + H][q] * gsc;
            const int c = col & 63;
            const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
            Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
          } else {
            Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i + 4 < N / 2; ++i) acc[r][i] = acc[r][i + 4];
  }
  return;
#endif
#pragma unroll
  for (int r = 0; r < H; ++r) {
    const int tile = rb * NW * H + warp * H + r;
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

// n6s3 TWM_EPI: staged epilogue (see tw_epi for the layouts). stg = this warp's 8 x 64 bf16 staging buffer in smem
// (1 KB, past the ring: +8 KB dyn smem in twm/twf/hot). Writes 16-B (Y rows) / 8-B (RT 4 inter) / 4-B (RT 2 inter) chunks.
// The 8-token-group loop is NOT unrolled (group gi reads acc[.][0..3], then the accumulators shift down by 4): moe_twm
// is ~16.7k SASS, far over the i-cache, and the fully unrolled epilogues of rare item kinds (N24/N32) ran cold from L2
// (H200 DIAG 54eb9bdb: 6-10 us per N24/N32 gemm13 item end; the time stayed with the global stores removed). Bitwise = tw_epi.
template <int RT, int N, bool GATED, int NW>
__device__ __forceinline__ void tw_epi2(float (&acc)[RT][N / 2], int lane, int warp, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out,
                                        __nv_bfloat16* stg) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int H = GATED ? RT / 2 : RT;
  const int col0 = (rb * NW * H + warp * H) * 16, c0 = col0 & 63;
#pragma unroll 1
  for (int gi = 0; gi < N / 8; ++gi) {
    if (gi * 8 >= nv) break;
    __syncwarp();
#pragma unroll
    for (int r = 0; r < H; ++r)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = t * 2 + (q & 1), cw = r * 16 + g + ((q & 2) ? 8 : 0);
        const float v = acc[r][q] * gsc;
        if (GATED) {
          const float u = acc[r + H][q] * gsc;
          const int c = (c0 + cw) & 63;
          const int lc = (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
          stg[tok * 64 + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        } else {
          stg[tok * 64 + cw] = __float2bfloat16(v);
        }
      }
    __syncwarp();
#if TWM_ORC & 128  // staging kept, global stores skipped at run time
    if (gsc != 1234.5f) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i + 4 < N / 2; ++i) acc[r][i] = acc[r][i + 4];
      continue;
    }
#endif
    if (GATED) {
      // the warp's positions in its 64-column block: lc bit 2 fixed (RT 4: 4-element chunks) or bits 1-2 fixed (RT 2: 2)
      constexpr int CE = H == 2 ? 4 : 2;
      const int fb = H == 2 ? (c0 >> 5) * 4 : (c0 >> 4) * 2;
#pragma unroll
      for (int i = lane; i < 64; i += 32) {
        const int tok = i >> 3, p = (i & 7) * 8 + fb;
        if (gi * 8 + tok < nv) {
          __nv_bfloat16* dst = Out + (size_t)(off + gi * 8 + tok) * NOUT + (col0 & ~63) + p;
          if (CE == 4) *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(stg + tok * 64 + p);
          else *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(stg + tok * 64 + p);
        }
      }
    } else {
      constexpr int CH = H * 2;  // 16-B chunks per token
#pragma unroll
      for (int i = lane; i < 8 * CH; i += 32) {
        const int tok = i / CH, k = i % CH;
        if (gi * 8 + tok < nv)
          *reinterpret_cast<uint4*>(Out + (size_t)sroutes[off + gi * 8 + tok] * NOUT + col0 + k * 8) =
              *reinterpret_cast<const uint4*>(stg + tok * 64 + k * 8);
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i + 4 < N / 2; ++i) acc[r][i] = acc[r][i + 4];
  }
  __syncwarp();
}

// DYN: one main loop for both GEMMs (moe_twf), epilogue picked by g2 at run time
template <int RT, int KG, int N, int NSTG, bool GATED, int NW, bool DYN = false, bool SA = false, int STRIDE = 0,
          int XBO = 0,  // s29: XBO = x tile bytes (weights' offset in the stage) when != the TWC default
          bool KS = true>  // n6s4: false = no K-split tail code (part must be null; i-cache)
__device__ __forceinline__ void tw_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                        const uint32_t* slut, int warp, int lane, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out,
                                        bool g2 = false, int nkc = 64, float* part = nullptr, int* tick = nullptr,
                                        int* s_flag = nullptr, int pofs = 0) {
  using C = TWC<RT, GATED, NW>;
  constexpr int XBv = XBO ? XBO : C::XB;
  constexpr int NA = N / 2;
  float acc[RT][NA];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[KG][RT][4], afB[KG][RT][4];
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);
    if (kc) mbar_wait(&full[st], ph);
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XBv)[(warp * RT + r) * 32 + lane];
    uint32_t sc[(RT + 1) / 2];
    if (NW == 16) {
      sc[0] = *reinterpret_cast<const uint16_t*>(sb + XBv + C::WB + (((warp >> 1) * 32 + lane) * 2 + (warp & 1)) * 2);
    } else if (RT == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(sb + XBv + C::WB + (warp * 32 + lane) * RT * 2);
      sc[0] = v.x; sc[RT / 2 - 1] = v.y;
    } else {
      sc[0] = *reinterpret_cast<const uint32_t*>(sb + XBv + C::WB + (warp * 32 + lane) * RT * 2);
    }
    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);
      if (SA) {
        const uint32_t pr = (r & 1) ? scp<1>(sc[r >> 1]) : scp<0>(sc[r >> 1]);
        s0[r] = blo(pr);
        s1[r] = bhi(pr);
      } else {
        s0[r] = slut[sw & 0xFF];
        s1[r] = slut[(sw >> 8) & 0xFF];
      }
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
#if TWM_ORC & 1
          { uint32_t* a_ = odd ? afB[jj][r] : afA[jj][r]; const uint32_t q_ = u4get(wv[r], j0 + jj);
            a_[0] = q_; a_[1] = q_ ^ s0[r]; a_[2] = q_ ^ s1[r]; a_[3] = q_; }
#else
          if (odd) dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afB[jj][r]);
          else dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afA[jj][r]);
#endif
        }
      wg_fence();
#if TWM_ORC & 8  // fresh f1s12: no wgmma; the dq output is kept alive by an XOR sink into acc (~2 LOP3 per 4 regs)
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          const uint32_t* a_ = odd ? afB[jj][r] : afA[jj][r];
          acc[r][jj & 3] = __int_as_float(__float_as_int(acc[r][jj & 3]) ^ a_[0] ^ a_[1] ^ a_[2] ^ a_[3]);
        }
#elif !(TWM_ORC & 2)
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r) wgmma_bf16<N>(acc[r], odd ? afB[jj][r] : afA[jj][r], d0 + (j0 + jj) * 2);
#endif
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
#if TWM_ORC & 4
  return;
#endif
#if TWM_ORC & 16  // runtime skips (code kept: same i-cache footprint); gsc is never 1234.5
  if (gsc != 1234.5f) return;
#endif
#if TWM_ORC & 32
  if (!g2 && gsc != 1234.5f) return;
#endif
#if TWM_ORC & 64
  if (g2 && gsc != 1234.5f) return;
#endif
  const int g = lane >> 2, t = lane & 3;
  if (NW == 16 && GATED) {
    float* xb = reinterpret_cast<float*>(const_cast<char*>(smem) + NSTG * C::STG) + (warp >> 1) * 32 * 32;
    if (warp & 1) {
#pragma unroll
      for (int i = 0; i < NA; ++i) xb[i * 32 + lane] = acc[0][i];
    }
    asm volatile("bar.sync %0, 64;" ::"r"(1 + (warp >> 1)) : "memory");
    if (!(warp & 1)) {
      const int tile = rb * 8 + (warp >> 1);
#pragma unroll
      for (int gi = 0; gi < N / 8; ++gi)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tok = gi * 8 + t * 2 + (q & 1);
          if (tok < nv) {
            const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
            const float v = acc[0][gi * 4 + q] * gsc;
            const float u = xb[(gi * 4 + q) * 32 + lane] * gsc;
            const int c = col & 63;
            const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
            Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
          }
        }
    }
    asm volatile("bar.sync %0, 64;" ::"r"(1 + (warp >> 1)) : "memory");
    return;
  }
  if constexpr (DYN) {
    if (KS && part) {  // s11 K-split tail item: the first half to finish stores its FP32 partial, the second adds it
      const int ct = warp * 32 + lane;
      asm volatile("bar.sync 1, 256;" ::: "memory");  // s_flag is free (previous split item done with it)
      if (ct == 0) *s_flag = atomicAdd(tick, 1);
      asm volatile("bar.sync 1, 256;" ::: "memory");
      const int o = *s_flag;
      if (!o) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < NA; ++i) __stcg(part + (r * NA + i) * 256 + ct, acc[r][i]);
        __threadfence();
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (ct == 0) asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(tick + 512), "r"(1) : "memory");
        return;
      }
      if (ct == 0) {  // the other half took its ticket first: wait for its partial
        int v;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(tick + 512) : "memory");
        } while (!v);
        tick[0] = 0;  // last user of both words: zero for the next call
        tick[512] = 0;
        __threadfence();
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      const float* ot = part + pofs;
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[r][i] += __ldcg(ot + (r * NA + i) * 256 + ct);
    }
#if TWM_EPI
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(const_cast<char*>(smem) + NSTG * (STRIDE ? STRIDE : C::STG) + warp * 1024);
    if (g2) tw_epi2<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out, stg);
    else tw_epi2<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out, stg);
#else
    if (g2) tw_epi<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
    else tw_epi<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
#endif
  } else {
    tw_epi<RT, N, GATED, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
  }
}

template <int RT, int KG, int NSTG, bool GATED, int NW>
__global__ void __launch_bounds__(TWC<RT, GATED, NW>::THREADS, 1) moe_tw(
    const char* __restrict__ Wb, const char* __restrict__ Sb, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out, int pfk) {
  using C = TWC<RT, GATED, NW>;
  constexpr int NRB = C::NRB;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[33], s_item[NSTG];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += C::THREADS) slut[i] = lut[i];
  // PDL: gemm13 waits for moe_gather (and so moe_route) before reading meta, then lets gemm2 launch; gemm2 reads
  // meta (complete by then) and only waits (producer) before its first copy of gemm13's output
  if (GATED) asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NW * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int np = tid < nact ? ((s_off[e + 1] - s_off[e] + C::PS - 1) / C::PS) * NRB : 0;
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
  if (warp >= NW) {
    if (NW == 8) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > NW || lane != 0) return;
    int* counter = meta + (GATED ? 66 : 67);
    int st = 0, ph = 0, i = 0;
    int item = atomicAdd(counter, 1);
    if (!GATED) {
      if (item < nitems && pfk) {  // L2-prefetch the first item's first pfk k chunks while gemm13 finishes
        int a = 0;
        while (s_cum[a + 1] <= item) ++a;
        const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
        const int e = s_e[a];
        const char* wsrc = Wb + ((size_t)e * NRB + rb) * 64 * C::WB;
        const char* ssrc = Sb + ((size_t)e * NRB + rb) * 64 * C::SB;
        prefetch_l2(wsrc, pfk * C::WB);
        prefetch_l2(ssrc, pfk * C::SB);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    for (;; item = atomicAdd(counter, 1)) {
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
      const int off = s_off[e] + p * C::PS, nv = min(C::PS, s_off[e + 1] - off);
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
  if (NW == 8) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    int a = 0;
    while (s_cum[a + 1] <= item) ++a;
    const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
    const int e = s_e[a];
    const int off = s_off[e] + p * C::PS, nv = min(C::PS, s_off[e + 1] - off);
    const float gsc = gs[e] * 256.0f;
#define TWI(NN) tw_item<RT, KG, NN, NSTG, GATED, NW>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, Out)
    if (C::PS == 32) {  // N <= 32 only: ptxas sizes the wgmma pipeline for the largest variant in the function
    switch ((nv + 7) >> 3) {
      case 1: TWI(8); break;
      case 2: TWI(16); break;
      case 3: TWI(24); break;
      default: TWI(32); break;
    }
    } else {
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
    }
#undef TWI
  }
}

template <int RT, int KG, int NSTG, bool GATED, int NW = 8>
static int launch_tw(const void* wb, const void* sb, const void* gs, const void* lut, const TMaps& maps,
                     const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  using C = TWC<RT, GATED, NW>;
  constexpr int SMEM = NSTG * C::STG + C::XCH + 1024;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tw<RT, KG, NSTG, GATED, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  return launch_k(moe_tw<RT, KG, NSTG, GATED, NW>, grid, C::THREADS, SMEM, st, (g_pdl & 4) != 0,
      (const char*)wb, (const char*)sb, (const float*)gs, (const uint32_t*)lut, maps,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out, g_twpf);
}

// xg[pos] = x[sroutes[pos] >> 3] with each 64-chunk K-permuted (logical word (j*2+h)*4+t <- physical word 8t+2j+h).
__global__ void __launch_bounds__(256) moe_gather(const uint32_t* __restrict__ X, const int* __restrict__ meta,
                                                  const int* __restrict__ sroutes, uint32_t* __restrict__ XG) {
  const int pos = blockIdx.x;
  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of moe_route
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
  return launch_k(moe_norm, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const __nv_bfloat16*)y.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (__nv_bfloat16*)out.data_ptr(), (const int*)nullptr);
}
// n4s1: moe_norm after moe_twm (+ moe_hot) built with NORM_EARLY 1: per-token readiness from meta[100 + e]
int64_t moe_norme_launch(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start,
                         int64_t count, double eps, torch::Tensor out, torch::Tensor meta) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const __nv_bfloat16*)y.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (__nv_bfloat16*)out.data_ptr(),
      (const int*)meta.data_ptr() + 100);
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
  for (int i = 0; i < 6; ++i) {
    cuuint64_t dims[2] = {(cuuint64_t)KD, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)KD * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)(i < 4 ? 8 << i : i == 4 ? 96 : 128)};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g_encode(&maps.m[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  }
}
int64_t moe_gather_launch(torch::Tensor x, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor xg) {
  return launch_k(moe_gather, (int)sroutes.numel(), 256, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const uint32_t*)x.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(), (uint32_t*)xg.data_ptr());
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
  if (cfg == 1118) {  // NW 16, RT 1 (uses the RT 2 packing)
    if (gated) return launch_tw<1, 1, 8, true, 16>(wb, sb, gs, lut, maps, meta, sr, out, st);
    return launch_tw<1, 1, 8, false, 16>(wb, sb, gs, lut, maps, meta, sr, out, st);
  }
  return -1;
}
// ---- moe_twf: gemm13 and gemm2 of the big-M path in ONE persistent launch (RT 4, 32-token passes). The work
// counter hands out all gemm13 items (heaviest experts first), then all gemm2 items in the same expert order. A
// gemm13 item's consumers bump done[e] after writing their inter rows; a gemm2 item's producer waits for done[e] ==
// (gemm13 items of e) before its first copy of inter. All gemm13 items are handed out before any gemm2 item and
// never wait, and the grid is 1 CTA per SM (all resident), so this cannot deadlock. CTAs that finish gemm13 early
// start on gemm2 instead of idling through gemm13's tail.
#ifndef NEEDK
#define NEEDK 1
#endif
template <int RT, int NSTG, bool SA = false>
__global__ void __launch_bounds__(384, 1) moe_twf(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit) {
  using CG = TWC<RT, true, 8>;
  using CN = TWC<RT, false, 8>;
  static_assert(CG::STG == CN::STG, "stage layout");
  constexpr int PS = CG::PS;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_flag;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
    const int npass = tid < nact ? (s_off[e + 1] - s_off[e] + PS - 1) / PS : 0;
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      int incl = npass * (gq ? CN::NRB : CG::NRB);
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      s_cum[gq][tid + 1] = incl;
      if (tid == 0) s_cum[gq][0] = 0;
    }
  }
  __syncthreads();
  // s11: the last ns gemm2 items (the lightest: the tail) run as 2 K halves each (items n13 + n2f + 2 j + h)
  // The split starts at an expert boundary (moe_route orders an expert's tokens nondeterministically: a token's
  // result must not depend on whether its pass is split), so ns may come out a bit below nsplit.
  const int n13 = s_cum[0][32], n2 = s_cum[1][32];
  int n2f = n2;
  if (nsplit > 0)
    for (int a = 32; a >= 0; --a)
      if (s_cum[1][a] >= n2 - nsplit) n2f = s_cum[1][a];
  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
  // item -> (gemm2?, active slot a, pass p, row block rb, K half h (-1: whole K), split slot j)
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb, int& h, int& j) {
    g2 = item >= n13;
    h = -1;
    j = 0;
    int it = g2 ? item - n13 : item;
    if (g2 && it >= n2f) {
      j = (it - n2f) >> 1;
      h = (it - n2f) & 1;
      it = n2f + j;
    }
    const int* cum = s_cum[g2];
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = g2 ? CN::NRB : CG::NRB;
    const int rem = it - cum[a];
    p = rem / nrb;
    rb = rem - p * nrb;
  };
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8 || lane != 0) return;
    int* counter = meta + 66;
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1);; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int g2, a, p, rb, h, j;
      decode(item, g2, a, p, rb, h, j);
      const int e = s_e[a];
      const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
      const int k0 = h < 0 ? 0 : h * 32, k1 = h < 0 ? 64 : k0 + 32;
      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const int nrb = g2 ? CN::NRB : CG::NRB;
      const char* wsrc = (g2 ? W2 : W13) + ((size_t)e * nrb + rb) * 64 * CG::WB;
      const char* ssrc = (g2 ? S2 : S13) + ((size_t)e * nrb + rb) * 64 * CG::SB;
      const CUtensorMap* map = g2 ? &maps2.m[mi] : &maps13.m[mi];
      const int bytes = CG::WB + CG::SB + (1024 << mi);
      if (g2) {
        const int need = ((s_off[e + 1] - s_off[e] + PS - 1) / PS) * CG::NRB;
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= need * NEEDK) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * CG::STG;
        if (kc == k0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s(sb + CG::XB, wsrc + (size_t)kc * CG::WB, CG::WB, &full[st]);
        bulk_g2s(sb + CG::XB + CG::WB, ssrc + (size_t)kc * CG::SB, CG::SB, &full[st]);
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
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);
    const int e = s_e[a];
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    float* pt = h < 0 ? nullptr : part + (size_t)(2 * j + h) * 16384;
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
#define TWI(NN) tw_item<RT, 1, NN, NSTG, true, 8, true, SA>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \
                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384)
    if constexpr (RT == 4) {
      switch ((nv + 7) >> 3) {
        case 1: TWI(8); break;
        case 2: TWI(16); break;
        case 3: TWI(24); break;
        default: TWI(32); break;
      }
    } else {  // s11 RT 2: 64-token passes (each dequantized weight serves up to 64 tokens)
      switch ((nv + 7) >> 3) {
        case 1: TWI(8); break;
        case 2: TWI(16); break;
        case 3: case 4: TWI(32); break;
        default: TWI(64); break;
      }
    }
#undef TWI
    if (!g2) {
      // release this item's inter rows to gemm2 producers on other SMs (their TMA reads them: async proxy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
  }
}

template <bool SA, int RT>
static int64_t twf_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_,
                       torch::Tensor s2_, torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_,
                       torch::Tensor sr_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_,
                       int64_t ns) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<RT, true, 8>::STG + 1024 + (TWM_EPI ? 8192 : 0);  // n6s3 staging
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_twf<RT, NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_twf<RT, NSTG, SA>, grid, 384, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(), (int*)tk_.data_ptr(),
      (float*)part_.data_ptr(), (int)std::min<int64_t>(ns, std::min<int64_t>(tk_.numel() / 2, part_.numel() / 32768)));
}
int64_t moe_twf_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_,
                       torch::Tensor s2_, torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_,
                       torch::Tensor sr_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_,
                       int64_t ns) {
#define TWFA w13_, s13_, gs13_, w2_, s2_, gs2_, lut_, xg_, meta_, sr_, inter_, y_, tk_, part_, ns
  if (g_sa & 2) return (g_sa & 1) ? twf_go<true, 2>(TWFA) : twf_go<false, 2>(TWFA);  // s11: RT 2, 64-token passes
  return (g_sa & 1) ? twf_go<true, 4>(TWFA) : twf_go<false, 4>(TWFA);
#undef TWFA
}
// s11 moe_twm = moe_twf with the item shape chosen PER EXPERT: experts with > 32 tokens run RT 2 items with 64-token
// passes (each dequantized weight serves 64 tokens; packing w*b2), the others RT 4 items with 32-token passes (w*b4).
// Stage stride = the RT 4 stage (the RT 2 stage is smaller and uses the same x / weight / scale offsets scheme).
#ifndef TWM_TL
#define TWM_TL 0
#endif
#ifndef HOT_EMU
#define HOT_XN 16  // fresh f1s6: moe_hot (NH 128) items at wgmma N = nv rounded up to HOT_XN (16: 80/96/112/128; 0 = N 128 always)
#define HOT_ET 1  // n7s39: moe_hot (per > 0) triggers moe_twm (hx) before its own wait; twm polls meta[67] bit 30
#define HOT_EMU 0  // s29: H200-proxy emulation in moe_twm/moe_hot (tests only; see moe_twm)
#endif
#ifndef TWQ_PREG
#define TWQ_PREG 32  // s17 moe_twq: producer warpgroup / consumer registers after setmaxnreg (640 threads: 96 at launch)
#define TWQ_CREG 112
#endif
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
// s17 moe_twm hot class (cl 3, -nsplit bit 23 = TWM_HOT): an expert with 65..96 tokens runs ONE pass (instead of two
// 64-token RT 2 passes = two dequantizations of every weight) with one 16-row tile per warp and N <= 96 (48
// accumulator registers: within twm's budget, unlike RT 2 x N 96). Item = half an RT 2 row block of the w*b2 packing
// (slots w8 4h..4h+3 are contiguous: 4 KB of weights + 512 B of scales per k chunk); stage = x (96-row box) at 0,
// weights at 12 KB, scales at 16 KB. Gated: warp 2i = gate tile, 2i+1 = its up tile; the up warp hands its FP32
// accumulators to the gate warp through the item's LAST ring slot (24 KB <= one stage; after a consumer barrier:
// the other warpgroup may still be reading that slot), released only afterwards.
template <int N, int NSTG, bool SA, int STRIDE>
__device__ __forceinline__ void tw_hot(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                       const uint32_t* slut, int warp, int lane, int nv, int off, int rb2, float gsc,
                                       const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out, bool g2) {
  constexpr int NA = N / 2, XB = 96 * 128, WB = 4096;
  float acc[1][NA];
#pragma unroll
  for (int i = 0; i < NA; ++i) acc[0][i] = 0.f;
  uint32_t afA[4], afB[4];
  const int so = XB + WB + ((warp >> 1) * 32 + lane) * 4 + (warp & 1) * 2;
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const char* sb = smem + st * STRIDE;
    if (kc) mbar_wait(&full[st], ph);
    const uint4 wv = reinterpret_cast<const uint4*>(sb + XB)[warp * 32 + lane];
    const uint32_t word = *reinterpret_cast<const uint16_t*>(sb + so);
    uint32_t s0, s1;
    if (SA) {
      const uint32_t pr = scp<0>(word); s0 = blo(pr); s1 = bhi(pr);
    } else {
      s0 = slut[word & 0xFF]; s1 = slut[word >> 8];
    }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const bool odd = j0 & 1;
      if (odd) dq(u4get(wv, j0), s0, s1, afB);
      else dq(u4get(wv, j0), s0, s1, afA);
      wg_fence();
      wgmma_bf16<N>(acc[0], odd ? afB : afA, d0 + j0 * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (odd) fence_r(afA[i]);
        else fence_r(afB[i]);
      }
      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);
    }
    pst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 4; ++i) { fence_r(afA[i]); fence_r(afB[i]); }
#pragma unroll
  for (int i = 0; i < NA; ++i) fence_f(acc[0][i]);
  if (g2) {
    mbar_arrive(&empty[pst]);
    tw_epi<1, N, false, 8>(acc, lane, warp, nv, off, rb2, gsc, sroutes, Out);
    return;
  }
  float* x = reinterpret_cast<float*>(const_cast<char*>(smem) + pst * STRIDE) + (warp >> 1) * 32 * NA;
  asm volatile("bar.sync 1, 256;" ::: "memory");  // the other warpgroup's MMAs may still read this slot's x tile
  if (warp & 1) {
#pragma unroll
    for (int i = 0; i < NA; ++i) x[i * 32 + lane] = acc[0][i];
  }
  asm volatile("bar.sync %0, 64;" ::"r"(2 + (warp >> 1)) : "memory");
  if (!(warp & 1)) {
    const int g = lane >> 2, t = lane & 3, tile = rb2 * 4 + (warp >> 1);
#pragma unroll
    for (int gi = 0; gi < N / 8; ++gi)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = gi * 8 + t * 2 + (q & 1);
        if (tok < nv) {
          const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
          const float v = acc[0][gi * 4 + q] * gsc;
          const float u = x[(gi * 4 + q) * 32 + lane] * gsc;
          const int c = col & 63;
          const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
          Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        }
      }
  }
  asm volatile("bar.sync %0, 64;" ::"r"(2 + (warp >> 1)) : "memory");
  mbar_arrive(&empty[pst]);  // the slot held the exchange: release it only now
}

template <int NSTG, bool SA = false, int PB = 64,  // n12: PB = big-class pass size (64; 48 = N48 passes)
          bool KS = (TWM_KSC != 0)>  // n6s4: false = no K-split tail code (a split request is ignored); i-cache
__global__ void __launch_bounds__(384, 1) moe_twm(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const char* __restrict__ W13h, const char* __restrict__ S13h, const char* __restrict__ W2h,
    const char* __restrict__ S2h,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit, int hx = 0) {
  // s29 hx: moe_hot (the PDL primary) runs the experts with 65..128 tokens: skip them here, do not wait for the
  // primary at entry (it runs concurrently; moe_gather completed before moe_hot's own wait returned), wait at the end
  // (so that moe_norm, this kernel's PDL secondary, sees moe_hot's Y)
  using CG = TWC<4, true, 8>;
  using CN = TWC<4, false, 8>;
  using HG = TWC<2, true, 8>;
  using HN = TWC<2, false, 8>;
  static_assert(CG::STG == CN::STG && HG::STG <= CG::STG && HG::XB == CG::XB, "stage layout");
  constexpr int STRIDE = CG::STG;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_flag, s_big[2][32], s_ord[4], s_cost;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
#if TWM_REL
  __shared__ __align__(8) uint64_t relb[4];
  __shared__ int s_rel[4];
#endif
  int* done = meta + 68;
  // s14: nsplit < 0 = no K split; instead the lightest (last) non-big experts whose RT 4 gemm2 items add up to
  // <= -nsplit run their gemm2 at RT 2 with 32-token passes (class 2: half-size items, shorter drain)
  // (-nsplit) >> 12 = the same for gemm13 (RT 4 gemm13 items of the lightest experts)
  // (-nsplit) >> 24 = item order (s14): bit 0 alternating heavy / light experts, bit 1 timeline stamps, bit 2
  // pass-minor items (rb-major: all passes of a row block adjacent: L2 reuse), bits 3-4 = merge the big (> 32 token)
  // experts' items evenly into the first W = phi * N positions of each phase (phi 1, .85, .7), light ones between;
  // bit 5: merge in gemm13 only; bit 6: merged set = multi-pass (> 64 token) experts only
  // s19: bit 22 set = bits 12-21 are a K-split count (TWF_SPLIT, combinable with order / hot; no tail classes then)
  const int sx = nsplit < 0 ? ((-nsplit) >> 22) & 1 : 0;
  const int tail = nsplit < 0 && !sx ? (-nsplit) & 4095 : 0, tail13 = nsplit < 0 && !sx ? ((-nsplit) >> 12) & 1023 : 0;
  const int order = nsplit < 0 ? (-nsplit) >> 24 : 0, hot = nsplit < 0 ? ((-nsplit) >> 23) & 1 : 0;
  nsplit = nsplit < 0 ? (sx ? ((-nsplit) >> 12) & 1023 : 0) : nsplit;
  if (!KS) nsplit = 0;
  // s20: split count 1022 / 1023 = AUTO: K-split the last r light gemm2 items, r = (total item cost mod gridDim) when
  // that last wave is at most half full (halves then fill it; else no split). 1022: every item costs 1; 1023: light
  // RT 4 items weighted by their pass size (N8 1, N16 1.25, N32 1.5). Shape/routing-derived on device, no host sync.
  const int autos = nsplit >= 1022 ? nsplit - 1021 : 0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  if (!hx) asm volatile("griddepcontrol.wait;" ::: "memory");
#if HOT_ET
  // n7s39 HOT_ET: moe_hot (per > 0) triggers this launch at its start (before its own wait), so the moe_gather output is
  // complete only once moe_hot CTA 0 sets meta[67] bit 30 (after its wait); moe_route zeroed meta[67]
  else {
    if (tid == 0) {
      int v;
      do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(meta + 67) : "memory"); } while (!(v & (1 << 30)));
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __syncthreads();
  }
#endif
#if HOT_EMU
  // s29 H200-proxy only (HOT_EMU = moe_hot's grid): the last HOT_EMU CTAs start only as moe_hot CTAs exit, like the
  // twm CTAs queued behind moe_hot on a full device
  if (hx && tid == 0 && (int)blockIdx.x >= (int)gridDim.x - HOT_EMU) {
    const int k = blockIdx.x - (gridDim.x - HOT_EMU);
    int v;
    do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(meta + 67) : "memory"); } while ((v >> 20) <= k);
  }
#endif
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
#if TWM_REL
    for (int i = 0; i < 4; ++i) mbar_init(&relb[i], 256);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int sl = (order & 1) ? ((tid & 1) ? nact - 1 - (tid >> 1) : tid >> 1) : tid;
    const int e = tid < nact ? meta[1 + sl] : 0;
    s_e[tid] = e;
    int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    if (hx && nt > 64 && nt <= 128) nt = 0;
    const int big = nt > 32;
    const int hc = hot && nt > 64 && nt <= 96;  // s17 hot class: one pass, 1 tile per warp
    const int npass = hc ? 1 : big ? (nt + 63) / 64 : (nt + 31) / 32;
    int c2 = hc ? 3 : big, c13 = hc ? 3 : big;
    if (tail | tail13) {
      int sfx = big ? 1 << 20 : npass;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_down_sync(0xffffffffu, sfx, o);
        if (tid + o < 32) sfx += v;
      }
      if (!big && nt > 0 && sfx * CN::NRB <= tail) c2 = 2;
      if (!big && nt > 0 && sfx * CG::NRB <= tail13) c13 = 2;
    }
    s_big[0][tid] = c13;
    s_big[1][tid] = c2;
    if (autos) {
      const int w = (autos == 2 && !c2 && nt > 8) ? (nt > 16 ? 6 : 5) : 4;
      const int n13i = npass * (c13 ? HG::NRB * (c13 == 3 ? 2 : 1) : CG::NRB), n2i = npass * (c2 ? HN::NRB * (c2 == 3 ? 2 : 1) : CN::NRB);
      int cst = tid < nact ? w * (n13i + n2i) : 0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) cst += __shfl_xor_sync(0xffffffffu, cst, o);
      if (tid == 0) s_cost = cst;
    }
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      const int c = gq ? c2 : c13;
      int incl = npass * (c ? (gq ? HN::NRB : HG::NRB) * (c == 3 ? 2 : 1) : (gq ? CN::NRB : CG::NRB));
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      s_cum[gq][tid + 1] = incl;
      if (tid == 0) s_cum[gq][0] = 0;
    }
  }
  __syncthreads();
#if TWM_ORC & 256  // fresh f1s12 ORACLE: flat grid-stride read of every active expert's w13+s13+w2+s2 bytes (hot ones too:
                   // moe_hot exits at once), no items/stages/x; then the per-expert flags moe_norme waits on
  {
    constexpr size_t B13 = (size_t)CG::NRB * 64 * CG::WB, Q13 = (size_t)CG::NRB * 64 * CG::SB;
    constexpr size_t B2 = (size_t)CN::NRB * 64 * CG::WB, Q2 = (size_t)CN::NRB * 64 * CG::SB;
    constexpr size_t PER = (B13 + Q13 + B2 + Q2) / 16;
    const int nact = meta[0];
    const size_t tot = PER * nact, stride = (size_t)gridDim.x * 384;
    uint32_t sink = 0;
    for (size_t u0 = (size_t)blockIdx.x * 384 + tid; u0 < tot; u0 += 16 * stride) {
      uint4 v[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const size_t u = u0 + k * stride;
        v[k] = make_uint4(0, 0, 0, 0);
        if (u < tot) {
          const int a = (int)(u / PER);
          size_t o = (u - (size_t)a * PER) * 16;
          const int e = s_e[a];
          const char* src;
          if (o < B13) src = W13 + (size_t)e * B13 + o;
          else if ((o -= B13) < Q13) src = S13 + (size_t)e * Q13 + o;
          else if ((o -= Q13) < B2) src = W2 + (size_t)e * B2 + o;
          else src = S2 + (size_t)e * Q2 + (o - B2);
          asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w) : "l"(src));
        }
      }
#pragma unroll
      for (int k = 0; k < 16; ++k) sink ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }
    if (sink == 0x9E3779B9u && hx == 12345) Y[tid] = __float2bfloat16(1.0f);  // never true: keeps the loads
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      if (atomicAdd(meta + 66, 1) == (int)gridDim.x - 1) {
        for (int a = 0; a < nact; ++a) { const int e = s_e[a]; atomicAdd(done + e, 1 << 24); atomicAdd(done + 32 + e, 1 << 20); }
        __threadfence();
      }
    }
    if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
    return;
  }
#endif
  // s11: the last ns gemm2 items (the lightest: the tail) run as 2 K halves each (items n13 + n2f + 2 j + h)
  // The split starts at an expert boundary (moe_route orders an expert's tokens nondeterministically: a token's
  // result must not depend on whether its pass is split), so ns may come out a bit below nsplit.
  const int n13 = s_cum[0][32], n2 = s_cum[1][32];
  int n2f = n2;
  if (autos) {
    const int GU = 4 * gridDim.x, r = s_cost % GU;
    nsplit = (r > 0 && 2 * r <= GU) ? min((r + 3) / 4, 66) : 0;
  }
  if (nsplit > 0)
    for (int a = 32; a >= 0; --a) {
      if (a < 32 && s_big[1][a] != 0) break;  // s19: only light (RT 4) experts split (hot/big items take no K half)
      if (s_cum[1][a] >= n2 - nsplit) n2f = s_cum[1][a];
    }
  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
#if NORM_EARLY
  __shared__ int s_nd2[32];
  if (tid < 32) {
    const int c0 = s_cum[1][tid], c1 = s_cum[1][tid + 1];
    s_nd2[tid] = (c1 - c0) + max(0, c1 - max(c0, n2f));  // gemm2 item ends of slot tid (K halves count twice)
  }
#endif
  if (tid == 0) {
    int nbig = 0;
    // n6s4 LAYOUT: rolled (-321 SASS) only in the KS false build. tools/n5s17/loop64k.py: rolled in the KS true build (or
    // unrolled in the KS false one) puts the RT 4 N16 MMA loop across the function's 64 KB boundary (n5s17: +4..12%)
    if constexpr (KS) {
      while (nbig < 32 && s_big[0][nbig] == 1 && (!(order & 64) || s_cum[0][nbig + 1] - s_cum[0][nbig] > HG::NRB)) ++nbig;
    } else {
#pragma unroll 1
      while (nbig < 32 && s_big[0][nbig] == 1 && (!(order & 64) || s_cum[0][nbig + 1] - s_cum[0][nbig] > HG::NRB)) ++nbig;
    }
    const int mrg = (order >> 3) & 3;
    s_ord[0] = nbig;
    s_ord[1] = mrg == 0 ? 0 : mrg == 1 ? 100 : mrg == 2 ? 85 : 70;
    s_ord[2] = order & 36;
  }
  __syncthreads();
  // item -> (gemm2?, active slot a, pass p, row block rb, K half h (-1: whole K), split slot j)
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb, int& h, int& j) {
    g2 = item >= n13;
    h = -1;
    j = 0;
    int it = g2 ? item - n13 : item;
    if (g2 && it >= n2f) {
      j = (it - n2f) >> 1;
      h = (it - n2f) & 1;
      it = n2f + j;
    }
    const int* cum = s_cum[g2];
    const int phx = s_ord[1], ob = s_ord[2];
    if (phx && h < 0 && !(g2 && (ob & 32))) {
      const int nA = cum[s_ord[0]], N = g2 ? n2f : cum[32];  // s19: K-split items stay at the end
      if (nA > 0 && nA < N) {
        const int W = max(nA, N * phx / 100);
        const int ca = it < W ? it * nA / W : nA, ca1 = it + 1 <= W ? (it + 1) * nA / W : nA;
        it = ca1 > ca ? ca : nA + it - ca;
      }
    }
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = s_big[g2][a] ? (g2 ? HN::NRB : HG::NRB) * (s_big[g2][a] == 3 ? 2 : 1) : (g2 ? CN::NRB : CG::NRB);
    const int rem = it - cum[a];
    if (ob & 4) {
      const int np = (cum[a + 1] - cum[a]) / nrb;
      rb = rem / np;
      p = rem - rb * np;
    } else {
      p = rem / nrb;
      rb = rem - p * nrb;
    }
  };
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
#if TWM_REL
    if (warp == 9) {  // n6s3 REL helper: releases finished items (4 slots; items are >> the release latency apart)
      // n6s4: gemm2 items too (NORM_EARLY flags). slot = e | g2 << 8 | slot a << 16, -1 = done. The mbarrier wait
      // (acquire) + this thread's gpu fence (cumulative) order the consumers' stores before the count, as the
      // bar.sync + thread-0 fence did. Consumers no longer wait for the fence (H200: twm is consumer-bound)
      if (lane == 0)
        for (int k = 0;; ++k) {
          mbar_wait(&relb[k & 3], (k >> 2) & 1);
          const int v = s_rel[k & 3];
          if (v < 0) break;
          const int e = v & 255;
          __threadfence();
#if NORM_EARLY
          if (v & 256) {
            if (atomicAdd(done + 32 + e, 1) == s_nd2[v >> 16] - 1) atomicAdd(done + 32 + e, 1 << 20);
          } else
#endif
          atomicAdd(done + e, 1);
        }
      return;
    }
#endif
    if (warp > 8 || lane != 0) return;
    int* counter = meta + 66;
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1);; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int g2, a, p, rb, h, j;
      decode(item, g2, a, p, rb, h, j);
      const int e = s_e[a], cl = s_big[g2][a], big = cl != 0, PS = cl == 3 ? 128 : cl == 1 ? 64 : 32;
      const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
      const int k0 = h < 0 ? 0 : h * 32, k1 = h < 0 ? 64 : k0 + 32;
      const int mi = cl == 3 ? 4 : nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const int nrb = big ? (g2 ? HN::NRB : HG::NRB) : (g2 ? CN::NRB : CG::NRB);
      const int WB = big ? HG::WB : CG::WB, SB = big ? HG::SB : CG::SB;
      // hot class: item rb = 2 * (RT 2 row block) + half; the half's slots are the 2nd half of each k chunk's block
      const int rbp = cl == 3 ? rb >> 1 : rb, hh = cl == 3 ? rb & 1 : 0;
      const char* wsrc = (big ? (g2 ? W2h : W13h) : (g2 ? W2 : W13)) + ((size_t)e * nrb + rbp) * 64 * WB + hh * (WB / 2);
      const char* ssrc = (big ? (g2 ? S2h : S13h) : (g2 ? S2 : S13)) + ((size_t)e * nrb + rbp) * 64 * SB + hh * (SB / 2);
      const CUtensorMap* map = g2 ? &maps2.m[mi] : &maps13.m[mi];
      const int xbytes = cl == 3 ? 96 * 128 : 1024 << mi, wbytes = cl == 3 ? WB / 2 : WB, sbytes = cl == 3 ? SB / 2 : SB;
      const int xo = cl == 3 ? 96 * 128 : CG::XB;  // weights' offset in the stage
      const int bytes = wbytes + sbytes + xbytes;
      if (g2) {
        const int b13 = s_big[0][a], PS13 = b13 == 3 ? 128 : b13 == 1 ? 64 : 32;
        const int need = ((s_off[e + 1] - s_off[e] + PS13 - 1) / PS13) * (b13 ? HG::NRB * (b13 == 3 ? 2 : 1) : CG::NRB);
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= need * NEEDK) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (kc == k0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
        {  // moe s21 (12d95a9): evict_last if a later pass of this expert re-reads these weights, else evict_first
          uint64_t pol;
          if (off + PS < s_off[e + 1]) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
          else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
          bulk_g2s_h(sb + xo, wsrc + (size_t)kc * WB, wbytes, &full[st], pol);
          bulk_g2s_h(sb + xo + wbytes, ssrc + (size_t)kc * SB, sbytes, &full[st], pol);
        }
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
#if TWM_TL
  // s14 timeline (build with TWM_TL 1, order & 2): per CTA 64 u64 in part: [0] start, [1] items, 3 per item
  // (start, end, item), [63] exit
  unsigned long long* tl = (order & 2) ? reinterpret_cast<unsigned long long*>(part) + blockIdx.x * 64 : nullptr;
  int ntl = 0;
  if (tl && tid == 0) tl[0] = gtimer();
#endif
  int nrel = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
#if TWM_TL
    if (tl && tid == 0) {
      if (item < 0) { tl[1] = ntl; tl[63] = gtimer(); }
      else if (ntl < 20) { tl[2 + 3 * ntl] = gtimer(); tl[4 + 3 * ntl] = item; }
    }
#endif
    if (item < 0) {
#if TWM_REL
      if (tid == 0) s_rel[nrel & 3] = -1;
      mbar_arrive(&relb[nrel & 3]);
#endif
      break;
    }
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);
    const int e = s_e[a], cl = s_big[g2][a], big = cl != 0, PS = cl == 3 ? 128 : cl == 1 ? 64 : 32;
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    float* pt = h < 0 ? nullptr : part + (size_t)(2 * j + h) * 16384;
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
#define TWI(RT, NN) tw_item<RT, 1, NN, NSTG, true, 8, true, SA, STRIDE, 0, KS>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \
                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384)
    if (cl == 3) {
#if TWM_HOTC  // n6s3: the in-twm hot class only when python TWM_HOT / TWM_HOT_M asks for it (i-cache: moe_twm 11.4k -> 8.0k SASS without)
      if (nv <= 80) tw_hot<80, NSTG, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2);
      else tw_hot<96, NSTG, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2);
#endif
    } else if (!big) {
      switch ((nv + 7) >> 3) {
        case 1: TWI(4, 8); break;
        case 2: TWI(4, 16); break;
        case 3: TWI(4, 24); break;
        default: TWI(4, 32); break;
      }
    } else {
      switch ((nv + 7) >> 3) {
        case 1: TWI(2, 8); break;
        case 2: TWI(2, 16); break;
        case 3: case 4: TWI(2, 32); break;
        default: TWI(2, 64); break;
      }
    }
#undef TWI
    if (!g2) {
      // release this item's inter rows to gemm2 producers on other SMs (their TMA reads them: async proxy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
#if TWM_REL  // n6s3/n6s4 REL: the helper warp 9 fences + counts (consumers go on to the next item)
      if (tid == 0) s_rel[nrel & 3] = e;
      mbar_arrive(&relb[nrel & 3]);
      ++nrel;
#else
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
#endif
    }
#if NORM_EARLY
    else {  // n4s1: Y rows of expert e complete after its last gemm2 item (a K-split's 2nd half writes Y before it counts)
#if TWM_REL
      if (tid == 0) s_rel[nrel & 3] = e | 256 | (a << 16);
      mbar_arrive(&relb[nrel & 3]);
      ++nrel;
#else
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        if (atomicAdd(done + 32 + e, 1) == s_nd2[a] - 1) atomicAdd(done + 32 + e, 1 << 20);
      }
#endif
    }
#endif
#if TWM_TL
    if (tl && tid == 0 && ntl < 20) tl[3 + 3 * ntl] = gtimer();
    ++ntl;
#endif
  }
  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// ---- s29 moe_hot: the experts with 65..128 tokens (layer T = 256: 1-3 per call) in ONE pass of RT 2 items with
// N 128 (w*b2 packing: the same item rows as moe_twm's big class) instead of twm's two 64-token passes (= every weight
// dequantized twice). Own launch: 128 accumulators exceed twm's register budget, and a 2nd wgmma N in one main loop
// serializes the HGMMAs (s28). moe_twm runs as its PDL secondary (hx) on the SMs this grid leaves free and skips these
// experts. Items: all gemm13 items, then all gemm2 items (they wait on done[e] as in twm; deadlock-free the same way).
// Stage = x 128-row box (16 KB, SW128) + weights 8 KB + scales 1 KB.
// s23 NH 64: the experts with 33..64 tokens in one N 64 pass (layer T = 128), with moe_twq (flags bit 20) skipping them.
template <int NSTG, bool SA, int NH = 128>
__global__ void __launch_bounds__(384, 1) moe_hot(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, int per = 0, int xn = 1) {
  using HG = TWC<2, true, 8>;
  using HN = TWC<2, false, 8>;
  constexpr int XB = NH * 128, STRIDE = XB + HG::WB + HG::SB;
  static_assert(STRIDE % 1024 == 0 && HG::WB == HN::WB && HG::SB == HN::SB, "stage layout");
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_item[NSTG], s_nh;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
#if HOT_ET
  if (per > 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // n7s39: moe_twm polls meta[67] bit 30
#endif
  asm volatile("griddepcontrol.wait;" ::: "memory");
#if HOT_ET
  if (blockIdx.x == 0 && tid == 0) { __threadfence(); atomicOr(meta + 67, 1 << 30); }  // item counter: & 0xFFFFF
#endif
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#if TWM_ORC & 256  // fresh f1s12: moe_twm reads every expert's bytes flat
  return;
#endif
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    const int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    const bool h = nt > NH / 2 && nt <= NH;
    const unsigned b = __ballot_sync(0xffffffffu, h);
    if (h) s_e[__popc(b & ((1u << tid) - 1))] = e;
    if (tid == 0) s_nh = __popc(b);
  }
  __syncthreads();
  const int nh = s_nh, n13 = nh * HG::NRB, nitems = n13 + nh * HN::NRB;
  // n7s32 HOT_PER: per > 0 = only max(per >> 8, (per & 255) * nh) CTAs work (<= the grid); the others
  // exit at once, so their SMs go to moe_twm (the PDL secondary). A fixed 32-CTA grid made calls with 3-4 hot experts
  // the critical path (H100 v32 sl, 4 hot: 434 vs 335 us at 64 CTAs): the judge's slow v32 draws.
  if (per > 0 && (int)blockIdx.x >= max(per >> 8, (per & 255) * nh)) return;
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8 || lane != 0) return;
    int* counter = meta + 67;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1) & 0xFFFFF;; item = atomicAdd(counter, 1) & 0xFFFFF) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      const int g2 = item >= n13, it = g2 ? item - n13 : item, nrb = g2 ? HN::NRB : HG::NRB;
      const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e];
      const char* wsrc = (g2 ? W2 : W13) + ((size_t)e * nrb + rb) * 64 * HG::WB;
      const char* ssrc = (g2 ? S2 : S13) + ((size_t)e * nrb + rb) * 64 * HG::SB;
      const CUtensorMap* map = g2 ? &maps2.m[NH == 128 ? 5 : 3] : &maps13.m[NH == 128 ? 5 : 3];
      if (g2) {
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= HG::NRB * NEEDK) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = 0; kc < 64; ++kc, ++i) {
        if (kc && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (!kc) s_item[st] = item;
        mbar_expect_tx(&full[st], STRIDE);
        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s_h(sb + XB, wsrc + (size_t)kc * HG::WB, HG::WB, &full[st], pol);
        bulk_g2s_h(sb + XB + HG::WB, ssrc + (size_t)kc * HG::SB, HG::SB, &full[st], pol);
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
    const int g2 = item >= n13, it = g2 ? item - n13 : item, nrb = g2 ? HN::NRB : HG::NRB;
    const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e], nv = s_off[e + 1] - off;
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
#define HOT_ITEM(NN) tw_item<2, 1, NN, NSTG, true, 8, true, SA, STRIDE, XB>(smem, full, empty, st, ph, slut, warp, lane, \
                                                                           nv, off, rb, gsc, sroutes, g2 ? Y : Inter, g2)
    // fresh f1s6 HOT_XN: the padded tokens' MMA work (moe n5s21: N 128 for 65..128-token experts) is skipped; the x box
    // stays 128 rows, so only the wgmma N and the accumulator count change (rows >= nv were never stored: exact)
    if constexpr (NH == 128 && HOT_XN == 16) {
      if (xn && nv <= 80) HOT_ITEM(80);
      else if (nv <= 96) HOT_ITEM(96);
      else if (nv <= 112) HOT_ITEM(112);
      else HOT_ITEM(128);
    } else if constexpr (NH == 128 && HOT_XN == 8) {
      if (xn && nv <= 72) HOT_ITEM(72);
      else if (nv <= 80) HOT_ITEM(80);
      else if (nv <= 88) HOT_ITEM(88);
      else if (nv <= 96) HOT_ITEM(96);
      else if (nv <= 104) HOT_ITEM(104);
      else if (nv <= 112) HOT_ITEM(112);
      else if (nv <= 120) HOT_ITEM(120);
      else HOT_ITEM(128);
    } else {
      HOT_ITEM(NH);
    }
#undef HOT_ITEM
    if (!g2) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
#if NORM_EARLY
    else {
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        if (atomicAdd(done + 32 + e, 1) == HN::NRB - 1) atomicAdd(done + 32 + e, 1 << 20);
      }
    }
#endif
  }
#if HOT_EMU
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (tid == 0) atomicAdd(meta + 67, 1 << 20);
#endif
}
template <bool SA, int NH>
static int64_t hot_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                      torch::Tensor inter_, torch::Tensor y_, int grid) {
  const int xn = grid < 100000000;  // fresh f1s9: grid + 1e8 = HOT_XN off for this call (tuner A/B)
  grid %= 100000000;
  const int per = (grid / 100000) << 8 | (grid / 1000) % 100;  // n7s32 HOT_PER: grid = base * 1e5 + per * 1e3 + CTAs
  grid %= 1000;
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * (NH * 128 + TWC<2, true, 8>::WB + TWC<2, true, 8>::SB) + 1024 + (TWM_EPI ? 8192 : 0);
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(moe_hot<NSTG, SA, NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    init = true;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_hot<NSTG, SA, NH>, grid, 384, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(), per, xn);
}
int64_t moe_hot_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_, int64_t grid) {
#define HOTA w13_, s13_, gs13_, w2_, s2_, gs2_, lut_, xg_, meta_, sr_, inter_, y_, (int)grid
  if (g_hot & 4) return (g_sa & 1) ? hot_go<true, 64>(HOTA) : hot_go<false, 64>(HOTA);  // s23: 33..64-token experts
  return (g_sa & 1) ? hot_go<true, 128>(HOTA) : hot_go<false, 128>(HOTA);
#undef HOTA
}
template <bool SA>
static int64_t twm_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor w13h_, torch::Tensor s13h_, torch::Tensor w2h_, torch::Tensor s2h_,
                      torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor inter_,
                      torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024 + (TWM_EPI ? 8192 : 0);  // n6s3 staging
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_twm<NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
#if TWM_KSC == 2
    cudaFuncSetAttribute(moe_twm<NSTG, SA, 64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
#endif
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  auto kf = moe_twm<NSTG, SA>;
#if TWM_KSC == 2  // n6s4: both builds; the no-K-split one when this call has no split (ns > 0 or bit 22 of -ns)
  if (!(ns > 0 || (ns < 0 && (((-ns) >> 22) & 1)))) kf = moe_twm<NSTG, SA, 64, false>;
#endif
  return launch_k(kf, grid, 384, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const char*)PTR(w13h_), (const char*)PTR(s13h_),
      (const char*)PTR(w2h_), (const char*)PTR(s2h_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(), (int*)tk_.data_ptr(),
      (float*)part_.data_ptr(), (int)std::min<int64_t>(ns, std::min<int64_t>(tk_.numel() / 2, part_.numel() / 32768)),
      g_hot);
}
int64_t moe_twm_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor w13h_, torch::Tensor s13h_, torch::Tensor w2h_,
                       torch::Tensor s2h_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns) {
#define TWMA w13_, s13_, gs13_, w2_, s2_, gs2_, w13h_, s13h_, w2h_, s2h_, lut_, xg_, meta_, sr_, inter_, y_, tk_, part_, ns
  return (g_sa & 1) ? twm_go<true>(TWMA) : twm_go<false>(TWMA);
#undef TWMA
}
// ---- s17 moe_twq: moe_twm with 16 consumer warps (4 warpgroups) x RT 2 instead of 8 x RT 4. An item covers the same
// 32 row tiles (it reads the w*b4 / w*sb4 packing: warp w16 takes slots of 8-warp warp w16 / 2), so the tensor work
// and the bytes per stage are unchanged, but each scheduler has 4 consumer warps instead of 2 (s17 ncu: twm items are
// latency-bound, 2 warps per scheduler, issue 43-62% busy). 640 threads: 4 consumer warpgroups (setmaxnreg 112; ptxas
// compiles the whole kernel at 96 registers) + a producer warpgroup (one thread works; setmaxnreg 32). Gated warp w16: gate tile rb*16 + w16 (slot r = w16 & 1), up = slot 2 + (w16 & 1);
// gemm2: tiles rb*32 + 2*w16 + {0, 1} (slots 2*(w16 & 1) + {0, 1}). All experts run 32-token passes (64-token
// passes need > 96 registers at 640 threads: wgmma serialized; an inline producer in warp 0 to get 128 registers was
// 2x slower, s17 NOTES): s17 proxy wins at T=128 (v16) only; T=256 (v32, hot experts of 90+ tokens) stays on twm.
template <int N, int NSTG, bool SA>
__device__ __forceinline__ void tq_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                        const uint32_t* slut, int warp, int lane, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out, bool g2,
                                        int nkc = 64, float* part = nullptr, int* tick = nullptr, int* s_flag = nullptr,
                                        int h = 0, int np = 1) {
  using C = TWC<4, true, 8>;
  constexpr int NA = N / 2, STRIDE = C::STG;
  float acc[2][NA];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[2][4], afB[2][4];
  const int w8 = warp >> 1, sub = warp & 1;
  const int wo0 = (w8 * 4 + (g2 ? 2 * sub : sub)) * 32 + lane, wo1 = wo0 + (g2 ? 32 : 64);
  const uint32_t sel = g2 ? (sub ? 0x7654u : 0x3210u) : (sub ? 0x7632u : 0x5410u);
  const int so = C::XB + C::WB + (w8 * 32 + lane) * 8;
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * STRIDE;
    if (kc) mbar_wait(&full[st], ph);
    const uint4* wp = reinterpret_cast<const uint4*>(sb + C::XB);
    uint4 wv[2];
    wv[0] = wp[wo0];
    wv[1] = wp[wo1];
    const uint2 v = *reinterpret_cast<const uint2*>(sb + so);
    const uint32_t word = __byte_perm(v.x, v.y, sel);
    uint32_t s0[2], s1[2];
    if (SA) {
      const uint32_t p0 = scp<0>(word), p1 = scp<1>(word);
      s0[0] = blo(p0); s1[0] = bhi(p0); s0[1] = blo(p1); s1[1] = bhi(p1);
    } else {
      s0[0] = slut[word & 0xFF]; s1[0] = slut[(word >> 8) & 0xFF];
      s0[1] = slut[(word >> 16) & 0xFF]; s1[1] = slut[word >> 24];
    }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const bool odd = j0 & 1;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (odd) dq(u4get(wv[r], j0), s0[r], s1[r], afB[r]);
        else dq(u4get(wv[r], j0), s0[r], s1[r], afA[r]);
      }
      wg_fence();
#pragma unroll
      for (int r = 0; r < 2; ++r) wgmma_bf16<N>(acc[r], odd ? afB[r] : afA[r], d0 + j0 * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (odd) fence_r(afA[r][i]);
          else fence_r(afB[r][i]);
        }
      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);
    }
    pst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { fence_r(afA[r][i]); fence_r(afB[r][i]); }
#pragma unroll
    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
  if (part) {  // n21 K-split tail item (np parts of 64 / np k chunks, gemm2 only): every part stores its FP32 partial,
    // then takes a ticket; the last one sums the np partials in part order (deterministic) and runs the epilogue
    const int ct = warp * 32 + lane;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < NA; ++i) __stcg(part + (size_t)h * 16384 + (r * NA + i) * 512 + ct, acc[r][i]);
    __threadfence();
    asm volatile("bar.sync 1, 512;" ::: "memory");  // s_flag is free (the previous split item is done with it)
    if (ct == 0) {
      const int o = atomicAdd(tick, 1);
      if (o == np - 1) { *tick = 0; __threadfence(); }  // last user: zero for the next call
      *s_flag = o;
    }
    asm volatile("bar.sync 1, 512;" ::: "memory");
    if (*s_flag != np - 1) return;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < NA; ++i) {
        float v = __ldcg(part + (r * NA + i) * 512 + ct);
        for (int q = 1; q < np; ++q) v += __ldcg(part + (size_t)q * 16384 + (r * NA + i) * 512 + ct);
        acc[r][i] = v;
      }
  }
  if (g2) tw_epi<2, N, false, 16>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
  else tw_epi<2, N, true, 16>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
}

template <int NSTG, bool SA>
__global__ void __launch_bounds__(640, 1) moe_twq(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, float* __restrict__ part, int flags, int* __restrict__ tick) {
  using C = TWC<4, true, 8>;
  constexpr int STRIDE = C::STG, NR13 = TWC<4, true, 8>::NRB, NR2 = TWC<4, false, 8>::NRB;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_big[32], s_flag;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // n21 K split: the last ~(flags >> 8 & 255) gemm2 items (from an expert boundary) run as 2^(flags >> 16 & 3) K parts
  const int kq = (flags >> 16) & 3, npt = 1 << kq;
  int nsplit = kq ? (flags >> 8) & 255 : 0;
  for (int i = tid; i < 256; i += 640) slut[i] = lut[i];
  // s23 hx (flags bit 20): moe_hot<NH 64> (the PDL primary) runs the experts with 33..64 tokens: skip them, no wait at
  // entry (moe_hot triggers after its own wait: the gather is complete), wait at the end (moe_norm sees moe_hot's Y)
  const int hx = (flags >> 20) & 1;
  if (!hx) asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 16 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    if (hx && nt > 32 && nt <= 64) nt = 0;
    const int big = 0;  // s17: 64-token passes do not fit (registers)
    const int npass = big ? (nt + 63) / 64 : (nt + 31) / 32;
    s_big[tid] = big;
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      int incl = npass * (gq ? NR2 : NR13);
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      s_cum[gq][tid + 1] = incl;
      if (tid == 0) s_cum[gq][0] = 0;
    }
  }
  __syncthreads();
  const int n13 = s_cum[0][32], n2 = s_cum[1][32];
  int n2f = n2;
  if (nsplit > 0)
    for (int a = 32; a >= 0; --a)
      if (s_cum[1][a] >= n2 - nsplit) n2f = s_cum[1][a];
  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
  // item -> (gemm2?, active slot a, pass p, row block rb, K half h (-1: whole K), split slot j)
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb, int& h, int& j) {
    g2 = item >= n13;
    h = -1;
    j = 0;
    int it = g2 ? item - n13 : item;
    if (g2 && it >= n2f) {
      j = (it - n2f) >> 1;
      h = (it - n2f) & 1;
      it = n2f + j;
    }
    const int* cum = s_cum[g2];
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = g2 ? NR2 : NR13, rem = it - cum[a];
    if (flags & 8) {  // pass-minor: all passes of a row block adjacent (later passes read the weights from L2)
      const int np = (cum[a + 1] - cum[a]) / nrb;
      rb = rem / np;
      p = rem - rb * np;
    } else {
      p = rem / nrb;
      rb = rem - p * nrb;
    }
  };
  if (warp >= 16) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(TWQ_PREG) : "memory");
    if (warp > 16 || lane != 0) return;
    int* counter = meta + 66;
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1);; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int g2, a, p, rb, h, j;
      decode(item, g2, a, p, rb, h, j);
      const int e = s_e[a], PS = s_big[a] ? 64 : 32;
      const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
      const int k0 = h < 0 ? 0 : h * (64 >> kq), k1 = h < 0 ? 64 : k0 + (64 >> kq);
      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const int nrb = g2 ? NR2 : NR13;
      const char* wsrc = (g2 ? W2 : W13) + ((size_t)e * nrb + rb) * 64 * C::WB;
      const char* ssrc = (g2 ? S2 : S13) + ((size_t)e * nrb + rb) * 64 * C::SB;
      const CUtensorMap* map = g2 ? &maps2.m[mi] : &maps13.m[mi];
      const int bytes = C::WB + C::SB + (1024 << mi);
      if (g2) {
        const int need = ((s_off[e + 1] - s_off[e] + PS - 1) / PS) * NR13;
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= need) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (kc == k0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s(sb + C::XB, wsrc + (size_t)kc * C::WB, C::WB, &full[st]);
        bulk_g2s(sb + C::XB + C::WB, ssrc + (size_t)kc * C::SB, C::SB, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(TWQ_CREG) : "memory");
  int st = 0, ph = 0;
#if TWM_TL
  unsigned long long* tl = (flags & 2) ? reinterpret_cast<unsigned long long*>(part) + blockIdx.x * 64 : nullptr;
  int ntl = 0;
  if (tl && tid == 0) tl[0] = gtimer();
#endif
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
#if TWM_TL
    if (tl && tid == 0) {
      if (item < 0) { tl[1] = ntl; tl[63] = gtimer(); }
      else if (ntl < 20) { tl[2 + 3 * ntl] = gtimer(); tl[4 + 3 * ntl] = item; }
    }
#endif
    if (item < 0) break;
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);
    const int e = s_e[a], PS = s_big[a] ? 64 : 32;
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
    float* pt = h < 0 ? nullptr : part + (size_t)j * npt * 16384;
#define TQI(NN) tq_item<NN, NSTG, SA>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \
                                      h < 0 ? 64 : 64 >> kq, pt, tick + j, &s_flag, h, npt)
    switch ((nv + 7) >> 3) {
      case 1: TQI(8); break;
      case 2: TQI(16); break;
      case 3: TQI(24); break;
      default: TQI(32); break;
    }
#undef TQI
    if (!g2) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 512;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
#if TWM_TL
    if (tl && tid == 0 && ntl < 20) tl[3 + 3 * ntl] = gtimer();
    ++ntl;
#endif
  }
  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// n21: clamp the K-split item count to the partial buffer (16384 floats per part) and the ticket words
static int64_t twq_flags(int64_t f, int64_t npart, int64_t ntk) {
  const int64_t kq = (f >> 16) & 3, ns = (f >> 8) & 255;
  if (!kq || !ns) return f & ~((int64_t)0x3FF00);
  const int64_t m = std::min<int64_t>(ns, std::min<int64_t>(npart / (16384 << kq), ntk));
  return (f & ~((int64_t)0x3FF00)) | (m << 8) | (m ? kq << 16 : 0);
}
template <bool SA>
static int64_t twq_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                      torch::Tensor inter_, torch::Tensor y_, torch::Tensor part_, torch::Tensor tk_, int64_t flags) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_twq<NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_twq<NSTG, SA>, grid, 640, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(),
      (float*)part_.data_ptr(), (int)twq_flags(flags, part_.numel(), tk_.numel()), (int*)tk_.data_ptr());
}
int64_t moe_twq_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_, torch::Tensor part_, torch::Tensor tk_, int64_t flags) {
#define TWQA w13_, s13_, gs13_, w2_, s2_, gs2_, lut_, xg_, meta_, sr_, inter_, y_, part_, tk_, flags
  return (g_sa & 1) ? twq_go<true>(TWQA) : twq_go<false>(TWQA);
#undef TWQA
}
// ---- moe_tk2f: the decode gemm13 and gemm2 in ONE launch (moe_tk2 rows, one routing, one ring). Phase 1 = gemm13
// rows, phase 2 = gemm2 rows (each split over the CTAs as in moe_tk2); the producer streams phase-2 weights as soon
// as its phase-1 copies are issued. A CTA publishes the phase-1 rows it finished of each (expert, group) segment in
// cnt[a * 4 + gi] (bar + threadfence + atomicAdd); a phase-2 segment waits (ld.acquire) for cnt == 256 before it
// loads its x (= inter) fragments. Deadlock-free: phase 1 never waits, and every CTA (grid = #SMs) becomes resident
// once earlier kernels free their SMs. cnt lives in the state, zero between calls: moe_norm2 (after this kernel) zeroes it again (a memset node cost ~2.5 us).
// layer s68 (moe n14 TK2F_SQ, r-moe 01326b4): SQ = Wq is the single-copy repack (prepare _sq_pack: [E][RPE][NQ][SB bytes]
// = {tile lr chunks, tile lr + RPE chunks, scales}, exactly the stage's smem image): ONE bulk copy per stage instead of 3.
template <int NRT, int NC, int NSTG, int KPS, bool SQ = false, bool S6 = false, bool ORN = false>
__device__ __forceinline__ void tk2f_produce(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, int r0,
                                             int r1, const int* s_rs, const int* s_e, uint4* smem, uint64_t* full,
                                             uint64_t* empty, int& st, int& ph, int& i) {
  constexpr int RPE = NRT / 2, NQ = 64 / (NC * KPS), SB = NC * KPS * (S6 ? 1120 : 1152), KB = NC * KPS;
  static_assert(!S6 || (SQ && KPS == 2), "S6: SQ stages, KPS 2");
  int a = 0;
#if TK2F_L2H
  // s21: an expert with > 8 tokens is streamed once per 8-token group; the groups' rows sit 256 rows apart (other
  // CTAs, other times), so without hints the re-read often comes from HBM again (H100 ncu d16 s7: +12% DRAM bytes).
  uint64_t pol_last, pol_first;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_last));
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_first));
#endif
  for (int R = r0; R < r1; ++R) {
    while (s_rs[a + 1] <= R) ++a;
    const int e = s_e[a];
    const int lr = (R - s_rs[a]) & (RPE - 1);
    const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
    const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
    const char* qsrc = reinterpret_cast<const char*>(Wq) + ((size_t)e * RPE + lr) * NQ * SB;  // SQ
#if TK2F_L2H
    // mode 4: every group of a multi-group expert (the other group's CTA may read the same row first)
    const bool nfin = TK2F_L2H == 4 ? s_rs[a + 1] - s_rs[a] > RPE : R + RPE < s_rs[a + 1];
    const int hk = nfin ? (TK2F_L2H == 3 ? 0 : 1) : (TK2F_L2H == 2 ? 0 : 2);  // 0 none, 1 evict_last, 2 evict_first
    const uint64_t pol = hk == 1 ? pol_last : pol_first;
#endif
    for (int q = 0; q < NQ; ++q, ++i) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      char* sbase = reinterpret_cast<char*>(smem) + st * SB;
      if (ORN) { mbar_arrive(&full[st]); if (++st == NSTG) { st = 0; ph ^= 1; } continue; }  // n5s25 orn: no bytes
      mbar_expect_tx(&full[st], SB);
#if TK2F_L2H
      if (SQ) {
        if (hk) bulk_g2s_h(sbase, qsrc + q * SB, SB, &full[st], pol);
        else bulk_g2s(sbase, qsrc + q * SB, SB, &full[st]);
      } else if (hk) {
        bulk_g2s_h(sbase, src + q * KB * 32, KB * 512, &full[st], pol);
        bulk_g2s_h(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st], pol);
        bulk_g2s_h(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st], pol);
      } else
#else
      if (SQ) bulk_g2s(sbase, qsrc + q * SB, SB, &full[st]);
      else
#endif
      {
        bulk_g2s(sbase, src + q * KB * 32, KB * 512, &full[st]);
        bulk_g2s(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st]);
        bulk_g2s(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st]);
      }
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
  }
}

// n6s6 TK2F_REL: the consumers' gemm13 publishes (bar.sync 2 + thread-0 gpu fence + atomicAdd at each segment end)
// move to a helper warp (NC + 1): consumers arrive on g_tkrel[k & 7] (release.cta) and go on with the next row. gemm13
// segments are 256 rows on 256-aligned row-space boundaries (tk2f GS 8 and tk2h), so the helper derives the k-th publish
// (segment a0 / 256 + k, its row count, its counter) from the CTA's rows [a0, a1) alone. Its mbarrier wait (acquire)
// + gpu fence (cumulative) order every consumer's inter stores before the count, as bar.sync + thread-0 fence did.
__shared__ __align__(8) uint64_t g_tkrel[8];
template <int NC, int DYN, int FN, int GS, int MRG, int PF>
__host__ __device__ constexpr bool tk2f_rel_on() {
  return TK2F_REL && (TK2F_REL != 2 || MRG != 0) && NC == 8 && !DYN && !FN && !(PF & 16) && !(PF & 524288) && !(PF & 65536) && !(PF & 256) &&
         (MRG == 1 || MRG == 2 || (GS == 8 && MRG == 0));
}
__device__ __forceinline__ void tk2f_rel_arrive(int seg_start, int r0) { mbar_arrive(&g_tkrel[((seg_start >> 8) - (r0 >> 8)) & 7]); }
template <bool SEGC>
__device__ __forceinline__ void tk2f_rel_helper(int* cnt, int a0, int a1) {
  if (a0 >= a1) return;
  const int s0 = a0 >> 8, s1 = (a1 - 1) >> 8;
  for (int s = s0; s <= s1; ++s) {
    const int k = s - s0;
    mbar_wait(&g_tkrel[k & 7], (k >> 3) & 1);
    __threadfence();
    atomicAdd(SEGC ? cnt + 64 + s : cnt, min(a1, (s + 1) * 256) - max(a0, s * 256));
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA = false, int PF = 0, bool REL = false>
__device__ __forceinline__ void tk2f_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int need, int r0, int r1,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint64_t* full, uint64_t* empty, int& st,
                                             int& ph, int& rb) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * ((PF & 1024) ? 1120 : 1152);
  constexpr bool S6 = (PF & 1024) != 0;  // layer s73 (moe n20): 6-bit scale codes
  constexpr bool BAL = (PF & 32) != 0;  // moe n8 BAL: per-segment counters cnt[64 + j]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      ci = a * 4 + gi;
      if (BAL && !GATED) {
        const int* cp = cnt + 64 + seg_start / RPE;
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cp) : "memory");
        } while (c < 2 * RPE);
      } else if (!GATED && !((PF & 2) && R != r0)) {  // all gemm13 rows written (by any CTA): see moe_tk2f. PF & 2 (s13): only
        int c;                                    // at the CTA's first gemm2 row (cnt never decreases within a call)
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);
      }
      const bool v = g < n;
      int row = 0;
      if (v) row = GATED ? (s_srt[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        if (GATED) {
          xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
          xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
        } else {  // written in this kernel: L2 only
          xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
          xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
        }
      }
    }
    if ((PF & 1) && R + 1 == seg_end && R + 1 < r1) {  // s13: L1 prefetch of the next segment's x during this row
      const int a2 = seg_end >= s_rs[a + 1] ? a + 1 : a;
      const int e2 = s_e[a2], off2 = s_off[e2] + ((seg_end - s_rs[a2]) / RPE) * 8;
      if (g < s_off[e2 + 1] - off2) {
        const int row2 = GATED ? (s_srt[off2 + g] >> 3) : (off2 + g);
        const char* p = reinterpret_cast<const char*>(X + (size_t)row2 * KD + 16 * t);
#pragma unroll
        for (int qq = 0; qq < NQ; ++qq) asm volatile("prefetch.global.L1 [%0];" ::"l"(p + (qq * NC + warp) * 128));
      }
    }
    constexpr int NA = (PF & 8) ? 2 : 1;  // s13 PF & 8: 2 accumulator chains per tile (j even / odd), summed at the end
    float acc[2][4], acc2[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = acc2[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      uint4 wv0[KPS], wv1[KPS];
      uint32_t sw[KPS], s6n = 0, s6h = 0;
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        wv0[h] = sw4[h * SW];
        wv1[h] = sw4[(KPS + h) * SW];
        if (!S6) sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];
      }
      if (S6) {
        const uint32_t* sb = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane);
        s6n = sb[warp * 32 + lane];
        s6h = reinterpret_cast<const unsigned short*>(sb + NC * 32)[warp * 32 + lane];
      }
      const int cs = st;  // s10: arrive on empty only AFTER the stage's registers are used (see below)
      if (++st == NSTG) { st = 0; ph ^= 1; }
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        uint32_t sc[4];
        if (S6) {
          s6dec(s6n, s6h, h, sc);
        } else if (SA) {
          { const uint32_t p0 = scp<0>(sw[h]), p1 = scp<1>(sw[h]); sc[0] = blo(p0); sc[1] = bhi(p0); sc[2] = blo(p1); sc[3] = bhi(p1); }
        } else {
          sc[0] = slut[sw[h] & 0xFF]; sc[1] = slut[(sw[h] >> 8) & 0xFF]; sc[2] = slut[(sw[h] >> 16) & 0xFF];
          sc[3] = slut[sw[h] >> 24];
        }
        const int qq = q * KPS + h;
        if (PF & 256) {  // layer s81 (moe n10 probe): timing only (wrong output), never selectable: no math, keep the loads
          acc[0][0] += __uint_as_float(sc[0] ^ wv0[h].x ^ wv1[h].y ^ xf[qq][0].x);
          continue;
        }
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
          const uint4 wv = ii ? wv1[h] : wv0[h];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t af[4];
            dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
            mma16816((NA == 2 && (j & 1)) ? acc2[ii] : acc[ii], af, u4get(xf[qq][j >> 1], (j & 1) * 2),
                   u4get(xf[qq][j >> 1], (j & 1) * 2 + 1));
          }
        }
      }
      // s10 RACE FIX: arriving right after ISSUING the ring LDS (as before) let the producer's next bulk copy
      // overwrite the stage before some LDS had read it (seen as wrong weights in the first row after an x reload,
      // when 16 LDGs per thread are in flight: the s7 "first gemm2 row" race). Arrive once the data is consumed.
      mbar_arrive(&empty[cs]);
    }
    if (NA == 2) {
#pragma unroll
      for (int ii = 0; ii < 2; ++ii)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[ii][r] += acc2[ii][r];
    }
    float* red = s_red + rb * (NC * 256);
    rb ^= 1;
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[((warp * 2 + ii) * 4 + r) * 32 + lane] = acc[ii][r];
    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
    const int rp = R - seg_start;
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
      if (R + 1 == seg_end || R + 1 == r1) {  // publish this CTA's rows of the segment
        if constexpr (REL) {
          tk2f_rel_arrive(seg_start, r0);
        } else {
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(BAL ? cnt + 64 + seg_start / RPE : cnt, R + 1 - max(seg_start, r0));
        }
        }
      }
    } else if (NC == 8 || ct < 256) {
      const int ii = ct >> 7, r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        if (PF & 4) reinterpret_cast<float*>(Out)[(size_t)s_srt[off + tok] * NOUT + col] = v * gsc;  // route slot order
        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

// s10 GS16: 16-token segments. One row of a segment with n > 8 tokens dequantizes every A fragment ONCE and runs
// it against both 8-token groups (TWO): group 0's x fragments in registers (xf), group 1's in smem (xs, lane-private,
// written by the same lane at the segment change). Dequant is ~70% of a row's instructions, so a 2-group row costs
// ~1.15x a 1-group row instead of 2x. s_red holds 2 groups x 2 tiles (double-buffered).
template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false, bool SLOT = false,
          bool S6 = false>
__device__ __forceinline__ void tk2g_row(const uint32_t* slut, const uint4 (&xf)[64 / NC][2], const uint4* xs, void* Out,
                                         int rp, int off, int n, float gsc, float* s_red, uint4* smem, uint64_t* full,
                                         uint64_t* empty, int& st, int& ph, int& rb, bool gated_rt = false,
                                         const int* s_srt = nullptr) {
  // SLOT (n6, = tk2f PF & 4): gemm2 Y row = route slot s_srt[off + tok], pre-scaled by gsc (= gs2 * 256): moe_norm3
  // RTG (s12): gated is a run-time flag (gemm13 and gemm2 share ONE main-loop body: i-cache), RPE from it
  const bool GATED = RTG ? gated_rt : GATED_;
  const int RPE = RTG ? (gated_rt ? 256 : 128) : NRT / 2;
  constexpr int NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * (S6 ? 1120 : 1152), NG = TWO ? 2 : 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint4* wsm = smem + warp * 32 + lane;
  float acc[NG][2][4];
#pragma unroll
  for (int h = 0; h < NG; ++h)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[h][i][r] = 0.f;
#pragma unroll
  for (int q = 0; q < NS; ++q) {
    uint4 xb[KPS][2];
    if (TWO) {
#pragma unroll
      for (int h = 0; h < KPS; ++h) { xb[h][0] = xs[((q * KPS + h) * 2) * 32]; xb[h][1] = xs[((q * KPS + h) * 2 + 1) * 32]; }
    }
    mbar_wait(&full[st], ph);
    const uint4* sw4 = wsm + st * (SB / 16);
    uint4 wv0[KPS], wv1[KPS];
    uint32_t sw[KPS], s6n = 0, s6h = 0;
#pragma unroll
    for (int h = 0; h < KPS; ++h) {
      wv0[h] = sw4[h * SW];
      wv1[h] = sw4[(KPS + h) * SW];
      if (!S6) sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];
    }
    if (S6) {  // layer s73 (moe n20)
      const uint32_t* sb = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane);
      s6n = sb[warp * 32 + lane];
      s6h = reinterpret_cast<const unsigned short*>(sb + NC * 32)[warp * 32 + lane];
    }
    const int cs = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
#pragma unroll
    for (int h = 0; h < KPS; ++h) {
      uint32_t sc[4];
      if (S6) {
        s6dec(s6n, s6h, h, sc);
      } else if (SA) {
        { const uint32_t p0 = scp<0>(sw[h]), p1 = scp<1>(sw[h]); sc[0] = blo(p0); sc[1] = bhi(p0); sc[2] = blo(p1); sc[3] = bhi(p1); }
      } else {
        sc[0] = slut[sw[h] & 0xFF]; sc[1] = slut[(sw[h] >> 8) & 0xFF]; sc[2] = slut[(sw[h] >> 16) & 0xFF];
        sc[3] = slut[sw[h] >> 24];
      }
      const int qq = q * KPS + h;
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        const uint4 wv = ii ? wv1[h] : wv0[h];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
          mma16816(acc[0][ii], af, u4get(xf[qq][j >> 1], (j & 1) * 2), u4get(xf[qq][j >> 1], (j & 1) * 2 + 1));
          if (TWO) mma16816(acc[NG - 1][ii], af, u4get(xb[h][j >> 1], (j & 1) * 2), u4get(xb[h][j >> 1], (j & 1) * 2 + 1));
        }
      }
    }
    mbar_arrive(&empty[cs]);  // only after the stage is consumed: see tk2f_consume (s10 race)
  }
  float* red = s_red + rb * (2 * NC * 256);
  rb ^= 1;
#pragma unroll
  for (int h = 0; h < NG; ++h)
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[(((h * NC + warp) * 2 + ii) * 4 + r) * 32 + lane] = acc[h][ii][r];
  asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
  const int ct = tid, l = ct & 31, gg = l >> 2, tt = l & 3;
  if (GATED) {
    const int h = ct >> 7;  // token group (threads 128-255 only for TWO)
    if (TWO || h == 0) {
      const int r = (ct >> 5) & 3;
      float v = 0.f, u = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) {
        v += red[(((h * NC + w) * 2) * 4 + r) * 32 + l];
        u += red[(((h * NC + w) * 2 + 1) * 4 + r) * 32 + l];
      }
      const int tok = h * 8 + 2 * tt + (r & 1);
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
#pragma unroll
    for (int h = 0; h < NG; ++h) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[(((h * NC + w) * 2 + ii) * 4 + r) * 32 + l];
      const int tok = h * 8 + 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        if (SLOT) reinterpret_cast<float*>(Out)[(size_t)s_srt[off + tok] * NOUT + col] = v * gsc;
        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA>
__device__ __forceinline__ void tk2g_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int need, int r0, int r1,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint4* s_x2, uint64_t* full, uint64_t* empty,
                                             int& st, int& ph, int& rb) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  uint4* xs = s_x2 + warp * (NQ * 2 * 32) + lane;
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0;
  float gsc = 0.f;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 16;
      n = min(16, s_off[e + 1] - off);
      if (!GATED) {  // all gemm13 rows written (by any CTA): see moe_tk2f
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);

      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (h && n <= 8) break;
        const bool v = h * 8 + g < n;
        int row = 0;
        if (v) row = GATED ? (s_srt[off + h * 8 + g] >> 3) : (off + h * 8 + g);
        const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
        for (int qq = 0; qq < NQ; ++qq) {
          const int kc = qq * NC + warp;
          uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
          if (v) {
            if (GATED) { x0 = __ldg(xr + kc * 8); x1 = __ldg(xr + kc * 8 + 1); }
            else { x0 = __ldg(xr + kc * 8); x1 = __ldg(xr + kc * 8 + 1); }
          }
          if (h) { xs[(qq * 2) * 32] = x0; xs[(qq * 2 + 1) * 32] = x1; }
          else { xf[qq][0] = x0; xf[qq][1] = x1; }
        }
      }
    }
    if (n > 8) tk2g_row<NRT, NC, NSTG, GATED, KPS, SA, true>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb);
    else tk2g_row<NRT, NC, NSTG, GATED, KPS, SA, false>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb);
    if (GATED && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
      asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(cnt, R + 1 - max(seg_start, r0));
      }
    }
  }
}

// s12 MERGED consumer (tk2h): both phases in ONE run-time loop, so the kernel holds one main-loop body per row kind
// (GS 8: 1 body; GS 16: 2) instead of 2 / 4. GS16's loss was instruction-cache misses: ncu stall_no_instruction at
// layer d1 923 -> 9418 (118 KB kernel vs 65 KB) with no 2-group row at all.
template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false, bool S6 = false, bool REL = false>
__device__ __forceinline__ void tk2h_consume(const uint32_t* slut, const float* __restrict__ gs13,
                                             const float* __restrict__ gs2, const __nv_bfloat16* X,
                                             __nv_bfloat16* inter, float* Y, int* cnt, int need, int a0, int a1, int b0,
                                             int b1, const int* s_rs13, const int* s_rs2, const int* s_e,
                                             const int* s_off, const int* s_srt, float* s_red, uint4* smem, uint4* s_x2,
                                             uint64_t* full, uint64_t* empty, int& st, int& ph, int& rb) {
  constexpr int NQ = 64 / NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  uint4* xs = GS == 16 ? s_x2 + warp * (NQ * 2 * 32) + lane : nullptr;
#pragma unroll 1
  for (int p = 0; p < 2; ++p) {
    const bool gated = p == 0;
    if (!gated) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // layer s58: as TK2F_LATE_TRIG 3
    const int RPE = gated ? 256 : 128;
    const int* s_rs = gated ? s_rs13 : s_rs2;
    const float* gs = gated ? gs13 : gs2;
    const __nv_bfloat16* Xp = gated ? X : inter;
    void* Out = gated ? (void*)inter : (void*)Y;
    const int r0 = gated ? a0 : b0, r1 = gated ? a1 : b1;
    int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0;
    float gsc = 0.f;
#pragma unroll 1
    for (int R = r0; R < r1; ++R) {
      if (R >= seg_end) {
        while (s_rs[a + 1] <= R) ++a;
        const int gi = (R - s_rs[a]) / RPE;
        seg_start = s_rs[a] + gi * RPE;
        seg_end = seg_start + RPE;
        const int e = s_e[a];
        gsc = gs[e] * 256.0f;
        off = s_off[e] + gi * GS;
        n = min(GS, s_off[e + 1] - off);
        if (!gated) {  // all gemm13 rows written (by any CTA): see moe_tk2f. PSC (s12): only this segment's 256 rows
          // (segment j = gemm13 rows [256 j, 256 j + 256) = gemm2 rows [128 j, 128 j + 128); counter cnt[64 + j], j < 64)
          const int* cp = PSC ? cnt + 64 + seg_start / 128 : cnt;
          const int nd = PSC ? 256 : need;
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cp) : "memory");
          } while (c < nd);
        }
#pragma unroll
        for (int h = 0; h < GS / 8; ++h) {
          if (h && n <= 8) break;
          const bool v = h * 8 + g < n;
          int row = 0;
          if (v) row = gated ? (s_srt[off + h * 8 + g] >> 3) : (off + h * 8 + g);
          const uint4* xr = reinterpret_cast<const uint4*>(Xp + (size_t)row * KD + 16 * t);
#pragma unroll
          for (int qq = 0; qq < NQ; ++qq) {
            const int kc = qq * NC + warp;
            uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
            if (v) { x0 = __ldcg(xr + kc * 8); x1 = __ldcg(xr + kc * 8 + 1); }  // layer s58: __ldcg (PDL / in-kernel producer)
            if (h) { xs[(qq * 2) * 32] = x0; xs[(qq * 2 + 1) * 32] = x1; }
            else { xf[qq][0] = x0; xf[qq][1] = x1; }
          }
        }
      }
      if (GS == 16 && n > 8)
        tk2g_row<0, NC, NSTG, false, KPS, SA, true, true, SLOT, S6>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated, s_srt);
      else
        tk2g_row<0, NC, NSTG, false, KPS, SA, false, true, SLOT, S6>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated, s_srt);
      if (gated && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
        if constexpr (REL) {
          tk2f_rel_arrive(seg_start, r0);
        } else {
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(PSC ? cnt + 64 + seg_start / 256 : cnt, R + 1 - max(seg_start, r0));
        }
        }
      }
    }
  }
}

// DYN variant (s8): CTA b first takes the static rows [b*S, (b+1)*S) of a phase (S = rows/grid - D), then single
// rows from the tail [grid*S, N) via a global counter (ctr, pulled by the producer one row ahead). The producer
// tells the consumers each row's index through s_rid (written before the row's first full-barrier arrive, which
// releases it); -1 = end of the phase (a stage with no data: plain arrive). Rows of a CTA stay increasing.
template <int NRT, int NC, int NSTG, int KPS, bool HARD>
__device__ __forceinline__ void tk2d_produce(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, int S,
                                             int N, int* ctr, const int* s_rs, const int* s_e, int* s_rid, uint64_t* ridb,
                                             uint4* smem, uint64_t* full, uint64_t* empty, int& st, int& ph, int& i,
                                             int& k) {
  constexpr int RPE = NRT / 2, NQ = 64 / (NC * KPS), SB = NC * KPS * 1152, KB = NC * KPS;
  const int G0 = gridDim.x * S;
  int a = 0, R = blockIdx.x * S;
  const int sEnd = R + S;
  int nxt = S == 0 ? atomicAdd(ctr, 1) : 0;  // next tail row, fetched one row early
  while (true) {
    int Rn;
    if (R < sEnd) {
      Rn = R++;
      if (R == sEnd) nxt = atomicAdd(ctr, 1);
    } else {
      Rn = G0 + nxt;
      if (Rn >= N) Rn = -1;
      else nxt = atomicAdd(ctr, 1);
    }
    if (Rn >= 0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
    s_rid[k & 3] = Rn;  // row id (-1: end of the phase) on its own arrive-only barrier: no wait for data
    mbar_arrive(&ridb[k & 3]);
    ++k;
    if (Rn < 0) return;
    while (s_rs[a + 1] <= Rn) ++a;
    const int e = s_e[a];
    const int lr = (Rn - s_rs[a]) & (RPE - 1);
    const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
    const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
    for (int q = 0; q < NQ; ++q, ++i) {
      if (q && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      char* sbase = reinterpret_cast<char*>(smem) + st * SB;
      mbar_expect_tx(&full[st], SB);
      bulk_g2s(sbase, src + q * KB * 32, KB * 512, &full[st]);
      bulk_g2s(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st]);
      bulk_g2s(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS>
__device__ __forceinline__ void tk2d_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int need, int S, const int* s_rid, uint64_t* ridb,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint64_t* full, uint64_t* empty, int& st,
                                             int& ph, int& rb, int& k) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * 1152;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, nd = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  const int R0 = blockIdx.x * S;
  for (int j = 0;; ++j, ++k) {
    int R = R0 + j;
    if (j >= S) {  // tail row: its index comes from the producer
      mbar_wait(&ridb[k & 3], (k >> 2) & 1);
      R = s_rid[k & 3];
      if (R < 0) { ++k; break; }
    }
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      if (!GATED && nd == 0) {  // all gemm13 rows written (by any CTA), once per CTA
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);
      }
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
    ++nd;
    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      uint4 wv0[KPS], wv1[KPS];
      uint32_t sw[KPS];
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        wv0[h] = sw4[h * SW];
        wv1[h] = sw4[(KPS + h) * SW];
        sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];
      }
      mbar_arrive(&empty[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        const uint32_t sc[4] = {slut[sw[h] & 0xFF], slut[(sw[h] >> 8) & 0xFF], slut[(sw[h] >> 16) & 0xFF], slut[sw[h] >> 24]};
        const int qq = q * KPS + h;
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
          const uint4 wv = ii ? wv1[h] : wv0[h];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t af[4];
            dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
            mma16816(acc[ii], af, u4get(xf[qq][j >> 1], (j & 1) * 2), u4get(xf[qq][j >> 1], (j & 1) * 2 + 1));
          }
        }
      }
    }
    float* red = s_red + rb * (NC * 256);
    rb ^= 1;
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[((warp * 2 + ii) * 4 + r) * 32 + lane] = acc[ii][r];
    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
    const int rp = R - seg_start;
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
  if (GATED) {  // publish this CTA's gemm13 rows
    asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
    if (tid == 0 && nd) {
      __threadfence();
      atomicAdd(cnt, nd);
    }
  }
}

// s9 fused post-expert norm (FN): every CTA takes a ticket (cnt[4], monotonic over calls: generation g = cnt[7], read
// at kernel start, before any ticket of the call) after its gemm2 rows; the last M take one token each, wait until all
// G CTAs have ticked, then do moe_norm2's work for it in one pass (y rows from L2, all loads in flight at once) and
// write out. The last ticket zeroes cnt[0] (every CTA is past its gemm2 wait) and bumps cnt[7]. Replaces moe_norm2
// and its launch (moe_norm2, used by the other paths, zeroes all counters: consistent with this too).
// s_rt[i] / s_rg[i]: route i's topk weight and global scale * 256 (0 when not local), s_inv[i]: its y row (-1).
template <int NC>
__device__ __forceinline__ void tk2f_norm(int* cnt, int M, const int* s_inv, const float* s_rt, const float* s_rg,
                                          const float* __restrict__ w1, const float* Y, float eps,
                                          __nv_bfloat16* __restrict__ out, float* s_ss, int* s_tk, int gen) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, G = gridDim.x;
  float wv[16];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float4 a = __ldg(wp + h * 512 + tid * 2), b = __ldg(wp + h * 512 + tid * 2 + 1);
      wv[h * 8 + 0] = a.x; wv[h * 8 + 1] = a.y; wv[h * 8 + 2] = a.z; wv[h * 8 + 3] = a.w;
      wv[h * 8 + 4] = b.x; wv[h * 8 + 5] = b.y; wv[h * 8 + 6] = b.z; wv[h * 8 + 7] = b.w;
    }
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  if (tid == 0) {
    __threadfence();
    const int t = atomicAdd(cnt + 4, 1) - gen * G;
    if (t == G - 1) {  // last ticket: all CTAs are done with cnt[0]
      cnt[0] = 0;
      __threadfence();
      atomicExch(cnt + 7, gen + 1);
    }
    s_tk[0] = t;
    s_tk[1] = gen;
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  const int tok = s_tk[0] - (G - M);
  if (tok < 0) return;
  gen = s_tk[1];
  int pos[8];
  float gsc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    pos[s] = max(s_inv[tok * 8 + s], 0);  // not local: row 0 (valid memory), scaled by 0
    gsc[s] = s_rg[tok * 8 + s];
  }
  {
    const int need = (gen + 1) * G;
    int c;
    do {
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt + 4) : "memory");
    } while (c < need);
  }
  float v[8][16], ss[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const float4* p = reinterpret_cast<const float4*>(Y + (size_t)pos[s] * NOUT);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float4 a = __ldg(p + h * 512 + tid * 2), b = __ldg(p + h * 512 + tid * 2 + 1);
      v[s][h * 8 + 0] = a.x; v[s][h * 8 + 1] = a.y; v[s][h * 8 + 2] = a.z; v[s][h * 8 + 3] = a.w;
      v[s][h * 8 + 4] = b.x; v[s][h * 8 + 5] = b.y; v[s][h * 8 + 6] = b.z; v[s][h * 8 + 7] = b.w;
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    ss[s] = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) { v[s][k] *= gsc[s] * wv[k]; ss[s] += v[s][k] * v[s][k]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[s] += __shfl_xor_sync(0xffffffffu, ss[s], o);
    if (lane == 0) s_ss[warp * 8 + s] = ss[s];
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  float acc[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) acc[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < NC; ++w) tot += s_ss[w * 8 + s];
    const float fs = gsc[s] != 0.f ? s_rt[tok * 8 + s] * rsqrtf(tot / NOUT + eps) : 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] += fs * v[s][k];
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint4 o;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[h * 8 + 2 * k], acc[h * 8 + 2 * k + 1]);
    reinterpret_cast<uint4*>(out + (size_t)tok * NOUT)[h * 256 + tid] = o;
  }
}

// GS16 weighted static split: first row of CTA b when rows of 2-group segments (n > 8) weigh wt/8 and others 1.
// Monotone in b, and b = gridDim.x gives the total row count, so CTA ranges tile the rows exactly.
__device__ __forceinline__ int tk2g_split(int b, int nact, int wt, int rps, const int* s_e, const int* s_off) {
  long long W = 0;
  for (int a = 0; a < nact; ++a) {
    const int c = s_off[s_e[a] + 1] - s_off[s_e[a]];
    for (int o = 0; o < c; o += 16) W += (long long)rps * (c - o > 8 ? wt : 8);
  }
  const long long p = W * b / gridDim.x;
  long long cum = 0;
  int row = 0;
  for (int a = 0; a < nact; ++a) {
    const int c = s_off[s_e[a] + 1] - s_off[s_e[a]];
    for (int o = 0; o < c; o += 16) {
      const int w = c - o > 8 ? wt : 8;
      if (p < cum + (long long)rps * w) return row + (int)((p - cum + w - 1) / w);
      cum += (long long)rps * w;
      row += rps;
    }
  }
  return row;
}

// n4s5: tk2g_split by warp 0 in 32-bit lane-parallel math (one lane per active expert; same bounds as tk2g_split).
// tk2g_split ran 64-bit loops in every thread: consumers started 1.8 (median) / 4.1 (max) us late at d16 (G88).
__device__ __forceinline__ int tk2g_bound(int b, int nact, int wt, int rps, const int* s_e, const int* s_off) {
  const int lane = threadIdx.x & 31;
  const int c = lane < nact ? s_off[s_e[lane] + 1] - s_off[s_e[lane]] : 0;
  int w = 0;  // this expert's weight / rps
  for (int o = 0; o < c; o += 16) w += c - o > 8 ? wt : 8;
  const int ng = (c + 15) >> 4;
  int incl = w * rps, rows = ng * rps;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o), r = __shfl_up_sync(0xffffffffu, rows, o);
    if (lane >= o) { incl += v; rows += r; }
  }
  const int W = __shfl_sync(0xffffffffu, incl, 31);
#if TK2G_PAR4
  const int p = (int)((unsigned)W * (unsigned)b / gridDim.x);  // n17: 32-bit (W * b < 2^28)
#else
  const int p = (int)((long long)W * b / gridDim.x);
#endif
  int cum = incl - w * rps, row = rows - ng * rps;
  int res = 0x7fffffff;
  if (ng && p >= cum && p < incl) {
    for (int o = 0; o < c; o += 16) {
      const int sw = c - o > 8 ? wt : 8;
      if (p < cum + rps * sw) { res = row + (p - cum + sw - 1) / sw; break; }
      cum += rps * sw;
      row += rps;
    }
  }
  if (p >= W) res = __shfl_sync(0xffffffffu, rows, 31);
  return __reduce_min_sync(0xffffffffu, res);
}

// s12 WEIGHTED static split (GS 8): a row of an expert's first 8-token group streams its weights from HBM, rows of the
// extra groups (hot experts) mostly hit L2 and cost ~wE/8 of it (H200 fit on layer routings: 4.4 / 6.97 us = 0.63).
// Warp 0 computes CTA b's row bounds [r(b), r(b+1)) of a phase with RPS rows per group; lane = active expert index.
// 32-bit math: total weight <= 64 groups x 256 rows x 8.
__device__ __forceinline__ int tk2w_bound(int b, int nact, int wE, int rps, const int* s_e, const int* s_off) {
  const int lane = threadIdx.x & 31;
  const int c = lane < nact ? s_off[s_e[lane] + 1] - s_off[s_e[lane]] : 0;
  const int ng = (c + 7) >> 3;
  const int w = ng ? rps * (8 + (ng - 1) * wE) : 0;  // this expert's weight
  int incl = w, rows = ng * rps;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o), r = __shfl_up_sync(0xffffffffu, rows, o);
    if (lane >= o) { incl += v; rows += r; }
  }
  const int W = __shfl_sync(0xffffffffu, incl, 31);
  const int p = (int)((long long)W * b / gridDim.x);
  const int cum = incl - w, row0 = rows - ng * rps;
  int res = 0x7fffffff;
  if (ng && p >= cum && p < incl) {  // p falls in this expert: group 0 (weight 8 per row), then groups of weight wE
    const int q = p - cum;
    if (q < rps * 8) res = row0 + (q + 7) / 8;
    else res = row0 + rps + (q - rps * 8 + wE - 1) / wE;
  }
  if (p >= W) res = __shfl_sync(0xffffffffu, rows, 31);
  res = __reduce_min_sync(0xffffffffu, res);
  return res;
}

template <int NC, int NSTG, int KPS, bool DYN, bool FN = false, bool SA = false, int GS = 8, int MRG = 0, int PF = 0>
__global__ void __launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(
    const uint4* __restrict__ Wq13, const uint32_t* __restrict__ S13, const uint4* __restrict__ Wq2,
    const uint32_t* __restrict__ S2, const uint32_t* __restrict__ lut, const float* __restrict__ gs13,
    const float* __restrict__ gs2, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids, int nroutes,
    int start, int count, int* __restrict__ meta, int* __restrict__ sroutes, int* __restrict__ inv,
    __nv_bfloat16* inter, float* __restrict__ Y, int* cnt, int D, const float* __restrict__ tw,
    const float* __restrict__ w1, __nv_bfloat16* __restrict__ out, int M, float eps) {
  static_assert(NC % 8 == 0, "routing uses warps 0..7 x 32 ids");
  extern __shared__ __align__(128) uint4 smem[];
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_rs13[33], s_rs2[33], s_srt[256], s_wc[8][32], s_nact;
  __shared__ float s_red[(GS == 16 || MRG ? 4 : 2) * NC * 256];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  __shared__ int s_rid[4];
  __shared__ __align__(8) uint64_t ridb[4];
  __shared__ int s_inv[FN ? 256 : 1], s_tk[2];
  __shared__ float s_rt[FN ? 256 : 1], s_rg[FN ? 256 : 1];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int gen = 0;  // FN: this call's generation (see tk2f_norm); thread 0 only
  if (FN && tid == 0) asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(gen) : "l"(cnt + 7) : "memory");
  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NC * 32); }
    for (int i = 0; i < 4; ++i) mbar_init(&ridb[i], 1);
    if constexpr (tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>())
      for (int i = 0; i < 8; ++i) mbar_init(&g_tkrel[i], NC * 32);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  {  // routing, as moe_tk2 GATED
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
    if (FN && warp < 8) s_inv[warp * 32 + lane] = e >= 0 ? s_wc[warp][e] + rank : -1;
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
  }
  __syncthreads();
  const int nact = s_nact;
  if (tid < 32) {
    const int e = s_e[tid];
    const int ng = tid < nact ? (s_off[e + 1] - s_off[e] + GS - 1) / GS : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_rs13[tid + 1] = incl * 256;
    s_rs2[tid + 1] = incl * 128;
    if (tid == 0) { s_rs13[0] = 0; s_rs2[0] = 0; }
  }
  __syncthreads();
  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if constexpr (DYN) {
    const int G = gridDim.x, ns13 = max(0, n13 / G - D), ns2 = max(0, n2 / G - D);
    int st = 0, ph = 0, k = 0;
    if (warp == NC) {
      if (lane == 0) {
        int i = 0;
        tk2d_produce<512, NC, NSTG, KPS, false>(Wq13, S13, ns13, n13, cnt + 1, s_rs13, s_e, s_rid, ridb, smem, full, empty, st, ph, i, k);
        tk2d_produce<256, NC, NSTG, KPS, true>(Wq2, S2, ns2, n2, cnt + 2, s_rs2, s_e, s_rid, ridb, smem, full, empty, st, ph, i, k);
      }
      return;
    }
    int rb = 0;
    tk2d_consume<512, NC, NSTG, true, KPS>(slut, gs13, X, inter, cnt, n13, ns13, s_rid, ridb, s_rs13, s_e, s_off, s_srt, s_red, smem,
                                          full, empty, st, ph, rb, k);
    tk2d_consume<256, NC, NSTG, false, KPS>(slut, gs2, inter, Y, cnt, n13, ns2, s_rid, ridb, s_rs2, s_e, s_off, s_srt, s_red, smem,
                                           full, empty, st, ph, rb, k);
    return;
  }
  int a0 = (int)((long long)n13 * blockIdx.x / gridDim.x), a1 = (int)((long long)n13 * (blockIdx.x + 1) / gridDim.x);
  int b0 = (int)((long long)n2 * blockIdx.x / gridDim.x), b1 = (int)((long long)n2 * (blockIdx.x + 1) / gridDim.x);
  if (GS == 8 && !DYN && D > 0) {  // s12 weighted split (wE = D / 8 for extra-group rows), see tk2w_bound
    __shared__ int s_ab[4];
    if (warp == 0) {
      const int v0 = tk2w_bound(blockIdx.x, nact, D, 256, s_e, s_off), v1 = tk2w_bound(blockIdx.x + 1, nact, D, 256, s_e, s_off);
      const int v2 = tk2w_bound(blockIdx.x, nact, D, 128, s_e, s_off), v3 = tk2w_bound(blockIdx.x + 1, nact, D, 128, s_e, s_off);
      if (lane == 0) { s_ab[0] = v0; s_ab[1] = v1; s_ab[2] = v2; s_ab[3] = v3; }
    }
    __syncthreads();
    a0 = s_ab[0]; a1 = s_ab[1]; b0 = s_ab[2]; b1 = s_ab[3];
  }
  if (GS == 16 && D > 8) {  // GS16: 2-group rows cost D/8 of a 1-group row: split the rows by weight, not count
    __shared__ int s_gb[4];
    if (TK2G_PAR) {  // n4s5: warp 0, lane-parallel (tk2g_bound)
#if TK2G_PAR4
      if (warp < 4) {  // n17: the 4 bounds on 4 warps
        const int v = tk2g_bound(blockIdx.x + (warp & 1), nact, D, (warp & 2) ? 128 : 256, s_e, s_off);
        if (lane == 0) s_gb[warp] = v;
      }
#else
      if (warp == 0) {
        const int v0 = tk2g_bound(blockIdx.x, nact, D, 256, s_e, s_off), v1 = tk2g_bound(blockIdx.x + 1, nact, D, 256, s_e, s_off);
        const int v2 = tk2g_bound(blockIdx.x, nact, D, 128, s_e, s_off), v3 = tk2g_bound(blockIdx.x + 1, nact, D, 128, s_e, s_off);
        if (lane == 0) { s_gb[0] = v0; s_gb[1] = v1; s_gb[2] = v2; s_gb[3] = v3; }
      }
#endif
      __syncthreads();
      a0 = s_gb[0]; a1 = s_gb[1]; b0 = s_gb[2]; b1 = s_gb[3];
    } else {
      a0 = tk2g_split(blockIdx.x, nact, D, 256, s_e, s_off);
      a1 = tk2g_split(blockIdx.x + 1, nact, D, 256, s_e, s_off);
      b0 = tk2g_split(blockIdx.x, nact, D, 128, s_e, s_off);
      b1 = tk2g_split(blockIdx.x + 1, nact, D, 128, s_e, s_off);
    }
  }
  if (PF & 32) {  // moe n8 BAL: gemm13's extra rows (n13 % G) on the LAST CTAs, gemm2's (n2 % G) on the FIRST ones
    const int G = gridDim.x, b = blockIdx.x, q1 = n13 / G, e1 = G - n13 % G, q2 = n2 / G, e2 = n2 % G;
    a0 = b * q1 + max(0, b - e1); a1 = a0 + q1 + (b >= e1);
    b0 = b * q2 + min(b, e2); b1 = b0 + q2 + (b < e2);
  }
  if constexpr ((PF & 4096) != 0) { a1 = a0; b1 = b0; }  // n5s25 orz: no rows (routing, launch, barriers' setup kept)
  constexpr bool REL = tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>();
  if constexpr (REL) {
    if (warp == NC + 1) {  // n6s6 TK2F_REL helper
      if (lane == 0) tk2f_rel_helper<MRG ? MRG == 2 : (PF & 32) != 0>(cnt, a0, a1);
      return;
    }
  }
  int st = 0, ph = 0;
  if (warp == NC) {
    if (lane == 0) {
      int i = 0;
      constexpr bool SQ = (PF & 512) != 0;  // layer s68 (moe n14): single-copy stages (Wq13 / Wq2 = the _sq_pack buffers)
      constexpr bool S6 = (PF & 1024) != 0;  // layer s73 (moe n20): 6-bit scale codes (the _s6_pack buffers)
      tk2f_produce<512, NC, NSTG, KPS, SQ, S6, (PF & 2048) != 0>(Wq13, S13, a0, a1, s_rs13, s_e, smem, full, empty, st, ph, i);
      tk2f_produce<256, NC, NSTG, KPS, SQ, S6, (PF & 2048) != 0>(Wq2, S2, b0, b1, s_rs2, s_e, smem, full, empty, st, ph, i);
    }
    return;
  }
  float rt = 0.f, rg = 0.f;  // FN: route tid's topk weight and gs2 * 256 (0: not local), used by tk2f_norm
  if (FN && tid < nroutes) {
    const int e = ids[tid] - start;
    if (e >= 0 && e < count) {
      rt = tw[tid];
      rg = gs2[e] * 256.0f;
    }
  }
  int rb = 0;
  if constexpr (MRG) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // layer s58: x is the PDL predecessor's output (TK2FX/TK2FJ)
    tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0, (PF & 1024) != 0, REL>(slut, gs13, gs2, X, inter, Y, cnt, n13, a0, a1, b0, b1, s_rs13, s_rs2, s_e, s_off,
                                        s_srt, s_red, smem, smem + NSTG * NC * KPS * 72, full, empty, st, ph, rb);
    return;
  }
  if constexpr (GS == 16) {
    uint4* s_x2 = smem + NSTG * NC * KPS * 72;  // after the ring: 8 KB per consumer warp
    tk2g_consume<512, NC, NSTG, true, KPS, SA>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem,
                                               s_x2, full, empty, st, ph, rb);
    tk2g_consume<256, NC, NSTG, false, KPS, SA>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem,
                                                s_x2, full, empty, st, ph, rb);
    return;
  }
  tk2f_consume<512, NC, NSTG, true, KPS, SA, PF, REL>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem, full,
                                    empty, st, ph, rb);
  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem, full,
                                     empty, st, ph, rb);
  if constexpr (FN) {
    s_rt[tid] = rt;
    s_rg[tid] = rg;
    tk2f_norm<NC>(cnt, M, s_inv, s_rt, s_rg, w1, Y, eps, out, s_red, s_tk, gen);
  }
}

template <int NC, int NSTG, int KPS, bool DYN = false, bool FN = false, bool SA = false, int GS = 8, int MRG = 0, int PF = 0>
static int64_t tk2f_go(int D, int nr, const void* w13, const void* s13, const void* w2, const void* s2, const void* lut,
                       const void* gs13, const void* gs2, const void* x, const void* ids, int start, int count,
                       void* meta, void* sr, void* inv, void* inter, void* y, void* cnt, const void* tw = nullptr,
                       const void* w1 = nullptr, void* out = nullptr, float eps = 0.f) {
  constexpr int SMEM = NSTG * NC * KPS * 1152 + (GS == 16 ? NC * 8 * 2 * 32 * 16 : 0);
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
#if TK2F_ORX
  if constexpr ((PF & (256 | 2048 | 4096)) == 0) {  // n5s25 DIAG: 1 orz, 2 orn, 3 ornc (= orn + PF 256 no math)
    if (g_orx == 1) return tk2f_go<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF | 4096>(D, nr, w13, s13, w2, s2, lut, gs13, gs2, x, ids, start, count, meta, sr, inv, inter, y, cnt, tw, w1, out, eps);
    if (g_orx == 2) return tk2f_go<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF | 2048>(D, nr, w13, s13, w2, s2, lut, gs13, gs2, x, ids, start, count, meta, sr, inv, inter, y, cnt, tw, w1, out, eps);
    if (g_orx == 3) return tk2f_go<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF | 2048 | 256>(D, nr, w13, s13, w2, s2, lut, gs13, gs2, x, ids, start, count, meta, sr, inv, inter, y, cnt, tw, w1, out, eps);
  }
#endif
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  return launch_k(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, grid, (NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, SMEM, st, false,
      (const uint4*)w13, (const uint32_t*)s13, (const uint4*)w2, (const uint32_t*)s2, (const uint32_t*)lut,
      (const float*)gs13, (const float*)gs2, (const __nv_bfloat16*)x, (const int*)ids, nr, start, count, (int*)meta,
      (int*)sr, (int*)inv, (__nv_bfloat16*)inter, (float*)y, (int*)cnt, D, (const float*)tw, (const float*)w1,
      (__nv_bfloat16*)out, nr / 8, eps);
}

// s9: moe_tk2f with the post-expert norm fused (FN): writes out [M, H] bf16 directly; no moe_norm2 after it.
int64_t moe_tk2fn_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor w2_, torch::Tensor s2_,
                         torch::Tensor lut_, torch::Tensor gs13_, torch::Tensor gs2_, torch::Tensor x_,
                         torch::Tensor ids_, int64_t start, int64_t count, torch::Tensor meta_, torch::Tensor sr_,
                         torch::Tensor inv_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor cnt_, int64_t cfg,
                         torch::Tensor tw_, torch::Tensor w1_, torch::Tensor out_, double eps) {
  const int nr = (int)ids_.numel();
  if (nr > 256) return -2;
#define TK2FN_ARGS 0, nr, PTR(w13_), PTR(s13_), PTR(w2_), PTR(s2_), PTR(lut_), PTR(gs13_), PTR(gs2_), PTR(x_), \
    PTR(ids_), (int)start, (int)count, meta_.data_ptr(), sr_.data_ptr(), inv_.data_ptr(), inter_.data_ptr(), \
    y_.data_ptr(), cnt_.data_ptr(), PTR(tw_), PTR(w1_), out_.data_ptr(), (float)eps
  if (cfg == 402) return tk2f_go<8, 2, 4, false, true>(TK2FN_ARGS);
  return tk2f_go<8, 4, 2, false, true>(TK2FN_ARGS);
#undef TK2FN_ARGS
}


// cfg = KPS * 100 + NSTG: 108 (s7 default), 204 (s8), 208, 402, 403, 206; + 1000 * (D + 1): DYN tail (204/402 only)
int64_t moe_tk2f_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor w2_, torch::Tensor s2_,
                        torch::Tensor lut_, torch::Tensor gs13_, torch::Tensor gs2_, torch::Tensor x_,
                        torch::Tensor ids_, int64_t start, int64_t count, torch::Tensor meta_, torch::Tensor sr_,
                        torch::Tensor inv_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor cnt_, int64_t cfg) {
#if TK2F_ORX
  g_orx = (int)(cfg / 100000000);
  cfg %= 100000000;
#endif
  const int nr = (int)ids_.numel();
  if (nr > 256) return -2;
#define TK2F_ARGS0 nr, PTR(w13_), PTR(s13_), PTR(w2_), PTR(s2_), PTR(lut_), PTR(gs13_), PTR(gs2_), PTR(x_), PTR(ids_), \
    (int)start, (int)count, meta_.data_ptr(), sr_.data_ptr(), inv_.data_ptr(), inter_.data_ptr(), y_.data_ptr(), \
    cnt_.data_ptr()
  const bool s6 = cfg >= 20000000;  // layer s73 (moe n20 S6): + 20000000 = SQ with 6-bit scale codes (_s6_pack buffers)
  const bool sq = cfg >= 10000000;  // layer s68 (moe n14 SQ): + 10000000 = single-copy stages (w13_ / w2_ = _sq_pack buffers)
  cfg %= 10000000;
  const bool bal = cfg >= 1000000;  // moe n8: + 1000000 = BAL (PF | 32; SA pf 4 / 5 only)
  cfg %= 1000000;
  if (s6) {  // SA 84204 (+ BAL), tk2h GS16 + PF4
    if (cfg >= 140000 && cfg < 150000 && cfg % 1000 == 4)
      return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4 | 512 | 1024>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
    TORCH_CHECK(cfg == 184204, "moe_tk2f S6: cfg ", cfg);
    if (bal) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36 | 512 | 1024>(0, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 512 | 1024>(0, TK2F_ARGS0);
  }
  if (sq) {  // SA 84204 only (+ BAL): d1 slot, T = 16 / 32
    if (cfg >= 140000 && cfg < 150000 && cfg % 1000 == 4)  // moe n16: tk2h GS16 + PF4 on SQ stages
      return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4 | 512>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
#if TK2F_DIAG
    // layer s81 DIAG (tuner timing only, never selectable): 184402/184403 = KPS 4 SQ stages (36 KB, _sq_pack nq 2 buffers),
    // NSTG 2/3; 184214 / 184412 / 184413 = the same (and 204) with no consumer math (PF 256: wrong output)
    if (cfg == 184402) return tk2f_go<8, 2, 4, false, false, true, 8, 0, 4 | 512>(0, TK2F_ARGS0);
    if (cfg == 184403) return tk2f_go<8, 3, 4, false, false, true, 8, 0, 4 | 512>(0, TK2F_ARGS0);
    if (cfg == 184214) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 512 | 256>(0, TK2F_ARGS0);
    if (cfg == 184412) return tk2f_go<8, 2, 4, false, false, true, 8, 0, 4 | 512 | 256>(0, TK2F_ARGS0);
    if (cfg == 184413) return tk2f_go<8, 3, 4, false, false, true, 8, 0, 4 | 512 | 256>(0, TK2F_ARGS0);
#endif
    TORCH_CHECK(cfg == 184204, "moe_tk2f SQ: cfg ", cfg);
    if (bal) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36 | 512>(0, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 512>(0, TK2F_ARGS0);
  }
  const bool sa = cfg >= 100000;  // + 100000: SA (arithmetic scale decode; prepare() checked the scales)
  cfg %= 100000;
  if (cfg > 80000 && cfg < 100000 && cfg % 1000 == 204) {  // s13 PF = cfg / 1000 % 10: 1 = L1 prefetch of the next segment's
    // x one row ahead, 2 = gemm2's counter spin only at the CTA's first gemm2 row, 4 = y32 in route slot order, pre-scaled
    // by gs2 * 256 (then moe_norm3 instead of moe_norm2), 8 = 2 accumulator chains per tile; built: 1, 2, 3, 4, 7, 12
    const int pf = (int)(cfg - 80000) / 1000;
    if (bal && sa && pf == 4) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36>(0, TK2F_ARGS0);
#ifdef MOE_BAL_NU8
    if (bal && sa && pf == 5) {
      TORCH_CHECK(nr == 64 && y_.size(0) >= 65 && g_ssw1 != nullptr, "tk2f PF 52: shape");
      return tk2f_go<8, 4, 2, false, false, true, 8, 0, 52>(0, TK2F_ARGS0, nullptr, g_ssw1);
    }
#endif
    if (pf == 12) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 12>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 12>(0, TK2F_ARGS0);
    if (pf == 4) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 4>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 4>(0, TK2F_ARGS0);
    if (pf == 7) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 7>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 7>(0, TK2F_ARGS0);
    if (pf == 1) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 1>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 1>(0, TK2F_ARGS0);
    if (pf == 2) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 2>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 2>(0, TK2F_ARGS0);
    return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 3>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 3>(0, TK2F_ARGS0);
  }
  if (sa && cfg >= 40000 && cfg < 50000 && cfg % 1000 == 4)  // layer s58: moe n6 tk2h GS16 + PF4 (h16w*p4), the only tk2h built
    return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
#if TK2_EXP
  if (cfg >= 30000) {  // s12 merged-phase consumer (tk2h): + 30000 GS 8, + 40000 GS 16 (cfg / 1000 % 10 = 2-group weight),
    // + 50000 / + 60000: the same with per-segment gemm13 -> gemm2 counters (PSC)
    const int kind = (int)(cfg / 10000);
    const int wt = 8 + (int)(cfg / 1000) % 10;
    if (cfg % 1000 == 4) {  // n6: + 4 = slot-ordered pre-scaled y32 (tk2f PF 4) -> moe_norm3
      if (kind == 3) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 1, 4>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 1, 4>(0, TK2F_ARGS0);
      return sa ? tk2f_go<8, 4, 2, false, false, true, 16, 1, 4>(wt, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 16, 1, 4>(wt, TK2F_ARGS0);
    }
    if (kind == 3) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 1>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 1>(0, TK2F_ARGS0);
    if (kind == 4) return sa ? tk2f_go<8, 4, 2, false, false, true, 16, 1>(wt, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 16, 1>(wt, TK2F_ARGS0);
    if (kind == 7) {  // + 70000: main consumer (tk2f_consume), weighted static split, wE = cfg / 1000 % 10
      const int wE = (int)(cfg / 1000) % 10;
      return sa ? tk2f_go<8, 4, 2, false, false, true>(wE, TK2F_ARGS0) : tk2f_go<8, 4, 2>(wE, TK2F_ARGS0);
    }
    if (kind == 5) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 2>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 2>(0, TK2F_ARGS0);
    return sa ? tk2f_go<8, 4, 2, false, false, true, 16, 2>(wt, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 16, 2>(wt, TK2F_ARGS0);
  }
#endif
  if (cfg >= 20000) {  // + 20000: 16 consumer warps (NC 16), cfg % 1000 = KPS * 100 + NSTG (static rows)
    const int c = (int)(cfg % 1000);
    if (sa) {
      if (c == 202) return tk2f_go<16, 2, 2, false, false, true>(0, TK2F_ARGS0);
      if (c == 104) return tk2f_go<16, 4, 1, false, false, true>(0, TK2F_ARGS0);
      if (c == 402) return tk2f_go<16, 2, 4, false, false, true>(0, TK2F_ARGS0);
      if (c == 103) return tk2f_go<16, 3, 1, false, false, true>(0, TK2F_ARGS0);
      return tk2f_go<16, 4, 2, false, false, true>(0, TK2F_ARGS0);
    }
    return tk2f_go<16, 4, 2>(0, TK2F_ARGS0);
  }
  const bool g16 = cfg >= 10000;  // + 10000: GS16 (16-token segments; static rows, KPS 2, NSTG 4)
  cfg %= 10000;
  if (g16) {  // (cfg / 1000) % 10 = extra weight of a 2-group row in 1/8 (see tk2g_split)
    const int wt = 8 + (int)(cfg / 1000) % 10;
    if (sa) return tk2f_go<8, 4, 2, false, false, true, 16>(wt, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, false, 16>(wt, TK2F_ARGS0);
  }
  const int D = (int)(cfg / 1000) - 1;  // cfg >= 1000: DYN with D = cfg / 1000 - 1
  cfg %= 1000;
  if (D >= 0) {
    if (cfg == 204) return tk2f_go<8, 4, 2, true>(D, TK2F_ARGS0);
    return tk2f_go<8, 2, 4, true>(D, TK2F_ARGS0);
  }
#define TK2F_ARGS 0, nr, PTR(w13_), PTR(s13_), PTR(w2_), PTR(s2_), PTR(lut_), PTR(gs13_), PTR(gs2_), PTR(x_), PTR(ids_), \
    (int)start, (int)count, meta_.data_ptr(), sr_.data_ptr(), inv_.data_ptr(), inter_.data_ptr(), y_.data_ptr(), \
    cnt_.data_ptr()
  if (sa && D < 0) {
    if (cfg == 402) return tk2f_go<8, 2, 4, false, false, true>(TK2F_ARGS);
    if (TK2F_RING_SA) {  // s12: deeper rings with SA, for the H200 diag (MOE_TK2F_RING_SA build flag)
      if (cfg == 208) return tk2f_go<8, 8, 2, false, false, true>(TK2F_ARGS);
      if (cfg == 206) return tk2f_go<8, 6, 2, false, false, true>(TK2F_ARGS);
      if (cfg == 403) return tk2f_go<8, 3, 4, false, false, true>(TK2F_ARGS);
      if (cfg == 108) return tk2f_go<8, 8, 1, false, false, true>(TK2F_ARGS);
    }
    return tk2f_go<8, 4, 2, false, false, true>(TK2F_ARGS);
  }
  if (cfg == 204) return tk2f_go<8, 4, 2>(TK2F_ARGS);
  if (cfg == 208) return tk2f_go<8, 8, 2>(TK2F_ARGS);
  if (cfg == 402) return tk2f_go<8, 2, 4>(TK2F_ARGS);
  if (cfg == 403) return tk2f_go<8, 3, 4>(TK2F_ARGS);
  if (cfg == 206) return tk2f_go<8, 6, 2>(TK2F_ARGS);
  return tk2f_go<8, 8, 1>(TK2F_ARGS);
#undef TK2F_ARGS
#undef TK2F_ARGS0
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
                         torch::Tensor gs, int64_t start, int64_t count, double eps, torch::Tensor out, torch::Tensor zc) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm2<false>, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 2) != 0,
      (const float*)acc2.data_ptr(), (const int*)inv.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (const float*)gs.data_ptr(), (int)start, (int)count, (float)eps,
      (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr());
}
// s13: after moe_tk2f with PF & 4 (y32 rows = route slots, pre-scaled by gs2 * 256)
int64_t moe_norm3_launch(torch::Tensor y32, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start,
                         int64_t count, double eps, torch::Tensor out, torch::Tensor zc) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm2<true>, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 2) != 0,
      (const float*)y32.data_ptr(), (const int*)nullptr, (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (const float*)nullptr, (int)start, (int)count, (float)eps,
      (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr());
}
"""


def _gen_wgmma():
    out = ["template <int N> __device__ __forceinline__ void wgmma_bf16(float* d, const uint32_t* a, uint64_t desc);"]
    for n in range(8, 136, 8):  # fresh f1s6: + 104/112/120 (HOT_XN)
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
        "int64_t moe_norme_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, torch::Tensor, torch::Tensor);"
        "int64_t moe_norm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor);"
        "int64_t moe_sk_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tk2_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tb_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gather_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_twf_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_twm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_tw_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route2_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t);"
        "int64_t moe_act_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
        "torch::Tensor, torch::Tensor);"
        "int64_t moe_norm2_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, double, torch::Tensor, torch::Tensor);"
        "int64_t moe_norm3_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor, torch::Tensor);"
        "int64_t moe_tk2f_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_tk2fn_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, double);"
        "int64_t moe_twq_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_set_pdl(int64_t, int64_t);"
        "int64_t moe_set_blk(int64_t);"
        "int64_t moe_set_sa(int64_t);"
        "int64_t moe_set_hot(int64_t);"
        "int64_t moe_hot_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);")


def _lib():
    global _LIB
    if _LIB is None:
        from torch.utils.cpp_extension import load_inline
        h = hashlib.sha256(CUDA_SRC.encode()).hexdigest()[:12]
        _LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_CPP, cuda_sources=CUDA_SRC,
                           functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch", "moe_sk_launch", "moe_tk2_launch", "moe_tk2f_launch", "moe_tk2fn_launch", "moe_tb_launch",
                                      "moe_gather_launch", "moe_tw_launch", "moe_twf_launch", "moe_twm_launch", "moe_twq_launch",
                                      "moe_route2_launch", "moe_act_launch", "moe_norm2_launch", "moe_norm3_launch", "moe_set_pdl", "moe_set_blk", "moe_set_sa", "moe_set_hot", "moe_hot_launch"],
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


def _sq_pack(wq, s2, nq=4):
    """layer s68 (moe n14 SQ): wq int32 [E, NRT, 64, 32, 4] + s2 int32 [E, NRT/2, 64, 32] -> int32 [E, NRT/2, nq, 2 * W + S]: per
    (row-tile pair lr, k quarter q) the tk2f stage image {tile lr kc q*16.., tile lr + NRT/2 same kc, scales}."""
    E, nrt = wq.shape[0], wq.shape[1]
    w = wq.reshape(E, 2, nrt // 2, nq, -1)
    sc = s2.reshape(E, nrt // 2, nq, -1)
    return torch.cat((w[:, 0], w[:, 1], sc), dim=-1).contiguous()


def _s6_pack(wq, s2, nq=4):
    """layer s73 (moe n20 S6): as _sq_pack, but the stage's scales as 6-bit codes c = e4m3 byte - 80 (caller checked
    80 <= byte <= 0x7E): per (lr, q) {tile lr, tile lr + NRT/2, nibbles int32 [8 warps][32 lanes], high bits int16
    [8][32]}; lane word bits 4k / 2k hold code k = 4 h + r. 17920 B/stage (lossless)."""
    E, nrt = wq.shape[0], wq.shape[1]
    w = wq.reshape(E, 2, nrt // 2, nq, -1)
    c = s2.contiguous().view(torch.uint8).reshape(E, nrt // 2, nq, 2, 8, 32, 4).long() - 80  # E lr q h warp lane r
    c = c.permute(0, 1, 2, 4, 5, 3, 6).reshape(E, nrt // 2, nq, 8, 32, 8)                     # ... warp lane k
    k = torch.arange(8, device=c.device)
    nib = ((c & 15) << (4 * k)).sum(-1)
    hi = ((c >> 4) << (2 * k)).sum(-1)
    nib = (nib - ((nib >> 31) << 32)).to(torch.int32).reshape(E, nrt // 2, nq, 256)
    hi = (hi - ((hi >> 15) << 16)).to(torch.int16).reshape(E, nrt // 2, nq, 256).view(torch.int32)
    return torch.cat((w[:, 0], w[:, 1], nib, hi), dim=-1).contiguous()


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
    # layer s73 (moe n20 S6): needs every block scale byte in [80, 0x7E] (positive normal, value >= 8), else 8-bit SQ
    s6 = TK2F_SQ and TK2F_S6 and all(bool(((b >= 80) & (b <= 0x7E)).all()) for b in (
        st["w13s2"].view(torch.uint8), st["w2s2"].view(torch.uint8)))
    if s6:
        st["w13s6"] = _s6_pack(st["w13q"], st["w13s2"])
        st["w2s6"] = _s6_pack(st["w2q"], st["w2s2"])
        for a_, b_ in (("w13s6", "w13s2"), ("w2s6", "w2s2")):  # lossless round trip
            assert torch.equal(_s6_unpack(st[a_]), st[b_].reshape(st[b_].shape[:4])), a_
    if TK2F_SQ and (not s6 or TK2F_S6_KEEPSQ):  # layer s68 (moe n14): tk2f single-copy stage images (lossless repack of w13q/w2q + w13s2/w2s2)
        st["w13sq"] = _sq_pack(st["w13q"], st["w13s2"])
        st["w2sq"] = _sq_pack(st["w2q"], st["w2s2"])
    if TK2F_DIAG:  # layer s81: KPS 4 (36 KB) single-copy stage images for the DIAG probes
        st["w13sq4"] = _sq_pack(st["w13q"], st["w13s2"], 2)
        st["w2sq4"] = _sq_pack(st["w2q"], st["w2s2"], 2)
    if TW_CFG:
        rt = 2 if TW_CFG >= 1000 else TW_CFG // 100
        if rt not in TB_PACK:
            st[f"w13b{rt}"], st[f"w13sb{rt}"] = _tb_pack(st["w13q"], st["w13s"], True, rt)
            st[f"w2b{rt}"], st[f"w2sb{rt}"] = _tb_pack(st["w2q"], st["w2s"], False, rt)
    for rt in sorted(set(TB_PACK) | ({2} if TWF_MIX or TWF_RT == 2 or 2 in TWF_RT_M.values() else set())):
        st[f"w13b{rt}"], st[f"w13sb{rt}"] = _tb_pack(st["w13q"], st["w13s"], True, rt)
        st[f"w2b{rt}"], st[f"w2sb{rt}"] = _tb_pack(st["w2q"], st["w2s"], False, rt)
    st["gs13"] = weights["w13_gs"][:LOCAL].float().contiguous()
    st["gs2"] = weights["w2_gs"][:LOCAL].float().contiguous()
    st["lut"] = _scale_lut(dev)
    # SA decode (s9) needs every block scale positive and normal (e4m3 exponent field 1..15, not NaN)
    sfb = [weights[k][:LOCAL].contiguous().view(torch.uint8) for k in ("w13_sf", "w2_sf")]
    st["sa"] = all(bool((((b & 0x80) == 0) & ((b & 0x78) != 0) & (b != 0x7F)).all()) for b in sfb)
    del sfb
    st["cnt"] = torch.zeros(128, dtype=torch.int32, device=dev)  # moe_tk2f counters (zero between calls)
    st["tk"] = torch.zeros(1024, dtype=torch.int32, device=dev)  # moe_twf split-item tickets (self-resetting)
    st["w1"] = (1.0 + weights["post_norm_w"].float()).contiguous()
    torch.cuda.synchronize()
    global _DIAG_DONE
    if DIAG and not _DIAG_DONE:
        _DIAG_DONE = True
        _diag(st)
    return st


_DIAG_DONE = False


# (R, G) per batch size for gemm13 / gemm2: R row tiles (x16 rows) per item, G token groups of 8 per pass.
def _plan(M):
    if M == 1:
        return 10, 10
    if M <= 32:
        return 11, 11
    return 118, 118


PLAN_OVERRIDE = None
# PDL / row order are set once in the library (g_pdl = 4: big-M kernels only, g_blk = 0); moe_set_pdl / moe_set_blk
# change them for experiments. Judge s5: decode PDL (gemm2 or norm2) and tk2_map blk 2 at bs32 lose on the H200.
TB_RT = 0  # big-M moe_tb on (1) or off (0: old moe_big path)
TB_CFG = 48  # RT*10 + stages
TW_CFG = 418  # big-M wgmma kernel moe_tw: RT*100 + KG*10 + stages (0: off)
TWF_RT = 4  # s11: 2 = moe_twf with RT 2 / 64-token passes (prepare then packs w13b2/w2b2): judge 7dda218c bs128 224.5,
# bs256 267.0 us (RT 4: ~200 / ~247): worse when experts have <= 32 tokens; H100 layer v16 -5, v32 -7 us (H100 misleads)
TWF_MIX = 1  # s11: 1 = moe_twm (RT 2 / 64-token passes for experts with > 32 tokens, RT 4 for the rest)
TWF_RT_M = {}  # s11: per-M override, e.g. {256: 2} for the layer's verify 32 (hot experts with 90+ tokens: 34 vs 24 passes)
TWM_TAIL = 0  # s14: moe_twm runs the gemm2 of the lightest experts (up to this many RT 4 items) at RT 2 / 32-token passes
TWM_TAIL_M = {}  # s14: per-M override of TWM_TAIL
TWM_TAIL13 = 0  # s14: same for the gemm13 items of the lightest experts (RT 4 gemm13 item budget)
TWM_TAIL13_M = {}
TWM_ORDER = 0  # s14: moe_twm item order (bits, see moe_twm): 76 = pass-minor + multi-pass experts' items merged evenly
#   over both phases (H100 layer v32 248.8 -> 239.9 us without TWM_TAIL (with it: 247), v16 = 0); 2 = timeline stamps
TWM_ORDER_M = {256: 76}  # s14: per-M override
HOT_M = {}  # s29: {M: grid}: moe_hot (65..128-token experts, one N 128 pass) on that many CTAs before moe_twm
TWM_HOT = 0  # s17: moe_twm hot class (experts with 65..96 tokens: one pass, 1 tile per warp, N <= 96)
TWM_HOT_M = {}  # s17: per-M override
TWQ = 0  # s17: big-M on moe_twq (16 consumer warps x RT 2, 32-token passes) if nonzero (value = flags; bit 1 = timeline,
#   bit 3 = pass-minor item order; use 4 for plain on)
TWQ_M = {}  # s17: per-M override, e.g. {128: 4} (proxy: v16 -6.5 proxy-us, v32 +4.6)
TWF = 1  # big-M (TW_CFG 418): gemm13 + gemm2 in one persistent launch (moe_twf)
TB_PACK = ()  # RT values packed in prepare for moe_tb (and moe_twf RT 2)
_CFGS = (10, 11, 22, 118)
SK_CFG = 3088  # decode kernel cfg: 2DM = moe_sk (D, MB); 3[NC][NSTG] = moe_tk
TK2_MAX = 32  # M above this: big-M path
TK2F_DYN = 84204  # M > 1 cfg. s13: 84204 = 204 + y32 in route-slot order pre-scaled + moe_norm3 (bitwise = 204; H200 diag adfb897e d16 -0.45, d32 -0.75 us). M = 1 (TK2F) stays 204: the layer's n2u reads the old y32 order. 5402 = DYN row tail (D = 4): judge no gain (b890d69b)
TWSA = 1  # s9: same for moe_twf (big-M)
TK2SA = 1  # s9: arithmetic e4m3 scale decode in moe_tk2f (when prepare() found all scales positive normal)
TK2FN = 0  # s9: 1 = post-expert norm fused into moe_tk2f (moe_tk2fn_launch): judge bs1 +2 us (check 168fa04c), dead
TK2G_W = 0  # s10: GS16 2-group row weight = (8 + TK2G_W) / 8 in the static row split
TK2G_MIN = 17  # s10: GS16 only from this M (H100: GS16 is slower on the layer routings; A/B on the judge)
TK2G = 0  # s10: 1 = 16-token segments in moe_tk2f (2 token groups share each dequantized weight fragment): judge bs32 +7 us (check b25182e8), H100 layer routings +6..18 us: off
TWF_DBG = 0
TWF_SPLIT_M = {}  # s11 per-M override of TWF_SPLIT; judge c1659443 {128: 132, 256: 66}: tie (tail -12 us, twf +10 us)
NORM_EARLY = 1  # n4s1: moe_norm after moe_twm waits per token on its experts' gemm2 flags (needs #define NORM_EARLY 1)
TWF_SPLIT = 0  # s11: moe_twf runs its last TWF_SPLIT gemm2 items as 2 K halves (FP32 partials, ticket): shorter tail
TK2F = 204  # decode: gemm13 + gemm2 in one launch (moe_tk2f, cfg KPS*100+NSTG); 0 = 3-launch moe_tk2 path
TK2_NSTG = 8  # decode path: moe_tk2 stages (0: the 5-launch SK_CFG path)
SK_ZERO = 64  # CTAs zeroing Acc13 in route2


# s12 DIAG (print only; never submit with DIAG on): H200 timing of expert variants on the LAYER's routings
# (r-layer reference front + top-k, 5 seeds per load; /tmp/moe10/routes.pt), cold L2, CUDA-graph replay medians.
DIAG = 0
_DIAG_HDR = [('decode 1', 7, 1), ('decode 1', 11, 1), ('decode 1', 13, 1), ('decode 1', 17, 1), ('decode 1', 99, 1), ('decode 16', 7, 16), ('decode 16', 11, 16), ('decode 16', 13, 16), ('decode 16', 17, 16), ('decode 16', 99, 16), ('decode 32', 7, 32), ('decode 32', 11, 32), ('decode 32', 13, 32), ('decode 32', 17, 32), ('decode 32', 99, 32), ('verify 1', 7, 8), ('verify 1', 11, 8), ('verify 1', 13, 8), ('verify 1', 17, 8), ('verify 1', 99, 8), ('verify 16', 7, 128), ('verify 16', 11, 128), ('verify 16', 13, 128), ('verify 16', 17, 128), ('verify 16', 99, 128), ('verify 32', 7, 256), ('verify 32', 11, 256), ('verify 32', 13, 256), ('verify 32', 17, 256), ('verify 32', 99, 256)]
_DIAG_R = "eNqVXIt22zgOLc02laBVxDpQorWm/v+/5IpvAASdWZ8zjSVRJAjicXFJj0cfPmC9X66/7+HC+a940//zy9+vP+fD589b/jc/f/L71/XbW7z7T338ljosLUjL699P2qdnnZHr55/wb5Buy//5/Hfx/ee6b1uzMJ9w+f5gt/22pakm4ZZHfcA6In/a59ejjbvlx0sR75f3jzDQdv0J9+zmtW62dqONvbVpRbFdnUGQHUK3J5Yeyheg3eabc3rgvMX03IkpnL71A/Haso4c8kHit0uyP0L8J2lhpZ7se3nwWFRNvodb12sumpezVbj0Vm7O1xhDd67czCM8sqoeW7+CmHtbiJCPjUmzcfU9sOjrbI/o7H7FRw8i2Pb+S5mgLT1es/zjiDarUUS548UfFNIslwFVleQ21ULCajw80RebEBXjGfrxuBE9Vjd4T3PZ4oToZ0mmvHRzt6z3M+leeMwmNBVuvNcGT6LpoIZ7bbjprif850lXs3VcvD4tRnlkPVn8rci8EL+N7dsMHBMgrl9aZpcWxD6T2i0WPYXLLU0q+v2SXl2ez63zibMp+EF96+yn/i5eoM3lx8SnkFtAnYUr75l8v/PQue9axhfTX7ApIaCUVGt3je/qeJcshkQuaIOa6yuWu45KvGalhEiFZe2t78w+3/TvinsLQ/b2Mbbcak+a3s/O2avBpuZoy92lBZ4lW0m8ezkXjVZbZ3yhp0eV9X2jcgAN/DPRV/zSWh4ip8rP33D7M7vgNdr+VR9dmfkrdv6b6+6enD7k93vUw80L9W7nnek6NPqntPrKDvv3tyLO1+Xdf6+/v5O4X2F4Y7KkX8yqMkwxv4m81+e/faf3T2bcCzc5U2cQb9ySWp3/WZfxBlXnWD3J/+AGNVM1YMxldRWm+mAfOHBLtqfmp1qGr8O7OFwexDSdH8SgoMSD9RIUOjgABV+okabiDOxCUhgkmR1IOUsUuiS6h/tLeu8mBwBPTSjIPxPhPsMVSi0BN/MEds66qCeXBUkwxE6d6dYHuX2IGKZow5LbRg4EqQtEEKuI/XKTqZ8lanA/j1dmdlyUs5dsDs2AWOVMMchk6LqffYTGFpzDBV+nWcz/1uaN10PrcWStrBtso8KwcRF/Ap/AAWSJMdsUComkhOkFdGrCMw1mcsCbRONaMUHNezHvml/7hOD0ROGa8xttukAAeOvAscQyl9SpDwFxAItrjwik35AMy/NFSK7P2xyXFYnwwsIdm4BNygV93bVlpR4JuoGEr2u6uLebLiEtBy2gYrPl90tFU+rkYGPNbNyls8W4kDxCOtTj3nkkOW5eACna3l2NLuELWozPQDcYrGWz/zClGyzrXq0JWgy4lyura+6q9GgGqrjY8nWO9t+tjqG9xTQXE8smIPH1bVbWMiKz2MqxBAcsVYCmsUvKo93HIour8566uGA6J4CJY3fg2bbgSSAh52Qhg+W9fZgAkUwo6JDmUdfbve30kPoxpR4PRSYLC3tejIXkizfuS1aJrfAKdFdHWcVs1uvLPWeSHf9FmWF7m4GcLWEQF3ei5T09tzTulW6vNpMkKPznQI5bTCNzfDG89BdikD0tC5QyO9z04IQpd9WMcFMn7L4pwGzyq0/q9HgWF8aILkoHZx7+TeJy5EH/4PK62q3z33ygY3ssyzb7oIJUjMfYnUYiJzTiOGUQVoH2VKiNo0zP+mHYHz5Jzv2x54kRUA2OIZZa57ZyMVgduRXzrGkNapx1bORY6c/Q8WvQgW+UrMFSGhy6ehn8zP7yR3FqL6Y/zKdTWW8sdjV/Zx4V/8CrsiSE5sYcOu4GwNeSKMWghgBWP/ZN010CntyOgQ+R/GDVJtuiy/zSPyRdkaA3qupAQZ2Y+mev5pNHM/MrOAgZpKBm/nvB5RsCf7ZKSBk9yd0KTgcdBMf3kM2vAczBKky5s0nXG7ZYsrZYYr0+T2/1oOIowzyJoIwvcWNgA5rCsMPRA54jp3UFt6/e1mpzfoVd1fszc+nDxS2UbhITtyOx9MaTsngrSC6B77mlPC+lXDX2IHvYLAc5kfKEfWY4qAId5/xvnTJJFYcfAzCqBvhZpo1r9CvOwqk0n9UMATFhDzIIipeLwSNZKaOtJKSciDlw717zkrMkEEfhw67tw3SG8iG9GE5hHzcbZcTYPVaY+1/rXwE+zcyGdV9DbY6KM7N9ilaOup64QE8Beh5wsv1AqT6De1PHvWMAXB+HjIr5Dhk70uo6A9/CiNpi3ULvQONy0qybR/6eQItJi5UgYnwf7TeAJQxnOgAuZXxWy+jzxC74JNsqRHXUFkgzDQAvsZVImUugSqJ6Uky2rBwTAcH1Bn7z345W8Cca5nSOaxtpploo8hUCJUGmxukBawzfIcww0d0TFOMSdIVJr152Hq0BB1FcxOhVwB+ejjX0Y19rsvVg2SrMaf3L3FfVPCWbGdYb7/TWWax+h5yeQTE3I3P5PmURkOXwxlbsfVFIUQNIn5yR2c08kdSU39g5iwGyI4YO2teqmSn2jPPATNB0KSBlOmTRH1LOhlmHepZ2cvRJhWzfDILKmtXV3JwkMEkvb1NlJhrbn+313hHhGnWK/jVfU9PAUYSDQHhx+NUWuVZ8qkmfvNtpGE1rIVbqh2ZPwJkN1zKhcZHvmZXRiTJbV7b09paUfIVnCzTs7LkesUWiAvWB69Ni22whLIkn0Q4alXrWplbqfveqCccCiXtgC5Ib5QsiPzKz/dQRYLAj3HuT2cR1Kmym2rZdIrRfG1PGz5IAjrO1axU5sAHMnbTjFmxQIhzsaZ5qSE5SnvN3OILgOxJ3b76Hpr1f5Y0YZJAY/WsAx9UV1GoNjeFiB7Dt+6YlcAPoPCpkUBqUih9Nozvw1CjBblvmIBXV+hIvRaO+Auxz1wOA3G3rduYYr0K52LU8NoRPwKQprgbzmlebid296ZotVkf42PiChQ5JN9mfOKjIU5PJ66xrejyxPtn+y6yts9isXL2gB/4dOwc7Schz4Imwpu098KlmXDSnZ06nvKDZ45FdjJAYli4vMv9TT8KtSDf3K/+PrPpvrKzghGRc56+CHy2bmK9bi1uizqdsCycFzKgeMSJuGV7obTruG3KAtDh1g0g8ZdLCCQ2APBsIqqIKvgYZZXg+sv1Wfb5l2lSRxQHUAmU583QChU/gvX+xVK1e+IEa30vjD+YUoFETN7Xuto1dBgbCuf3X8G1rcLHXsmjISwlMH/kUhlImIGPWavwLGPYoIXNt/D3QebvqEfNglUHFka6/iTp/Hjr/zXFtlXpl893GUSlRWxNKGCDi5cKeHyXcg+bfHU/NDTecEDq6l49xwJy/42VAxASnzvZGCkkveCjmlgqv/4vCRUjbXk4Mv8iiXDZoEZGfbGL70tNIbcpCN35OxaFHOpfslKZWSzFLY1sCZfKDByjUVB8Pca0gyNnqPvYbHrjbMMhbMoASt3TbnuLvqtAB1g9LsGE0ONamZqpuZIFKXwtgS2qr5WKfOl6kmtuQP69j9IdgwOnOw0KK4haW2H1BZTX+Tx1+Y/Vx4RUnhS0j9f9cUwWyJRrPsj80hcN3LtnXth9I6XHr1W7MS4pNDww12prINzqe/6zw95vSM/qTnWjJfpZO2SDZcDb/H/ZgcWUp0zWzLDKbQ2WK7Bw4pdUKKsfKkiJuefzGqW01XyH8hwBtpCi8/mbBpjBgqrpRLacm48mpDNT9G4p5HpUdcR/dvk/DtROvQ2DgReM9tp+J5c0s8sLkmYV9SjfZdaxqYZAQ23k6b90RG995Ac2A5YRkT4HnI8GLOO7/TpKuTo/i7duPXAqIjX1yrDqhhsO2h0+p0gop04683RsfZEHZBHKtAFXP+UKa5ylpMxobul1BpvlSrwtiHGOZ+Iz8I0jcw6GXspAfHUMo+ZZY/HC2T+7fWZoTQU1rt+rL0AHn0PaoYR+VzTUm35gRsTx4AM84DhT2qzdp5L8ikvBJzRgfr89rQF1VltR5hxT9r2T+kM/LsbM7u6KYU2rrGDBarN2c7Q5eMMbKxofx9ugZM2TVnaHHytLCmfYdi3u9AQGKcf267d1ZWuAq5YNBKj+0Thi/Ho7eGq9Xz0nhuyBsHeNsxYElJvi80Sh4sEVypK5zCs7u9O/4OtDtXcqkQasV+DnzLixBy2Tl3o1Y/G3MHa2DWlzkrFqcEZbjI294mKmrm0WEBwWRQVNcf/7nxvSk/sDS/u394Wy4AQn650fpFX44jPNZTMuy+OA4ydWZ6dkZJWp2ioUC1OpDJ36wRd3lGXcqioYc2ZSbXEcseLqZL8FJJL78LpZDHtM/m0tN4F7CaWVX22lpjZxXLAHqlADAarL49KuZpbwP/Mj7mbb/QxJfSydA92+Qkj4SqHjPCaAsnlPr+XVKmwbl9WOQHW4Kj3pmRGt/oOCvQHp6+ZWS06BEOoDvBHDubcbW7VK2fpFfQKs4oxv9zsh/VKC6vqqtgHtt5oAmBVV4ti/i+e9rjtfVW/628v2mQjVapXXx8FlYqNMyTDOEE6gdr7yAcS92mOpB7vQSysdhf8C20yTsTC2WLz9v7X7k6o50mrq2fc5ZWwaBRVNfdzAbAUTzb95QWf03p0MWFZ7PBO+psPg2XDs0r8BU3bzCgkVhkudGtrmdTb9mOA/XriZV/lNMuekxSfO3GbS5UdHc9v4+6rqj8PMSb6BhCm2vmzGwqIA40HAa89xZhjTs8/osVfTmxa+ebOsind3XkVO8fygIm+9Bv13LfKCWjBLM3erPPySfkc71oFqT5pEj7HA6QrGy8i/4s56aQfa/LLDgfnUra/Lbqd1uSZweni+D0QH63r0qc4lt2+jkOwJiH1rYtPXphxadNc9KFVoTcwvstz6UNQTC9msO4s/YfvzCp/jRR3V3NPl/km4/FJo9ffsf7xy7jA=="
DIAG_VARIANTS = {  # s14 queued: H200 A/B of the twm item orders (H100: o76 v32 -9 us vs main, v16 0)
    "dec": [("gs8", {})],
    "big": [("main", {})] + [(f"o{o}", {"TWM_ORDER": o, "TWM_TAIL_M": {}}) for o in (0, 76, 12, 28, 92)]
           + [("o76t", {"TWM_ORDER": 76}), ("mainb", {})],
}


def _diag(st):
    import zlib, base64
    g = globals()
    raw = zlib.decompress(base64.b64decode(_DIAG_R))
    routes, p = [], 0
    for load, seed, T in _DIAG_HDR:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda()
        p += T * 8
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": H, "local_start": st["start"], "local_count": st["count"]}}
        routes.append((load, seed, T, data))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    res = {}
    for load, seed, T, data in routes:
        vs = DIAG_VARIANTS["dec" if T <= TK2_MAX else "big"]
        graphs = []
        for name, ov in vs:
            old = {k: g[k] for k in ov}
            g.update(ov)
            try:
                kernel_fn(data, st)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    o = kernel_fn(data, st)
                gr.replay()
                torch.cuda.synchronize()
                graphs.append((name, gr, o))
            finally:
                g.update(old)
        base = graphs[0][2].float()
        ts = {n: [] for n, _, _ in graphs}
        for _ in range(30):
            for n, gr, _ in graphs:
                torch.sum(flush, 0, out=sink)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); gr.replay(); b.record()
                torch.cuda.synchronize()
                ts[n].append(a.elapsed_time(b) * 1e3)
        for n, _, o in graphs:
            t = sorted(ts[n])[len(ts[n]) // 2]
            err = (o.float() - base).abs().max().item()
            res.setdefault((load, n), []).append(t)
            print(f"diag12 {load} s{seed} {n}: {t:.1f} us (maxdiff vs {graphs[0][0]} {err:.3g})", flush=True)
        del graphs
    for (load, n), v in res.items():
        print(f"diag12 MEAN {load} {n}: {sum(v) / len(v):.1f} us", flush=True)
    if DIAG_TL:
        for load, seed, T, data in routes:
            if T > TK2_MAX:
                for o in DIAG_TL:
                    _diag_tl(st, data, f"{load} s{seed}", o)
    del flush
    torch.cuda.synchronize()


DIAG_TL = ()  # s14: twm item orders to timeline in the diag (needs TWM_TL 1 in the CUDA source: diag copies only)


def _diag_tl(st, data, name, order):
    """s14: per-CTA moe_twm timeline (TWM_TL build): exit spread, HBM-rate estimate per 10 us, item durations by kind."""
    import collections
    g = globals()
    old = (g["TWM_ORDER"], g["TWM_TAIL_M"], g["TWM_ORDER_M"])
    g["TWM_ORDER"], g["TWM_TAIL_M"], g["TWM_ORDER_M"] = order | 2, {}, {}
    try:
        for _ in range(3):
            kernel_fn(data, st)
        torch.cuda.synchronize()
    finally:
        g["TWM_ORDER"], g["TWM_TAIL_M"], g["TWM_ORDER_M"] = old
    part, meta = st.pop("_tl")
    tl = part.view(torch.int64)[:132 * 64].view(132, 64).cpu()
    meta = meta.cpu()
    t0 = tl[:, 0].min().item()
    us = lambda v: (v - t0) / 1000
    n = tl[:, 1]
    ex = sorted(us(tl[c, 63].item()) for c in range(132))
    rows = sorted((us(tl[c, 2 + 3 * k].item()), us(tl[c, 3 + 3 * k].item()), tl[c, 4 + 3 * k].item())
                  for c in range(132) for k in range(min(n[c].item(), 20)))
    na = meta[0].item(); off = meta[33:66].tolist()
    E = [meta[1 + a].item() for a in range(na)]
    nt = [off[e + 1] - off[e] for e in E]; big = [x > 32 for x in nt]
    PS = [64 if b else 32 for b in big]; npass = [(x + q - 1) // q for x, q in zip(nt, PS)]
    nbig = 0
    while nbig < na and big[nbig] and (not (order & 64) or npass[nbig] > 1): nbig += 1
    items = []
    for g2, nrbl in ((0, [32 if b else 16 for b in big]), (1, [16 if b else 8 for b in big])):
        lst = []
        for a in range(na):
            if order & 4:
                lst += [(g2, a, q) for rb in range(nrbl[a]) for q in range(npass[a])]
            else:
                lst += [(g2, a, q) for q in range(npass[a]) for rb in range(nrbl[a])]
        mrg = (order >> 3) & 3
        if mrg and not (g2 and order & 32):
            phx = {1: 100, 2: 85, 3: 70}[mrg]
            nA = sum(npass[a] * nrbl[a] for a in range(nbig)); N = len(lst)
            if 0 < nA < N:
                W = max(nA, N * phx // 100); out = []
                for it in range(N):
                    ca = it * nA // W if it < W else nA; ca1 = (it + 1) * nA // W if it + 1 <= W else nA
                    out.append(lst[ca] if ca1 > ca else lst[nA + it - ca])
                lst = out
        items += lst
    T = ex[-1]; hb = [0.0] * (int(T / 10) + 2)
    kinds = collections.defaultdict(list)
    for a_, b_, it in rows:
        g2, a, q = items[it]
        nv = min(PS[a], nt[a] - q * PS[a])
        kinds[(g2, int(big[a]), q > 0, (nv + 7) // 8 * 8)].append(b_ - a_)
        if q: continue
        mb = 0.55 if big[a] else 1.09
        d = max(b_ - a_, 0.1)
        for z in range(int(a_ / 10), int(b_ / 10) + 1):
            lo, hi = max(a_, z * 10), min(b_, z * 10 + 10)
            if hi > lo: hb[z] += mb * (hi - lo) / d
    print(f"diagtl {name} o{order}: experts {na} exit min/med/max {ex[0]:.1f}/{ex[66]:.1f}/{ex[-1]:.1f} "
          f"first g2 start {min(a for a, b, it in rows if items[it][0]):.1f} last g13 end "
          f"{max(b for a, b, it in rows if not items[it][0]):.1f} tokens {nt}", flush=True)
    print(f"diagtl {name} o{order} HBM TB/s per 10us: " + " ".join(f"{v / 10:.2f}" for v in hb), flush=True)
    print(f"diagtl {name} o{order} kinds (g2 big reuse N: n mean): " + "; ".join(
        f"{k[0]}{k[1]}{int(k[2])} {k[3]}: {len(v)} {sum(v) / len(v):.1f}" for k, v in sorted(kinds.items())), flush=True)


TK2H = 0  # s12: merged-phase tk2f consumer (one main-loop body, i-cache): 8 = GS 8, 16 = GS 16 (from M >= TK2H_MIN16)
TK2H_MIN16 = 1
TK2H_MIN = 1  # n4s5: TK2H applies from this M (below: the main tk2f cfg); 9 = only calls that can have 2-group rows
TK2W = 0  # s12: weighted static row split for M > 1 (extra-group rows weigh TK2W / 8; 0 = by count), cfg + 70000
TK2H_PSC = 0  # s12: 1 = per-segment gemm13 -> gemm2 counters (cnt[64 + seg]) instead of the global one
TK2H_PF4 = 0  # n6: 1 = tk2h writes y32 like tk2f PF 4 (slot order, pre-scaled) -> moe_norm3 (cfg + 4; kinds 3/4 only)
TK2H_W = 0  # GS16 2-group row weight (8 + W) / 8 in the static split (tk2g_split)
TK2H_M = {}  # layer s58: {M: W + 1}: tk2h GS16 + PF4 (SA only) at that M, 2-group weight (8 + W) / 8


TK2F_BAL_M = set()  # moe n8 BAL: M values whose tk2f (84204 / NU8 85204, SA) runs balanced (cfg + 1000000)


def _tk2f_cfg(M, st):
    return _tk2f_cfg0(M, st) + (1000000 if M in TK2F_BAL_M and TK2SA and st["sa"] else 0)


TK2F_S6 = 0  # layer s73 (moe n20, r-moe 4a87466): 6-bit scale-code SQ stages when every scale byte is in [80, 0x7E]. OFF: H200 moe diag 13b528a8 tk2f d16 +0.7..1.0, d32 +2.9, tk2h W3 d32 +1.6 us (H100 -1..-3 = DRAM-bound artefact)
TK2F_S6_ON = True  # run tk2f / tk2h on them (False: the 8-bit SQ stages; needs TK2F_S6_KEEPSQ)
TK2F_S6_KEEPSQ = False  # also build the 8-bit SQ images when S6 is on (A/B tests)


def _s6_unpack(p):
    """inverse of _s6_pack's scale part -> int32 view of the paired scales (prepare-time round-trip check)."""
    E, rpe, nq = p.shape[0], p.shape[1], p.shape[2]
    t = p[..., -384:]
    nib = t[..., :256].reshape(E, rpe, nq, 8, 32, 1).long() & 0xFFFFFFFF
    hi = t[..., 256:].contiguous().view(torch.int16).reshape(E, rpe, nq, 8, 32, 1).long() & 0xFFFF
    k = torch.arange(8, device=p.device)
    c = ((nib >> (4 * k)) & 15) | (((hi >> (2 * k)) & 3) << 4)                 # E lr q warp lane k(= 4h + r)
    c = (c + 80).reshape(E, rpe, nq, 8, 32, 2, 4).permute(0, 1, 2, 5, 3, 4, 6)  # E lr q h warp lane r
    return c.to(torch.uint8).contiguous().view(torch.int32).reshape(E, rpe, nq * 16, 32)


TK2F_SQ = 1  # layer s68 (moe n14, H200 d32 -1.5, d16 -0.55 us, bitwise = off): 184204 (+ BAL) on the _sq_pack buffers


def _tk2f_w(c, st):
    """(cfg, w13, w2) for moe_tk2f_launch: SA 84204 (+ BAL) -> + 10000000 on the single-copy buffers (SQ)."""
    sq = c % 1000000 == 184204 or (140000 <= c % 1000000 < 150000 and c % 1000 == 4)  # 84204 (+ BAL) / moe n16 tk2h
    if TK2F_SQ and sq and TK2F_S6_ON and "w13s6" in st:  # layer s73 (moe n20 S6)
        return c + 20000000, st["w13s6"], st["w2s6"]
    if TK2F_SQ and sq and "w13sq" in st:
        return c + 10000000, st["w13sq"], st["w2sq"]
    return c, st["w13q"], st["w2q"]


TK2F_ORX = False  # layer n5s25: build moe n5s32's tk2f oracle probes (tk2x 7 orz, 8 orn, 9 ornc; DIAG check copies only)
TK2F_DIAG = False  # layer s81: build the tk2f DIAG probes (DIAG check copies only; sed to True)
TK2X_M = {}  # layer s81 DIAG: M -> probe code (tuner timing only): 1 nm204, 2 sq402, 3 sq403, 4 nm402, 5 nm403, 6 plain SQ 204


def _tk2x(c, st, code):
    """layer s81 DIAG: (cfg, w13, w2) of probe `code` (c = the SQ 184204 cfg) -- wrong outputs for the nm codes."""
    assert TK2F_DIAG and c == 10184204, c
    if code == 6:
        return c, st["w13sq"], st["w2sq"]
    if code == 1:
        return 10184214, st["w13sq"], st["w2sq"]
    return 10000000 + (184402, 184403, 184412, 184413)[code - 2], st["w13sq4"], st["w2sq4"]


def _tk2f_cfg0(M, st):
    sa = 100000 if TK2SA and st["sa"] else 0
    if TK2H_M.get(M) and sa:
        return sa + 40000 + (TK2H_M[M] - 1) * 1000 + 4
    if TK2W and M > 1:
        return sa + 70000 + TK2W * 1000
    if TK2H and M >= TK2H_MIN:
        return sa + (40000 + TK2H_W * 1000 if TK2H == 16 and M >= TK2H_MIN16 else 30000) + 20000 * TK2H_PSC + 4 * (TK2H_PF4 and not TK2H_PSC)
    return (TK2F_DYN if M > 1 else TK2F) + sa + (10000 + TK2G_W * 1000 if TK2G and M >= TK2G_MIN else 0)


def kernel_fn(data, st):
    cfg = data["config"]
    assert cfg["local_start"] == st["start"] and cfg["local_count"] == st["count"]
    lib = st["lib"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].float().contiguous()
    M = x.shape[0]
    dev = x.device
    meta = torch.empty(132, dtype=torch.int32, device=dev)  # [100, 132): NORM_EARLY gemm2 flags
    sroutes = torch.empty(M * TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((M * TOP_K, I), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M * TOP_K, H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    if M <= TK2_MAX and TK2_NSTG:
        inv = torch.empty(M * TOP_K, dtype=torch.int32, device=dev)
        y32 = torch.empty((M * TOP_K, H), dtype=torch.float32, device=dev)
        if TK2F and TK2FN:
            lib.moe_tk2fn_launch(st["w13q"], st["w13s2"], st["w2q"], st["w2s2"], st["lut"], st["gs13"], st["gs2"], x,
                                 ids, st["start"], st["count"], meta, sroutes, inv, inter, y32, st["cnt"], TK2F, tw,
                                 st["w1"], out, POST_NORM_EPS)
            return out
        if TK2F:
            c, w13_, w2_ = _tk2f_w(_tk2f_cfg(M, st), st)
            if TK2X_M.get(M, 0) >= 7:  # layer n5s25 DIAG: oracle code in cfg / 100000000 (same buffers and cfg otherwise)
                assert TK2F_ORX, TK2X_M
                c += (TK2X_M[M] - 6) * 100000000
            elif TK2X_M.get(M):  # layer s81 DIAG
                c, w13_, w2_ = _tk2x(c, st, TK2X_M[M])
            lib.moe_tk2f_launch(w13_, st["w13s2"], w2_, st["w2s2"], st["lut"], st["gs13"], st["gs2"], x, ids,
                                st["start"], st["count"], meta, sroutes, inv, inter, y32, st["cnt"], c)
            c5 = c % 1000000 % 100000
            if (80000 < c5 < 100000 and (c5 - 80000) // 1000 & 4) or (30000 <= c5 < 50000 and c5 % 1000 == 4):  # s13 slot-ordered y32 (n6: tk2h + 4)
                lib.moe_norm3_launch(y32, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out, st["cnt"])
                return out
            lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], POST_NORM_EPS, out,
                                 st["cnt"])
            return out
        lib.moe_tk2_launch(1, TK2_NSTG, st["w13q"], st["w13s2"], st["lut"], st["gs13"], x, ids, st["start"], st["count"],
                           meta, sroutes, inv, inter)
        lib.moe_tk2_launch(0, TK2_NSTG, st["w2q"], st["w2s2"], st["lut"], st["gs2"], inter, ids, st["start"],
                           st["count"], meta, sroutes, inv, y32)
        lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], POST_NORM_EPS, out, st["cnt"])
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
        lib.moe_norm2_launch(acc2, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], POST_NORM_EPS, out, st["cnt"])
        return out
    p13, p2 = PLAN_OVERRIDE(M) if PLAN_OVERRIDE else _plan(M)
    assert p13 in _CFGS and p2 in _CFGS, (p13, p2)
    rg = bool(data.get("rg")) and TW_CFG and data.get("gather_norm") is not None and data.get("bucket") is None
    if rg:  # layer s52: route + gathern in one launch below
        pass
    elif data.get("bucket") is None:
        lib.moe_route_launch(ids, st["start"], st["count"], meta, sroutes)
    else:  # layer s27: bucketed on the layer's router stream (meta, sroutes of moe_route_launch)
        meta, sroutes = data["bucket"]
    if TW_CFG:
        # inter here holds gemm2's B in the K-permuted order (see moe_tw)
        rt = 2 if TW_CFG >= 1000 else TW_CFG // 100
        xg = torch.empty((M * TOP_K, H), dtype=torch.bfloat16, device=dev)
        if rg:
            assert st["start"] == 0
            lib.moe_routegathern_launch(ids, st["count"], x, *data["gather_norm"], meta, sroutes, xg, data["gather_eps"])
        elif data.get("gather_norm") is None:
            lib.moe_gather_launch(x, meta, sroutes, xg)
        else:  # layer s27 GNORM: x = the pooler output; the pre-expert rmsnorm happens in the gather
            lib.moe_gathern_launch(x, *data["gather_norm"], meta, sroutes, xg, data["gather_eps"])
        if TWF and rt == 4:
            trt = TWF_RT_M.get(M, TWF_RT)
            trt = trt if f"w2b{trt}" in st else 4
            lib.moe_set_sa((1 if TWSA and st["sa"] else 0) | (2 if trt == 2 else 0))
            ns = TWF_SPLIT_M.get(M, TWF_SPLIT)
            part = (torch.full if TWF_DBG else torch.empty)((max(66 if ns >= 1022 else ns, 1) * 32768,), *((float("nan"),) if TWF_DBG else ()), dtype=torch.float32, device=dev)
            if TWQ_M.get(M, TWQ):
                q = TWQ_M.get(M, TWQ)
                qs, qk = (q >> 8) & 255, (q >> 16) & 3  # n21: K-split tail in the flags: q | items << 8 | log2 parts << 16
                if qs and qk and not DIAG_TL:
                    part = torch.empty((qs << qk) * 16384, dtype=torch.float32, device=dev)
                if DIAG_TL and TWM_ORDER_M.get(M, TWM_ORDER) & 2: st["_tl"] = (part, meta); q |= 2
                hq = (q >> 24) & 127  # moe s23: q | grid << 24 = moe_hot<NH 64> on grid CTAs first; moe_twq skips its experts
                q &= (1 << 24) - 1
                if hq:
                    lib.moe_set_hot(4)
                    lib.moe_hot_launch(st["w13b2"], st["w13sb2"], st["gs13"], st["w2b2"], st["w2sb2"], st["gs2"], st["lut"],
                                       xg, meta, sroutes, inter, y, hq)
                    q |= 1 << 20
                lib.moe_twq_launch(st["w13b4"], st["w13sb4"], st["gs13"], st["w2b4"], st["w2sb4"], st["gs2"], st["lut"], xg,
                                   meta, sroutes, inter, y, part, st["tk"], q)
            elif TWF_MIX and "w2b2" in st:
                tl = TWM_TAIL_M.get(M, TWM_TAIL) + 4096 * TWM_TAIL13_M.get(M, TWM_TAIL13) + (TWM_ORDER_M.get(M, TWM_ORDER) << 24)
                tl += (1 << 23) if TWM_HOT_M.get(M, TWM_HOT) else 0
                if tl and ns:  # s19: K-split tail together with order / hot (tail classes off)
                    tl = (tl & ~((1 << 23) - 1)) + (1 << 22) + 4096 * min(ns, 1023)
                ns = -tl if tl else ns
                if DIAG_TL and tl >> 24 & 2: st["_tl"] = (part, meta)
                hg = HOT_M.get(M, 0)
                lib.moe_set_hot(1 if hg else 0)
                if hg:  # s29: experts with 65..128 tokens in one N 128 pass, moe_twm (PDL secondary) skips them
                    lib.moe_hot_launch(st["w13b2"], st["w13sb2"], st["gs13"], st["w2b2"], st["w2sb2"], st["gs2"], st["lut"],
                                       xg, meta, sroutes, inter, y, hg)
                lib.moe_twm_launch(st["w13b4"], st["w13sb4"], st["gs13"], st["w2b4"], st["w2sb4"], st["gs2"], st["w13b2"],
                                   st["w13sb2"], st["w2b2"], st["w2sb2"], st["lut"], xg, meta, sroutes, inter, y, st["tk"],
                                   part, ns)
                if NORM_EARLY:
                    lib.moe_norme_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out, meta)
                    return out
            else:
              lib.moe_twf_launch(st[f"w13b{trt}"], st[f"w13sb{trt}"], st["gs13"], st[f"w2b{trt}"], st[f"w2sb{trt}"], st["gs2"], st["lut"], xg,
                               meta, sroutes, inter, y, st["tk"], part, ns)
            lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], POST_NORM_EPS, out)
            if TWF_DBG: st["dbg"] = (y, inter, meta, sroutes)
            return out
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
