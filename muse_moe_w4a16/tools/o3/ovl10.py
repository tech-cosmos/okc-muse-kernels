# o1 (lead 2026-10-31 09:55Z pipe-balanced dq ORACLE): ovl8 + roles 14/15 (dq without the 4 HMUL2: o_i passed
# straight to the wgmma = speed bound of ANY scale-step replacement, wrong values) and 16/17 (each HMUL2 -> one IADD3 with
# the scale: = a 1-int-op-per-pair replacement, also wrong values). Timing only.
# was: n11a (lead 03:50Z LANE A, DIAG ONLY, never submitted): FP8 W4A8 variant of the twm stage vs production twm.
# Base = tools/n7s47/ovl2.py (same stage: RT row tiles, 4 u32 of e2m1 codes per tile per lane = 32 weights, 8 static smem
# stages, x tile = SW128 B from smem, 2 consumer WGs each timing itself, producer WG idle with setmaxnreg 40/232).
# Roles: 1 twm BF16 (prod dq -> 4 RS k16 wgmma) | 2 BF16 dq only | 3 BF16 RS wgmma const A
#        7 FP8 F0: e2m1 -> e4m3 by bit masks only (exact x 2^-6, block scale NOT applied: oracle lower bound) -> 2 RS k32 e4m3 wgmma
#        8 FP8 F1: block scale folded lossily: e2m1 -> f16x2 (mask, = code*2^-14), HMUL2 by f16 scale, cvt.rn.satfinite.e4m3x2.f16x2, pack
#        9 F1 dq only | 10 FP8 RS wgmma const A | 11 F0 dq only
# Same weight bytes per stage in all roles; FP8 x tile = half the bytes per K (timing-only data).
import torch, os, sys, fcntl, subprocess, glob, re
from torch.utils.cpp_extension import load_inline
def gen_rs0(n):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return (f"template <> __device__ __forceinline__ void wg_rs0<{n}>(float* d, const uint32_t* a, uint64_t desc, int z) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(z));\n}}\n')
def gen_rs(n, fp8):
    na = n // 2; dl = ", ".join(f"%{i}" for i in range(na)); al = ", ".join(f"%{na + i}" for i in range(4))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    ins = 'k32.f32.e4m3.e4m3' if fp8 else 'k16.f32.bf16.bf16'
    tail = 'p, 1, 1' if fp8 else 'p, 1, 1, 0'
    nm = 'wg_rs8' if fp8 else 'wg_rs'
    return (f"template <> __device__ __forceinline__ void {nm}<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}{ins} {{{dl}}}, {{{al}}}, %{na + 4}, {tail};\\n}}\\n"\n'
            f'    : {outs} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}\n')
NS = (8, 16, 24, 32, 40, 48, 56, 64)
SRC = r"""
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t desc);
template <int N> __device__ __forceinline__ void wg_rs0(float* d, const uint32_t* a, uint64_t desc, int z);
template <int N> __device__ __forceinline__ void wg_rs8(float* d, const uint32_t* a, uint64_t desc);
@@GEN@@
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void vfma(float& a, float s, float p) { asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a) : "f"(s), "f"(p)); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ uint32_t hmul(uint32_t a, uint32_t b) {
  uint32_t d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ uint32_t cvt8(uint32_t lo, uint32_t hi) {  // two f16x2 -> 4 e4m3 bytes
  uint32_t d; asm("{.reg .b16 a, b;\n cvt.rn.satfinite.e4m3x2.f16x2 a, %1;\n cvt.rn.satfinite.e4m3x2.f16x2 b, %2;\n mov.b32 %0, {a, b};}"
                  : "=r"(d) : "r"(lo), "r"(hi)); return d; }
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0); a[1] = bmul(o1, s1); a[2] = bmul(o2, s0); a[3] = bmul(o3, s1);
}
// F0: prepare would place one code per byte in bits {7,4,3,2} (out 0) and the other in bits {6,5,1,0} (out 1):
// e4m3 byte s|0|0|e1 e0 m|0 0 = code * 2^-6 exactly (incl. 0.5 -> subnormal). Rows g / g+8 = outputs 0 / 1.
__device__ __forceinline__ void dq8f0(uint32_t q, uint32_t* a) {
  a[0] = q & 0x9C9C9C9Cu;
  a[1] = ((q << 2) & 0x8C8C8C8Cu) | ((q >> 2) & 0x10101010u);
}
// F1: same masks as dq() but f16 (s at 15, e1 e0 m at 11..9: code * 2^-14), HMUL2 by the f16 scale (s * 2^7), cvt -> e4m3
__device__ __forceinline__ void dq8f1(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x8E008E00u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = cvt8(hmul(o0, s0), hmul(o2, s0));
  a[1] = cvt8(hmul(o1, s1), hmul(o3, s1));
}
// F2: per (row, stage) 8-byte LUT of e4m3(code_k * s * 2^-4), k = 0..7 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}: = s byte
// + exponent shifts for {.5, 1, 2, 4} and s15 = e4m3(1.5 s) + shifts for {1.5, 3, 6} (one rounding = F1's). Weights
// repacked so q's low half = out-0 magnitudes (nibbles, bit 3 = out-0 sign of byte i at 8i + 7) and the high half = out 1
// (its signs at 8i + 3): out 0 = prmt(LUT_g, q & 0x7777) | (q & 0x80808080), out 1 = prmt(LUT_g8, q >> 16 & 0x7777) | (q << 4 & ..).
__device__ __forceinline__ void lut8(uint32_t w, uint32_t* L) {  // w low 16 = scale bytes (row g, row g + 8)
  uint32_t h, b15;
  asm("{.reg .b16 t; mov.b32 {t, _}, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "r"(w));
  asm("{.reg .b16 t; .reg .b32 m; mul.rn.f16x2 m, %1, %2; cvt.rn.satfinite.e4m3x2.f16x2 t, m; cvt.u32.u16 %0, t;}" : "=r"(b15) : "r"(h), "r"(0x3E003E00u));
  // row g: s = byte 0 of w, s15 = byte 0 of b15 (prmt index 4); row g8: byte 1 / index 5
  L[0] = (__byte_perm(w, b15, 0x4000u) & 0xFFFFFF00u) - 0x20202800u;
  L[1] = __byte_perm(w, b15, 0x4040u) - 0x10101818u;
  L[2] = (__byte_perm(w, b15, 0x5111u) & 0xFFFFFF00u) - 0x20202800u;
  L[3] = __byte_perm(w, b15, 0x5151u) - 0x10101818u;
}
__device__ __forceinline__ void dq8f2(uint32_t q, const uint32_t* L, uint32_t* a) {
  a[0] = __byte_perm(L[0], L[1], q & 0x7777u) | (q & 0x80808080u);
  a[1] = __byte_perm(L[2], L[3], (q >> 16) & 0x7777u) | ((q << 4) & 0x80808080u);
}
__device__ __forceinline__ void dqn(uint32_t q, uint32_t* a) {  // oracle: no scale step
  const uint32_t M = 0x81C081C0u;
  a[0] = q & M; a[1] = __funnelshift_l(q, q, 10) & M; a[2] = __funnelshift_l(q, q, 13) & M;
  a[3] = ((q & 0x001B001Bu) * 16648u) & M;
}
__device__ __forceinline__ uint32_t iadd(uint32_t a, uint32_t b) { uint32_t d; asm volatile("add.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ void dqi(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {  // oracle: 1 int op per pair
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M, o1 = __funnelshift_l(q, q, 10) & M, o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = iadd(o0, s0); a[1] = iadd(o1, s1); a[2] = iadd(o2, s0); a[3] = iadd(o3, s1);
}
__device__ __forceinline__ uint32_t u4get(const uint4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }
constexpr int NSTG = 8, XB = 8192;
template <int RT> struct SZ { static constexpr int WB = 8 * RT * 512, STG = XB + WB + 8 * 32 * 8; };

template <int ROLE, int RT, int N>
__device__ __forceinline__ unsigned long long body(const char* sm, int warp, int lane, int wg, int nst, float* sinkp) {
  constexpr int STG = SZ<RT>::STG, XBW = XB + SZ<RT>::WB, D = 2;
  constexpr bool F8 = ROLE >= 7 && ROLE < 14;
  float acc[RT][N / 2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[r][i] = 0.f;
  constexpr bool SD = ROLE >= 18;
  float P[SD ? 2 : 1][RT][SD ? N / 2 : 1];
#pragma unroll
  for (int d = 0; d < (SD ? 2 : 1); ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < (SD ? N / 2 : 1); ++i) P[d][r][i] = 0.f;
  uint32_t af[D][RT][4];
  const uint32_t seed = (uint32_t)clock() & 0x10u;
  const int zz = nst == -12345;  // runtime 0: scale-d false, unknown to ptxas
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < 4; ++i) af[d][r][i] = (0x3C383C38u ^ ((d * 16 + r * 4 + i) << 1)) | seed;
  uint32_t xs = 0;
  int st = 0;
  const bool DQ = ROLE == 1 || ROLE == 2 || ROLE == 7 || ROLE == 8 || ROLE == 9 || ROLE == 11 || ROLE == 12 || ROLE == 13 || (ROLE >= 14 && ROLE != 20);
  const bool MMA = ROLE == 1 || ROLE == 3 || ROLE == 7 || ROLE == 8 || ROLE == 10 || ROLE == 12 || ROLE == 14 || ROLE == 16 || ROLE >= 18;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int kc = 0; kc < nst; ++kc) {
    const char* sb = sm + st * STG;
    uint4 wv[RT];
    uint32_t s0[RT], s1[RT], Lt[RT][4];
    uint2 sv = make_uint2(0, 0);
    if (DQ) {
#pragma unroll
      for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XB)[(warp * RT + r) * 32 + lane];
      sv = reinterpret_cast<const uint2*>(sb + XBW)[warp * 32 + lane];
#pragma unroll
      for (int r = 0; r < RT; ++r) if (ROLE == 12 || ROLE == 13) { const uint32_t w = (r & 2) ? sv.y : sv.x; lut8((r & 1) ? (w >> 16) : w, Lt[r]); } else { const uint32_t w = (r & 2) ? sv.y : sv.x; s0[r] = (w & 0x00FF00FFu) | 0x3F003F00u; s1[r] = ((w >> 8) & 0x00FF00FFu) | 0x3F003F00u; }
    }
    const uint64_t d0 = sw128_desc(sb);
    constexpr int NJ = F8 ? 2 : 4;
#pragma unroll
    for (int j0 = 0; j0 < NJ; ++j0) {
      const int sl = j0 & 1;
      if (DQ) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (ROLE == 14 || ROLE == 15 || ROLE >= 18) dqn(u4get(wv[r], j0), af[sl][r]);
          else if (ROLE == 16 || ROLE == 17) dqi(u4get(wv[r], j0), s0[r], s1[r], af[sl][r]);
          else if (!F8) dq(u4get(wv[r], j0), s0[r], s1[r], af[sl][r]);
          else if (ROLE == 12 || ROLE == 13) { dq8f2(u4get(wv[r], 2 * j0), Lt[r], af[sl][r]); dq8f2(u4get(wv[r], 2 * j0 + 1), Lt[r], af[sl][r] + 2); }
          else if (ROLE == 7 || ROLE == 11) { dq8f0(u4get(wv[r], 2 * j0), af[sl][r]); dq8f0(u4get(wv[r], 2 * j0 + 1), af[sl][r] + 2); }
          else { dq8f1(u4get(wv[r], 2 * j0), s0[r], s1[r], af[sl][r]); dq8f1(u4get(wv[r], 2 * j0 + 1), s0[r], s1[r], af[sl][r] + 2); }
        }
      }
      if (!MMA) {
#pragma unroll
        for (int r = 0; r < RT; ++r) xs ^= af[sl][r][0] ^ af[sl][r][1] ^ af[sl][r][2] ^ af[sl][r][3];
      } else {
        if (ROLE == 21) {
          wg_wait<0>();
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int i = 0; i < N / 2; ++i) fence_f(P[0][r][i]);
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            const uint32_t w = (r & 2) ? sv.y : sv.x, b = (r & 1) ? (w >> 16) : w;
            const float sa = __uint_as_float(((b >> (2 * j0)) & 0x7Fu) * (1u << 20) + 0x3C000000u);
            const float sb = __uint_as_float(((b >> (2 * j0 + 8)) & 0x7Fu) * (1u << 20) + 0x3C000000u);
#pragma unroll
            for (int i = 0; i < N / 2; ++i) vfma(acc[r][i], (i & 2) ? sb : sa, P[0][r][i]);
          }
        }
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int r = 0; r < RT; ++r) { if (ROLE == 21) wg_rs0<N>(P[0][r], af[sl][r], d0 + j0 * 2, zz); else if (SD) wg_rs0<N>(P[sl][r], af[sl][r], d0 + j0 * 2, zz); else if (F8) wg_rs8<N>(acc[r], af[sl][r], d0 + j0 * 2); else wg_rs<N>(acc[r], af[sl][r], d0 + j0 * 2); }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        if (SD) {
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int i = 0; i < N / 2; ++i) fence_f(P[ROLE == 21 ? 0 : sl][r][i]);
        }
        if (ROLE != 21) wg_wait<1>();
        if (ROLE == 22) {  // promotion ALU cost only: same FFMAs + scale decodes, operand = acc (not a wgmma register): no dependency, no C7514
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            const uint32_t w = (r & 2) ? sv.y : sv.x, b = (r & 1) ? (w >> 16) : w;
            const float sa = __uint_as_float(((b >> (2 * j0)) & 0x7Fu) * (1u << 20) + 0x3C000000u);
            const float sb = __uint_as_float(((b >> (2 * j0 + 8)) & 0x7Fu) * (1u << 20) + 0x3C000000u);
#pragma unroll
            for (int i = 0; i < N / 2; ++i) vfma(acc[r][i], (i & 2) ? sb : sa, acc[r][(i + 1) % (N / 2)]);
          }
        }
        if (ROLE == 18) {
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int i = 0; i < N / 2; ++i) fence_f(P[sl ^ 1][r][i]);  // previous k16's partial is complete: acc += s(row) * P (FP32), scale e4m3 -> f32 = (b & 0x7F) << 20 + 120 << 23
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            const uint32_t w = (r & 2) ? sv.y : sv.x, b = (r & 1) ? (w >> 16) : w;
            const float sa = __uint_as_float(((b >> (2 * j0)) & 0x7Fu) * (1u << 20) + 0x3C000000u);
            const float sb = __uint_as_float(((b >> (2 * j0 + 8)) & 0x7Fu) * (1u << 20) + 0x3C000000u);
#pragma unroll
            for (int i = 0; i < N / 2; ++i) acc[r][i] = fmaf((i & 2) ? sb : sa, P[sl ^ 1][r][i], acc[r][i]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_r(af[(j0 + 1) & 1][r][i]);
    }
    if (++st == NSTG) st = 0;
  }
  wg_wait<0>();
  if (SD) {
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < N / 2; ++i) { fence_f(P[0][r][i]); fence_f(P[1][r][i]); }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i < N / 2; ++i) acc[r][i] += P[0][r][i] + P[1][r][i];
  }
  const unsigned long long t1 = clock64();
  float s = __uint_as_float(xs);
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += acc[r][i];
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int r = 0; r < RT; ++r) s += __uint_as_float(af[d][r][0] ^ af[d][r][3]);
  *sinkp = s;
  return t1 - t0;
}

template <int R0, int R1, int RT, int N>
__global__ void __launch_bounds__(384, 1) ovl(unsigned long long* out, float* sink, int nst) {
  extern __shared__ __align__(1024) char sm[];
  constexpr int STG = SZ<RT>::STG;
  const int tid = threadIdx.x, wg = tid >> 7, warp = (tid >> 5) & 7, lane = tid & 31;
  for (int i = tid; i < NSTG * STG / 4; i += 384) reinterpret_cast<uint32_t*>(sm)[i] = (i * 2654435761u) & 0x3F3F3F3Fu;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (wg == 2) { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n"); return; }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  unsigned long long t = 0;
  float* sp = sink + blockIdx.x * 384 + tid;
  if (wg == 0) t = body<R0, RT, N>(sm, warp, lane, 0, nst, sp);
  else t = body<R1, RT, N>(sm, warp, lane, 1, nst, sp);
  if ((tid & 127) == 0) out[wg] = t;
}
int64_t run(int64_t cfg, torch::Tensor out, torch::Tensor sink, int64_t nst) {
  constexpr int SMB = NSTG * SZ<4>::STG + 1024;
  int c = 0;
  #define GO(R0, R1, RT, NN) if (cfg == c++) { auto k = ovl<R0, R1, RT, NN>; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB); \
    k<<<1, 384, SMB>>>((unsigned long long*)out.data_ptr(), (float*)sink.data_ptr(), (int)nst); return 0; }
  @@CFGS@@
  return -1;
}
"""
CF = [("BF16 twm||twm (prod)", 1, 1), ("BF16 dq||dq", 2, 2), ("BF16 mma||mma", 3, 3),
      ("FP8 F0 twm||twm (no scale)", 7, 7), ("FP8 F1 twm||twm (scale fold)", 8, 8),
      ("FP8 F0 dq||dq", 11, 11), ("FP8 F1 dq||dq", 9, 9), ("FP8 mma||mma", 10, 10),
      ("FP8 F2 twm||twm (LUT fold)", 12, 12), ("FP8 F2 dq||dq", 13, 13),
      ("NOH twm||twm (no HMUL2)", 14, 14), ("NOH dq||dq", 15, 15), ("IAD twm||twm (HMUL2->IADD)", 16, 16), ("IAD dq||dq", 17, 17),
      ("SD0F twm||twm (sd0 + FFMA scale)", 18, 18), ("SD0N twm||twm (sd0, no scale)", 19, 19), ("SD0M mma||mma (sd0)", 20, 20), ("SD0W twm||twm (sd0, wait0+FFMA)", 21, 21), ("SD0X twm||twm (sd0 + FFMA ALU, no dep)", 22, 22)]
SH = [(4, 32), (2, 64)] if not os.environ.get("SH") else [tuple(int(x) for x in t.split("x")) for t in os.environ["SH"].split(",")]
CFS = os.environ.get("CFS")
CF = [c for c in CF if not CFS or c[0].split()[0] + " " + c[0].split()[1] in CFS.split(";")]
ALLC = [(lab, r0, r1, rt, n) for (rt, n) in SH for (lab, r0, r1) in CF]
gos = "\n  ".join(f"GO({r0}, {r1}, {rt}, {n})" for (lab, r0, r1, rt, n) in ALLC)
SRC = SRC.replace("@@GEN@@", "".join(gen_rs(n, f) for n in NS for f in (False, True)) + "".join(gen_rs0(n) for n in NS)).replace("@@CFGS@@", gos)
NAME = "o3_ovl10_" + os.environ.get("TAG", "a")
lib = load_inline(NAME, cpp_sources="int64_t run(int64_t, torch::Tensor, torch::Tensor, int64_t);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=bool(os.environ.get("VERB")))
bdir = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}"))[0]
so = f"{bdir}/{NAME}.so"
cuo = "/usr/local/cuda-13.0/bin/cuobjdump" if os.path.exists("/usr/local/cuda-13.0/bin/cuobjdump") else "cuobjdump"
try:
    sass = subprocess.run([cuo, "-sass", so], capture_output=True, text=True).stdout
except Exception:
    sass = ""
info = {}
for blk in sass.split("Function : ")[1:]:
    nm = blk.split()[0]
    m = re.match(r"_Z3ovlILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", nm)
    if m: info[tuple(int(x) for x in m.groups())] = (blk.count("DEPBAR.LE gsb0, 0x0"), len(re.findall(r"[HQ]GMMA", blk)), len(re.findall(r"\bSTL\b|\bLDL\b", blk)))
out = torch.zeros(4, dtype=torch.int64, device="cuda"); sink = torch.zeros(384, device="cuda")
NST, REP = int(os.environ.get("NST", "512")), int(os.environ.get("REP", "5"))
for ci, (lab, r0, r1, rt, n) in enumerate(ALLC):
    out.zero_(); lib.run(ci, out, sink, 16); torch.cuda.synchronize()
    best = None
    for _ in range(REP):
        with open('/tmp/okc-gpu.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            lib.run(ci, out, sink, NST); torch.cuda.synchronize()
        v = [out[i].item() / NST for i in range(2)]
        if best is None or max(v) < max(best): best = v
    dep0, hg, lm = info.get((r0, r1, rt, n), ("?", "?", "?"))
    print(f"RT{rt} N{n:2d} {lab:30s} WG0 {best[0]:6.0f} WG1 {best[1]:6.0f} cyc/stage (64 rows x 64 K x RT per WG)  (dep0 {dep0} hg {hg} lmem {lm})", flush=True)
