// proj n7s9: fa1 = batch-1 full-attention decode (d1 full: 1 query token, 16 q heads, 2 KV heads, D 64, FP8 KV,
// page 1) with one CTA per SM that holds its whole K/V slice in shared memory.
//   pre-wait : page-table rows -> cp.async gather of the CTA's K/V rows (128 B each) into swizzled smem
//   post-wait: q/k/v of the new token from qkv (bf16 [2304]); QK and PV on mma.sync m16n8k16 f16 (e4m3 -> f16 is
//              exact; queries bf16-rounded as in the reference, bf16 -> f16 exact in range), FP32 accumulators;
//              per-warp max, CTA merge in smem, then FP32 red.add into NR replicas at reference max 0 (log2 units);
//              the last CTA (ticket, epoch counter) normalizes, applies v_scale, the per-head RMSNorm and the
//              sigmoid gate, writes bf16 out [1024], and re-zeroes the replicas.
// Out-of-range CTA max (|m| > 100 in log2 units) sets flag[0] (prototype: no rare path yet).
// Diag modes (NR >> 8): 0 normal; 1 loads only; EVERY mode >= 2 SKIPS THE KV LOADS: 2 no loads, 3 + return after
// compute, 4 + no new token, 5 + return before the REDs, 6 + return after REDs + fence, 7/8/9/10 cache-store probes.
// H100 (n7s9, cold L2, graph, minus empty graph): full 18.3 us | loads only 10.5 | no loads 12.2 | layer attn_src
// (_gd_full + _red_fin) 13.7. L2-warm: loads only 2.7, compute (mode 3) 3.6, +merge 4.0, +REDs/fence 5.6, all 10.0.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace fa1 {
constexpr int HQ = 16, HKV = 2, D = 64, G = 8, NT = 512, KSW = 5, WPH = NT / 64;  // WPH warps per KV head  // KSW: max 16-token k-steps per warp

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ uint32_t cvt2(uint32_t v16) {  // two e4m3 (low 16 bits) -> f16x2
  uint32_t r;
  asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}\n" : "=r"(r) : "r"(v16));
  return r;
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// smem: K rows [CH][128 B] then V rows [CH][128 B]; 16-B chunk j of row r at K: j ^ ((r & 1) << 2), V: j ^ (((r >> 1) & 3) << 1)
__global__ void __launch_bounds__(NT, 1) attn(const __nv_bfloat16* __restrict__ qkv, uint8_t* kc, uint8_t* vc,
    const float* __restrict__ ksc, const float* __restrict__ vsc, const int* __restrict__ pt,
    const int* __restrict__ seqp, const long* __restrict__ loc, float* red, unsigned* cnt, int* flag,
    __nv_bfloat16* __restrict__ out, int CH, float qsf, float smsc, int NR) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int mode = NR >> 8;  // diag: 1 loads only, 2 no loads
  NR &= 255;
  __shared__ __align__(16) __half sQ[HQ * D];
  __shared__ float sM[2 * WPH][G], sL[2 * WPH][G];
  __shared__ float sO[2 * WPH][G][D + 1];
  __shared__ float sMc[HQ], sSc[HQ], sKs[4], sKN[HKV * D], sVN[HKV * D];
  __shared__ unsigned sLast;
  const int t = threadIdx.x, c0 = blockIdx.x * CH;
  uint8_t* sK = sm;
  uint8_t* sV = sm + CH * 128;
  // ---- pre-wait: gather this CTA's rows (pos < seq; rows past seq are zero-filled and masked) ----
  const int seq = __ldg(seqp);
  const long lc = __ldg(loc);
  if (blockIdx.x == gridDim.x - 1 && t < 2 && mode != 10)  // the new token's cache rows into L2 before they are written
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"((t ? vc : kc) + lc * 128));
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
  int* sPT = reinterpret_cast<int*>(sm + CH * 256);
  for (int r = t; r < CH; r += NT) sPT[r] = c0 + r < seq ? __ldg(pt + c0 + r) : -1;
  __syncthreads();
  for (int i = t; i < (mode >= 2 ? 0 : CH * 16); i += NT) {
    const int r = i >> 4, part = i & 15, j = part & 7;
    const int pr = sPT[r];
    const bool ok = pr >= 0;
    const long row = ok ? (long)pr : 0;
    if (part < 8) cp16(sbase + r * 128 + ((j ^ ((r & 1) << 2)) << 4), kc + row * 128 + j * 16, ok ? 16 : 0);
    else cp16(sbase + CH * 128 + r * 128 + ((j ^ (((r >> 1) & 3) << 1)) << 4), vc + row * 128 + j * 16, ok ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  // ---- q: per-head RMSNorm, * qk_scale_factor, bf16-rounded (reference), stored as f16 ----
  if (t < 256) {
    const int h = t >> 4, l = t & 15;
    const uint2 raw = *reinterpret_cast<const uint2*>(qkv + h * D + l * 4);
    float x[4] = {__low2float(*(__nv_bfloat162*)&raw.x), __high2float(*(__nv_bfloat162*)&raw.x),
                  __low2float(*(__nv_bfloat162*)&raw.y), __high2float(*(__nv_bfloat162*)&raw.y)};
    float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rs = rsqrtf(ss / D + 1e-5f);
    for (int e = 0; e < 4; e++) sQ[h * D + l * 4 + e] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(x[e] * rs * qsf)));
  }
  const float ks = __ldg(ksc), vs = __ldg(vsc);
  const bool newtok = blockIdx.x == gridDim.x - 1 && mode != 4;  // the last CTA has the fewest past tokens
  float xk = 0.f, xv = 0.f;
  if (newtok && t >= 256 && t < 256 + HKV * D) {  // the new token's k (RMS over each head's 64) and v
    xk = __bfloat162float(qkv[HQ * D + (t - 256)]);
    xv = __bfloat162float(qkv[HQ * D + HKV * D + (t - 256)]);
    float ss = xk * xk;
    for (int o = 1; o < 32; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((t & 31) == 0) sKs[(t - 256) >> 5] = ss;
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  if (mode == 1) return;
  if (newtok && t >= 256 && t < 256 + HKV * D) {  // FP8 k/v into the cache row and smem
    const int t2 = t - 256, hh = t2 >> 6;
    const float rs = rsqrtf((sKs[2 * hh] + sKs[2 * hh + 1]) / D + 1e-5f);
    const __nv_fp8_e4m3 fk(xk * rs / ks), fv(xv / vs);
    // pack 4 bytes per lane (32-bit stores); these threads skip the pre-ticket fence (the rows are read after the kernel)
    uint32_t bk = *reinterpret_cast<const uint8_t*>(&fk), bv = *reinterpret_cast<const uint8_t*>(&fv);
    bk |= __shfl_down_sync(0xffffffffu, bk, 1) << 8; bv |= __shfl_down_sync(0xffffffffu, bv, 1) << 8;
    bk |= __shfl_down_sync(0xffffffffu, bk, 2) << 16; bv |= __shfl_down_sync(0xffffffffu, bv, 2) << 16;
    if ((t2 & 3) == 0 && mode != 7) {
      uint8_t* kd = mode == 8 ? reinterpret_cast<uint8_t*>(flag + 64) : kc + (mode == 9 ? 0 : lc) * 128;
      uint8_t* vd = mode == 8 ? reinterpret_cast<uint8_t*>(flag + 128) : vc + (mode == 9 ? 0 : lc) * 128;
      *reinterpret_cast<uint32_t*>(kd + t2) = bk;
      *reinterpret_cast<uint32_t*>(vd + t2) = bv;
    }
    sKN[t2] = float(fk);
    sVN[t2] = float(fv);
  }
  const float sc2 = smsc * ks * 1.4426950408889634f;  // score -> log2 units
  const int w = t >> 5, ln = t & 31, g = ln >> 2, cq = ln & 3, kh = w / WPH, sub = w % WPH;
  // this warp's k-steps: CH / 16 split 4 ways
  const int nks = CH >> 4, kq = nks / WPH, krem = nks % WPH;
  const int k0 = sub * kq + min(sub, krem), nk = kq + (sub < krem ? 1 : 0);
  uint32_t qa[8];
  {
    const uint4 q0 = *reinterpret_cast<const uint4*>(sQ + (kh * G + g) * D + cq * 16);
    const uint4 q1 = *reinterpret_cast<const uint4*>(sQ + (kh * G + g) * D + cq * 16 + 8);
    qa[0] = q0.x; qa[1] = q0.y; qa[2] = q0.z; qa[3] = q0.w; qa[4] = q1.x; qa[5] = q1.y; qa[6] = q1.z; qa[7] = q1.w;
  }
  float s[2 * KSW][2];
  float mx = -INFINITY;
#pragma unroll
  for (int kk = 0; kk < KSW; kk++) {
    if (kk < nk) {
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const int r = (k0 + kk) * 16 + hf * 8 + g;
        const uint4 kw = *reinterpret_cast<const uint4*>(sK + r * 128 + ((kh * 4 + cq) ^ ((r & 1) << 2)) * 16);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        mma(c, qa[0], qa[1], cvt2(kw.x & 0xffff), cvt2(kw.x >> 16));
        mma(c, qa[2], qa[3], cvt2(kw.y & 0xffff), cvt2(kw.y >> 16));
        mma(c, qa[4], qa[5], cvt2(kw.z & 0xffff), cvt2(kw.z >> 16));
        mma(c, qa[6], qa[7], cvt2(kw.w & 0xffff), cvt2(kw.w >> 16));
        const int p0 = c0 + (k0 + kk) * 16 + hf * 8 + 2 * cq;
        s[2 * kk + hf][0] = p0 < seq ? c[0] * sc2 : -INFINITY;
        s[2 * kk + hf][1] = p0 + 1 < seq ? c[1] * sc2 : -INFINITY;
        mx = fmaxf(mx, fmaxf(s[2 * kk + hf][0], s[2 * kk + hf][1]));
      }
    }
  }
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  const float mref = mx == -INFINITY ? 0.f : mx;
  float o[8][2];
#pragma unroll
  for (int n = 0; n < 8; n++) o[n][0] = o[n][1] = 0.f;
  float lsum = 0.f;
#pragma unroll
  for (int kk = 0; kk < KSW; kk++) {
    if (kk < nk) {
      const float p00 = exp2f(s[2 * kk][0] - mref), p01 = exp2f(s[2 * kk][1] - mref);
      const float p10 = exp2f(s[2 * kk + 1][0] - mref), p11 = exp2f(s[2 * kk + 1][1] - mref);
      lsum += (p00 + p01) + (p10 + p11);
      const uint32_t a0 = pk(p00, p01), a2 = pk(p10, p11);
      const int rb = (k0 + kk) * 16 + 2 * cq;  // rows rb, rb+1 (b0) and rb+8, rb+9 (b1); (r >> 1) & 3 == cq
      const int off = kh * 64 + g * 8, ch = (off >> 4) ^ (cq << 1), bo = (ch << 4) | (off & 15);
      const uint2 A = *reinterpret_cast<const uint2*>(sV + rb * 128 + bo);
      const uint2 B = *reinterpret_cast<const uint2*>(sV + (rb + 1) * 128 + bo);
      const uint2 C = *reinterpret_cast<const uint2*>(sV + (rb + 8) * 128 + bo);
      const uint2 E = *reinterpret_cast<const uint2*>(sV + (rb + 9) * 128 + bo);
      uint32_t x0 = __byte_perm(A.x, B.x, 0x5140), x1 = __byte_perm(A.x, B.x, 0x7362);
      uint32_t x2 = __byte_perm(A.y, B.y, 0x5140), x3 = __byte_perm(A.y, B.y, 0x7362);
      uint32_t y0 = __byte_perm(C.x, E.x, 0x5140), y1 = __byte_perm(C.x, E.x, 0x7362);
      uint32_t y2 = __byte_perm(C.y, E.y, 0x5140), y3 = __byte_perm(C.y, E.y, 0x7362);
      float c[4];
#define PV(n, X, Y, SH)                                                     \
  c[0] = o[n][0]; c[1] = o[n][1]; c[2] = 0.f; c[3] = 0.f;                   \
  mma(c, a0, a2, cvt2((X >> SH) & 0xffff), cvt2((Y >> SH) & 0xffff));        \
  o[n][0] = c[0]; o[n][1] = c[1];
      PV(0, x0, y0, 0) PV(1, x0, y0, 16) PV(2, x1, y1, 0) PV(3, x1, y1, 16)
      PV(4, x2, y2, 0) PV(5, x2, y2, 16) PV(6, x3, y3, 0) PV(7, x3, y3, 16)
#undef PV
    }
  }
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
  // warp partial -> smem: o[n][e] = O[head g][d = 8 * (2 cq + e) + n]
  if (cq == 0) { sM[w][g] = mx; sL[w][g] = lsum; }
#pragma unroll
  for (int n = 0; n < 8; n++) { sO[w][g][8 * (2 * cq) + n] = o[n][0]; sO[w][g][8 * (2 * cq + 1) + n] = o[n][1]; }
  __syncthreads();
  // ---- CTA merge (+ the new token in CTA 0), then red.add at reference 0 ----
  if (mode == 3) return;
  if (newtok) {  // new token score per q head: q . dequant(k_new)
    const int h = t >> 5, l = t & 31;
    float p = __half2float(sQ[h * D + 2 * l]) * sKN[(h >> 3) * D + 2 * l] + __half2float(sQ[h * D + 2 * l + 1]) * sKN[(h >> 3) * D + 2 * l + 1];
    for (int o = 1; o < 32; o <<= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
    if (l == 0) sSc[h] = p * sc2;
    __syncthreads();
  }
  if (t < HQ) {
    const int hh = t >> 3, gg = t & 7;
    float m = -INFINITY;
    for (int q = 0; q < WPH; q++) m = fmaxf(m, sM[hh * WPH + q][gg]);
    if (newtok) m = fmaxf(m, sSc[t]);
    sMc[t] = m;
    if (m != -INFINITY && !(fabsf(m) <= 100.f)) atomicOr(flag, 1);
  }
  __syncthreads();
  if (mode == 5) return;
  if (t < 256) {
    const int hd = t >> 4;  // head 0..15, 4 d each
    const int hh = hd >> 3, gg = hd & 7;
    const float m = sMc[hd];
    float f[WPH];
    for (int q = 0; q < WPH; q++) f[q] = m == -INFINITY ? 0.f : exp2f(sM[hh * WPH + q][gg]);  // reference max 0
    const int rep = blockIdx.x % NR;
    float* R = red + rep * (HQ * D + HQ);
    float v[4];
    for (int e = 0; e < 4; e++) {
      const int d = (t & 15) * 4 + e;
      v[e] = 0.f;
      for (int q = 0; q < WPH; q++) v[e] += sO[hh * WPH + q][gg][d] * f[q];
      if (newtok) v[e] += exp2f(sSc[hd]) * sVN[hh * D + d];
    }
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(R + hd * D + (t & 15) * 4), "f"(v[0]), "f"(v[1]),
                 "f"(v[2]), "f"(v[3]) : "memory");
    if ((t & 15) == 0) {
      float l = 0.f;
      for (int q = 0; q < WPH; q++) l += sL[hh * WPH + q][gg] * f[q];
      if (newtok) l += exp2f(sSc[hd]);
      atomicAdd(R + HQ * D + hd, l);
    }
  }
  if (t < 256) __threadfence();  // orders this thread's REDs before the ticket
  __syncthreads();
  if (mode == 6) return;
  if (t == 0) sLast = (atomicAdd(cnt, 1u) + 1u) % gridDim.x == 0u;
  __syncthreads();
  if (!sLast) return;
  __threadfence();
  if (t < 256) {
    const int hd = t >> 4, l16 = t & 15;
    float v[4], l = 0.f;
    for (int e = 0; e < 4; e++) v[e] = 0.f;
    float4 xs[4];
    float ls[4];
    for (int r0 = 0; r0 < NR; r0 += 4) {  // NR % 4 == 0: 4 loads in flight at once
#pragma unroll
      for (int rp = 0; rp < 4; rp++) {
        const float* R = red + (r0 + rp) * (HQ * D + HQ);
        xs[rp] = __ldcg(reinterpret_cast<const float4*>(R + hd * D + l16 * 4));
        ls[rp] = __ldcg(R + HQ * D + hd);
      }
#pragma unroll
      for (int rp = 0; rp < 4; rp++) { v[0] += xs[rp].x; v[1] += xs[rp].y; v[2] += xs[rp].z; v[3] += xs[rp].w; l += ls[rp]; }
    }
    float ss = 0.f;
    for (int e = 0; e < 4; e++) { v[e] = v[e] / l * vs; ss += v[e] * v[e]; }
    for (int o2 = 1; o2 < 16; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rs = rsqrtf(ss / D + 1e-5f);
    for (int e = 0; e < 4; e++) {
      const int d = l16 * 4 + e;
      const float gt = __bfloat162float(qkv[HQ * D + 2 * HKV * D + hd * D + d]);
      out[hd * D + d] = __float2bfloat16_rn(v[e] * rs * (1.f / (1.f + __expf(-gt))));
    }
    for (int rp = 0; rp < NR; rp++) {
      float* R = red + rp * (HQ * D + HQ);
      *reinterpret_cast<float4*>(R + hd * D + l16 * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (l16 == 0) R[HQ * D + hd] = 0.f;
    }
  }
}
}  // namespace fa1

void fa1_launch(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vs,
                torch::Tensor pt, torch::Tensor seq, torch::Tensor loc, torch::Tensor red, torch::Tensor cnt,
                torch::Tensor flag, torch::Tensor out, int64_t nb, int64_t ch, double qsf, double smsc, int64_t nr) {
  const int smem = (int)ch * 260;
  static bool attr = false;
  cudaFuncSetAttribute(fa1::attn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  fa1::attn<<<(int)nb, fa1::NT, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)qkv.data_ptr(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), ks.data_ptr<float>(),
      vs.data_ptr<float>(), pt.data_ptr<int>(), seq.data_ptr<int>(), (const long*)loc.data_ptr(), red.data_ptr<float>(),
      (unsigned*)cnt.data_ptr(), flag.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(), (int)ch, (float)qsf,
      (float)smsc, (int)nr);
}
