// one row-tile pair rp of a segment: 8 ring stages, cross-warp reduce, epilogue. TWO: a 16-token segment, the
// second 8-token group's x fragments come from smem (xs, lane-private) and share every dequantized A fragment.
template <int NRT, int NC, int NSTG, bool GATED, bool TWO>
__device__ __forceinline__ void tk2f_row(const uint32_t* slut, const uint4 (&xf)[64 / NC][2], const uint4* xs, void* Out,
                                         int rp, int off, int n, float gsc, float* s_red, uint4* smem, uint64_t* full,
                                         uint64_t* empty, int& st, int& ph, int& rb) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC, SW = NC * 32, SB = NC * 1152, NG = TWO ? 2 : 1;
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
  for (int q = 0; q < NQ; ++q) {
    uint4 xb0, xb1;
    if (TWO) { xb0 = xs[(q * 2) * 32]; xb1 = xs[(q * 2 + 1) * 32]; }
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
        mma16816(acc[0][ii], af, u4get(xf[q][j >> 1], (j & 1) * 2), u4get(xf[q][j >> 1], (j & 1) * 2 + 1));
        if (TWO) {
          const uint4 xb = (j >> 1) ? xb1 : xb0;
          mma16816(acc[NG - 1][ii], af, u4get(xb, (j & 1) * 2), u4get(xb, (j & 1) * 2 + 1));
        }
      }
    }
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
        reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

// GS = tokens per segment (8, or 16 with the 2-group rows); counters cnt[a * 4 + segment index in the expert]
template <int NRT, int NC, int NSTG, bool GATED>
__device__ __forceinline__ void tk2f_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int r0, int r1,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint4* s_x2, int GS, uint64_t* full,
                                             uint64_t* empty, int& st, int& ph, int& rb) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  uint4* xs = s_x2 + warp * (NQ * 2 * 32) + lane;
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;
  float gsc = 0.f;
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
      ci = a * 4 + gi;
      if (!GATED) {  // all 256 gemm13 rows of this segment written (by any CTA)
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt + ci) : "memory");
        } while (c < 256);
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
          uint4 x0, x1;
          if (GATED) {
            x0 = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
            x1 = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
          } else {  // written in this kernel: L2 only
            x0 = v ? __ldcg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
            x1 = v ? __ldcg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
          }
          if (h) { xs[(qq * 2) * 32] = x0; xs[(qq * 2 + 1) * 32] = x1; }
          else { xf[qq][0] = x0; xf[qq][1] = x1; }
        }
      }
    }
    if (n > 8) tk2f_row<NRT, NC, NSTG, GATED, true>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb);
    else tk2f_row<NRT, NC, NSTG, GATED, false>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb);
    if (GATED && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
      asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(cnt + ci, R + 1 - max(seg_start, r0));
      }
    }
  }
}
