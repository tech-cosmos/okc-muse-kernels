# n6s3 EPI2: tw_item's DYN epilogue (moe_twm / moe_twf / moe_hot RT 2) stages each warp's 8-token group in smem (1 KB per
# warp, past the ring: +8 KB dynamic smem) and writes it back as 16-B (gemm2 Y rows) / 8-B (RT 4 gated) / 4-B (RT 2 gated)
# chunks instead of one 2-byte store per value. Same values, same rounding: bitwise identical.
# usage: python tools/n6s3/mk_epi.py <in.py> <out.py>
import sys
s = open(sys.argv[1]).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)
EPI2 = r'''
// n6s3 EPI2: staged epilogue (see tw_epi for the layouts). stg = this warp's 8 x 64 bf16 staging buffer in smem.
template <int RT, int N, bool GATED, int NW>
__device__ __forceinline__ void tw_epi2(float (&acc)[RT][N / 2], int lane, int warp, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out,
                                        __nv_bfloat16* stg) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int H = GATED ? RT / 2 : RT;
  const int col0 = (rb * NW * H + warp * H) * 16, c0 = col0 & 63;
#pragma unroll
  for (int gi = 0; gi < N / 8; ++gi) {
    if (gi * 8 >= nv) break;
    __syncwarp();
#pragma unroll
    for (int r = 0; r < H; ++r)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = t * 2 + (q & 1), cw = r * 16 + g + ((q & 2) ? 8 : 0);
        const float v = acc[r][gi * 4 + q] * gsc;
        if (GATED) {
          const float u = acc[r + H][gi * 4 + q] * gsc;
          const int c = (c0 + cw) & 63;
          const int lc = (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
          stg[tok * 64 + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        } else {
          stg[tok * 64 + cw] = __float2bfloat16(v);
        }
      }
    __syncwarp();
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
  }
  __syncwarp();
}
'''
rep('// DYN: one main loop for both GEMMs (moe_twf), epilogue picked by g2 at run time\n',
    EPI2 + '\n// DYN: one main loop for both GEMMs (moe_twf), epilogue picked by g2 at run time\n')
rep('''    if (g2) tw_epi<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
    else tw_epi<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
''', '''    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(const_cast<char*>(smem) + NSTG * (STRIDE ? STRIDE : C::STG) + warp * 1024);
    if (g2) tw_epi2<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out, stg);
    else tw_epi2<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out, stg);
''')
rep('  constexpr int SMEM = NSTG * TWC<RT, true, 8>::STG + 1024;\n', '  constexpr int SMEM = NSTG * TWC<RT, true, 8>::STG + 1024 + 8192;  // n6s3 EPI2 staging\n')
rep('  constexpr int SMEM = NSTG * ((NH > 64 ? 128 : NH) * 128 + TWC<2, true, 8>::WB + TWC<2, true, 8>::SB) + 1024;\n',
    '  constexpr int SMEM = NSTG * ((NH > 64 ? 128 : NH) * 128 + TWC<2, true, 8>::WB + TWC<2, true, 8>::SB) + 1024 + 8192;  // n6s3 EPI2\n')
i = s.find('static int64_t twm_go(')
j = s.find('  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;\n', i)
s = s[:j] + '  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024 + 8192;  // n6s3 EPI2 staging\n' + s[j + len('  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;\n'):]
if len(sys.argv) > 3 and sys.argv[3] == "roll":
    # n6s3 ROLL: the 8-token-group loop is NOT unrolled (i-cache: moe_twm is ~16.5k SASS); group gi always reads
    # acc[.][0..3], then the accumulators shift down by 4 (register moves, constant indices)
    rep("""#pragma unroll
  for (int gi = 0; gi < N / 8; ++gi) {
    if (gi * 8 >= nv) break;
    __syncwarp();""", """#pragma unroll 1
  for (int gi = 0; gi < N / 8; ++gi) {
    if (gi * 8 >= nv) break;
    __syncwarp();""")
    rep("        const float v = acc[r][gi * 4 + q] * gsc;\n        if (GATED) {\n          const float u = acc[r + H][gi * 4 + q] * gsc;\n          const int c = (c0 + cw) & 63;",
        "        const float v = acc[r][q] * gsc;\n        if (GATED) {\n          const float u = acc[r + H][q] * gsc;\n          const int c = (c0 + cw) & 63;")
    rep("""              *reinterpret_cast<const uint4*>(stg + tok * 64 + k * 8);
      }
    }
  }
  __syncwarp();""", """              *reinterpret_cast<const uint4*>(stg + tok * 64 + k * 8);
      }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int i = 0; i + 4 < N / 2; ++i) acc[r][i] = acc[r][i + 4];
  }
  __syncwarp();""")
open(sys.argv[2], 'w').write(s)
print('ok')
