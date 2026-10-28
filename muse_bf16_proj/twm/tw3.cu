// ---- proj n5s10 moe_tw3: moe_twm's big-M work with 3 INDEPENDENT consumer warpgroups per CTA (416 threads: 3 x 128
// consumers + 1 producer warp, no setmaxnreg, ptxas cap 152 registers). Each warpgroup g has its own NSTG ring and
// streams HALF-items: one warpgroup's 256 (RT 4) / 128 (RT 2) rows of a twm item = warps 4*hf .. 4*hf+3 of the w*b4 /
// w*b2 packing, which are contiguous inside every k chunk (no repack). Per stage: x tile (TMA, 8 KB max) + half the
// weights + half the scales. One producer thread serves the 3 rings round-robin and never blocks (test_wait on the
// empty slots, one look at done[e] per round for gemm2 items), so a ring waiting on gemm13 does not stall the others.
// Half-item k of the global counter = twm item k >> 1, half k & 1. done[e] / done[32 + e] count halves (x2).
// Default twm order (heaviest experts first, pass-major), no hot class / tail classes / K split.
__device__ __forceinline__ bool mbar_test(uint64_t* b, int parity) {
  uint32_t ok;
  asm volatile(
      "{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0;
}

template <int RT, int N, int NSTG, bool SA, int STRIDE>
__device__ __forceinline__ void tw3_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                         const uint32_t* slut, int wl, int wq, int lane, int nv, int off, int rb,
                                         float gsc, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out,
                                         bool g2) {
  constexpr int XB = 64 * 128, WBH = 8 * RT * 512 / 2;
  constexpr int NA = N / 2;
  float acc[RT][NA];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[RT][4], afB[RT][4];
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const char* sb = smem + st * STRIDE;
    if (kc) mbar_wait(&full[st], ph);
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XB)[(wl * RT + r) * 32 + lane];
    uint32_t sc[(RT + 1) / 2];
    if (RT == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(sb + XB + WBH + (wl * 32 + lane) * RT * 2);
      sc[0] = v.x; sc[RT / 2 - 1] = v.y;
    } else {
      sc[0] = *reinterpret_cast<const uint32_t*>(sb + XB + WBH + (wl * 32 + lane) * RT * 2);
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
    for (int j0 = 0; j0 < 4; ++j0) {
      const bool odd = j0 & 1;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (odd) dq(u4get(wv[r], j0), s0[r], s1[r], afB[r]);
        else dq(u4get(wv[r], j0), s0[r], s1[r], afA[r]);
      }
      wg_fence();
#pragma unroll
      for (int r = 0; r < RT; ++r) wgmma_bf16<N>(acc[r], odd ? afB[r] : afA[r], d0 + j0 * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int r = 0; r < RT; ++r)
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
  for (int r = 0; r < RT; ++r) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { fence_r(afA[r][i]); fence_r(afB[r][i]); }
#pragma unroll
    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
  if (g2) tw_epi<RT, N, false, 8>(acc, lane, wq, nv, off, rb, gsc, sroutes, Out);
  else tw_epi<RT, N, true, 8>(acc, lane, wq, nv, off, rb, gsc, sroutes, Out);
}

#ifndef TW3_NWG
#define TW3_NWG 3  // consumer warpgroups (2 or 3)
#endif
#ifndef TW3_NSTG
#define TW3_NSTG (TW3_NWG == 3 ? 4 : 6)
#endif
#ifndef TW3_REG
#define TW3_REG (TW3_NWG == 3 ? 152 : 224)
#endif
template <int NSTG, bool SA, int PB = 64>
__global__ void __maxnreg__(TW3_REG) moe_tw3(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const char* __restrict__ W13h, const char* __restrict__ S13h, const char* __restrict__ W2h,
    const char* __restrict__ S2h,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y) {
  using CG = TWC<4, true, 8>;
  using CN = TWC<4, false, 8>;
  using HG = TWC<2, true, 8>;
  using HN = TWC<2, false, 8>;
  constexpr int XB = CG::XB, STRIDE = XB + CG::WB / 2 + CG::SB / 2;
  static_assert(STRIDE % 1024 == 0 && HG::WB / 2 + HG::SB / 2 <= CG::WB / 2 + CG::SB / 2, "stage layout");
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem0 = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[TW3_NWG][NSTG], s_big[32], s_nd2[32];
  __shared__ __align__(8) uint64_t full[TW3_NWG][NSTG], empty[TW3_NWG][NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid < 256) slut[tid] = lut[tid];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int g = 0; g < TW3_NWG; ++g)
      for (int i = 0; i < NSTG; ++i) { mbar_init(&full[g][i], 1); mbar_init(&empty[g][i], 4 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    const int big = nt > 32;
    const int npass = big ? (nt + PB - 1) / PB : (nt + 31) / 32;
    s_big[tid] = big;
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      int incl = npass * (big ? (gq ? HN::NRB : HG::NRB) : (gq ? CN::NRB : CG::NRB));
      if (gq) s_nd2[tid] = 2 * incl;
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
  const int n13 = s_cum[0][32], nh = 2 * (n13 + s_cum[1][32]);
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb) {
    g2 = item >= n13;
    const int it = g2 ? item - n13 : item;
    const int* cum = s_cum[g2];
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = s_big[a] ? (g2 ? HN::NRB : HG::NRB) : (g2 ? CN::NRB : CG::NRB);
    const int rem = it - cum[a];
    p = rem / nrb;
    rb = rem - p * nrb;
  };
  if (warp == 4 * TW3_NWG) {
    if (lane) return;
    int* counter = meta + 66;
    // per ring: state 0 = needs an item, 1 = gemm2 item waiting for done[e], 2 = streaming, 3 = sentinel, 4 = finished
    int sta[TW3_NWG], hi[TW3_NWG], kc[TW3_NWG], st[TW3_NWG], ph[TW3_NWG], n[TW3_NWG], off[TW3_NWG], mi[TW3_NWG], g2v[TW3_NWG], ev[TW3_NWG], need[TW3_NWG], wb[TW3_NWG], sbb[TW3_NWG], keep[TW3_NWG];
    const char* ws[TW3_NWG];
    const char* ss[TW3_NWG];
#pragma unroll
    for (int g = 0; g < TW3_NWG; ++g) { sta[g] = 0; st[g] = 0; ph[g] = 0; n[g] = 0; kc[g] = 0; }
    int live = TW3_NWG;
    while (live) {
#pragma unroll
      for (int g = 0; g < TW3_NWG; ++g) {
        if (sta[g] == 0) {
          hi[g] = atomicAdd(counter, 1);
          if (hi[g] >= nh) {
            sta[g] = 3;
          } else {
            int g2, a, p, rb;
            decode(hi[g] >> 1, g2, a, p, rb);
            const int hf = hi[g] & 1, e = s_e[a], big = s_big[a], PS = big ? PB : 32;
            const int o = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - o);
            const int nrb = big ? (g2 ? HN::NRB : HG::NRB) : (g2 ? CN::NRB : CG::NRB);
            const int WB = big ? HG::WB : CG::WB, SB = big ? HG::SB : CG::SB;
            ws[g] = (big ? (g2 ? W2h : W13h) : (g2 ? W2 : W13)) + ((size_t)e * nrb + rb) * 64 * WB + hf * (WB / 2);
            ss[g] = (big ? (g2 ? S2h : S13h) : (g2 ? S2 : S13)) + ((size_t)e * nrb + rb) * 64 * SB + hf * (SB / 2);
            wb[g] = WB / 2;
            sbb[g] = SB / 2;
            off[g] = o;
            mi[g] = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
            g2v[g] = g2;
            keep[g] = o + PS < s_off[e + 1];
            ev[g] = e;
            kc[g] = 0;
            if (g2) {
              const int b13 = s_big[a], PS13 = b13 ? PB : 32;
              need[g] = 2 * ((s_off[e + 1] - s_off[e] + PS13 - 1) / PS13) * (b13 ? HG::NRB : CG::NRB);
              sta[g] = 1;
            } else {
              sta[g] = 2;
            }
          }
        }
        if (sta[g] == 1) {
          int v;
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + ev[g]) : "memory");
          if (v >= need[g]) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            sta[g] = 2;
          }
        }
        if (sta[g] >= 2 && sta[g] <= 3 && (n[g] < NSTG || mbar_test(&empty[g][st[g]], ph[g] ^ 1))) {
          char* sb = smem0 + (g * NSTG + st[g]) * STRIDE;
          if (sta[g] == 3) {
            s_item[g][st[g]] = -1;
            mbar_arrive(&full[g][st[g]]);
            sta[g] = 4;
            --live;
          } else {
            if (kc[g] == 0) s_item[g][st[g]] = hi[g];
            const int xbytes = 1024 << mi[g];
            mbar_expect_tx(&full[g][st[g]], xbytes + wb[g] + sbb[g]);
            tma2d(sb, g2v[g] ? &maps2.m[mi[g]] : &maps13.m[mi[g]], kc[g] * 64, off[g], &full[g][st[g]]);
            uint64_t pol;
            if (keep[g]) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
            else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
            bulk_g2s_h(sb + XB, ws[g] + (size_t)kc[g] * (2 * wb[g]), wb[g], &full[g][st[g]], pol);
            bulk_g2s_h(sb + XB + wb[g], ss[g] + (size_t)kc[g] * (2 * sbb[g]), sbb[g], &full[g][st[g]], pol);
            ++n[g];
            if (++st[g] == NSTG) { st[g] = 0; ph[g] ^= 1; }
            if (++kc[g] == 64) sta[g] = 0;
          }
        }
      }
    }
    return;
  }
  const int g = warp >> 2, wl = warp & 3;
  const char* smem = smem0 + g * NSTG * STRIDE;
  uint64_t* F = full[g];
  uint64_t* E = empty[g];
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&F[st], ph);
    const int hi = s_item[g][st];
    if (hi < 0) break;
    int g2, a, p, rb;
    decode(hi >> 1, g2, a, p, rb);
    const int hf = hi & 1, wq = hf * 4 + wl;
    const int e = s_e[a], big = s_big[a], PS = big ? PB : 32;
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
#define TWI3(RT, NN) tw3_item<RT, NN, NSTG, SA, STRIDE>(smem, F, E, st, ph, slut, wl, wq, lane, nv, off, rb, gsc, sroutes, out, g2)
    if (!big) {
      switch ((nv + 7) >> 3) {
        case 1: TWI3(4, 8); break;
        case 2: TWI3(4, 16); break;
        case 3: TWI3(4, 24); break;
        default: TWI3(4, 32); break;
      }
    } else {
      switch ((nv + 7) >> 3) {
        case 1: TWI3(2, 8); break;
        case 2: TWI3(2, 16); break;
        case 3: case 4: TWI3(2, 32); break;
        default: TWI3(2, PB); break;
      }
    }
#undef TWI3
    if (!g2) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync %0, 128;" ::"r"(1 + g) : "memory");
      if (wl == 0 && lane == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
#if NORM_EARLY
    else {
      asm volatile("bar.sync %0, 128;" ::"r"(1 + g) : "memory");
      if (wl == 0 && lane == 0) {
        __threadfence();
        if (atomicAdd(done + 32 + e, 1) == s_nd2[a] - 1) atomicAdd(done + 32 + e, 1 << 20);
      }
    }
#endif
  }
}

template <bool SA>
static int64_t tw3_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor w13h_, torch::Tensor s13h_, torch::Tensor w2h_, torch::Tensor s2h_,
                      torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor inter_,
                      torch::Tensor y_) {
  constexpr int NSTG = TW3_NSTG;
  constexpr int SMEM = TW3_NWG * NSTG * (TWC<4, true, 8>::XB + TWC<4, true, 8>::WB / 2 + TWC<4, true, 8>::SB / 2) + 1024;
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tw3<NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, moe_tw3<NSTG, SA>);
  TORCH_CHECK(fa.maxThreadsPerBlock >= 128 * TW3_NWG + 32, "moe_tw3: ", fa.numRegs, " regs allow only ", fa.maxThreadsPerBlock, " threads");
  return launch_k(moe_tw3<NSTG, SA>, grid, 128 * TW3_NWG + 32, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const char*)PTR(w13h_), (const char*)PTR(s13h_),
      (const char*)PTR(w2h_), (const char*)PTR(s2h_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr());
}
int64_t moe_tw3_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor w13h_, torch::Tensor s13h_, torch::Tensor w2h_,
                       torch::Tensor s2h_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_) {
#define TW3A w13_, s13_, gs13_, w2_, s2_, gs2_, w13h_, s13h_, w2h_, s2h_, lut_, xg_, meta_, sr_, inter_, y_
  return (g_sa & 1) ? tw3_go<true>(TW3A) : tw3_go<false>(TW3A);
#undef TW3A
}
