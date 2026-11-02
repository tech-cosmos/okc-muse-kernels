// layer n14 (moe/slack, NOT for submission until the organizers approve): moe's slack_kernel rule (slack_src.cu) as a
// device function for small T, run by the block that just routed (pr8's finish block, T = 8): no extra launch.
// Same decisions as slack_kernel: A per token (warp t), B greedy S + B2 redundancy pass in warp 0 (lane = local
// expert, no block barriers inside), C per token picks, then the floor check and the in-place overwrite.
// lgs: logits [T][256] (shared), bs: bias [256] (shared); ids/lids/wts [T][8] global, already written by this block.
// SLK8_RT/KT/TIE/DM/RM: margins (defined by layer.py). Needs blockDim >= 32 * T threads, T <= MT, >= 1.2 KB dynamic smem.
namespace slk8 {
constexpr int E = 256, TOPK = 8, LOC = 32, MT = 8;

__device__ __forceinline__ uint32_t okey(float k) {
  uint32_t b = __float_as_uint(k);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __noinline__ void select(const float* lgs, const float* bs, int ls, int T, int* ids, int* lids, float* wts,
                                       float mult, float reps) {
  // scratch in the caller's DYNAMIC smem (unused by the calling block): static smem would grow the whole kernel's
  // footprint (pr8: +640 B static = +4 us at v1 on the H100 even when never executed)
  extern __shared__ __align__(16) unsigned char slk8_dyn[];
  uint32_t* sL = (uint32_t*)slk8_dyn; uint32_t* sK = sL + MT; uint32_t* sR = sK + MT; uint32_t* sN = sR + MT;
  uint32_t* sSp = sN + MT;
  int* sNeed = (int*)(sSp + 4); int (*sNew)[TOPK] = (int (*)[TOPK])(sNeed + MT); int* sChg = sNeed + MT + MT * TOPK;
  int* sLm = sChg + MT; float (*sW)[TOPK] = (float (*)[TOPK])(sLm + MT);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();  // ids/lids/wts of every token written (global, this block)
  // ---- A: per token (warp), lane j holds experts (ls + j + 32 k) mod 256; k = 0 is local expert j
  if (warp < T) {
    const int t = warp;
    const int myid = lane < TOPK ? __ldcg(ids + t * TOPK + lane) : -1;
    float key[8]; bool in8[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int e = (ls + lane + 32 * k) & (E - 1);
      key[k] = 1.f / (1.f + __expf(-lgs[t * E + e] * mult)) + bs[e];
      in8[k] = false;
    }
#pragma unroll
    for (int s = 0; s < TOPK; s++) {
      const int id = __shfl_sync(0xffffffffu, myid, s);
      const int k = ((id - ls - lane) & (E - 1));
#pragma unroll
      for (int kk = 0; kk < 8; kk++) in8[kk] |= (k == 32 * kk);
    }
    float v8 = 3.4e38f, v9 = -3.4e38f;
#pragma unroll
    for (int k = 0; k < 8; k++) { if (in8[k]) v8 = fminf(v8, key[k]); else v9 = fmaxf(v9, key[k]); }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      v8 = fminf(v8, __shfl_xor_sync(0xffffffffu, v8, o));
      v9 = fmaxf(v9, __shfl_xor_sync(0xffffffffu, v9, o));
    }
    const float thr = v8 - SLK8_RT;
    const uint32_t L = __ballot_sync(0xffffffffu, key[0] >= thr);
    const uint32_t K = __ballot_sync(0xffffffffu, in8[0] && key[0] - v9 > SLK8_KT);
    const uint32_t R = __ballot_sync(0xffffffffu, in8[0]);
    const uint32_t N = __ballot_sync(0xffffffffu, !in8[0] && key[0] >= v8 - SLK8_TIE);
    int nl = 0;
#pragma unroll
    for (int k = 1; k < 8; k++) nl += __popc(__ballot_sync(0xffffffffu, key[k] >= thr));
    if (lane == 0) { sL[t] = L; sK[t] = K; sR[t] = R; sN[t] = N; sNeed[t] = max(0, TOPK - nl); }
  }
  __syncthreads();
  // ---- B + B2 (warp 0): greedy S, then drop greedy additions that turned out redundant (highest index first)
  if (warp == 0) {
    uint32_t gM = 0, gU = 0, gN = 0; int gRef = 0, gNr = 0;
    for (int t = 0; t < T; t++) {
      gM |= sK[t]; gU |= sR[t]; gN |= sN[t]; gRef += __popc(sR[t]); gNr += __popc(sN[t]);
    }
    const float floor_d = (float)(__popc(gU | gN) + SLK8_DM) * 0.85f - 2.f;
    uint32_t S = gM;
    for (int it = 0; it <= LOC; it++) {
      bool any = false; int cnt = 0;
      for (int t = 0; t < T; t++) {
        const bool def = __popc(sL[t] & S) < sNeed[t];
        any |= def;
        if (def && ((sL[t] & ~S) >> lane & 1u)) cnt += 1024;
        if ((sR[t] & ~S) >> lane & 1u) cnt += 1;
      }
      if (!(any || (float)__popc(S) < floor_d)) break;
      int bc = cnt;
#pragma unroll
      for (int o = 16; o; o >>= 1) bc = max(bc, __shfl_xor_sync(0xffffffffu, bc, o));
      if (bc <= 0) break;
      const int be = __ffs(__ballot_sync(0xffffffffu, cnt == bc)) - 1;  // lowest index on ties
      S |= 1u << be;
    }
    for (int e = LOC - 1; e >= 0; e--) {
      if (!((S & ~gM) >> e & 1u)) continue;
      const uint32_t S1 = S & ~(1u << e);
      bool bad = (float)__popc(S1) < floor_d;
      for (int t = 0; t < T; t++) bad |= __popc(sL[t] & S1) < sNeed[t];
      if (!bad) S = S1;
    }
    if (lane == 0) *sSp = S;
  }
  __syncthreads();
  // ---- C: per token picks
  const uint32_t S = *sSp;
  if (warp < T) {
    const int t = warp;
    const bool chg = (sR[t] & ~S) != 0u;
    const int myid = lane < TOPK ? __ldcg(ids + t * TOPK + lane) : -1;
    if (!chg) {
      if (lane == 0) { sChg[t] = 0; sLm[t] = (int)sR[t]; }
    } else {
      float key[8]; bool in8[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const int e = (ls + lane + 32 * k) & (E - 1);
        key[k] = 1.f / (1.f + __expf(-lgs[t * E + e] * mult)) + bs[e];
        in8[k] = false;
      }
#pragma unroll
      for (int s = 0; s < TOPK; s++) {
        const int id = __shfl_sync(0xffffffffu, myid, s);
        const int k = ((id - ls - lane) & (E - 1));
#pragma unroll
        for (int kk = 0; kk < 8; kk++) in8[kk] |= (k == 32 * kk);
      }
      float v8 = 3.4e38f;
#pragma unroll
      for (int k = 0; k < 8; k++) if (in8[k]) v8 = fminf(v8, key[k]);
#pragma unroll
      for (int o = 16; o; o >>= 1) v8 = fminf(v8, __shfl_xor_sync(0xffffffffu, v8, o));
      const float thr = v8 - SLK8_RT;
      uint64_t pk[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const bool ok = !in8[k] && key[k] >= thr && (k > 0 || ((S >> lane) & 1u));
        const int e = (ls + lane + 32 * k) & (E - 1);
        pk[k] = ok ? ((uint64_t)(k == 0) << 41) | ((uint64_t)okey(key[k]) << 8) | (uint64_t)(E - 1 - e) : 0ull;
      }
      const uint32_t drop = sR[t] & ~S;
      const int nd = __popc(drop);
      // lane s < 8 holds slot s; the r-th dropped slot (slot order) takes the r-th best candidate (no local arrays:
      // a stack frame in pr8 costs ~+4 us at v1 even when never executed)
      const int myl = (myid - ls) & (E - 1);
      const bool dr = lane < TOPK && myl < LOC && ((drop >> myl) & 1u);
      const int rank = __popc(__ballot_sync(0xffffffffu, dr) & ((1u << lane) - 1u));
      int nid = myid;
      for (int r = 0; r < nd; r++) {
        uint64_t b = 0; int bk = 0;
#pragma unroll
        for (int k = 0; k < 8; k++) if (pk[k] > b) { b = pk[k]; bk = k; }
        uint64_t w = b;
#pragma unroll
        for (int o = 16; o; o >>= 1) { const uint64_t x = __shfl_xor_sync(0xffffffffu, w, o); w = x > w ? x : w; }
        if (dr && rank == r) nid = E - 1 - (int)(w & 0xFF);
        if (b == w && w != 0ull) {
#pragma unroll
          for (int k = 0; k < 8; k++) if (k == bk) pk[k] = 0ull;
        }
      }
      const float sc = lane < TOPK ? 1.f / (1.f + expf(-lgs[t * E + nid] * mult)) : 0.f;
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < TOPK; j++) sum += __shfl_sync(0xffffffffu, sc, j);
      const int l2 = (nid - ls) & (E - 1);
      uint32_t lmask = 0;
#pragma unroll
      for (int j = 0; j < TOPK; j++) {
        const int lj = __shfl_sync(0xffffffffu, l2, j);
        if (lj < LOC) lmask |= 1u << lj;
      }
      if (lane < TOPK) { sNew[t][lane] = nid; sW[t][lane] = sc / (sum + reps); }
      if (lane == 0) { sChg[t] = 1; sLm[t] = (int)lmask; }
    }
  }
  __syncthreads();
  int routes = 0; uint32_t use = 0, gU = 0, gN = 0; int gRef = 0, gNr = 0;
  for (int t = 0; t < T; t++) {
    routes += __popc((uint32_t)sLm[t]); use |= (uint32_t)sLm[t];
    gU |= sR[t]; gN |= sN[t]; gRef += __popc(sR[t]); gNr += __popc(sN[t]);
  }
  const float floor_d = (float)(__popc(gU | gN) + SLK8_DM) * 0.85f - 2.f;
  const float floor_r = (float)(gRef + gNr + SLK8_RM) * 0.92f - 4.f;
  if (!((float)routes >= floor_r && (float)__popc(use) >= floor_d)) return;
  for (int i = threadIdx.x; i < T * TOPK; i += blockDim.x) {
    const int t = i / TOPK, s = i % TOPK;
    if (!sChg[t]) continue;
    const int id = sNew[t][s];
    const int l = (id - ls) & (E - 1);
    ids[i] = id; lids[i] = l < LOC ? l : -1; wts[i] = sW[t][s];
  }
}
}  // namespace slk8
