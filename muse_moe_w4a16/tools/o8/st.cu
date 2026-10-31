// moe/slack (moe s5 2026-10-31): ROUTING-SLACK selection step. NOT for submission until the organizers approve it.
// After the normal routing (ids = biased top 8, lowest index on ties), choose a smaller set S of this GPU's experts and
// re-route each token inside TASK.md's routing tolerance (reference.check_routing), so fewer local experts are read:
//   - every pick scores (sigmoid + bias) >= its 8th best - RT (RT = ROUTE_TOL minus a margin)
//   - every local top-8 expert beating the 9th best by > KT (KEEP_TOL minus a margin) is kept (S starts as their union)
//   - S grows greedily (most deficit tokens covered, then most top-8 routes; lowest index on ties) until every token can
//     fill 8 picks from its local candidates in S plus its non-local candidates, and |S| >= the distinct floor; then
//     greedy additions that became redundant are dropped again (= fresh x8's exact minimum on 11/12 scored loads)
//   - a token keeps its top 8 where allowed; a dropped local expert is replaced by its best local candidate in S, else
//     its best non-local candidate (key order, lowest index on ties)
//   - if the local routes or distinct experts fall below the floors (with margins), the routing is left unchanged.
// Grid ceil(T / 8) CTAs x 8 warps, warp = token (phase A: per-token masks to global); the last CTA (self-resetting
// ticket) runs the greedy (B) and re-picks the changed tokens only (C). Deterministic (integer/bitmask decisions from
// the same FP32 keys), no host sync: CUDA-graph safe.
// logits[t][e] = sum_k part[k * SKS + t * ST + e]; ids/lids/wts [T][8] are read and overwritten in place.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace slk {
constexpr int E = 256, TOPK = 8, LOC = 32, NW = 8, MAXT = 256;

struct Args {
  const float* part; int ks, sks, st;
  const float* bias; const int* ls;
  int* ids; int* lids; float* wts;
  uint32_t* msk;  // [5][MAXT]: L, K, R, N, need
  int* cnt;       // ticket (zero, self-resetting)
  int T; float mult, reps, rt, kt, tie, dmarg, rmarg;
  int trig;  // trigger dependents at once (= the routing kernel's TRIG at this site)
};

__device__ __forceinline__ uint32_t okey(float k) {  // order-preserving bits
  uint32_t b = __float_as_uint(k);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// lane j holds experts (ls + j + 32 k) mod 256; k = 0 is local expert j. Returns v8 (the 8th best key); sg[k] = the
// routing score (sigmoid, as the routing kernel computes it) for the weights. All global loads are issued before any use
// (one round trip: the selection is latency-bound, o6).
__device__ __forceinline__ float token_keys(const Args& a, int t, int ls, int lane, float (&key)[8], float (&sg)[8],
                                            uint32_t& in8m, int& myid_out) {
  const int myid = lane < TOPK ? __ldcg(a.ids + t * TOPK + lane) : 0;
  float l0[8], l1[8], bb[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int e = (ls + lane + 32 * k) & (E - 1);
    l0[k] = __ldcg(a.part + (size_t)t * a.st + e);
    l1[k] = a.ks > 1 ? __ldcg(a.part + (size_t)a.sks + (size_t)t * a.st + e) : 0.f;
    bb[k] = __ldg(a.bias + e);
  }
#pragma unroll
  for (int k = 0; k < 8; k++) if (a.ks > 1) l0[k] += l1[k];
  for (int q = 2; q < a.ks; q++) {  // split-K routers at small T (ks 8/16): one round trip per partial
#pragma unroll
    for (int k = 0; k < 8; k++) l1[k] = __ldcg(a.part + (size_t)q * a.sks + (size_t)t * a.st + ((ls + lane + 32 * k) & (E - 1)));
#pragma unroll
    for (int k = 0; k < 8; k++) l0[k] += l1[k];
  }
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const float lg = l0[k];
    key[k] = 1.f / (1.f + __expf(-lg * a.mult)) + bb[k];
    sg[k] = 1.f / (1.f + expf(-lg * a.mult));
  }
  // in8m bit k: expert k of this lane is in the top 8. id's lane = (id - ls) & 31, its k = ((id - ls) & 255) >> 5
  myid_out = myid;
  const int rel = (myid - ls) & (E - 1);
  in8m = 0;
#pragma unroll
  for (int s = 0; s < TOPK; s++) {
    const int r = __shfl_sync(0xffffffffu, rel, s);
    if ((r & 31) == lane) in8m |= 1u << (r >> 5);
  }
  float v8 = 3.4e38f;
#pragma unroll
  for (int k = 0; k < 8; k++) if ((in8m >> k) & 1u) v8 = fminf(v8, key[k]);
#pragma unroll
  for (int o = 16; o; o >>= 1) v8 = fminf(v8, __shfl_xor_sync(0xffffffffu, v8, o));
  return v8;
}

__global__ void __launch_bounds__(NW * 32) slack_kernel(Args a) {
  __shared__ uint32_t sL[MAXT], sR[MAXT];
  __shared__ int sNeed[MAXT], sList[MAXT];
  __shared__ int cnt[LOC];
  __shared__ uint32_t gM, gU, gN, gS, gUse;
  __shared__ int gRef, gNr, gRoutes, gAny, gDone, gLast, gNc;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, T = a.T;
  unsigned long long* ST = (unsigned long long*)(a.cnt + 8) + 16 * (blockIdx.x == 0 ? 0 : 1);
  unsigned long long _t0; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t0));
  // PDL secondary of the routing kernel; the experts wait for our completion (griddepcontrol.wait = full grid done)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (a.trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if(threadIdx.x==0) ST[0]=_t0;
  const int ls = *a.ls;
  // ---- A: one token per warp
  {
    const int t = blockIdx.x * NW + warp;
    if (t < T) {
      float key[8], sg[8]; uint32_t in8m; int myid;
      const float v8 = token_keys(a, t, ls, lane, key, sg, in8m, myid);
      float v9 = -3.4e38f;
#pragma unroll
      for (int k = 0; k < 8; k++) if (!((in8m >> k) & 1u)) v9 = fmaxf(v9, key[k]);
#pragma unroll
      for (int o = 16; o; o >>= 1) v9 = fmaxf(v9, __shfl_xor_sync(0xffffffffu, v9, o));
      const float thr = v8 - a.rt;
      const bool in0 = in8m & 1u;
      const uint32_t L = __ballot_sync(0xffffffffu, key[0] >= thr);
      const uint32_t K = __ballot_sync(0xffffffffu, in0 && key[0] - v9 > a.kt);
      const uint32_t R = __ballot_sync(0xffffffffu, in0);
      // local experts just outside the top 8 (within TIE of the 8th): the reference's top 8 may hold them instead
      const uint32_t N = __ballot_sync(0xffffffffu, !in0 && key[0] >= v8 - a.tie);
      int nl = 0;
#pragma unroll
      for (int k = 1; k < 8; k++) nl += key[k] >= thr;
#pragma unroll
      for (int o = 16; o; o >>= 1) nl += __shfl_xor_sync(0xffffffffu, nl, o);
      if (lane < 5) {
        const uint32_t v = lane == 0 ? L : lane == 1 ? K : lane == 2 ? R : lane == 3 ? N : (uint32_t)max(0, TOPK - nl);
        __stcg(a.msk + lane * MAXT + t, v);
      }
    }
  }
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[1]=_t;}
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(a.cnt, 1);
    gLast = prev == (int)gridDim.x - 1;
    if (gLast) *a.cnt = 0;  // self-reset for the next launch / replay
  }
  __syncthreads();
  if (!gLast) return;
  ST = (unsigned long long*)(a.cnt + 8) + 32; if(threadIdx.x==0) ST[0]=_t0;
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[1]=_t;}
  __threadfence();
  if (threadIdx.x == 0) { gM = 0; gU = 0; gN = 0; gUse = 0; gRef = 0; gNr = 0; gRoutes = 0; gNc = 0; }
  if (threadIdx.x < LOC) cnt[threadIdx.x] = 0;
  __syncthreads();
  {
    uint32_t m = 0, u = 0, n = 0; int r = 0, nr = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
      const uint32_t L = __ldcg(a.msk + t), K = __ldcg(a.msk + MAXT + t), R = __ldcg(a.msk + 2 * MAXT + t),
                     N = __ldcg(a.msk + 3 * MAXT + t);
      sL[t] = L; sR[t] = R; sNeed[t] = (int)__ldcg(a.msk + 4 * MAXT + t);
      m |= K; u |= R; n |= N; r += __popc(R); nr += __popc(N);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      m |= __shfl_xor_sync(0xffffffffu, m, o); u |= __shfl_xor_sync(0xffffffffu, u, o);
      n |= __shfl_xor_sync(0xffffffffu, n, o); r += __shfl_xor_sync(0xffffffffu, r, o);
      nr += __shfl_xor_sync(0xffffffffu, nr, o);
    }
    if (lane == 0) { atomicOr(&gM, m); atomicOr(&gU, u); atomicOr(&gN, n); atomicAdd(&gRef, r); atomicAdd(&gNr, nr); }
  }
  __syncthreads();
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[2]=_t;}
  // ---- B: greedy S. Floors from the reference's counts, at their largest over near-tie flips (+ fixed margins)
  const float floor_d = (float)(__popc(gU | gN) + a.dmarg) * 0.85f - 2.f;
  const float floor_r = (float)(gRef + gNr + a.rmarg) * 0.92f - 4.f;
  if (threadIdx.x == 0) gS = gM;
  __syncthreads();
  for (int it = 0; it <= LOC; it++) {
    const uint32_t S = gS;
    if (threadIdx.x == 0) gAny = 0;
    __syncthreads();
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
      const bool def = __popc(sL[t] & S) < sNeed[t];
      uint32_t b = def ? (sL[t] & ~S) : 0u;
      if (def) gAny = 1;
      while (b) { const int e = __ffs(b) - 1; b &= b - 1; atomicAdd(&cnt[e], 1024); }
      b = sR[t] & ~S;
      while (b) { const int e = __ffs(b) - 1; b &= b - 1; atomicAdd(&cnt[e], 1); }
    }
    __syncthreads();
    if (warp == 0) {  // argmax over the 32 counters (lowest index on ties)
      const int c = cnt[lane];
      int bv = c * 64 + (63 - lane);
#pragma unroll
      for (int o = 16; o; o >>= 1) bv = max(bv, __shfl_xor_sync(0xffffffffu, bv, o));
      cnt[lane] = 0;
      if (lane == 0) {
        gDone = !(gAny || (float)__popc(S) < floor_d);
        if (!gDone) { if (bv < 64) gDone = 1; else gS = S | (1u << (63 - (bv & 63))); }
      }
    }
    __syncthreads();
    if (gDone) break;
  }
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[3]=_t;}
  if(threadIdx.x==0) ST[10]=__popc(gS);
  // ---- B2: drop greedy additions that turned out redundant (highest index first)
  {
    uint32_t add = gS & ~gM;
    while (add) {
      const int e = 31 - __clz(add); add &= ~(1u << e);
      const uint32_t S1 = gS & ~(1u << e);
      if (threadIdx.x == 0) gAny = (float)__popc(S1) < floor_d;
      __syncthreads();
      for (int t = threadIdx.x; t < T; t += blockDim.x)
        if (__popc(sL[t] & S1) < sNeed[t]) gAny = 1;
      __syncthreads();
      if (threadIdx.x == 0 && !gAny) gS = S1;
      __syncthreads();
    }
  }
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[4]=_t;}
  // ---- C: changed tokens (a top-8 local expert outside S) re-pick; the others keep their routes
  const uint32_t S = gS;
  {
    int r = 0; uint32_t use = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
      const uint32_t R = sR[t];
      if (R & ~S) sList[atomicAdd(&gNc, 1)] = t;
      else { r += __popc(R); use |= R; }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) { r += __shfl_xor_sync(0xffffffffu, r, o); use |= __shfl_xor_sync(0xffffffffu, use, o); }
    if (lane == 0) { atomicAdd(&gRoutes, r); atomicOr(&gUse, use); }
  }
  __syncthreads();
  const int nc = gNc;
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[5]=_t;}
  if(threadIdx.x==0) ST[11]=nc;
  // new picks in registers of lanes < 8 (slot = lane), at most MAXT / NW tokens per warp
  constexpr int PER = MAXT / NW;
  int nid[PER]; float nw[PER];
  int nmine = 0;
  for (int i = warp; i < nc; i += NW, nmine++) {
    const int t = sList[i];
    float key[8], sg[8]; uint32_t in8m; int myid;
    const float v8 = token_keys(a, t, ls, lane, key, sg, in8m, myid);
    const float thr = v8 - a.rt;
    uint64_t pk[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const bool ok = !((in8m >> k) & 1u) && key[k] >= thr && (k > 0 || ((S >> lane) & 1u));
      const int e = (ls + lane + 32 * k) & (E - 1);
      pk[k] = ok ? ((uint64_t)(k == 0) << 41) | ((uint64_t)okey(key[k]) << 8) | (uint64_t)(E - 1 - e) : 0ull;
    }
    // slot s (lane s < 8) dropped if its id is local and outside S; the r-th dropped slot gets the r-th best candidate
    const int rel = (myid - ls) & (E - 1);
    const bool dropped = lane < TOPK && rel < LOC && !((S >> rel) & 1u);
    const uint32_t dm = __ballot_sync(0xffffffffu, dropped);
    const int myrank = __popc(dm & ((1u << lane) - 1));
    int id = myid;
    for (int r = 0; r < __popc(dm); r++) {
      uint64_t b = 0; int bk = 0;
#pragma unroll
      for (int k = 0; k < 8; k++) if (pk[k] > b) { b = pk[k]; bk = k; }
      uint64_t w = b;
#pragma unroll
      for (int o = 16; o; o >>= 1) { const uint64_t x = __shfl_xor_sync(0xffffffffu, w, o); w = x > w ? x : w; }
      if (b == w) {  // the owner removes it (ids unique in the low byte, so one owner)
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == bk) pk[k] = 0ull;
      }
      if (dropped && myrank == r) id = E - 1 - (int)(w & 0xFF);
    }
    // the pick's score from the lane holding it (no reload: lane (id - ls) & 31, slot ((id - ls) & 255) >> 5)
    const int prel = (id - ls) & (E - 1);
    float sc = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float v = __shfl_sync(0xffffffffu, sg[k], prel & 31);
      if ((prel >> 5) == k) sc = v;
    }
    if (lane >= TOPK) sc = 0.f;
    float sum = sc;
#pragma unroll
    for (int o = 4; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);  // lanes 0..7
    const int l2 = (id - ls) & (E - 1);
    const uint32_t lm = __ballot_sync(0xffffffffu, lane < TOPK && l2 < LOC);
    uint32_t um = 0;
#pragma unroll
    for (int s = 0; s < TOPK; s++) { const int x = __shfl_sync(0xffffffffu, l2, s); if (x < LOC) um |= 1u << x; }
    if (lane == 0) { atomicAdd(&gRoutes, __popc(lm)); atomicOr(&gUse, um); }
#pragma unroll
    for (int j = 0; j < PER; j++) if (j == nmine) { nid[j] = id; nw[j] = sc / (sum + a.reps); }
  }
  __syncthreads();
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[6]=_t;}
  if ((float)gRoutes < floor_r || (float)__popc(gUse) < floor_d) return;  // floors missed: routing unchanged
  int j = 0;
  for (int i = warp; i < nc; i += NW, j++) {
    if (lane >= TOPK) continue;
    const int t = sList[i];
    int id = 0; float w = 0.f;
#pragma unroll
    for (int q = 0; q < PER; q++) if (q == j) { id = nid[q]; w = nw[q]; }
    const int l = (id - ls) & (E - 1);
    a.ids[t * TOPK + lane] = id; a.lids[t * TOPK + lane] = l < LOC ? l : -1; a.wts[t * TOPK + lane] = w;
  }
  __syncthreads();
  {unsigned long long _t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(_t)); if(threadIdx.x==0) ST[7]=_t;}
}
}  // namespace slk

void slack_launch(torch::Tensor part, int64_t ks, int64_t sks, int64_t st, torch::Tensor bias, torch::Tensor ls,
                  torch::Tensor ids, torch::Tensor lids, torch::Tensor wts, torch::Tensor scratch, int64_t T,
                  double mult, double reps, double rt, double kt, double tie, double dmarg, double rmarg, int64_t trig) {
  TORCH_CHECK(T >= 1 && T <= slk::MAXT, "slack: T out of range");
  TORCH_CHECK(ks >= 1, "slack: ks >= 1");
  // scratch int32 [>= 5 * 256 + 1]: masks, then the ticket (zero at first use; the kernel resets it)
  slk::Args a{part.data_ptr<float>(), (int)ks, (int)sks, (int)st, bias.data_ptr<float>(), ls.data_ptr<int>(),
              ids.data_ptr<int>(), lids.data_ptr<int>(), wts.data_ptr<float>(), (uint32_t*)scratch.data_ptr<int>(),
              scratch.data_ptr<int>() + 5 * slk::MAXT, (int)T, (float)mult, (float)reps, (float)rt, (float)kt,
              (float)tie, (float)dmarg, (float)rmarg, (int)trig};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((T + slk::NW - 1) / slk::NW); cfg.blockDim = dim3(slk::NW * 32);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, slk::slack_kernel, a);
}
