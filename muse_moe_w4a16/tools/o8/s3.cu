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

// o8: one thread-block cluster (CN <= 8 CTAs, one token per warp). The masks go to the leader CTA's smem over DSMEM,
// the leader runs B/B2 and publishes S, every warp re-picks its own token from the keys it still holds in registers
// (no reload trip), the new routes/experts are counted into the leader by DSMEM atomics. Same decisions and float
// ops as slack_kernel (bitwise equal outputs).
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
template <typename P> __device__ __forceinline__ P* lead_ptr(P* p) {  // generic address of *p in the leader CTA (rank 0)
  uint64_t g = (uint64_t)p, o;
  asm volatile("mapa.u64 %0, %1, 0;" : "=l"(o) : "l"(g));
  return (P*)o;
}

__global__ void __launch_bounds__(1024, 1) slack2_kernel(Args a) {
  __shared__ uint32_t sL[MAXT], sR[MAXT], sK[MAXT], sN[MAXT];
  __shared__ int sNeed[MAXT];
  __shared__ int cnt[LOC];
  __shared__ uint32_t gM, gU, gN, gS, gUse;
  __shared__ int gRef, gNr, gRoutes, gAny, gDone;
  __shared__ float gFd, gFr;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, T = a.T, nw = blockDim.x >> 5;
  const uint32_t rank = cl_rank();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (a.trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int ls = *a.ls;
  const int t = (int)rank * nw + warp;
  const bool have = t < T;
  float key[8], sg[8], v8 = 0.f; uint32_t in8m = 0, R = 0, L = 0, K = 0, N = 0; int myid = 0, nl = 0;
  // ---- A: one token per warp
  if (have) {
    v8 = token_keys(a, t, ls, lane, key, sg, in8m, myid);
    float v9 = -3.4e38f;
#pragma unroll
    for (int k = 0; k < 8; k++) if (!((in8m >> k) & 1u)) v9 = fmaxf(v9, key[k]);
#pragma unroll
    for (int o = 16; o; o >>= 1) v9 = fmaxf(v9, __shfl_xor_sync(0xffffffffu, v9, o));
    const float thr = v8 - a.rt;
    const bool in0 = in8m & 1u;
    L = __ballot_sync(0xffffffffu, key[0] >= thr);
    K = __ballot_sync(0xffffffffu, in0 && key[0] - v9 > a.kt);
    R = __ballot_sync(0xffffffffu, in0);
    N = __ballot_sync(0xffffffffu, !in0 && key[0] >= v8 - a.tie);
#pragma unroll
    for (int k = 1; k < 8; k++) nl += key[k] >= thr;
#pragma unroll
    for (int o = 16; o; o >>= 1) nl += __shfl_xor_sync(0xffffffffu, nl, o);
  }
  cl_sync();  // every CTA of the cluster has started (before any DSMEM access)
  if (have && lane == 0) {
    *lead_ptr(&sL[t]) = L; *lead_ptr(&sK[t]) = K; *lead_ptr(&sR[t]) = R; *lead_ptr(&sN[t]) = N;
    *lead_ptr(&sNeed[t]) = max(0, TOPK - nl);
  }
  cl_sync();  // #1: masks in the leader
  if (rank == 0) {
    if (threadIdx.x == 0) { gM = 0; gU = 0; gN = 0; gUse = 0; gRef = 0; gNr = 0; gRoutes = 0; }
    if (threadIdx.x < LOC) cnt[threadIdx.x] = 0;
    __syncthreads();
    {
      uint32_t m = 0, u = 0, n = 0; int r = 0, nr = 0;
      for (int q = threadIdx.x; q < T; q += blockDim.x) {
        m |= sK[q]; u |= sR[q]; n |= sN[q]; r += __popc(sR[q]); nr += __popc(sN[q]);
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
    // ---- B: greedy S (as slack_kernel)
    const float floor_d = (float)(__popc(gU | gN) + a.dmarg) * 0.85f - 2.f;
    const float floor_r = (float)(gRef + gNr + a.rmarg) * 0.92f - 4.f;
    if (threadIdx.x == 0) { gS = gM; gFd = floor_d; gFr = floor_r; }
    __syncthreads();
    for (int it = 0; it <= LOC; it++) {
      const uint32_t S = gS;
      if (threadIdx.x == 0) gAny = 0;
      __syncthreads();
      for (int q = threadIdx.x; q < T; q += blockDim.x) {
        const bool def = __popc(sL[q] & S) < sNeed[q];
        uint32_t b = def ? (sL[q] & ~S) : 0u;
        if (def) gAny = 1;
        while (b) { const int e = __ffs(b) - 1; b &= b - 1; atomicAdd(&cnt[e], 1024); }
        b = sR[q] & ~S;
        while (b) { const int e = __ffs(b) - 1; b &= b - 1; atomicAdd(&cnt[e], 1); }
      }
      __syncthreads();
      if (warp == 0) {
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
    // ---- B2: drop greedy additions that turned out redundant (highest index first)
    {
      uint32_t add = gS & ~gM;
      while (add) {
        const int e = 31 - __clz(add); add &= ~(1u << e);
        const uint32_t S1 = gS & ~(1u << e);
        if (threadIdx.x == 0) gAny = (float)__popc(S1) < floor_d;
        __syncthreads();
        for (int q = threadIdx.x; q < T; q += blockDim.x)
          if (__popc(sL[q] & S1) < sNeed[q]) gAny = 1;
        __syncthreads();
        if (threadIdx.x == 0 && !gAny) gS = S1;
        __syncthreads();
      }
    }
    // unchanged tokens' routes
    {
      const uint32_t S = gS;
      int r = 0; uint32_t use = 0;
      for (int q = threadIdx.x; q < T; q += blockDim.x) {
        const uint32_t Rq = sR[q];
        if (!(Rq & ~S)) { r += __popc(Rq); use |= Rq; }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) { r += __shfl_xor_sync(0xffffffffu, r, o); use |= __shfl_xor_sync(0xffffffffu, use, o); }
      if (lane == 0) { atomicAdd(&gRoutes, r); atomicOr(&gUse, use); }
    }
  }
  cl_sync();  // #2: S, floors, unchanged counts published
  const uint32_t S = *lead_ptr(&gS);
  const bool changed = have && (R & ~S);
  int id = myid; float wv = 0.f;
  if (changed) {  // ---- C: re-pick from the keys in registers
    const float thr = v8 - a.rt;
    uint64_t pk[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const bool ok = !((in8m >> k) & 1u) && key[k] >= thr && (k > 0 || ((S >> lane) & 1u));
      const int e = (ls + lane + 32 * k) & (E - 1);
      pk[k] = ok ? ((uint64_t)(k == 0) << 41) | ((uint64_t)okey(key[k]) << 8) | (uint64_t)(E - 1 - e) : 0ull;
    }
    const int rel = (myid - ls) & (E - 1);
    const bool dropped = lane < TOPK && rel < LOC && !((S >> rel) & 1u);
    const uint32_t dm = __ballot_sync(0xffffffffu, dropped);
    const int myrank = __popc(dm & ((1u << lane) - 1));
    for (int r = 0; r < __popc(dm); r++) {
      uint64_t b = 0; int bk = 0;
#pragma unroll
      for (int k = 0; k < 8; k++) if (pk[k] > b) { b = pk[k]; bk = k; }
      uint64_t w = b;
#pragma unroll
      for (int o = 16; o; o >>= 1) { const uint64_t x = __shfl_xor_sync(0xffffffffu, w, o); w = x > w ? x : w; }
      if (b == w) {
#pragma unroll
        for (int k = 0; k < 8; k++) if (k == bk) pk[k] = 0ull;
      }
      if (dropped && myrank == r) id = E - 1 - (int)(w & 0xFF);
    }
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
    for (int o = 4; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const int l2 = (id - ls) & (E - 1);
    const uint32_t lm = __ballot_sync(0xffffffffu, lane < TOPK && l2 < LOC);
    uint32_t um = 0;
#pragma unroll
    for (int s = 0; s < TOPK; s++) { const int x = __shfl_sync(0xffffffffu, l2, s); if (x < LOC) um |= 1u << x; }
    if (lane == 0) { atomicAdd(lead_ptr(&gRoutes), __popc(lm)); atomicOr(lead_ptr(&gUse), um); }
    wv = sc / (sum + a.reps);
  }
  cl_sync();  // #3: all counts in
  const int routes = *lead_ptr(&gRoutes); const uint32_t use = *lead_ptr(&gUse);
  const float fd = *lead_ptr(&gFd), fr = *lead_ptr(&gFr);
  cl_sync();  // #4: the leader's smem stays alive until every CTA has read it
  if (changed && lane < TOPK && !((float)routes < fr || (float)__popc(use) < fd)) {
    const int l = (id - ls) & (E - 1);
    a.ids[t * TOPK + lane] = id; a.lids[t * TOPK + lane] = l < LOC ? l : -1; a.wts[t * TOPK + lane] = wv;
  }
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
  const int cn = T < 8 ? (int)T : 8, nwc = ((int)T + cn - 1) / cn;
  cfg.gridDim = dim3(cn); cfg.blockDim = dim3(nwc * 32);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension; attr[1].val.clusterDim.x = cn; attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 2;
  cudaLaunchKernelEx(&cfg, slk::slack2_kernel, a);
}
