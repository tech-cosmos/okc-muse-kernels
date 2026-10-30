// n9s3 v2: STREAM-ONLY prototype, T=128 qkv 12-bit split-K, cluster 16 (K slices of 512), G clusters x NR rows.
// Changes vs sk128.cu (pull reduce 5 us/pass): consumers PUSH their pass partial (v2 st.async into the owner rank's
// recv buffer, mbarrier complete_tx), NRED reducer warps sum 16 slices + store BF16 while consumers go on streaming;
// x slice arrives as 8 TMA tensor boxes [64 k x 128 tok] SW128 (one mbarrier each), stage q of a pass waits box q only.
// build: nvcc -O3 -arch=sm_90a -std=c++17 -DGG=7 -DNRR=336 -o /tmp/sk128b sk128b.cu -lcuda
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_bf16.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#ifndef GG
#define GG 7
#endif
#ifndef NRR
#define NRR 336
#endif
#ifndef PF
#define PF 0
#endif
#ifndef NRED
#define NRED 4
#endif
constexpr int T = 128, N = 2304, K = 8192, S = 16, G = GG, KC = K / S, NR = NRR, NPASS = 3, PR = NR / NPASS, RO = PR / S;
constexpr int NXC = KC / 64;                 // 8 x boxes
constexpr int WB = NR * KC * 3 / 2;
#ifdef SMALLSM
constexpr int XB = 0;
#else
constexpr int XB = T * KC * 2;
#endif
#ifdef SMALLSM
constexpr int RB = 0;
#else
constexpr int RB = S * T * RO * 4;
#endif
           // recv buffer = one pass of partials
constexpr int NCT = 256, NTH = NCT + 32 + 32 * NRED;
static_assert(PR % S == 0 && RO % 1 == 0, "rows");

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t b, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(c)); }
__device__ __forceinline__ void mb_expect(uint32_t b, int tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mb_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory"); }
__device__ __forceinline__ void mb_wait_cl(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra W; }" :: "r"(b), "r"(ph) : "memory"); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, int r) { uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o; }
__device__ __forceinline__ void mb_arrive_remote(uint32_t ra) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(ra) : "memory"); }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory"); }
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(bar) : "memory"); }
__device__ __forceinline__ void st_async2(uint32_t ra, float a, float b, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               :: "r"(ra), "f"(a), "f"(b), "r"(rbar) : "memory"); }
__device__ __forceinline__ void st_async4(uint32_t ra, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(ra), "f"(a), "f"(b), "f"(c), "f"(d), "r"(rbar) : "memory"); }
__device__ __forceinline__ uint32_t crank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
#define STAMP(i) if (tl) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); tl[blockIdx.x * 16 + (i)] = t_; }

template <int SB, int NST>
__global__ void __launch_bounds__(NTH, 1) sk(const uint8_t* __restrict__ W, const __grid_constant__ CUtensorMap xm,
                                             __nv_bfloat16* __restrict__ out, unsigned long long* tl) {
  constexpr int NSTG = WB / SB, SPP = NSTG / NPASS;
  static_assert(WB % SB == 0 && NSTG % NPASS == 0 && (SPP == NXC || XB == 0), "stage = one 64-k chunk of a pass");
  extern __shared__ __align__(1024) uint8_t sm_[];
  uint8_t* sm = (uint8_t*)(((uintptr_t)sm_ + 1023) & ~(uintptr_t)1023);
  uint8_t* xs = sm; uint8_t* ring = sm + XB; float* recv = (float*)(ring + NST * SB);
  uint64_t* bars = (uint64_t*)((uint8_t*)recv + RB);   // full[NST], empty[NST], xb[NXC], rfull, freeall
  const uint32_t full0 = sa(bars), empty0 = sa(bars + NST), xb0 = sa(bars + 2 * NST), rfull = sa(bars + 2 * NST + NXC),
                 freeall = sa(bars + 2 * NST + NXC + 1);
  const int tid = threadIdx.x, rank = crank(), grp = blockIdx.x / S;
  if (tid == 0) {
    for (int i = 0; i < NST; i++) { mb_init(full0 + 8 * i, 1); mb_init(empty0 + 8 * i, NCT / 32); }
    for (int i = 0; i < NXC; i++) mb_init(xb0 + 8 * i, 1);
    mb_init(rfull, 1); mb_init(freeall, S);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync_all();
  if (tid >= NCT && tid < NCT + 32) {  // producer warp
    if (tid == NCT) {
      STAMP(0)
      const uint8_t* wp = W + (size_t)blockIdx.x * WB;
      int pre = NST < NSTG ? NST : NSTG;
      for (int s = 0; s < pre; s++) {
        
#ifdef NOW
        mb_arrive(full0 + 8 * s);
#else
        mb_expect(full0 + 8 * s, SB); bulk_g2s(sa(ring + s * SB), wp + (size_t)s * SB, SB, full0 + 8 * s);
#endif
#ifndef NOX
        if (s < NXC) { mb_expect(xb0 + 8 * s, T * 128); tma2d(sa(xs + s * T * 128), &xm, rank * KC + s * 64, 0, xb0 + 8 * s); }
#endif
      }
#if PF > 0
      for (int s = pre; s < pre + PF && s < NSTG; s++) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(wp + (size_t)s * SB), "r"(SB) : "memory");
#endif
#ifndef NOX
      for (int c = pre; c < NXC; c++) { mb_expect(xb0 + 8 * c, T * 128); tma2d(sa(xs + c * T * 128), &xm, rank * KC + c * 64, 0, xb0 + 8 * c); }
#endif
      for (int s = pre; s < NSTG; s++) {
        int sl = s % NST; mb_wait(empty0 + 8 * sl, ((s / NST) - 1) & 1);
        #ifdef NOW
        mb_arrive(full0 + 8 * sl);
#else
        mb_expect(full0 + 8 * sl, SB); bulk_g2s(sa(ring + sl * SB), wp + (size_t)s * SB, SB, full0 + 8 * sl);
#endif
#if PF > 0
        if (s + PF < NSTG) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(wp + (size_t)(s + PF) * SB), "r"(SB) : "memory");
#endif
      }
    }
    return;
  }
  if (tid >= NCT + 32) {  // reducer warps
#ifdef NOPUSH
    return;
#endif
    const int rt = tid - NCT - 32;
    for (int p = 0; p < NPASS; p++) {
      if (rt == 0) mb_expect(rfull, RB);
      mb_wait_cl(rfull, p & 1);
      if (rt == 0) STAMP(4 + 3 * p)
      // recv layout [src][r in RO][tok]; out element (tok, r)
      for (int e = rt; e < RO * T / 2; e += 32 * NRED) {
        float2 sum = make_float2(0.f, 0.f);
#pragma unroll
        for (int c = 0; c < S; c++) { float2 v = ((const float2*)(recv + c * RO * T))[e]; sum.x += v.x; sum.y += v.y; }
        int r = (2 * e) / T, tok = (2 * e) % T;
        int n = grp * NR + p * PR + rank * RO + r;
        if (n < N) { out[(size_t)tok * N + n] = __float2bfloat16(sum.x); out[(size_t)(tok + 1) * N + n] = __float2bfloat16(sum.y); }
      }
      asm volatile("bar.sync 2, %0;" :: "n"(32 * NRED) : "memory");
      if (rt == 0) STAMP(5 + 3 * p)
      if (rt < S) mb_arrive_remote(mapa(freeall, rt));
    }
    mb_wait_cl(freeall, (NPASS - 1) & 1);   // all peers done with my... (their arrivals to me have landed)
    if (rt == 0) STAMP(15)
    return;
  }
  // consumers
  float acc = 0.f;
  for (int p = 0; p < NPASS; p++) {
    for (int q = 0; q < SPP; q++) {
      int s = p * SPP + q, sl = s % NST;
      mb_wait(full0 + 8 * sl, (s / NST) & 1);
#ifndef NOX
      if (p == 0) mb_wait(xb0 + 8 * q, 0);
#endif
      if (p == 0 && q == 0 && tid == 0) STAMP(1)
      acc += __int_as_float(((const uint32_t*)(ring + sl * SB))[tid] & 0x3f800000u);  // stream-only: touch one word
      if (XB) acc += __bfloat162float(((const __nv_bfloat16*)(xs + q * T * 128))[tid]);
      __syncwarp();
      if ((tid & 31) == 0) mb_arrive(empty0 + 8 * sl);
    }
    if (tid == 0) STAMP(2 + 3 * p)
#ifdef NOPUSH
    if (tid == 0 && acc == 1234.f) out[0] = __float2bfloat16(acc);
    continue;
#endif
    if (p > 0) mb_wait_cl(freeall, (p - 1) & 1);           // every owner has freed its recv buffer
    // push the pass partial: PR x T floats, thread holds PR*T/NCT; pair (row, tok, tok+1) -> owner row/RO
    const uint32_t rbase = sa(recv) + rank * RO * T * 4;    // my slot [src = rank] in every owner's buffer
#ifdef V4
#pragma unroll 4
    for (int i = 0; i < PR * T / 4 / NCT; i++) {
      int e = i * NCT + tid, row = e / (T / 4), tok = 4 * (e % (T / 4));
      int ow = row / RO, r = row % RO;
      st_async4(mapa(rbase + (r * T + tok) * 4, ow), acc + i, acc - i, acc, acc * i, mapa(rfull, ow));
    }
#else
#pragma unroll 4
    for (int i = 0; i < PR * T / 2 / NCT; i++) {
      int e = i * NCT + tid, row = e / (T / 2), tok = 2 * (e % (T / 2));
      int ow = row / RO, r = row % RO;
      st_async2(mapa(rbase + (r * T + tok) * 4, ow), acc + i, acc - i, mapa(rfull, ow));
    }
#endif
    if (tid == 0) STAMP(3 + 3 * p)
  }
}

__global__ void flush(const float4* b, size_t n, float* o) { float s = 0; for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += b[i].x; if (s == 1234.5f) *o = s; }

template <int SB, int NST> void run(int reps, uint8_t* W, const CUtensorMap& xm, __nv_bfloat16* out, float4* fb, size_t fn, float* fo) {
  auto kf = sk<SB, NST>;
  int smem = XB + NST * SB + RB + 8 * (2 * NST + NXC + 2) + 1024;
  CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
  cfg.gridDim = dim3(S * G); cfg.blockDim = dim3(NTH); cfg.dynamicSmemBytes = smem;
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = getenv("CL") ? atoi(getenv("CL")) : S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int ncl = 0; CK(cudaOccupancyMaxActiveClusters(&ncl, kf, &cfg));
  cudaStream_t st; CK(cudaStreamCreate(&st)); cfg.stream = st;
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  CK(cudaLaunchKernelEx(&cfg, kf, (const uint8_t*)W, xm, out, (unsigned long long*)nullptr));
  CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
  int fd = open("/tmp/okc-gpu.lock", O_CREAT | O_RDWR, 0666); flock(fd, LOCK_EX);
  std::vector<float> ts;
  for (int r = 0; r < reps; r++) {
    flush<<<264, 1024, 0, st>>>(fb, fn, fo);
    cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms * 1000.f);
  }
  unsigned long long* tl; CK(cudaMalloc(&tl, S * G * 16 * 8)); CK(cudaMemset(tl, 0, S * G * 16 * 8));
  flush<<<264, 1024, 0, st>>>(fb, fn, fo);
  CK(cudaLaunchKernelEx(&cfg, kf, (const uint8_t*)W, xm, out, tl)); CK(cudaStreamSynchronize(st));
  flock(fd, LOCK_UN); close(fd);
  CK(cudaGetLastError());
  std::sort(ts.begin(), ts.end());
  printf("G %d NR %d SB %6d NST %d stages %3d smem %6d maxcl %d: median %.2f us (p10 %.2f p90 %.2f)\n", G, NR, SB, NST, WB / SB, smem, ncl,
         ts[reps / 2], ts[reps / 10], ts[reps * 9 / 10]);
  if (getenv("TL")) {
    std::vector<unsigned long long> h(S * G * 16); CK(cudaMemcpy(h.data(), tl, h.size() * 8, cudaMemcpyDeviceToHost));
    unsigned long long t0 = ~0ull; for (int b = 0; b < S * G; b++) t0 = std::min(t0, h[b * 16]);
    const char* nm[16] = {"prod start", "cons 1st stage", "p0 consumed", "p0 pushed", "red p0 got", "red p0 done", "p1 consumed", "p1 pushed",
                          "red p1 got", "red p1 done", "p2 consumed", "p2 pushed", "red p2 got", "red p2 done", "", "red exit"};
    for (int i = 0; i < 16; i++) { std::vector<double> v; for (int b = 0; b < S * G; b++) if (h[b * 16 + i]) v.push_back((h[b * 16 + i] - t0) / 1e3);
      if (v.empty()) continue; std::sort(v.begin(), v.end()); printf("  %-15s min %6.2f med %6.2f max %6.2f\n", nm[i], v[0], v[v.size() / 2], v.back()); }
  }
  cudaFree(tl);
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 50;
  uint8_t* W; __nv_bfloat16 *x, *out; float4* fb; float* fo;
  size_t wbytes = (size_t)WB * S * G, fn = (256u << 20) / 16;
  CK(cudaMalloc(&W, wbytes)); CK(cudaMalloc(&x, (size_t)T * K * 2)); CK(cudaMalloc(&out, (size_t)T * N * 2));
  CK(cudaMalloc(&fb, fn * 16)); CK(cudaMalloc(&fo, 4));
  CK(cudaMemset(W, 0x11, wbytes)); CK(cudaMemset(x, 0, (size_t)T * K * 2)); CK(cudaMemset(fb, 0, fn * 16));
  CUtensorMap xm; cuuint64_t dims[2] = {K, T}; cuuint64_t strides[1] = {K * 2}; cuuint32_t box[2] = {64, T}, es[2] = {1, 1};
  CUresult cr = cuTensorMapEncodeTiled(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (cr != CUDA_SUCCESS) { printf("tensormap %d\n", cr); return 1; }
  printf("W %.1f MB 12-bit; HBM at 3.24 TB/s %.2f us\n", wbytes / 1e6, wbytes / 3.24e6);
  constexpr int SB0 = PR * 64 * 3 / 2;
#ifdef SMALLSM
  run<SB0, 4>(reps, W, xm, out, fb, fn, fo); run<SB0, 6>(reps, W, xm, out, fb, fn, fo); run<SB0, 8>(reps, W, xm, out, fb, fn, fo);
  run<SB0, 12>(reps, W, xm, out, fb, fn, fo); run<SB0, 16>(reps, W, xm, out, fb, fn, fo); run<2 * SB0, 8>(reps, W, xm, out, fb, fn, fo);
  return 0;
#endif
  run<SB0, 2>(reps, W, xm, out, fb, fn, fo);
  run<SB0, 3>(reps, W, xm, out, fb, fn, fo);
  if (XB + 4 * SB0 + RB + 1200 <= 232448) run<SB0, 4>(reps, W, xm, out, fb, fn, fo);
  return 0;
}
