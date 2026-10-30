// n9s3: STREAM-ONLY prototype of a T=128 qkv 12-bit split-K GEMM (lead 02:30Z option (a)).
// Cluster of S=16 CTAs = 16 K slices (Kc 512) of one N group (G=8 groups x 288 rows). x slice [128 x 512] BF16 = 128 KB
// resident in smem (loaded once); W bytes (12-bit = 1.5 B/weight; 221184 B per CTA, contiguous) stream through a TMA ring;
// 3 passes of 96 rows: consumers "accumulate" (stream-only: touch one word per stage), write a 96x128 FP32 partial
// to smem, cluster reduce-scatter over DSMEM (mbarrier flags, no global round trips), BF16 store.
// build: nvcc -O3 -arch=sm_90a -std=c++17 -o /tmp/sk128 sk128.cu ; run: /tmp/sk128 [SB] [NST] [reps]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_bf16.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#ifndef GG
#define GG 8
#endif
constexpr int T = 128, N = 2304, K = 8192, S = 16, G = GG, KC = K / S, NR = 288, NPASS = 3, PR = NR / NPASS;  // PR 96
constexpr int WB = NR * KC * 3 / 2;          // 221184 B per CTA
constexpr int XB = T * KC * 2;               // 131072 B
constexpr int PB = PR * T * 4;               // 49152 B partial
constexpr int NCT = 256;                      // consumer threads (2 WGs)
constexpr int NTH = NCT + 32;                 // + producer warp

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
__device__ __forceinline__ float2 ld_dsm2(uint32_t a) { float2 v; asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t crank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

template <int SB, int NST>
__global__ void __launch_bounds__(NTH, 1) sk128(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, unsigned long long* tl) {
  constexpr int NSTG = WB / SB, SPP = NSTG / NPASS;
  static_assert(WB % SB == 0 && NSTG % NPASS == 0, "stage split");
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* xs = sm; uint8_t* ring = sm + XB; float* part = (float*)(ring + NST * SB);
  uint64_t* bars = (uint64_t*)((uint8_t*)part + PB);   // full[NST], empty[NST], xbar, ready, consumed
  const uint32_t full0 = sa(bars), empty0 = sa(bars + NST), xbar = sa(bars + 2 * NST), ready = sa(bars + 2 * NST + 1),
                 consumed = sa(bars + 2 * NST + 2);
  const int tid = threadIdx.x, rank = crank(), grp = blockIdx.x / S;
  if (tid == 0) {
    for (int i = 0; i < NST; i++) { mb_init(full0 + 8 * i, 1); mb_init(empty0 + 8 * i, NCT / 32); }
    mb_init(xbar, 1); mb_init(ready, S); mb_init(consumed, S);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync_all();
  if (tid >= NCT) {  // producer warp
    if (tid == NCT) {
      const uint8_t* wp = W + (size_t)blockIdx.x * WB;
      int pre = NST < NSTG ? NST : NSTG;
      for (int s = 0; s < pre; s++) { mb_expect(full0 + 8 * s, SB); bulk_g2s(sa(ring + s * SB), wp + (size_t)s * SB, SB, full0 + 8 * s); }
      mb_expect(xbar, XB);
      for (int t = 0; t < T; t++) bulk_g2s(sa(xs + t * KC * 2), x + (size_t)t * K + rank * KC, KC * 2, xbar);
      for (int s = pre; s < NSTG; s++) {
        int sl = s % NST; mb_wait(empty0 + 8 * sl, ((s / NST) - 1) & 1);
        mb_expect(full0 + 8 * sl, SB); bulk_g2s(sa(ring + sl * SB), wp + (size_t)s * SB, SB, full0 + 8 * sl);
      }
    }
    return;
  }
  // consumers
#define STAMP(i) if (tid == 0 && tl) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); tl[blockIdx.x * 16 + (i)] = t_; }
  STAMP(0)
  float acc = 0.f;
  bool xw = false;
  for (int p = 0; p < NPASS; p++) {
    for (int q = 0; q < SPP; q++) {
      int s = p * SPP + q, sl = s % NST;
      mb_wait(full0 + 8 * sl, (s / NST) & 1);
      if (!xw) { STAMP(1) mb_wait(xbar, 0); xw = true; STAMP(2) }
      acc += __int_as_float(((const uint32_t*)(ring + sl * SB))[tid] & 0x3f800000u);  // stream-only: touch one word
      __syncwarp();
      if ((tid & 31) == 0) mb_arrive(empty0 + 8 * sl);
    }
    STAMP(3 + 4 * p)
    if (p > 0) mb_wait_cl(consumed, (p - 1) & 1);           // peers finished reading my previous partial
    // fake partial (48 floats/thread): depends on acc and on resident x
    float xv = __bfloat162float(((const __nv_bfloat16*)xs)[tid * 64 + p]);
#pragma unroll
    for (int i = 0; i < PB / 4 / NCT; i++) part[i * NCT + tid] = acc + xv * (float)i;
    asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
    if (tid < S) mb_arrive_remote(mapa(ready, tid));
    STAMP(4 + 4 * p)
    mb_wait_cl(ready, p & 1);
    STAMP(5 + 4 * p)
    // reduce-scatter: rank owns rows rank*6..+5 of this pass; partial layout [tok][PR]
    constexpr int RO = PR / S;          // 6
    for (int e = tid; e < T * RO / 2; e += NCT) {   // 384 float2
      int tok = e / (RO / 2), r2 = e % (RO / 2);
      uint32_t la = sa(part + tok * PR + rank * RO + 2 * r2);
      float2 sum = make_float2(0.f, 0.f);
#pragma unroll
      for (int c = 0; c < S; c++) { float2 v = ld_dsm2(mapa(la, c)); sum.x += v.x; sum.y += v.y; }
      int n = grp * NR + p * PR + rank * RO + 2 * r2;
      *(__nv_bfloat162*)(out + (size_t)tok * N + n) = __floats2bfloat162_rn(sum.x, sum.y);
    }
    asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");
    if (tid < S) mb_arrive_remote(mapa(consumed, tid));
    STAMP(6 + 4 * p)
  }
  mb_wait_cl(consumed, (NPASS - 1) & 1); STAMP(15)  // nobody reads my smem any more
}

__global__ void flush(const float4* b, size_t n, float* o) { float s = 0; for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += b[i].x; if (s == 1234.5f) *o = s; }

template <int SB, int NST> void run(int reps, uint8_t* W, __nv_bfloat16* x, __nv_bfloat16* out, float4* fb, size_t fn, float* fo) {
  auto kf = sk128<SB, NST>;
  int smem = XB + NST * SB + PB + 64;
  CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CK(cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
  cfg.gridDim = dim3(S * G); cfg.blockDim = dim3(NTH); cfg.dynamicSmemBytes = smem;
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = S; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int ncl = 0; CK(cudaOccupancyMaxActiveClusters(&ncl, kf, &cfg));
  cudaStream_t st; CK(cudaStreamCreate(&st)); cfg.stream = st;
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  CK(cudaLaunchKernelEx(&cfg, kf, (const uint8_t*)W, (const __nv_bfloat16*)x, out, (unsigned long long*)nullptr));
  CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  // empty-ish reference graph: none; report raw
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
  int fd = open("/tmp/okc-gpu.lock", O_CREAT | O_RDWR, 0666); flock(fd, LOCK_EX);
  std::vector<float> ts;
  for (int r = 0; r < reps; r++) {
    flush<<<264, 1024, 0, st>>>(fb, fn, fo);
    cudaEventRecord(e0, st); cudaGraphLaunch(ge, st); cudaEventRecord(e1, st); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms * 1000.f);
  }
  flock(fd, LOCK_UN); close(fd);
  CK(cudaGetLastError());
  { unsigned long long* tl; CK(cudaMalloc(&tl, S * G * 16 * 8)); CK(cudaMemset(tl, 0, S * G * 16 * 8));
    flush<<<264, 1024, 0, st>>>(fb, fn, fo);
    CK(cudaLaunchKernelEx(&cfg, kf, (const uint8_t*)W, (const __nv_bfloat16*)x, out, tl)); CK(cudaStreamSynchronize(st));
    std::vector<unsigned long long> h(S * G * 16); CK(cudaMemcpy(h.data(), tl, h.size() * 8, cudaMemcpyDeviceToHost));
    unsigned long long t0 = ~0ull; for (int b = 0; b < S * G; b++) t0 = std::min(t0, h[b * 16]);
    for (int i = 0; i < 16; i++) { std::vector<double> v; for (int b = 0; b < S * G; b++) if (h[b*16+i]) v.push_back((h[b * 16 + i] - t0) / 1e3);
      if (v.empty()) continue; std::sort(v.begin(), v.end()); printf("  stamp %2d: min %6.2f med %6.2f max %6.2f\n", i, v[0], v[v.size()/2], v.back()); }
    cudaFree(tl); }
  std::sort(ts.begin(), ts.end());
  printf("SB %6d NST %d stages %3d smem %6d maxclusters %d: median %.2f us (p10 %.2f p90 %.2f)\n", SB, NST, WB / SB, smem, ncl,
         ts[reps / 2], ts[reps / 10], ts[reps * 9 / 10]);
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 50;
  uint8_t* W; __nv_bfloat16 *x, *out; float4* fb; float* fo;
  size_t wbytes = (size_t)WB * S * G, fn = (256u << 20) / 16;
  CK(cudaMalloc(&W, wbytes)); CK(cudaMalloc(&x, (size_t)T * K * 2)); CK(cudaMalloc(&out, (size_t)T * N * 2));
  CK(cudaMalloc(&fb, fn * 16)); CK(cudaMalloc(&fo, 4));
  CK(cudaMemset(W, 0x11, wbytes)); CK(cudaMemset(x, 0, (size_t)T * K * 2)); CK(cudaMemset(fb, 0, fn * 16));
  printf("W %.1f MB 12-bit; HBM floor at 3.24 TB/s %.2f us\n", wbytes / 1e6, wbytes / 3.24e6);
  run<9216, 4>(reps, W, x, out, fb, fn, fo);
  run<12288, 3>(reps, W, x, out, fb, fn, fo);
  run<18432, 2>(reps, W, x, out, fb, fn, fo);
  return 0;
}
