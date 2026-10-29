// fresh f1s1 hop microbench (FRESH.md O1 follow-up): cost of ONE grid-wide dependency hop inside a persistent kernel.
// 132 CTAs x NH hops. Hop h: every CTA writes its 1/132 slice of an XN-float vector, fences, bumps cnt[h]; then every CTA
// spins until cnt[h] == grid, and loads the whole vector (XN floats from L2) and reduces it (the next phase's x).
// Optional: a producer warp per CTA keeps a bulk-copy weight stream in flight the whole time (STREAM=1), like a ring prefetch.
// Build: nvcc -O3 -arch=sm_90a -o /tmp/fresh/hop hop.cu ; run /tmp/fresh/hop
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int NH = 64, NTHR = 288, STAGE = 16384, NSTG = 8;
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

__global__ void __launch_bounds__(NTHR, 1) hop(int* cnt, float* vec, int xn, const char* W, long long wbytes, int stream,
                                               volatile int* stop, unsigned long long* ts, float* sink, int depth, unsigned long long* nbytes) {
  extern __shared__ __align__(128) char ring[];
  __shared__ uint64_t full[NSTG];
  __shared__ float red[8];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  if (tid == 0) for (int s = 0; s < NSTG; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(&full[s])));
  __syncthreads();
  if (warp == 8) {  // producer: endless bulk stream over W (strided by CTA) until the consumers finish
    if (lane || !stream) return;
    long long off = (long long)b * STAGE; int s = 0, par = 0, issued = 0;
    while (!*stop) {
      if (issued >= depth) {  // wait for the oldest copy before reusing its slot
        asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(sa(&full[s])), "r"(par ^ 1) : "memory");
      }
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(&full[s])), "r"(STAGE) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   ::"r"(sa(ring + s * STAGE)), "l"(W + off), "r"(STAGE), "r"(sa(&full[s])) : "memory");
      off += (long long)G * STAGE; if (off + STAGE > wbytes) off = (long long)b * STAGE;
      ++issued; if (++s == depth) { s = 0; par ^= 1; }
    }
    // drain outstanding copies so the CTA can exit cleanly
    nbytes[b] = (unsigned long long)issued * STAGE;
    for (int i = 0; i < min(issued, depth); ++i) {
      asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(sa(&full[s])), "r"(par ^ 1) : "memory");
      if (++s == depth) { s = 0; par ^= 1; }
    }
    return;
  }
  float acc = 0.f;
  const int per = (xn + G - 1) / G;
  for (int h = 0; h < NH; ++h) {
    float* v = vec + (size_t)h * xn;
    // a little fake work so CTAs arrive at slightly different times
    for (int i = tid; i < per; i += 256) { int k = b * per + i; if (k < xn) v[k] = acc + k; }
    __threadfence();
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (tid == 0) { ts[(h * G + b) * 3 + 0] = gt(); atomicAdd(&cnt[h], 1); long long it = 0; while (ld_acq(&cnt[h]) < G) { if (++it > (1LL << 24)) { printf("CTA %d stuck at hop %d cnt %d\n", b, h, ld_acq(&cnt[h])); asm volatile("trap;"); } } ts[(h * G + b) * 3 + 1] = gt(); }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    float a = 0.f;
    for (int i0 = 0; i0 < xn; i0 += 256 * 8) {
      float t[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) { int i = i0 + u * 256 + tid; t[u] = i < xn ? __ldcg(&v[i]) : 0.f; }
#pragma unroll
      for (int u = 0; u < 8; ++u) a += t[u];
    }
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
    if (!lane) red[warp] = a;
    asm volatile("bar.sync 1, 256;" ::: "memory");
    if (tid == 0) { float s = 0.f; for (int w = 0; w < 8; ++w) s += red[w]; acc += s * 1e-30f; ts[(h * G + b) * 3 + 2] = gt(); }
    asm volatile("bar.sync 1, 256;" ::: "memory");
  }
  if (tid == 0) { sink[b] = acc; if (atomicAdd(&cnt[NH], 1) == G - 1) *stop = 1; }
}

int main() {
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int G = std::min(nsm, 132);
  long long wbytes = 1LL << 30;
  char* W; CK(cudaMalloc(&W, wbytes)); CK(cudaMemset(W, 1, wbytes));
  int* cnt; CK(cudaMalloc(&cnt, (NH + 1) * 4));
  int* stop; CK(cudaMalloc(&stop, 4));
  float* vec; CK(cudaMalloc(&vec, (size_t)NH * 12288 * 4));
  unsigned long long* ts; CK(cudaMalloc(&ts, (size_t)NH * G * 3 * 8));
  float* sink; CK(cudaMalloc(&sink, G * 4));
  unsigned long long* nb; CK(cudaMalloc(&nb, G * 8));
  CK(cudaFuncSetAttribute(hop, cudaFuncAttributeMaxDynamicSharedMemorySize, NSTG * STAGE));
  int fd = open("/tmp/okc-gpu.lock", O_CREAT | O_RDWR, 0666);
  for (int depth : {0, 1, 2, 3, 4, 6, 8})
    for (int xn : {8192}) {
      const int stream = depth > 0;
      double bw = 0;
      std::vector<double> seen, xok;
      for (int rep = 0; rep < 5; ++rep) { 
        CK(cudaMemset(cnt, 0, (NH + 1) * 4)); CK(cudaMemset(stop, 0, 4));
        flock(fd, LOCK_EX);
        CK(cudaMemset(nb, 0, G * 8));
        hop<<<G, NTHR, NSTG * STAGE>>>(cnt, vec, xn, W, wbytes, stream, stop, ts, sink, std::max(depth, 1), nb);
        CK(cudaDeviceSynchronize());
        flock(fd, LOCK_UN);
        std::vector<unsigned long long> h((size_t)NH * G * 3);
        CK(cudaMemcpy(h.data(), ts, h.size() * 8, cudaMemcpyDeviceToHost));
        { std::vector<unsigned long long> n(G); CK(cudaMemcpy(n.data(), nb, G * 8, cudaMemcpyDeviceToHost));
          unsigned long long tot = 0; for (auto v : n) tot += v;
          unsigned long long t0 = ~0ull, t1 = 0; for (int c = 0; c < G; ++c) { t0 = std::min(t0, h[c * 3]); t1 = std::max(t1, h[((NH - 1) * G + c) * 3 + 2]); }
          bw += (double)tot / (double)(t1 - t0) / 5.0; }
        for (int k = 4; k < NH; ++k) {  // per hop: last arrival -> median/last 'seen' and median/last 'x loaded'
          unsigned long long last = 0; std::vector<unsigned long long> s, x;
          for (int c = 0; c < G; ++c) { last = std::max(last, h[(k * G + c) * 3]); s.push_back(h[(k * G + c) * 3 + 1]); x.push_back(h[(k * G + c) * 3 + 2]); }
          std::sort(s.begin(), s.end()); std::sort(x.begin(), x.end());
          seen.push_back((double)(s.back() - last) / 1e3); xok.push_back((double)(x.back() - last) / 1e3);
        }
      }
      std::sort(seen.begin(), seen.end()); std::sort(xok.begin(), xok.end());
      printf("in flight %5.1f MB, stream %.2f TB/s | ", depth * STAGE * G / 1e6, bw / 1e3);
      printf("stream %d  x %5d floats: last arrival -> all CTAs see the counter: median %.2f us (p90 %.2f); -> all have x: median %.2f us (p90 %.2f)\n",
             stream, xn, seen[seen.size() / 2], seen[seen.size() * 9 / 10], xok[xok.size() / 2], xok[xok.size() * 9 / 10]);
    }
  return 0;
}
