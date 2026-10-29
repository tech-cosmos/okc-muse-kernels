// fresh f1s1 ORACLE O1 (FRESH.md H1): does a persistent kernel that prefetches each next phase's weights before it waits
// on that phase's dependency stream the T=1 layer chain at ~bytes/BW, versus one PDL kernel per phase (today's structure)?
// Synthetic: real byte counts of the d1 layer (12-bit dense sizes, 3 experts), BF16 rows, GEMV dot products, real data
// dependencies (each phase reads the previous phase's output; the expert addresses come from a routing step).
// Timing-only: the numbers it computes are meaningless. Build: nvcc -O3 -arch=sm_90a -o /tmp/fresh/chain chain_oracle.cu
// Run:   /tmp/fresh/chain [reps]   (takes /tmp/okc-gpu.lock around each timed loop)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int NCTA_MAX = 132, NTHR = 288, NCW = 8;  // 8 consumer warps + 1 producer warp
#ifndef NSTG_
#define NSTG_ 10
#endif
constexpr int STAGE = 16384, NSTG = NSTG_, XMAX = 12288;  // ring 160 KB + x 48 KB
constexpr int MAXP = 12, NEXP = 32, EXP_SEL = 3;

struct Phase {
  int kind;            // 0 gemv, 1 routing (CTA 0), 2 final norm (CTA 0)
  int R, K;            // rows, row length (bf16)
  const char* base;    // weights; for expert phases the base of expert 0
  long long seg_stride;// bytes between experts (0: not an expert phase)
  int seg_rows;        // rows per selected expert
  int npart;           // participating CTAs (0..npart-1)
  int dep_mask;        // phases whose completion the consumers wait for
  const float* src; int src_len; int src_seg; int norm;  // x[seg*K + k] = src[seg*src_seg + k % src_len] (* rms^-1)
  float* out;
};
struct Plan { int np; Phase ph[MAXP]; int target[MAXP]; int route_phase; };
__constant__ Plan c_plan;
__constant__ int c_list[NCTA_MAX][MAXP + 1];  // per CTA: phase order, -1 terminated (persistent mode)

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(n)); }
__device__ __forceinline__ void mb_expect(uint64_t* b, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, int par) {
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(sa(b)), "r"(par) : "memory"); }
__device__ __forceinline__ void bulk(void* dst, const void* src, int bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void spin(const int* p, int t) { while (ld_acq(p) < t) __nanosleep(32); }

__device__ __forceinline__ void rows_of(const Phase& P, int idx, int& r0, int& r1) {
  r0 = (int)((long long)P.R * idx / P.npart); r1 = (int)((long long)P.R * (idx + 1) / P.npart); }

// One CTA runs the phases in `list` (a single phase in multi-kernel mode). Producer warp streams every chunk of the
// CTA's work list into the ring as far ahead as the ring allows, across phase boundaries; it waits only for the routing
// result before the first expert chunk. Consumers wait for each phase's dependencies, build x, and dot the rows.
__device__ unsigned long long g_ts[8][MAXP][4];
__device__ unsigned long long g_t0[8];
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define STAMP(i) do { if (blockIdx.x < 8 && threadIdx.x == 0) g_ts[blockIdx.x][p][i] = gt(); } while (0)
__global__ void __launch_bounds__(NTHR, 1) chain(int single, int* cnt, int* ids) {
  extern __shared__ __align__(128) char smem[];
  char* ring = smem;
  float* xs = reinterpret_cast<float*>(smem + NSTG * STAGE);
  __shared__ uint64_t full[NSTG], empty[NSTG];
  __shared__ float rowacc[16];
  __shared__ float red[NCW];
  __shared__ int sids[EXP_SEL];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int list[MAXP + 1];
  if (single >= 0) { list[0] = single; list[1] = -1; }
  else for (int i = 0; i <= MAXP; ++i) list[i] = c_list[blockIdx.x][i];
  const int me = blockIdx.x;  // participant index (phases use CTAs 0..npart-1)
  if (tid == 0) { for (int s = 0; s < NSTG; ++s) { mb_init(&full[s], 1); mb_init(&empty[s], NCW); } }
  if (tid < 16) rowacc[tid] = 0.f;
  if (blockIdx.x < 8 && tid == 0) g_t0[blockIdx.x] = gt();
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;");
  if (warp == NCW) {  // ---------------- producer
    if (lane) return;
    int s = 0, par = 0, idsok = 0; int e[EXP_SEL] = {0, 0, 0};
    for (int li = 0; list[li] >= 0; ++li) {
      const Phase& P = c_plan.ph[list[li]];
      if (P.kind) continue;
      int r0, r1; rows_of(P, me, r0, r1);
      const int rb = P.K * 2, per = max(1, STAGE / rb);
      if (P.seg_stride && !idsok) {
        spin(&cnt[c_plan.route_phase], 1);
        for (int j = 0; j < EXP_SEL; ++j) e[j] = __ldcg(&ids[j]);
        idsok = 1;
      }
      for (int r = r0; r < r1;) {
        int n = min(per, r1 - r);
        const char* src;
        if (P.seg_stride) { int j = r / P.seg_rows, rr = r % P.seg_rows; n = min(n, P.seg_rows - rr);
                            src = P.base + e[j] * P.seg_stride + (long long)rr * rb; }
        else src = P.base + (long long)r * rb;
        mb_wait(&empty[s], par ^ 1);
        mb_expect(&full[s], n * rb);
        bulk(ring + s * STAGE, src, n * rb, &full[s]);
        r += n; if (++s == NSTG) { s = 0; par ^= 1; }
      }
    }
    return;
  }
  // ---------------- consumers (256 threads, named barrier 1)
  int s = 0, par = 0;
  for (int li = 0; list[li] >= 0; ++li) {
    const int p = list[li];
    const Phase& P = c_plan.ph[p];
    if (tid == 0) for (int d = 0; d < c_plan.np; ++d) if (P.dep_mask >> d & 1) spin(&cnt[d], c_plan.target[d]);
    asm volatile("bar.sync 1, 256;" ::: "memory");
    STAMP(0);
    if (P.kind == 1) {  // routing: pick EXP_SEL distinct experts from the router outputs
      if (tid == 0) {
        int used = 0;
        for (int j = 0; j < EXP_SEL; ++j) {
          int ex = (int)(fabsf(__ldcg(&P.src[j])) * 1000.f) & (NEXP - 1);
          while (used >> ex & 1) ex = (ex + 1) & (NEXP - 1);
          used |= 1 << ex; ids[j] = ex;
        }
        __threadfence(); atomicAdd(&cnt[p], 1);
      }
      continue;
    }
    // x (with an optional rms norm of the source, computed redundantly per CTA: no extra hop)
    float sc = 1.f;
    if (P.norm) {
      float a = 0.f;
      for (int i0 = 0; i0 < P.src_len; i0 += 256 * 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { int i = i0 + u * 256 + tid; v[u] = i < P.src_len ? __ldcg(&P.src[i]) : 0.f; }
#pragma unroll
        for (int u = 0; u < 8; ++u) a += v[u] * v[u];
      }
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
      if (!lane) red[warp] = a;
      asm volatile("bar.sync 1, 256;" ::: "memory");
      a = 0.f; for (int w = 0; w < NCW; ++w) a += red[w];
      sc = rsqrtf(a / P.src_len + 1e-5f);
    }
    if (P.kind == 2) {
      if (tid == 0) { P.out[0] = sc; __threadfence(); atomicAdd(&cnt[p], 1); }
      continue;
    }
    const int nseg = P.seg_stride ? EXP_SEL : 1;
    {  // x[j*K + k] = src[j*SS + (k & (L2 - 1))], L2 = largest power of two <= src_len (values don't matter here)
      const int K = P.K, SS = P.src_seg; int L2 = 1; while (L2 * 2 <= P.src_len) L2 *= 2;
      const float* src = P.src;
      for (int j = 0; j < nseg; ++j)
        for (int k0 = 0; k0 < K; k0 += 256 * 8) {
          float v[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) { int k = k0 + u * 256 + tid; v[u] = k < K ? __ldcg(&src[j * SS + (k & (L2 - 1))]) : 0.f; }
#pragma unroll
          for (int u = 0; u < 8; ++u) { int k = k0 + u * 256 + tid; if (k < K) xs[j * K + k] = v[u] * sc; }
        }
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    STAMP(1);
    int r0, r1; rows_of(P, me, r0, r1);
    const int rb = P.K * 2, per = max(1, STAGE / rb), vpr = P.K / 8;
    const float ivpr = 1.f / vpr;
    for (int r = r0; r < r1;) {
      int n = min(per, r1 - r), j = 0;
      if (P.seg_stride) { j = r / P.seg_rows; n = min(n, P.seg_rows - r % P.seg_rows); }
      mb_wait(&full[s], par);
      if (r == r0) STAMP(2);
      const uint4* w = reinterpret_cast<const uint4*>(ring + s * STAGE);
      const float* x = xs + j * P.K;
      const int nv = n * vpr;
      float acc = 0.f; int row = -1;
#ifndef NOMATH
#pragma unroll 4
      for (int v = tid; v < nv; v += 256) {
        const int rr = (int)((v + 0.5f) * ivpr);
        if (rr != row) {
          if (row >= 0) { for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o); if (!lane) atomicAdd(&P.out[r + row], acc); }
          row = rr; acc = 0.f;
        }
        const uint4 u = w[v];
        const float4 x0 = *reinterpret_cast<const float4*>(x + (v - rr * vpr) * 8);
        const float4 x1 = *reinterpret_cast<const float4*>(x + (v - rr * vpr) * 8 + 4);
        acc += __uint_as_float(u.x << 16) * x0.x + __uint_as_float(u.x & 0xffff0000u) * x0.y
             + __uint_as_float(u.y << 16) * x0.z + __uint_as_float(u.y & 0xffff0000u) * x0.w
             + __uint_as_float(u.z << 16) * x1.x + __uint_as_float(u.z & 0xffff0000u) * x1.y
             + __uint_as_float(u.w << 16) * x1.z + __uint_as_float(u.w & 0xffff0000u) * x1.w;
      }
      if (row >= 0) { for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o); if (!lane) atomicAdd(&P.out[r + row], acc); }
#endif
      __syncwarp();
      if (!lane) mb_arrive(&empty[s]);
      r += n; if (++s == NSTG) { s = 0; par ^= 1; }
    }
    __threadfence();
    asm volatile("bar.sync 1, 256;" ::: "memory");
    STAMP(3);
    if (tid == 0) { __threadfence(); atomicAdd(&cnt[p], 1); }
  }
}

__global__ void fill(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    // two bf16 in [-1/64, 1/64): sign + exponent 0x79..0x7b + random mantissa
    uint32_t a = (h & 0x807f807fu) | 0x3c003c00u; p[i] = a;
  }
}
__global__ void flush(const float4* p, size_t n, float* out) {
  float a = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = p[i]; a += v.x + v.w; }
  if (a == 12345.f) out[0] = a;
}

static int lockfd = -1;
static void lock() { lockfd = open("/tmp/okc-gpu.lock", O_CREAT | O_RDWR, 0666); flock(lockfd, LOCK_EX); }
static void unlock() { flock(lockfd, LOCK_UN); close(lockfd); }

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 200;
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int N = std::min(nsm, NCTA_MAX);
  // ---- T=1 layer chain, bytes as in FRESH.md (12-bit dense sizes):
  struct Def { const char* name; int kind, R, K; int expert; int npart; int deps; int src; int srcseg; int norm; };
  // src: index of the phase whose output feeds x (-1 = layer input)
  std::vector<Def> D = {
    {"qkv",     0, 1728, 8192, 0, N, 0,                -1, 0, 0},   // 28.3 MB
    {"attn",    0,   32, 8192, 0, 32, 1 << 0,           0, 0, 0},   // 0.5 MB KV, 32 CTAs
    {"o",       0, 8192,  768, 0, N, 1 << 1,            1, 0, 0},   // 12.6 MB
    {"router",  0,  512, 8192, 0, N, 1 << 2,            2, 0, 1},   // 8.4 MB
    {"routing", 1,    0,    0, 0, 1, 1 << 3,            3, 0, 0},
    {"pooler",  0, 3072, 8192, 0, N, 1 << 2,            2, 0, 1},   // 50.3 MB
    {"gemm13",  0, 3 * 2304, 4096, 1, N, (1 << 5) | (1 << 4), 5, 0, 1},  // 3 x 18.9 MB
    {"gemm2",   0, 3 * 1152, 4096, 1, N, 1 << 6,        6, 2304, 0},  // 3 x 9.4 MB
    {"unpool",  0,  384, 8192, 0, N, 1 << 7,            7, 0, 0},   // 6.3 MB
    {"final",   2,    0,    0, 0, 1, 1 << 8,            8, 0, 1},
  };
  const int NP = (int)D.size();
  Plan plan{}; plan.np = NP; plan.route_phase = 4;
  float* input; CK(cudaMalloc(&input, 8192 * 4)); fill<<<64, 256>>>((uint32_t*)input, 8192, 7);
  std::vector<float*> outs(NP);
  size_t wbytes = 0;
  for (auto& d : D) if (!d.kind) wbytes += (size_t)(d.expert ? (d.R / EXP_SEL) * NEXP : d.R) * d.K * 2;
  char* W; CK(cudaMalloc(&W, wbytes)); fill<<<1024, 256>>>((uint32_t*)W, wbytes / 4, 11);
  size_t off = 0; double moved = 0;
  for (int p = 0; p < NP; ++p) {
    auto& d = D[p]; Phase& P = plan.ph[p];
    CK(cudaMalloc(&outs[p], std::max(d.R, 16) * 4)); CK(cudaMemset(outs[p], 0, std::max(d.R, 16) * 4));
    P.kind = d.kind; P.R = d.R; P.K = d.K; P.npart = d.npart; P.dep_mask = d.deps; P.norm = d.norm; P.out = outs[p];
    P.src = d.src < 0 ? input : outs[d.src];
    P.src_len = d.src < 0 ? 8192 : std::max(D[d.src].R, 1);
    if (d.srcseg) { P.src_len = d.srcseg; P.src_seg = d.srcseg; }
    if (!d.kind) {
      P.base = W + off;
      if (d.expert) { P.seg_rows = d.R / EXP_SEL; P.seg_stride = (long long)P.seg_rows * d.K * 2; off += (size_t)P.seg_stride * NEXP; }
      else off += (size_t)d.R * d.K * 2;
      moved += (double)d.R * d.K * 2;
    }
    plan.target[p] = d.npart;
  }
  // gemm13's x comes from the pooler (phase 5); routing's src is the router (phase 3)
  printf("chain: %d phases, %.1f MB streamed, %d CTAs\n", NP, moved / 1e6, N);
  CK(cudaMemcpyToSymbol(c_plan, &plan, sizeof(plan)));
  int lst[NCTA_MAX][MAXP + 1];
  for (int c = 0; c < NCTA_MAX; ++c) { int k = 0; for (int p = 0; p < NP; ++p) if (c < D[p].npart) lst[c][k++] = p; lst[c][k] = -1; }
  CK(cudaMemcpyToSymbol(c_list, lst, sizeof(lst)));
  int *cnt, *ids; CK(cudaMalloc(&cnt, 64 * 4)); CK(cudaMalloc(&ids, 64 * 4));
  const int SMEM = NSTG * STAGE + XMAX * 4;
  CK(cudaFuncSetAttribute(chain, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  size_t fl_n = (size_t)1 << 28;  // 1 GB flush buffer (float4 count = 64M)
  float4* fl; CK(cudaMalloc(&fl, fl_n * 4)); CK(cudaMemset(fl, 0, fl_n * 4));
  float* sink; CK(cudaMalloc(&sink, 64));
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto launch = [&](int single, int grid, bool pdl) {
    cudaLaunchConfig_t cfg{}; cfg.gridDim = grid; cfg.blockDim = NTHR; cfg.dynamicSmemBytes = SMEM; cfg.stream = st;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&cfg, chain, single, cnt, ids));
  };
  auto capture = [&](int mode) {  // 0 memset only, 1 persistent, 2 kernel per phase + PDL, 3 kernel per phase no PDL
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    CK(cudaMemsetAsync(cnt, 0, 64 * 4, st));
    if (mode == 1) launch(-1, N, false);
    if (mode >= 2) for (int p = 0; p < NP; ++p) launch(p, D[p].npart, mode == 2 && p > 0);
    CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0)); return ge;
  };
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  {  // solo: each streaming phase alone, every dependency pre-satisfied (cnt = 0x7f7f7f7f, ids = 1,2,3)
    int h[3] = {1, 2, 3}; CK(cudaMemcpy(ids, h, 12, cudaMemcpyHostToDevice));
    for (int p = 0; p < NP; ++p) {
      if (D[p].kind) continue;
      cudaGraph_t g; cudaGraphExec_t ge;
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      CK(cudaMemsetAsync(cnt, 0x7f, 64 * 4, st)); launch(p, D[p].npart, false);
      CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
      for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, st));
      CK(cudaStreamSynchronize(st));
      std::vector<float> t; lock();
      for (int i = 0; i < 50; ++i) {
        flush<<<4 * N, 512, 0, st>>>(fl, fl_n / 4, sink);
        CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms * 1000.f);
      }
      unlock(); std::sort(t.begin(), t.end());
      double mb = (double)D[p].R * D[p].K * 2 / 1e6;
      printf("solo %-8s %6.1f MB  %7.1f us (incl. memset ~5)  %.2f TB/s excl. memset\n", D[p].name, mb, t[25], mb / (t[25] - 5.0) / 1e6 * 1e6 / 1e6);
    }
  }
  const char* names[] = {"memset only", "B persistent+prefetch", "A kernel/phase PDL", "A kernel/phase plain"};
  for (int mode = 0; mode < 4; ++mode) {
    cudaGraphExec_t ge = capture(mode);
    for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    std::vector<float> t;
    lock();
    for (int i = 0; i < reps; ++i) {
      flush<<<4 * N, 512, 0, st>>>(fl, fl_n / 4, sink);
      CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms * 1000.f);
    }
    unlock();
    std::sort(t.begin(), t.end());
    if (mode == 1) {
      unsigned long long ts[8][MAXP][4], t0[8]; CK(cudaMemcpyFromSymbol(ts, g_ts, sizeof(ts))); CK(cudaMemcpyFromSymbol(t0, g_t0, sizeof(t0)));
      for (int b = 0; b < 2; ++b) for (int p = 0; p < NP; ++p) if (!D[p].kind && b < D[p].npart)
        printf("   cta%d %-7s deps-ok %6.2f x-built %6.2f first-data %6.2f end %6.2f\n", b, D[p].name, (ts[b][p][0]-t0[b])/1e3,
               (ts[b][p][1]-t0[b])/1e3, (ts[b][p][2]-t0[b])/1e3, (ts[b][p][3]-t0[b])/1e3);
    }
    printf("%-24s median %7.1f us  p10 %7.1f  p90 %7.1f\n", names[mode], t[t.size() / 2], t[t.size() / 10], t[t.size() * 9 / 10]);
  }
  printf("bytes/3.0 TB/s (H100 stream) %.1f us, bytes/4.3 TB/s (H200 stream) %.1f us\n", moved / 3.0e6, moved / 4.3e6);
  return 0;
}
