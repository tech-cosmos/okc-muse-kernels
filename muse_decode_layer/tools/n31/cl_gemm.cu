// n31 oracle: CUTLASS 4.1 sm90 TMA warp-specialized GEMM (stream-K / split-K scheduler, cluster multicast) for the layer's
// T=128/256 qkv: out[M,N] = x[M,K] @ W[N,K]^T, BF16 in/out, FP32 accumulate. Compile-time config via -D:
// SWAP (0: A=x, 1: A=W, D col-major), TM TN TK, CM CN (cluster), SCHED (0 persistent, 1 stream-K), KS (cooperative=1/pingpong=0).
// Runtime: ./a.out M [splits] ; prints median us over 50 cold-L2 runs (cudaEvent per run, 200 MB flush between).
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;
#ifndef SWAP
#define SWAP 0
#endif
#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 128
#endif
#ifndef TK
#define TK 64
#endif
#ifndef CM
#define CM 1
#endif
#ifndef CN
#define CN 1
#endif
#ifndef SCHED
#define SCHED 1
#endif
#ifndef COOP
#define COOP 1
#endif
using Ebf = cutlass::bfloat16_t;
using LA = cutlass::layout::RowMajor;
using LB = cutlass::layout::ColumnMajor;
#if SWAP
using LD = cutlass::layout::ColumnMajor;
#else
using LD = cutlass::layout::RowMajor;
#endif
using TileS = Shape<Int<TM>, Int<TN>, Int<TK>>;
using Clu = Shape<Int<CM>, Int<CN>, _1>;
#if COOP
using KSch = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ESch = cutlass::epilogue::TmaWarpSpecializedCooperative;
#else
using KSch = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESch = cutlass::epilogue::TmaWarpSpecialized;
#endif
using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileS, Clu, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LD, 8, Ebf, LD, 8, ESch>::CollectiveOp;
using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Ebf, LA, 8, Ebf, LB, 8, float, TileS, Clu,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>, KSch>::CollectiveOp;
#if SCHED
using Sch = cutlass::gemm::StreamKScheduler;
#else
using Sch = cutlass::gemm::PersistentScheduler;
#endif
using GK = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, Sch>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;

__global__ void fill(Ebf* p, size_t n, float s, unsigned seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed; h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15;
    p[i] = Ebf(s * ((h & 0xffff) / 32768.f - 1.f));
  }
}
__global__ void refk(const Ebf* x, const Ebf* w, float* o, int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y; if (n >= N) return;
  float a = 0; for (int k = 0; k < K; ++k) a += float(x[(size_t)m * K + k]) * float(w[(size_t)n * K + k]); o[(size_t)m * N + n] = a;
}
int main(int argc, char** argv) {
  int M = atoi(argv[1]), N = 2304, K = 8192, splits = argc > 2 ? atoi(argv[2]) : 1;
  int mode = argc > 3 ? atoi(argv[3]) : 0;  // 0 heuristic, 1 split-K, 2 stream-K, 3 data-parallel
  Ebf *x, *w, *o; float* r; char* fl;
  cudaMalloc(&x, (size_t)M * K * 2); cudaMalloc(&w, (size_t)N * K * 2); cudaMalloc(&o, (size_t)M * N * 2);
  cudaMalloc(&r, (size_t)M * N * 4); cudaMalloc(&fl, 200 << 20);
  fill<<<512, 256>>>(x, (size_t)M * K, 1.f, 1); fill<<<512, 256>>>(w, (size_t)N * K, 0.02f, 2);
  refk<<<dim3((N + 255) / 256, M), 256>>>(x, w, r, M, N, K);
  int gm = SWAP ? N : M, gn = SWAP ? M : N;
  const Ebf* A = SWAP ? w : x; const Ebf* B = SWAP ? x : w;
  auto sA = cutlass::make_cute_packed_stride(typename GK::StrideA{}, make_shape(gm, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename GK::StrideB{}, make_shape(gn, K, 1));
  auto sD = cutlass::make_cute_packed_stride(typename GK::StrideD{}, make_shape(gm, gn, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {gm, gn, K, 1}, {A, sA, B, sB},
                                {{1.f, 0.f}, nullptr, sD, o, sD}};
#if SCHED
  args.scheduler.splits = splits;
  using DM = cutlass::gemm::kernel::detail::DecompositionMode;
  args.scheduler.decomposition_mode = mode == 1 ? DM::SplitK : mode == 2 ? DM::StreamK : mode == 3 ? DM::DataParallel : DM::Heuristic;
  args.scheduler.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;
#endif
  Gemm g; size_t ws = Gemm::get_workspace_size(args); void* wsp; cudaMalloc(&wsp, ws + 16);
  if (g.can_implement(args) != cutlass::Status::kSuccess) { printf("cannot implement\n"); return 1; }
  if (g.initialize(args, wsp) != cutlass::Status::kSuccess) { printf("init fail\n"); return 1; }
  fprintf(stderr, "ws=%zu\n", ws); g.run(); fprintf(stderr, "launched %s\n", cudaGetErrorString(cudaGetLastError())); cudaDeviceSynchronize(); fprintf(stderr, "first run ok\n");
  std::vector<Ebf> ho((size_t)M * N); std::vector<float> hr((size_t)M * N);
  cudaMemcpy(ho.data(), o, ho.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hr.data(), r, hr.size() * 4, cudaMemcpyDeviceToHost);
  double e2 = 0, r2 = 0; for (size_t i = 0; i < ho.size(); ++i) { double d = float(ho[i]) - hr[i]; e2 += d * d; r2 += (double)hr[i] * hr[i]; }
  cudaEvent_t a, a2, b; cudaEventCreate(&a); cudaEventCreate(&a2); cudaEventCreate(&b); std::vector<float> ts, ts2;
  int lk = open("/tmp/okc-gpu.lock", O_RDWR | O_CREAT, 0666); fprintf(stderr, "wait lock\n"); flock(lk, LOCK_EX); fprintf(stderr, "got lock\n"); alarm(20);
  for (int i = 0; i < 60; ++i) {
    cudaMemsetAsync(fl, i, 200 << 20); cudaEventRecord(a); g.initialize(args, wsp); cudaEventRecord(a2); g.run(); cudaEventRecord(b); cudaEventSynchronize(b);
    float t, t2; cudaEventElapsedTime(&t, a, b); cudaEventElapsedTime(&t2, a2, b); if (i >= 10) { ts.push_back(t * 1000); ts2.push_back(t2 * 1000); }
  }
  flock(lk, LOCK_UN);
  std::sort(ts.begin(), ts.end()); std::sort(ts2.begin(), ts2.end());
  printf("M=%d swap=%d tile=%dx%dx%d clu=%dx%d sched=%d coop=%d splits=%d mode=%d ws=%zu: incl-init %.1f us, kernel-only %.1f (min %.1f) relerr %.1e\n",
         M, SWAP, TM, TN, TK, CM, CN, SCHED, COOP, splits, mode, ws, ts[ts.size() / 2], ts2[ts2.size() / 2], ts2[0], sqrt(e2 / r2));
  return 0;
}
