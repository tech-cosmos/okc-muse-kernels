// fresh f1s6: CUTLASS 3.x sm90 BF16 GEMM, swap-AB (D^T[N,M] = W[N,K] . x[M,K]^T), for the layer's T=128/256 dense GEMMs.
// Compile-time knobs (-D): TM TN TK (tile: TM over W rows, TN over tokens), CM CN (cluster), SCHED 0 coop / 1 pingpong,
// SK 0 persistent / 1 stream-K scheduler (runtime: splits, mode), OUTF32 0/1.
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;

#if OUTF32
using ElementD = float;
#else
using ElementD = cutlass::bfloat16_t;
#endif
using ElementA = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
constexpr int AlignD = 16 / sizeof(ElementD);
using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
using ClusterShape = Shape<Int<CM>, Int<CN>, _1>;
#if SCHED == 0
using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ESched = cutlass::epilogue::TmaWarpSpecializedCooperative;
#else
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
#endif
#if SK
using TSched = cutlass::gemm::StreamKScheduler;
#else
using TSched = cutlass::gemm::PersistentScheduler;
#endif

using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    void, LayoutD, AlignD, ElementD, LayoutD, AlignD, ESched>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, 8, ElementB, LayoutB, 8, float,
    TileShape, ClusterShape,
#ifdef STAGES
    cutlass::gemm::collective::StageCount<STAGES>,
#else
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
#endif
    KSched>::CollectiveOp;
using Kern = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, TSched>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kern>;

static Gemm g_gemm;

static typename Gemm::Arguments make_args(void* w, void* x, void* out, int n, int m, int k, int splits, int mode) {
  using SA = typename Kern::StrideA; using SB = typename Kern::StrideB; using SD = typename Kern::StrideD;
  SA sa = cutlass::make_cute_packed_stride(SA{}, make_shape(n, k, 1));
  SB sb = cutlass::make_cute_packed_stride(SB{}, make_shape(m, k, 1));
  SD sd = cutlass::make_cute_packed_stride(SD{}, make_shape(n, m, 1));
  int L = 1, kk = k;
  if (mode >= 100) {  // explicit split-K as a batch of L = mode % 100 K slices, D = L slabs [m][n] (reduce outside)
    L = mode % 100; kk = k / L;  // mode >= 200: W repacked slice-major [L][n][kk] (prepare-time, lossless)
    get<2>(sa) = kk; get<2>(sb) = kk; get<2>(sd) = (int64_t)n * m;
    if (mode >= 200) { get<0>(sa) = kk; get<2>(sa) = (int64_t)n * kk; }
    splits = 1; mode = 3;
  }
  typename Gemm::Arguments a{cutlass::gemm::GemmUniversalMode::kBatched, {n, m, kk, L},
                             {(ElementA*)w, sa, (ElementB*)x, sb},
                             {{1.f, 0.f}, nullptr, sd, (ElementD*)out, sd}};
#if SK
  a.scheduler.splits = splits;
  using DM = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using RM = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;
  a.scheduler.reduction_mode = mode >= 10 ? RM::Nondeterministic : RM::Deterministic;
  mode %= 10;
  a.scheduler.decomposition_mode = mode == 0 ? DM::Heuristic : mode == 1 ? DM::SplitK : mode == 2 ? DM::StreamK : DM::DataParallel;
#endif
  return a;
}

extern "C" size_t cg_ws(int n, int m, int k, int splits, int mode) {
  return Gemm::get_workspace_size(make_args(nullptr, nullptr, nullptr, n, m, k, splits, mode));
}
// init once (host work), then cg_run per call (graph-capturable launch)
extern "C" int cg_init(void* w, void* x, void* out, int n, int m, int k, int splits, int mode, void* ws, void* stream) {
  auto a = make_args(w, x, out, n, m, k, splits, mode);
  if (g_gemm.can_implement(a) != cutlass::Status::kSuccess) return 1;
  if (g_gemm.initialize(a, ws, (cudaStream_t)stream) != cutlass::Status::kSuccess) return 2;
  return 0;
}
extern "C" int cg_run(void* stream) {
  return g_gemm.run((cudaStream_t)stream) == cutlass::Status::kSuccess ? 0 : 3;
}
