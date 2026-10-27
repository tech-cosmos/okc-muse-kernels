// cuBLASLt BF16 GEMM with an explicit algo config: out[m,n] = x[m,k] @ W[n,k]^T, FP32 compute and accumulation.
// Same library and kernel family SGLang's F.linear reaches (algo 66 on sm90), but with the tile / split-K / cluster
// config picked per shape in prepare() (timed on the judge's own GPU) instead of cuBLAS's heuristic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

static cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) cublasLtCreate(&h);
  return h;
}

struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
};

// cfg = (algo id, tile, stages, split k, reduction scheme, swizzle, custom option, cluster shape).
// Returns 0 if the config is valid for the shape and fits the workspace.
static int lt_make(LtPlan& p, int64_t m, int64_t n, int64_t k, const std::vector<int64_t>& cfg, int64_t wsbytes) {
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, k, n, k);  // W, column-major [k, n]
  cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);  // x, column-major [k, m]
  cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n);  // out, column-major [n, m]
  if (cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                             CUDA_R_16BF, (int)cfg[0], &p.algo) != CUBLAS_STATUS_SUCCESS) return 1;
  uint32_t tile = cfg[1], stages = cfg[2], red = cfg[4], swz = cfg[5], cust = cfg[6];
  int32_t sk = cfg[3];
  uint16_t cl = cfg[7];
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4);
  if (sk > 1) cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cl, 2);
  cublasLtMatmulHeuristicResult_t r;
  if (cublasLtMatmulAlgoCheck(lt_handle(), p.op, p.a, p.b, p.c, p.c, &p.algo, &r) != CUBLAS_STATUS_SUCCESS) return 2;
  if (r.workspaceSize > (size_t)wsbytes) return 3;
  return 0;
}

static void lt_free(LtPlan& p) {
  cublasLtMatrixLayoutDestroy(p.a);
  cublasLtMatrixLayoutDestroy(p.b);
  cublasLtMatrixLayoutDestroy(p.c);
  cublasLtMatmulDescDestroy(p.op);
}

// Returns the cuBLAS status (0 = launched).
int64_t lt_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, std::vector<int64_t> cfg) {
  int64_t m = x.size(0), k = x.size(1), n = out.size(1);
  LtPlan p;
  int64_t st = lt_make(p, m, n, k, cfg, ws.numel());
  if (st == 0) {
    float one = 1.f, zero = 0.f;
    st = 100 + (int64_t)cublasLtMatmul(lt_handle(), p.op, &one, w.data_ptr(), p.a, x.data_ptr(), p.b, &zero,
                                       out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo, ws.data_ptr(), ws.numel(),
                                       at::cuda::getCurrentCUDAStream());
    if (st == 100) st = 0;
  }
  lt_free(p);
  return st;
}

// Print-only diag helper (s68): every valid algo-66 config for the shape as cfg tuples [N, 8] (stages auto, swizzle 0,
// reduction scheme 2 when split): tiles from the algo caps x custom options cust_lo..cust_hi x cluster shapes x split-K.
torch::Tensor lt_cands(int64_t m, int64_t n, int64_t k, int64_t wsbytes, int64_t cust_lo, int64_t cust_hi) {
  cublasLtMatmulAlgo_t a;
  std::vector<int64_t> keep;
  if (cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                             CUDA_R_16BF, 66, &a) == CUBLAS_STATUS_SUCCESS) {
    size_t sz = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
    std::vector<int> tiles(sz / 4);
    cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
    const int clusters[] = {0, 2, 3, 4, 5, 6, 8};
    const int sks[] = {1, 2, 3, 4, 6, 8};
    for (int t : tiles) for (int64_t cu = cust_lo; cu <= cust_hi; cu++) for (int c : clusters) for (int sk : sks) {
      std::vector<int64_t> cfg = {66, t, 35, sk, sk > 1 ? 2 : 0, 0, cu, c};
      LtPlan p;
      if (lt_make(p, m, n, k, cfg, wsbytes) == 0) keep.insert(keep.end(), cfg.begin(), cfg.end());
      lt_free(p);
    }
  }
  auto out = torch::empty({(int64_t)keep.size() / 8, 8}, torch::kInt64);
  if (!keep.empty()) memcpy(out.data_ptr<int64_t>(), keep.data(), keep.size() * 8);
  return out;
}
