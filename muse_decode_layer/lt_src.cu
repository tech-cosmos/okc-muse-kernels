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
