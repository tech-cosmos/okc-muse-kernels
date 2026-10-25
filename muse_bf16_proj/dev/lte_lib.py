"""cuBLASLt algo enumeration for out[m,n] = x[m,k] @ W[n,k]^T (BF16, FP32 compute)."""
import os, sys, torch
from torch.utils.cpp_extension import load_inline

CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>

static cublasLtHandle_t H = nullptr;
static cublasLtHandle_t h() { if (!H) cublasLtCreate(&H); return H; }

struct Desc { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; };
static Desc mk(int64_t m, int64_t n, int64_t k) {
  Desc d;
  cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatrixLayoutCreate(&d.a, CUDA_R_16BF, k, n, k);   // W col-major [k, n]
  cublasLtMatrixLayoutCreate(&d.b, CUDA_R_16BF, k, m, k);   // x col-major [k, m]
  cublasLtMatrixLayoutCreate(&d.c, CUDA_R_16BF, n, m, n);   // out col-major [n, m]
  return d;
}
static void fr(Desc& d) { cublasLtMatrixLayoutDestroy(d.a); cublasLtMatrixLayoutDestroy(d.b); cublasLtMatrixLayoutDestroy(d.c); cublasLtMatmulDescDestroy(d.op); }

// returns [na, 8] int64 algo blobs (heuristic order)
// copy of algo with split-K / reduction scheme set; returns ok flag via validity check
torch::Tensor lt_variant(torch::Tensor algo, int64_t splitk, int64_t red, int64_t m, int64_t n, int64_t k, int64_t wsbytes) {
  cublasLtMatmulAlgo_t a; memcpy(&a, algo.data_ptr<int64_t>(), sizeof(a));
  int32_t sk = splitk; uint32_t rs = red;
  auto out = torch::zeros({9}, torch::kInt64);
  if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk)) != CUBLAS_STATUS_SUCCESS) return out;
  if (splitk > 1 && cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, sizeof(rs)) != CUBLAS_STATUS_SUCCESS) return out;
  Desc d = mk(m, n, k);
  cublasLtMatmulHeuristicResult_t r;
  auto st = cublasLtMatmulAlgoCheck(h(), d.op, d.a, d.b, d.c, d.c, &a, &r);
  fr(d);
  if (st != CUBLAS_STATUS_SUCCESS || r.workspaceSize > (size_t)wsbytes) return out;
  memcpy(out.data_ptr<int64_t>(), &a, sizeof(a)); out[8] = 1;
  return out;
}

torch::Tensor lt_enum(int64_t m, int64_t n, int64_t k, int64_t wsbytes, int64_t maxn) {
  Desc d = mk(m, n, k);
  std::vector<cublasLtMatmulAlgo_t> keep;
  cublasLtMatmulAlgo_t a;
  cublasLtMatmulAlgoInit(h(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 66, &a);
  size_t sz = 0;
  cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
  std::vector<int> tiles(sz / 4); cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
  int clusters[] = {0, 2, 3, 4, 5, 8, 6};
  int sks[] = {1, 2, 3, 4, 6, 8};
  for (int t : tiles) for (int cu = 0; cu < 8; cu++) for (int c : clusters) for (int sk : sks) {
    cublasLtMatmulAlgo_t b = a;
    uint32_t tt = t, s = 35, cc = cu, rs = 2; int32_t skk = sk; uint16_t c16 = c;
    cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_TILE_ID, &tt, 4);
    cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_STAGES_ID, &s, 4);
    cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cc, 4);
    cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &c16, 2);
    cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &skk, 4);
    if (sk > 1) cublasLtMatmulAlgoConfigSetAttribute(&b, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, 4);
    cublasLtMatmulHeuristicResult_t r;
    if (cublasLtMatmulAlgoCheck(h(), d.op, d.a, d.b, d.c, d.c, &b, &r) == CUBLAS_STATUS_SUCCESS && r.workspaceSize <= (size_t)wsbytes)
      keep.push_back(b);
    if ((int64_t)keep.size() >= maxn) goto done;
  }
done:
  fr(d);
  auto out = torch::zeros({(int64_t)keep.size(), 8}, torch::kInt64);
  for (size_t i = 0; i < keep.size(); i++) memcpy(out[i].data_ptr<int64_t>(), &keep[i], sizeof(cublasLtMatmulAlgo_t));
  return out;
}

torch::Tensor lt_desc(torch::Tensor algo) {
  cublasLtMatmulAlgo_t a; memcpy(&a, algo.data_ptr<int64_t>(), sizeof(a));
  cublasLtMatmulAlgoConfigAttributes_t at[] = {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
  auto out = torch::zeros({9}, torch::kInt64);
  size_t szs[] = {4, 4, 4, 4, 4, 4, 4, 2, 2};
  for (int i = 0; i < 9; i++) { uint64_t v = 0; size_t sz; cublasLtMatmulAlgoConfigGetAttribute(&a, at[i], &v, szs[i], &sz); out[i] = (int64_t)v; }
  return out;
}

torch::Tensor lt_caps(int64_t id) {
  cublasLtMatmulAlgo_t a;
  cublasLtMatmulAlgoInit(h(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, id, &a);
  size_t sz = 0; std::vector<int> v;
  cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz); int nt = sz / 4;
  cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz); int ns = sz / 4;
  std::vector<int> st(ns); if (ns) cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_STAGES_IDS, st.data(), ns*4, &sz);
  int32_t cm = 0; cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cm, 4, &sz);
  auto out = torch::zeros({3 + ns}, torch::kInt64); out[0] = nt; out[1] = ns; out[2] = cm;
  for (int i = 0; i < ns; i++) out[3 + i] = st[i];
  return out;
}

torch::Tensor lt_algos(int64_t m, int64_t n, int64_t k, int64_t wsbytes, int64_t maxa) {
  Desc d = mk(m, n, k);
  cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p);
  size_t ws = wsbytes;
  cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  std::vector<cublasLtMatmulHeuristicResult_t> r(maxa);
  int got = 0;
  cublasLtMatmulAlgoGetHeuristic(h(), d.op, d.a, d.b, d.c, d.c, p, maxa, r.data(), &got);
  auto out = torch::zeros({got, 8}, torch::kInt64);
  for (int i = 0; i < got; i++) memcpy(out[i].data_ptr<int64_t>(), &r[i].algo, sizeof(cublasLtMatmulAlgo_t));
  cublasLtMatmulPreferenceDestroy(p); fr(d);
  return out;
}

int64_t lt_mm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor algo, torch::Tensor ws) {
  int64_t m = x.size(0), k = x.size(1), n = out.size(1);
  Desc d = mk(m, n, k);
  cublasLtMatmulAlgo_t a; memcpy(&a, algo.data_ptr<int64_t>(), sizeof(a));
  float one = 1.f, zero = 0.f;
  auto st = cublasLtMatmul(h(), d.op, &one, w.data_ptr(), d.a, x.data_ptr(), d.b, &zero, out.data_ptr(), d.c, out.data_ptr(), d.c,
                 &a, ws.data_ptr(), ws.numel(), at::cuda::getCurrentCUDAStream());
  fr(d);
  return (int64_t)st;
}
"""
mod = load_inline("pj_lt", CPP, functions=["lt_caps", "lt_desc", "lt_algos", "lt_mm", "lt_variant", "lt_enum"], extra_ldflags=["-lcublasLt"], with_cuda=True, verbose=False)

if __name__ == "__main__":
    sys.path.insert(0, os.path.expanduser("~/r-proj/muse_bf16_proj"))
    import bench, reference
    W = reference.generate_weights(reference.WEIGHTS_SEED)
    wsb = 64 << 20
    ws = torch.empty(wsb, dtype=torch.uint8, device="cuda")
    ms = [int(a) for a in sys.argv[1].split(",")]
    layers = sys.argv[2].split(",")
    for m in ms:
        for layer in layers:
            data = reference.generate_input(m, layer, seed=7)
            x = data["x"]; n = data["config"]["n"]; k = x.shape[1]
            wt = W[layer][:n].contiguous()
            ref = torch.nn.functional.linear(x, wt)
            tc = bench.time_graph(lambda: torch.nn.functional.linear(x, wt)) - bench.empty_cost()
            al0 = mod.lt_algos(m, n, k, wsb, 64)
            al = [al0[i] for i in range(al0.shape[0])]
            for i in range(al0.shape[0]):
                for sk in (1, 2, 3, 4, 6, 8, 12, 16):
                    for red in (1, 2, 4):
                        v = mod.lt_variant(al0[i], sk, red, m, n, k, wsb)
                        if v[8] == 1: al.append(v[:8].clone())
            res = []
            for i in range(len(al)):
                out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                a = al[i].clone()
                mod.lt_mm(wt, x, out, a, ws); torch.cuda.synchronize()
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                t = bench.time_graph(lambda: mod.lt_mm(wt, x, out, a, ws), reps=40) - bench.empty_cost()
                res.append((t, i, err))
            res.sort()
            print(f"m={m} {layer} cublas {tc:.2f} | n_algos {len(res)}/{al0.shape[0]} best " + " ".join(f"#{i}:{t:.2f}{'' if e < 1e-2 else 'BAD'}" for t, i, e in res[:5]) + f" | heur#0 {[r[0] for r in res if r[1]==0][0]:.2f}", flush=True)
