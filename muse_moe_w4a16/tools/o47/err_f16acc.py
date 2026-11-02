# o47 (c) accuracy: FP16-accumulate MMA vs FP32 accumulate on one real expert (reference.generate_weights(5252), expert E)
# x 256 tokens (generate_input bs 256 full, dist normal|skewed), full K = 4096. Operands are EXACT f16: W = E2M1 * E4M3 * 2^B
# (global scale and 2^-B applied after, in FP32), x bf16 -> f16 (exactness counted). gemm13 then gemm2 (inter = silu*up of
# the FP64 reference rounded to bf16, like SGLang). mma.sync m16n8k16 row.col; P = 0 FP16 acc over all K; P = 1/2/4 FP16
# partial over P k16 steps promoted (added) into FP32; F32 = f32 accumulate. Errors vs FP64 of the same operands.
import torch, sys
sys.path.insert(0, "../..")
from torch.utils.cpp_extension import load_inline
import reference as R
from nvfp4 import dequantize
SRC = r"""
#include <cstdint>
#include <cuda_fp16.h>
template <int P>  // P < 0: f32 accumulate
__global__ void g(const __half* __restrict__ W, const __half* __restrict__ X, float* __restrict__ Y, int M, int N, int K) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int tm = wid / (N / 8), tn = wid % (N / 8);
  if (tm * 16 >= M) return;
  const __half* w0 = W + (size_t)(tm * 16 + gq) * K; const __half* w1 = w0 + (size_t)8 * K;
  const __half* x0 = X + (size_t)(tn * 8 + gq) * K;
  float f[4] = {0.f, 0.f, 0.f, 0.f}; uint32_t h[2] = {0u, 0u};
  for (int k = 0, s = 0; k < K; k += 16, ++s) {
    uint32_t a0 = *(const uint32_t*)(w0 + k + 2 * t), a1 = *(const uint32_t*)(w1 + k + 2 * t);
    uint32_t a2 = *(const uint32_t*)(w0 + k + 8 + 2 * t), a3 = *(const uint32_t*)(w1 + k + 8 + 2 * t);
    uint32_t b0 = *(const uint32_t*)(x0 + k + 2 * t), b1 = *(const uint32_t*)(x0 + k + 8 + 2 * t);
    if (P < 0) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(h[0]), "+r"(h[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      if (P > 0 && (s + 1) % P == 0) {
        float2 u = __half22float2(*reinterpret_cast<__half2*>(&h[0])), v = __half22float2(*reinterpret_cast<__half2*>(&h[1]));
        f[0] += u.x; f[1] += u.y; f[2] += v.x; f[3] += v.y; h[0] = h[1] = 0u;
      }
    }
  }
  if (P == 0) {
    float2 u = __half22float2(*reinterpret_cast<__half2*>(&h[0])), v = __half22float2(*reinterpret_cast<__half2*>(&h[1]));
    f[0] = u.x; f[1] = u.y; f[2] = v.x; f[3] = v.y;
  }
  const int r0 = tm * 16 + gq, c0 = tn * 8 + 2 * t;  // Y [N, M] (token-major)
  Y[(size_t)c0 * M + r0] = f[0]; Y[(size_t)(c0 + 1) * M + r0] = f[1];
  Y[(size_t)c0 * M + r0 + 8] = f[2]; Y[(size_t)(c0 + 1) * M + r0 + 8] = f[3];
}
void run(int64_t p, torch::Tensor W, torch::Tensor X, torch::Tensor Y) {
  const int M = W.size(0), K = W.size(1), N = X.size(0), warps = (M / 16) * (N / 8);
  auto w = (const __half*)W.data_ptr(); auto x = (const __half*)X.data_ptr(); auto y = (float*)Y.data_ptr();
  const int b = 256, gr = (warps * 32 + b - 1) / b;
  if (p < 0) g<-1><<<gr, b>>>(w, x, y, M, N, K); else if (p == 0) g<0><<<gr, b>>>(w, x, y, M, N, K);
  else if (p == 1) g<1><<<gr, b>>>(w, x, y, M, N, K); else if (p == 2) g<2><<<gr, b>>>(w, x, y, M, N, K);
  else g<4><<<gr, b>>>(w, x, y, M, N, K);
}
"""
lib = load_inline("o47_err", cpp_sources="void run(int64_t, torch::Tensor, torch::Tensor, torch::Tensor);",
                  cuda_sources="#include <torch/extension.h>\n" + SRC, functions=["run"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
E = int(sys.argv[1]) if len(sys.argv) > 1 else 0
wt = R.generate_weights(R.WEIGHTS_SEED)
ex = {k: wt[k][E].clone() for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
del wt; torch.cuda.empty_cache()
def wq(w, sf):  # E2M1 * E4M3 exactly (global scale 1)
    return dequantize(w, sf, torch.tensor(1.0, device="cuda")).double()
def mm(p, Wd, X16, B):
    W16 = (Wd * 2.0 ** B).half()
    assert torch.equal(W16.double(), Wd * 2.0 ** B), "W not exact in f16"
    Y = torch.empty(X16.shape[0], Wd.shape[0], device="cuda"); lib.run(p, W16, X16, Y); torch.cuda.synchronize()
    return Y.double() * 2.0 ** -B
def report(tag, Wd, Xb, B):
    X16 = Xb.half(); inexact = (X16.double() != Xb.double()).sum().item()
    ref = Xb.double() @ Wd.T
    am = ref.abs().max().item()
    out = [f"{tag} B={B:+d} x->f16 inexact {inexact} | max|ref| {am:.4g} (x2^B {am * 2 ** B:.4g})"]
    for p, nm in ((-1, "F32"), (0, "F16all"), (4, "F16p4"), (2, "F16p2"), (1, "F16p1")):
        y = mm(p, Wd, X16, B); d = y - ref
        nf = (~torch.isfinite(y)).sum().item(); d = torch.where(torch.isfinite(y), d, torch.zeros_like(d))
        rel = (d.abs() / ref.abs().clamp(min=1e-3 * am)).max().item()
        out.append(f"   {nm:7s} maxabs {d.abs().max().item():.3e} (={d.abs().max().item() / am:.2e} of max|ref|) maxrel(|ref|>1e-3max) {rel:.3e} relL2 {(d.norm() / ref.norm()).item():.3e} nonfinite {nf}")
    print("\n".join(out), flush=True)
    return ref
W13 = wq(ex["w13"], ex["w13_sf"]); W2 = wq(ex["w2"], ex["w2_sf"])
g13, g2 = ex["w13_gs"].item(), ex["w2_gs"].item()
print(f"expert {E}: |W13 q| max {W13.abs().max().item():.4g}, |W2 q| max {W2.abs().max().item():.4g}, gs13 {g13:.4g} gs2 {g2:.4g}")
for dist in ("normal", "skewed"):
    x = R.generate_input(256, "full", 7, dist=dist)["hidden_states"]
    for B in (-7, -10, -14):
        ref13 = report(f"{dist} gemm13", W13, x, B)
    h = ref13 * g13
    inter = (torch.nn.functional.silu(h[:, :R.D_EXPERT]) * h[:, R.D_EXPERT:]).to(torch.bfloat16)
    for B in (-7, -10, -14):
        report(f"{dist} gemm2 ", W2, inter, B)
