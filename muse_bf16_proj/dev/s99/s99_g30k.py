"""proj s99: g30k (stream-K g30 on G CTAs) vs g30 (S1/S2) at T=16/32: exactness (rel err vs FP32, one-hot bit-exact over
escape columns, run-to-run bitwise), cold-L2 timing after a 3 us PDL pred. Run from a copy of r-layer muse_decode_layer/
with this file in its dev/ and g30k_part.cu next to it.
env: M (32), KEYS (pooler,unpool), GS (grid list, e.g. 96,108,120), KDEFS (g30k defines, e.g. G30_NW=20,G30_NS=8),
     SDEFS (g30 baseline 'S:NW:NS' list, e.g. 2:20:16,1:20:8), PRED ns (3000), REPS (2)."""
import os, sys, hashlib
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(D))
import torch, torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import bench, reference
M = int(os.environ.get("M", 32))
L = bench.load_sub("submission.py")
SRC = open(os.path.join(os.path.dirname(D), "g30_src.cu")).read()
KSRC = SRC + open(os.path.join(D, "g30k_part.cu")).read()
CU = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]


def build(src, dfs, fn, sig):
    h = hashlib.sha256((src + " ".join(dfs)).encode()).hexdigest()[:10]
    return load_inline(name=f"pj99_{fn}_{h}", cpp_sources=sig, cuda_sources=src, functions=[fn],
                       extra_cuda_cflags=CU + dfs, extra_ldflags=["-lcuda"])


KSIG = "void g30k_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, torch::Tensor, torch::Tensor, int64_t);"
KB = os.environ.get("KDEFS", "G30_NW=20,G30_NS=8")
KV = os.environ.get("KVARS", "").split("|")  # extra define sets on top of KDEFS; '' = KDEFS alone
klibs = {v: build(KSRC, [f"-D{d}" for d in (KB + "," + v).split(",") if d] + [f"-DG30_MTOK={M}"], "g30k_gemm", KSIG) for v in KV}
klib = klibs[KV[0]]
slibs = {}
for s in os.environ.get("SDEFS", "2:20:16,1:20:8").split(","):
    if s:
        S, NW, NS = s.split(":")
        slibs[s] = build(SRC, [f"-DG30_MTOK={M}", f"-DG30_S={S}", f"-DG30_NW={NW}", f"-DG30_NS={NS}"], "g30_gemm",
                         "void g30_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);")
PRED = int(os.environ.get("PRED", 3000))
pm = load_inline(name="pj95_pred", cpp_sources="void pred(torch::Tensor, torch::Tensor, int64_t);", cuda_sources=r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void pk(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
void pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    pk<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * 2 / 16, ns_);
}""", functions=["pred"], extra_cuda_cflags=CU)
GS = [int(g) for g in os.environ.get("GS", "96,108,120,132").split(",")]
REPS = int(os.environ.get("REPS", 2))
W = reference.generate_weights(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for key in os.environ.get("KEYS", "pooler,unpool").split(","):
    w = W["unpooler"] if key == "unpool" else W[key]
    N, K = w.shape
    code, sm, hdr, scale = L._compress(w)
    wp = L._pack30(code, sm)
    slab = torch.empty(N // 64 * 4 * (M // 2) * 128, device="cuda", dtype=torch.float32)
    tick = torch.zeros(N // 64, device="cuda", dtype=torch.int32)
    x = torch.randn(M, K, device="cuda").bfloat16()
    ref32 = x.float() @ w.float().t()
    rel = lambda o: ((o.float() - ref32).norm() / ref32.norm()).item()  # noqa: E731
    # one-hot exactness over escape columns
    esc_cols = torch.unique(hdr[:, :, 0][hdr[:, :, 1] != 2.0 ** -15].contiguous().view(torch.int32)).tolist()
    cols = esc_cols + torch.randperm(K)[:64].tolist()
    KP = K // 128
    P = N // 64 * KP
    nsl = lambda G: KP // (P // G) + 2  # noqa: E731  (slab bound used by the kernel check)
    def run(kl, v, x, G, pdl=0, o=None):
        if "G30K_F32=1" in v:
            o = o if o is not None else torch.zeros(nsl(G), M, N, device="cuda", dtype=torch.float32)
            kl.g30k_gemm(x, wp, hdr, o, scale, pdl, zt, tick, G)
            return o
        o = o if o is not None else torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        kl.g30k_gemm(x, wp, hdr, o, scale, pdl, slab, tick, G)
        return o
    zt = torch.ones(int(os.environ.get("ZN", 0)), device="cuda")  # F32 mode: zeroed-ahead buffer (ZN floats)
    fin = lambda o: o.sum(0).bfloat16() if o.dim() == 3 else o  # noqa: E731
    msg = []
    for v, klib in klibs.items():
      if "PROBE" in v or "NOSLAB" in v:
        continue
      for G in GS:
        bad = 0
        for p in range(0, len(cols), M):
            cs = (cols[p:p + M] + cols[:M] + cols[:M])[:M]
            xo = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
            xo[torch.arange(M), torch.tensor(cs)] = 1
            out = fin(run(klib, v, xo, G))
            bad += int((out.view(torch.int16) != w[:, cs].t().contiguous().view(torch.int16)).sum())
        ob = run(klib, v, x, G)
        o1 = fin(ob).clone()
        nd = 0
        for _ in range(5):
            run(klib, v, x, G, o=ob)
            nd += int((fin(ob).view(torch.int16) != o1.view(torch.int16)).sum())
        zbad = int(zt.count_nonzero())
        msg.append(f"{v}:G{G}: zn left {zbad}, onehot bad {bad}, rerun diff {nd}, err {rel(o1):.2e}, tick {int(tick.sum())}")
    print(f"M{M} {key} {N}x{K} esc cols {len(esc_cols)} cuBLAS err {rel(F.linear(x, w)):.2e} | " + " | ".join(msg), flush=True)
    xsrc = x.clone()
    pre = lambda: pm.pred(xsrc, x, PRED)  # noqa: E731
    e0 = bench.time_graph(pre)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    res = {}
    for r in range(REPS):
        for v, kl in klibs.items():
            for G in GS:
                ob = run(kl, v, x, G)
                res.setdefault(f"k{v}:{G}", []).append(bench.time_graph(lambda: (pre(), run(kl, v, x, G, 1, ob))) - e0)
        for s, lb in slibs.items():
            if K % (128 * int(s.split(":")[0])) == 0:
                res.setdefault(f"S{s}", []).append(bench.time_graph(lambda: (pre(), lb.g30_gemm(x, wp, hdr, out, scale, 1))) - e0)
        res.setdefault("cuBLAS", []).append(bench.time_graph(lambda: (pre(), F.linear(x, w))) - e0)
    print(f"M{M} {key} us after {PRED} ns pred: " + "  ".join(f"{k} " + "/".join(f"{v:.2f}" for v in vs) for k, vs in res.items()), flush=True)
