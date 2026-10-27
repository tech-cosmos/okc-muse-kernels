"""moe n5s1: pr32 = T=16/32 pooler (g30n 12-bit, S2 cluster) + router (hi/lo BF16) + routing in one launch.
Exactness vs references + cold-L2 timing (3 us PDL pred like proj s95). Run from anywhere: python pr32.py [M]"""
import os, sys
LAYER = "/home/daytona/r-layer/muse_decode_layer"
sys.path.insert(0, LAYER)
os.chdir(LAYER)
import torch, torch.nn.functional as F
import bench, reference
from torch.utils.cpp_extension import load_inline
M = int(os.environ.get("M", 32))
HERE = os.path.dirname(os.path.abspath(__file__))
L = bench.load_sub("submission.py")
DEFS = f"G30_MTOK {M},G30_S {os.environ.get('S', 2)},G30_NW {os.environ.get('NW', 16)},G30_NS {os.environ.get('NS', 8)}," + os.environ.get("DEFS", "")
src = "".join(f"#define {d}\n" for d in DEFS.split(",") if d) + open(os.path.join(HERE, os.environ.get("SRC", "pr32_src.cu"))).read()
tag = "n5pr32_" + "_".join(d.replace(" ", "") for d in DEFS.split(",") if d)
lib = load_inline(name=tag, cpp_sources="void pr32_launch(torch::Tensor x, torch::Tensor wp, torch::Tensor hdr, torch::Tensor out, double scale, torch::Tensor rw, torch::Tensor part, torch::Tensor logits, torch::Tensor cnt, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor lids, torch::Tensor wts, double mult, double reps, int64_t local, int64_t pdl, torch::Tensor tl);",
                  cuda_sources=src, functions=["pr32_launch"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                  extra_ldflags=["-lcuda"], verbose=False)


def pack_router(r):
    """[512, 8192] BF16 -> [8 G][128 KS][4 j][128 thr][8 bf16] (A-fragment order, kernel PR_RST stages)."""
    n, k = r.shape
    G, KS = n // 64, k // 64
    t = r.contiguous().view(torch.int16)
    t = t.view(G, 4, 2, 8, KS, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 2, 8).reshape(G, KS, 128, 4, 8)
    return t.permute(0, 1, 3, 2, 4).contiguous().view(torch.uint8)


W = reference.generate_weights(reference.WEIGHTS_SEED)
router = W["router"].contiguous()
hi = router.to(torch.bfloat16); lo = (router - hi.float()).to(torch.bfloat16)
HL = "PR_HL 1" in os.environ.get("DEFS", "")
# PR_HL: each 64-row group = 32 hi rows + the same 32 lo rows (hi + lo folded in-CTA)
rw = pack_router(torch.cat([hi.view(8, 32, -1), lo.view(8, 32, -1)], 1).reshape(512, -1) if HL else torch.cat([hi, lo]))
code, smv, hdr, scale = L._compress(W["pooler"])
wp = L._pack30(code, smv)
dev = "cuda"
RR = int(os.environ.get("RS", 3))
part = torch.empty(8 * 8 * int(os.environ.get('S', 2)) * M * 64, dtype=torch.float32, device=dev)
logits = torch.zeros(M, 512, dtype=torch.float32, device=dev)
cnt = torch.zeros(16, dtype=torch.int32, device=dev)
bias = W["router_bias"].float().contiguous()
ls = torch.tensor([int(os.environ.get("LS", 64))], dtype=torch.int32, device=dev)
ids = torch.empty(M, 8, dtype=torch.int32, device=dev); lids = torch.empty_like(ids); wts = torch.empty(M, 8, dtype=torch.float32, device=dev)
out = torch.empty(M, 4096, dtype=torch.bfloat16, device=dev)
TL = torch.zeros(132 * 12, dtype=torch.int64, device=dev)
MULT, REPS, LOCAL = L.ROUTER_MULTIPLIER, L.ROUTE_EPS, L.LOCAL_EXPERTS
run = lambda x, pdl=0: lib.pr32_launch(x, wp, hdr, out, scale, rw, part, logits, cnt, bias, ls, ids, lids, wts, MULT, REPS, LOCAL, pdl, TL)  # noqa
bad = 0
for seed in range(RR):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(M, 8192, device=dev, generator=g) * (0.5 + seed)).bfloat16()
    for rep in range(3):
        run(x); torch.cuda.synchronize()
        ref = F.linear(x, W["pooler"])
        pe = (out.float() - (x.float() @ W["pooler"].float().t())).abs().max().item()
        same_pool = torch.equal(out, ref)
        lg = logits[:, :256] + logits[:, 256:]
        lref = x.double() @ router.double().t()
        le = (lg.double() - lref).abs().max().item()
        # routing vs the layer's Triton routing on our logits (exact) and on the FP64 logits
        i2 = torch.empty_like(ids); l2 = torch.empty_like(ids); w2 = torch.empty_like(wts)
        L._routing[(M,)](logits, bias, ls, i2, l2, w2, M, E=256, KS=2, TOPK=8, LOCAL=LOCAL, MULT=MULT, REPS=REPS, SKS=256, ST=512, num_warps=1)
        i3 = torch.empty_like(ids); l3 = torch.empty_like(ids); w3 = torch.empty_like(wts)
        lr32 = lref.float().contiguous()
        L._routing[(M,)](lr32, bias, ls, i3, l3, w3, M, E=256, KS=1, TOPK=8, LOCAL=LOCAL, MULT=MULT, REPS=REPS, num_warps=1)
        torch.cuda.synchronize()
        ok = (torch.equal(ids, i2) and torch.equal(lids, l2) and (wts - w2).abs().max().item() < 1e-6)
        same64 = torch.equal(torch.sort(ids, 1)[0], torch.sort(i3, 1)[0])
        print(f"seed {seed} rep {rep}: pooled==F.linear {same_pool} maxerr {pe:.3e} | logit err {le:.2e} | routing==triton(ours) {ok} "
              f"wdiff {(wts - w2).abs().max().item():.1e} | ids==fp64 route {same64} | cnt {cnt.tolist()[:9]}", flush=True)
        bad += (not ok) or cnt.abs().sum().item() != 0
print("BAD", bad)
if "PR_TL" in DEFS:
    for _ in range(3):
        TL.zero_(); x = torch.randn(M, 8192, device=dev).bfloat16(); torch.cuda.synchronize(); run(x); torch.cuda.synchronize()
    tl = TL.view(-1, 12)[:66 * int(os.environ.get('S', 2))].cpu().double()
    t0 = tl[tl[:, 8] > 0, 8].min()
    rel = torch.where(tl > 0, (tl - t0) / 1000, torch.full_like(tl, float('nan')))
    names = ["rstore", "rpart", "fingest", "fxchg", "pstart", "pdone", "fpoll", "froute", "entry", "xwait", "end"]
    for i, n in enumerate(names):
        c = rel[:, i]; c = c[~c.isnan()]
        if len(c): print(f"{n:7s} n {len(c):3d} min {c.min():6.2f} med {c.median():6.2f} max {c.max():6.2f} us")
    pr = rel[4:].view(-1, 2, 12)
    pd = torch.maximum(pr[:, 0, 5], pr[:, 1, 5]); tail = pr[:, 0, 10] - pd; imb = (pr[:, 0, 5] - pr[:, 1, 5]).abs()
    print(f"pair tail end-max(pdone) min {tail.min():.2f} med {tail.median():.2f} max {tail.max():.2f} | pair pdone imbalance med {imb.median():.2f} max {imb.max():.2f}")
    print("rg reducers (cta, rgred):", [(i, round(float(rel[i, 2]), 2)) for i in range(rel.shape[0]) if not rel[i, 2].isnan()])

if os.environ.get("TIME", "1") == "1":
    PRED = 3000
    pm = load_inline(name="n5_pred", cpp_sources="void pred(torch::Tensor, torch::Tensor, int64_t);", cuda_sources=r"""
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
}""", functions=["pred"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    x = torch.randn(M, 8192, device=dev).bfloat16(); xs = x.clone()
    WARM = float(os.environ.get("WARM", 0))  # n5s4: fraction of every CTA's pooler stream (+ all router W) L2-warm first
    if WARM > 0:
        wm = load_inline(name="n5_warm", cpp_sources="void warm(torch::Tensor, torch::Tensor, torch::Tensor);", cuda_sources=r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void wk(const long long* __restrict__ off, const int* __restrict__ len, const uint8_t* base, unsigned long long* sink) {
    const uint4* p = (const uint4*)(base + off[blockIdx.x]);
    unsigned s = 0;
    for (int i = threadIdx.x; i < len[blockIdx.x] / 16; i += blockDim.x) { uint4 v = __ldcg(p + i); s ^= v.x ^ v.w; }
    if (s == 0x12345678u) *sink = s;
}
void warm(torch::Tensor off, torch::Tensor len, torch::Tensor sink) {
    wk<<<off.numel(), 512, 0, c10::cuda::getCurrentCUDAStream()>>>((const long long*)off.data_ptr(), (const int*)len.data_ptr(), (const uint8_t*)0, (unsigned long long*)sink.data_ptr());
}""", functions=["warm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
        KS = 128; WST = wp.numel() * wp.element_size() // (64 * KS)
        offs, lens = [], []
        for c in range(rw.numel() // (64 << 10) if os.environ.get("WR", "1") == "1" else 0):
            offs.append(rw.data_ptr() + c * (64 << 10)); lens.append(64 << 10)
        for gq in range(64):
            for y in range(2):
                i0 = 2 * (y * (KS // 2) // 2)
                WA, WB = (int(v) for v in os.environ.get("WPAT", "0/0").split("/"))
                if WB:  # n5s4: stages i with i % WB < WA (interleaved), WARM ignored
                    for i in range(KS // 2):
                        if i % WB < WA:
                            offs.append(wp.data_ptr() + (gq * KS + i0 + i) * WST); lens.append(WST)
                    continue
                n = int(WARM * (KS // 2)) * WST
                offs.append(wp.data_ptr() + (gq * KS + i0) * WST); lens.append(n)
        woff = torch.tensor(offs, dtype=torch.int64, device=dev); wlen = torch.tensor(lens, dtype=torch.int32, device=dev)
        wsink = torch.zeros(1, dtype=torch.int64, device=dev)
        print(f"WARM {WARM}: {sum(lens) / 1e6:.1f} MB in L2 (WST {WST})")
        pre = lambda: (wm.warm(woff, wlen, wsink), pm.pred(xs, x, PRED))  # noqa
    else:
        pre = lambda: pm.pred(xs, x, PRED)  # noqa
    e0 = bench.time_graph(pre)
    for _ in range(3):
        print(f"M{M} pr32 {DEFS}: {bench.time_graph(lambda: (pre(), run(x, 1))) - e0:.2f} us (over a {PRED/1000:.0f} us PDL pred)", flush=True)
