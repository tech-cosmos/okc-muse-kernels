"""csl (decode 1 sliding) per-CTA globaltimer timeline on the H100: cold (512 MB read before each replay) and warm.
Patches submission._CSL_SRC with stamps (thread 0 of each CTA): 0 entry, 1 page table landed (kslot[0]), 2 K/V landed,
3 P@V done, 4 pre-merge __syncthreads, 5 cluster wait done, 6 owner mbarrier done (rank < 8), 7 owner store done.
Run from muse_attention_decode/: python tools/s31/cslstamp.py [batch]"""
import fcntl, re, statistics, sys, torch
sys.path.insert(0, ".")
from torch.utils.cpp_extension import load_inline
import reference

txt = open("submission.py").read()
src = re.search(r"_CSL_SRC = r'''(.*?)'''", txt, re.S).group(1)
a, b = src.split("namespace vtk", 1)
TS = ('if (threadIdx.x == 0 && ({c}) != 0x7fffffffu) {{ unsigned long long t; '
      'asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); g_ts[blockIdx.x * 8 + {k}] = t; if ({k} == 0) {{ asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_ts[8192 + blockIdx.x] = t; }} }}\n')
def ins(s, anchor, code, after=True):
    assert s.count(anchor) == 1, anchor
    return s.replace(anchor, anchor + code if after else code + anchor)
a = ins(a, "namespace csl {", "\n__device__ unsigned long long g_ts[1024 * 9];\n")
a = ins(a, "    asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\" ::: \"memory\");\n", TS.format(c="0u", k=0))
a = ins(a, "#undef CSL_LOADS\n", TS.format(c="(unsigned)kslot[0]", k=1) + TS.format(c="kr[0].x ^ vr[7].x ^ kr[3].w", k=2))
a = ins(a, "    // Pre-merge the 4 warps", TS.format(c="__float_as_uint(acc[0][0] + acc[3][3])", k=3), after=False)
a = ins(a, "    __syncthreads();\n    {\n        const int h2 = h", "", after=True)
a = a.replace("    __syncthreads();\n    {\n        const int h2 = h", "    __syncthreads();\n" + TS.format(c="0u", k=4) + "    {\n        const int h2 = h")
a = ins(a, "        const unsigned owner = row;\n", TS.format(c="0u", k=5))
a = ins(a, "        float M = -1e30f;\n#pragma unroll\n        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);\n",
        TS.format(c="__float_as_uint(M)", k=6))
a = ins(a, "__floats2bfloat162_rn(o0, o1);\n", TS.format(c="0u", k=7))
import os
VAR = os.environ.get("VAR", "")
if "fdiv" in VAR:
    a = a.replace("float r = vscale / den;", "float r = __fdividef(vscale, den);")
    a = a.replace("o0 *= ni / (1.f + __expf(-g0));", "o0 *= __fdividef(ni, 1.f + __expf(-g0));")
    a = a.replace("o1 *= ni / (1.f + __expf(-g1));", "o1 *= __fdividef(ni, 1.f + __expf(-g1));")
if "sge" in VAR:
    anc = "    if (warp == 0 && rank < 8) {\n        asm volatile(\"{\\n\\t.reg .pred p;"
    assert a.count(anc) == 1
    a = a.replace(anc, "    const float sg0 = __fdividef(1.f, 1.f + __expf(-g0)), sg1 = __fdividef(1.f, 1.f + __expf(-g1));\n" + anc)
    a = a.replace("o0 *= __fdividef(ni, 1.f + __expf(-g0));", "o0 *= ni * sg0;")
    a = a.replace("o1 *= __fdividef(ni, 1.f + __expf(-g1));", "o1 *= ni * sg1;")
if "s7loop" in VAR:  # move stamp 7 to right after the 16-partial loop
    i = a.index("g_ts[blockIdx.x * 8 + 7]"); j = a.rindex("if (threadIdx.x == 0", 0, i); k = a.index("\n", i)
    st7 = a[j:k + 1]; a = a[:j] + a[k + 1:]
    assert a.count("        float r = ") == 1; a = a.replace("        float r = ", st7.replace("(0u)", "(__float_as_uint(o0 + o1 + den))") + "        float r = ")
if "w4" in VAR:  # 4 warps each fold 4 partials, one named barrier, warp 0 finishes
    o1_ = """    if (warp == 0 && rank < 8) {
        asm volatile("{\\n\\t.reg .pred p;\\n\\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\\n\\t@!p bra W;\\n\\t}"
                     ::"r"(mb) : "memory");"""
    assert a.count(o1_) == 1, "w4 a"
    a = a.replace(o1_, o1_.replace("warp == 0 && rank < 8", "rank < 8"))
    o2_ = """        for (int s = 0; s < 16; ++s) {
            float w = exp2f(rml[s][0] - M);
            den += w * rml[s][1];
            float2 a = *(const float2*)&racc[s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
"""
    assert a.count(o2_) == 1, "w4 b"
    a = a.replace(o2_, """        for (int s4 = 0; s4 < 4; ++s4) {
            const int s = 4 * warp + s4;
            float w = exp2f(rml[s][0] - M);
            den += w * rml[s][1];
            float2 a = *(const float2*)&racc[s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
        if (warp) { *(float2*)&lpart[0][warp][0][2 * lane] = make_float2(o0, o1); if (lane == 0) lpart[0][warp][1][0] = den; }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (warp) return;
#pragma unroll
        for (int k = 1; k < 4; ++k) { float2 v = *(const float2*)&lpart[0][k][0][2 * lane]; o0 += v.x; o1 += v.y; den += lpart[0][k][1][0]; }
""")
src = a + "namespace vtk" + b + r'''
void csl_ts(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), csl::g_ts, o.numel() * 8); }
'''
m = load_inline(name="muse_csl_stamp31" + VAR, cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double); void csl_ts(torch::Tensor);",
                cuda_sources=src, functions=["csl_decode", "csl_ts"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"])
import submission  # for _cos_sin / LOG2E only
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
c = dict(batch=B, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=1, dist="soft")
if B > 1: c["spread"] = "agentic"
data = reference.generate_input(seed=11, **c)
cfg = data["config"]; dev = data["qkv"].device
out = torch.empty((B, 1, 1024), dtype=torch.bfloat16, device=dev)
cs = submission._cos_sin(cfg["rope_theta"], 64, dev)
def run():
    m.csl_decode(data["qkv"], cs, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"],
                 data["cache_loc"], data["k_scale"], data["v_scale"], out, cfg["page_size"],
                 cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E)
run(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): run()
flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
ts = torch.zeros(1024 * 9, dtype=torch.int64)
n = 32 * B
GHZ = float(sys.argv[2]) if len(sys.argv) > 2 else 1.98
for mode in ("cold", "warm"):
    rows = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in range(30):
            if mode == "cold": flush.sum()
            else: g.replay()
            torch.cuda.synchronize()
            g.replay(); torch.cuda.synchronize()
            m.csl_ts(ts); t = ts[:n * 8].view(n, 8).double()
            gt = ts[8192:8192 + n].double()
            rows.append((t - t[:, :1]) / GHZ + (gt - gt.min())[:, None])
    R = torch.stack(rows)  # reps, n, 8
    own = [i for i in range(n) if (i % 16) < 8]
    print(f"--- {mode}: us from the first CTA entry (median over 30 replays of: median / max over CTAs)")
    for k, nm in enumerate(["entry", "pt landed", "K/V landed", "P@V done", "premerge sync", "cluster wait",
                            "owner mbar", "owner store"]):
        sel = R[:, own, k] if k >= 6 else R[:, :, k]
        med = statistics.median(sel.median(dim=1).values.tolist()) / 1000
        mx = statistics.median(sel.max(dim=1).values.tolist()) / 1000
        print(f"{k} {nm:14s} med {med:6.2f}  max {mx:6.2f}")
