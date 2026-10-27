"""proj s91: g30 (layer s99 T=128 12-bit wgmma GEMM) + escape fixups. Correctness vs FP64 x @ W (true weights, natural
+ adversarial escapes, probe/spike token rows) and cold-L2 graph timing vs cuBLAS and vs the no-fixup build (H100).
usage: python g30e.py [name:N:K:S ...]   env DEFS="-DG30_NS=10", ESCW=1 adversarial escapes, REPS=n."""
import os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
LD = os.path.join(HERE, "../../../muse_decode_layer")
sys.path.insert(0, LD); sys.path.insert(0, os.path.join(LD, "dev"))
import torch
from torch.utils.cpp_extension import load_inline
import bench
os.chdir(LD)
layer = bench.load_sub(os.path.join(LD, "submission.py"))
sys.path.insert(0, HERE); import s99_g30_pack as P

SRC = open(os.path.join(HERE, os.environ.get("CU", "g30f.cu"))).read()  # g30f = layer s99 857a9cb + escapes; g30e = s99 5269ca7 + escapes
DEFS = os.environ.get("DEFS", "").split()


def build(defs):
    return load_inline("g30e_" + hashlib.md5((" ".join(defs) + SRC).encode()).hexdigest()[:10],
                       cpp_sources="void g30_run(torch::Tensor x, torch::Tensor wp, torch::Tensor out, int64_t N, int64_t S, double scale, torch::Tensor hdr);",
                       cuda_sources=SRC, functions=["g30_run"],
                       extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-lineinfo"] + defs,
                       extra_ldflags=["-lcuda"], verbose=False)


if not os.environ.get("NOMAIN"):
    mod, mod0 = build(DEFS), build(DEFS + ["-DG30_ESC=0"])
ESCW = os.environ.get("ESCW", "0") not in ("", "0")


def run(name, N, K, S):
    g_ = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=g_) / K ** 0.5).bfloat16()
    ce = torch.randperm(K, device="cuda", generator=g_)[:32]
    if ESCW:  # bench.py test_esc / g19.py ESCW injection: tiny / zero / subnormal values at 32 columns, ~75% of rows
        mx = w.float().abs().max().item()
        vals = torch.tensor([0.0, -0.0, 2.0 ** -133, -3 * 2.0 ** -131, 2.0 ** -100, -1.37 * 2.0 ** -60,
                             mx * 2.0 ** -17, -mx * 1.9 * 2.0 ** -18, 1.5 * 2.0 ** -126], device="cuda").bfloat16()
        for _ in range(3):
            rows = torch.arange(N, device="cuda")[torch.rand(N, device="cuda", generator=g_) < 0.75]
            w[rows, ce[torch.randint(0, 32, (len(rows),), device="cuda", generator=g_)]] = vals[torch.randint(0, len(vals), (len(rows),), device="cuda", generator=g_)]
    c = layer._compress(w)
    assert c is not None, "row with > G13_ESC escapes"
    code, sm, hdr, scale = c
    nesc = int((hdr[:, :, 1] != 2.0 ** -15).sum())
    v = hdr[:, :, 1][hdr[:, :, 1] != 2.0 ** -15]
    nsub = int(((v != 0) & (v.abs() < 2.0 ** -126)).sum())
    assert torch.equal(v.bfloat16().float(), v), "g30 escape values not exact in BF16 (prepare() must fall back for this weight)"
    wp = P.pack30(code, sm)
    x = torch.randn(128, K, device="cuda", generator=g_).bfloat16()
    for j in range(0, 64, 2):  # spike tokens: one x at an escape-heavy column (2 tokens per column), rest 0
        x[j].zero_(); x[j, ce[(j // 2) % 32]] = 8.0
    ref = x.double() @ w.double().t()
    out = torch.empty(128, N, device="cuda", dtype=torch.bfloat16) if S == 1 else torch.empty(S, 128, N, device="cuda")
    cb = (torch.mm(x, w.t()) if S == 1 else torch.mm(x.float(), w.float().t())).double()
    tol = (ref.abs() * 2.0 ** -8 + 1e-30) if S == 1 else (ref.abs() * 2.0 ** -20 + 2.0 ** -100)
    tol = torch.maximum(tol, 2 * (cb - ref).abs())  # same-precision baseline: within 2x cuBLAS's own error
    print(f"  {name} cuBLAS{' (fp32)' if S > 1 else ''}: rel err {((cb - ref).norm() / ref.norm()).item():.2e}", flush=True)
    for m_, tag in ((mod, "fix"), (mod0, "nofix")):
        worst = 0.0
        for rep in range(int(os.environ.get("REPS", 3))):
            out.fill_(7.0)
            m_.g30_run(x, wp, out, N, S, scale, hdr); torch.cuda.synchronize()
            got = (out.double() if S == 1 else out.double().sum(0))
            # BF16 out: within 1 BF16 ulp of the FP64 dot (plus a tiny absolute floor for FP32 rounding near 0)
            bad = int(((got - ref).abs() > tol).sum())
            err = ((got - ref).norm() / ref.norm()).item()
            spike_bad = int(((got[0:64:2] - ref[0:64:2]).abs() > tol[0:64:2]).sum())
            worst = max(worst, err)
        print(f"  {name} {tag}: rel err {worst:.2e}  elements out of tol {bad}  spike-token bad {spike_bad}", flush=True)
    e0 = bench.empty_cost()
    t = bench.time_graph(lambda: mod.g30_run(x, wp, out, N, S, scale, hdr)) - e0
    t0 = bench.time_graph(lambda: mod0.g30_run(x, wp, out, N, S, scale, hdr)) - e0
    tc = bench.time_graph(lambda: torch.mm(x, w.t())) - e0
    print(f"{name} N{N} K{K} S{S} {' '.join(DEFS)} ESCW={int(ESCW)} escapes {nesc} ({nesc / N:.2f}/row, {nsub} subnormal v): g30+fix {t:.2f} | "
          f"g30 nofix {t0:.2f} | cuBLAS {tc:.2f} us", flush=True)
    del w, x, out, wp, ref


for sh in [] if os.environ.get("NOMAIN") else sys.argv[1:] or ["unp:8192:4096:1", "pool:4096:8192:2"]:
    nm, N, K, S = sh.split(":")
    run(nm, int(N), int(K), int(S))
