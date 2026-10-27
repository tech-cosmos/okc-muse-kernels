"""s99 g30 prototype: build + correctness (vs x @ decoded-W) + cold-L2 graph timing vs cuBLAS (H100).
usage: python dev/s99_g30.py [name:N:K:S ...]  env DEFS="-DG30_NS=8"."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch
from torch.utils.cpp_extension import load_inline
import bench
layer = bench.load_sub("submission.py")

DEFS = os.environ.get("DEFS", "").split()
mod = load_inline("g30_" + __import__("hashlib").md5((" ".join(DEFS) + open(os.path.join(HERE, "s99_g30.cu")).read()).encode()).hexdigest()[:10],
                  cpp_sources="void g30_run(torch::Tensor x, torch::Tensor wp, torch::Tensor out, int64_t N, int64_t S, double scale);",
                  cuda_sources=open(os.path.join(HERE, "s99_g30.cu")).read(), functions=["g30_run"],
                  extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a", "-lineinfo"] + DEFS,
                  extra_ldflags=["-lcuda"], verbose=False)


def pack30(code, sm):
    """[N, K] codes / sign|mantissa bytes -> [N/64][K/64][6144 B] stages: S bytes [2][128 thr][16], words [128][4]."""
    n, k = code.shape
    G, KS = n // 64, k // 64
    # row = 16 w + 8 rr + gq ; k = 16 j + 8 kh + 2 q + h ; element e = (kh, rr, h) ; thread = (w, gq, q)
    perm = lambda t: t.view(G, 4, 2, 8, KS, 4, 2, 4, 2).permute(0, 4, 1, 3, 7, 5, 6, 2, 8).reshape(G, KS, 128, 4, 8)  # noqa: E731
    b3, v = layer._dec3(perm(sm), perm(code))  # [G, KS, 128, 4, 8] uint8, [G, KS, 128, 4] int32
    Sb = b3.view(G, KS, 128, 2, 16).permute(0, 1, 3, 2, 4).reshape(G, KS, 4096)
    Nb = v.contiguous().view(torch.uint8).view(G, KS, 2048)
    return torch.cat([Sb, Nb], 2).contiguous()


def run(name, N, K, S):
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    x = torch.randn(128, K, device="cuda").bfloat16()
    c = layer._compress(w)
    assert c is not None
    code, sm, hdr, scale = c
    wp = pack30(code, sm)
    dec = (((sm.int() & 0x80) << 8) | ((code.int() + 112) << 7) | (sm.int() & 0x7F))
    dec = torch.where(dec >= 32768, dec - 65536, dec).to(torch.int16).view(torch.bfloat16)  # w / scale, escapes = 2^-15
    ref = (x.float() @ dec.float().t()) * scale
    out = torch.empty(128, N, device="cuda", dtype=torch.bfloat16) if S == 1 else torch.empty(S, 128, N, device="cuda")
    mod.g30_run(x, wp, out, N, S, scale)
    torch.cuda.synchronize()
    got = out.float() if S == 1 else out.sum(0)
    err = ((got - ref).norm() / ref.norm()).item()
    e0 = bench.empty_cost()
    t = bench.time_graph(lambda: mod.g30_run(x, wp, out, N, S, scale)) - e0
    tc = bench.time_graph(lambda: torch.mm(x, w.t())) - e0
    mb = N * K * 1.5 / 1e6
    print(f"{name} N{N} K{K} S{S} {' '.join(DEFS)}: err {err:.1e} | g30 {t:.2f} us ({mb / t:.2f} TB/s 12-bit) | "
          f"cuBLAS {tc:.2f} us ({N * K * 2 / 1e6 / tc:.2f} TB/s)", flush=True)
    del w, x, out, wp, dec, ref


for sh in sys.argv[1:] or ["unp:8192:4096:1", "pool:4096:8192:2"]:
    nm, N, K, S = sh.split(":")
    run(nm, int(N), int(K), int(S))
