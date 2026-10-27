"""proj s91: interleaved timing of g30e variants (cold-L2 graphs, H100). usage: VARS="-DG30_ESC=0|;-DG30_ESC_NOFIX=1" python t30.py [name:N:K:S ...]
(';' separates variants, each a space-separated DEFS list; '' = default build). ROUNDS=n (default 3)."""
import os, sys
sys.argv, shapes = sys.argv[:1], sys.argv[1:]
os.environ["NOMAIN"] = "1"
import g30e as G  # noqa
import torch
import bench
VARS = os.environ.get("VARS", "-DG30_ESC=0;").split(";")
mods = [G.build(v.split()) for v in VARS]
for sh in shapes or ["unp:8192:4096:1", "pool:4096:8192:2"]:
    nm, N, K, S = sh.split(":"); N, K, S = int(N), int(K), int(S)
    g_ = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=g_) / K ** 0.5).bfloat16()
    code, sm, hdr, scale = G.layer._compress(w)
    wp = G.P.pack30(code, sm); x = torch.randn(128, K, device="cuda", generator=g_).bfloat16()
    out = torch.empty(128, N, device="cuda", dtype=torch.bfloat16) if S == 1 else torch.empty(S, 128, N, device="cuda")
    e0 = bench.empty_cost(); res = [[] for _ in mods]; rc = []
    for r in range(int(os.environ.get("ROUNDS", 3))):
        for j, m in enumerate(mods):
            res[j].append(bench.time_graph(lambda: m.g30_run(x, wp, out, N, S, scale, hdr)) - e0)
        rc.append(bench.time_graph(lambda: torch.mm(x, w.t())) - e0)
    print(f"{nm} N{N} K{K} S{S}: " + " | ".join(f"[{v or 'default'}] " + "/".join(f"{t:.2f}" for t in res[j]) for j, v in enumerate(VARS))
          + " | cuBLAS " + "/".join(f"{t:.2f}" for t in rc), flush=True)
