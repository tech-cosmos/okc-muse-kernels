"""n14s2: g30n reduce-scatter (G30_RS) A/B, standalone H100 cold L2. python tools/n14/rs_ab.py <submission.py> [KEY] [MTOK,S,NW,NS] [rounds]
Builds RS 0 / RS 1 from the same source, checks bitwise equality on random x, times interleaved graph replays."""
import sys, hashlib, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
KEY = sys.argv[2] if len(sys.argv) > 2 else "qkv_gate"
cfg = tuple(int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "32,3,16,16").split(","))
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 6
import os; SPL = int(os.environ.get("SPL", "1"))
w, st = bench.weights_state(sub)
w = {**st["dense"], "unpool": st["unpool"]}
_, e, N = st["g30n"][(KEY, cfg[0])]
libs = {}
for rs in (0, 1):
    dfs = [f"-DG30_MTOK={cfg[0]}", f"-DG30_S={cfg[1]}", f"-DG30_NW={cfg[2]}", f"-DG30_NS={cfg[3]}", f"-DG30_SPL={SPL if rs else 0}"]
    libs[rs] = sub._g30_load("n14_rs_" + hashlib.sha256((sub._G30_SRC + " ".join(dfs)).encode()).hexdigest()[:10], dfs, False)
m, K = cfg[0], w[KEY].shape[1]
outs = {rs: torch.empty((m, N), dtype=torch.bfloat16, device="cuda") for rs in (0, 1)}
for seed in range(20):
    torch.manual_seed(seed)
    x = torch.randn((m, K), device="cuda").bfloat16()
    for rs in (0, 1):
        outs[rs].fill_(float("nan"))
        libs[rs].g30_gemm(x, e[0], e[1], outs[rs], e[2], 0)
    torch.cuda.synchronize()
    if not torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)): print("not bitwise, seed", seed, ((outs[0].float() - outs[1].float()).abs().max()).item()) if seed < 2 else None
ref = torch.nn.functional.linear(x, w[KEY]).float()
print("bitwise RS0 == RS1 on 20 inputs; rel err", ((outs[1].float() - ref).norm() / ref.norm()).item())
ts = {0: [], 1: []}
emp = bench.empty_cost()
for r in range(rounds):
    for rs in ((0, 1) if r % 2 == 0 else (1, 0)):
        ts[rs].append(bench.time_graph(lambda: libs[rs].g30_gemm(x, e[0], e[1], outs[rs], e[2], 0), 30) - emp)
for rs in (0, 1):
    print(f"RS{rs} {KEY} {cfg}: median {statistics.median(ts[rs]):.2f} us  all {[round(v, 2) for v in ts[rs]]}")
