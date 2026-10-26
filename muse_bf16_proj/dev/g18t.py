# gemv18 timeline (G18_TRACE): python dev/g18t.py M,KC,AP,NS layer   -> per-block event times (us from the earliest start)
import os, sys, torch, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
M, KC, AP, NS = map(int, sys.argv[1].split(","))
os.environ["DEFS"] = ",".join(["G18_TRACE"] + [d for d in os.environ.get("TDEFS", "").split(",") if d])
os.environ["SH"] = "none"
import importlib.util
spec = importlib.util.spec_from_file_location("g18m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "g18.py"))
src = open(spec.origin).read().replace('functions=["g18_gemm"]', 'functions=["g18_gemm", "g18_trace"]').replace(
    'int64_t, int64_t, int64_t, int64_t, int64_t);",', 'int64_t, int64_t, int64_t, int64_t, int64_t); torch::Tensor g18_trace();",')
g = {"__file__": spec.origin, "__name__": "g18m"}
exec(compile(src, spec.origin, "exec"), g)
layer = sys.argv[2]
n = {"qkv_gate": 2304, "pooler": 4096, "unpooler": 8192}[layer]
W = g["Wr"][layer][:n].contiguous(); k = W.shape[1]
P, hdr, scale = g["pack18"](W, KC, AP)
x = torch.randn(M, k, device="cuda").bfloat16(); out = torch.empty(M, n, dtype=torch.bfloat16, device="cuda")
fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
for _ in range(3):
    fl.add_(1); torch.cuda.synchronize()
    g["mod"].g18_gemm(P, hdr, scale, x, out, g["wsf"], g["cnt"], KC, AP, NS, g["GRID"], 0); torch.cuda.synchronize()
tr = g["mod"].g18_trace().cpu()
t0 = tr[:, 0].min()
names = {0: "start", 1: "prodEnd", 2: "x1", 3: "x2", 4: "consEnd"}
for b in list(range(0, 132, 13)) + [131]:
    row = tr[b]
    ev = [(int(i), (int(row[i]) - int(t0)) / 1000) for i in range(64) if row[i] > 0 and int(row[i]) >= int(t0)]
    print(b, " ".join(f"{names.get(i, i)}:{v:.1f}" for i, v in ev))
print("max consEnd", (tr[:, 4].max() - t0).item() / 1000, "max any", (tr.max() - t0).item() / 1000)
