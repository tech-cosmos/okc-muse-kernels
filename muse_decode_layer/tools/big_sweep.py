"""moe_big variants (NS, NW, MB) on the layer's own routing at T=128/256 (reference front -> biased top 8,
local ids): gemm13 + gemm2 time, graph replay, cold L2, median. Also checks outputs equal the default's."""
import sys, re, fcntl, statistics, types, torch
sys.path.insert(0, ".")
import reference, bench
VARS = {118: (8, 8, 2), 201: (4, 8, 2), 202: (4, 16, 2), 203: (8, 16, 1), 204: (4, 8, 3), 205: (4, 16, 3), 206: (8, 4, 4)}
src = open("moe_src.py").read()
cases = "".join(f"    case {c}: return launch_big<GATED, {a}, {b}, {m}>(wq, ws, gs, lut, x, meta, sr, out, st);     \\\n"
                for c, (a, b, m) in VARS.items() if c != 118)
src = src.replace("    case 118: return launch_big<GATED, 8, 8, 2>(wq, ws, gs, lut, x, meta, sr, out, st);     \\\n",
                  "    case 118: return launch_big<GATED, 8, 8, 2>(wq, ws, gs, lut, x, meta, sr, out, st);     \\\n" + cases)
src = src.replace("_CFGS = (10, 11, 22, 118)", "_CFGS = (10, 11, 22, " + ", ".join(map(str, VARS)) + ")")
mod = types.ModuleType("moe_sweep"); exec(compile(src, "moe_sweep", "exec"), mod.__dict__)
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = mod.prepare({"w13": w["w13"], "w13_sf": w["w13_sf"], "w13_gs": w["w13_gs"], "w2": w["w2"], "w2_sf": w["w2_sf"],
                  "w2_gs": w["w2_gs"], "post_norm_w": w["post_expert_norm"]})
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
def t(fn, n=30):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(n):
            torch.sum(flush.view(torch.int64), 0, out=sink); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)
lib = st["lib"]
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or load not in sys.argv[1:]:
        continue
    d = bench.make_input(c, w, 99)
    fr = reference.ref_front(d, w)
    ids = fr["biased"].topk(8, dim=-1).indices.int()
    loc = ((ids - fr["local_start"]) % 256).int(); loc = torch.where(loc < 32, loc, -1).int().contiguous()
    T = ids.shape[0]
    x = (torch.randn(T, 4096, device="cuda") ).bfloat16()
    meta = torch.empty(68, dtype=torch.int32, device="cuda"); sr = torch.empty(T * 8, dtype=torch.int32, device="cuda")
    inter = torch.empty((T * 8, 4096), dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(inter)
    ref = None
    for code, v in VARS.items():
        def run(code=code):
            lib.moe_route_launch(loc, 0, 32, meta, sr)
            lib.moe_gemm13(code, st["w13q"], st["w13s"], st["gs13"], st["lut"], x, meta, sr, inter)
            lib.moe_gemm2(code, st["w2q"], st["w2s"], st["gs2"], st["lut"], inter, meta, sr, y)
        def r13(code=code):
            lib.moe_route_launch(loc, 0, 32, meta, sr)
            lib.moe_gemm13(code, st["w13q"], st["w13s"], st["gs13"], st["lut"], x, meta, sr, inter)
        run(); torch.cuda.synchronize()
        n = int((loc >= 0).sum())
        out = y.clone()
        if ref is None: ref = out
        same = torch.equal(out[:0], ref[:0])
        sel = torch.sort(sr[:n].long()).values
        diff = (out[sel].float() - ref[sel].float()).abs().max().item()
        print(f"{load} T={T} cfg {code} {v}: route+13+2 {t(run):7.2f} us  route+13 {t(r13):7.2f}  maxdiff {diff:.2e}")
