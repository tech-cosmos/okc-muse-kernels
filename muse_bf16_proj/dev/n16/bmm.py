"""n16: router m128/m256 as ONE strided-batched cuBLAS GEMM over K splits (batch s = K slice; N = 512 = [hi; lo])
into an FP32 ws [S, m, 512], then ONE reduce kernel summing splits and halves (vs torch.mm stacked + splitKreduce + add)."""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import bench, reference

@triton.jit
def _red(ws, out, M, S: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLK + tl.arange(0, BLK)
    r = o // 256; c = o % 256
    ok = r < M
    acc = tl.zeros([BLK], tl.float32)
    for s in tl.static_range(S):
        base = ws + s * M * 512 + r * 512 + c
        acc += tl.load(base, mask=ok, other=0.) + tl.load(base + 256, mask=ok, other=0.)
    tl.store(out + o, acc, mask=ok)

W = reference.generate_weights(reference.WEIGHTS_SEED)["router"].cuda()
hi = W.to(torch.bfloat16); lo = (W - hi.float()).to(torch.bfloat16)
hlt = torch.cat([hi, lo], 0).t().contiguous()  # [K, 512]
e = bench.empty_cost()
def base(x):
    acc = torch.mm(x, hlt, out_dtype=torch.float32)
    return acc[:, :256] + acc[:, 256:]
for m in [int(a) for a in sys.argv[1:]] or [128, 256]:
    x = torch.randn(m, 8192, device="cuda").to(torch.bfloat16)
    ref = x.double() @ W.double().t()
    t0 = bench.time_graph(lambda: base(x), reps=200) - e
    res = [f"base {t0:.2f}"]
    for S in [int(v) for v in os.environ.get("SS", "2,4,8").split(",")]:
        KC = 8192 // S
        ws = torch.empty(S, m, 512, device="cuda")
        out = torch.empty(m, 256, device="cuda")
        xb = x.view(m, S, KC).transpose(0, 1)          # [S, m, KC], batch stride KC
        wb = hlt.view(S, KC, 512)                       # [S, KC, 512]
        def f():
            torch.bmm(xb, wb, out_dtype=torch.float32, out=ws) if False else torch.bmm(xb, wb, out=None)
        try:
            def g():
                torch.baddbmm(ws, xb, wb, beta=0, out_dtype=torch.float32, out=ws) if False else None
            def run():
                r = torch.bmm(xb, wb, out_dtype=torch.float32)
                _red[(m * 256 + 1023) // 1024,](r, out, m, S, 1024)
                return out
            o = run(); torch.cuda.synchronize()
            err = ((o.double() - ref).norm() / ref.norm()).item()
            t = bench.time_graph(run, reps=200) - e
            res.append(f"S{S} {t:.2f} (err {err:.1e})")
        except Exception as ex:
            res.append(f"S{S} fail {str(ex)[:80]}")
    print(f"m{m}: " + " | ".join(res), flush=True)
