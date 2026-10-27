import sys, fcntl, zlib, base64, torch
sys.path.insert(0, "/tmp/moe18")
import reference, submission as S
w = reference.generate_weights(reference.WEIGHTS_SEED); st = S.prepare(w)
raw = zlib.decompress(base64.b64decode(S._DIAG_R)); p = 0; R = []
for load, seed, T in S._DIAG_HDR:
    loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda(); p += T * 8
    if T in (1, 8, 16, 32) and seed in (7, 11):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        R.append((load, seed, {"hidden_states": torch.randn(T, S.H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
            "topk_weights": tw / tw.sum(-1, keepdim=True), "config": {"bs": T, "d_hidden": S.H, "local_start": st["start"], "local_count": st["count"]}}))
lib = S._lib()
class W:
    def __init__(s, l, skip): s.l, s.skip = l, skip
    def __getattr__(s, k):
        if k in s.skip: return lambda *a: 0
        return getattr(s.l, k)
flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64); sink = torch.empty((), dtype=torch.int64, device="cuda")
def tg(data, skip):
    S._lib = lambda: W(lib, skip)
    S.kernel_fn(data, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): S.kernel_fn(data, st)
    S._lib = lambda: lib
    ts = []
    for _ in range(30):
        torch.sum(flush, 0, out=sink)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    st["cnt"].zero_()
    return sorted(ts)[15]
with open('/tmp/okc-gpu.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for load, seed, data in R:
        a = tg(data, ()); b = tg(data, ("moe_norm3_launch", "moe_norm2_launch"))
        print(f"{load} s{seed}: full {a:.1f} no-norm {b:.1f} (norm costs {a - b:.1f})", flush=True)
