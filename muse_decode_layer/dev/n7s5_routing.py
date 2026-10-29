"""n7s5: _routing with one packed int64 max per top-k step (key bits + index) instead of max + argmin + score-sum.
Checks bitwise equality of ids/lids/wts vs layer._routing on random partials (incl. forced ties), then times both
in a CUDA graph (chain of launches, cold-ish partials). python dev/n7s5_routing.py"""
import sys, fcntl, torch, triton, triton.language as tl
sys.path.insert(0, ".")
import bench
L = bench.load_sub("submission.py")


@triton.jit
def _routing2(part_ptr, bias_ptr, ls_ptr, ids_ptr, lids_ptr, wts_ptr, T, E: tl.constexpr, KS: tl.constexpr,
              TOPK: tl.constexpr, LOCAL: tl.constexpr, MULT: tl.constexpr, REPS: tl.constexpr,
              SKS: tl.constexpr = 0, ST: tl.constexpr = 0, TRIG: tl.constexpr = False, PDL: tl.constexpr = False):
    if TRIG:
        L.gdc_launch_dependents()
    if PDL:
        L.gdc_wait()
    t = tl.program_id(0)
    e = tl.arange(0, E)
    logit = tl.zeros((E,), dtype=tl.float32)
    for ks in tl.static_range(KS):
        if ST:
            logit += tl.load(part_ptr + ks * SKS + t * ST + e)
        else:
            logit += tl.load(part_ptr + (ks * T + t) * E + e)
    score = 1.0 / (1.0 + tl.exp(-logit * MULT))
    key = score + tl.load(bias_ptr + e)
    kb = key.to(tl.int32, bitcast=True)
    kb = kb ^ ((kb >> 31) & 0x7FFFFFFF)  # order-preserving signed int of the float (no NaN, no -0.0 here)
    pk = (kb.to(tl.int64) << 32) | (E - 1 - e).to(tl.int64)  # max: largest key, then lowest index
    j = tl.arange(0, TOPK)
    ids = tl.zeros((TOPK,), dtype=tl.int32)
    for i in tl.static_range(TOPK):
        best = tl.max(pk, 0)
        idx = (E - 1 - (best & 0xFFFFFFFF)).to(tl.int32)
        ids = tl.where(j == i, idx, ids)
        pk = tl.where(e == idx, -9223372036854775807 - 1, pk)
    picked = tl.gather(score, ids, 0)
    wts = picked / (tl.sum(picked, 0) + REPS)
    ls = tl.load(ls_ptr)
    loc = (ids - ls + E) % E
    loc = tl.where(loc < LOCAL, loc, -1)
    tl.store(ids_ptr + t * TOPK + j, ids)
    tl.store(lids_ptr + t * TOPK + j, loc)
    tl.store(wts_ptr + t * TOPK + j, wts)


def run(kern, part, bias, ls, T, KS, nw, ST=0, SKS=0):
    ids = torch.empty((T, 8), dtype=torch.int32, device="cuda"); lids = torch.empty_like(ids)
    wts = torch.empty((T, 8), dtype=torch.float32, device="cuda")
    kern[(T,)](part, bias, ls, ids, lids, wts, T, E=256, KS=KS, TOPK=8, LOCAL=32, MULT=L.ROUTER_MULTIPLIER,
               REPS=L.ROUTE_EPS, SKS=SKS, ST=ST, num_warps=nw)
    return ids, lids, wts


if __name__ == "__main__":
    torch.manual_seed(0)
    bad = 0; n = 0
    for T, KS, nw in ((16, 16, 2), (32, 16, 2), (128, 2, 1), (256, 2, 1), (1, 8, 2), (8, 16, 2)):
        for trial in range(60):
            part = torch.randn((KS, T, 256), device="cuda") * (0.3 if trial % 3 else 3.0)
            if trial % 4 == 0:  # force exact ties in key: duplicate logits and zero bias on some experts
                part[:, :, 128:192] = part[:, :, 0:64]
            bias = torch.randn(256, device="cuda") * 0.01 * (trial % 5)
            if trial % 4 == 0:
                bias[128:192] = bias[0:64]
            ls = torch.tensor([32 * (trial % 8)], dtype=torch.int32, device="cuda")
            a = run(L._routing, part, bias, ls, T, KS, nw); b = run(_routing2, part, bias, ls, T, KS, nw)
            n += 1
            if not all(torch.equal(x, y) for x, y in zip(a, b)):
                bad += 1
                print("MISMATCH", T, KS, trial)
    print(f"bitwise: {bad}/{n} bad")
    # timing: graph of 50 dependent-free launches each, per launch us (includes launch gaps; same for both)
    for T, KS, nw in ((16, 16, 2), (32, 16, 2), (128, 2, 1), (256, 2, 1)):
        part = torch.randn((KS, T, 256), device="cuda"); bias = torch.zeros(256, device="cuda")
        ls = torch.zeros(1, dtype=torch.int32, device="cuda")
        res = {}
        for name, k in (("old", L._routing), ("new", _routing2)):
            run(k, part, bias, ls, T, KS, nw)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(50):
                    run(k, part, bias, ls, T, KS, nw)
            ts = []
            with open("/tmp/okc-gpu.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                for _ in range(30):
                    s = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
                    s.record(); g.replay(); e2.record(); torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e2) * 1e3 / 50)
            res[name] = sorted(ts)[len(ts) // 2]
        print(f"T={T} KS={KS} warps={nw}: old {res['old']:.2f} us/launch, new {res['new']:.2f}")
