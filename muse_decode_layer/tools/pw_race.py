"""s63 attention-level race test for CSL_PW / vtk PW (pre-wait K/V gathers). Per input D and its step A (= D moved back
by L tokens, writing D's last L context rows), with a FIXED qkv per step (so attention is bitwise deterministic):
  ref    : eager A (no PW), sync, eager B (no PW)
  split  : graph GA = [attn A]; graph GB = [layer's qkv GEMM (trigger at start), attn B as its PDL secondary (PW)];
           GA then GB back to back, no host sync (cold-L2 flush between, as the judge) = the layer's situation
  split_nf: the same with no flush kernel between the graphs
  joint  : one graph [attn A, qkv GEMM, attn B PW] (positive control: A's cache writes are NOT ordered before B's
           pre-wait gathers here, so this may differ: shows the test can see a race)
  joint2 : one graph [attn A, attn B PW] (B a direct PDL secondary of A, which triggers at start: stronger control)
python tools/pw_race.py [submission.py] [n] [loads: d16,v1]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
loads = (sys.argv[3] if len(sys.argv) > 3 else "d16,v1").split(",")
sub = bench.load_sub(path)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = sub.prepare(w)
KEYS = ("positions", "cache_loc", "cache_seqlens", "cu_seqlens_q", "k_cache", "v_cache", "k_scale", "v_scale",
        "page_table", "seq_lens")


def back(d, L):
    a = dict(d)
    a["seq_lens"], a["positions"], a["cache_seqlens"] = d["seq_lens"] - L, d["positions"] - L, d["cache_seqlens"] - L
    B = d["seq_lens"].shape[0]
    pos = (a["seq_lens"].long()[:, None] + torch.arange(L, device="cuda")).reshape(B, L)
    page = d["config"]["page_size"]
    a["cache_loc"] = (torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page).reshape(-1)
    return a


for ld in loads:
    load = {"d16": "decode 16", "v1": "verify 1", "d1": "decode 1", "v16": "verify 16"}[ld]
    c = next(c for l, c in bench.SCORED if l == load and c["layer"] == "sliding")
    L, T = c["q_len"], c["batch"] * c["q_len"]
    sub._SLIDING[0] = True
    d0 = bench.make_input(c, w, 7000)
    cfg = d0["config"]
    wq = st["dense"]["qkv_gate"]
    # captured buffers: A and B share the caches
    bufA = {k: d0[k].clone() for k in KEYS}
    bufB = {k: (bufA[k] if k in ("k_cache", "v_cache") else d0[k].clone()) for k in KEYS}
    x = d0["attn_input"].clone()
    qA = torch.empty((c["batch"], L, wq.shape[0]), dtype=torch.bfloat16, device="cuda")
    qB = torch.empty_like(qA)

    def attn(buf, q, pw):
        return sub.attn_kernel_fn({**buf, "qkv": q, "scheduler_metadata": None, "config": cfg, "o_pf": (0, 0),
                                   "csl_pw": pw, "qkv_ss": 0, "qkv_slabs": None})

    def gemm():
        if T == 16:
            r = sub._g19(x, "qkv_gate", st, trig=1)
            assert r is not None
        else:
            r = sub._linear(x, wq, "qkv_gate", st)
        return r

    def cap(fn):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            with torch.cuda.graph(g, stream=s):
                r = fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        return g, r

    gA, oA = cap(lambda: attn(bufA, qA, 0))
    gB, oB = cap(lambda: (gemm(), attn(bufB, qB, 1))[1])
    gJ, oJ = cap(lambda: (attn(bufA, qA, 0), gemm(), attn(bufB, qB, 1))[2])
    gK, oK = cap(lambda: (attn(bufA, qA, 0), attn(bufB, qB, 1))[1])
    cnt = dict(nondet=0, split=0, joint=0, joint2=0, split_nf=0, skipped=0)
    for i in range(n):
        d = bench.make_input(c, w, 9001 + 31 * i)
        if int(d["seq_lens"].min()) <= L:
            cnt["skipped"] += 1
            continue
        a = back(d, L)
        q_a = torch.randn(qA.shape, device="cuda").to(torch.bfloat16)
        q_b = torch.randn(qB.shape, device="cuda").to(torch.bfloat16)
        x.copy_(d["attn_input"])
        kc0, vc0 = d["k_cache"].clone(), d["v_cache"].clone()

        def load_inputs():
            for k in KEYS:
                if k not in ("k_cache", "v_cache"):
                    bufA[k].copy_(a[k]); bufB[k].copy_(d[k])
            bufA["k_cache"].copy_(kc0); bufA["v_cache"].copy_(vc0)
            qA.copy_(q_a); qB.copy_(q_b)

        refs = []
        for _ in range(2):
            load_inputs(); torch.cuda.synchronize()
            attn(bufA, qA, 0); torch.cuda.synchronize()
            refs.append((attn(bufB, qB, 0).clone(), bufA["k_cache"].clone()))
            torch.cuda.synchronize()
        cnt["nondet"] += not torch.equal(refs[0][0], refs[1][0])
        load_inputs()
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gA.replay()
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gB.replay()
        torch.cuda.synchronize()
        cnt["split"] += not (torch.equal(oB, refs[0][0]) and torch.equal(bufA["k_cache"], refs[0][1]))
        load_inputs()
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gA.replay()
        gB.replay()  # no flush kernel between the two graphs
        torch.cuda.synchronize()
        cnt["split_nf"] += not (torch.equal(oB, refs[0][0]) and torch.equal(bufA["k_cache"], refs[0][1]))
        load_inputs()
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gJ.replay()
        torch.cuda.synchronize()
        cnt["joint"] += not torch.equal(oJ, refs[0][0])
        load_inputs()
        torch.sum(flush.view(torch.int64), 0, out=sink)
        gK.replay()
        torch.cuda.synchronize()
        cnt["joint2"] += not torch.equal(oK, refs[0][0])
        del d, a, kc0, vc0
    print(f"{load} sliding (T={T}) n {n}: {cnt}", flush=True)
    del gA, gB, gJ, gK, bufA, bufB
    torch.cuda.empty_cache()
