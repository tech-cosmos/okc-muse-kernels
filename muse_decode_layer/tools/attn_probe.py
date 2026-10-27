"""Key-coverage oracle for the layer's attention (proj s78, lane (b)). The full-layer check's 3-15% bound can hide a
dropped/duplicated tile or a window off-by-one (one 128-key tile of 80K keys moves the output ~0.2%). Here:
  K cache = 0 (every cached key scores 0), V cache = 0 except PROBES: for each (request, KV head), value dim i holds
  1.0 at one cached position p_i (64 probes per pass); qkv's k and v parts = 0 (new keys score 0 too: uniform
  softmax, independent of the query's precision), q/gate random (BF16 qkv as given).
So each output row before the norm is (sum of the VISIBLE probe one-hots) / softmax denominator: a missed key zeroes a
dim, an extra key (outside the window, a masked verify key) adds one: ~1/sqrt(#probes) relative error, far above the
FP32/BF16 noise (~1e-3). Probes sweep [first visible - 2, seq) of every request; passes cover all positions when
--passes 0 (exhaustive), else that many evenly spread passes (always incl. the first and last).
Reference = reference._attention with qkv_gate = I (exact FP32: TF32 off), on our BF16 qkv. Also checks that the new cache
entries (K and V at cache_loc) are written as 0 (no stale/garbage rows).
python tools/attn_probe.py [submission.py] [--passes=N (8)] [--only=substring] [--tol=0.03] [--control]
Cases: the 12 scored shapes + the 13 test shapes (+ a few odd lengths). Calls sub.attn_kernel_fn eagerly (csl_pw off,
BF16 qkv): every attention path the layer dispatches to at those shapes (csl/vtk/nvs/nds/gluon/triton)."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
path = args[0] if args else "submission.py"
npass_max = int(flags.get("--passes", 8))
only = flags.get("--only", "")
tol = float(flags.get("--tol", 0.03))
control = "--control" in flags
torch.backends.cuda.matmul.allow_tf32 = False
sub = bench.load_sub(path)
w = reference.generate_weights(reference.WEIGHTS_SEED)
sub.prepare(w)  # resolves the attention extensions (csl/nds/nvs JIT)
D, HQ, HKV = reference.HEAD_DIM, reference.HQ, reference.HKV
NQKV = 2 * reference.Q_SIZE + 2 * reference.KV_SIZE
EYE = {"qkv_gate": torch.eye(NQKV, device="cuda")}
ONE = torch.tensor(1.0).to(reference.FP8).view(torch.uint8).item()

cases = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED] + [(f"test{i}", c) for i, c in enumerate(bench.TESTS)]
cases += [("odd sl b3 L8 p64", dict(batch=3, ctx=2049, layer="sliding", q_len=8, page=64)),
          ("odd sl b2 L1 c2050", dict(batch=2, ctx=2050, layer="sliding")),
          ("odd fu b7 L1 c4097", dict(batch=7, ctx=4097, layer="full", spread="mixed", page=1)),
          ("odd fu b24 L8 c9001", dict(batch=24, ctx=9001, layer="full", q_len=8, spread="agentic", page=1))]


def probe_lists(data):
    """Per request: the probed positions [lo, seq) (lo = 2 below the first key any of its queries sees)."""
    cfg, out = data["config"], []
    for b in range(cfg["batch"]):
        seq = int(data["seq_lens"][b])
        first = max(0, seq - cfg["window"] + 1) if cfg["window"] else 0
        out.append(torch.arange(max(0, first - 2), seq, device="cuda"))
    return out


def run_case(name, c, seed):
    data = bench.make_input(c, w, seed)
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn((T, NQKV), device="cuda", generator=g)
    qkv[:, reference.Q_SIZE: reference.Q_SIZE + 2 * reference.KV_SIZE] = 0  # k and v parts: every score is 0
    qkv = qkv.to(torch.bfloat16)
    kc, vc, pt, size = data["k_cache"], data["v_cache"], data["page_table"].long(), cfg["page_size"]
    k8, v8 = kc.view(torch.uint8), vc.view(torch.uint8)
    vflat = v8.view(-1, HKV, D)
    lists = probe_lists(data)
    need = max((len(p) + D - 1) // D for p in lists)
    if npass_max and need > npass_max:
        sel = sorted({round(i * (need - 1) / (npass_max - 1)) for i in range(npass_max)}) if npass_max > 1 else [0]
    else:
        sel = list(range(need))
    worst, worst_at, bad, kbad = 0.0, None, 0, 0.0
    for p in sel:
        k8.zero_()
        v8.zero_()
        k8.view(-1, HKV, D)[data["cache_loc"]] = 0x44  # stale bytes in the new rows: the kernel must write 0 there
        vflat[data["cache_loc"]] = 0x44
        for b, pos in enumerate(lists):
            n = len(pos)
            stride = (n + D - 1) // D
            for h in range(HKV):
                pp = (p + h * (stride // 2)) % stride if stride else 0
                idx = pp + stride * torch.arange(D, device="cuda")
                dims = torch.arange(D, device="cuda")[idx < n]
                idx = idx[idx < n]
                q = pos[idx]
                rows = pt[b, q // size] * size + q % size
                vflat[rows, h, dims] = ONE
                if b == 0 and h == 0:
                    hide = (rows[len(rows) // 2], dims[len(dims) // 2])
        ref, k_new, _ = reference._attention(data, qkv.float(), EYE)
        if control:  # positive control: the kernel misses one of request 0's probes (= one dropped key)
            vflat[hide[0], 0, hide[1]] = 0
        out = sub.attn_kernel_fn({
            "o_pf": (0, 0), "csl_pw": False, "qkv": qkv.view(B, L, -1), "positions": data["positions"],
            "cache_loc": data["cache_loc"], "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
            "scheduler_metadata": data["scheduler_metadata"], "k_cache": kc, "v_cache": vc,
            "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": data["page_table"],
            "seq_lens": data["seq_lens"], "config": cfg, "qkv_ss": 0, "qkv_slabs": None})
        torch.cuda.synchronize()
        o = out.float().view(T, HQ, D)
        r = ref.view(T, HQ, D)
        err = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
        m = float(err.max())
        bad += int((err > tol).sum())
        if m > worst:
            t, hh = divmod(int(err.argmax()), HQ)
            worst, worst_at = m, (p, t // L, t % L, hh)
        nk, nv = k8.view(-1, HKV, D)[data["cache_loc"]], vflat[data["cache_loc"]]
        kbad = max(kbad, float(((nk & 0x7F) != 0).sum()), float(((nv & 0x7F) != 0).sum()))  # new K/V = +-0
        if kbad and p == sel[0]:
            print("  new K bytes", torch.unique(nk).tolist()[:8], "new V bytes", torch.unique(nv).tolist()[:8])
    ok = bad == 0 and kbad == 0
    print(f"{'PASS' if ok else 'FAIL'} {name:22s} seed {seed} passes {len(sel)}/{need} worst {worst:.2e} "
          f"(pass,b,l,head)={worst_at} rows>{tol}: {bad} newkv {kbad:.2e}", flush=True)
    del data
    torch.cuda.empty_cache()
    return ok


nbad = 0
for name, c in cases:
    if only and only not in name:
        continue
    nbad += not run_case(name, c, 777)
print(f"bad {nbad}")
