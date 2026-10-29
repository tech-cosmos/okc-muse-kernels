"""n7s50: vLLM Humming MoE (NVFP4 W4A16, INDEXED) vs our submission on the moe board's routings (H100).
Benchmark only (Humming is never imported by a submission). Run from muse_moe_w4a16/:
  PYTHONPATH=/tmp/n7s50/humming python tools/n7s50/hum.py [bs ...]
Humming path = gemm13 (indexed, top_k 8) -> silu*up (torch) -> gemm2 (indexed, top_k 1) -> norm + weighted sum (torch).
moe_align (sorted_ids) is built OUTSIDE the timed graph (generous to Humming). Times: CUDA-graph replay, cold L2, median.
"""
import json, math, sys, statistics, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import bench, reference, submission
from humming.layer import HummingLayer
from humming.config import GemmType
from humming.tune import get_heuristics_config

E, D, H = 32, reference.D_EXPERT, reference.D_HIDDEN
W, ST = bench.state()

def mk(n, k, w, sf, gs):
    L = HummingLayer(shape_n=n, shape_k=k, num_experts=E,
                     weight_config={"quant_method": "modelopt", "quant_algo": "nvfp4"},
                     torch_dtype=torch.bfloat16).cuda()
    L.load_from_tensors({"weight": w[:E].contiguous(), "weight_scale": sf[:E].contiguous(),
                         "weight_scale_2": gs[:E].contiguous()})
    L.transform()
    return L

L13 = mk(2 * D, H, W["w13"], W["w13_sf"], W["w13_gs"])
L2 = mk(H, D, W["w2"], W["w2_sf"], W["w2_gs"])
CC = json.dumps({"use_f16_accum": False, "use_m_major_input_scale": False, "gemm_type": "indexed"})

def tcfg(L, routed):
    tc = get_heuristics_config(layer_config=L.humming_config, gemm_type=GemmType.INDEXED)
    for lo, hi, c in tc:
        if lo < routed <= hi:
            return tc, c
    raise RuntimeError(routed)

def align(ids_flat, bsz):
    st, ex = [], []
    for e in range(E):
        t = torch.where(ids_flat == e)[0]
        nb = math.ceil(t.numel() / bsz)
        st.append(F.pad(t, (0, nb * bsz - t.numel()), value=ids_flat.numel()))
        ex += [e] * nb
    s = torch.cat(st).int()
    return s, torch.tensor(ex, dtype=torch.int32, device="cuda"), torch.tensor(s.numel(), dtype=torch.int32, device="cuda")

def humming_fns(data):
    x, ids, tw = data["hidden_states"], data["topk_ids"], data["topk_weights"]
    M = x.shape[0]
    fl = ids.reshape(-1).long()
    loc = (fl < E)
    routed = int(loc.sum())
    tc13, c13 = tcfg(L13, M * 8)
    tc2, c2 = tcfg(L2, M * 8)
    s13, e13, n13 = align(torch.where(loc, fl, torch.full_like(fl, 10**6)), c13["block_shape"][0])
    s2, e2, n2 = align(torch.where(loc, fl, torch.full_like(fl, 10**6)), c2["block_shape"][0])
    tc13, tc2 = json.dumps(tc13), json.dumps(tc2)
    o13 = torch.zeros((M * 8, 2 * D), dtype=torch.bfloat16, device="cuda")
    o2 = torch.zeros((M * 8, H), dtype=torch.bfloat16, device="cuda")
    pw = W["post_norm_w"].float() + 1.0
    wl = (tw.reshape(-1) * loc).view(M, 8, 1)
    def g13():
        L13(inputs=x, outputs=o13, sorted_ids=s13, expert_ids=e13, num_tokens_padded=n13, top_k=8,
            compute_config=CC, tuning_config=tc13)
    def g2(inter):
        L2(inputs=inter, outputs=o2, sorted_ids=s2, expert_ids=e2, num_tokens_padded=n2, top_k=1,
           compute_config=CC, tuning_config=tc2)
    state = {}
    def full():
        g13()
        inter = (F.silu(o13[:, :D].float()) * o13[:, D:].float()).to(torch.bfloat16)
        g2(inter)
        y = o2.float() * pw
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + reference.POST_NORM_EPS)
        return (y.view(M, 8, H) * wl).sum(1).to(torch.bfloat16)
    inter0 = torch.randn((M * 8, D), device="cuda").to(torch.bfloat16)
    return dict(full=full, g13=g13, g2=lambda: g2(inter0)), (c13, c2, routed, int(e13.numel()))

def main():
    bss = [int(a) for a in sys.argv[1:]] or [1, 16, 32, 8, 128, 256]
    seeds = (7, 8, 9)
    # correctness of the Humming path on one routing per bs
    for bs in bss:
        d = reference.generate_input(bs=bs, sharding="ep8", seed=2000 + bs)
        fns, _ = humming_fns(d)
        ok, msg = reference.check_implementation(d, fns["full"](), W)
        print(f"humming correctness bs {bs}: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
    ec = bench.empty_cost()
    print(f"{'bs':>4} {'ours':>8} {'hum_full':>9} {'hum_g13':>8} {'hum_g2':>8} {'g13+g2':>8}  cfg13 | cfg2", flush=True)
    for bs in bss:
        r = {k: [] for k in ("ours", "full", "g13", "g2")}
        for sd in seeds:
            d = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
            r["ours"].append(bench.time_graph(lambda: submission.kernel_fn(d, ST), 30) - ec)
            fns, info = humming_fns(d)
            for k in ("full", "g13", "g2"):
                r[k].append(bench.time_graph(fns[k], 30) - ec)
        m = {k: statistics.median(v) for k, v in r.items()}
        c13, c2, routed, nblk = info
        f = lambda c: f"{c.get('mma_type')} blk{tuple(c['block_shape'])} wp{tuple(c['warp_shape'])} sk{int(c.get('use_stream_k', True))} " \
                      f"stg{c.get('num_stages')} cps{c.get('num_ctas_per_sm', c.get('min_ctas_per_sm'))} ws{int(bool(c.get('use_warp_spec')))} tma{int(bool(c.get('use_tma')))}"
        print(f"{bs:4d} {m['ours']:8.1f} {m['full']:9.1f} {m['g13']:8.1f} {m['g2']:8.1f} {m['g13'] + m['g2']:8.1f}  {f(c13)} | {f(c2)}", flush=True)
        if bs == bss[0]:
            print("  full cfg13:", json.dumps(c13), flush=True)

main()
