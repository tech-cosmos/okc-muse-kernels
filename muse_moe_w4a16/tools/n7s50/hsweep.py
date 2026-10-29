"""n7s50: sweep Humming tuning configs around its heuristic pick (per GEMM, one routing, seed 7). Run like hum.py:
  PYTHONPATH=/tmp/n7s50/humming python tools/n7s50/hsweep.py bs [bs ...]
Prints the heuristic time and the best variant per GEMM; illegal/failed configs are skipped."""
import copy, json, sys, itertools, torch
sys.argv, BSS = sys.argv[:1], [int(a) for a in sys.argv[1:]]
src = open("tools/n7s50/hum.py").read().replace("\nmain()\n", "\n")
hum = type(sys)("hum"); exec(compile(src, "hum.py", "exec"), hum.__dict__)
import bench, reference

def variants(c):
    out = []
    mt = c["mma_type"]
    bm = c["block_shape"][0]
    for bn, bk, cps, stg, sk in itertools.product((64, 128, 256), (64, 128, 256), (1, 2, 3, 4), (3, 4, 6, 8), (True, False)):
        v = copy.deepcopy(c)
        v["block_shape"] = [bm, bn, bk]
        wn = min(64, bn // 2) if mt == "mma" else c["warp_shape"][1]
        v["warp_shape"] = [bm, wn, min(bk // 2, 64) if mt == "mma" else c["warp_shape"][2]]
        v["num_ctas_per_sm"], v["num_stages"], v["use_stream_k"] = cps, stg, sk
        out.append(v)
    return out

ec = bench.empty_cost()
for bs in BSS:
    d = reference.generate_input(bs=bs, sharding="ep8", seed=7)
    fns, (c13, c2, routed, nblk) = hum.humming_fns(d)
    M = bs
    for name, L, c, fn in (("g13", hum.L13, c13, "g13"), ("g2", hum.L2, c2, "g2")):
        base = bench.time_graph(fns[fn], 20) - ec
        res = []
        import random; random.seed(bs)
        vs = variants(c); random.shuffle(vs)
        for v in vs[:40]:
            tc = json.dumps([[0, 1 << 30, v]])
            try:
                o = torch.zeros((M * 8, L.shape_n), dtype=torch.bfloat16, device="cuda")
                x = d["hidden_states"] if name == "g13" else torch.randn((M * 8, hum.D), device="cuda").to(torch.bfloat16)
                s, e, n = hum.align(torch.where(d["topk_ids"].reshape(-1).long() < 32, d["topk_ids"].reshape(-1).long(), 10**6), v["block_shape"][0])
                tk = 8 if name == "g13" else 1
                f = lambda: L(inputs=x, outputs=o, sorted_ids=s, expert_ids=e, num_tokens_padded=n, top_k=tk, compute_config=hum.CC, tuning_config=tc)
                f(); torch.cuda.synchronize()
                res.append((bench.time_graph(f, 20) - ec, v))
            except Exception:
                continue
        res.sort(key=lambda r: r[0])
        b = res[0] if res else (float("nan"), {})
        print(f"bs {bs} {name}: heuristic {base:.1f} us | best of {len(res)} legal {b[0]:.1f} us "
              f"blk{b[1].get('block_shape')} cps{b[1].get('num_ctas_per_sm')} stg{b[1].get('num_stages')} sk{b[1].get('use_stream_k')}", flush=True)
