"""Judge-like routing stress: per scored shape, capture kernel_fn once as a CUDA graph, then for many fresh
inputs copy them into the captured input tensors, flush L2, replay, and check the routing (and keep margins)
against reference.ref_front.  python tools/route_judge.py [submission.py] [n_inputs] [router_seeds] [filter]
Prints per shape: fails, worst pick gap (limit 0.05), worst dropped-keep margin (limit 0.03/0.04), and the
same for the FP8-query reference's own top 8 (what SGLang-like numerics would give)."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [11]
flt = sys.argv[4] if len(sys.argv) > 4 else ""
sub = bench.load_sub(path)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")


def margins(ids, biased, local_start):
    """(worst pick gap below the 8th best, worst reference-top8-local expert dropped: its lead over the 9th)."""
    best = biased.topk(9, dim=-1)
    top8, ninth = best.values[:, :8], best.values[:, 8:9]
    gap = float((top8[:, 7:8] - biased.gather(1, ids.long())).max())
    kept = (best.indices[:, :8, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices[:, :8] - local_start) % reference.N_EXPERTS < reference.LOCAL_EXPERTS
    lead = (top8 - ninth).masked_fill(~(local & ~kept), -1.0)
    return gap, float(lead.max())


def copy_into(dst, src):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            assert dst[k].shape == v.shape, k
            dst[k].copy_(v)


for rs in seeds:
    w = reference.generate_weights(reference.WEIGHTS_SEED, router_seed=rs)
    st = sub.prepare(w)
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name:
            continue
        data = bench.make_input(c, w, 7000)
        sub.kernel_fn(data, st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            sub.kernel_fn(data, st)
            with torch.cuda.graph(g, stream=s):
                out = sub.kernel_fn(data, st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        fails, wg, wl, eg, el = 0, 0.0, -1.0, 0.0, -1.0
        msgs = []
        for i in range(n):
            d = bench.make_input(c, w, 7001 + 17 * i + 1000 * rs)
            copy_into(data, d)
            torch.sum(flush.view(torch.int64), 0, out=sink)
            g.replay()
            torch.cuda.synchronize()
            front = reference.ref_front(d, w)
            ls = front["local_start"]
            err = reference.check_routing(out[2], front["biased"], ls)
            if err:
                fails += 1
                msgs.append(f"input {i}: {err}")
            gp, ld = margins(out[2], front["biased"], ls)
            wg, wl = max(wg, gp), max(wl, ld)
            f8 = reference.ref_front(d, w, torch.float8_e4m3fn)
            ids8 = f8["biased"].topk(8, dim=-1).indices.to(torch.int32)
            gp, ld = margins(ids8, front["biased"], ls)
            eg, el = max(eg, gp), max(el, ld)
            del d, front, f8
        print(f"rs {rs} {name:16s} fails {fails}/{n}  ours gap {wg:.4f} drop-lead {wl:.4f} | "
              f"fp8ref gap {eg:.4f} drop-lead {el:.4f}", flush=True)
        for m in msgs[:3]:
            print("   ", m, flush=True)
        del g, data, out
        torch.cuda.empty_cache()
    del st, w
    torch.cuda.empty_cache()
