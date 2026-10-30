"""fresh x5 oracle (4): print distinct local experts + their unique weight bytes for one scored shape, then run it 2x (for ncu).
python tools/fresh/sw/uniq.py submission.py '<load> <layer>'  (from muse_decode_layer/)"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference as R
sub = bench.load_sub(sys.argv[1])
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == sys.argv[2]:
        data = bench.make_input(c, w, 99)
        f = R.ref_front(data, w)
        ids = f["biased"].topk(R.TOP_K, dim=-1).indices
        loc = (ids.long() - f["local_start"]) % R.N_EXPERTS
        loc = loc[loc < R.LOCAL_EXPERTS]
        cnt = torch.bincount(loc.flatten(), minlength=32)
        per = sum(w[k][0].numel() * w[k][0].element_size() for k in ("w13", "w13_sf", "w2", "w2_sf"))
        n = int((cnt > 0).sum())
        print(f"T={ids.shape[0]} pairs={int(cnt.sum())} distinct={n} per-expert {per/1e6:.2f} MB unique {n*per/1e6:.1f} MB hist {sorted(cnt.tolist(), reverse=True)}", flush=True)
        for _ in range(2):
            sub.kernel_fn(data, st)
        torch.cuda.synchronize()
