"""moe o44 (lead 11-01 20:20Z): is any expert weight re-read from DRAM in-layer? (IBM col-major/L2-reuse question)
Run from muse_decode_layer (needs bench.py):
  ncu --profile-from-start off -k regex:moe_ --cache-control all --metrics dram__bytes_read.sum,... \
      python ~/r-moe/muse_moe_w4a16/tools/o44/l2reuse.py submission.py
Prints per scored shape: routed tokens->distinct local experts, unique expert weight bytes; ncu prints per-kernel metrics
(one profiler range per shape, in the printed order)."""
import sys, fcntl, torch
sys.path.insert(0, ".")
import bench, reference as R

sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
w, st = bench.weights_state(sub)
EB = sum(w[k][0].numel() * w[k][0].element_size() for k in ("w13", "w13_sf", "w2", "w2_sf"))
print(f"O44 expert bytes {EB} ({EB/1e6:.2f} MB)")
flt = sys.argv[2] if len(sys.argv) > 2 else ""
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name:
        continue
    d = bench.make_input(c, w, 99)
    ids = R.ref_front(d, w)["biased"].topk(R.TOP_K, dim=-1).indices
    loc = (ids.long() - int(d["local_start"])) % R.N_EXPERTS
    loc = loc[loc < R.LOCAL_EXPERTS]
    ne = int(torch.unique(loc).numel())
    T = ids.shape[0]
    print(f"O44 {name}: T {T} local routes {loc.numel()} distinct {ne} unique weight MB {ne*EB/1e6:.1f}"
          f" act MB {T*R.D_EXPERT*2/1e6:.3f}", flush=True)
    sub.kernel_fn(d, st); torch.cuda.synchronize()
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        torch.cuda.profiler.start()
        sub.kernel_fn(d, st)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
