"""n57: VJX off vs on, eager, per (shape, seed): reference check message + topk-id agreement + local-route count."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]
w, st = bench.weights_state(sub)
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name:
        continue
    for sd in seeds:
        outs = []
        for vt in ((), (1, 8, 16, 32, 128, 256)):
            sub.VJX_TOKENS = vt
            data = bench.make_input(c, w, sd)
            out = sub.kernel_fn(data, st)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
            ids = out[-1]
            ls = data["config"].get("local_start", None)
            outs.append(ids)
            print(f"{name} s{sd} vjx={bool(vt)} {'PASS' if ok else 'FAIL'} {msg[:150]}", flush=True)
            del data
        a, b = outs
        print(f"   topk ids equal {(a == b).float().mean().item():.4f}; shape {tuple(a.shape)}", flush=True)
