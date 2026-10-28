"""n5s29 PR correctness: python dev/n5s29/pr_test.py <sub> [W] : bench TESTS + scored d16/d32 x seeds with tk2pr = W + 1 at T=16/32,
and the max |diff| of the layer outputs vs the same input with PR off."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
w, st = bench.weights_state(sub)
def setpr(on):
    sub._ab(16); 
    for T in (16, 32):
        for s in (True, False):
            d = sub.TUNE_BASE.setdefault((T, s), {})
            if on: d["tk2pr"] = W + 1
            else: d.pop("tk2pr", None)
bad = 0
for sd in [int(x) for x in __import__("os").environ.get("PRS", "99,7,13,11,17,3").split(",")]:
    for l, c in bench.SCORED:
        if c["batch"] * c["q_len"] not in (16, 32): continue
        data = bench.make_input(c, w, sd)
        setpr(False); o0 = sub.kernel_fn(data, st); torch.cuda.synchronize()
        o0 = [t.clone() if torch.is_tensor(t) else t for t in o0]
        setpr(True); o1 = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, o1), w)
        dm = max(((a.float() - b.float()).abs().max().item() if torch.is_tensor(a) and a.is_floating_point() else 0.0) for a, b in zip(o0, o1))
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} s{sd} {l} {c['layer']} maxdiff vs off {dm:.4g} {msg[:100]}", flush=True)
setpr(True)
bad += bench.test(sub) if __import__("os").environ.get("PRT", "1") == "1" else 0
print("BAD", bad)
