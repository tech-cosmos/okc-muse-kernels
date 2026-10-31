"""moe o25 held-out-weights mode 2 in the layer: default weights with a zero and a subnormal e4m3 block scale
(w13_sf[:,0,0]=0, w13_sf[:,7,3]=1, w2_sf[:,5,7]=3 as raw bytes), so prepare()'s st["sa"] is False.
python tools/n37/nonsa.py <file> <scored seeds>   -> 13 tests + every scored shape x seeds."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1]); seeds = int(sys.argv[2])
w = reference.generate_weights(reference.WEIGHTS_SEED)
v = w["w13_sf"].view(torch.uint8); v[:, 0, 0] = 0; v[:, 7, 3] = 1; w["w2_sf"].view(torch.uint8)[:, 5, 7] = 3
bench._W["w"] = w; bench._W["s"] = st = sub.prepare(w)
print("sa", st.get("sa"), flush=True)
bad = bench.test(sub)
for load, c in bench.SCORED:
    for sd in range(seeds):
        data = bench.make_input(c, w, 1000 + sd)
        out = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
        bad += not ok
        print("PASS" if ok else "FAIL", load, c["layer"], sd, msg[:120], flush=True)
        del data, out
        getattr(reference.generate_input, "cache_clear", lambda: None)()
print("BAD", bad)
