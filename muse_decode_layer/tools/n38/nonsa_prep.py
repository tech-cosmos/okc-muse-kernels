"""Cold prepare() wall time on non-SA weights (n37 fix rebuilds the moe lib). Run with fresh
TORCH_EXTENSIONS_DIR / TRITON_CACHE_DIR for a cold measurement. python tools/n38/nonsa_prep.py <file> [sa]"""
import sys, time, torch
sys.path.insert(0, ".")
import bench, reference
t0 = time.time()
sub = bench.load_sub(sys.argv[1])
w = reference.generate_weights(reference.WEIGHTS_SEED)
if len(sys.argv) < 3:
    v = w["w13_sf"].view(torch.uint8); v[:, 0, 0] = 0; v[:, 7, 3] = 1; w["w2_sf"].view(torch.uint8)[:, 5, 7] = 3
t1 = time.time()
st = sub.prepare(w); torch.cuda.synchronize()
print("nsa", sub._MOE_NSA if hasattr(sub, "_MOE_NSA") else "?", "import+weights %.1f s, prepare %.1f s" % (t1 - t0, time.time() - t1), flush=True)
