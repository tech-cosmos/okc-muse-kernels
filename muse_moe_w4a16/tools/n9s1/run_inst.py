"""n9s1: python run_inst.py sub.py filter [reps] -> per-shape twm consumer counters (cycles, sum over warps)."""
import sys, ctypes, fcntl, torch
sys.path.insert(0, '.')
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
w, st = bench.weights_state(sub)
rt = ctypes.CDLL("libcudart.so.13")
def rd():
    torch.cuda.synchronize()
    p = sub._moe__LIB.n9_ptr()
    buf = (ctypes.c_uint64 * 16)()
    assert rt.cudaMemcpy(buf, ctypes.c_void_p(p), 128, 2) == 0
    return list(buf)
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if not any(f in name for f in flt.split(',')): continue
    for seed in (99, 1, 2):
        d = bench.make_input(c, w, seed)
        g = bench.capture(lambda: sub.kernel_fn(d, st))
        g.replay(); a = rd()
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for _ in range(reps): g.replay()
            b = rd()
        v = [(y - x) / reps for x, y in zip(a, b)]
        if v[8] == 0: print(name, seed, "RAW", v, flush=True); v[8] = 1
        W = 132 * 8  # consumer warps
        us = lambda cyc: cyc / W / 1.98e3  # per-warp mean, us at 1.98 GHz (clock64 = SM clock)
        print(f"{name} s{seed}: clk {v[5]/v[8]:.3f} GHz life {v[8]/W/1e3:.1f} us | life {us(v[5]):.1f} | between-item wait {us(v[4]):.1f} | item {us(v[6]):.1f} "
              f"(loop {us(v[1]):.1f} of which ring wait {us(v[0]):.1f}; epi+rest {us(v[6]-v[1]):.1f}) | items/warp {v[3]/W:.2f} big {v[7]/W:.2f}", flush=True)
        print(f"   release {v[11]/132/8/1.98e3:.2f} us per warp;", end=" ")
        print(f"   producer g2 spin {v[9]/132/1.98e3:.1f} us per CTA over {v[10]/132:.2f} g2 items/CTA", flush=True)
        del g, d
