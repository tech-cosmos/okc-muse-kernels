"""moe o42: per-CTA %globaltimer stamps of the layer's T=8 expert chain in-layer (pr8 -> [rmsnorm] -> tk2f -> nu8 -> bnd2).
Run from a COPY of r-layer/muse_decode_layer (needs its bench.py):
  cd /tmp/o42/layer && python ~/r-moe/muse_moe_w4a16/tools/o42/stamp_v1.py submission.py ['verify 1 sliding'] [reps] [seed]
NOSTAMP=1 runs the unpatched file (replay time reference). Times are us from the first pr8 CTA entry of each replay;
columns = medians over replays of the per-CTA min / median / max."""
import sys, os, fcntl, statistics, torch
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
import stp_patch as sp

src = sys.argv[1]
flt = sys.argv[2] if len(sys.argv) > 2 else "verify 1 sliding"
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 99
NR = 5
buf = torch.zeros(NR * sp.NC * sp.NK, dtype=torch.int64, device="cuda")
s = open(src).read()
inj = ""
if os.environ.get("NU8"):  # o42 nu8 variant: v4 | store (oracle) | ks1 | mmq | ssq | fast
    inj += "_stpp.nu8_mode(globals(), %r)\n" % os.environ["NU8"]
if not os.environ.get("NOSTAMP"):
    inj += "_stpp.patch(globals(), %d)\n" % buf.data_ptr()
if inj:
    k = s.index("_PREBUILD = [")
    s = s[:k] + ("import sys as _stps\n_stps.path.insert(0, %r)\nimport stp_patch as _stpp\n" % os.path.dirname(os.path.abspath(__file__))) + inj + s[k:]
out = os.path.join(os.path.dirname(os.path.abspath(src)), f"_o42_stamped_{os.getpid()}.py")
open(out, "w").write(s)
try:
    sub = bench.load_sub(out)
    w, st = bench.weights_state(sub)
    (name, c), = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if f"{l} {c['layer']}" == flt]
    d = bench.make_input(c, w, seed)
    sub.kernel_fn(d, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sub.kernel_fn(d, st)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rows, tot = [], []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for r in range(reps):
            (None if os.environ.get("NOFLUSH") else flush.zero_()); buf.zero_(); torch.cuda.synchronize()
            ev0.record(); g.replay(); ev1.record(); torch.cuda.synchronize()
            tot.append(ev0.elapsed_time(ev1) * 1000)
            rows.append(buf.view(NR, sp.NC, sp.NK).cpu().clone())
finally:
    os.remove(out)
print(name, f"seed {seed} layer replay median {statistics.median(tot):.1f} us ({'unstamped' if os.environ.get('NOSTAMP') else 'stamped'})")
if os.environ.get("NOSTAMP"):
    sys.exit()
lines = []
for reg in range(NR):
    for k, nm in sp.SLOTS[reg].items():
        meds = []
        for t in rows:
            p8 = t[0, :, 0]
            t0 = p8[p8 > 0].min().item() if (p8 > 0).any() else t[t > 0].min().item()
            v = t[reg, :, k]
            v = v[v > 0]
            if len(v) == 0:
                break
            v = (v - t0).double() / 1000.0
            meds.append((v.min().item(), v.median().item(), v.max().item(), len(v)))
        if len(meds) == len(rows):
            lines.append((statistics.median(m[0] for m in meds), f"{sp.NAMES[reg]:8s} {nm:14s} n {meds[0][3]:3d}  min {statistics.median(m[0] for m in meds):7.2f}  med {statistics.median(m[1] for m in meds):7.2f}  max {statistics.median(m[2] for m in meds):7.2f} us"))
for _, l in sorted(lines):
    print(l)
