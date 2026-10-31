"""proj n24: the judge tuner's 'confirm' loop times T=256 full ~3% slower than its first pass (25/25 judge logs, same graphs,
same data). Reproduce on the H100: one layer submission, tuner off, graphs of one scored case; phases in the tuner's order
(A: 3 graphs round-robin x 43, B: 2 graphs alternating x 40) and a long single-graph series, with a 256 MB flush.zero_()
before each replay like _tune. nvidia-smi clocks/power sampled in the background.
python conf.py <layer_dir> <sub.py> <case filter, e.g. 'verify 32 full'> [reps]"""
import sys, os, fcntl, statistics, subprocess, time, torch
ldir = sys.argv[1]; sys.path.insert(0, ldir); os.chdir(ldir)
import bench, reference
os.makedirs("/tmp/n24", exist_ok=True)
sub = bench.load_sub(sys.argv[2]); sub.TUNE = {}
flt = sys.argv[3]; R = int(sys.argv[4]) if len(sys.argv) > 4 else 300
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = sub.prepare(w)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def cap(data):
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sub.kernel_fn(data, st)
        with torch.cuda.graph(g, stream=s):
            sub.kernel_fn(data, st)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    return g


def rp(g):
    flush.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); b.synchronize()
    return a.elapsed_time(b) * 1e3


for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name: continue
    data = bench.make_input(c, w, 7000)
    gs = [cap(data) for _ in range(3)]  # 3 captures of the same call (tuner: 3 candidates)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw,clocks_throttle_reasons.active,temperature.gpu", "--format=csv,noheader",
                            "-lms", "20"], stdout=open(f"/tmp/n24/smi_{name.replace(' ', '_')}.csv", "w"))
    time.sleep(0.5)
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        t0 = time.time()
        A = [[] for _ in gs]
        for r in range(46):  # tuner first pass: round-robin, reversed every other round, 3 warm-up rounds
            for i in ((0, 1, 2) if r % 2 == 0 else (2, 1, 0)):
                t = rp(gs[i])
                if r >= 3: A[i].append(t)
        B = [[], []]
        for r in range(40):  # tuner confirm: graphs 0 and 1 alternating
            for j, i in enumerate((0, 1) if r % 2 == 0 else (1, 0)):
                B[i].append(rp(gs[i]))
        tA = time.time()
        C = [rp(gs[0]) for _ in range(R)]  # long single-graph series
        tC = time.time()
    time.sleep(0.2); smi.terminate(); smi.wait()
    rows = [r.split(", ") for r in open(f"/tmp/n24/smi_{name.replace(' ', '_')}.csv").read().strip().splitlines()]
    def ts(r):
        import datetime
        return datetime.datetime.strptime(r[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
    for lo, hi, ph in ((t0, tA, "A+B"), (tA, tC, "C")):
        sel = [r for r in rows if len(r) >= 5 and lo - 0.05 <= ts(r) <= hi + 0.05]
        print(f"  smi {ph}: " + " ".join(f"{r[1].split()[0]}MHz/{r[2].split()[0]}W/{r[3][-4:]}/{r[4]}C" for r in sel), flush=True)
    med = lambda v: statistics.median(v)  # noqa: E731
    print(f"{name}: A (3-way) medians {' '.join(f'{med(v):.1f}' for v in A)} | B (2-way) {med(B[0]):.1f} {med(B[1]):.1f}"
          f" | A+B {tA - t0:.2f} s, C {tC - tA:.2f} s", flush=True)
    print("  A per-round-block of 10 (graph 0): " + " ".join(f"{med(A[0][k:k + 10]):.1f}" for k in range(0, len(A[0]), 10)))
    print("  B per-block of 10 (graph 0):       " + " ".join(f"{med(B[0][k:k + 10]):.1f}" for k in range(0, len(B[0]), 10)))
    print("  C per-block of R/12:               " + " ".join(f"{med(C[k:k + R // 12]):.1f}" for k in range(0, len(C), R // 12)), flush=True)
    del gs, data; torch.cuda.empty_cache()
