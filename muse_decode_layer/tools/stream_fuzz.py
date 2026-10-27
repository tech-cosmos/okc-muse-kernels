"""proj s83: deterministic cross-stream race fuzzer (the s82 nu8d race class). Monkeypatches torch.cuda.stream so that
every `with torch.cuda.stream(side or hi stream):` in layer.py injects a torch.cuda._sleep spin kernel:
  side  on the other stream at context entry   -> the side/hi stream's work runs late (main reads its outputs too early?)
  main  on the main stream right after the fork -> main runs late (side overwrites/reuses what main still reads?)
then runs every TUNE option set (opt_test's loop: each T's candidates on TUNE_BASE, eager) on the scored shape of that T and
checks with reference.check_implementation. Input seeds rotate between consecutive runs (2000-2002): with one fixed seed a
stale buffer still holds the right values from the previous run and hides the race. The sleep only adds ordering slack (it never removes a stream/PDL edge), so a
correct layer passes in every mode; the s82 bug (unfixed v128) fails 'side' deterministically.
python tools/stream_fuzz.py [submission.py] [modes=side,main] [cycles=400000] [seeds=1] [T list] [reps=12]
modes gside/gmain: the same sleeps captured into a CUDA graph (judge path), replayed reps times on fresh inputs, only
for option sets that fork (no eager runs in those modes).
seeds = runs per option set that never forks; reps = runs per option set that does (failures are probabilistic: the
wrong read only shows when the stale buffer differs enough; unfixed v128 T=8 {'pr1': False}: 1 of 6 at reps 1)."""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
modes = (sys.argv[2] if len(sys.argv) > 2 else "side,main").split(",")
cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 12  # runs per option set that forks a stream
Ts = {int(t) for t in sys.argv[5].split(",")} if len(sys.argv) > 5 and sys.argv[5] else set(sub.TUNE)
w, st = bench.weights_state(sub)
TL = {1: "decode 1", 16: "decode 16", 32: "decode 32", 8: "verify 1", 128: "verify 16", 256: "verify 32"}
_orig = torch.cuda.stream
MODE, FORKS = [None], [0]


class _Fuzz:
    def __init__(self, s):
        self.s, self.ctx = s, _orig(s)

    def __enter__(self):
        other = self.s is not None and any(self.s == st.get(k) for k in ("side_stream", "hi_stream"))
        if other and MODE[0].endswith("main"):
            torch.cuda._sleep(cycles)  # on main, after the fork's wait_stream/wait_event
        r = self.ctx.__enter__()
        if other and MODE[0].endswith("side"):
            torch.cuda._sleep(cycles)
        FORKS[0] += other
        return r

    def __exit__(self, *a):
        return self.ctx.__exit__(*a)


torch.cuda.stream = _Fuzz
bad = n = 0
for mode in modes:
    MODE[0] = mode
    for T, cands in sub.TUNE.items():
        if T not in Ts:
            continue
        for load, c in bench.SCORED:
            if load != TL[T]:
                continue
            key = (T, c["layer"] == "sliding")
            f0, b0 = FORKS[0], bad
            for opts in cands:
                st["tune"] = {key: opts}
                fk, r = FORKS[0], 0
                if mode.startswith("g"):  # graph: capture once (sleeps become graph nodes), replay on fresh inputs
                    data = bench.make_input(c, w, 7000)
                    out = [None]
                    g = bench.capture(lambda: out.__setitem__(0, sub.kernel_fn(data, st)))
                    if FORKS[0] == fk:
                        del g, data, out
                        continue
                    for i in range(reps):
                        d = bench.make_input(c, w, 7001 + n)
                        for k, v in d.items():
                            if isinstance(v, torch.Tensor):
                                data[k].copy_(v)
                        g.replay()
                        torch.cuda.synchronize()
                        ok, msg = reference.check_implementation(data, reference.collect(data, out[0]), w)
                        bad += not ok; n += 1
                        if not ok:
                            print("FAIL", mode, load, c["layer"], opts, i, msg[:150], flush=True)
                        del d
                    del g, data, out
                    torch.cuda.empty_cache()
                    continue
                fk, r = FORKS[0], 0
                while True:
                    data = bench.make_input(c, w, 2000 + n % 3)  # rotate: stale buffers from the last run must differ
                    out = sub.kernel_fn(data, st)
                    torch.cuda.synchronize()
                    ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
                    bad += not ok; n += 1; r += 1
                    if not ok:
                        print("FAIL", mode, load, c["layer"], opts, r, msg[:150], flush=True)
                    del data, out
                    getattr(reference.generate_input, "cache_clear", lambda: None)()
                    if r >= (reps if FORKS[0] > fk else seeds):
                        break
            print("done", mode, load, c["layer"], len(cands), "sets, forks", FORKS[0] - f0, "bad", bad - b0, flush=True)
print("bad", bad, "of", n)
