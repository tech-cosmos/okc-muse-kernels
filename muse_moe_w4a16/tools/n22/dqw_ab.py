# n22: DQW 0 vs 1 (dq() rotations on the FMA pipe) on the layer's routings: bitwise check + cold-L2 time (DRAM-bound
# on the H100) + warm-L2 time (graph of 5 back-to-back calls, no flush: consumer-rate proxy). A B A B rounds.
# python tools/n22/dqw_ab.py [rounds]
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import submission as S
import layer_ab as L
st = L.st
st["w13sq"] = S._sq_pack(st["w13q"], st["w13s2"]); st["w2sq"] = S._sq_pack(st["w2q"], st["w2s2"])
libs = {0: st["lib"]}
S.DQW = 1; libs[1] = S._lib(); S.DQW = 0; S._lib()
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
loads = sys.argv[2].split(",") if len(sys.argv) > 2 else ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
def warm(fn, reps=60):
    import fcntl
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(5): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / 5)
    ts.sort(); return ts[len(ts) // 2]
for load in loads:
    c = {0: [], 1: []}; w = {0: [], 1: []}; bad = 0
    for (l, sd), loc in sorted(L.R.items()):
        if l != load: continue
        d = L.mk(loc, sd)
        o = {}
        for q in (0, 1):
            st["lib"] = libs[q]; o[q] = S.kernel_fn(d, st).clone()
        bad += int(not torch.equal(o[0], o[1]))
        for _ in range(rounds):
            for q in (0, 1):
                st["lib"] = libs[q]
                c[q].append(L.tg(lambda: S.kernel_fn(d, st)))
                w[q].append(warm(lambda: S.kernel_fn(d, st)))
    st["lib"] = libs[0]
    m = lambda v: sum(v) / len(v)
    print(f"{load}: cold q0 {m(c[0]):.2f} q1 {m(c[1]):.2f} d {m(c[1]) - m(c[0]):+.2f} | warm q0 {m(w[0]):.2f} q1 {m(w[1]):.2f} "
          f"d {m(w[1]) - m(w[0]):+.2f} us | not bitwise {bad}", flush=True)
