# s27: consumer-bound proxy for moe_twm (main big-M path): ONE local expert (L2-resident), ntok tokens routed to it
# (T = 40 -> M > 32 -> twm), warm L2, 5 kernel_fn calls per graph. python tools/s27m/twm_warm.py (from muse_moe_w4a16/)
import sys, torch, fcntl
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
import layer_ab as L
m = L.submission; st = L.st
def mkd(T, ntok, ne=1):
    loc = torch.full((T, 8), -1, dtype=torch.int8)
    for j in range(ne): loc[:ntok, j] = 3 + j
    return L.mk(loc, 1)
def tw(data):
    fn = lambda: [m.kernel_fn(data, st) for _ in range(5)]
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g): fn()
    ts = []
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        g.replay()
        for _ in range(20):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / 5)
    ts.sort(); return ts[len(ts) // 2]
if __name__ == "__main__":
  ne = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  for ntok in (1, 8, 16, 24, 32):
      print(f"experts {ne} tokens {ntok}: " + "/".join(f"{tw(mkd(40, ntok, ne)):.1f}" for _ in range(3)), flush=True)
