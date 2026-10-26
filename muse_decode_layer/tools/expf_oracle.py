# needs dev/expf_s55.patch (the _EXPF hook in layer.py) applied + build.py; then a tuner-off variant (TUNE = {}).
"""s55 oracle: L2-prefetch (evict_last) part of the routed local experts' w13q on the side stream while the g19
pooler runs (T=16/32). python tools/expf_oracle.py <file> <filter> <rounds> spec...
spec = off | <when>:<unit_bytes>:<prefix_bytes>[:w2] with when = pre (before the router, oracle ids recorded from an
untimed run) or post (after the routing, real ids). unit 0 = the whole expert."""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void expf_k(const int* __restrict__ ids, int n, const char* base, long long estride, long long unit,
                       long long pre, int nunits, int chunk) {
  __shared__ unsigned smask;
  if (threadIdx.x == 0) smask = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) { int e = __ldcg(ids + i); if (e >= 0 && e < 32) atomicOr(&smask, 1u << e); }
  __syncthreads();
  const unsigned m = smask;
  const int ne = __popc(m);
  const long long cpu = pre / chunk;               // chunks per unit
  const long long total = (long long)ne * nunits * cpu;
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  for (long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x; w < total; w += (long long)gridDim.x * blockDim.x) {
    long long c = w / ((long long)ne * nunits);     // chunk index inside the unit: outermost (prefix order)
    long long r = w % ((long long)ne * nunits);
    int u = r / ne, j = r % ne;
    unsigned mm = m;
    for (int t = 0; t < j; t++) mm &= mm - 1;
    int e = __ffs(mm) - 1;
    const char* a = base + e * estride + u * unit + c * chunk;
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(a), "r"(chunk), "l"(pol) : "memory");
  }
}
void l2pf_ex(torch::Tensor ids, torch::Tensor w, long long unit, long long pre, int nunits, int chunk, int grid) {
  expf_k<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(), ids.numel(), (const char*)w.data_ptr(),
      w.stride(0) * w.element_size(), unit, pre, nunits, chunk);
}
"""
lib = load_inline("expf_s55", cpp_sources="void l2pf_ex(torch::Tensor ids, torch::Tensor w, long long unit, long long pre, int nunits, int chunk, int grid);",
                  cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + SRC, functions=["l2pf_ex"],
                  extra_cuda_cflags=["-O3", "-arch=sm_90a"], build_directory="/tmp/expf", verbose=False)

sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; rounds = int(sys.argv[3]); specs = sys.argv[4:]
w, st = bench.weights_state(sub)
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
data = {n: bench.make_input(c, w, 99) for n, c in names}
rec = {}


def make_hook(spec, name):
    if spec == "off":
        return None
    p = spec.split(":")
    when, unit, pre = p[0], int(p[1]), int(p[2])
    wk = "w2q" if len(p) > 3 and p[3] == "w2" else "w13q"

    def hook(state, tokens, lids):
        if tokens not in (16, 32):
            return
        if lids is not None and when == "rec":
            rec[name] = lids.clone()
            return
        if (when == "pre") != (lids is None):
            return
        ids = rec[name] if when == "pre" else lids
        W = state["moe"][wk]
        es = W.stride(0) * W.element_size()
        u = unit or es
        cur = torch.cuda.current_stream()
        forked[0] = True
        pfs.wait_stream(cur)
        with torch.cuda.stream(pfs):
            lib.l2pf_ex(ids, W, u, pre, es // u, 16384 if pre % 16384 == 0 else 4096, GRID)
    return hook


pfs = torch.cuda.Stream()
forked = [False]
import os
GRID = int(os.environ.get('GRID', 2))
kfn = sub.kernel_fn


def kfn_join(d, s):
    forked[0] = False
    out = kfn(d, s)
    if forked[0]:
        torch.cuda.current_stream().wait_stream(pfs)
    return out


for n, _ in names:  # record the real local ids once (untimed, eager)
    sub._EXPF = make_hook("rec:0:0", n)
    kfn_join(data[n], st); torch.cuda.synchronize()
e0 = bench.empty_cost()
res = {(n, i): [] for n, _ in names for i in range(len(specs))}
for r in range(rounds):
    for i, s in enumerate(specs):
        for n, _ in names:
            sub._EXPF = make_hook(s, n)
            res[(n, i)].append(bench.time_graph(lambda: kfn_join(data[n], st)) - e0)
sub._EXPF = None
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{i}] {statistics.median(res[(n, i)]):7.2f}" for i in range(len(specs))))
print("specs:", " | ".join(f"[{i}] {s}" for i, s in enumerate(specs)))

import os
if os.environ.get("TL"):
    import json, fcntl
    from torch.profiler import profile, ProfilerActivity
    for spec in os.environ["TL"].split(","):
        n = names[0][0]
        sub._EXPF = make_hook(spec, n)
        g = bench.capture(lambda: kfn_join(data[n], st))
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            torch.sum(flush.view(torch.int64), 0, out=sink); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
        p.export_chrome_trace("/tmp/expf/tl.json")
        ev = sorted([e for e in json.load(open("/tmp/expf/tl.json"))["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
        t0 = ev[0]["ts"]
        print("== TL", n, spec)
        for e in ev:
            print(f"{e['ts']-t0:8.1f} {e['ts']+e['dur']-t0:8.1f} {e['dur']:7.1f} s{e['args'].get('stream')} {e['name'][:60]}")
        del flush
