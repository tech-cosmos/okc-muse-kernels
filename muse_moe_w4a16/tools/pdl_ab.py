# A/B of PDL variants (local graph timing, cold L2), several routings per bs.
# usage: pdl_ab.py "<bs,...>" "<pdl:twpf,...>"   e.g. pdl_ab.py 128,256 0:0,1:0,1:16
import sys, statistics
sys.path.insert(0, ".")
import torch, bench, submission, reference
w, st = bench.state()
lib = st["lib"]
bss = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "1,8,16,32").split(",")]
vs = [tuple(int(t) for t in v.split(":")) for v in (sys.argv[2] if len(sys.argv) > 2 else "0:0,1:16").split(",")]
for bs in bss:
    res = {v: [] for v in vs}
    for sd in range(9000, 9006):
        d = reference.generate_input(bs=bs, sharding="ep8", seed=sd)
        for _ in range(2):
            for v in vs:
                lib.moe_set_pdl(*v)
                res[v].append(bench.time_graph(lambda: submission.kernel_fn(d, st), 30))
    print(bs, "  ".join("%s %.1f" % (v, statistics.median(res[v])) for v in vs), flush=True)
lib.moe_set_pdl(4, 0)
