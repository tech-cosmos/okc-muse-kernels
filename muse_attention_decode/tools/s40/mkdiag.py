"""s40: H200 DIAG copy (NEVER submit) of submission.py: batch-16/32 full-attention bounds: main vs all-L2 page table
(consumer-bound time), per-request sequential rows (dense DRAM order) and the hfloor random/dense gather floors.
Derived from tools/s32/mkdiag.py
(variants/floors from ~/attn exp/hdiag5 840db1a). Usage: python tools/s32/mkdiag.py submission.py /tmp/s32d/submission.py
On the first eager (uncaptured) kernel_fn call of a diag shape, each variant is graph-captured and replayed after a
512 MB read (cold L2), round-robin, and 'diag40 ...' lines (median minus an empty graph) go to the run's log.
(v1 of this tool renamed kernels for the judge profile: the profile lists only the top 12 kernels, so floors were cut.)"""
import subprocess, sys

src = open(sys.argv[1]).read()
anchor = "    plan = _plan(cfg, max_pages, device)\n"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + "    plan = {**plan, **_PLAN_OV}\n")
kf = "\ndef kernel_fn(data: Inputs) -> torch.Tensor:\n"
assert src.count(kf) == 1
src = src.replace(kf, "\ndef _kfn(data: Inputs) -> torch.Tensor:\n")
hd = subprocess.run(["git", "-C", "/home/daytona/attn", "show", "840db1a:submission.py"], capture_output=True,
                    text=True, check=True).stdout
tail = hd[hd.index("def _diag_floor(data, r, dense):"):]
tail = tail[:tail.index("\n\n\n") + 1] if "\n\n\n" in tail else tail
flo = hd[hd.index("_FLOOR_SRC = r\"\"\""):hd.index("def _diag_floor(data, r, dense):")]
flo = flo.replace("HL(2, 0) HL(4, 0) HL(2, 1)", "HL(2, 0) HL(4, 0) HL(8, 0) HL(2, 1)")

src += r'''

# ---- s40 DIAG (never submit) ----
_PLAN_OV = {}
_GW = dict(gluon=True, splits=127, block_n=64, stages=3, lag=1, gpeel=True, pw=4, pregs=40, cregs=96, maxnreg=80,
           lsum=True, gfuse=False, gbal=False)  # s94's non-bal Gluon plan at verify 1 full
_GB = {**_GW, "gbal": True, "hb": 2, "cw": 4}
_DIAG = {
    ("decode", 16, "full"): [("d16a", {}), ("d16L2", {"_pt": "l2"}), ("d16seq", {"_pt": "seq"}), ("d16b", {}),
                            ("d16fl2", "floor"), ("d16fl4", "floor4"), ("d16fl8", "floor8"), ("d16dense", "dense")],
    ("decode", 32, "full"): [("d32a", {}), ("d32L2", {"_pt": "l2"}), ("d32seq", {"_pt": "seq"}), ("d32b", {}),
                            ("d32fl2", "floor"), ("d32fl4", "floor4"), ("d32fl8", "floor8"), ("d32dense", "dense")],
    ("verify", 16, "full"): [("v16a", {}), ("v16L2", {"_pt": "l2"}), ("v16seq", {"_pt": "seq"}), ("v16b", {}),
                            ("v16fl2", "floor"), ("v16fl4", "floor4"), ("v16fl8", "floor8"), ("v16dense", "dense")],
    ("verify", 32, "full"): [("v32a", {}), ("v32L2", {"_pt": "l2"}), ("v32seq", {"_pt": "seq"}), ("v32b", {}),
                            ("v32fl2", "floor"), ("v32fl4", "floor4"), ("v32fl8", "floor8"), ("v32dense", "dense")],
}


def _pt_variant(data, kind):
    """Diag data: page table remapped so the gathers hit L2 (all rows among the first 1024 entries' rows) or
    read consecutive rows (dense DRAM order). Output is then not the reference's; timing only."""
    pt = data["page_table"]
    n = pt.shape[1]
    idx = torch.arange(n, device=pt.device)
    if kind == "l2":
        pt2 = pt[:, idx % 1024].contiguous()
    else:
        b = torch.arange(pt.shape[0], device=pt.device).unsqueeze(1)
        pt2 = (b * n + idx.unsqueeze(0)).to(torch.int32).contiguous()  # distinct consecutive rows per request
    return {**data, "page_table": pt2}


_DFLUSH = {}
_DONE = set()


def _diag_flush(device):
    if device not in _DFLUSH:
        _DFLUSH[device] = (torch.zeros(64 << 20, dtype=torch.int64, device=device),
                           torch.empty(1, dtype=torch.int64, device=device))
    buf, sink = _DFLUSH[device]
    torch.sum(buf, 0, out=sink[0])


def kernel_fn(data: Inputs) -> torch.Tensor:
    out = _kfn(data)
    cfg = data["config"]
    key = ("decode" if cfg["q_len"] == 1 else "verify", cfg["batch"], "sliding" if cfg["window"] else "full")
    if key in _DIAG and cfg["num_kv_heads"] == 2 and cfg["page_size"] == 1 and key not in _DONE \
            and not torch.cuda.is_current_stream_capturing():
        _DONE.add(key)
        _diag_time(data, key)
    return out


def _diag_time(data, key, reps=40):
    global _PLAN_OV
    import statistics
    device = data["qkv"].device
    _diag_flush(device)

    def mk(ov):
        if ov is None:
            return lambda: None
        if isinstance(ov, str):
            return lambda: _diag_floor(data, {"floor": 2, "floor4": 4, "floor8": 8, "dense": 2}[ov], int(ov == "dense"))

        dd = _pt_variant(data, ov["_pt"]) if "_pt" in ov else data
        ov = {k: v for k, v in ov.items() if k != "_pt"}

        def f():
            global _PLAN_OV
            _PLAN_OV = ov
            try:
                return _kfn(dd)
            finally:
                _PLAN_OV = {}
        return f

    ref = _kfn(data).float()
    vs = []
    for name, ov in [("empty", None)] + _DIAG[key]:
        try:
            o = mk(ov)()
            torch.cuda.synchronize()
            if not isinstance(ov, (str, type(None))):
                print(f"diag40 {name} maxdiff {(o.float() - ref).abs().max().item():.3e}", flush=True)
            vs.append((name, ov))
        except Exception as ex:
            print(f"diag40 {name} FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
    graphs = []
    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for name, ov in vs:
            f = mk(ov)
            f(); f()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                f()
            graphs.append((g, f))  # f keeps the variant's remapped page table alive for the replays
    torch.cuda.current_stream(device).wait_stream(s)
    torch.cuda.synchronize()
    t = {n: [] for n, _ in vs}
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in vs]
    for r in range(reps):
        for (name, _), (g, _f), (a, b) in zip(vs, graphs, ev):
            _diag_flush(device)
            a.record(); g.replay(); b.record()
            b.synchronize()
            t[name].append(a.elapsed_time(b) * 1000)
    e = statistics.median(t["empty"])
    for name, _ in vs:
        x = sorted(t[name])
        print(f"diag40 {key[0]}{key[1]} {name:10s} med {statistics.median(x) - e:7.2f} p25 {x[len(x) // 4] - e:7.2f} "
              f"min {x[0] - e:7.2f} us (raw empty {e:.2f})", flush=True)


''' + flo + "_FLOOR = []\n\n\n" + tail
open(sys.argv[2], "w").write(src)
print("wrote", sys.argv[2])
