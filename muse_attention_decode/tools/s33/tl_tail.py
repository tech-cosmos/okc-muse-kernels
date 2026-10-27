

# ---- s33 timeline DIAG (never submit) ----
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
    if cfg["q_len"] == 8 and cfg["batch"] == 1 and not cfg["window"] and cfg["num_kv_heads"] == 2 \
            and cfg["page_size"] == 1 and "v1" not in _DONE and not torch.cuda.is_current_stream_capturing():
        _DONE.add("v1")
        try:
            _tl_run(data)
        except Exception as ex:
            print(f"tl33 FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    return out


def _pt_l2(data):
    pt = data["page_table"]
    idx = torch.arange(pt.shape[1], device=pt.device)
    return {**data, "page_table": pt[:, idx % 1024].contiguous()}


def _tl_run(data, reps=30):
    global _TS33
    import statistics
    import numpy as np
    device = data["qkv"].device
    _TSB33[device] = torch.zeros(65536 + 4096, dtype=torch.int64, device=device)
    _diag_flush(device)
    ref = _kfn(data).float()
    dl2 = _pt_l2(data)
    variants = [("empty", None, 0), ("ts0", data, 0), ("ts1", data, 1), ("ts2", data, 2), ("ts1L2", dl2, 1),
                ("ts2L2", dl2, 2)]
    graphs = []
    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for name, dd, ts in variants:
            _TS33 = ts
            try:
                if dd is not None:
                    o = _kfn(dd)
                    if dd is data:
                        torch.cuda.synchronize()
                        print(f"tl33 {name} maxdiff {(o.float() - ref).abs().max().item():.3e}", flush=True)
                    _kfn(dd)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    if dd is not None:
                        _kfn(dd)
                graphs.append((name, g, ts))
            finally:
                _TS33 = 0
    torch.cuda.current_stream(device).wait_stream(s)
    torch.cuda.synchronize()
    tb = _TSB33[device]
    times = {n: [] for n, _, _ in graphs}
    stamps = {n: [] for n, _, ts in graphs if ts}
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(reps):
        for name, g, ts in graphs:
            tb.zero_()
            _diag_flush(device)
            a.record(); g.replay(); b.record()
            b.synchronize()
            times[name].append(a.elapsed_time(b) * 1000)
            if ts:
                stamps[name].append(tb.cpu().numpy().copy())
    e = statistics.median(times["empty"])
    for name, _, _ in graphs:
        x = sorted(times[name])
        print(f"tl33 time {name:6s} med {statistics.median(x) - e:7.2f} min {x[0] - e:7.2f} us", flush=True)
    for name, st in stamps.items():
        _tl_report(name, st)


def _tl_report(name, st):
    import numpy as np
    rows = {}

    def add(k, v):
        rows.setdefault(k, []).append(v)
    for t in st:
        A = t[:65536].reshape(-1, 32)
        A = A[A[:, 0] != 0]
        A = A[A[:, 9] > 0]  # (the tsep new-token program has no key tiles)
        C = t[65536:].reshape(-1, 4)
        C = C[C[:, 0] != 0]
        if len(A) == 0:
            continue
        ghz = np.median((A[:, 6] - A[:, 1]) / np.maximum(A[:, 7] - A[:, 0], 1))
        t0 = A[:, 0].min()
        aend = A[:, 7].max()
        add("ghz", [ghz])
        add("n_prog", [len(A)])
        add("a_start", np.percentile(A[:, 0] - t0, [50, 90, 100]) / 1e3)
        add("a_end", np.percentile(A[:, 7] - t0, [0, 50, 90, 100]) / 1e3)
        names = ["q", "new", "fix", "loop", "fin"]
        for i, nm in enumerate(names):
            d = (A[:, 2 + i] - A[:, 1 + i]) / ghz / 1e3
            add("ph_" + nm, np.percentile(d, [50, 90, 100]))
        add("ph_total", np.percentile((A[:, 6] - A[:, 1]) / ghz / 1e3, [50, 90, 100]))
        nt = A[:, 9]
        add("ntiles", np.percentile(nt, [0, 50, 100]))
        last = np.argmax(A[:, 7])
        add("last_prog", [(A[last, 0] - t0) / 1e3] + list((np.diff(A[last, 1:7])) / ghz / 1e3) + [A[last, 9]])
        if A[:, 10].any():
            tiles = []
            for row in A:
                k = int(row[9])
                if k <= 0 or row[10] == 0:
                    continue
                prev = row[4]
                seq = [row[10 + i] for i in range(min(k, 20)) if row[10 + i] != 0]
                d = np.diff([prev] + seq) / ghz / 1e3
                tiles.append(d)
            mx = max(len(x) for x in tiles)
            per = [np.median([x[i] for x in tiles if len(x) > i]) for i in range(mx)]
            add("tile_med", per)
        if len(C):
            c0 = C[:, 0] - t0
            add("c_start", np.percentile(c0, [0, 50, 100]) / 1e3)
            add("c_wait_rel", np.percentile(C[:, 1] - aend, [0, 50, 100]) / 1e3)
            add("c_load", np.percentile(C[:, 2] - C[:, 1], [50, 100]) / 1e3)
            add("c_end_rel", np.percentile(C[:, 3] - aend, [0, 50, 100]) / 1e3)
            add("total", [(C[:, 3].max() - t0) / 1e3])
    for k, v in rows.items():
        L = min(len(x) for x in v)
        m = np.median(np.array([x[:L] for x in v]), 0)
        print(f"tl33 {name:6s} {k:10s} " + " ".join(f"{x:7.3f}" for x in m), flush=True)
