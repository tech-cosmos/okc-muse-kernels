# Appended to layer.py for a diagnostic check only (NEVER submit): profiles each step of the Marlin expert path
# in prepare (untimed) and prints kernel names and CUDA-graph replay times of each step on the judge's GPU.
def _diag(state):
    import statistics
    from torch.profiler import profile, ProfilerActivity
    dev = "cuda"
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    sink = torch.zeros((), dtype=torch.int64, device=dev)

    def gtime(fn, reps=30):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.sum(flush.view(torch.int64), 0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        return statistics.median(ts)

    e0 = gtime(lambda: None, 50)
    bank = state["bank"]
    for T in (1, 8, 32, 256):
        gen = torch.Generator(device=dev).manual_seed(T)
        x = torch.randn((T, D_EXPERT), generator=gen, device=dev).to(torch.bfloat16)
        lids = torch.stack([torch.randperm(48, generator=gen, device=dev)[:TOP_K] for _ in range(T)]).to(torch.int32)
        lids = torch.where(lids < LOCAL_EXPERTS, lids, -1).contiguous()
        wts = torch.rand((T, TOP_K), generator=gen, device=dev)
        for block_size_m in BLOCK_SIZE_M_CANDIDATES:
            if T * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        steps = {"align": lambda: moe_align_block_size(lids, block_size_m, N_EXPERTS),
                 "all": lambda: _experts_marlin(x, wts, lids, state)}
        for name, fn in steps.items():
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                fn(); torch.cuda.synchronize()
            ks = [e.name[:60] for e in p.events() if e.device_type.name == "CUDA"]
            print(f"DIAG T={T} {name}: {gtime(fn) - e0:.2f} us; kernels: {ks}", flush=True)
        print(f"DIAG T={T} prenorm {gtime(lambda: _rmsnorm_gain[(T,)](x, state['pre_expert'], x.clone(), N=D_EXPERT, EPS=RMS_EPS, PDL=False, num_warps=8)) - e0:.2f}")
    del flush


_prepare_orig = prepare


def prepare(weights):
    state = _prepare_orig(weights)
    try:
        _diag(state)
    except Exception as e:  # noqa: BLE001
        print("DIAG failed", repr(e))
    return state
