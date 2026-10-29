# fresh f1s6: H100/H200 standalone A/B of CUTLASS sm90 builds (/tmp/fcg/*.so from build.sh) vs F.linear and the
# layer's cuBLASLt picks (LT_CFG) for the T=128/256 dense GEMMs. Cold L2 (256 MB read flush before each call),
# CUDA-graph replays, time = (flush+gemm) - (flush only), median over rounds. GPU lock only around timed replays.
# usage: python cgb.py SHAPES SOS [splits:mode,...]   SHAPES e.g. qkv:128,o:256 ; SOS = comma list of .so names or "-"
import ctypes, fcntl, os, sys, statistics
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER = os.path.abspath(os.path.join(HERE, "../../.."))
SHAPES = {"qkv": (2304, 8192, False), "o": (8192, 1024, False), "pool": (4096, 8192, False),
          "fr": (4608, 8192, True), "unp": (8192, 4096, False)}
LTKEY = {"qkv": "qkv_gate", "o": "o", "pool": "pooler", "fr": "fr", "unp": "unpool"}
REPS, ROUNDS = 20, 7
dev = torch.device("cuda")
flushbuf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
sink = torch.empty(1, dtype=torch.int64, device=dev)


def flush():
    torch.sum(flushbuf.view(torch.int64), dim=0, out=sink)


def lock():
    f = open("/tmp/okc-gpu.lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def gtime(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); flush()
    torch.cuda.synchronize()
    gs = []
    for body in (lambda: flush(), lambda: (flush(), fn())):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(REPS):
                body()
        gs.append(g)
    res = []
    lk = lock()
    try:
        for _ in range(ROUNDS):
            t = []
            for g in gs:
                g.replay(); torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                t.append(e0.elapsed_time(e1) * 1000 / REPS)
            res.append(t[1] - t[0])
    finally:
        lk.close()
    return statistics.median(res)


_LT = None


def lt():
    global _LT
    if _LT is None:
        from torch.utils.cpp_extension import load_inline
        src = open(os.path.join(LAYER, "lt_src.cu")).read()
        _LT = load_inline("fresh_lt", cpp_sources="int64_t lt_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>);", cuda_sources=src, functions=["lt_gemm"],
                          extra_ldflags=["-lcublasLt"], verbose=False)
    return _LT


def main():
    import ast
    tree = ast.parse(open(os.path.join(LAYER, "layer.py")).read())
    LT_CFG = next(ast.literal_eval(nd.value) for nd in tree.body
                  if isinstance(nd, ast.Assign) and getattr(nd.targets[0], "id", "") == "LT_CFG")
    shapes = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[1].split(",")]
    sos = [] if sys.argv[2] == "-" else sys.argv[2].split(",")
    sms = [(1, 0)] if len(sys.argv) < 4 else [tuple(int(v) for v in t.split(":")) for t in sys.argv[3].split(",")]
    torch.manual_seed(0)
    ws = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    for name, m in shapes:
        n, k, f32 = SHAPES[name]
        w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
        x = torch.randn(m, k, device=dev).bfloat16()
        odt = torch.float32 if f32 else torch.bfloat16
        ref = (x.float() @ w.float().t())
        out = torch.empty(m, n, device=dev, dtype=odt)
        row = [f"{name} m{m}"]
        if f32:
            row.append(f"mm {gtime(lambda: torch.mm(x, w.t(), out_dtype=torch.float32)):.2f}")
        else:
            row.append(f"F {gtime(lambda: F.linear(x, w, out=out)):.2f}")
        cfg = LT_CFG.get((LTKEY[name], m))
        if cfg is not None:
            st = lt().lt_gemm(w, x, out, ws, list(cfg))
            err = ((out.float() - ref).norm() / ref.norm()).item()
            row.append(f"lt {gtime(lambda: lt().lt_gemm(w, x, out, ws, list(cfg))):.2f} (st{st} e{err:.1e})")
        for so in sos:
            lib = ctypes.CDLL(f"/tmp/fcg/{so}.so")
            for sp, mode in sms:
                out.zero_()
                strm = torch.cuda.current_stream().cuda_stream
                need = lib.cg_ws(n, m, k, sp, mode)
                if need > ws.numel():
                    row.append(f"{so}/{sp}:{mode} ws{need}"); continue
                ws.zero_()
                L = mode % 100 if mode >= 100 else 0
                wk = w.view(n, L, k // L).permute(1, 0, 2).contiguous() if mode >= 200 else w
                slab = torch.empty(L * m * n, dtype=torch.float32, device=dev) if L else out
                dst = slab
                rc = lib.cg_init(ctypes.c_void_p(wk.data_ptr()), ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(dst.data_ptr()),
                                 n, m, k, sp, mode, ctypes.c_void_p(ws.data_ptr()), ctypes.c_void_p(strm))
                if rc:
                    row.append(f"{so}/{sp}:{mode} rc{rc}"); continue
                wsv = ws[:need]
                red = out
                def call():
                    if need:
                        wsv.zero_()
                    lib.cg_run(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
                    if L:
                        torch.sum(slab.view(L, m, n), 0, out=red) if f32 else out.copy_(slab.view(L, m, n).sum(0))
                t = gtime(call)
                torch.cuda.synchronize()
                err = ((out.float() - ref).norm() / ref.norm()).item()
                row.append(f"{so}/{sp}:{mode} {t:.2f}{'' if need == 0 else ' ws%dK' % (need >> 10)}{'' if err < 1e-2 else ' BAD%.1e' % err}")
        print(" | ".join(row), flush=True)
        del w, x, out


if __name__ == "__main__":
    main()
