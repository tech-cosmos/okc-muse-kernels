# n7s21: H200 timing of the T=256 unpool GEMM (cuBLAS torch.mm vs ug<NT>, tools/n7s21/ug.py MC=1 form) inside a moe-board
# check's prepare() (DIAG copy; outputs stay correct). usage (muse_moe_w4a16/): python tools/n7s21/mkdiag_ug.py submission.py /tmp/n7s21/submission.py
# Output: grep "ugdiag" profiles/submission.<job>.log
import sys, os
s = open(sys.argv[1]).read()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ug_src.cu")).read()
a = '''    if DIAG and not _DIAG_DONE:
        _DIAG_DONE = True
        _diag(st)'''
assert s.count(a) == 1
s = s.replace(a, '''    if not _DIAG_DONE:
        _DIAG_DONE = True
        _ugdiag()''')
s += '''

_UG_SRC = r\'\'\'''' + src + '''\'\'\'


def _ugdiag():
    import time
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    lib = load_inline(name="ug_n7s21_diag", cpp_sources="void ugemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t nt, int64_t ntile);",
                      cuda_sources=_UG_SRC, functions=["ugemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"],
                      extra_ldflags=["-lcuda"])
    print(f"ugdiag build {time.time() - t0:.1f} s", flush=True)
    g0 = torch.Generator(device="cuda"); g0.manual_seed(0)
    flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda")  # 256 MB
    sink = torch.empty(1, dtype=torch.float32, device="cuda")
    for KK, NW in ((4096, 8192), (2048, 8192)):
        W = (torch.randn(NW, KK, device="cuda", generator=g0) * 0.02).bfloat16()
        x = torch.randn(256, KK, device="cuda", generator=g0).bfloat16()
        ref = x.float() @ W.float().t()
        o = torch.empty(256, NW, device="cuda", dtype=torch.bfloat16)
        fns = [("cublas256", lambda: torch.mm(x, W.t(), out=o), 256)]
        for nt in (128, 112, 104, 96):
            fns.append((f"ug{nt}", (lambda nt=nt: lib.ugemm(x, W, o, nt, 2)), 2 * nt))
        errs = {}
        graphs = []
        for n, fn, rows in fns:
            o.zero_(); fn(); torch.cuda.synchronize()
            errs[n] = (o[:rows].float() - ref[:rows]).abs().max().item()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            graphs.append((n, gr))
        for cold in (1, 0):
            ts = {n: [] for n, _ in graphs}
            for _ in range(40):
                for n, gr in graphs:
                    if cold:
                        torch.sum(flush, 0, out=sink)
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record(); gr.replay(); b.record()
                    torch.cuda.synchronize()
                    ts[n].append(a.elapsed_time(b) * 1e3)
            print(f"ugdiag K{KK} {'cold' if cold else 'warm'}: " + " | ".join(
                f"{n} {sorted(v)[len(v) // 2]:.2f}" for n, v in ts.items()) + " | maxerr " + " ".join(f"{errs[n]:.3g}" for n in errs), flush=True)
        del graphs, W, x, ref, o
    del flush
    torch.cuda.synchronize()
'''
open(sys.argv[2], "w").write(s)
