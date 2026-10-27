"""proj n5s4: g30 fixed cost vs slope. Times g30 (proj build, (m, S, 16, 16)) on the first rows/K-columns of a weight
(N rows, K cols) as a 1-kernel graph after a cold-L2 flush, minus an empty-kernel graph; fits t = a + bytes / rate.
Run from muse_bf16_proj/: python dev/n5s4/kfit.py [layer=qkv_gate] [m=16] [S=3]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference
L = bench.submission
layer = sys.argv[1] if len(sys.argv) > 1 else "qkv_gate"
m = int(sys.argv[2]) if len(sys.argv) > 2 else 16
S = int(sys.argv[3]) if len(sys.argv) > 3 else 3
W = reference.generate_weights(reference.WEIGHTS_SEED)[layer]
lib = L._G30[(m, S, 16, 16)]
o = torch.zeros(1, device="cuda")
e0 = bench.time_graph(lambda: o.zero_())
rows = []
for N in (W.shape[0], W.shape[0] // 2):
    for K in (W.shape[1], W.shape[1] // 2, W.shape[1] // 4):
        w = W[:N, :K].contiguous()
        code, sm, hdr, scale = L._compress(w)
        P = L._pack30(code, sm)
        x = torch.randn(m, K, device="cuda").bfloat16()
        out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
        t = min(bench.time_graph(lambda: lib.g30_gemm(x, P, hdr, out, scale, 0)) for _ in range(2)) - e0
        nb = P.numel() * P.element_size()
        rows.append((N, K, nb, t))
        print(f"{layer} m{m} S{S} N{N} K{K} ({N // 64 * S} CTAs) {nb / 1e6:.2f} MB: {t:.2f} us over a 1-kernel graph ({nb / t / 1e6:.2f} TB/s)", flush=True)
for N in sorted({r[0] for r in rows}):
    rs = [r for r in rows if r[0] == N]
    A = torch.tensor([[1.0, r[2] / 1e6] for r in rs]); y = torch.tensor([r[3] for r in rs])
    a, b = torch.linalg.lstsq(A, y[:, None]).solution.flatten().tolist()
    print(f"N{N}: fit t = {a:.2f} us + MB x {b:.3f} us/MB ({1 / b:.2f} TB/s marginal)")
