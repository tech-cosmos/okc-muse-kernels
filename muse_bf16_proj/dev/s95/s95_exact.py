"""s100: bit-exact W decode through g30: x = one-hot rows (token t picks column c_t) -> out[t][n] must equal W[n][c_t]
bitwise. Columns cover every escape column (several passes). usage: python dev/s100_g30exact.py [seed ...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import bench, reference
L = bench.load_sub("submission.py")
L._G30_SRC = "".join(f"#define {d}\n" for d in (f"G30_MTOK {os.environ.get('M',128)}," + os.environ.get("DEFS", "")).split(",") if d) + open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g30_src.cu")).read()  # the dev copy
L._g30_build()
M = int(os.environ.get("M", 128))
lib = L._G30["lib"]
for seed in [int(a) for a in sys.argv[1:]] or [0]:
    w = reference.generate_weights(seed)[os.environ.get("KEY", "unpooler")]
    code, sm, hdr, scale = L._compress(w)
    wp = L._pack30(code, sm)
    esc_cols = torch.unique(hdr[:, :, 0][hdr[:, :, 1] != 2.0 ** -15].contiguous().view(torch.int32)).tolist()
    cols = esc_cols + torch.randperm(w.shape[1])[:256].tolist()
    bad = 0
    for p in range(0, len(cols), M):
        cs = (cols[p:p + M] + cols[:M] + cols[:M])[:M]
        x = torch.zeros(M, w.shape[1], device="cuda", dtype=torch.bfloat16)
        x[torch.arange(M), torch.tensor(cs)] = 1
        out = torch.empty(M, w.shape[0], device="cuda", dtype=torch.bfloat16)
        lib.g30_gemm(x, wp, hdr, out, scale, 0)
        ref = w[:, cs].t().contiguous()
        bad += int((out.view(torch.int16) != ref.view(torch.int16)).sum())
    print(f"seed {seed}: {len(esc_cols)} escape columns + 256 random, {len(cols) * w.shape[0]} weights checked: {bad} bit mismatches")
