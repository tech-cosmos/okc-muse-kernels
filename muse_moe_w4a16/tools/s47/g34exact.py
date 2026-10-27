"""moe s47: g34 bit-exactness: x = one-hot rows at escape columns (+ random columns) -> out[t] == scale-exact W[:, col] (FP32)."""
import os, sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g34.py")).read().split("e0 = bench")[0])
bad = 0
for name in ["pool", "unp"]:
    w = W[{"pool": "pooler", "unp": "unpooler"}[name]]
    code, sm, hdr, scale = layer._compress(w)
    wp = layer._pack30(code, sm); hdr = hdr.contiguous()
    N, K = w.shape
    escc = torch.unique(hdr[:, :, 0][hdr[:, :, 1] != 2.0 ** -15].contiguous().view(torch.int32))
    cols = torch.cat([escc, torch.randint(0, K, (4 * MT,), device="cuda")]).long()
    for S in (1, 2):
        for c0 in range(0, len(cols), MT):
            cc = cols[c0:c0 + MT]
            x = torch.zeros(MT, K, device="cuda", dtype=torch.bfloat16)
            x[torch.arange(len(cc), device="cuda"), cc] = 1
            out = torch.zeros(MT, N, device="cuda")
            mod.g34_gemm(x, wp, hdr, out, scale, S, 0)
            ref = torch.zeros(MT, N, device="cuda"); ref[:len(cc)] = w[:, cc].float().t()
            bad += int((out != ref).sum())
    print(name, "escape cols", len(escc), "bad so far", bad, flush=True)
print("MISMATCHES", bad)
