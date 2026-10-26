import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1] if len(sys.argv) > 1 else "submission.py")
w_, s = bench.weights_state(sub)
lib = sub._G10["lib"]
g = torch.Generator(device="cuda").manual_seed(3)
for key in sub.G13_CFG:
    W = s["dense"][key].clone()
    n, k = W.shape
    # adversarial escapes: per probed row, 0, a tiny normal and a subnormal far below the window
    rows = torch.randint(0, n, (64,), device="cuda", generator=g)
    cols = torch.randint(0, k, (64, 3), device="cuda", generator=g)
    vals = torch.tensor([0.0, -1e-20, 3e-39], device="cuda").bfloat16()
    for j in range(3): W[rows, cols[:, j]] = vals[j]
    pk13 = sub._pack13(W); pk14 = sub._pack14(W, 1024 * sub.G14_CFG[key][0]) if key in sub.G14_CFG else None
    assert pk13 is not None
    bad = 0
    for r in range(64):
        for j in range(3):
            c = int(cols[r, j]); x = torch.zeros(1, k, device="cuda", dtype=torch.bfloat16); x[0, c] = 1
            out = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
            lib.g13_gemv(pk13[0], pk13[1], pk13[2], x, out, pk13[3], *sub.G13_CFG[key])
            bad += not torch.equal(out[0].view(torch.int16), W[:, c].view(torch.int16))
            if pk14 is not None:
                x8 = torch.zeros(8, k, device="cuda", dtype=torch.bfloat16); x8[:, c] = 1
                o8 = torch.empty(8, n, device="cuda", dtype=torch.bfloat16)
                lib.g14_gemm(pk14[0], pk14[1], pk14[2], pk14[3], x8, o8, s["g12_ws"], s["g12_cnt"], n // 8, *sub.G14_CFG[key])
                bad += not torch.equal(o8.view(torch.int16), W[:, c].expand(8, n).contiguous().view(torch.int16))
    x = torch.randn(8, k, device="cuda", generator=g).bfloat16()
    ref = (x.float() @ W.float().t())
    o8 = torch.empty(8, n, device="cuda", dtype=torch.bfloat16)
    lib.g14_gemm(pk14[0], pk14[1], pk14[2], pk14[3], x, o8, s["g12_ws"], s["g12_cnt"], n // 8, *sub.G14_CFG[key])
    o1 = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
    lib.g13_gemv(pk13[0], pk13[1], pk13[2], x[:1].contiguous(), o1, pk13[3], *sub.G13_CFG[key])
    torch.cuda.synchronize()
    print(key, "probe mismatches", bad, "rand rel err g14", ((o8.float()-ref).norm()/ref.norm()).item(), "g13", ((o1.float()-ref[:1]).norm()/ref[:1].norm()).item())
