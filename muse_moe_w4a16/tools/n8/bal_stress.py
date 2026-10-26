import sys, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1])
w, s = bench.weights_state(sub)
st = s["moe"]
print("sa", st["sa"])
g = torch.Generator(device="cuda").manual_seed(11)
bad = tot = 0
N = int(sys.argv[2])
for T in (2, 5, 8, 9, 16, 24, 32):
    for r in range(N):
        x = (torch.randn(T, 4096, device="cuda", generator=g)).bfloat16()
        ids = torch.randint(0, (40, 12, 64, 34)[r % 4], (T, 8), device="cuda", generator=g, dtype=torch.int32)
        ids = torch.where(ids >= 32, torch.full_like(ids, -1), ids)
        tw = torch.rand(T, 8, device="cuda", generator=g)
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": st["start"], "local_count": st["count"]}}
        sub._moe_TK2F_BAL_M.clear(); a = sub._moe_kernel_fn(d, st).clone()
        sub._moe_TK2F_BAL_M.add(T)
        assert sub._moe__tk2f_cfg(T, st) >= 1000000
        for rep in range(2):
            b = sub._moe_kernel_fn(d, st)
            torch.cuda.synchronize()
            tot += 1
            if not torch.equal(a, b):
                bad += 1; print("MISMATCH", T, r, (a.float()-b.float()).abs().max().item())
        sub._moe_TK2F_BAL_M.clear()
    print(T, "done", bad, tot, flush=True)
print("bad", bad, "of", tot)
