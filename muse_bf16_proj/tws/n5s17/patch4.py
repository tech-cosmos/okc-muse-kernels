import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b, n=1):
    global s
    c = s.count(a); assert c == n, (c, a[:90]); s = s.replace(a, b)
rep('TORCH_CHECK(!(g_hot & 1) && ns == 0, "TWS: no hot / K split");', 'TORCH_CHECK(ns == 0, "TWS: no K split");')
rep('''        "int64_t moe_twm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"''',
'''        "int64_t moe_twm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, int64_t);"''')
rep('''            ns = TWF_SPLIT_M.get(M, TWF_SPLIT)
            part = (torch.full if TWF_DBG else torch.empty)((67 * 32768 if DIAG_TL else max(66 if ns >= 1022 else ns, 1) * 32768,), *((float("nan"),) if TWF_DBG else ()), dtype=torch.float32, device=dev)''',
'''            ns = TWF_SPLIT_M.get(M, TWF_SPLIT)
            stw = TWM_STEAL_M.get(M, 0)  # n5s17 TWS (moe_twm work-stealing gemm2 tail): NV | MS << 10 | B << 16
            if stw: ns = 0
            part = (torch.full if TWF_DBG else torch.empty)((67 * 32768 + ((stw & 1023) + 132 * 4) * 16384 if stw else 67 * 32768 if DIAG_TL else max(66 if ns >= 1022 else ns, 1) * 32768,), *((float("nan"),) if TWF_DBG else ()), dtype=torch.float32, device=dev)''')
rep('''                                   st["w13sb2"], st["w2b2"], st["w2sb2"], st["lut"], xg, meta, sroutes, inter, y, st["tk"],
                                   part, ns)''', '''                                   st["w13sb2"], st["w2b2"], st["w2sb2"], st["lut"], xg, meta, sroutes, inter, y, st["tk"],
                                   part, ns, st["tws"], stw)''')
open(p, "w").write(s)
print("ok")
