"""n78: dev/n65/ncu_drv.py with a vjx build variant: python dev/n78/ncu_drv.py <batch> <l2|hbm> <base|ss3|ovl|ovls>"""
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "dev/n78")
import bench, reference, vjx_src, vjx_dyn, vjx_qlo, vjx_ovl
B, mode, var = int(sys.argv[1]), sys.argv[2], sys.argv[3]
s = vjx_dyn.patch(vjx_src.CUDA_SRC, 100, 2)
tag = {"base": "n75b"}.get(var, "n78" + var)
if var != "base":
    s = vjx_qlo.patch(s, "ss3")
if var.startswith("ovl"):
    s = vjx_ovl.patch(s, fast=var != "ovls")
vjx_src.CUDA_SRC = s
vjx_src._MOD = vjx_src._build(tag=tag)
w = reference.generate_weights(reference.WEIGHTS_SEED)
c = dict(batch=B, ctx=80000, layer="full", page=1, q_len=8, dist="soft", spread="agentic")
data = bench.make_input(c, w, 7)
torch.manual_seed(0)
qkv = torch.randn(B, 8, reference.QKV_SIZE, device="cuda", dtype=torch.bfloat16)
pt = data["page_table"]
if mode == "l2":
    pt = torch.where(pt >= 0, pt % 8192, pt)
vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens")}
vd.update(qkv=qkv, config=data["config"], page_table=pt)
for _ in range(3):
    vjx_src._launch(vd, pdl=False)
torch.cuda.synchronize()
print("ok")
