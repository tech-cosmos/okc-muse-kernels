"""n65: eager driver for ncu on vjx's attention kernel (no submission prepare): random BF16 qkv, layer inputs.
python dev/n65/ncu_drv.py <batch> <q_len> <l2|hbm> [iters]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference, vjx_src
B, L, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
it = int(sys.argv[4]) if len(sys.argv) > 4 else 3
w = reference.generate_weights(reference.WEIGHTS_SEED)
c = dict(batch=B, ctx=80000, layer="full", page=1, q_len=L, dist="soft")
if B > 1:
    c["spread"] = "agentic"
data = bench.make_input(c, w, 7)
cfg = data["config"]
torch.manual_seed(0)
qkv = torch.randn(B, L, reference.QKV_SIZE, device="cuda", dtype=torch.bfloat16)
pt = data["page_table"]
if mode == "l2":
    pt = torch.where(pt >= 0, pt % 8192, pt)
vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens")}
vd.update(qkv=qkv, config=cfg, page_table=pt)
for _ in range(it):
    vjx_src._launch(vd, pdl=False)
torch.cuda.synchronize()
print("ok")
