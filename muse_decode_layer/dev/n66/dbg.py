import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "dev/n66")
import bench, vjx_src, dyn, reference
B = int(sys.argv[1]); f = int(sys.argv[2])
base_src = vjx_src.CUDA_SRC
mb = vjx_src._build(tag="n66b")
vjx_src.CUDA_SRC = dyn.patch(base_src, f)
md = vjx_src._build(tag=f"n66d{f}")
w = reference.generate_weights(reference.WEIGHTS_SEED)
c = dict(batch=B, ctx=80000, layer="full", page=1, q_len=8, dist="soft", spread="agentic")
data = bench.make_input(c, w, 7)
cfg = data["config"]
torch.manual_seed(0)
qkv = torch.randn(B, 8, reference.QKV_SIZE, device="cuda", dtype=torch.bfloat16)
vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens", "page_table")}
vd.update(qkv=qkv, config=cfg)
print("seq", data["seq_lens"].tolist())
print("plan", vjx_src._plan(B, cfg["num_kv_heads"] // 2, 8, cfg["window"], data["page_table"].shape[1], 1, "cuda"))
a = vjx_src._launch(vd, _mod=mb).float()
for r in range(3):
    d = vjx_src._launch(vd, _mod=md).float()
    torch.cuda.synchronize()
    e = ((d - a).norm(dim=(1, 2)) / a.norm(dim=(1, 2)))
    print("per-req err", [f"{x:.1e}" for x in e.tolist()])
    print("ctr", vjx_src._WS["cuda:0"][2][31000:31000 + B].tolist() if "cuda:0" in vjx_src._WS else list(vjx_src._WS))
for ns in (260, 256):
    b2 = vjx_src._launch(vd, _mod=mb, n_splits=ns).float()
    print("base ns", ns, "vs base264 err", f"{((b2 - a).norm() / a.norm()).item():.1e}", "vs dyn", f"{((b2 - d).norm() / d.norm()).item():.1e}")
