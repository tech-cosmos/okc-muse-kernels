"""n65: ablations of vjx's q_len-8 softmax (timing only; outputs wrong for noexp/nocvt): which pipe binds v16/v32 full.
variants: base, noexp (P's ex2 -> 1 LOP3 giving ~1.0), nocvt (P's f32->e4m3 cvt -> PRMT/LOP), both.
HBM (real page ids) and L2 (page ids % 8192) timings, CUDA-graph replay, cold L2.
python dev/n65/abl.py <file> <seed> <rounds> [variants]"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
import vjx_src
sub = bench.load_sub(sys.argv[1]); sd = int(sys.argv[2]); rounds = int(sys.argv[3])
names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["base", "noexp", "nocvt", "both"]
src0 = vjx_src.CUDA_SRC
FAKE = r"""
__device__ __forceinline__ float fex2(float x) { return __int_as_float(0x3f800000 | (__float_as_int(x) & 7)); }
#define FEXQ(x, q) fex2(x)
__device__ __forceinline__ uint32_t fe4m3x4f(float x0, float x1, float x2, float x3) {
  return __byte_perm(__byte_perm(__float_as_uint(x0), __float_as_uint(x1), 0x0073),
                     __byte_perm(__float_as_uint(x2), __float_as_uint(x3), 0x0073), 0x5410) & 0x38383838u;
}
"""
def variant(n):
    s = src0
    a = s.index("// four f16 (two f16x2 words")
    s = s[:a] + FAKE + s[a:]
    # only P's packs: the lines in the PV section
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        if "pa[kk][" in ln and "e4m3x4f(" in ln:
            if n in ("noexp", "both"):
                ln = ln.replace("ex2(fmaf(", "fex2(fmaf(").replace("EXQ(fmaf(", "FEXQ(fmaf(")
            if n in ("nocvt", "both"):
                ln = ln.replace("= e4m3x4f(", "= fe4m3x4f(")
            lines[i] = ln
    return "\n".join(lines)
mods = {}
for n in names:
    vjx_src.CUDA_SRC = variant(n)
    mods[n] = vjx_src._build(tag="n65" + n)
vjx_src.CUDA_SRC = src0
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
for load, c in bench.SCORED:
    if c["layer"] != "full" or c["q_len"] != 8 or c["batch"] == 1:
        continue
    data = bench.make_input(c, w, sd)
    cfg = data["config"]
    qkv = (data["attn_input"] @ st["dense"]["qkv_gate"].T).view(cfg["batch"], cfg["q_len"], -1)
    pt = data["page_table"]
    fns = {}
    for tag, p in (("hbm", pt), ("l2", torch.where(pt >= 0, pt % 8192, pt))):
        vd = {k: data[k] for k in ("cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "seq_lens")}
        vd.update(qkv=qkv, config=cfg, page_table=p)
        for n in names:
            fns[f"{n}_{tag}"] = (lambda vd=vd, m=mods[n]: vjx_src._launch(vd, _mod=m, pdl=False))
    t = {k: [] for k in fns}
    for r in range(rounds):
        for k in (list(fns) if r % 2 == 0 else list(fns)[::-1]):
            t[k].append(bench.time_graph(fns[k]) - e0)
    print(f"{load} full s{sd}: " + "  ".join(f"{k} {statistics.median(v):7.2f}" for k, v in t.items()), flush=True)
    del data, qkv, fns
