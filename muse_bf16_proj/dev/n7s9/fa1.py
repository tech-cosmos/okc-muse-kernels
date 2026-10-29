"""proj n7s9: fa1 (d1 full attention, 1 CTA/SM, whole KV slice in smem) test + H100/H200 timing vs the layer's
attn_src path (_gd_full + _red_fin). Usage: python fa1.py [test|bench] (LAYER=<dir with attn_src.py, reference.py>)."""
import math, os, sys
sys.dont_write_bytecode = True
import torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER = os.environ.get("LAYER", os.path.expanduser("~/r-layer/muse_decode_layer"))
sys.path.insert(0, LAYER)
import reference as R  # noqa: E402
import fcntl  # noqa: E402


class uc:  # time_graph from dev/n7s8/uc.py: CUDA graph, 512 MB read flush, GPU lock around the timed loop
    @staticmethod
    def time_graph(fn, reps=100):
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
        fs = torch.empty((), dtype=torch.int64, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        ts = []
        with open("/tmp/okc-gpu.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(reps):
                torch.sum(flush, 0, out=fs)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                ts.append(a.elapsed_time(b) * 1000)
        del flush
        ts.sort()
        return ts[len(ts) // 2]


def build():
    src = open(os.path.join(HERE, "fa1.cu")).read()
    return load_inline("pj_fa1_" + str(abs(hash(src)) % 10**8), cpp_sources="void fa1_launch(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vs, torch::Tensor pt, torch::Tensor seq, torch::Tensor loc, torch::Tensor red, torch::Tensor cnt, torch::Tensor flag, torch::Tensor out, int64_t nb, int64_t ch, double qsf, double smsc, int64_t nr);", cuda_sources=src,
                       functions=["fa1_launch"], extra_cuda_cflags=["-O3", "-arch=sm_90a", "-lineinfo"], verbose=False)


def ref_attn(data, qkv):
    """reference._attention from a given qkv row (batch 1, q_len 1), FP32."""
    cfg = data["config"]
    x = qkv.float().view(1, -1)
    q, k, v, gate = x.split([R.Q_SIZE, R.KV_SIZE, R.KV_SIZE, R.Q_SIZE], dim=-1)
    q = R._rms(q.view(-1, R.HQ, R.HEAD_DIM))
    k = R._rms(k.view(-1, R.HKV, R.HEAD_DIM))
    q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16).float().view(R.HQ, R.HEAD_DIM)
    k_new = R._quant(k, data["k_scale"]).view(R.HKV, R.HEAD_DIM)
    v_new = R._quant(v.view(-1, R.HKV, R.HEAD_DIM), data["v_scale"]).view(R.HKV, R.HEAD_DIM)
    seq = int(data["seq_lens"][0])
    rows = data["page_table"][0].long()[:seq]
    kf = data["k_cache"].view(-1, R.HKV, R.HEAD_DIM)
    vf = data["v_cache"].view(-1, R.HKV, R.HEAD_DIM)
    keys = torch.cat([R._dequant(kf[rows], data["k_scale"]), R._dequant(k_new[None], data["k_scale"])])
    vals = torch.cat([R._dequant(vf[rows], data["v_scale"]), R._dequant(v_new[None], data["v_scale"])])
    att = torch.einsum("kgd,tkd->kgt", q.view(R.HKV, 8, R.HEAD_DIM), keys) * cfg["sm_scale"]
    out = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vals).reshape(1, R.HQ, R.HEAD_DIM)
    return (R._rms(out).reshape(-1, R.Q_SIZE) * torch.sigmoid(gate)), k_new, v_new


class Fa1:
    def __init__(self, mod, data, nb=None):
        dev = data["k_cache"].device
        self.mod, self.data = mod, data
        self.nb = nb or torch.cuda.get_device_properties(dev).multi_processor_count
        cap = data["page_table"].shape[1] - 1  # past tokens at most (q_len 1)
        self.ch = (math.ceil(cap / self.nb) + 15) // 16 * 16
        self.nr = int(os.environ.get("NR", 4))
        self.red = torch.zeros(self.nr * (16 * 64 + 16), dtype=torch.float32, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=dev)
        self.flag = torch.zeros(256, dtype=torch.int32, device=dev)
        self.out = torch.empty(1024, dtype=torch.bfloat16, device=dev)

    def __call__(self, qkv):
        d, c = self.data, self.data["config"]
        self.mod.fa1_launch(qkv, d["k_cache"], d["v_cache"], d["k_scale"], d["v_scale"], d["page_table"], d["seq_lens"],
                            d["cache_loc"], self.red, self.cnt, self.flag, self.out, self.nb, self.ch,
                            c["qk_scale_factor"], c["sm_scale"], self.nr)
        return self.out


def make(seed, ctx=80000, kv_scale=1.0, dist="soft"):
    data = R.generate_input(1, ctx, "full", seed, q_len=1, dist=dist, page=1, kv_scale=kv_scale, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    qkv = (torch.randn(2304, generator=g, device="cuda") * 2.0).bfloat16()
    return data, qkv


def test(mod):
    bad = 0
    for seed, ctx, kvs, dist in [(1, 80000, 1.0, "soft"), (2, 80000, 1.0, "normal"), (3, 3000, 1.0, "soft"),
                                 (4, 20000, 0.015625, "soft"), (5, 80000, 1.0, "soft"), (6, 10, 1.0, "normal")]:
        data, qkv = make(seed, ctx, kvs, dist)
        loc = int(data["cache_loc"][0])
        exp, kn, vn = ref_attn(data, qkv)
        f = Fa1(mod, data)
        for rep in range(3):
            got = f(qkv).float().view(1, -1)
        torch.cuda.synchronize()
        err = ((got - exp).norm() / exp.norm()).item()
        kok = torch.equal(data["k_cache"].view(-1, 2, 64)[loc].view(torch.uint8), kn.view(torch.uint8))
        vok = torch.equal(data["v_cache"].view(-1, 2, 64)[loc].view(torch.uint8), vn.view(torch.uint8))
        ok = err < 2e-3 and kok and vok and int(f.flag[0]) == 0
        bad += not ok
        print(f"seed {seed} ctx {ctx} kvs {kvs} {dist}: rel_l2 {err:.2e} k {kok} v {vok} flag {int(f.flag[0])} ch {f.ch}"
              f" {'OK' if ok else 'FAIL'}")
    print("ALL OK" if not bad else f"{bad} FAIL")


def bench(mod):
    data, qkv = make(11)
    f = Fa1(mod, data); f.nr0 = f.nr
    e = uc.time_graph(lambda: None, 200)
    t = uc.time_graph(lambda: f(qkv), 200) - e
    print(f"empty {e:.2f} | fa1 nb {f.nb} ch {f.ch}: {t:.2f} us")
    for md in (0, 10, 8):
        f.nr = f.nr0 + (md << 8)
        print(f"  mode {md}: {uc.time_graph(lambda: f(qkv), 200) - e:.2f} us")
    f.nr = f.nr0
    fl = uc.time_graph
    import types
    def warm(fn, reps=200):  # no flush: KV stays in L2 from the previous replay (H100 50 MB L2)
        g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
        ts.sort(); return ts[len(ts) // 2]
    ew = warm(lambda: None)
    for md in (0, 4, 7):
        f.nr = f.nr0 + (md << 8)
        print(f"  L2-warm mode {md}: {warm(lambda: f(qkv)) - ew:.2f} us")
    f.nr = f.nr0
    for nb in (128,):
        f2 = Fa1(mod, data, nb)
        print(f"  fa1 nb {nb} ch {f2.ch}: {uc.time_graph(lambda: f2(qkv), 200) - e:.2f} us")
    try:
        import attn_src as A
        d2 = dict(data)
        d2["qkv"] = qkv.view(1, 1, -1)
        ref = A.kernel_fn(d2)
        t2 = uc.time_graph(lambda: A.kernel_fn(d2), 200) - e
        print(f"attn_src.kernel_fn (layer _gd_full+_red_fin path): {t2:.2f} us")
        exp = ref_attn(data, qkv)[0]
        print("  attn_src rel_l2", ((ref.float().view(1, -1) - exp).norm() / exp.norm()).item(),
              "fa1 rel_l2", ((f(qkv).float().view(1, -1) - exp).norm() / exp.norm()).item())
    except Exception as ex:  # noqa: BLE001
        print("attn_src baseline failed:", repr(ex)[:300])


def main():
    mod = build()
    {"test": test, "bench": bench, "kvwarm": kvwarm}[sys.argv[1] if len(sys.argv) > 1 else "test"](mod)


def kvwarm(mod):
    """fa1 with ONLY the KV rows L2-resident (as a KV_PF-style prefetch in the qkv g13 would leave them): graph =
    [fa1 loads-only pass (mode 1) ; fa1], minus the graph [mode 1]. Code/pt also warm-ish (same kernel): optimistic."""
    data, qkv = make(11)
    f, f1 = Fa1(mod, data), Fa1(mod, data)
    f1.nr = f1.nr + (1 << 8)
    e = uc.time_graph(lambda: None, 200)
    a = uc.time_graph(lambda: f1(qkv), 200) - e
    b = uc.time_graph(lambda: (f1(qkv), f(qkv)), 200) - e
    c = uc.time_graph(lambda: f(qkv), 200) - e
    import attn_src as A
    d2 = dict(data); d2["qkv"] = qkv.view(1, 1, -1)
    A.kernel_fn(d2)
    r = uc.time_graph(lambda: (f1(qkv), A.kernel_fn(d2)), 200) - e
    r0 = uc.time_graph(lambda: A.kernel_fn(d2), 200) - e
    print(f"gather pass {a:.2f} | fa1 cold {c:.2f} | fa1 after gather {b - a:.2f} | attn_src cold {r0:.2f} | attn_src after gather {r - a:.2f} us")


if __name__ == "__main__":
    main()
