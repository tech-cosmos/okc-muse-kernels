"""s63 two-step decode stress (for pre-wait loads, e.g. CSL_PW): per scored sliding shape, capture kernel_fn once; per
input D, first replay step A = D moved back by q_len tokens (seq_lens - L, positions - L, cache_loc = the L slots before
D's), which writes D's last L context rows into the cache; then copy D's inputs EXCEPT the caches and replay step B,
which must gather those rows. Both steps are checked against reference.
python tools/step_test.py [submission.py] [n_inputs] [filter] [force: T:key=val,...]"""
import sys, torch
sys.path.insert(0, ".")
import bench, reference

path = sys.argv[1] if len(sys.argv) > 1 else "submission.py"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
flt = sys.argv[3] if len(sys.argv) > 3 else ""
force = [f.split(":") for f in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
sub = bench.load_sub(path)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = sub.prepare(w)
for t, kv in force:
    k, v = kv.split("=")
    st.setdefault("tune", {}).setdefault((int(t), True), {})[k] = eval(v)
print("tune", {k: v for k, v in st.get("tune", {}).items() if k[1]})


def back(d, L):
    """D moved back by L tokens: same caches/page table, the L slots before D's new ones become the new ones."""
    a = dict(d)
    a["seq_lens"] = d["seq_lens"] - L
    a["positions"] = d["positions"] - L
    a["cache_seqlens"] = d["cache_seqlens"] - L
    B = d["seq_lens"].shape[0]
    pos = (a["seq_lens"].long()[:, None] + torch.arange(L, device="cuda")).reshape(B, L)
    page = d["config"]["page_size"]
    a["cache_loc"] = (torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page).reshape(-1)
    return a


def copy_into(dst, src, skip=()):
    for k, v in src.items():
        if isinstance(v, torch.Tensor) and k not in skip:
            assert dst[k].shape == v.shape, k
            dst[k].copy_(v)


for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if c["layer"] != "sliding" or flt not in name:
        continue
    L = c["q_len"]
    data = bench.make_input(c, w, 7000)
    sub.kernel_fn(data, st)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sub.kernel_fn(data, st)
        with torch.cuda.graph(gr, stream=s):
            out = sub.kernel_fn(data, st)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    fails = [0, 0]
    for i in range(n):
        d = bench.make_input(c, w, 9001 + 31 * i)
        if int(d["seq_lens"].min()) <= L:
            continue
        for step, src in enumerate((back(d, L), d)):
            copy_into(data, src, skip=("k_cache", "v_cache") if step else ())
            torch.sum(flush.view(torch.int64), 0, out=sink)
            gr.replay()
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out), w)
            if not ok:
                fails[step] += 1
                print("    FAIL", i, "step", step, msg[:200], flush=True)
        del d
    print(f"{name:16s} fails step A {fails[0]}/{n} step B {fails[1]}/{n}", flush=True)
    del gr, data, out
    torch.cuda.empty_cache()
