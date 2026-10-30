"""fresh x5 oracle (3) split: is the page-table oracle's gain the address CHAIN or the KV rows' physical ORDER?
Times files on the same input with page_table (and cache_loc) = orig (random pages, as the judge) | sorted per request (same
slots, ascending) | identity (b*maxp + t). python tools/fresh/sw/pt_order.py <filter> <rounds> <file> ..."""
import sys, statistics, importlib.util, torch
sys.path.insert(0, ".")
import bench, reference
flt = sys.argv[1]; rounds = int(sys.argv[2]); files = sys.argv[3:]
w = reference.generate_weights(reference.WEIGHTS_SEED)
subs = []
for i, f in enumerate(files):
    spec = importlib.util.spec_from_file_location(f"sub{i}", f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    subs.append((m, m.prepare(w)))
def variant(d, kind):
    d = dict(d); pt = d["page_table"].clone(); B, MP = pt.shape
    if kind == "sorted":
        for b in range(B):
            n = int((pt[b] >= 0).sum()); pt[b, :n] = pt[b, :n].sort().values
    elif kind == "ident":
        n = (pt >= 0); pt = torch.where(n, torch.arange(B * MP, device=pt.device, dtype=pt.dtype).view(B, MP), pt)
    # move the cache rows with the table, so every logical position keeps its key/value (same output, same routing)
    old, v = d["page_table"].long(), pt >= 0
    for key in ("k_cache", "v_cache"):
        src = d[key]; dst = src.clone()
        dst.view(src.shape[0], -1)[pt.long()[v]] = src.view(src.shape[0], -1)[old[v]]
        d[key] = dst
    d["page_table"] = pt
    pos = (d["seq_lens"].long()[:, None] + torch.arange(d["cache_loc"].numel() // B, device=pt.device)).view(B, -1)
    d["cache_loc"] = torch.gather(pt.long(), 1, pos).reshape(-1).to(d["cache_loc"].dtype)
    return d
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
KINDS = ["orig", "sorted", "ident"]
data = {}
for n, c in names:
    d0 = bench.make_input(c, w, 99)
    print(n, "k_cache", tuple(d0["k_cache"].shape), f"{d0['k_cache'].nbytes / 1e6:.0f} MB", flush=True)
    for k in KINDS: data[(n, k)] = d0 if k == "orig" else variant(d0, k)
m0, s0 = subs[0]
for n, _ in names:  # same logical input -> same output (and routing) for every layout and file
    ref = reference.collect(data[(n, "orig")], m0.kernel_fn(data[(n, "orig")], s0)).float()
    for i, (m, st) in enumerate(subs):
        for k in KINDS[1:] if i == 0 else KINDS[2:]:
            o = reference.collect(data[(n, k)], m.kernel_fn(data[(n, k)], st)).float()
            print(f"eq {n} [{i}]{k}: max|diff| {(o - ref).abs().max().item():.3g}", flush=True)
res = {}
e0 = bench.empty_cost()
for r in range(rounds):
    for i, (m, st) in enumerate(subs):
        for n, _ in names:
            for k in KINDS:
                res.setdefault((n, i, k), []).append(bench.time_graph(lambda: m.kernel_fn(data[(n, k)], st)) - e0)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"[{i}]{k} {statistics.median(res[(n, i, k)]):7.2f}" for i in range(len(files)) for k in KINDS), flush=True)
