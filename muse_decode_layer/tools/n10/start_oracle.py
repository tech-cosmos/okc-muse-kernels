"""n10s1 LAYER START oracle (H100, in-layer): python tools/n10/start_oracle.py <file> <filter> <rounds> <variants...>
variants: base | free (qkv launches nothing: its output is computed once eagerly and replayed from a clone; timing only)
| rd<N> (free + a bare Triton reader of qkv's packed W bytes on N CTAs as the first op) | pre (empty kernel before qkv)
| prerd<N> (free + empty kernel + reader). Prints per-shape medians (min) of cold-L2 graph replays minus empty replay."""
import sys, os, statistics, torch, triton, triton.language as tl
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; rounds = int(sys.argv[3]); variants = sys.argv[4:]
w, st = bench.weights_state(sub)

@triton.jit
def _rd(p, n, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0); np_ = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for i in range(pid * BLOCK, n, np_ * BLOCK):
        o = i + tl.arange(0, BLOCK)
        acc += tl.load(p + o, mask=o < n, other=0)
    tl.store(out + pid, tl.sum(acc, 0))

@triton.jit
def _nop(out):
    pass

SINK = torch.zeros(4096, dtype=torch.int64, device="cuda")
MODE = {"v": "base", "path": None, "cache": {}, "wbytes": None}
o_g19, o_lin, o_bmm = sub._g19, sub._linear, torch.bmm

def _pre():
    v = MODE["v"]
    if v.startswith("pre"):
        _nop[(1,)](SINK)

def _reader(t):
    v = MODE["v"]
    if "rd" in v:
        n = int(v.split("rd")[1])
        x = t.view(torch.int64) if t.numel() * t.element_size() % 8 == 0 else t
        _rd[(n,)](x.view(-1), x.numel(), SINK, BLOCK=1024, num_warps=8)

def _cached(key, fn):
    v = MODE["v"]
    if v == "base" or v == "pre":
        return fn()
    c = MODE["cache"]
    if key not in c:
        r = fn()
        c[key] = None if r is None else r.clone()
        return r
    return None if c[key] is None else c[key].clone()

def g19(x, key, state, trig=False):
    if key != "qkv_gate":
        return o_g19(x, key, state, trig)
    if "rd" not in MODE["v"]:
        _pre()
    r = _cached(("g19", x.shape[0]), lambda: o_g19(x, key, state, trig))
    if r is not None:
        MODE["path"] = "g19"
    return r

def lin(x, wt, key, state=None, sliding=False, kv_pf=None):
    if key != "qkv_gate":
        return o_lin(x, wt, key, state, sliding, kv_pf)
    MODE["path"] = MODE["path"] or "linear"
    return _cached(("lin", x.shape[0]), lambda: o_lin(x, wt, key, state, sliding, kv_pf))

sub._g19, sub._linear = g19, lin
names = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if flt in f"{l} {c['layer']}"]
data = {n: bench.make_input(c, w, 99) for n, c in names}
# W bytes the qkv kernel streams: packed 12-bit if the pick has one, else BF16 dense
def wsrc(T):  # qkv's 12-bit byte count (N 2304 x K 8192 x 1.5 B), read from the BF16 dense weight's first bytes
    d = st["dense"]["qkv_gate"].view(-1).view(torch.uint8)
    return "dense-12bit-bytes", d[: 2304 * 8192 * 3 // 2]
e0 = bench.empty_cost()
res = {}
for r in range(rounds):
    for v in variants:
        for n, c in names:
            MODE.update(v=v, path=None, cache={})
            T = c["batch"] * c["q_len"]
            src, t = wsrc(T)
            def fn():
                if "rd" in v:
                    _pre(); _reader(t)
                return sub.kernel_fn(data[n], st)
            res.setdefault((n, v), []).append(bench.time_graph(fn) - e0)
            if r == 0 and v == variants[0]:
                print(f"# {n}: qkv path {MODE['path']} wsrc {src} {t.numel() * t.element_size() / 1e6:.1f} MB", flush=True)
for n, _ in names:
    print(f"{n:18s}", "  ".join(f"{v} {statistics.median(res[(n, v)]):7.2f} ({min(res[(n, v)]):6.2f})" for v in variants), flush=True)
