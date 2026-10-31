"""Routing-check risk vs query precision (reference only, no submission, no timing).
python tools/fresh/x16/qerr.py <filter> <n_inputs> <router_seeds>
Per variant (fp8 = E4M3 queries like ours/SGLang; hilo = q8 + e4m3(q - q8), two E4M3 terms): check_routing fails, max |dbiased|,
p99.9 |dbiased| over (token, expert), worst dropped-keep lead (limit 0.03 at <= 256 tokens)."""
import sys, inspect, torch
sys.path.insert(0, ".")
import bench, reference as R

src = inspect.getsource(R._attention).replace("def _attention(", "def _attention_v(")
src = src.replace("""    else:  # FP8 on the normalized query, qk_scale_factor applied after
        q = q.to(query_dtype).float() * cfg["qk_scale_factor"]""",
"""    elif query_dtype == "hilo":
        q8 = q.to(torch.float8_e4m3fn).float()
        q = (q8 + (q - q8).to(torch.float8_e4m3fn).float()) * cfg["qk_scale_factor"]
    else:
        q = q.to(query_dtype).float() * cfg["qk_scale_factor"]""")
assert "hilo" in src
exec(src, R.__dict__)


def front(d, w, qd):
    o, _, _ = R._attention_v(d, d["attn_input"].float(), w, qd)
    a = o @ w["o"].float().T
    a1, b1 = R.residual_coefficients(w["attn_residual_gate"])
    s1 = a1 * d["stream"].float() + b1 * R._rms(a, R.POST_NORM_EPS)
    h = R._rms(s1) * (1.0 + w["pre_ffn_norm"].float())
    return R.route(h, w)[1]


flt, n = sys.argv[1], int(sys.argv[2])
seeds = [int(s) for s in sys.argv[3].split(",")]
for rs in seeds:
    w = R.generate_weights(R.WEIGHTS_SEED, router_seed=rs)
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name:
            continue
        per = []
        st = {v: [0, 0.0, [], -1.0] for v in ("fp8", "hilo")}
        for i in range(n):
            d = bench.make_input(c, w, 9001 + 31 * i + 1000 * rs)
            ref = front(d, w, torch.bfloat16)
            ls = int(d["local_start"])
            best = ref.topk(9, dim=-1)
            for v, qd in (("fp8", torch.float8_e4m3fn), ("hilo", "hilo")):
                b = front(d, w, qd)
                ids = b.topk(8, dim=-1).indices.to(torch.int32)
                s = st[v]
                if R.check_routing(ids, ref, ls):
                    s[0] += 1
                dd = (b - ref).abs()
                s[1] = max(s[1], float(dd.max()))
                s[2].append(dd.flatten()[::7].cpu())
                kept = (best.indices[:, :8, None] == ids.long()[:, None, :]).any(-1)
                local = (best.indices[:, :8] - ls) % R.N_EXPERTS < R.LOCAL_EXPERTS
                lead = (best.values[:, :8] - best.values[:, 8:9]).masked_fill(~(local & ~kept), -1.0)
                s[3] = max(s[3], float(lead.max()))
                if v == 'fp8': per.append(float(lead.max()))
            del d
        out = []
        for v, s in st.items():
            q = torch.cat(s[2]).float().quantile(torch.tensor([0.99, 0.9999])).tolist() if False else None
            allv = torch.cat(s[2]).float()
            k = max(1, int(allv.numel() * 1e-4))
            p9999 = float(allv.topk(k).values[-1])
            out.append(f"{v}: fails {s[0]}/{n} max|d| {s[1]:.4f} p99.99 {p9999:.4f} worst drop-lead {s[3]:.4f}")
        print(f"rs {rs} {name:16s} | " + " | ".join(out), flush=True)
        print('PER', rs, name, ' '.join(f'{x:.4f}' for x in per), flush=True)
