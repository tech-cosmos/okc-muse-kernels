"""n57: in-layer VJX off/on A/B with the routing pinned: the scored input (real lengths, cache, page table) with its
stream replaced by the tuner's skewed synthetic stream (_synth_stream: 8 router rows + noise, so both attention paths
route alike; topk agreement printed). python dev/n57/pin_ab.py <file> <filter> <seeds> <rounds>"""
import sys, statistics, torch
sys.path.insert(0, ".")
import bench
sub = bench.load_sub(sys.argv[1]); flt = sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]; rounds = int(sys.argv[4])
w, st = bench.weights_state(sub)
e0 = bench.empty_cost()
ON, OFF = (1, 8, 16, 32, 128, 256), ()
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if flt not in name:
        continue
    for sd in seeds:
        data = bench.make_input(c, w, sd)
        T = data["config"]["tokens"]
        g = torch.Generator(device="cuda").manual_seed(sd)
        s = sub._synth_stream(st["router"], T, g)
        data["stream"] = s
        data["attn_input"] = (s * torch.rsqrt(s.square().mean(-1, keepdim=True) + sub.RMS_EPS)).to(torch.bfloat16)
        key = (T, False)
        ids = []
        for vt in (OFF, ON):
            sub.VJX_TOKENS = vt
            ids.append(sub.kernel_fn(data, st)[-1].clone())
        agree = (ids[0] == ids[1]).float().mean().item()
        t = [[], []]
        for r in range(rounds):
            for i in ((0, 1) if r % 2 == 0 else (1, 0)):
                sub.VJX_TOKENS = (OFF, ON)[i]
                t[i].append(bench.time_graph(lambda: sub.kernel_fn(data, st)) - e0)
        m0, m1 = statistics.median(t[0]), statistics.median(t[1])
        print(f"{name} s{sd}: off {m0:7.2f}  on {m1:7.2f}  d {m1 - m0:+6.2f} us  topk agree {agree:.3f}  tune {st['tune'].get(key)}", flush=True)
        del data
