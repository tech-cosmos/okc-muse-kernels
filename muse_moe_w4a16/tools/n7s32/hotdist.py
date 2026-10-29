# per router seed x input seed: T=256 sliding (v32) tokens per local expert; nh = experts with 65..128 tokens
import sys, collections, torch
sys.path.insert(0, "/home/daytona/r-layer/muse_decode_layer")
import reference as R
c = dict(batch=32, ctx=80000, layer="sliding", page=1, q_len=8, dist="soft", spread="agentic")
W = R.generate_weights(R.WEIGHTS_SEED)
cache = {}
def hs(seed):
    d = R.generate_input(**c, seed=seed, local_start=0)
    h = R.ref_front(d, W)["h"]
    del d; getattr(R.generate_input, "cache_clear", lambda: None)(); torch.cuda.empty_cache()
    return h
H = [hs(s) for s in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6)]
for rs in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
    rt = R._router(torch.Generator(device="cuda").manual_seed(1000 + rs), "cuda", W["pre_ffn_norm"])
    row = []
    for h in H:
        ids = R.route(h, rt)[1].topk(8, dim=-1).indices
        pairs = torch.bincount((ids // 32).flatten(), minlength=8)
        ls = 32 * int(pairs.argmax())
        loc = ids[(ids >= ls) & (ids < ls + 32)] - ls
        cnt = sorted(torch.bincount(loc, minlength=32).tolist(), reverse=True)
        nh = sum(65 <= x <= 128 for x in cnt); big = sum(x > 128 for x in cnt)
        row.append(f"nh{nh}{'+'+str(big) if big else ''} r{sum(cnt)} d{sum(x>0 for x in cnt)} top{cnt[:4]}")
    print("router", rs, " | ".join(row), flush=True)
