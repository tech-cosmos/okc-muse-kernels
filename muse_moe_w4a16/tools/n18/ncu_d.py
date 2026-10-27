# ncu driver: python ncu_d.py <load e.g. 'decode 16'> <seed> <variant json>
import sys, json, zlib, base64, torch
sys.path.insert(0, "/tmp/moe18")
import reference, submission as S
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = S.prepare(w)
raw = zlib.decompress(base64.b64decode(S._DIAG_R)); p = 0
for load, seed, T in S._DIAG_HDR:
    loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda(); p += T * 8
    if load == sys.argv[1] and seed == int(sys.argv[2]): break
gen = torch.Generator(device="cuda").manual_seed(seed)
ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
tw = torch.rand(T, 8, device="cuda", generator=gen)
data = {"hidden_states": torch.randn(T, S.H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
        "topk_weights": tw / tw.sum(-1, keepdim=True),
        "config": {"bs": T, "d_hidden": S.H, "local_start": st["start"], "local_count": st["count"]}}
for k, v in json.loads(sys.argv[3]).items():
    setattr(S, k, {int(a): b for a, b in v.items()} if isinstance(v, dict) else v)
S.kernel_fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
S.kernel_fn(data, st); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
