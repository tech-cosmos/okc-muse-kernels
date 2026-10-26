# python dev/t19lib.py: g19_lib round trip (pack19 from _compress, FP32 out, cvt)
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj/dev"); sys.path.insert(0, "/home/daytona/r-proj/muse_bf16_proj")
import g19_lib, reference, importlib.util
spec = importlib.util.spec_from_file_location("st", "/home/daytona/r-proj/muse_bf16_proj/dev/sub_template.py")
src = open(spec.origin).read()
ns = {}
exec(src[:src.index("def _nib")].split("def _compress")[0].split("\n# ")[0] if False else "", ns)
# pull G13_ESC and _compress only
g = {"torch": torch}
import re
esc = re.search(r"^G13_ESC\s*=\s*(\d+)", src, re.M).group(1); g["G13_ESC"] = int(esc)
exec(src[src.index("def _compress"):src.index("def _nib")], g)
W = reference.generate_weights(reference.WEIGHTS_SEED)
lib = g19_lib.build([(16, 1024, 4, 4), (32, 512, 4, 4)])
ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
for name in ["qkv_gate", "pooler", "unpooler"]:
    w = W[name]; c = g["_compress"](w); assert c is not None
    for m, kc, ap, nsg in [(16, 1024, 4, 4), (32, 512, 4, 4)]:
        P = g19_lib.pack19(c, kc, ap)
        x = torch.randn(m, w.shape[1], device="cuda").bfloat16()
        out = torch.full((m, w.shape[0]), 3.0, device="cuda"); ob = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device="cuda")
        lib.g19_gemm(P, c[2], c[3], x, out, ctr, kc, ap, nsg, 1, 132, 0, 2, 0); lib.g19_cvt(out, ob); torch.cuda.synchronize()
        ref = x.double() @ w.double().T
        print(name, m, "rel err fp32", ((out.double() - ref).norm() / ref.norm()).item(), "bf16 == round(fp32)", torch.equal(ob, out.bfloat16()),
              "vs F.linear", ((ob.double() - F.linear(x, w).double()).norm() / ref.norm()).item())
print("ctr", ctr.tolist())
