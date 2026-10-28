import re, zlib, base64, numpy as np, ast
src = open('/home/daytona/r-moe/muse_moe_w4a16/submission.py').read()
hdr = ast.literal_eval(re.search(r'^_DIAG_HDR = (.*)$', src, re.M).group(1))
R = re.search(r'^_DIAG_R = "(.*)"$', src, re.M).group(1)
raw = zlib.decompress(base64.b64decode(R)); p = 0
PE = 3*4096*4096*(0.5+1/16)
for load, seed, T in hdr:
    loc = np.frombuffer(raw[p:p+T*8], dtype=np.int8).reshape(T, 8); p += T*8
    if T < 128: continue
    c = np.bincount(loc[loc >= 0].astype(int), minlength=32)
    cs = sorted(c.tolist(), reverse=True); nz = int((c > 0).sum())
    # twm passes: >32 tok -> 64-token passes; hot class: moe_hot 65..128 one pass (h32 variant)
    p64 = sum(-(-x//64) for x in cs if x > 32) + sum(1 for x in cs if 0 < x <= 32)
    phot = sum((1 if x <= 128 else -(-x//64)) for x in cs if x > 64) + sum(1 for x in cs if 0 < x <= 64)
    print(load, seed, T, "routes", c.sum(), "distinct", nz, "uniqMB %.0f" % (nz*PE/1e6), "t@4.0 %.0f" % (nz*PE/4e6),
          "passes64", p64, "passes_hot", phot, "rereadMB64 %.0f" % ((p64-nz)*PE/1e6), cs[:6])
