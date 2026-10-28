# consumer SM-time model for twm || moe_hot on the layer routings (H100 in-kernel warm item times, SCB, us)
import re, zlib, base64, numpy as np, ast
src = open('/home/daytona/r-moe/muse_moe_w4a16/submission.py').read()
hdr = ast.literal_eval(re.search(r'^_DIAG_HDR = (.*)$', src, re.M).group(1))
raw = zlib.decompress(base64.b64decode(re.search(r'^_DIAG_R = "(.*)"$', src, re.M).group(1))); p = 0
PE = 3*4096*4096*(0.5+1/16)
RT4 = {8: 34.7, 16: 38.5, 24: 40.1, 32: 46.0}   # 24 items/expert (1.13 MB)
def cost(n, hot):
    if n <= 32: return 24 * RT4[min(c for c in RT4 if c >= n)], 0
    if n <= 64 or (not hot and False): return 48 * 35.0, 0
    if hot and n <= 128: return 0, 48 * 51.5
    return 48 * 35.0 * -(-n // 64), 0
for load, seed, T in hdr:
    loc = np.frombuffer(raw[p:p+T*8], dtype=np.int8).reshape(T, 8); p += T*8
    if T < 128: continue
    c = np.bincount(loc[loc >= 0].astype(int), minlength=32); nz = int((c > 0).sum())
    for hot in ((False,) if T == 128 else (False, True)):
        tw = sum(cost(n, hot)[0] for n in c if n); ht = sum(cost(n, hot)[1] for n in c if n)
        fl = sum(2*n*3*4096*4096 for n in c) / 1e9
        print(f"{load} s{seed:<3} hot{int(hot)} uniq {nz*PE/1e6:4.0f} MB bytes@4.2 {nz*PE/4.2e6:5.1f} us | cons SM-us twm {tw:6.0f} hot {ht:5.0f}"
              f" -> /132 {(tw+ht)/132:5.1f} us | twm on {132-32*hot} SMs {tw/(132-32*hot):5.1f}, hot on 32 {ht/32:5.1f} | {fl:.1f} GFLOP")
