import sys
s=open('/tmp/s28/sub_nc.py').read()
A="""    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XBv)[(warp * RT + r) * 32 + lane];"""
B="""    if (kc) mbar_wait(&full[st], ph);
    uint4 wv[RT];"""
assert s.count(A)==1 and s.count(B)==1
v=sys.argv[1]
if v=='tid':
    s=s.replace("  int pst = -1;\n#pragma unroll 1\n  for (int kc = 0; kc < nkc; ++kc) {\n",
      "  int pst = -1;\n  uint32_t wl = (warp * RT) * 32 + lane, sl = (warp * 32 + lane) * RT * 2;\n#pragma unroll 1\n  for (int kc = 0; kc < nkc; ++kc) {\n    fence_r(wl); fence_r(sl);\n",1)
    s=s.replace(A,A.replace("[(warp * RT + r) * 32 + lane]","[wl + r * 32]"))
    s=s.replace("const uint2 v = *reinterpret_cast<const uint2*>(sb + XBv + C::WB + (warp * 32 + lane) * RT * 2);","const uint2 v = *reinterpret_cast<const uint2*>(sb + XBv + C::WB + sl);")
elif v=='dw':
    s=s.replace(B,"    if (kc) { mbar_wait(&full[st], ph); mbar_wait(&full[st], ph); }\n    uint4 wv[RT];")
elif v=='nw':
    s=s.replace(A,"    for (int r = 0; r < RT; ++r) { uint32_t q = lane * (r + 1) + kc; fence_r(q); wv[r] = make_uint4(q, q ^ 0x1234, q * 3, q + 77); }")
elif v=='nwait':  # skip wait (nocopy only); keep ordering via empty
    s=s.replace(B,"    uint4 wv[RT];")
assert s!=open('/tmp/s28/sub_nc.py').read()
open(f'/tmp/s28/sub_{v}.py','w').write(s)
