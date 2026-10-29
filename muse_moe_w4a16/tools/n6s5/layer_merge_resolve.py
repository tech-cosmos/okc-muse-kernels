import re
s = open('merged.py').read()
blocks = list(re.finditer(r'<<<<<<< layer_moe.py\n(.*?)=======\n(.*?)>>>>>>> moe_new.py\n', s, re.S))
assert len(blocks) == 8, len(blocks)
res = []
L = [b.group(1) for b in blocks]; M = [b.group(2) for b in blocks]
# 1 defines
res.append(L[0] + "#define TWM_EPI 1        // n6s3 (moe a60019d): staged, rolled tw_item / tw_epi epilogues (i-cache); bitwise\n"
 "#define TWM_HOTC 0       // n6s3 (moe 0be4fa0): moe_twm's in-twm hot class compiled out (layer: twm_hot is in no TUNE list); set 1 to use twm_hot\n"
 "#define TWM_REL 1        // n6s4 (moe 7d2a079): moe_twm gemm13+gemm2 releases by helper warp 9\n"
 "#define TWM_KSC 2        // n6s4 (moe 7d2a079): moe_twm built with + without the K-split tail code (picked per call); 0 = never split, 1 = always built in\n")
# 2 template
res.append(M[1])
# 3 TWI macro: moe's minus the TWM_TL block
t = M[2]; i = t.index('#define TWI'); res.append(t[i:])
# 4 SMEM: layer's + staging
x = L[3]; res.append(x.replace(' + 1024;', ' + 1024 + (TWM_EPI ? 8192 : 0);'))
# 5 attr
res.append('#if TWM_KSC == 2\n    cudaFuncSetAttribute(moe_twm<NSTG, SA, 64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);\n#endif\n')
# 6 launch
res.append('  auto kf = moe_twm<NSTG, SA>;\n#if TWM_KSC == 2  // n6s4: both builds; the no-K-split one when this call has no split (ns > 0 or bit 22 of -ns)\n'
 '  if (!(ns > 0 || (ns < 0 && (((-ns) >> 22) & 1)))) kf = moe_twm<NSTG, SA, 64, false>;\n#endif\n'
 + L[5].replace('launch_k(moe_twm<NSTG, SA>,', 'launch_k(kf,'))
# 7,8 python: layer's
res.append(L[6]); res.append(L[7])
out = []; p = 0
for b, r in zip(blocks, res):
    out.append(s[p:b.start()]); out.append(r); p = b.end()
out.append(s[p:]); open('moe_src_new.py', 'w').write(''.join(out))
