# n7s14: twm epilogue-store oracle (timing only, WRONG results): tw_epi2's global stores + gemm13's fence.proxy.async removed
# (LDS kept alive by a magic-value test). usage (muse_moe_w4a16/): python tools/n7s14/mk_orx.py submission.py /tmp/x/orx.py
# H200: mkdir /tmp/x/ov; cp submission.py /tmp/x/ov/base.py; cp /tmp/x/orx.py /tmp/x/ov/
#       python tools/h200.py muse_moe_w4a16 x=/tmp/x/ov --cmd "python tools/n6s3/ab2.py base.py orx.py 2"
# ALWAYS check the variant's layout/serialization first: tools/n6s4/sass.py + tools/n5s17/loop64k.py, and count
# 'DEPBAR.LE gsb0, 0x0' (an oracle that drops the epilogue entirely made ptxas serialize every wgmma: 96 vs 8, +23..36 us).
import sys
s = open(sys.argv[1]).read()
a = '''          if (CE == 4) *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(stg + tok * 64 + p);
          else *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(stg + tok * 64 + p);'''
assert s.count(a) == 1
s = s.replace(a, '''          if (EPI_ORX) { if (*reinterpret_cast<const uint32_t*>(stg + tok * 64 + p) == 0x7fc17fc1u) *dst = stg[0]; } else
''' + a)
b = '''        if (gi * 8 + tok < nv)
          *reinterpret_cast<uint4*>(Out + (size_t)sroutes[off + gi * 8 + tok] * NOUT + col0 + k * 8) ='''
assert s.count(b) == 1
s = s.replace(b, '''        if (EPI_ORX) { if (gi * 8 + tok < nv && *reinterpret_cast<const uint32_t*>(stg + tok * 64 + k * 8) == 0x7fc17fc1u) Out[col0] = stg[0]; } else
''' + b)
c = '''      asm volatile("fence.proxy.async.global;" ::: "memory");
#if TWM_REL  // n6s3/n6s4 REL'''
assert s.count(c) == 1
s = s.replace(c, '''#if !EPI_ORX
      asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
#if TWM_REL  // n6s3/n6s4 REL''')
s = s.replace('#define TWM_EPI 0 ', '#define EPI_ORX 1\n#define TWM_EPI 0 ', 1)
open(sys.argv[2], 'w').write(s)
