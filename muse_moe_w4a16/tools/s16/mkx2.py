import sys
n = int(sys.argv[1])
s = open('/tmp/moe16/sub_tlg.py').read()
s = s.replace('CUDA_SRC = r"""\n', f'CUDA_SRC = r"""\n#define TWX {n}\n', 1)
a = "    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + C::XB)[(warp * RT + r) * 32 + lane];"
assert s.count(a) == 1
s = s.replace(a, "    for (int r = 0; r < RT; ++r) wv[r] = (TWX & 4) ? make_uint4(kc, r, lane, warp) : reinterpret_cast<const uint4*>(sb + C::XB)[(warp * RT + r) * 32 + lane];")
b = """          if (odd) dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afB[jj][r]);
          else dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afA[jj][r]);"""
assert s.count(b) == 1, s.count(b)
s = s.replace(b, """#if TWX & 1
          { uint32_t q = u4get(wv[r], j0 + jj); uint32_t* af = odd ? afB[jj][r] : afA[jj][r];
            af[0] = q ^ s0[r]; af[1] = q + s1[r]; af[2] = q; af[3] = q ^ 0x1234; }
#else
""" + b + "\n#endif")
c = "        for (int r = 0; r < RT; ++r) wgmma_bf16<N>(acc[r], odd ? afB[jj][r] : afA[jj][r], d0 + (j0 + jj) * 2);"
assert s.count(c) == 1
s = s.replace(c, """        for (int r = 0; r < RT; ++r) {
#if TWX & 2
          const uint32_t* af = odd ? afB[jj][r] : afA[jj][r]; acc[r][0] += __uint_as_float(af[0] ^ af[1] ^ af[2] ^ af[3]);
#else
          wgmma_bf16<N>(acc[r], odd ? afB[jj][r] : afA[jj][r], d0 + (j0 + jj) * 2);
#endif
        }""")
d = """        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);"""
assert s.count(d) == 1
s = s.replace(d, """#if TWX & 8
        mbar_expect_tx(&full[st], bytes - (1024 << mi));
#else
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
#endif""")
for f in ("      wg_fence();\n", "      wg_commit();\n      wg_wait<1>();\n"):
    assert s.count(f) >= 1, f
s = s.replace("      wg_fence();\n#pragma unroll\n      for (int jj = 0; jj < KG; ++jj)", "      if (!(TWX & 32)) wg_fence();\n#pragma unroll\n      for (int jj = 0; jj < KG; ++jj)")
s = s.replace("      wg_commit();\n      wg_wait<1>();\n", "      if (!(TWX & 32)) { wg_commit();\n      wg_wait<1>(); }\n")
open(f'/tmp/moe16/sub_y{n}.py', 'w').write(s)
