# n6s3: TWM_TL build with a 4th stamp per item: tw_item's loop end (after wg_wait<0>), so item = [first stage arrived ->
# loop end] + [epilogue + release]. 15 items per CTA, 4 slots each (start, end, desc, loopend). -> /tmp/n6s3/sub_tl.py
s = open('submission.py').read()
def rep(a, b, n=1):
    global s
    assert s.count(a) >= n, a
    s = s.replace(a, b, n)
rep('#define TWM_TL 0\n', '#define TWM_TL 1\n')
# stamp in tw_item (first wg_wait<0>() is tw_item's)
i = s.find('__device__ __forceinline__ void tw_item(')
j = s.find('  wg_wait<0>();\n', i)
s = s[:j] + '  wg_wait<0>();\n  if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_le[blockIdx.x] = t; }\n' + s[j + len('  wg_wait<0>();\n'):]
# global decl before the tw_item template line
k = s.rfind('template', 0, i)
s = s[:k] + '__device__ unsigned long long g_le[1024];\n' + s[k:]
# twm loop (first occurrence = moe_twm)
a = s.find('      else if (ntl < 20) { tl[2 + 3 * ntl] = gtimer(); tl[4 + 3 * ntl] = item; }')
s = s[:a] + s[a:].replace('      else if (ntl < 20) { tl[2 + 3 * ntl] = gtimer(); tl[4 + 3 * ntl] = item; }',
                          '      else if (ntl < 15) { tl[2 + 4 * ntl] = gtimer(); tl[4 + 4 * ntl] = item; }', 1)
s = s.replace('    if (tl && tid == 0 && ntl < 20) tl[4 + 3 * ntl] = g2 | (cl << 1)', '    if (tl && tid == 0 && ntl < 15) tl[4 + 4 * ntl] = g2 | (cl << 1)', 1)
a = s.find('    if (tl && tid == 0 && ntl < 20) tl[3 + 3 * ntl] = gtimer();')
s = s[:a] + '    if (tl && tid == 0 && ntl < 15) { tl[3 + 4 * ntl] = gtimer(); tl[5 + 4 * ntl] = g_le[blockIdx.x]; }' + s[a + len('    if (tl && tid == 0 && ntl < 20) tl[3 + 3 * ntl] = gtimer();'):]
# n6s3b: 2nd stamp after tw_epi in tw_item's DYN branch -> g_le2; stored at tl[63 - 0]? no: pack (le2 - le) in ns into desc bits 32+
i = s.find('__device__ __forceinline__ void tw_item(')
j = s.find('    if (g2) tw_epi<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);\n    else tw_epi<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);\n', i)
assert j > 0
L = len('    if (g2) tw_epi<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);\n    else tw_epi<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);\n')
s = s[:j + L] + '    if (threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); g_le[512 + blockIdx.x] = t; }\n' + s[j + L:]
s = s.replace('tl[5 + 4 * ntl] = g_le[blockIdx.x]; }', 'tl[5 + 4 * ntl] = g_le[blockIdx.x]; tl[4 + 4 * ntl] |= (g_le[512 + blockIdx.x] - g_le[blockIdx.x]) << 32; }', 1)
open('/tmp/n6s3/sub_tl.py', 'w').write(s)
print("ok")
