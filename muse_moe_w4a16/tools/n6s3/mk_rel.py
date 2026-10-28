# n6s3 REL: moe_twm's gemm13 release (threadfence + atomicAdd done[e]) moves from consumer thread 0 (after a 256-thread
# bar.sync) to a helper warp (warp 9, idle until now): consumers fence.proxy.async + arrive on an mbarrier and go on.
# usage: python tools/n6s3/mk_rel.py <in.py> <out.py>   (applies to main or to the TL build)
import sys
s = open(sys.argv[1]).read()
a = s.find('moe_twm(\n'); assert a > 0
b = s.find('  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");\n}', a); assert b > 0
t = s[a:b]
def rep(x, y):
    global t
    assert t.count(x) == 1, x
    t = t.replace(x, y)
rep('  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];\n',
    '  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG], relb[2];\n  __shared__ int s_rel[2];\n')
rep('mbar_init(&empty[i], 8 * 32); }\n', 'mbar_init(&empty[i], 8 * 32); }\n    mbar_init(&relb[0], 256); mbar_init(&relb[1], 256);\n')
rep('    if (warp > 8 || lane != 0) return;\n', '''    if (warp == 9) {  // n6s3 REL helper: release finished gemm13 items (2 slots; items are >> the release latency apart)
      if (lane == 0)
        for (int k = 0;; ++k) {
          mbar_wait(&relb[k & 1], (k >> 1) & 1);
          const int e = s_rel[k & 1];
          if (e < 0) break;
          __threadfence();
          atomicAdd(done + e, 1);
        }
      return;
    }
    if (warp > 8 || lane != 0) return;
''')
rep('  for (;;) {\n    mbar_wait(&full[st], ph);\n', '  int nrel = 0;\n  for (;;) {\n    mbar_wait(&full[st], ph);\n')
rep('    if (item < 0) break;\n', '    if (item < 0) {\n      if (tid == 0) s_rel[nrel & 1] = -1;\n      mbar_arrive(&relb[nrel & 1]);\n      break;\n    }\n')
rep('''      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
''', '''      asm volatile("fence.proxy.async.global;" ::: "memory");
      if (tid == 0) s_rel[nrel & 1] = e;
      mbar_arrive(&relb[nrel & 1]);
      ++nrel;
''')
s = s[:a] + t + s[b:]
open(sys.argv[2], 'w').write(s)
print('ok')
