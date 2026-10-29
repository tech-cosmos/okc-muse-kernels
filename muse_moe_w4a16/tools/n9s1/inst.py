"""n9s1: clock64 counters in moe_twm consumers (lane 0 of each consumer warp, summed over CTAs).
usage: python inst.py in.py out.py"""
import sys
s = open(sys.argv[1]).read()
def rep(a, b, cnt=1, last=False):
    global s
    n = s.count(a)
    assert n >= 1 and (last or n == cnt), (a[:60], n)
    if last:
        i = s.rindex(a); s = s[:i] + b + s[i + len(a):]
    else:
        s = s.replace(a, b)
rep("static int g_hot = 0;", "static int g_hot = 0;\n__device__ unsigned long long g_n9[16];\n"
    "int64_t n9_ptr() { void* p = nullptr; cudaGetSymbolAddress(&p, g_n9); return (int64_t)p; }")
rep('_moe__CPP = _moe__CPP + "int64_t moe_set_idf(int64_t);"',
    '_moe__CPP = _moe__CPP + "int64_t moe_set_idf(int64_t);"\n_moe__CPP = _moe__CPP + "int64_t n9_ptr();"')
rep('"moe_set_hot", "moe_hot_launch", "moe_set_idf"],', '"moe_set_hot", "moe_hot_launch", "moe_set_idf", "n9_ptr"],', last=True)
# tw_item: ring waits + main-loop time
a = """  uint32_t afA[KG][RT][4], afB[KG][RT][4];
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);
    if (kc) mbar_wait(&full[st], ph);"""
assert s.count(a) == 1
s = s.replace(a, """  uint32_t afA[KG][RT][4], afB[KG][RT][4];
  int pst = -1;
  long long n9t0 = clock64(), n9w = 0;
#pragma unroll 1
  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);
    if (kc) { long long a_ = clock64(); mbar_wait(&full[st], ph); n9w += clock64() - a_; }""")
a = """    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
#if TWM_ORC & 4"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, """    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
  if (lane == 0) { atomicAdd(g_n9 + 0, (unsigned long long)n9w); atomicAdd(g_n9 + 1, (unsigned long long)(clock64() - n9t0)); atomicAdd(g_n9 + 3, 1ull); }
#if TWM_ORC & 4""")
# moe_twm consumer loop (the only one with "int nrel = 0;\n  for (;;) {\n    mbar_wait(&full[st], ph);")
a = """  int nrel = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, """  int nrel = 0;
  const long long n9L = clock64();
  for (;;) {
    { long long a_ = clock64(); mbar_wait(&full[st], ph); if (lane == 0) atomicAdd(g_n9 + 4, (unsigned long long)(clock64() - a_)); }
    const int item = s_item[st];""")
a = """    } else if (!big) {
      switch ((nv + 7) >> 3) {
        case 1: TWI(4, 8); break;"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, """    } else if (!big) {
      const long long n9i = clock64();
      switch ((nv + 7) >> 3) {
        case 1: TWI(4, 8); break;""")
a = """        case 3: TWI(4, 24); break;
        default: TWI(4, 32); break;
      }
    } else {
      switch ((nv + 7) >> 3) {
        case 1: TWI(2, 8); break;
        case 2: TWI(2, 16); break;
        case 3: case 4: TWI(2, 32); break;
        default: TWI(2, 64); break;
      }
    }
#undef TWI"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, """        case 3: TWI(4, 24); break;
        default: TWI(4, 32); break;
      }
      if (lane == 0) atomicAdd(g_n9 + 6, (unsigned long long)(clock64() - n9i));
    } else {
      const long long n9i = clock64();
      switch ((nv + 7) >> 3) {
        case 1: TWI(2, 8); break;
        case 2: TWI(2, 16); break;
        case 3: case 4: TWI(2, 32); break;
        default: TWI(2, 64); break;
      }
      if (lane == 0) { atomicAdd(g_n9 + 6, (unsigned long long)(clock64() - n9i)); atomicAdd(g_n9 + 7, 1ull); }
    }
#undef TWI""")
a = """#endif
  }
  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");"""
n = s.count(a); print("loop-end matches", n)
i = s.index("const long long n9L = clock64();")
j = s.index(a, i)
s = s[:j] + """#endif
  }
  if (lane == 0) atomicAdd(g_n9 + 5, (unsigned long long)(clock64() - n9L));
  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");""" + s[j + len(a):]
open(sys.argv[2], "w").write(s)
