s = open('/home/daytona/r-moe/muse_moe_w4a16/submission.py').read()
s = s.replace('grid = nsm;', 'grid = getenv("MOE_GRID") ? atoi(getenv("MOE_GRID")) : nsm;')
def rep(a, b, start=0):
    global s
    i = s.find(a, start); assert i >= 0, a
    s = s[:i] + b + s[i + len(a):]
rep('template <typename... KA, typename... AA>\nstatic int launch_k(', '''__device__ unsigned long long g_td[1024 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TD(k) g_td[blockIdx.x * 8 + (k)] = gtime()
int64_t moe_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), g_td, 1024 * 8 * 8); return 0; }
template <typename... KA, typename... AA>
static int launch_k(''')
i = s.find('__device__ __forceinline__ void tk2f_consume(')
rep('''  bool pre_ok = false;''', '''  bool pre_ok = false;
  unsigned long long tspin = 0;
  if (tid == 0) TD(GATED ? 0 : 2);''', i)
rep('''        if (!pre_ok) {
          const int* cp = cnt + 64 + seg_start / RPE;
          int c;''', '''        if (!pre_ok) {
          const unsigned long long ts0 = gtime();
          const int* cp = cnt + 64 + seg_start / RPE;
          int c;''', i)
rep('''          } while (c < 2 * RPE);
        }''', '''          } while (c < 2 * RPE);
          tspin += gtime() - ts0;
        }''', i)
rep('''        } while (c < need);
      }''', '''        } while (c < need);
      }
      if (tid == 0 && !GATED && R == r0) TD(4);''', i)
# end of consume: find the closing of the R loop: insert before the final "}\n}\n\n// s10 GS16"
rep('''  }
}

// s10 GS16''', '''  }
  if (tid == 0) { TD(GATED ? 1 : 3); g_td[blockIdx.x * 8 + 5 + !GATED] = tspin; }
}

// s10 GS16''', i)
rep('"int64_t moe_set_pdl(int64_t, int64_t);"', '"int64_t moe_set_pdl(int64_t, int64_t);" "int64_t moe_tdbg(torch::Tensor);"')
rep('functions=["moe_gemm13",', 'functions=["moe_tdbg", "moe_gemm13",')
open('/tmp/moe7/sub_i.py', 'w').write(s)
