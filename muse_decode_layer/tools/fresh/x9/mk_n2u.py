import sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
a = s.index("_N2U_SRC = r'''"); b = s.index("'''", a + 20)
k = s[a:b]
def kr(o, n):
    global k
    assert k.count(o) == 1, (k.count(o), o[:60]); k = k.replace(o, n)
kr("namespace n2u {\n", "namespace n2u {\n__device__ unsigned long long g_nt[16384];\n#define NT(j, dep) do { unsigned long long t_; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t_) : \"f\"(dep)); if (tid == 0) g_nt[blockIdx.x * 8 + (j)] = t_; } while (0)\n")
kr('    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n    // s14:', '    NT(0, 0.f);\n    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n    // s14:')
kr('    asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (blockIdx.x == 0 && tid < 128) zc[tid] = 0;\n',
   '    asm volatile("griddepcontrol.wait;" ::: "memory");\n    NT(1, 0.f);\n    if (blockIdx.x == 0 && tid < 128) zc[tid] = 0;\n')
kr("    float v[8][8], ss[8];\n", "    { float d_ = 0.f; for (int s = 0; s < 8; ++s) d_ += twv[s] + (loc[s] ? 1.f : 0.f); NT(2, d_); }\n    float v[8][8], ss[8];\n")
kr("""        if (lane == 0) ssum[warp][s] = x;
    }
    __syncthreads();
""", """        if (lane == 0) ssum[warp][s] = x;
    }
    { float d_ = 0.f; for (int s = 0; s < 8; ++s) d_ += ss[s]; NT(3, d_); }
    __syncthreads();
    NT(4, 0.f);
""")
kr("""        if (lane == 0) red[warp][r] = t;
    }
    __syncthreads();
    if (tid < R) {""", """        if (lane == 0) red[warp][r] = t;
    }
    NT(5, red[warp][0]);
    __syncthreads();
    if (tid < R) {""")
kr("""        out[r0 + tid] = __float2bfloat16(t);
    }
    if (bc == nullptr) return;""", """        out[r0 + tid] = __float2bfloat16(t);
    }
    NT(6, 0.f);
    if (bc == nullptr) return;""")
k += "\nint64_t n2u_tdbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), n2u::g_nt, 16384 * 8); return 0; }\n"
s = s[:a] + k + s[b:]
assert s.count('+ "int64_t pr1_tdbg(torch::Tensor);') == 1
s = s.replace('+ "int64_t pr1_tdbg(torch::Tensor);', '+ "int64_t pr1_tdbg(torch::Tensor); int64_t n2u_tdbg(torch::Tensor);')
assert s.count('"pr1_launch", "pr1_tdbg"') == 1, s.count('"pr1_launch", "pr1_tdbg"')
s = s.replace('"pr1_launch", "pr1_tdbg"', '"pr1_launch", "pr1_tdbg", "n2u_tdbg"')
open(out, "w").write(s)
print("ok")
