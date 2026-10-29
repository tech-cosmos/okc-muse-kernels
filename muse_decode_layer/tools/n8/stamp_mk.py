"""n8s1: add per-CTA globaltimer stamps to the nvs kernel (v16 path) of a variant file: python tools/n8/stamp_mk.py in out.
nv::g_st[cta*16 + k]: 0 entry, 1 producer pt staged, 2..9 tile i ready, 10 consumer0 Q done, 11 consumer0 end, 12 consumer0 first tile ready seen."""
import sys
s = open(sys.argv[1]).read()
a = s.index("_NVS_SRC = r'''"); b = s.index("'''", a + 20)
k = s[a:b]
def rep(o, n):
    global k
    assert o in k, o[:50]; k = k.replace(o, n, 1)
rep("namespace nv {\n", "namespace nv {\n__device__ unsigned long long g_st[512 * 16];\n__device__ __forceinline__ void st(int i) { unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); g_st[blockIdx.x * 16 + i] = t; }\n")
rep("    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n    const int tid = threadIdx.x, wg = tid >> 7;",
    "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n    if (threadIdx.x == 0) st(0);\n    const int tid = threadIdx.x, wg = tid >> 7;")
rep("    cp_wait<0>();\n    asm volatile(\"bar.sync 1, 128;\" ::: \"memory\");\n", "    cp_wait<0>();\n    asm volatile(\"bar.sync 1, 128;\" ::: \"memory\");\n    if (tid == 0) st(1);\n")
rep("            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);", "            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);\n            if (tid == 0 && i - LAG < 8) st(2 + i - LAG);")
rep("        asm volatile(\"bar.sync %0, 128;\" ::\"r\"(2 + h) : \"memory\");\n    }\n    float o[32], s[32];",
    "        asm volatile(\"bar.sync %0, 128;\" ::\"r\"(2 + h) : \"memory\");\n    }\n    if (tid == 0 && h == 0) st(10);\n    float o[32], s[32];")
rep("    const int ra = w * 16 + (lane >> 2);\n    float* pa0", "    if (tid == 0) st(11);\n    const int ra = w * 16 + (lane >> 2);\n    float* pa0")
rep("if (n > 0) {\n        mb_wait(&sm.ready[0], 0);\n", "if (n > 0) {\n        mb_wait(&sm.ready[0], 0);\n        if (tid == 0 && h == 0) st(12);\n")
k += "\nvoid nvs_dbg(torch::Tensor o) { cudaMemcpyFromSymbol(o.data_ptr(), nv::g_st, 512 * 16 * 8); }\n"
s = s[:a] + k + s[b:]
o1 = 'cuda_sources=_NVS_SRC, functions=["nvs"],'
assert s.count(o1) == 1; s = s.replace(o1, 'cuda_sources=_NVS_SRC, functions=["nvs", "nvs_dbg"],')
c = s.index('cuda_sources=_NVS_SRC'); d = s.rindex('cpp_sources="', 0, c); e = s.index('",', s.index(';', d))
s = s[:e] + ' void nvs_dbg(torch::Tensor);' + s[e:]
open(sys.argv[2], "w").write(s)
