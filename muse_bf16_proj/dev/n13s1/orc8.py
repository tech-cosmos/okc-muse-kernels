"""proj n13s1: pr8 (v1 pooler+router) flat oracle: sub_rt8.py = the layer submission with pr8 doing a grid-stride read of
the same bytes (12-bit WS + WN planes + FP32 router; all 129 blocks, 4 x 16 B in flight per thread, griddepcontrol.wait
after the first batch, no outputs) when env PR8_FLAT is set at launch (= at graph capture). python orc8.py <sub> <out.py>"""
import sys, pathlib
full = pathlib.Path(sys.argv[1]).read_text()
i0 = full.index("_GPR8_SRC = r'''"); i1 = full.index("'''", i0 + 20); src = full[i0:i1]
a = """// s36 timing probe (from gemv19)"""
assert src.count(a) == 1
flat = """// n13s1 oracle only: flat grid-stride read of pr8's bytes (WS, WN, router), PDL like pr8 (trigger at start, wait after the 1st batch)
__global__ void __launch_bounds__(1024, 1) flat8(const uint4* ws, long ns, const uint4* wn, long nn, const uint4* rw, long nr, float* sink) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const long tot = ns + nn + nr, stride = (long)gridDim.x * blockDim.x;
    unsigned acc = 0; bool waited = false;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { const long q = i + u * stride; v[u] = q >= tot ? make_uint4(0, 0, 0, 0) : __ldcs(q < ns ? ws + q : q < ns + nn ? wn + (q - ns) : rw + (q - ns - nn)); }
        if (!waited) { asm volatile("griddepcontrol.wait;" ::: "memory"); waited = true; }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (!waited) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (acc == 0x9e3779b9u) sink[0] = 1.f;
}
""" + a
b = """    PR8_CONFIGS
"""
assert src.count(b) == 1
b2 = """    if (getenv("PR8_FLAT")) {  // n13s1 oracle only
        lc.gridDim = bk + 1; lc.blockDim = 1024; lc.dynamicSmemBytes = 0;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, pr8::flat8, (const uint4*)ws_.data_ptr(), (long)(ws_.numel() * ws_.element_size() / 16),
                                          (const uint4*)wn.data_ptr(), (long)(wn.numel() * wn.element_size() / 16),
                                          (const uint4*)rw.data_ptr(), (long)(rw.numel() * 4 / 16), logits.data_ptr<float>()));
        return;
    }
""" + b
src2 = src.replace(a, flat).replace(b, b2)
pathlib.Path(sys.argv[2]).write_text(full[:i0] + src2 + full[i1:])
print("ok")
