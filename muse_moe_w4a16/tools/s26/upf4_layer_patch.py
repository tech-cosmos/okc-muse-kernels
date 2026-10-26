# s26 UPF4: moe_twm's producer L2-prefetches (evict_last) the first SEG bytes of every unpool row, spread over the
# items handed out between F0% and F1% of the work counter. Usage: patch.py <layer dir>
import sys, pathlib
d = pathlib.Path(sys.argv[1]); p = d / "layer.py"; s = p.read_text()
anchor = '_moe__CPP = _moe__CPP + "int64_t moe_set_upf(int64_t, int64_t, int64_t, int64_t);"\n'
assert s.count(anchor) == 1
code = r'''# moe s26 UPF4: at T in UPF4 = {T: (seg bytes per unpool row, f0 %, f1 %)}, moe_twm's producer lanes L2-prefetch
# (evict_last) the first seg bytes of every row of the BF16 unpooler, rows spread over the items numbered
# f0..f1 % of nitems (twm at T >= 128 is dq/MMA-bound: HBM has spare bandwidth during its main phase).
_i = _moe_CUDA_SRC.index("__global__ void __launch_bounds__(384, 1) moe_twm(")
_a = "    __nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit) {\n"
_j = _moe_CUDA_SRC.index(_a, _i)
assert _j - _i < 1000
_moe_CUDA_SRC = (_moe_CUDA_SRC[:_j] + _a.replace("int nsplit) {", "int nsplit,\n    const char* upf, int upf_rows, int upf_rb, int upf_seg) {")
                 + _moe_CUDA_SRC[_j + len(_a):])
_a = "      int g2, a, p, rb, h, j;\n      decode(item, g2, a, p, rb, h, j);\n"
_j = _moe_CUDA_SRC.index(_a, _i)
_moe_CUDA_SRC = (_moe_CUDA_SRC[:_j] + """      if (upf != nullptr) {
        const int i0 = nitems * ((upf_seg >> 16) & 255) / 100, nq = nitems * (upf_seg >> 24) / 100 - i0, q = item - i0;
        if (q >= 0 && q < nq) {
          const int r1 = (int)((long long)upf_rows * (q + 1) / nq), sg = upf_seg & 65535;
          uint64_t pl;
          asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pl));
          for (int r = (int)((long long)upf_rows * q / nq); r < r1; ++r)
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(upf + (long long)r * upf_rb), "r"(sg), "l"(pl) : "memory");
        }
      }
""" + _moe_CUDA_SRC[_j:])
_i = _moe_CUDA_SRC.index("static int64_t twm_go(")
_a = "      (float*)part_.data_ptr(), (int)std::min<int64_t>(ns, std::min<int64_t>(tk_.numel() / 2, part_.numel() / 32768)));\n}\n"
_j = _moe_CUDA_SRC.index(_a, _i)
assert _j - _i < 3000
_moe_CUDA_SRC = (_moe_CUDA_SRC[:_j] + _a.replace(")));\n}", ")),\n      g_upf4, g_upf4_rows, g_upf4_rb, g_upf4_seg);\n}") + _moe_CUDA_SRC[_j + len(_a):])
_a = "\nstatic const char* g_upf = nullptr;"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + "\nstatic const char* g_upf4 = nullptr;\nstatic int g_upf4_rows = 0, g_upf4_rb = 0, g_upf4_seg = 0;\n"
                                      "int64_t moe_set_upf4(int64_t p, int64_t rows, int64_t rb, int64_t seg) {\n"
                                      "  g_upf4 = (const char*)p; g_upf4_rows = (int)rows; g_upf4_rb = (int)rb; g_upf4_seg = (int)seg; return 0;\n}")
assert _moe_CUDA_SRC.index("int64_t moe_set_upf4(") < _moe_CUDA_SRC.index("static int64_t twm_go(")
_moe__CPP = _moe__CPP + "int64_t moe_set_upf4(int64_t, int64_t, int64_t, int64_t);"
UPF4 = {}
'''
s = s.replace(anchor, anchor + code)
a = '"moe_set_upf", "moe_set_ss",'
assert s.count(a) == 1
s = s.replace(a, '"moe_set_upf", "moe_set_upf4", "moe_set_ss",')
a = "        upf = UPF3.get(m) if m <= TK2_MAX_TOKENS else None\n"
assert s.count(a) == 1
s = s.replace(a, a + """        upf4 = _opt(state, m, "upf4", UPF4.get(m)) if m > TK2_MAX_TOKENS else None
        if upf4:
            u = state["unpool"]
            lib.moe_set_upf4(u.data_ptr(), u.shape[0], u.shape[1] * u.element_size(), upf4[0] | (upf4[1] << 16) | (upf4[2] << 24))
""")
a = "            lib.moe_set_upf(0, 0, 0, 0)\n"
assert s.count(a) == 1
i = s.index(a); ind = s[s.rindex("\n", 0, i) + 1:i]
s = s.replace(a, a + "        if upf4:\n            lib.moe_set_upf4(0, 0, 0, 0)\n")
p.write_text(s)
print("ok")
