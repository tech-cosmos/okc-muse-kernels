"""moe o42: per-CTA %globaltimer stamps of the layer's T=8 expert chain (timing copy only; never in a submission).
patch(g, addr) rewrites the layer module's CUDA source strings (after all of its own runtime .replace()s) so that
pr8 / its routing finisher / moe_tk2f / nu8 / bnd2 write u64 stamps to buf[(region * NC + cta) * NK + slot].
Regions: 0 pr8 CTAs, 1 pr8 finisher (routing), 2 tk2f, 3 nu8, 4 bnd2. See SLOTS for the slot names."""
import os
NC, NK = 1024, 16
SLOTS = {
    0: {0: "entry", 1: "past wait", 2: "end"},
    1: {0: "entry", 1: "past wait", 2: "all partials", 3: "routed"},
    2: {0: "entry", 1: "routed", 2: "past X wait", 3: "g13 1st full", 4: "g13 end", 6: "g2 cnt ok", 7: "g2 1st full",
        8: "end", 9: "prod start", 10: "prod last TMA"},
    3: {0: "entry", 1: "past wait", 2: "Y loaded", 4: "scaled+ssq", 7: "w0 mma end", 8: "w15 mma end", 9: "w0 issued", 10: "w0 ks0", 11: "w0 ks1", 12: "w0 ks2", 13: "w0 ks5", 5: "mma done", 6: "red stored", 3: "end"},
    4: {0: "entry", 1: "past wait", 2: "end"},
}
NAMES = {0: "pr8", 1: "pr8 fin", 2: "tk2f", 3: "nu8", 4: "bnd2"}


def hdr(addr):
    return ('\n#ifndef STP_ADDR\n#define STP_ADDR %dULL\n'
            '__device__ __forceinline__ unsigned long long stp_gt() { unsigned long long t; asm volatile("mov.u64 %%0, %%%%globaltimer;" : "=l"(t)); return t; }\n'
            '#define TSR(r, k) (reinterpret_cast<unsigned long long*>(STP_ADDR))[((r) * %d + blockIdx.x + gridDim.x * blockIdx.y) * %d + (k)] = stp_gt()\n'
            '#endif\n') % (addr, NC, NK)


def ins(s, at, h):
    """insert h before the line holding index at, or before its template <...> line"""
    ls = s.rfind("\n", 0, at) + 1
    pl = s.rfind("\n", 0, ls - 1) + 1
    if s[pl:ls].startswith("template <"):
        ls = pl
    return s[:ls] + h + s[ls:]


class P:
    def __init__(self, s):
        self.s = s

    def rep(self, a, b, start=0, end=None, after=None):
        if after is not None:
            start = self.s.index(after, start)
        i = self.s.find(a, start, end if end is not None else len(self.s))
        assert i >= 0, a[:90]
        self.s = self.s[:i] + b + self.s[i + len(a):]
        return i


def patch(g, addr):
    H = hdr(addr)
    # ---- moe (tk2f) ----
    m = P(g["_moe_CUDA_SRC"])
    i0 = m.s.index("__device__ __forceinline__ void tk2f_consume(")
    m.s = ins(m.s, i0, H)
    ic = m.s.index("__device__ __forceinline__ void tk2f_consume(")
    ie = m.s.index("\n}\n", ic)
    for spin in ("        } while (c < 64);\n", "        } while (c < 2 * RPE);\n", "        } while (c < need);\n"):
        m.rep(spin, spin + "        if (!GATED && tid == 0 && R_ == r0) TSR(2, 6);\n", ic, ie)
        ie = m.s.index("\n}\n", ic)
    w = "      mbar_wait(&full[st], ph);\n      const uint4* sw4 = wsm + st * (SB / 16);\n"
    m.rep(w, w + "      if (tid == 0 && R_ == r0 && q == 0) TSR(2, GATED ? 3 : 7);\n", ic, ie)
    # producer: first / last TMA issue of the tk2f_produce used by moe_tk2f (stamp per call: GATED rows 512 / 256)
    km = m.s.index("__global__ void __launch_bounds__((NC + 1 + tk2f_rel_on<NC, DYN, FN, GS, MRG, PF>()) * 32, 1) moe_tk2f(")
    m.rep("  int gen = 0;\n", "  if (threadIdx.x == 0) TSR(2, 0);\n  int gen = 0;\n", km)
    m.rep("  const int nact = s_nact;\n", "  const int nact = s_nact;\n  if (tid == 0) TSR(2, 1);\n", km)
    a = '  asm volatile("griddepcontrol.wait;" ::: "memory");  // s27 TK2FX: x is the PDL predecessor\'s output\n'
    m.rep(a, a + "  if (tid == 0) TSR(2, 2);\n", km)
    a = '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  tk2f_consume<256, NC, NSTG, false'
    m.rep(a, "  if (tid == 0) TSR(2, 4);\n" + a, km)
    a = "  if (upf != nullptr && tid < 32)\n"
    m.rep(a, "  if (tid == 0) TSR(2, 8);\n" + a, km)
    m.rep("      int i = 0;\n      constexpr bool SQ", "      int i = 0;\n      TSR(2, 9);\n      constexpr bool SQ", km)
    a = "      tk2f_produce<256, NC, NSTG, KPS, SQ, S6, (PF & 2048) != 0>(Wq2, S2, b0, b1, s_rs2, s_e, smem, full, empty, st, ph, i);\n"
    m.rep(a, a + "      TSR(2, 10);\n", km)
    g["_moe_CUDA_SRC"] = m.s
    # ---- pr8 ----
    p = P(g["_GPR8_SRC"])
    k = p.s.index("namespace pr8 {")
    p.s = p.s[:k] + H + p.s[k:]
    kf = p.s.index("__device__ __forceinline__ void finish(const Args& a) {")
    p.rep("    if (t < E) bs_[t] = __ldg(a.bias + t);\n", "    if (t == 0) TSR(1, 0);\n    if (t < E) bs_[t] = __ldg(a.bias + t);\n", kf)
    p.rep('    asm volatile("griddepcontrol.wait;" ::: "memory");\n', '    asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (t == 0) TSR(1, 1);\n', kf)
    p.rep("        done = 1;\n", "        done = 1;\n        TSR(1, 2);\n", kf)
    p.rep("    if (t == 0) a.cnt[0] = 0;\n", "    if (t == 0) a.cnt[0] = 0;\n    __syncthreads();\n    if (t == 0) TSR(1, 3);\n", kf)
    kk = p.s.index("fused(const Args a) {")
    p.rep('    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n    if ((int)blockIdx.x == a.bk) {',
          '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n    if (threadIdx.x == 0 && (int)blockIdx.x != a.bk) TSR(0, 0);\n    if ((int)blockIdx.x == a.bk) {', kk)
    p.rep('    asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (has_r && rlate)',
          '    asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (threadIdx.x == 0) TSR(0, 1);\n    if (has_r && rlate)', kk)
    p.rep("    TR(5)\n}", "    TR(5)\n    if (threadIdx.x == 0) TSR(0, 2);\n}", kk)
    g["_GPR8_SRC"] = p.s
    # ---- nu8 (in _N2U_SRC) ----
    n = P(g["_N2U_SRC"])
    k = n.s.index("namespace nu8 {")
    n.s = n.s[:k] + H + n.s[k:]
    k = n.s.index("namespace nu8 {")
    n.rep('    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n',
          '    if (tid == 0) TSR(3, 0);\n    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n', k)
    n.rep('    asm volatile("griddepcontrol.wait;" ::: "memory");\n', '    asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (tid == 0) TSR(3, 1);\n', k)
    a = '        asm volatile("cp.async.wait_all;" ::: "memory");\n        __syncthreads();\n        for (int i = 0; i < nl; ++i) ys[i * YS + tid] *= wv;\n'
    n.rep(a, a.replace("__syncthreads();\n", "__syncthreads();\n        if (tid == 0) TSR(3, 2);\n"), k)
    a = "            atomicAdd(out + (size_t)s_list[i] * 1024 + r0 + r, v);\n        }\n        return;\n"
    n.rep(a, a.replace("        return;\n", "        __syncthreads();\n        if (tid == 0) TSR(3, 3);\n        return;\n"), k)
    a = "        const int mt = warp & 3, kq = warp >> 2, g = lane >> 2, tq = lane & 3, nt = (nl + 7) >> 3;\n        float c[8][4];\n"
    n.rep(a, "        if (tid == 0) TSR(3, 4);\n" + a, k)
    if os.environ.get("O42_KS"):
        a = "                }\n            }\n        }\n        __syncthreads();\n        float* red = ys;\n"
        n.rep(a, a.replace("            }\n        }\n        __sync", "            }\n            if (lane == 0 && warp == 0 && (ks < 3 || ks == 5) && __float_as_uint(c[0][0] + c[1][3]) != 0x7fc0deadu) TSR(3, ks == 5 ? 13 : 10 + ks);\n        }\n        __sync", 1), k)
    a = "        __syncthreads();\n        float* red = ys;\n#pragma unroll\n        for (int n = 0; n < 8; ++n) {\n"
    n.rep(a, "        if (tid == 0) TSR(3, 9);\n        if (lane == 0 && (warp == 0 || warp == 15) && __float_as_uint(c[0][0] + c[1][3]) != 0x7fc0deadu) TSR(3, warp == 0 ? 7 : 8);\n" + a, k)
    a = "        __syncthreads();\n        float* red = ys;\n#pragma unroll\n        for (int n = 0; n < 8; ++n) {\n"
    n.rep(a, a.replace("        float* red", "        if (tid == 0) TSR(3, 5);\n        float* red"), k)
    a = "        __syncthreads();\n        for (int idx = tid; idx < nl * 64; idx += 512) {\n"
    n.rep(a, a.replace("        for (int idx", "        if (tid == 0) TSR(3, 6);\n        for (int idx"), k)
    g["_N2U_SRC"] = n.s
    # ---- bnd2 ----
    b = P(g["_GBND2_SRC"])
    k = b.s.index("__global__ void __launch_bounds__(NT) bnd2(")
    b.s = ins(b.s, k, H)
    k = b.s.index("__global__ void __launch_bounds__(NT) bnd2(")
    b.rep('    if (TRIG) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n',
          '    if (threadIdx.x == 0) TSR(4, 0);\n    if (TRIG) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n', k)
    b.rep('    asm volatile("griddepcontrol.wait;" ::: "memory");\n', '    asm volatile("griddepcontrol.wait;" ::: "memory");\n    if (threadIdx.x == 0) TSR(4, 1);\n', k)
    a = "        *reinterpret_cast<uint4*>(hid + (size_t)row * N + c0) = o4;\n    }\n}\n"
    b.rep(a, a[:-2] + "    if (threadIdx.x == 0) TSR(4, 2);\n}\n", k)
    g["_GBND2_SRC"] = b.s


NU8_A = ("        for (int idx = tid; idx < nl * 64; idx += 512) {\n"
         "            const int i = idx >> 6, r = idx & 63;\n"
         "            const float v = red[i * 64 + r] + red[(64 + i) * 64 + r] + red[(128 + i) * 64 + r] + red[(192 + i) * 64 + r];\n"
         "            atomicAdd(out + (size_t)s_list[i] * 1024 + r0 + r, v);\n"
         "        }\n")
NU8_V4 = ("        for (int idx = tid; idx < nl * 16; idx += 512) {  // o42: one red.v4 per 4 rows\n"
          "            const int i = idx >> 4, r = (idx & 15) * 4;\n"
          "            const float4 v0 = *reinterpret_cast<const float4*>(red + i * 64 + r), v1 = *reinterpret_cast<const float4*>(red + (64 + i) * 64 + r);\n"
          "            const float4 v2 = *reinterpret_cast<const float4*>(red + (128 + i) * 64 + r), v3 = *reinterpret_cast<const float4*>(red + (192 + i) * 64 + r);\n"
          "            atomicAdd(reinterpret_cast<float4*>(out + (size_t)s_list[i] * 1024 + r0 + r),\n"
          "                      make_float4(v0.x + v1.x + v2.x + v3.x, v0.y + v1.y + v2.y + v3.y, v0.z + v1.z + v2.z + v3.z, v0.w + v1.w + v2.w + v3.w));\n"
          "        }\n")


def nu8_mode(g, mode):
    """mode 'v4': vector red (sum order per element unchanged: ((r0 + r1) + r2) + r3, then the atomic);
    'store': plain stores instead of atomics (WRONG output, timing oracle only)."""
    s = g["_N2U_SRC"]
    assert s.count(NU8_A) == 1
    if mode == "v4":
        s = s.replace(NU8_A, NU8_V4)
    elif mode == "store":
        s = s.replace(NU8_A, NU8_A.replace("atomicAdd(out + (size_t)s_list[i] * 1024 + r0 + r, v);", "out[(size_t)s_list[i] * 1024 + r0 + r] = v;"))
    elif mode == "ks1":  # D path: rolled ks loop (same per-accumulator order: bitwise)
        k = s.index("    if constexpr (D) {")
        a = "#pragma unroll\n        for (int ks = 0; ks < KB / 64; ++ks) {\n"
        i = s.index(a, k)
        s = s[:i] + "#pragma unroll 1\n" + s[i + len("#pragma unroll\n"):]
    if mode in ("fast", "mmq"):  # D path: MMA on a compile-time slot-tile count (no runtime n < nt guards)
        k = s.index("    if constexpr (D) {")
        a0 = "#pragma unroll\n        for (int ks = 0; ks < KB / 64; ++ks) {\n"
        i = s.index(a0, k)
        j = s.index("        __syncthreads();\n        float* red = ys;\n", i)
        body = s[i:j].replace("                if (n < nt) {", "                if (n < NTC) {")
        body = "\n".join(("    " + l) if l.strip() and not l.startswith("#") else l for l in body.split("\n"))
        new = ("        auto mmq = [&](auto ntc) {  // o42: nt as a template constant\n"
               "            constexpr int NTC = decltype(ntc)::v;\n" + body +
               "        };\n"
               "        switch (nt) {\n" + "".join("            case %d: mmq(O42IC<%d>{}); break;\n" % (q, q) for q in range(1, 9)) +
               "            default: break;\n        }\n")
        s = s[:i] + new + s[j:]
        k = s.index("namespace nu8 {")
        s = s[:k] + "template <int N> struct O42IC { static constexpr int v = N; };\n" + s[k:]
    if mode in ("fast", "ssq"):  # D path: slot sums of squares spread over the row groups (rg r: slots i = r mod NRG)
        k = s.index("    if constexpr (D) {")
        a = "        if (rg == 0) {\n            for (int i = 0; i < nl; ++i) {\n"
        assert s.index(a, k) < s.index("float c[8][4];", k)
        s = s[:k] + s[k:].replace(a, "        {\n            for (int i = rg; i < nl; i += gridDim.x / NKC) {\n", 1)
        a = "        if (rg == 0 && tid < nl) {\n"
        s = s[:k] + s[k:].replace(a, "        if (tid < nl && tid % (gridDim.x / NKC) == rg) {\n", 1)
    g["_N2U_SRC"] = s


if __name__ == "__main__":
    # python stp_patch.py <layer submission.py> <out.py> <nu8 mode: mmq | ssq | fast | v4>: bake a nu8 variant into a file
    import sys
    s = open(sys.argv[1]).read()
    a = s.index("_N2U_SRC = r'''")
    b = s.index("'''", a + 20)
    g = {"_N2U_SRC": s[a:b]}
    nu8_mode(g, sys.argv[3])
    open(sys.argv[2], "w").write(s[:a] + g["_N2U_SRC"] + s[b:])
