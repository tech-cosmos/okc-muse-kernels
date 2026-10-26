# proj s33: gemv13 (12-bit) for the layer's T=1 o projection as a PDL secondary of attention.
import sys
p = sys.argv[1]
s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)
rep('G13_CFG = {"qkv_gate": (2, 4, 1, 1, 132), "pooler": (2, 4, 1, 1, 132)}  # (u, nv, minb, pf, blocks), k = 8192',
    'G13_CFG = {"qkv_gate": (2, 4, 1, 1, 132), "pooler": (2, 4, 1, 1, 132)}  # (u, nv, minb, pf, blocks), k = 8192\n'
    '# s33 (proj): o at T=1 on gemv13 (12-bit W, 12.6 vs 16.8 MB) as a PDL secondary: W prologue (.nc) before\n'
    '# griddepcontrol.wait, x (__ldcg) after. k = 1024. TUNE option "g13o" at T=1 (H100 d1 full -1.0..-1.3 us, sliding\n'
    '# +-0.3: attention and o share HBM, so W bytes moved before the wait slow csl about as much as they save; NOTES-proj s33).\n'
    'G13_O = (2, 1, 4, 1, 528)  # (u, nv, minb, pf, blocks); None = no o pack\n'
    'G13_K = {"o": 1024}')
rep('            src13 = (_G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(8192,%d,%d,%d,%d)" % c[:4] for c in sorted(set(G13_CFG.values()) | {a for v in G13_ALT.values() for a in v})))',
    '            src13 = (_G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(8192,%d,%d,%d,%d)" % c[:4] for c in sorted(set(G13_CFG.values()) | {a for v in G13_ALT.values() for a in v}))\n'
    '                                      + " " + " ".join("G13_CASE(1024,%d,%d,%d,%d)" % c[:4] for c in [G13_O] if c))')
# kernel: pdl flag; x after the wait in PDL mode
rep('                 "                                                       const char* __restrict__ pfv, int pfn, int pfrb) {"),',
    '                 "                                                       const char* __restrict__ pfv, int pfn, int pfrb, int pdl) {"),\n'
    '                ("        if (i < KK / 8) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);\\n",\n'
    '                 "        if (i < KK / 8 && !pdl) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);\\n"),')
rep('                 "            }\\n        }\\n    }\\n"),',
    '                 "            }\\n        }\\n    }\\n"\n'
    '                 "    if (pdl) {  // s33: PDL secondary: W prologue above, x (the predecessor\'s output) after the wait\\n"\n'
    '                 "        asm volatile(\\"griddepcontrol.wait;\\" ::: \\"memory\\");\\n"\n'
    '                 "#pragma unroll\\n"\n'
    '                 "        for (int q = 0; q < XV; ++q) {\\n"\n'
    '                 "            const int i = threadIdx.x + q * 256 * NV;\\n"\n'
    '                 "            if (i < KK / 8) xv[q] = __ldcg(reinterpret_cast<const uint4*>(x) + i);\\n"\n'
    '                 "        }\\n    }\\n"),')
rep('                 "              torch::Tensor pft, torch::Tensor pfk, torch::Tensor pfv, int64_t pfn, int64_t pfrb) {"),',
    '                 "              torch::Tensor pft, torch::Tensor pfk, torch::Tensor pfv, int64_t pfn, int64_t pfrb, int64_t pdl) {"),\n'
    '                ("        g13::gemv<U_, V_, B_, K_, P_><<<blocks, 256 * V_, 0, st>>>(",\n'
    '                 "        cudaLaunchConfig_t lc = {}; lc.gridDim = blocks; lc.blockDim = 256 * V_; lc.dynamicSmemBytes = 0; \\\\\\n"\n'
    '                 "        lc.stream = st; cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \\\\\\n"\n'
    '                 "        la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0; \\\\\\n"\n'
    '                 "        cudaLaunchKernelEx(&lc, g13::gemv<U_, V_, B_, K_, P_>, "),')
rep('                 "        (const int*)pft.data_ptr(), (const char*)pfk.data_ptr(), (const char*)pfv.data_ptr(), (int)pfn, (int)pfrb); return; }"),',
    '                 "        (const int*)pft.data_ptr(), (const char*)pfk.data_ptr(), (const char*)pfv.data_ptr(), (int)pfn, (int)pfrb, (int)pdl); return; }"),')
# prepare: pack o
rep('    state["g13"] = {key: pk for key in G13_CFG if (pk := _pack13(w[key])) is not None}',
    '    state["g13"] = {key: pk for key in list(G13_CFG) + (["o"] if G13_O else []) if (pk := _pack13(w[key])) is not None}')
# _linear
rep('        if pk is not None and key not in GEMV_PDL_SECONDARY.get((x.shape[0], sliding), ()):',
    '        o13 = x.shape[0] == 1 and key == "o" and bool(_opt(state, 1, "g13o", False))\n'
    '        if pk is not None and (key not in GEMV_PDL_SECONDARY.get((x.shape[0], sliding), ()) or o13) and (key != "o" or o13):')
rep('                lib.g13_gemv(pk[0], pk[1], pk[2], x.contiguous(), out, pk[3],\n'
    '                              *_cfg(state, 1, "g13", key, G13_CFG), *pf)',
    '                lib.g13_gemv(pk[0], pk[1], pk[2], x.contiguous(), out, pk[3],\n'
    '                              *(G13_O if o13 else _cfg(state, 1, "g13", key, G13_CFG)), *pf,\n'
    '                              int(o13 and UNPOOL_PDL and key in GEMV_PDL_SECONDARY.get((1, sliding), ())))')
open(p, "w").write(s)
print("ok")
s = open(p).read()
rep('"int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);\\n"\n                            "void g14_gemm(',
    '"int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\\n"\n                            "void g14_gemm(')
open(p, "w").write(s)
print("ok2")
s = open(p).read()
rep('            {"rf_trig": False}),\n', '            {"rf_trig": False}, {"g13o": True}),\n')
open(p, "w").write(s)
print("ok3")
