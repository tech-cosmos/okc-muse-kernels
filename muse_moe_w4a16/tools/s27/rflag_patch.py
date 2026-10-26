"""moe s27 RFLAG: T=1 moe_tk2f launched at the pre-expert rmsnorm's START (TRIG, not TRIG2), i.e. while pr1 still runs;
it waits for pr1's routing with a flag (moe st["cnt"][97], release-stored by pr1's routing warp after ids/lids/wts;
zeroed by n2u after tk2f completes, like the rest of cnt[0..127]). pr1's ticket moves to st["cnt"][96] (call-time view:
clone-safe). tk2f gets the flag pointer through its (non-FN, unused) tw arg via moe_set_rflag (one-shot host static).
Usage: python rflag_patch.py <layer dir>  (edits layer.py + pr1_src.cu in place; then python build.py)."""
import sys, os
d = sys.argv[1]
L = os.path.join(d, "layer.py"); P = os.path.join(d, "pr1_src.cu")
s = open(L).read(); p = open(P).read()

def rep(t, a, b, n=1):
    assert t.count(a) == n, (a[:80], t.count(a))
    return t.replace(a, b)

# pr1: flag store after the routing writes
p = rep(p, """        a.wts[lane] = mysc / (sum + a.reps);
    }
    if (lane == 0) *a.cnt = 0;
}""", """        a.wts[lane] = mysc / (sum + a.reps);
    }
    if (lane == 0) *a.cnt = 0;
#ifdef PR1_RFLAG
    __syncwarp();
    if (lane == 0) {  // moe s27 RFLAG: routing ready for the early-launched moe_tk2f (a.cnt = moe cnt + 96)
        __threadfence();
        asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(a.cnt + 1), "r"(1) : "memory");
    }
#endif
}""")
p = "#define PR1_RFLAG 1\n" + p

# moe: host static + setter, spin before the routing, __ldcg ids
anchor = '_moe__CPP = _moe__CPP + "int64_t moe_set_tk2fpdl(int64_t);"\n'
add = r'''# moe s27 RFLAG (see tools/s27/rflag_patch.py in r-moe)
_a = "\nstatic const char* g_upf = nullptr;"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "\nstatic const void* g_rflag = nullptr;\nint64_t moe_set_rflag(int64_t p) { g_rflag = (const void*)p; return 0; }" + _a)
_moe__CPP = _moe__CPP + "int64_t moe_set_rflag(int64_t);"
_a = "  return launch_k(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, grid, (NC + 1) * 32, SMEM, st, pdlx,"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "  if (!FN && !tw && g_rflag) { tw = g_rflag; }\n  g_rflag = nullptr;  // one-shot\n" + _a)
_a = "  {  // routing, as moe_tk2 GATED\n    int e = -1, rank = 0;\n"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, """  if (!FN && tw != nullptr) {  // moe s27 RFLAG: launched before pr1 finished: wait for its routing flag
    if (tid == 0) {
      int v;
      do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(tw) : "memory"); } while (v == 0);
    }
    __syncthreads();
  }
""" + _a)
# (ids are already read with __ldcg: the layer's own later patch)
# RSPLIT (eps arg, unused without FN): Q * 1000 + W: the first Q CTAs (they land on pr1's early-exiting router SMs)
# get W/100 of a normal CTA's gemm13 rows (gemm2 stays uniform: it starts after the phase barrier anyway)
_a = "  int b0 = (int)((long long)n2 * blockIdx.x / gridDim.x), b1 = (int)((long long)n2 * (blockIdx.x + 1) / gridDim.x);\n"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + """  if (!FN && eps > 0.f) {  // moe s27 RSPLIT
    const int q_ = (int)eps / 1000, w_ = (int)eps % 1000;
    auto cum = [&](int b) -> long long { return b <= q_ ? (long long)b * w_ : (long long)q_ * w_ + (long long)(b - q_) * 100; };
    const long long tot = cum(gridDim.x);
    a0 = (int)((long long)n13 * cum(blockIdx.x) / tot); a1 = (int)((long long)n13 * cum(blockIdx.x + 1) / tot);
  }
""")
_a = "  if (!FN && !tw && g_rflag) { tw = g_rflag; }\n"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + "  if (!FN && g_rflag && %d > 0) eps = (float)%d;\n" % (RSPLIT, RSPLIT))
'''
s = rep(s, anchor, anchor + add)
s = rep(s, '"moe_set_tk2fpdl", ', '"moe_set_tk2fpdl", "moe_set_rflag", ')
s = rep(s, "TK2FX_DEFAULT = True", "TK2FX_DEFAULT = True\nRFLAG = True  # moe s27")
s = rep(s, "\nimport torch\n", "\nimport torch\nRSPLIT = 0  # moe s27 RSPLIT (Q * 1000 + W)\n")
# pr1 call: ticket at moe cnt[96], flag at cnt[97]
s = rep(s, """_G10["lib"].pr1_launch(pk[0], pk[1], pk[2], h, pooled, pk[3], state["router"], h32, logits, state["pr1_cnt"],""",
        """_G10["lib"].pr1_launch(pk[0], pk[1], pk[2], h, pooled, pk[3], state["router"], h32, logits, state["moe"]["cnt"][96:],""")
# rmsnorm: trigger at start when RFLAG at T=1
s = rep(s, "launch_pdl=rpdl, z_ptr=y32[64] if nu8 else None, ZN=256 if nu8 else 0, TRIG2=tk2fx or tk2fj)",
        "launch_pdl=rpdl, z_ptr=y32[64] if nu8 else None, ZN=256 if nu8 else 0, TRIG2=(tk2fx or tk2fj) and not (RFLAG and tokens == 1 and tk2fx))")
s = rep(s, "TRIG=tk2_pdl, ROUND=pooled.dtype", "TRIG=tk2_pdl or (RFLAG and tokens == 1 and tk2fx), ROUND=pooled.dtype")
s = rep(s, """        state["moe"]["lib"].moe_set_tk2fpdl(1)  # the next moe_tk2f launch (one-shot) is this rmsnorm's PDL secondary\n""",
        """        state["moe"]["lib"].moe_set_tk2fpdl(1)  # the next moe_tk2f launch (one-shot) is this rmsnorm's PDL secondary\n        if RFLAG and tokens == 1 and tk2fx:\n            state["moe"]["lib"].moe_set_rflag(state["moe"]["cnt"].data_ptr() + 97 * 4)\n""")
open(L, "w").write(s); open(P, "w").write(p)
print("ok")
