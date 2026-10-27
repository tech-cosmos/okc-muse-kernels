# okc: problem=muse_decode_layer
"""Muse Spark Open decoder layer (TP8/EP8, one GPU), as SGLang's MuseSpark1xDecoderLayer.forward computes it,
with fewer kernels:

  qkv_gate projection      cuBLAS (F.linear), as SGLang's bf16_gemm_dispatch on H200
  attention                our muse_attention_decode kernels (attn_src.py, inlined by build.py): QK-norm, RoPE,
                           FP8 cache store, attention over the paged FP8 cache, output RMSNorm x sigmoid(gate)
  o projection             cuBLAS
  boundary 1               one Triton kernel: post-attention norm, residual gate, pre-FFN norm (BF16 h, and the
                           FP32 h SGLang's router reads up to 8 tokens)
  router + routing         side stream, overlapping the pooler: a split-K Triton GEMV (FP32 up to 8 tokens, BF16
                           hi/lo weight pair with FP32 accumulation above, as MuseSpark1xRouter), then one Triton
                           kernel: sum the splits, sigmoid, biased top-8, renormalize, this GPU's expert ids
  pooler                   cuBLAS
  pre-expert norm          Triton
  experts                  SGLang's Marlin W4A16 MoE path (judge image), a dequantizing fallback locally
  unpooler                 cuBLAS (this GPU's 1024-row slice up to 8 tokens)
  boundary 2               one Triton kernel: post-FFN norm (the all-gather's zero columns implicit), residual gate,
                           next layer's pre-attention norm"""

from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents

try:
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    HAVE_SGL = True
except ImportError:
    HAVE_SGL = False

# s68: every JIT build (attention csl/nvs/nds, moe, g10, lt, g24) runs at once on this pool from import on; prepare()
# joins them (cold H100: serial import 97 s + prepare 217 s). Each build has its own name/dir; a build that fails on the
# pool is retried serially in prepare (attention: Triton fallback as before).
import concurrent.futures as _cf
_BUILD_POOL = _cf.ThreadPoolExecutor(8)


class _LazyBuild:
    def __init__(self, fut, name):
        self.fut, self.name = fut, name

    def get(self):
        try:
            return self.fut.result()
        except Exception as e:  # noqa: BLE001
            print(f"{self.name} build failed, using Triton:", e)
            return None


# ---- attention (attn_src.py) ----
"""Gated attention decode in two Triton launches.

1. _attn_split: one program per (split of the context, request, KV head). It normalizes (and
   on sliding layers ropes) its group's queries straight from the fused projection output,
   streams its slice of the paged FP8 cache through an online softmax, and writes a partial
   (max, sum, accumulator). The split-0 program also normalizes/ropes the block's new keys,
   quantizes the new keys and values to FP8, stores them in the cache, and attends them
   causally from registers.
2. _combine: one program per (token, KV head). It merges the splits, then applies the
   per-head RMSNorm and sigmoid(gate)."""

from typing import TypedDict

import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

try:  # Gluon (Triton's low-level dialect, ships with triton 3.8): the verify full-layer kernel.
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, warpgroup_mma,
                                                                    warpgroup_mma_wait)
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2
    HAVE_GLUON = True
except ImportError:
    HAVE_GLUON = False


from triton.runtime.jit import constexpr_function as _constexpr_function


@_constexpr_function
def _np2_jit(n):
    return 1 << max(0, int(n) - 1).bit_length()


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


# Decode sliding layers at batch 1: one native kernel (tools/s72/csl.cu), a 16-CTA cluster per request whose split
# partials merge through distributed shared memory (st.async + mbarrier), instead of _attn_split + a merge kernel.
# Built once at import; the Triton path is the fallback if the build fails.
def _load_inline(**kw):  # s68: built on layer's _BUILD_POOL, resolved in prepare (_join_builds)
    from torch.utils.cpp_extension import load_inline
    return _LazyBuild(_BUILD_POOL.submit(load_inline, **kw), kw['name'])
_CSL_SRC = r'''// Batch-1-style decode sliding attention as ONE kernel: per (request, KV head) one 16-CTA thread-block cluster
// (s91: a cluster per KV head halves each SM's gathered bytes; H100 paired decode 1/16 sliding -0.58/-0.14 us).
// CTA rank r covers window keys [128 r, 128 r + 128) of its KV head (4 warps, 32 keys each). QK^T and P@V are mma.sync m16n8k32 E4M3 (P in E4M3 with FA3's max offset 8, V^T built in registers
// by byte permutes), FP32 partials pushed by each warp straight into the owning CTA's shared memory
// (st.shared::cluster), one cluster barrier, then CTA r < 8 merges query head 8 h + r (RMSNorm, sigmoid gate, BF16).
// Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 1.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef CSL_SPEC
#define CSL_SPEC 1
#endif
namespace csl {
constexpr int HQ = 16, HKV = 2, D = 64, W = 2048, QKV_W = (2 * HQ + 2 * HKV) * D;

__device__ __forceinline__ unsigned pk8(float hi, float lo) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void mma8(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                     unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned cta) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}
__device__ __forceinline__ void st4(unsigned addr, float a, float b, float c, float d) {
    asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d)
                 : "memory");
}
__device__ __forceinline__ void st2(unsigned addr, float a, float b) {
    asm volatile("st.shared::cluster.v2.f32 [%0], {%1,%2};" ::"r"(addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void sa4(unsigned addr, unsigned mbar, float a, float b, float c, float d) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
                 ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d), "r"(mbar) : "memory");
}
__device__ __forceinline__ void sa2(unsigned addr, unsigned mbar, float a, float b) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1,%2}, [%3];"
                 ::"r"(addr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
// layer s28 (QF32): qkv may be the FP32 output of the qkv GEMM (layer g19 at T=16); each element is rounded to BF16
// (RN, as the torch cast it replaces) at its load, so everything downstream is bit-identical to the BF16 input.
__device__ __forceinline__ unsigned pkbf(float lo, float hi) {
    __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
    return *(unsigned*)&t;
}
__device__ __forceinline__ unsigned ld2(const unsigned short* p) { return *(const unsigned*)p; }
__device__ __forceinline__ unsigned ld2(const float* p) { const float2 f = *(const float2*)p; return pkbf(f.x, f.y); }
__device__ __forceinline__ void ld16(const unsigned short* p, uint4& a, uint4& b) {
    a = *(const uint4*)p; b = *(const uint4*)(p + 8);
}
__device__ __forceinline__ void ld16(const float* p, uint4& a, uint4& b) {
    const float4 f0 = *(const float4*)p, f1 = *(const float4*)(p + 4), f2 = *(const float4*)(p + 8),
                 f3 = *(const float4*)(p + 12);
    a = make_uint4(pkbf(f0.x, f0.y), pkbf(f0.z, f0.w), pkbf(f1.x, f1.y), pkbf(f1.z, f1.w));
    b = make_uint4(pkbf(f2.x, f2.y), pkbf(f2.z, f2.w), pkbf(f3.x, f3.y), pkbf(f3.z, f3.w));
}

template <int PAGE, typename QT, bool PW = false>
__global__ void __launch_bounds__(128) decode(const QT* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec,
                                              const unsigned char* __restrict__ pf, long long pfb) {
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) float lpart[2][4][8][D];
    __shared__ __align__(16) float lml[2][4][8][2];
    __shared__ __align__(16) unsigned char newkv[2][2][D];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = (blockIdx.x >> 4) & 1, wk = warp;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int b = blockIdx.x / 32;
    // The owner's receive barrier counts the 16 partials' bytes (st.async complete_tx). Peers must have started
    // (and initialized it) before anyone pushes: arrive now, wait before the pushes.
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0 && rank < 8) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (16 * 16 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Every load that doesn't need the page table goes out first (warps issue in order, and the gathers' addresses
    // wait for the page-table trip): queries, the new token, the gate, scales; then cos/sin (needs seq).
    // layer s62 (PW): a PDL secondary of the qkv GEMM: the page table, cos/sin and K/V gathers (inputs only) go out
    // before griddepcontrol.wait, the qkv rows after it.
    const QT* qp = qkv + (long)b * QKV_W + (h * 8 + g) * D + 16 * c;
    uint4 qa, qb;
    const QT* kp = qkv + (long)b * QKV_W + (HQ + h) * D + 2 * lane;
    unsigned kw, vw, gw;
#define CSL_QLOADS                                                                                             \
    ld16(qp, qa, qb);                                                                                          \
    kw = ld2(kp); vw = ld2(kp + HKV * D);                                                                      \
    gw = ld2(qkv + (long)b * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);
    if (!PW) { CSL_QLOADS }
    const long nloc = loc[b];
    const float kscale = *ks, vscale = *vs;
    // spec >= 0 (batch 1, page 1): SGLang slices the page table to the batch's longest length, so the window's
    // page-table range is known from the table width: gather from it without waiting for seq_lens, then compare
    // with the real length and redo the loads if it differs (never on SGLang's inputs; correct either way).
    int seq = spec >= 0 ? spec : seqs[b];
    const float* csr;
    float4 c0, c1, s0, s1;
    float co, si;
    int base, w0, tmax;
    const int* ptr = pt + (long)b * MP;
    long kslot[4], vslot[8];
    uint4 kr[4];
    uint2 vr[8];
#define CSL_LOADS                                                                                              \
    csr = cs + (long)seq * D + 8 * c;                                                                          \
    c0 = *(const float4*)csr; c1 = *(const float4*)(csr + 4);                                                  \
    s0 = *(const float4*)(csr + 32); s1 = *(const float4*)(csr + 36);                                          \
    co = cs[(long)seq * D + lane]; si = cs[(long)seq * D + 32 + lane];                                         \
    base = seq - (W - 1);                                                                                      \
    w0 = rank * 128 + wk * 32;                                                                                 \
    tmax = max(seq - 1, 0);                                                                                    \
    /* Keys of this thread: K rows 8j + g (j 0..3), V rows {2c, 2c+1, 8+2c, 9+2c, 16+2c, ...}. */             \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                            \
        int t = min(max(base + w0 + 8 * j + g, 0), tmax);                                                      \
        kslot[j] = PAGE == 1 ? (long)max(ptr[t], 0) : (long)max(ptr[t / PAGE], 0) * PAGE + t % PAGE;           \
    }                                                                                                          \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                            \
        int k = 8 * (i >> 1) + 2 * c + (i & 1);                                                                \
        int t = min(max(base + w0 + k, 0), tmax);                                                              \
        vslot[i] = PAGE == 1 ? (long)max(ptr[t], 0) : (long)max(ptr[t / PAGE], 0) * PAGE + t % PAGE;           \
    }                                                                                                          \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c); \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
    CSL_LOADS
    if (spec >= 0) {
        const int real = seqs[b];
        if (real != seq) { seq = real; CSL_LOADS }
    }
#undef CSL_LOADS
    if (PW) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        CSL_QLOADS
    }
#undef CSL_QLOADS
    const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
    float x[16];
    {
        unsigned u[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    const float inv = rsqrtf(ss / D + 1e-5f);
    unsigned q8[4];
    {
        float cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float sn[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        float y[16];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
            y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) q8[w] = pk8(y[4 * w + 1], y[4 * w]) | (pk8(y[4 * w + 3], y[4 * w + 2]) << 16);
    }

    // The new token (window key 2047): CTA 15, the last warp of each head.
    const bool newer = rank == 15 && wk == 3;
    if (newer) {
        float k0 = bf(kw & 0xffff), k1 = bf(kw >> 16);
        float s = k0 * k0 + k1 * k1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        float ki = rsqrtf(s / D + 1e-5f) / kscale;
        float r0 = (k0 * co - k1 * si) * ki, r1 = (k1 * co + k0 * si) * ki;
        unsigned short k8 = pk8(r1, r0);
        unsigned short v8 = pk8(bf(vw >> 16) / vscale, bf(vw & 0xffff) / vscale);
        *(unsigned short*)&newkv[h][0][2 * lane] = k8;
        *(unsigned short*)&newkv[h][1][2 * lane] = v8;
        long slot = nloc * (HKV * D) + h * D + 2 * lane;
        *(unsigned short*)(kc + slot) = k8;
        *(unsigned short*)(vc + slot) = v8;
        __syncwarp();
        if (g == 7) kr[3] = *(const uint4*)&newkv[h][0][16 * c];
        if (c == 3) vr[7] = *(const uint2*)&newkv[h][1][8 * g];
    }

    // S = Q K^T for row g: tile j holds keys 8j + 2c + {0, 1}.
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], q8[0], 0u, q8[1], 0u, kr[j].x, kr[j].y);
        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);
    }
    const float scale = qk_scale * kscale;
    float m = -1e30f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            bool ok = base + w0 + 8 * j + 2 * c + e >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            m = fmaxf(m, sc[j][e]);
        }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    const float mo = m - 8.f;  // FA3's FP8 max offset: P <= 2^8
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    const unsigned pb0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
    const unsigned pb1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);

    // O^T = V^T P^T: m-tile i holds d 8g + 2i (rows m = g) and 8g + 2i + 1 (m = g + 8).
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int bsel = (i & 1) ? 0x7362 : 0x5140;
        unsigned w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
        unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
        unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632), __byte_perm(t45, t67, 0x5410),
             __byte_perm(t45, t67, 0x7632), pb0, pb1);
    }

    if (pfb > 0 && warp == 1) {  // layer s34 (O_PF): K/V consumed: L2-prefetch this CTA's share of the next GEMM's weight
        const long long per = ((pfb / gridDim.x) + 4095) & ~4095LL, end = min(pfb, per * (blockIdx.x + 1));
        for (long long o = per * blockIdx.x + lane * 16384LL; o < end; o += 32 * 16384LL)
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pf + o), "r"((int)min(16384LL, end - o)) : "memory");
    }
    // Pre-merge the 4 warps of each KV head in shared memory, then push one partial per (head, row) to its owner.
    {
        float* lp = &lpart[h][wk][0][0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = lp + (2 * c + e) * D + 8 * g;
            *(float4*)r = make_float4(acc[0][e], acc[0][e + 2], acc[1][e], acc[1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[2][e], acc[2][e + 2], acc[3][e], acc[3][e + 2]);
        }
        if (c == 0) *(float2*)&lml[h][wk][g][0] = make_float2(mo, l);
    }
    __syncthreads();
    {
        const int h2 = h, row = (tid >> 4) & 7, dc = tid & 15;
        float mk[4], lk[4], M = -1e30f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 v = *(const float2*)&lml[h2][k][row][0];
            mk[k] = v.x; lk[k] = v.y; M = fmaxf(M, v.x);
        }
        float L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float w = exp2f(mk[k] - M);
            L += w * lk[k];
            float4 a = *(const float4*)&lpart[h2][k][row][4 * dc];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
        }
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        const unsigned owner = row;
        const unsigned rmb = mapa(mb, owner);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[rank][4 * dc]), owner), rmb, o[0], o[1], o[2], o[3]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[rank][0]), owner), rmb, M, L);
    }

    // Merge query head `rank` from its 16 partials: one warp, two d per lane, no block barriers.
    if (warp == 0 && rank < 8) {
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            float w = exp2f(rml[s][0] - M);
            den += w * rml[s][1];
            float2 a = *(const float2*)&racc[s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
        float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        float ni = rsqrtf(s2 / D + 1e-5f);
        o0 *= ni / (1.f + __expf(-g0));
        o1 *= ni / (1.f + __expf(-g1));
        *(__nv_bfloat162*)(out + (long)b * HQ * D + (h * 8 + rank) * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace csl

void csl_decode(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t pf_ptr, int64_t pf_bytes, int64_t pw) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && CSL_SPEC) ? mp - 1 : -1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(32 * batch);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pw ? 2 : 1;
#define CSL_ARGS(QT) (const QT*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, spec, \
    (const unsigned char*)pf_ptr, (long long)pf_bytes
#define CSL_LAUNCH1(P, QT, PW) { \
        static bool set = false; \
        if (!set) { cudaFuncSetAttribute(csl::decode<P, QT, PW>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
                    if (PW) cudaFuncSetAttribute(csl::decode<P, QT, PW>, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
                    set = true; } \
        cudaLaunchKernelEx(&cfg, csl::decode<P, QT, PW>, CSL_ARGS(QT)); }
#define CSL_LAUNCH(P, QT) { if (pw) CSL_LAUNCH1(P, QT, true) else CSL_LAUNCH1(P, QT, false) }
    TORCH_CHECK(page == 1 || page == 64, "csl: page 1 or 64");
    const bool f32 = qkv.scalar_type() == at::kFloat;
    if (page == 1) {
        if (f32) CSL_LAUNCH(1, float) else CSL_LAUNCH(1, unsigned short)
    } else {
        if (f32) CSL_LAUNCH(64, float) else CSL_LAUNCH(64, unsigned short)
    }
#undef CSL_LAUNCH
#undef CSL_LAUNCH1
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// s96 vtk: verify sliding as csl clusters, one 16-CTA cluster per (request, block token, KV head) (256 CTAs at batch 1):
// each cluster is csl's decode layout for one token; the 8 new tokens (norm, RoPE, quantize; branch-free, no IEEE
// divides) sit at window keys 2047 - j + n in CTA 15's last warp, cluster j writes token j's cache row.
// H100 verify 1 sliding 7.65-7.82 -> 6.43-6.50 us (tools/s96/vtk.cu).
namespace vtk {
constexpr int L = 8, HQ = 16, HKV = 2, D = 64, W = 2048, QKV_W = (2 * HQ + 2 * HKV) * D;

__device__ __forceinline__ unsigned pk8(float hi, float lo) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void mma8(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                     unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned cta) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}
__device__ __forceinline__ void st4(unsigned addr, float a, float b, float c, float d) {
    asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d)
                 : "memory");
}
__device__ __forceinline__ void st2(unsigned addr, float a, float b) {
    asm volatile("st.shared::cluster.v2.f32 [%0], {%1,%2};" ::"r"(addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void sa4(unsigned addr, unsigned mbar, float a, float b, float c, float d) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
                 ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d), "r"(mbar) : "memory");
}
__device__ __forceinline__ void sa2(unsigned addr, unsigned mbar, float a, float b) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1,%2}, [%3];"
                 ::"r"(addr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }

// layer s38: QT float = the FP32 qkv of the layer's g19 at T=8, rounded to BF16 (RN) at each load (as csl's QF32)
template <int PAGE, typename QT, bool PW = false>
__global__ void __launch_bounds__(128) verify(const QT* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec) {
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) float lpart[2][4][8][D];
    __shared__ __align__(16) float lml[2][4][8][2];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = (blockIdx.x >> 4) & 1, wk = warp;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tok = blockIdx.x >> 5, b = tok / L, jt = tok % L;  // this cluster's query token (row of qkv / out)
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    // The owner's receive barrier counts the 16 partials' bytes (st.async complete_tx). Peers must have started
    // (and initialized it) before anyone pushes: arrive now, wait before the pushes.
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0 && rank < 8) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (16 * 16 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Every load that doesn't need the page table goes out first (warps issue in order, and the gathers' addresses
    // wait for the page-table trip): queries, the new token, the gate, scales; then cos/sin (needs seq).
    // layer s62 (PW): a PDL secondary of the qkv GEMM: the qkv rows (queries, gate, new K/V) after griddepcontrol.wait,
    // everything else (page table, cos/sin, K/V gathers) before it.
    const QT* qp = qkv + (long)tok * QKV_W + (h * 8 + g) * D + 16 * c;
    uint4 qa, qb;
    // The block's new tokens 0..jt (only the new-token warp needs them).
    const bool newer = rank == 15 && wk == 3;
    unsigned gw;
    unsigned nkw[L], nvw[L];
#define VTK_QLOADS                                                                                             \
    csl::ld16(qp, qa, qb);                                                                                     \
    gw = csl::ld2(qkv + (long)tok * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);         \
    _Pragma("unroll") for (int n = 0; n < L; ++n) {                                                            \
        const int nn = min(n, jt);                                                                             \
        const QT* kp = qkv + (long)(b * L + nn) * QKV_W + (HQ + h) * D + 2 * lane;                             \
        nkw[n] = newer ? csl::ld2(kp) : 0u; nvw[n] = newer ? csl::ld2(kp + HKV * D) : 0u;                      \
    }
    if (!PW) { VTK_QLOADS }
    const long nloc = loc[tok];
    const float kscale = *ks, vscale = *vs;
    const float rks = 1.f / kscale, rvs = 1.f / vscale;
    // spec >= 0 (batch 1, page 1): SGLang slices the page table to the batch's longest length, so the window's
    // page-table range is known from the table width: gather from it without waiting for seq_lens, then compare
    // with the real length and redo the loads if it differs (never on SGLang's inputs; correct either way).
    int seq = spec >= 0 ? spec : seqs[b];
    // The block's new tokens 0..jt (only the new-token warp needs them), cos/sin rows and cache rows, all issued
    // before the gathers (clamped indices so the loads are unconditional).
    float nco[L], nsi[L];
#pragma unroll
    for (int n = 0; n < L; ++n) {
        // Predicated (not branched) so they stay in the first basic block, issued with the gathers.
        const int nn = min(n, jt);
        nco[n] = newer ? cs[(long)(seq + nn) * D + lane] : 0.f; nsi[n] = newer ? cs[(long)(seq + nn) * D + 32 + lane] : 0.f;
    }
    int pos;
    const float* csr;
    float4 c0, c1, s0, s1;
    int base, w0, tmax;
    const int* ptr = pt + (long)b * MP;
    long kslot[4], vslot[8];
    uint4 kr[4];
    uint2 vr[8];
#define CSL_LOADS                                                                                              \
    pos = seq + jt;                                                                                            \
    csr = cs + (long)pos * D + 8 * c;                                                                          \
    c0 = *(const float4*)csr; c1 = *(const float4*)(csr + 4);                                                  \
    s0 = *(const float4*)(csr + 32); s1 = *(const float4*)(csr + 36);                                          \
    base = pos - (W - 1);                                                                                      \
    w0 = rank * 128 + wk * 32;                                                                                 \
    tmax = max(seq - 1, 0);                                                                                    \
    /* Keys of this thread: K rows 8j + g (j 0..3), V rows {2c, 2c+1, 8+2c, 9+2c, 16+2c, ...}. */             \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                            \
        int t = min(max(base + w0 + 8 * j + g, 0), tmax);                                                      \
        kslot[j] = PAGE == 1 ? (long)max(ptr[t], 0) : (long)max(ptr[t / PAGE], 0) * PAGE + t % PAGE;           \
    }                                                                                                          \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                            \
        int k = 8 * (i >> 1) + 2 * c + (i & 1);                                                                \
        int t = min(max(base + w0 + k, 0), tmax);                                                              \
        vslot[i] = PAGE == 1 ? (long)max(ptr[t], 0) : (long)max(ptr[t / PAGE], 0) * PAGE + t % PAGE;           \
    }                                                                                                          \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c); \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
    CSL_LOADS
    if (spec >= 0) {
        const int real = seqs[b];
        if (real != seq) {
            seq = real; CSL_LOADS
            if (newer) {
#pragma unroll
                for (int n = 0; n < L; ++n) { nco[n] = cs[(long)(seq + n) * D + lane]; nsi[n] = cs[(long)(seq + n) * D + 32 + lane]; }
            }
        }
    }
#undef CSL_LOADS
    if (PW) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        VTK_QLOADS
    }
#undef VTK_QLOADS
    const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
    float x[16];
    {
        unsigned u[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    const float inv = rsqrtf(ss / D + 1e-5f);
    unsigned q8[4];
    {
        float cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float sn[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        float y[16];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
            y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) q8[w] = pk8(y[4 * w + 1], y[4 * w]) | (pk8(y[4 * w + 3], y[4 * w + 2]) << 16);
    }

    // The new tokens n <= jt sit at window keys 2047 - jt + n: all in CTA 15's last warp, K tile 3 (keys 24 + g),
    // V rows 24 + 2c + e. Cluster jt writes token jt's cache row for its head.
    if (newer) {
        __shared__ __align__(16) unsigned char newkv[L][2][D];
#pragma unroll
        unsigned short kj = 0, vj = 0;
#pragma unroll
        for (int n = 0; n < L; ++n) {
            {  // branch-free: n > jt repeats token jt (same values, same smem slot)
                const float co = nco[n], si = nsi[n];
                float k0 = bf(nkw[n] & 0xffff), k1 = bf(nkw[n] >> 16);
                float s = k0 * k0 + k1 * k1;
#pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
                float ki = rsqrtf(s / D + 1e-5f) * rks;
                float r0 = (k0 * co - k1 * si) * ki, r1 = (k1 * co + k0 * si) * ki;
                unsigned short k8 = pk8(r1, r0);
                unsigned short v8 = pk8(bf(nvw[n] >> 16) * rvs, bf(nvw[n] & 0xffff) * rvs);
                *(unsigned short*)&newkv[n][0][2 * lane] = k8;  // slots n > jt are never read
                *(unsigned short*)&newkv[n][1][2 * lane] = v8;
                kj = n == jt ? k8 : kj; vj = n == jt ? v8 : vj;
            }
        }
        {
            long slot = nloc * (HKV * D) + h * D + 2 * lane;
            *(unsigned short*)(kc + slot) = kj;
            *(unsigned short*)(vc + slot) = vj;
        }
        __syncwarp();
        const int nk = g + jt - 7;
        if (nk >= 0) kr[3] = *(const uint4*)&newkv[nk][0][16 * c];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int nv = 2 * c + e + jt - 7;
            if (nv >= 0) vr[6 + e] = *(const uint2*)&newkv[nv][1][8 * g];
        }
    }

    // S = Q K^T for row g: tile j holds keys 8j + 2c + {0, 1}.
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], q8[0], 0u, q8[1], 0u, kr[j].x, kr[j].y);
        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);
    }
    const float scale = qk_scale * kscale;
    float m = -1e30f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            bool ok = base + w0 + 8 * j + 2 * c + e >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            m = fmaxf(m, sc[j][e]);
        }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    const float mo = m - 8.f;  // FA3's FP8 max offset: P <= 2^8
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    const unsigned pb0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
    const unsigned pb1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);

    // O^T = V^T P^T: m-tile i holds d 8g + 2i (rows m = g) and 8g + 2i + 1 (m = g + 8).
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int bsel = (i & 1) ? 0x7362 : 0x5140;
        unsigned w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
        unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
        unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632), __byte_perm(t45, t67, 0x5410),
             __byte_perm(t45, t67, 0x7632), pb0, pb1);
    }

    // Pre-merge the 4 warps of each KV head in shared memory, then push one partial per (head, row) to its owner.
    {
        float* lp = &lpart[h][wk][0][0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = lp + (2 * c + e) * D + 8 * g;
            *(float4*)r = make_float4(acc[0][e], acc[0][e + 2], acc[1][e], acc[1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[2][e], acc[2][e + 2], acc[3][e], acc[3][e + 2]);
        }
        if (c == 0) *(float2*)&lml[h][wk][g][0] = make_float2(mo, l);
    }
    __syncthreads();
    {
        const int h2 = h, row = (tid >> 4) & 7, dc = tid & 15;
        float mk[4], lk[4], M = -1e30f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 v = *(const float2*)&lml[h2][k][row][0];
            mk[k] = v.x; lk[k] = v.y; M = fmaxf(M, v.x);
        }
        float L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float w = exp2f(mk[k] - M);
            L += w * lk[k];
            float4 a = *(const float4*)&lpart[h2][k][row][4 * dc];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
        }
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        const unsigned owner = row;
        const unsigned rmb = mapa(mb, owner);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[rank][4 * dc]), owner), rmb, o[0], o[1], o[2], o[3]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[rank][0]), owner), rmb, M, L);
    }

    // Merge query head `rank` from its 16 partials: one warp, two d per lane, no block barriers.
    if (warp == 0 && rank < 8) {
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            float w = exp2f(rml[s][0] - M);
            den += w * rml[s][1];
            float2 a = *(const float2*)&racc[s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
        float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        float ni = rsqrtf(s2 / D + 1e-5f);
        o0 *= ni / (1.f + __expf(-g0));
        o1 *= ni / (1.f + __expf(-g1));
        *(__nv_bfloat162*)(out + (long)tok * HQ * D + (h * 8 + rank) * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace vtk

void vtk_verify(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t pw) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && CSL_SPEC) ? mp - vtk::L : -1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(32 * batch * vtk::L);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pw ? 2 : 1;
#define VTK_ARGS(QT) (const QT*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, spec
#define VTK_LAUNCH1(P, QT, PW) { \
        static bool set = false; \
        if (!set) { cudaFuncSetAttribute(vtk::verify<P, QT, PW>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; } \
        cudaLaunchKernelEx(&cfg, vtk::verify<P, QT, PW>, VTK_ARGS(QT)); }
#define VTK_LAUNCH(P, QT) { if (pw) VTK_LAUNCH1(P, QT, true) else VTK_LAUNCH1(P, QT, false) }
    const bool f32 = qkv.scalar_type() == at::kFloat;
    TORCH_CHECK(page == 1 || page == 64, "vtk: page 1 or 64");
    if (page == 1) {
        if (f32) VTK_LAUNCH(1, float) else VTK_LAUNCH(1, unsigned short)
    } else {
        if (f32) VTK_LAUNCH(64, float) else VTK_LAUNCH(64, unsigned short)
    }
#undef VTK_LAUNCH
#undef VTK_LAUNCH1
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
try:
    _CSL = _load_inline(
        name="muse_csl_s96_" + __import__("hashlib").sha256(_CSL_SRC.encode()).hexdigest()[:10],
        cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t, int64_t, int64_t);"
                    "void vtk_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);",
        cuda_sources=_CSL_SRC, functions=["csl_decode", "vtk_verify"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("csl build failed, using Triton:", _e)
    _CSL = None
VTK_MAX_BATCH = 1  # verify sliding on vtk (one cluster per (request, token, KV head)): H100 B1 7.7 -> 6.45 us
CSL_MAX_BATCH = 16  # decode sliding up to this batch on the cluster kernel (H100 paired: B1 4.70 -> 3.84, B16 10.61 -> 9.14 us; B32 15.84 -> 15.94)

# agent2 s3: native verify-sliding attention (tools/a2/nvs.cu): one CTA per (request, key split), a producer warpgroup
# gathering both KV heads' 128-B rows with cp.async (+ the FA3-style FP8 V transpose) and one consumer warpgroup per
# KV head (FP8 wgmma QK^T, online softmax with FA3's offset-8 E4M3 P, RS wgmma P@V, QK(i) overlapping PV(i-1)). The
# last split's consumers first attend the block's 8 new tokens as one causal tensor-core tile (and store them in the
# cache) while the cached tiles' gathers are in flight. FP32 partials in _attn_split's layout, merged by _combine_hb.
_NVS_SRC = r'''// agent2 nvs: native verify-SLIDING attention (L 8, G 8, D 64, HKV 2, page 1), from tools/s71/nverify.cu.
// One CTA per (split, request): warpgroup 0 gathers both KV heads' K/V rows (128-B lines) with cp.async and
// transposes V to FP8 V^T; warpgroups 1/2 (KV head 0/1) normalize + RoPE their queries, run QK^T (FP8 wgmma),
// the online softmax and P@V with P in E4M3 (FA3's FP8 path), QK(i) overlapping PV(i-1). Partials in
// _attn_split's layout [B, HKV, S, 64, 64] FP32 for _combine_hb. NVS_NEW: split S-1 also attends the 8 new tokens.
#define NV_NS 5
#define NV_LAG 2
#define NV_PFD 4
#define NV_TR 1
#define NV_LAZY 0
#define NV_PREGS 40
#define NV_CREGS 160
#define NV_MINB 1
#define NV_HALF 1
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace nv {
constexpr int BN = 64, HD = 64, MR = 64;
#ifndef NV_LAG
#define NV_LAG 1
#endif
#ifndef NV_PFD
#define NV_PFD 4
#endif
#ifndef NV_TR
#define NV_TR 1
#endif
#ifndef NV_LAZY
#define NV_LAZY 1
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t mdesc(uint32_t addr, uint32_t sbo, uint32_t mode) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(sbo >> 4) << 32) | ((uint64_t)mode << 62);
}
__device__ __forceinline__ uint32_t swz64(uint32_t o) { return o ^ (((o >> 7) & 3) << 4); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_regs(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
#define D32(a) "+f"(a[0]), "+f"(a[1]), "+f"(a[2]), "+f"(a[3]), "+f"(a[4]), "+f"(a[5]), "+f"(a[6]), "+f"(a[7]), \
    "+f"(a[8]), "+f"(a[9]), "+f"(a[10]), "+f"(a[11]), "+f"(a[12]), "+f"(a[13]), "+f"(a[14]), "+f"(a[15]), \
    "+f"(a[16]), "+f"(a[17]), "+f"(a[18]), "+f"(a[19]), "+f"(a[20]), "+f"(a[21]), "+f"(a[22]), "+f"(a[23]), \
    "+f"(a[24]), "+f"(a[25]), "+f"(a[26]), "+f"(a[27]), "+f"(a[28]), "+f"(a[29]), "+f"(a[30]), "+f"(a[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
__device__ __forceinline__ void qk_k32(float (&s)[32], uint64_t da, uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", %32, %33, p, 1, 1;\n}\n"
                 : D32(s) : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ void pv8_k32(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n"
                 ::"r"(smem_u32(b)), "r"(par) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NS> struct Smem {
    alignas(1024) uint8_t k[NS][BN * 128];     // both heads' K rows, SW128 (wgmma B, K-major)
    alignas(1024) uint8_t vs[NS][BN * 128];    // V rows as gathered; 16-byte chunk c of row r at c ^ vsw(r)
    alignas(1024) uint8_t vt[NS][2][HD * BN];  // per head V^T [d][key slot], SW64 (wgmma B, K-major), keys permuted
    alignas(1024) uint8_t q[2][MR * HD];       // per head FP8 queries [row][d], SW64
    int ptr[8][(1 + NV_HALF) * BN];                            // page 1: page-table entries of tiles i .. i + NV_PFD (ring of 8)
    int lim[8];                                // and each tile's last valid key
    uint64_t ready[NS], empty[NS];
#if NV_HALF
    int lim2[8][2];
    float xo[128 * 36];                        // NV_HALF: consumer 1's o / m / l for the in-CTA merge
#endif
    alignas(1024) uint8_t knew[BN * 128];      // last split: the 8 new keys as a K tile (rows 0-7; rows 8-63 zero)
    alignas(1024) uint8_t vtnew[2][HD * BN];   // and their V^T (key slots 0-7 of group 0; the rest zero)
};
// Staging swizzle: SW128 (r & 7) for the ldmatrix transpose (8 consecutive rows per phase), else conflict-free for
// the 4-row register transpose.
__device__ __forceinline__ uint32_t vsw(int r) { return NV_TR ? (r & 7) : (((r >> 1) & 3) | (((r >> 4) & 1) << 2)); }


struct Range { int b, lo, hi, n, seq; };
__device__ __forceinline__ Range rng(const int* __restrict__ seq, int b, int split, int ch, int win) {
    Range r;
    r.b = b;
    r.seq = seq[b];
    const int base = win > 0 ? max(r.seq - (win - 1), 0) : 0;
    r.lo = base + split * ch;
    r.hi = min(r.lo + ch, r.seq);
    r.n = r.hi > r.lo ? (r.hi - r.lo + BN - 1) / BN : 0;
    return r;
}
__device__ __forceinline__ void cp4ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}

#if NV_HALF
// NV_HALF: the CTA serves one KV head H; smem row r of tile i holds key lo0 + 64 i + r of half 0 in bytes 0-63 and key
// lo1 + 64 i + r of half 1 in bytes 64-127 (both head H), so the consumer / transpose layout is the 2-head one with
// "head" = half.
template <int NS, int LAG>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, Range r0, Range r1, int H, int maxp) {
    const int tid = threadIdx.x;  // 0..127
    const int total = r0.n;
    const int* ptr = pt + (long long)r0.b * maxp;
    auto stage = [&](int j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const Range& rg = c ? r1 : r0;
            const int kb = rg.lo + j * BN;
            if (kb + tid < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][c * BN + tid]), ptr + kb + tid);
            if (kb + tid + 32 < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][c * BN + tid + 32]), ptr + kb + tid + 32);
            if (tid == 0) sm.lim2[j & 7][c] = rg.hi - 1 - kb;
        }
    };
    if (tid < 32)
        for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
    cp_commit();
    cp_wait<0>();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int grow = tid >> 3;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    const int lane = tid & 31;
    const int ch = tid & 7, cc = ch >> 2;
    const int nmine = cc ? r1.n : r0.n;
    const uint8_t* kcb = kc + H * HD + (ch & 3) * 16;
    const uint8_t* vcb = vc + H * HD + (ch & 3) * 16;
    for (int i = 0; i < total + LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            int cr[4];
            if (i < nmine) {
                const int lim = sm.lim2[i & 7][cc];
#pragma unroll
                for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][cc * BN + min(grow + 16 * j, lim)];
            }
            if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
            mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
            if (i < nmine) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = grow + 16 * j;
                    const long long off = (long long)cr[j] * 128;
                    cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kcb + off);
                    cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vcb + off);
                }
            }
        }
#else
template <int NS, int LAG>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, Range rg, int maxp) {
    const int tid = threadIdx.x;  // 0..127
    const int total = rg.n;
    const int* ptr = pt + (long long)rg.b * maxp;
    // Warp 0 stages each tile's page-table entries NV_PFD tiles ahead in sm.ptr (2 x 4-byte cp.async per lane: the
    // window start is not 16-byte aligned).
    auto stage = [&](int j) {
        const int kb = rg.lo + j * BN;
        if (kb + tid < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid]), ptr + kb + tid);
        if (kb + tid + 32 < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid + 32]), ptr + kb + tid + 32);
        if (tid == 0) sm.lim[j & 7] = rg.hi - 1 - kb;
    };
    if (tid < 32)
        for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
    cp_commit();
    cp_wait<0>();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int grow = tid >> 3;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    const int lane = tid & 31;
    for (int i = 0; i < total + LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            const int lim = sm.lim[i & 7];
            int cr[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][min(grow + 16 * j, lim)];
            if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
            mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
            const int ch = tid & 7;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = grow + 16 * j;
                const long long off = (long long)cr[j] * 128 + ch * 16;
                cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kc + off);
                cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vc + off);
            }
        }
#endif
        cp_commit();
        if (i >= LAG) {
            const int s = (i - LAG) % NS;
            cp_wait<LAG>();
            asm volatile("bar.sync 1, 128;" ::: "memory");
            {
                const uint32_t vsb = smem_u32(&sm.vs[s][0]), vtb = smem_u32(&sm.vt[s][0][0]);
                const int wq = tid >> 5, mi = lane >> 3, ri = lane & 7;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int u = wq * 4 + q, gg = u & 3, hc = u >> 2;
                    const int r = 16 * gg + (mi & 1) * 8 + ri, cc = 2 * hc + (mi >> 1);
                    uint32_t R0, R1, R2, R3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(vsb + r * 128 + ((cc ^ (r & 7)) << 4)));
                    const uint32_t S0 = prmt(R0, R1, 0x6420), S1 = prmt(R0, R1, 0x7531);
                    const uint32_t S2 = prmt(R2, R3, 0x6420), S3 = prmt(R2, R3, 0x7531);
                    const int d = 32 * (hc & 1) + 16 * (mi >> 1) + 2 * ri + (mi & 1);
                    const uint32_t o = (uint32_t)(hc >> 1) * (HD * BN) + swz64(d * 64 + 16 * gg);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                                 :: "r"(vtb + o), "r"(S0), "r"(S1), "r"(S2), "r"(S3) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);
        }
    }
}

__device__ __forceinline__ void softmax(float (&s)[32], float (&m)[2], float (&l)[2], uint32_t (&p)[8],
                                        float (&alpha)[2], float scale) {
    float mx0 = s[0], mx1 = s[2];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        mx0 = fmaxf(mx0, fmaxf(s[j * 4 + 0], s[j * 4 + 1]));
        mx1 = fmaxf(mx1, fmaxf(s[j * 4 + 2], s[j * 4 + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    float mn0 = fmaxf(m[0], mx0 * scale), mn1 = fmaxf(m[1], mx1 * scale);
    if (NV_LAZY) {
        bool grow = (mn0 > m[0] + 8.f) || (mn1 > m[1] + 8.f);
        if (!__any_sync(0xffffffff, grow)) { mn0 = m[0]; mn1 = m[1]; }
    }
    alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    const float off = NV_LAZY ? 0.f : 8.f;
    const float o0 = mn0 - off, o1 = mn1 - off;
    float e[32], l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        e[j * 4 + 0] = ex2(fmaf(s[j * 4 + 0], scale, -o0)); e[j * 4 + 1] = ex2(fmaf(s[j * 4 + 1], scale, -o0));
        e[j * 4 + 2] = ex2(fmaf(s[j * 4 + 2], scale, -o1)); e[j * 4 + 3] = ex2(fmaf(s[j * 4 + 3], scale, -o1));
        l0 += e[j * 4 + 0] + e[j * 4 + 1]; l1 += e[j * 4 + 2] + e[j * 4 + 3];
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
        const int b0 = 16 * kk;
        p[kk * 4 + 0] = pack_e4(e[b0 + 0], e[b0 + 1], e[b0 + 4], e[b0 + 5]);
        p[kk * 4 + 1] = pack_e4(e[b0 + 2], e[b0 + 3], e[b0 + 6], e[b0 + 7]);
        p[kk * 4 + 2] = pack_e4(e[b0 + 8], e[b0 + 9], e[b0 + 12], e[b0 + 13]);
        p[kk * 4 + 3] = pack_e4(e[b0 + 10], e[b0 + 11], e[b0 + 14], e[b0 + 15]);
    }
    l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1;
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2]) {
    if (NV_LAZY && __all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}
__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[8], uint32_t va) {
    wg_fence();
    pv8_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
    pv8_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
    wg_commit();
}
// Score masks: key column col (of this tile) is out if col >= nvalid (past the split's end) or col < thr0 / thr1
// (rows of token 2w / 2w + 1: before the sliding window's start).
__device__ __forceinline__ void mask(float (&s)[32], int lane, int nvalid, int thr0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid || col < thr0) s[j * 4 + 0] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0) s[j * 4 + 1] = -INFINITY;
        if (col >= nvalid || col < thr0 + 1) s[j * 4 + 2] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0 + 1) s[j * 4 + 3] = -INFINITY;
    }
}

// layer s72 (QSS at verify sliding): qkv may be 2 FP32 split-K slabs qss elements apart (one batched GEMM, no
// split-K reduce kernel): the partials are added in FP32 and rounded to BF16 (RN) at the load (= splitKreduce + BF16 out).
struct qs2 { float v; };
__device__ __forceinline__ unsigned qpk(float lo, float hi) {
    __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
    return *(unsigned*)&t;
}
__device__ __forceinline__ uint4 qld8(const __nv_bfloat16* p, int) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint4 qld8(const qs2* q, int qss) {
    const float* p = reinterpret_cast<const float*>(q);
    const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    const float4 c = reinterpret_cast<const float4*>(p + qss)[0], d = reinterpret_cast<const float4*>(p + qss)[1];
    return make_uint4(qpk(a.x + c.x, a.y + c.y), qpk(a.z + c.z, a.w + c.w), qpk(b.x + d.x, b.y + d.y), qpk(b.z + d.z, b.w + d.w));
}
__device__ __forceinline__ uint2 qld4(const __nv_bfloat16* p, int) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ uint2 qld4(const qs2* q, int qss) {
    const float* p = reinterpret_cast<const float*>(q);
    const float4 a = *reinterpret_cast<const float4*>(p), c = *reinterpret_cast<const float4*>(p + qss);
    return make_uint2(qpk(a.x + c.x, a.y + c.y), qpk(a.z + c.z, a.w + c.w));
}
template <int NS, typename QT>
__device__ __forceinline__ void consumer(Smem<NS>& sm, int h, const QT* __restrict__ qkv,
                                         const float* __restrict__ cs, Range rg, int split, int S, int qkv_w,
                                         float scale, int win, float* __restrict__ pacc, float* __restrict__ pml,
                                         uint8_t* __restrict__ kc,
                                         uint8_t* __restrict__ vc, const long long* __restrict__ loc, float ksv,
                                         float vsv, int H, int total, int qss) {
    // NV_HALF: h is the consumer's half (smem slot) of KV head H's split; else h is the KV head (H unused).
    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;
    const int HH = NV_HALF ? H : h;
    const bool isnew = NV_HALF ? (split == S - 1 && h == 1) : split == S - 1;
    const uint32_t qa = smem_u32(&sm.q[h][0]);
    const int n = rg.n, lo = rg.lo, hi = rg.hi, b = rg.b;
    {   // Queries: row r = token r / 8, head h * 8 + r % 8; 2 threads per row (32 d each); RMS-norm, RoPE, FP8.
        const int r = tid >> 1, half = tid & 1;
        const int pos = rg.seq + (r >> 3);
        const QT* src = qkv + (long long)(b * 8 + (r >> 3)) * qkv_w + (HH * 8 + (r & 7)) * HD + half * 32;
        uint4 raw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) raw[j] = qld8(src + 8 * j, qss);
        float4 cv[4], sv[4];
        const float4* cp = reinterpret_cast<const float4*>(cs + (long long)pos * HD + half * 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) { cv[j] = cp[j]; sv[j] = cp[j + 8]; }
        float x[32], ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&raw[j]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(hp[k]);
                x[j * 8 + 2 * k] = f.x; x[j * 8 + 2 * k + 1] = f.y;
                ss += f.x * f.x + f.y * f.y;
            }
        }
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        const float rs = rsqrtf(ss / HD + 1e-5f);
        const float* cf = reinterpret_cast<const float*>(cv);
        const float* sf = reinterpret_cast<const float*>(sv);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float a = x[2 * i], bb = x[2 * i + 1];
            x[2 * i] = (a * cf[i] - bb * sf[i]) * rs;
            x[2 * i + 1] = (bb * cf[i] + a * sf[i]) * rs;
        }
        uint32_t pk[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) pk[j] = pack_e4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
        if (isnew) {
            // The block's new token t = tid / 16, d 4 (tid % 16) + 0..3 of this KV head: RMS-norm + RoPE, E4M3 (as
            // _new), stored into the cache and into the new-token K / V^T tiles.
            const int t = tid >> 4, d0 = (tid & 15) * 4, tok = b * 8 + t, npos = rg.seq + t;
            const QT* kp = qkv + (long long)tok * qkv_w + (16 + HH) * HD + d0;
            const uint2 kr = qld4(kp, qss), vr = qld4(kp + 2 * HD, qss);
            const float2 c2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + d0 / 2);
            const float2 s2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + 32 + d0 / 2);
            const long long slot = loc[tok];
            const float2 k01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.x));
            const float2 k23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.y));
            const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.x));
            const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.y));
            float kss = k01.x * k01.x + k01.y * k01.y + k23.x * k23.x + k23.y * k23.y;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) kss += __shfl_xor_sync(0xffffffff, kss, o);
            const float krs = rsqrtf(kss / HD + 1e-5f);
            const uint32_t kq = pack_e4((k01.x * c2.x - k01.y * s2.x) * krs / ksv, (k01.y * c2.x + k01.x * s2.x) * krs / ksv,
                                        (k23.x * c2.y - k23.y * s2.y) * krs / ksv, (k23.y * c2.y + k23.x * s2.y) * krs / ksv);
            const uint32_t vq = pack_e4(v01.x / vsv, v01.y / vsv, v23.x / vsv, v23.y / vsv);
            *reinterpret_cast<uint32_t*>(kc + slot * 128 + HH * HD + d0) = kq;
            *reinterpret_cast<uint32_t*>(vc + slot * 128 + HH * HD + d0) = vq;
            {
                // Zero the tiles' unused rows / slots, then place this lane's 4 key bytes (row t, SW128 chunk) and
                // 4 value bytes (V^T rows d0 .. d0 + 3, key slot of t in group 0: word t / 2, byte t % 2).
                const uint4 z = make_uint4(0, 0, 0, 0);
                uint4* vz = reinterpret_cast<uint4*>(&sm.vtnew[h][0]);
                vz[tid] = z; vz[tid + 128] = z;
#if NV_HALF
                {
                    uint4* kz = reinterpret_cast<uint4*>(&sm.knew[8 * 128]);
                    for (int e = tid; e < 448; e += 128) kz[e] = z;
                }
#else
                if (tid < 112) {
                    uint4* kz = reinterpret_cast<uint4*>(&sm.knew[(8 + 28 * h) * 128]);
                    kz[tid] = z; kz[tid + 112] = z;
                }
#endif
                asm volatile("bar.sync %0, 128;" ::"r"(2 + h) : "memory");
                const int c = h * 4 + (d0 >> 4);
                *reinterpret_cast<uint32_t*>(&sm.knew[t * 128 + ((c ^ (t & 7)) << 4) + (d0 & 15)]) = kq;
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    sm.vtnew[h][swz64((d0 + e) * 64) + 4 * (t >> 1) + (t & 1)] = (uint8_t)(vq >> (8 * e));
            }
        }
        uint8_t* qd = &sm.q[h][0];
        *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32 + 16)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync %0, 128;" ::"r"(2 + h) : "memory");
    }
    float o[32], s[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
    uint32_t pa[8], pb[8];
    // Window start for this warp's token 2w rows, relative to key lo: keys t < seq - (win - 1) + 2w are out.
    const int thr = win > 0 ? rg.seq - (win - 1) + 2 * w - lo : -(1 << 30);
    bool newdone = false;
    if (isnew) {
        // The new tokens first, as one causal 8-key tile on the tensor cores, while the cached tiles' gathers fly.
        issue_qk(s, qa, smem_u32(&sm.knew[0]) + h * 64);
        wg_wait<0>();
        fence_regs(s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + (lane & 3) * 2;
            if (col > 2 * w) s[j * 4 + 0] = -INFINITY;
            if (col + 1 > 2 * w) s[j * 4 + 1] = -INFINITY;
            if (col > 2 * w + 1) s[j * 4 + 2] = -INFINITY;
            if (col + 1 > 2 * w + 1) s[j * 4 + 3] = -INFINITY;
        }
        softmax(s, m, l, pa, alpha, scale);
        fence_regs(pa);
        issue_pv(o, pa, smem_u32(&sm.vtnew[h][0]));
        wg_wait<0>();
        fence_regs(o);
        newdone = true;
    }
    if (n > 0) {
        mb_wait(&sm.ready[0], 0);
        issue_qk(s, qa, smem_u32(&sm.k[0][0]) + h * 64);
        wg_wait<0>();
        fence_regs(s);
        if (hi - lo < BN || thr + 1 > 0) mask(s, lane, hi - lo, thr);
        softmax(s, m, l, pa, alpha, scale);
        if (newdone) { rescale(o, alpha); fence_regs(o); }
        auto step = [&](uint32_t (&pin)[8], uint32_t (&pout)[8], int i) {
            const int sp = (i - 1) % NS, sc = i % NS;
            mb_wait(&sm.ready[sc], (i / NS) & 1);
            issue_qk(s, qa, smem_u32(&sm.k[sc][0]) + h * 64);
            fence_regs(pin);
            issue_pv(o, pin, smem_u32(&sm.vt[sp][h][0]));
            wg_wait<1>();
            fence_regs(s);
            const int nv = hi - lo - i * BN;
            if (nv < BN || thr + 1 - i * BN > 0) mask(s, lane, nv, thr - i * BN);
            softmax(s, m, l, pout, alpha, scale);
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
            rescale(o, alpha);
            fence_regs(o);
        };
        auto last = [&](uint32_t (&pin)[8]) {
            const int sp = (n - 1) % NS;
            fence_regs(pin);
            issue_pv(o, pin, smem_u32(&sm.vt[sp][h][0]));
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
        };
        int i = 1;
        for (; i + 1 < n; i += 2) { step(pa, pb, i); step(pb, pa, i + 1); }
        if (i < n) { step(pa, pb, i); last(pb); } else last(pa);
    }
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
#if NV_HALF
    // Release the producer's tiles this half has no keys for, then merge half 1 into half 0 through smem.
    for (int i = n; i < total; ++i) {
        mb_wait(&sm.ready[i % NS], (i / NS) & 1);
        if (lane == 0) mb_arrive(&sm.empty[i % NS]);
    }
    if (h == 1) {
        float4* xo = reinterpret_cast<float4*>(&sm.xo[tid * 36]);
#pragma unroll
        for (int j = 0; j < 8; ++j) xo[j] = make_float4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
        xo[8] = make_float4(m[0], m[1], l[0], l[1]);
        asm volatile("bar.sync 4, 256;" ::: "memory");
        return;
    }
    asm volatile("bar.sync 4, 256;" ::: "memory");
    {
        const float4* xo = reinterpret_cast<const float4*>(&sm.xo[tid * 36]);
        const float4 x = xo[8];
        const float mn0 = fmaxf(m[0], x.x), mn1 = fmaxf(m[1], x.y);
        const float a0 = ex2(m[0] - mn0), b0 = ex2(x.x - mn0), a1 = ex2(m[1] - mn1), b1 = ex2(x.y - mn1);
        m[0] = mn0; m[1] = mn1;
        l[0] = l[0] * a0 + x.z * b0; l[1] = l[1] * a1 + x.w * b1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 y = xo[j];
            o[4 * j + 0] = o[4 * j + 0] * a0 + y.x * b0; o[4 * j + 1] = o[4 * j + 1] * a0 + y.y * b0;
            o[4 * j + 2] = o[4 * j + 2] * a1 + y.z * b1; o[4 * j + 3] = o[4 * j + 3] * a1 + y.w * b1;
        }
    }
    const long long part = (((long long)b * 2 + H) * S + split) * MR;
#else
    const long long part = (((long long)b * 2 + h) * S + split) * MR;
#endif
    const int ra = w * 16 + (lane >> 2);
    float* pa0 = pacc + (part + ra) * HD;
    float* pa1 = pa0 + 8 * HD;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        *reinterpret_cast<float2*>(pa0 + col) = make_float2(o[j * 4 + 0], o[j * 4 + 1]);
        *reinterpret_cast<float2*>(pa1 + col) = make_float2(o[j * 4 + 2], o[j * 4 + 3]);
    }
    if ((lane & 3) == 0) {
        const float off = NV_LAZY ? 0.f : 8.f;
        *reinterpret_cast<float2*>(pml + (part + ra) * 2) = make_float2(m[0] - off, l[0]);
        *reinterpret_cast<float2*>(pml + (part + ra + 8) * 2) = make_float2(m[1] - off, l[1]);
    }
}

template <int NS, int LAG, typename QT>
__global__ void __launch_bounds__(384, NV_MINB) nvs(const QT* __restrict__ qkv, const float* __restrict__ cs,
                                              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                              const float* __restrict__ ks, const float* __restrict__ vsc,
                                              const long long* __restrict__ loc, const int* __restrict__ pt,
                                              const int* __restrict__ seq, float* __restrict__ pacc,
                                              float* __restrict__ pml, float qk_scale, int S, int ch, int maxp,
                                              int qkv_w, int win, const uint8_t* __restrict__ pf, long long pfb, int qss) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x, wg = tid >> 7;
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mb_init(&sm.ready[i], 4); mb_init(&sm.empty[i], 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int split = blockIdx.x % S;
#if NV_HALF
    // CTA (request, KV head H, split); consumer c takes half c of the split's ch keys.
    const int H = (blockIdx.x / S) & 1, bb = blockIdx.x / (2 * S);
    const Range r0 = rng(seq, bb, 2 * split, ch / 2, win), r1 = rng(seq, bb, 2 * split + 1, ch / 2, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS, LAG>(sm, kc, vc, pt, r0, r1, H, maxp);
        if (pfb > 0 && tid < 32) {  // layer s34 (O_PF): after this CTA's last KV load is issued, L2-prefetch its share of the next GEMM's weight
            const long long per = ((pfb / gridDim.x) + 4095) & ~4095LL, end = min(pfb, per * (blockIdx.x + 1));
            for (long long o = per * blockIdx.x + tid * 16384LL; o < end; o += 32 * 16384LL)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pf + o), "r"((int)min(16384LL, end - o)) : "memory");
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS, QT>(sm, wg - 1, qkv, cs, wg == 1 ? r0 : r1, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc,
                     loc, ks[0], vsc[0], H, r0.n, qss);
    }
#else
    const Range rg = rng(seq, blockIdx.x / S, split, ch, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS, LAG>(sm, kc, vc, pt, rg, maxp);
        if (pfb > 0 && tid < 32) {  // layer s34 (O_PF): after this CTA's last KV load is issued, L2-prefetch its share of the next GEMM's weight
            const long long per = ((pfb / gridDim.x) + 4095) & ~4095LL, end = min(pfb, per * (blockIdx.x + 1));
            for (long long o = per * blockIdx.x + tid * 16384LL; o < end; o += 32 * 16384LL)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pf + o), "r"((int)min(16384LL, end - o)) : "memory");
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS, QT>(sm, wg - 1, qkv, cs, rg, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc, loc, ks[0], vsc[0], 0, 0, qss);
    }
#endif
}
}  // namespace nv

void nvs(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vsc,
         torch::Tensor loc, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, double qk_scale, int64_t S, int64_t ch, int64_t win,
         int64_t pf_ptr, int64_t pf_bytes, int64_t qss) {
    // Producer lag NV_LAG (1) at every S: LAG 2 at 4 splits was H100 -0.18 us but H200 +0.2 (check 9249146d, verify 32).
    // qss > 0 (layer s72): qkv = FP32 slab 0 of 2 split-K slabs qss elements apart, summed at the loads.
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    const int B = (int)seq.size(0);
#define NVS_LAUNCH(QT_) { \
        auto f = nv::nvs<NV_NS, NV_LAG, QT_>; \
        static bool once = false; \
        if (!once) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; } \
        f<<<dim3((unsigned)(S * B * (1 + NV_HALF))), 384, smem, c10::cuda::getCurrentCUDAStream()>>>( \
            (const QT_*)qkv.data_ptr(), (const float*)cs.data_ptr(), (uint8_t*)kc.data_ptr(), \
            (uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const float*)vsc.data_ptr(), \
            (const long long*)loc.data_ptr(), (const int*)pt.data_ptr(), \
            (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale, (int)S, (int)ch, \
            (int)pt.size(1), (int)qkv.size(-1), (int)win, (const uint8_t*)pf_ptr, (long long)pf_bytes, (int)qss); }
    if (qss) { TORCH_CHECK(qkv.scalar_type() == at::kFloat, "nvs: QSS needs FP32 slabs"); NVS_LAUNCH(nv::qs2) }
    else { TORCH_CHECK(qkv.scalar_type() == at::kBFloat16, "nvs: qkv bf16"); NVS_LAUNCH(__nv_bfloat16) }
}
'''
try:
    _NVS = _load_inline(
        name="muse_nvs_a2s13h_" + __import__("hashlib").sha256(_NVS_SRC.encode()).hexdigest()[:10],
        cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, "
                    "int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_NVS_SRC, functions=["nvs"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("nvs build failed, using Triton:", _e)
    _NVS = None
NVS_HALF = '#define NV_HALF 1' in _NVS_SRC
# moe s31 (r-moe 6c4fee3, agent2 halfpfkr): verify-32 sliding nvs with the 2 splits of each (request, KV head) as one
# 2-CTA cluster merged over DSMEM (writes out; no _combine_hb). H200 standalone v32 sl 15.51 -> 14.07/14.39 us (moe checks
# 362ff219/18cf3bfb). BF16 qkv, no O_PF / QSS: used by _verify_nvs only when both are off (their T=256 defaults).
_NVSCL_SRC = r'''// agent2 nvs: native verify-SLIDING attention (L 8, G 8, D 64, HKV 2, page 1), from tools/s71/nverify.cu.
// One CTA per (split, request): warpgroup 0 gathers both KV heads' K/V rows (128-B lines) with cp.async and
// transposes V to FP8 V^T; warpgroups 1/2 (KV head 0/1) normalize + RoPE their queries, run QK^T (FP8 wgmma),
// the online softmax and P@V with P in E4M3 (FA3's FP8 path), QK(i) overlapping PV(i-1). Partials in
// _attn_split's layout [B, HKV, S, 64, 64] FP32 for _combine_hb. NVS_NEW: split S-1 also attends the 8 new tokens.
#define NV_NS 5
#define NV_LAG 2
#define NV_PFD 4
#define NV_TR 1
#define NV_LAZY 0
#define NV_PREGS 40
#define NV_CREGS 160
#define NV_MINB 1
#define NV_HALF 1
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace nv {
constexpr int BN = 64, HD = 64, MR = 64;
#ifndef NV_LAG
#define NV_LAG 1
#endif
#ifndef NV_PFD
#define NV_PFD 4
#endif
#ifndef NV_TR
#define NV_TR 1
#endif
#ifndef NV_LAZY
#define NV_LAZY 1
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t mdesc(uint32_t addr, uint32_t sbo, uint32_t mode) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(sbo >> 4) << 32) | ((uint64_t)mode << 62);
}
#ifndef NV_STAMP
#define NV_STAMP 0
#endif
__device__ __forceinline__ void stamp(float* pml, int k) {
#if NV_STAMP
    if ((threadIdx.x & 127) == 0) {
        uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        reinterpret_cast<unsigned long long*>(pml)[blockIdx.x * 8 + k] = t;
    }
#endif
}
__device__ __forceinline__ uint32_t swz64(uint32_t o) { return o ^ (((o >> 7) & 3) << 4); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_regs(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
#define D32(a) "+f"(a[0]), "+f"(a[1]), "+f"(a[2]), "+f"(a[3]), "+f"(a[4]), "+f"(a[5]), "+f"(a[6]), "+f"(a[7]), \
    "+f"(a[8]), "+f"(a[9]), "+f"(a[10]), "+f"(a[11]), "+f"(a[12]), "+f"(a[13]), "+f"(a[14]), "+f"(a[15]), \
    "+f"(a[16]), "+f"(a[17]), "+f"(a[18]), "+f"(a[19]), "+f"(a[20]), "+f"(a[21]), "+f"(a[22]), "+f"(a[23]), \
    "+f"(a[24]), "+f"(a[25]), "+f"(a[26]), "+f"(a[27]), "+f"(a[28]), "+f"(a[29]), "+f"(a[30]), "+f"(a[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
__device__ __forceinline__ void qk_k32(float (&s)[32], uint64_t da, uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", %32, %33, p, 1, 1;\n}\n"
                 : D32(s) : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ void pv8_k32(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n"
                 ::"r"(smem_u32(b)), "r"(par) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NS> struct Smem {
    alignas(1024) uint8_t k[NS][BN * 128];     // both heads' K rows, SW128 (wgmma B, K-major)
    alignas(1024) uint8_t vs[NS][BN * 128];    // V rows as gathered; 16-byte chunk c of row r at c ^ vsw(r)
    alignas(1024) uint8_t vt[NS][2][HD * BN];  // per head V^T [d][key slot], SW64 (wgmma B, K-major), keys permuted
    alignas(1024) uint8_t q[2][MR * HD];       // per head FP8 queries [row][d], SW64
    int ptr[8][(1 + NV_HALF) * BN];                            // page 1: page-table entries of tiles i .. i + NV_PFD (ring of 8)
    int lim[8];                                // and each tile's last valid key
    uint64_t ready[NS], empty[NS];
#if NV_HALF
    int lim2[8][2];
    float xo[128 * 36];                        // NV_HALF: consumer 1's o / m / l for the in-CTA merge
    alignas(128) float xo2[128 * 36];                       // cl: the cluster's rank-1 CTA's merged o / m / l (DSMEM push)
    uint64_t clbar;                            // cl: rank 1's 128 remote arrivals after that push
#endif
    alignas(1024) uint8_t knew[BN * 128];      // last split: the 8 new keys as a K tile (rows 0-7; rows 8-63 zero)
    alignas(1024) uint8_t vtnew[2][HD * BN];   // and their V^T (key slots 0-7 of group 0; the rest zero)
};
// Staging swizzle: SW128 (r & 7) for the ldmatrix transpose (8 consecutive rows per phase), else conflict-free for
// the 4-row register transpose.
__device__ __forceinline__ uint32_t vsw(int r) { return NV_TR ? (r & 7) : (((r >> 1) & 3) | (((r >> 4) & 1) << 2)); }


struct Range { int b, lo, hi, n, seq; };
__device__ __forceinline__ Range rng(const int* __restrict__ seq, int b, int split, int ch, int win) {
    Range r;
    r.b = b;
    r.seq = seq[b];
    const int base = win > 0 ? max(r.seq - (win - 1), 0) : 0;
    r.lo = base + split * ch;
    r.hi = min(r.lo + ch, r.seq);
    r.n = r.hi > r.lo ? (r.hi - r.lo + BN - 1) / BN : 0;
    return r;
}
__device__ __forceinline__ void cp4ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}

#if NV_HALF
// NV_HALF: the CTA serves one KV head H; smem row r of tile i holds key lo0 + 64 i + r of half 0 in bytes 0-63 and key
// lo1 + 64 i + r of half 1 in bytes 64-127 (both head H), so the consumer / transpose layout is the 2-head one with
// "head" = half.
template <int NS, int LAG>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, Range r0, Range r1, int H, int maxp) {
    const int tid = threadIdx.x;  // 0..127
    const int total = r0.n;
    const int* ptr = pt + (long long)r0.b * maxp;
    auto stage = [&](int j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const Range& rg = c ? r1 : r0;
            const int kb = rg.lo + j * BN;
            if (kb + tid < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][c * BN + tid]), ptr + kb + tid);
            if (kb + tid + 32 < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][c * BN + tid + 32]), ptr + kb + tid + 32);
            if (tid == 0) sm.lim2[j & 7][c] = rg.hi - 1 - kb;
        }
    };
    if (tid < 32)
        for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
    cp_commit();
    cp_wait<0>();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int grow = tid >> 3;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    const int lane = tid & 31;
    const int ch = tid & 7, cc = ch >> 2;
    const int nmine = cc ? r1.n : r0.n;
    const uint8_t* kcb = kc + H * HD + (ch & 3) * 16;
    const uint8_t* vcb = vc + H * HD + (ch & 3) * 16;
    for (int i = 0; i < total + LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            int cr[4];
            if (i < nmine) {
                const int lim = sm.lim2[i & 7][cc];
#pragma unroll
                for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][cc * BN + min(grow + 16 * j, lim)];
            }
            if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
            mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
            if (i < nmine) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = grow + 16 * j;
                    const long long off = (long long)cr[j] * 128;
                    cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kcb + off);
                    cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vcb + off);
                }
            }
        }
#else
template <int NS, int LAG>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, Range rg, int maxp) {
    const int tid = threadIdx.x;  // 0..127
    const int total = rg.n;
    const int* ptr = pt + (long long)rg.b * maxp;
    // Warp 0 stages each tile's page-table entries NV_PFD tiles ahead in sm.ptr (2 x 4-byte cp.async per lane: the
    // window start is not 16-byte aligned).
    auto stage = [&](int j) {
        const int kb = rg.lo + j * BN;
        if (kb + tid < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid]), ptr + kb + tid);
        if (kb + tid + 32 < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid + 32]), ptr + kb + tid + 32);
        if (tid == 0) sm.lim[j & 7] = rg.hi - 1 - kb;
    };
    if (tid < 32)
        for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
    cp_commit();
    cp_wait<0>();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int grow = tid >> 3;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    const int lane = tid & 31;
    for (int i = 0; i < total + LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            const int lim = sm.lim[i & 7];
            int cr[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][min(grow + 16 * j, lim)];
            if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
            mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
            const int ch = tid & 7;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = grow + 16 * j;
                const long long off = (long long)cr[j] * 128 + ch * 16;
                cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kc + off);
                cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vc + off);
            }
        }
#endif
        cp_commit();
        if (i >= LAG) {
            const int s = (i - LAG) % NS;
            cp_wait<LAG>();
            asm volatile("bar.sync 1, 128;" ::: "memory");
            {
                const uint32_t vsb = smem_u32(&sm.vs[s][0]), vtb = smem_u32(&sm.vt[s][0][0]);
                const int wq = tid >> 5, mi = lane >> 3, ri = lane & 7;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int u = wq * 4 + q, gg = u & 3, hc = u >> 2;
                    const int r = 16 * gg + (mi & 1) * 8 + ri, cc = 2 * hc + (mi >> 1);
                    uint32_t R0, R1, R2, R3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(vsb + r * 128 + ((cc ^ (r & 7)) << 4)));
                    const uint32_t S0 = prmt(R0, R1, 0x6420), S1 = prmt(R0, R1, 0x7531);
                    const uint32_t S2 = prmt(R2, R3, 0x6420), S3 = prmt(R2, R3, 0x7531);
                    const int d = 32 * (hc & 1) + 16 * (mi >> 1) + 2 * ri + (mi & 1);
                    const uint32_t o = (uint32_t)(hc >> 1) * (HD * BN) + swz64(d * 64 + 16 * gg);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                                 :: "r"(vtb + o), "r"(S0), "r"(S1), "r"(S2), "r"(S3) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);
        }
    }
}

__device__ __forceinline__ void softmax(float (&s)[32], float (&m)[2], float (&l)[2], uint32_t (&p)[8],
                                        float (&alpha)[2], float scale) {
    float mx0 = s[0], mx1 = s[2];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        mx0 = fmaxf(mx0, fmaxf(s[j * 4 + 0], s[j * 4 + 1]));
        mx1 = fmaxf(mx1, fmaxf(s[j * 4 + 2], s[j * 4 + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    float mn0 = fmaxf(m[0], mx0 * scale), mn1 = fmaxf(m[1], mx1 * scale);
    if (NV_LAZY) {
        bool grow = (mn0 > m[0] + 8.f) || (mn1 > m[1] + 8.f);
        if (!__any_sync(0xffffffff, grow)) { mn0 = m[0]; mn1 = m[1]; }
    }
    alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    const float off = NV_LAZY ? 0.f : 8.f;
    const float o0 = mn0 - off, o1 = mn1 - off;
    float e[32], l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        e[j * 4 + 0] = ex2(fmaf(s[j * 4 + 0], scale, -o0)); e[j * 4 + 1] = ex2(fmaf(s[j * 4 + 1], scale, -o0));
        e[j * 4 + 2] = ex2(fmaf(s[j * 4 + 2], scale, -o1)); e[j * 4 + 3] = ex2(fmaf(s[j * 4 + 3], scale, -o1));
        l0 += e[j * 4 + 0] + e[j * 4 + 1]; l1 += e[j * 4 + 2] + e[j * 4 + 3];
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
        const int b0 = 16 * kk;
        p[kk * 4 + 0] = pack_e4(e[b0 + 0], e[b0 + 1], e[b0 + 4], e[b0 + 5]);
        p[kk * 4 + 1] = pack_e4(e[b0 + 2], e[b0 + 3], e[b0 + 6], e[b0 + 7]);
        p[kk * 4 + 2] = pack_e4(e[b0 + 8], e[b0 + 9], e[b0 + 12], e[b0 + 13]);
        p[kk * 4 + 3] = pack_e4(e[b0 + 10], e[b0 + 11], e[b0 + 14], e[b0 + 15]);
    }
    l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1;
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2]) {
    if (NV_LAZY && __all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}
__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[8], uint32_t va) {
    wg_fence();
    pv8_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
    pv8_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
    wg_commit();
}
// Score masks: key column col (of this tile) is out if col >= nvalid (past the split's end) or col < thr0 / thr1
// (rows of token 2w / 2w + 1: before the sliding window's start).
__device__ __forceinline__ void mask(float (&s)[32], int lane, int nvalid, int thr0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid || col < thr0) s[j * 4 + 0] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0) s[j * 4 + 1] = -INFINITY;
        if (col >= nvalid || col < thr0 + 1) s[j * 4 + 2] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0 + 1) s[j * 4 + 3] = -INFINITY;
    }
}

template <int NS>
__device__ __forceinline__ void consumer(Smem<NS>& sm, int h, const __nv_bfloat16* __restrict__ qkv,
                                         const float* __restrict__ cs, Range rg, int split, int S, int qkv_w,
                                         float scale, int win, float* __restrict__ pacc, float* __restrict__ pml,
                                         uint8_t* __restrict__ kc,
                                         uint8_t* __restrict__ vc, const long long* __restrict__ loc, float ksv,
                                         float vsv, int H, int total, int cl, __nv_bfloat16* __restrict__ out) {
    // NV_HALF: h is the consumer's half (smem slot) of KV head H's split; else h is the KV head (H unused).
    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;
    const int HH = NV_HALF ? H : h;
    const bool isnew = NV_HALF ? (split == S - 1 && h == 1) : split == S - 1;
    const uint32_t qa = smem_u32(&sm.q[h][0]);
    const int n = rg.n, lo = rg.lo, hi = rg.hi, b = rg.b;
    {   // Queries: row r = token r / 8, head h * 8 + r % 8; 2 threads per row (32 d each); RMS-norm, RoPE, FP8.
        const int r = tid >> 1, half = tid & 1;
        const int pos = rg.seq + (r >> 3);
        const __nv_bfloat16* src = qkv + (long long)(b * 8 + (r >> 3)) * qkv_w + (HH * 8 + (r & 7)) * HD + half * 32;
        uint4 raw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) raw[j] = reinterpret_cast<const uint4*>(src)[j];
        float4 cv[4], sv[4];
        const float4* cp = reinterpret_cast<const float4*>(cs + (long long)pos * HD + half * 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) { cv[j] = cp[j]; sv[j] = cp[j + 8]; }
        float x[32], ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&raw[j]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(hp[k]);
                x[j * 8 + 2 * k] = f.x; x[j * 8 + 2 * k + 1] = f.y;
                ss += f.x * f.x + f.y * f.y;
            }
        }
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        const float rs = rsqrtf(ss / HD + 1e-5f);
        const float* cf = reinterpret_cast<const float*>(cv);
        const float* sf = reinterpret_cast<const float*>(sv);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float a = x[2 * i], bb = x[2 * i + 1];
            x[2 * i] = (a * cf[i] - bb * sf[i]) * rs;
            x[2 * i + 1] = (bb * cf[i] + a * sf[i]) * rs;
        }
        uint32_t pk[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) pk[j] = pack_e4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
        if (isnew) {
            // The block's new token t = tid / 16, d 4 (tid % 16) + 0..3 of this KV head: RMS-norm + RoPE, E4M3 (as
            // _new), stored into the cache and into the new-token K / V^T tiles.
            const int t = tid >> 4, d0 = (tid & 15) * 4, tok = b * 8 + t, npos = rg.seq + t;
            const __nv_bfloat16* kp = qkv + (long long)tok * qkv_w + (16 + HH) * HD + d0;
            const uint2 kr = *reinterpret_cast<const uint2*>(kp), vr = *reinterpret_cast<const uint2*>(kp + 2 * HD);
            const float2 c2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + d0 / 2);
            const float2 s2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + 32 + d0 / 2);
            const long long slot = loc[tok];
            const float2 k01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.x));
            const float2 k23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.y));
            const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.x));
            const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.y));
            float kss = k01.x * k01.x + k01.y * k01.y + k23.x * k23.x + k23.y * k23.y;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) kss += __shfl_xor_sync(0xffffffff, kss, o);
            const float krs = rsqrtf(kss / HD + 1e-5f);
            const uint32_t kq = pack_e4((k01.x * c2.x - k01.y * s2.x) * krs / ksv, (k01.y * c2.x + k01.x * s2.x) * krs / ksv,
                                        (k23.x * c2.y - k23.y * s2.y) * krs / ksv, (k23.y * c2.y + k23.x * s2.y) * krs / ksv);
            const uint32_t vq = pack_e4(v01.x / vsv, v01.y / vsv, v23.x / vsv, v23.y / vsv);
            *reinterpret_cast<uint32_t*>(kc + slot * 128 + HH * HD + d0) = kq;
            *reinterpret_cast<uint32_t*>(vc + slot * 128 + HH * HD + d0) = vq;
            {
                // Zero the tiles' unused rows / slots, then place this lane's 4 key bytes (row t, SW128 chunk) and
                // 4 value bytes (V^T rows d0 .. d0 + 3, key slot of t in group 0: word t / 2, byte t % 2).
                const uint4 z = make_uint4(0, 0, 0, 0);
                uint4* vz = reinterpret_cast<uint4*>(&sm.vtnew[h][0]);
                vz[tid] = z; vz[tid + 128] = z;
#if NV_HALF
                {
                    uint4* kz = reinterpret_cast<uint4*>(&sm.knew[8 * 128]);
                    for (int e = tid; e < 448; e += 128) kz[e] = z;
                }
#else
                if (tid < 112) {
                    uint4* kz = reinterpret_cast<uint4*>(&sm.knew[(8 + 28 * h) * 128]);
                    kz[tid] = z; kz[tid + 112] = z;
                }
#endif
                asm volatile("bar.sync %0, 128;" ::"r"(2 + h) : "memory");
                const int c = h * 4 + (d0 >> 4);
                *reinterpret_cast<uint32_t*>(&sm.knew[t * 128 + ((c ^ (t & 7)) << 4) + (d0 & 15)]) = kq;
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    sm.vtnew[h][swz64((d0 + e) * 64) + 4 * (t >> 1) + (t & 1)] = (uint8_t)(vq >> (8 * e));
            }
        }
        uint8_t* qd = &sm.q[h][0];
        *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32 + 16)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("bar.sync %0, 128;" ::"r"(2 + h) : "memory");
    }
    float o[32], s[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
    uint32_t pa[8], pb[8];
    // Window start for this warp's token 2w rows, relative to key lo: keys t < seq - (win - 1) + 2w are out.
    const int thr = win > 0 ? rg.seq - (win - 1) + 2 * w - lo : -(1 << 30);
    bool newdone = false;
    if (isnew) {
        // The new tokens first, as one causal 8-key tile on the tensor cores, while the cached tiles' gathers fly.
        issue_qk(s, qa, smem_u32(&sm.knew[0]) + h * 64);
        wg_wait<0>();
        fence_regs(s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + (lane & 3) * 2;
            if (col > 2 * w) s[j * 4 + 0] = -INFINITY;
            if (col + 1 > 2 * w) s[j * 4 + 1] = -INFINITY;
            if (col > 2 * w + 1) s[j * 4 + 2] = -INFINITY;
            if (col + 1 > 2 * w + 1) s[j * 4 + 3] = -INFINITY;
        }
        softmax(s, m, l, pa, alpha, scale);
        fence_regs(pa);
        issue_pv(o, pa, smem_u32(&sm.vtnew[h][0]));
        wg_wait<0>();
        fence_regs(o);
        newdone = true;
    }
    if (n > 0) {
        mb_wait(&sm.ready[0], 0);
        issue_qk(s, qa, smem_u32(&sm.k[0][0]) + h * 64);
        wg_wait<0>();
        fence_regs(s);
        if (hi - lo < BN || thr + 1 > 0) mask(s, lane, hi - lo, thr);
        softmax(s, m, l, pa, alpha, scale);
        if (newdone) { rescale(o, alpha); fence_regs(o); }
        auto step = [&](uint32_t (&pin)[8], uint32_t (&pout)[8], int i) {
            const int sp = (i - 1) % NS, sc = i % NS;
            mb_wait(&sm.ready[sc], (i / NS) & 1);
            issue_qk(s, qa, smem_u32(&sm.k[sc][0]) + h * 64);
            fence_regs(pin);
            issue_pv(o, pin, smem_u32(&sm.vt[sp][h][0]));
            wg_wait<1>();
            fence_regs(s);
            const int nv = hi - lo - i * BN;
            if (nv < BN || thr + 1 - i * BN > 0) mask(s, lane, nv, thr - i * BN);
            softmax(s, m, l, pout, alpha, scale);
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
            rescale(o, alpha);
            fence_regs(o);
        };
        auto last = [&](uint32_t (&pin)[8]) {
            const int sp = (n - 1) % NS;
            fence_regs(pin);
            issue_pv(o, pin, smem_u32(&sm.vt[sp][h][0]));
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
        };
        int i = 1;
        for (; i + 1 < n; i += 2) { step(pa, pb, i); step(pb, pa, i + 1); }
        if (i < n) { step(pa, pb, i); last(pb); } else last(pa);
    }
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
    if (cl) stamp(pml, 1 + h);
#if NV_HALF
    // Release the producer's tiles this half has no keys for, then merge half 1 into half 0 through smem.
    for (int i = n; i < total; ++i) {
        mb_wait(&sm.ready[i % NS], (i / NS) & 1);
        if (lane == 0) mb_arrive(&sm.empty[i % NS]);
    }
    if (h == 1) {
        float4* xo = reinterpret_cast<float4*>(&sm.xo[tid * 36]);
#pragma unroll
        for (int j = 0; j < 8; ++j) xo[j] = make_float4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
        xo[8] = make_float4(m[0], m[1], l[0], l[1]);
        asm volatile("bar.sync 4, 256;" ::: "memory");
        return;
    }
    uint32_t rank = 0;
    __nv_bfloat162 gr[2][8];
    float2 gt[2][8];
    if (cl) {
        asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
        if ((w >> 1) == (int)rank) {  // rows this rank finishes: prefetch their gates (2 rows x 16 columns)
            const int ra = w * 16 + (lane >> 2);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const __nv_bfloat16* g = qkv + (long long)(b * 8 + (ra >> 3) + t) * qkv_w + 20 * HD +
                                         (H * 8 + (ra & 7)) * HD + (lane & 3) * 2;
#pragma unroll
                for (int j = 0; j < 8; ++j) gr[t][j] = *reinterpret_cast<const __nv_bfloat162*>(g + j * 8);
            }
        }
    }
    asm volatile("bar.sync 4, 256;" ::: "memory");
    if (cl) stamp(pml, 3);
    {
        const float4* xo = reinterpret_cast<const float4*>(&sm.xo[tid * 36]);
        const float4 x = xo[8];
        const float mn0 = fmaxf(m[0], x.x), mn1 = fmaxf(m[1], x.y);
        const float a0 = ex2(m[0] - mn0), b0 = ex2(x.x - mn0), a1 = ex2(m[1] - mn1), b1 = ex2(x.y - mn1);
        m[0] = mn0; m[1] = mn1;
        l[0] = l[0] * a0 + x.z * b0; l[1] = l[1] * a1 + x.w * b1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 y = xo[j];
            o[4 * j + 0] = o[4 * j + 0] * a0 + y.x * b0; o[4 * j + 1] = o[4 * j + 1] * a0 + y.y * b0;
            o[4 * j + 2] = o[4 * j + 2] * a1 + y.z * b1; o[4 * j + 3] = o[4 * j + 3] * a1 + y.w * b1;
        }
    }
    if (cl) {
        // 2-CTA cluster = the 2 splits of (request, H). Rank r finishes rows 32 r .. 32 r + 31 (warps 2 r, 2 r + 1):
        // the other 2 warps stage their merged o / m / l locally and one bulk DSMEM copy (9 KB) puts them into the
        // peer's xo2, completing on the peer's clbar; the owners merge and write the gated, normalized output.
        if ((w >> 1) != (int)rank) {
            float4* xl = reinterpret_cast<float4*>(&sm.xo2[tid * 36]);
#pragma unroll
            for (int j = 0; j < 8; ++j) xl[j] = make_float4(o[4 * j], o[4 * j + 1], o[4 * j + 2], o[4 * j + 3]);
            xl[8] = make_float4(m[0], m[1], l[0], l[1]);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // the peer's clbar init is visible
            asm volatile("bar.sync 5, 64;" ::: "memory");
            if ((tid & 63) == 0) {
                const uint32_t src = smem_u32(&sm.xo2[tid * 36]);
                uint32_t dst = src, bar = smem_u32(&sm.clbar);
                asm volatile("mapa.shared::cluster.u32 %0, %0, %1;" : "+r"(dst) : "r"(rank ^ 1));
                asm volatile("mapa.shared::cluster.u32 %0, %0, %1;" : "+r"(bar) : "r"(rank ^ 1));
                asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             ::"r"(dst), "r"(src), "n"(64 * 36 * 4), "r"(bar) : "memory");
                asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;" ::: "memory");
            }
            stamp(pml, 4);
            return;
        }
        // sigmoid(gate) while rank 1 finishes (as _norm_gate_store: o * sigmoid(gate))
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 g = __bfloat1622float2(gr[t][j]);
                gt[t][j] = make_float2(__fdividef(1.f, 1.f + __expf(-g.x)), __fdividef(1.f, 1.f + __expf(-g.y)));
            }
        fence_regs(*reinterpret_cast<float(*)[32]>(&gt[0][0]));  // keep the sigmoids before the wait
        asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n@!p bra W;\n}\n"
                     ::"r"(smem_u32(&sm.clbar)) : "memory");
        stamp(pml, 4);
        const float4* xo = reinterpret_cast<const float4*>(&sm.xo2[tid * 36]);
        const float4 x = xo[8];
        const float mn0 = fmaxf(m[0], x.x), mn1 = fmaxf(m[1], x.y);
        const float a0 = ex2(m[0] - mn0), b0 = ex2(x.x - mn0), a1 = ex2(m[1] - mn1), b1 = ex2(x.y - mn1);
        const float l0 = l[0] * a0 + x.z * b0, l1 = l[1] * a1 + x.w * b1;
        const float f0 = __fdividef(vsv, l0), f1 = __fdividef(vsv, l1);
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 y = xo[j];
            o[4 * j + 0] = (o[4 * j + 0] * a0 + y.x * b0) * f0; o[4 * j + 1] = (o[4 * j + 1] * a0 + y.y * b0) * f0;
            o[4 * j + 2] = (o[4 * j + 2] * a1 + y.z * b1) * f1; o[4 * j + 3] = (o[4 * j + 3] * a1 + y.w * b1) * f1;
            s0 += o[4 * j] * o[4 * j] + o[4 * j + 1] * o[4 * j + 1];
            s1 += o[4 * j + 2] * o[4 * j + 2] + o[4 * j + 3] * o[4 * j + 3];
        }
        if (NV_STAMP) { fence_regs(o); stamp(pml, 6); }
        s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
        s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
        const float r0 = rsqrtf(s0 / HD + 1e-5f), r1 = rsqrtf(s1 / HD + 1e-5f);
        const int ra = w * 16 + (lane >> 2);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            __nv_bfloat16* dsto = out + (long long)(b * 8 + (ra >> 3) + t) * (16 * HD) + (H * 8 + (ra & 7)) * HD +
                                  (lane & 3) * 2;
            const float rr = t ? r1 : r0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 g = gt[t][j];
                const __nv_bfloat162 v = __floats2bfloat162_rn(o[4 * j + 2 * t] * rr * g.x, o[4 * j + 2 * t + 1] * rr * g.y);
                *reinterpret_cast<__nv_bfloat162*>(dsto + j * 8) = v;
            }
        }
        stamp(pml, 5);
        return;
    }
    const long long part = (((long long)b * 2 + H) * S + split) * MR;
#else
    const long long part = (((long long)b * 2 + h) * S + split) * MR;
#endif
    const int ra = w * 16 + (lane >> 2);
    float* pa0 = pacc + (part + ra) * HD;
    float* pa1 = pa0 + 8 * HD;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        *reinterpret_cast<float2*>(pa0 + col) = make_float2(o[j * 4 + 0], o[j * 4 + 1]);
        *reinterpret_cast<float2*>(pa1 + col) = make_float2(o[j * 4 + 2], o[j * 4 + 3]);
    }
    if ((lane & 3) == 0) {
        const float off = NV_LAZY ? 0.f : 8.f;
        *reinterpret_cast<float2*>(pml + (part + ra) * 2) = make_float2(m[0] - off, l[0]);
        *reinterpret_cast<float2*>(pml + (part + ra + 8) * 2) = make_float2(m[1] - off, l[1]);
    }
}

template <int NS, int LAG>
__global__ void __launch_bounds__(384, NV_MINB) nvs(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cs,
                                              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                              const float* __restrict__ ks, const float* __restrict__ vsc,
                                              const long long* __restrict__ loc, const int* __restrict__ pt,
                                              const int* __restrict__ seq, float* __restrict__ pacc,
                                              float* __restrict__ pml, float qk_scale, int S, int ch, int maxp,
                                              int qkv_w, int win, int cl, __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x, wg = tid >> 7;
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mb_init(&sm.ready[i], 4); mb_init(&sm.empty[i], 8); }
#if NV_HALF
        if (cl) {  // 1 arrival (its own, with the expected bytes) + the peer's bulk copy of 9 KB
            mb_init(&sm.clbar, 1);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&sm.clbar)), "n"(64 * 36 * 4) : "memory");
        }
#endif
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (cl) {  // the peer's bulk copy completes on this CTA's clbar on rank 0's clbar later: its init must be visible cluster-wide
        // arrive now, wait only in the warps that copy into the peer (just before the copy)
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    } else {
        __syncthreads();
    }
    if (cl && wg == 1) {
        stamp(pml, 0);
        if (NV_STAMP && threadIdx.x == 128) { uint32_t sid; asm("mov.u32 %0, %%smid;" : "=r"(sid)); reinterpret_cast<unsigned long long*>(pml)[blockIdx.x * 8 + 7] = sid; }
    }
    const int split = blockIdx.x % S;
#if NV_HALF
    // CTA (request, KV head H, split); consumer c takes half c of the split's ch keys.
    const int H = (blockIdx.x / S) & 1, bb = blockIdx.x / (2 * S);
    const Range r0 = rng(seq, bb, 2 * split, ch / 2, win), r1 = rng(seq, bb, 2 * split + 1, ch / 2, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS, LAG>(sm, kc, vc, pt, r0, r1, H, maxp);
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS>(sm, wg - 1, qkv, cs, wg == 1 ? r0 : r1, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc,
                     loc, ks[0], vsc[0], H, r0.n, cl, out);
    }
#else
    const Range rg = rng(seq, blockIdx.x / S, split, ch, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS, LAG>(sm, kc, vc, pt, rg, maxp);
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS>(sm, wg - 1, qkv, cs, rg, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc, loc, ks[0], vsc[0], 0, 0, 0, nullptr);
    }
#endif
}
}  // namespace nv

void nvs(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vsc,
         torch::Tensor loc, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, double qk_scale, int64_t S, int64_t ch, int64_t win,
         torch::Tensor out, int64_t cl) {
    // Producer lag NV_LAG (1) at every S: LAG 2 at 4 splits was H100 -0.18 us but H200 +0.2 (check 9249146d, verify 32).
    auto f = nv::nvs<NV_NS, NV_LAG>;
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    static bool once = false;
    if (!once) {
        cudaFuncSetAttribute(nv::nvs<NV_NS, NV_LAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        once = true;
    }
    const int B = (int)seq.size(0);
    // cl (NV_HALF, S 2): the 2 splits of each (request, KV head) run as one 2-CTA cluster and merge over DSMEM.
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(S * B * (1 + NV_HALF)));
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = cl ? 2 : 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, f,
        (const __nv_bfloat16*)qkv.data_ptr(), (const float*)cs.data_ptr(), (uint8_t*)kc.data_ptr(),
        (uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const float*)vsc.data_ptr(),
        (const long long*)loc.data_ptr(), (const int*)pt.data_ptr(),
        (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale, (int)S, (int)ch,
        (int)pt.size(1), (int)qkv.size(-1), (int)win, (int)cl, (__nv_bfloat16*)out.data_ptr());
}
'''
try:
    _NVSCL = _load_inline(
        name="muse_nvscl_" + __import__("hashlib").sha256(_NVSCL_SRC.encode()).hexdigest()[:10],
        cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, "
                    "int64_t, torch::Tensor, int64_t);",
        cuda_sources=_NVSCL_SRC, functions=["nvs"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("nvscl build failed:", _e)
    _NVSCL = None
assert '#define NV_HALF 1' in _NVSCL_SRC and '#define NV_STAMP 0' in _NVSCL_SRC
NVS_CL_BATCH = 32  # cl from this batch on (S 2)
NVS_MIN_BATCH = 16  # verify sliding from this batch on nvs (verify 32: H100 paired -1.66 us, H200 16.64 -> 15.28/15.07;
# verify 16 with 8 splits: H100 paired -0.38 us in s9 (6/7), but the H200 tied 12.63 vs 12.63 in check baea7057)

_NDS_SRC = r'''#define NV_NS 3
#define NV_MAXR 96
#define NV_PREGS 40
#define NV_CREGS 152
// agent2 nds (decode sliding, both KV heads in one consumer warpgroup) built on nvs: native verify-SLIDING attention (L 8, G 8, D 64, HKV 2, page 1), from tools/s71/nverify.cu.
// One CTA per (split, request): warpgroup 0 gathers both KV heads' K/V rows (128-B lines) with cp.async and
// transposes V to FP8 V^T; warpgroups 1/2 (KV head 0/1) normalize + RoPE their queries, run QK^T (FP8 wgmma),
// the online softmax and P@V with P in E4M3 (FA3's FP8 path), QK(i) overlapping PV(i-1). Partials in
// _attn_split's layout [B, HKV, S, 64, 64] FP32 for _combine_hb. NVS_NEW: split S-1 also attends the 8 new tokens.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef NV_NS
#define NV_NS 4
#endif
#ifndef NV_PREGS
#define NV_PREGS 40
#endif
#ifndef NV_CREGS
#define NV_CREGS 208
#endif
#ifndef NV_MINB
#define NV_MINB 1
#endif
namespace nv {
constexpr int BN = 64, HD = 64, MR = 64;
#ifndef NV_LAG
#define NV_LAG 1
#endif
#ifndef NV_PFD
#define NV_PFD 4
#endif
#ifndef NV_TR
#define NV_TR 1
#endif
#ifndef NV_LAZY
#define NV_LAZY 0
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t mdesc(uint32_t addr, uint32_t sbo, uint32_t mode) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(sbo >> 4) << 32) | ((uint64_t)mode << 62);
}
__device__ __forceinline__ uint32_t swz64(uint32_t o) { return o ^ (((o >> 7) & 3) << 4); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_regs(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
#define D32(a) "+f"(a[0]), "+f"(a[1]), "+f"(a[2]), "+f"(a[3]), "+f"(a[4]), "+f"(a[5]), "+f"(a[6]), "+f"(a[7]), \
    "+f"(a[8]), "+f"(a[9]), "+f"(a[10]), "+f"(a[11]), "+f"(a[12]), "+f"(a[13]), "+f"(a[14]), "+f"(a[15]), \
    "+f"(a[16]), "+f"(a[17]), "+f"(a[18]), "+f"(a[19]), "+f"(a[20]), "+f"(a[21]), "+f"(a[22]), "+f"(a[23]), \
    "+f"(a[24]), "+f"(a[25]), "+f"(a[26]), "+f"(a[27]), "+f"(a[28]), "+f"(a[29]), "+f"(a[30]), "+f"(a[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
__device__ __forceinline__ void qk_k32(float (&s)[32], uint64_t da, uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", %32, %33, p, 1, 1;\n}\n"
                 : D32(s) : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ void pv8_k32(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n"
                 ::"r"(smem_u32(b)), "r"(par) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NS> struct Smem {
    alignas(1024) uint8_t k[NS][BN * 128];     // both heads' K rows, SW128 (wgmma B, K-major)
    alignas(1024) uint8_t vs[NS][BN * 128];    // V rows as gathered; 16-byte chunk c of row r at c ^ vsw(r)
    alignas(1024) uint8_t vt[NS][2][HD * BN];  // per head V^T [d][key slot], SW64 (wgmma B, K-major), keys permuted
    alignas(1024) uint8_t q[2][MR * HD];       // per head FP8 queries [row][d], SW64
    int ptr[8][BN];                            // page 1: page-table entries of tiles i .. i + NV_PFD (ring of 8)
    int lim[8];                                // and each tile's last valid key
    uint64_t ready[NS], empty[NS];
    alignas(1024) uint8_t knew[BN * 128];      // last split: the 8 new keys as a K tile (rows 0-7; rows 8-63 zero)
    alignas(1024) uint8_t vtnew[2][HD * BN];   // and their V^T (key slots 0-7 of group 0; the rest zero)
};
// Staging swizzle: SW128 (r & 7) for the ldmatrix transpose (8 consecutive rows per phase), else conflict-free for
// the 4-row register transpose.
__device__ __forceinline__ uint32_t vsw(int r) { return NV_TR ? (r & 7) : (((r >> 1) & 3) | (((r >> 4) & 1) << 2)); }


struct Range { int b, lo, hi, n, seq; };
__device__ __forceinline__ Range rng(const int* __restrict__ seq, int b, int split, int ch, int win) {
    Range r;
    r.b = b;
    r.seq = seq[b];
    const int base = win > 0 ? max(r.seq - (win - 1), 0) : 0;
    r.lo = base + split * ch;
    r.hi = min(r.lo + ch, r.seq);
    r.n = r.hi > r.lo ? (r.hi - r.lo + BN - 1) / BN : 0;
    return r;
}
__device__ __forceinline__ void cp4ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}

template <int NS>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, Range rg, int maxp) {
    const int tid = threadIdx.x;  // 0..127
    const int total = rg.n;
    const int* ptr = pt + (long long)rg.b * maxp;
    // Warp 0 stages each tile's page-table entries NV_PFD tiles ahead in sm.ptr (2 x 4-byte cp.async per lane: the
    // window start is not 16-byte aligned).
    auto stage = [&](int j) {
        const int kb = rg.lo + j * BN;
        if (kb + tid < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid]), ptr + kb + tid);
        if (kb + tid + 32 < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid + 32]), ptr + kb + tid + 32);
        if (tid == 0) sm.lim[j & 7] = rg.hi - 1 - kb;
    };
    if (tid < 32)
        for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
    cp_commit();
    cp_wait<0>();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int grow = tid >> 3;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    const int lane = tid & 31;
    for (int i = 0; i < total + NV_LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            const int lim = sm.lim[i & 7];
            int cr[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][min(grow + 16 * j, lim)];
            if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
            mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
            const int ch = tid & 7;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = grow + 16 * j;
                const long long off = (long long)cr[j] * 128 + ch * 16;
                cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kc + off);
                cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vc + off);
            }
        }
        cp_commit();
        if (i >= NV_LAG) {
            const int s = (i - NV_LAG) % NS;
            cp_wait<NV_LAG>();
            asm volatile("bar.sync 1, 128;" ::: "memory");
            {
                const uint32_t vsb = smem_u32(&sm.vs[s][0]), vtb = smem_u32(&sm.vt[s][0][0]);
                const int wq = tid >> 5, mi = lane >> 3, ri = lane & 7;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int u = wq * 4 + q, gg = u & 3, hc = u >> 2;
                    const int r = 16 * gg + (mi & 1) * 8 + ri, cc = 2 * hc + (mi >> 1);
                    uint32_t R0, R1, R2, R3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(vsb + r * 128 + ((cc ^ (r & 7)) << 4)));
                    const uint32_t S0 = prmt(R0, R1, 0x6420), S1 = prmt(R0, R1, 0x7531);
                    const uint32_t S2 = prmt(R2, R3, 0x6420), S3 = prmt(R2, R3, 0x7531);
                    const int d = 32 * (hc & 1) + 16 * (mi >> 1) + 2 * ri + (mi & 1);
                    const uint32_t o = (uint32_t)(hc >> 1) * (HD * BN) + swz64(d * 64 + 16 * gg);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                                 :: "r"(vtb + o), "r"(S0), "r"(S1), "r"(S2), "r"(S3) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);
        }
    }
}

__device__ __forceinline__ void softmax(float (&s)[32], float (&m)[2], float (&l)[2], uint32_t (&p)[8],
                                        float (&alpha)[2], float scale) {
    float mx0 = s[0], mx1 = s[2];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        mx0 = fmaxf(mx0, fmaxf(s[j * 4 + 0], s[j * 4 + 1]));
        mx1 = fmaxf(mx1, fmaxf(s[j * 4 + 2], s[j * 4 + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    float mn0 = fmaxf(m[0], mx0 * scale), mn1 = fmaxf(m[1], mx1 * scale);
    if (NV_LAZY) {
        bool grow = (mn0 > m[0] + 8.f) || (mn1 > m[1] + 8.f);
        if (!__any_sync(0xffffffff, grow)) { mn0 = m[0]; mn1 = m[1]; }
    }
    alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    const float off = NV_LAZY ? 0.f : 8.f;
    const float o0 = mn0 - off, o1 = mn1 - off;
    float e[32], l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        e[j * 4 + 0] = ex2(fmaf(s[j * 4 + 0], scale, -o0)); e[j * 4 + 1] = ex2(fmaf(s[j * 4 + 1], scale, -o0));
        e[j * 4 + 2] = ex2(fmaf(s[j * 4 + 2], scale, -o1)); e[j * 4 + 3] = ex2(fmaf(s[j * 4 + 3], scale, -o1));
        l0 += e[j * 4 + 0] + e[j * 4 + 1]; l1 += e[j * 4 + 2] + e[j * 4 + 3];
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
        const int b0 = 16 * kk;
        p[kk * 4 + 0] = pack_e4(e[b0 + 0], e[b0 + 1], e[b0 + 4], e[b0 + 5]);
        p[kk * 4 + 1] = pack_e4(e[b0 + 2], e[b0 + 3], e[b0 + 6], e[b0 + 7]);
        p[kk * 4 + 2] = pack_e4(e[b0 + 8], e[b0 + 9], e[b0 + 12], e[b0 + 13]);
        p[kk * 4 + 3] = pack_e4(e[b0 + 10], e[b0 + 11], e[b0 + 14], e[b0 + 15]);
    }
    l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1;
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2]) {
    if (NV_LAZY && __all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}
__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[8], uint32_t va) {
    wg_fence();
    pv8_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
    pv8_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
    wg_commit();
}
// Score masks: key column col (of this tile) is out if col >= nvalid (past the split's end) or col < thr0 / thr1
// (rows of token 2w / 2w + 1: before the sliding window's start).
__device__ __forceinline__ void mask(float (&s)[32], int lane, int nvalid, int thr0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid || col < thr0) s[j * 4 + 0] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0) s[j * 4 + 1] = -INFINITY;
        if (col >= nvalid || col < thr0 + 1) s[j * 4 + 2] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0 + 1) s[j * 4 + 3] = -INFINITY;
    }
}


#define D64(a) D32(a), "+f"(a[32]), "+f"(a[33]), "+f"(a[34]), "+f"(a[35]), "+f"(a[36]), "+f"(a[37]), "+f"(a[38]), \
    "+f"(a[39]), "+f"(a[40]), "+f"(a[41]), "+f"(a[42]), "+f"(a[43]), "+f"(a[44]), "+f"(a[45]), "+f"(a[46]), "+f"(a[47]), \
    "+f"(a[48]), "+f"(a[49]), "+f"(a[50]), "+f"(a[51]), "+f"(a[52]), "+f"(a[53]), "+f"(a[54]), "+f"(a[55]), "+f"(a[56]), \
    "+f"(a[57]), "+f"(a[58]), "+f"(a[59]), "+f"(a[60]), "+f"(a[61]), "+f"(a[62]), "+f"(a[63])
#define R64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27," \
    "%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55," \
    "%56,%57,%58,%59,%60,%61,%62,%63}"
__device__ __forceinline__ void pv16_k32(float (&o)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " R64 ", {%64,%65,%66,%67}, %68, 1, 1, 1;\n"
                 : D64(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
// QK over the whole 128-B K row (both heads; each query row is zero in the other head's 64 d).
__device__ __forceinline__ void issue_qk2(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
#pragma unroll
    for (int k = 0; k < 4; ++k) qk_k32(s, mdesc(qa + 32 * k, 1024, 1), mdesc(ka + 32 * k, 1024, 1), k);
    wg_commit();
}
// P@V into both heads' 128 output columns (V^T of head 0 then head 1: 128 SW64 rows).
__device__ __forceinline__ void issue_pv2(float (&o)[64], const uint32_t (&p)[8], uint32_t va) {
    wg_fence();
    pv16_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
    pv16_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
    wg_commit();
}
__device__ __forceinline__ void dmask(float (&s)[32], int lane, int nvalid, int thr) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid || col < thr) { s[j * 4 + 0] = -INFINITY; s[j * 4 + 2] = -INFINITY; }
        if (col + 1 >= nvalid || col + 1 < thr) { s[j * 4 + 1] = -INFINITY; s[j * 4 + 3] = -INFINITY; }
    }
}
// Row lane / 4 is head 0 (output columns 0-63), row lane / 4 + 8 head 1 (columns 64-127).
__device__ __forceinline__ void drescale(float (&o)[64], const float (&alpha)[2]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0];
        o[32 + j * 4 + 2] *= alpha[1]; o[32 + j * 4 + 3] *= alpha[1];
    }
}

// Decode (L 1, G 8): one consumer warpgroup for both KV heads. m64 rows 0-7 = head 0's 8 query heads (d 0-63 of the
// K row; 64-127 zero), rows 8-15 = head 1's (d 64-127), rows 16-63 zero: only warp 0 has live rows and runs the
// softmax; warps 1-3 just take part in the wgmmas (P = 0).
// layer s29 (QF32 at T=32): qkv may be the FP32 output of the qkv GEMM; each element is rounded to BF16 (RN, as the
// torch cast / GEMM output it replaces) at its load: everything downstream is bit-identical to the BF16 input.
__device__ __forceinline__ unsigned qpk(float lo, float hi) {
    __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
    return *(unsigned*)&t;
}
__device__ __forceinline__ uint4 qld8(const __nv_bfloat16* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint4 qld8(const float* p) {
    const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    return make_uint4(qpk(a.x, a.y), qpk(a.z, a.w), qpk(b.x, b.y), qpk(b.z, b.w));
}
__device__ __forceinline__ uint2 qld4(const __nv_bfloat16* p) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ uint2 qld4(const float* p) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    return make_uint2(qpk(a.x, a.y), qpk(a.z, a.w));
}
// proj s22 (QSS): qkv as 2 FP32 split-K slabs qss elements apart (one batched GEMM, no split-K reduce kernel): the
// two partials are added in FP32 and rounded to BF16 (RN) at the load, as cuBLAS's splitKreduce + BF16 output does.
struct qs2 { float v; };
__device__ __forceinline__ uint4 qld8(const __nv_bfloat16* p, int) { return qld8(p); }
__device__ __forceinline__ uint4 qld8(const float* p, int) { return qld8(p); }
__device__ __forceinline__ uint4 qld8(const qs2* q, int qss) {
    const float* p = reinterpret_cast<const float*>(q);
    const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    const float4 c = reinterpret_cast<const float4*>(p + qss)[0], d = reinterpret_cast<const float4*>(p + qss)[1];
    return make_uint4(qpk(a.x + c.x, a.y + c.y), qpk(a.z + c.z, a.w + c.w), qpk(b.x + d.x, b.y + d.y), qpk(b.z + d.z, b.w + d.w));
}
__device__ __forceinline__ uint2 qld4(const __nv_bfloat16* p, int) { return qld4(p); }
__device__ __forceinline__ uint2 qld4(const float* p, int) { return qld4(p); }
__device__ __forceinline__ uint2 qld4(const qs2* q, int qss) {
    const float* p = reinterpret_cast<const float*>(q);
    const float4 a = *reinterpret_cast<const float4*>(p), c = *reinterpret_cast<const float4*>(p + qss);
    return make_uint2(qpk(a.x + c.x, a.y + c.y), qpk(a.z + c.z, a.w + c.w));
}
template <int NS, typename QT>
__device__ __forceinline__ void dconsumer(Smem<NS>& sm, const QT* __restrict__ qkv,
                                          const float* __restrict__ cs, Range rg, int split, int S, int qkv_w,
                                          float scale, int win, float* __restrict__ pacc, float* __restrict__ pml,
                                          uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                          const long long* __restrict__ loc, float ksv, float vsv, __nv_bfloat16* __restrict__ out, int qss) {
    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;
    uint8_t* qd = &sm.q[0][0];  // [64 rows][128 B] SW128
    const uint32_t qa = smem_u32(qd);
    const int n = rg.n, lo = rg.lo, hi = rg.hi, b = rg.b;
    const bool last_split = split == S - 1;
    {
        const uint4 z = make_uint4(0, 0, 0, 0);
        if (last_split) {
            uint4* vz = reinterpret_cast<uint4*>(&sm.vtnew[0][0]);
#pragma unroll
            for (int j = 0; j < 4; ++j) vz[tid + 128 * j] = z;
        }
        if (w > 0) {
            uint4* qz = reinterpret_cast<uint4*>(qd + 16 * 128);
#pragma unroll
            for (int j = 0; j < 4; ++j) qz[tid - 32 + 96 * j] = z;
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
    }
    if (w == 0) {   // Queries: row r = query head r (KV head r / 8); 2 lanes per row (32 d each).
        const int r = lane >> 1, half = lane & 1, h = r >> 3;
        const int pos = rg.seq;
        const QT* src = qkv + (long long)b * qkv_w + r * HD + half * 32;
        uint4 raw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) raw[j] = qld8(src + 8 * j, qss);
        float4 cv[4], sv[4];
        const float4* cp = reinterpret_cast<const float4*>(cs + (long long)pos * HD + half * 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) { cv[j] = cp[j]; sv[j] = cp[j + 8]; }
        float x[32], ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&raw[j]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(hp[k]);
                x[j * 8 + 2 * k] = f.x; x[j * 8 + 2 * k + 1] = f.y;
                ss += f.x * f.x + f.y * f.y;
            }
        }
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        const float rs = rsqrtf(ss / HD + 1e-5f);
        const float* cf = reinterpret_cast<const float*>(cv);
        const float* sf = reinterpret_cast<const float*>(sv);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float a = x[2 * i], bb = x[2 * i + 1];
            x[2 * i] = (a * cf[i] - bb * sf[i]) * rs;
            x[2 * i + 1] = (bb * cf[i] + a * sf[i]) * rs;
        }
        uint32_t pk[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) pk[j] = pack_e4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
        const int c0 = h * 4 + half * 2, z0 = (1 - h) * 4 + half * 2;  // 16-B chunks: live d, zero d
        *reinterpret_cast<uint4*>(qd + r * 128 + ((c0 ^ (r & 7)) << 4)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(qd + r * 128 + (((c0 + 1) ^ (r & 7)) << 4)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        *reinterpret_cast<uint4*>(qd + r * 128 + ((z0 ^ (r & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(qd + r * 128 + (((z0 + 1) ^ (r & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        if (last_split) {
            // The step's new token: lane = KV head (lane / 16), d 4 (lane % 16) + 0..3: RMS-norm + RoPE, E4M3 (as
            // _new), stored into the cache and into the new-token K row 0 / V^T key slot 0.
            const int hh = lane >> 4, d0 = (lane & 15) * 4, npos = rg.seq;
            const QT* kp = qkv + (long long)b * qkv_w + (16 + hh) * HD + d0;
            const uint2 kr = qld4(kp, qss), vr = qld4(kp + 2 * HD, qss);
            const float2 c2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + d0 / 2);
            const float2 s2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + 32 + d0 / 2);
            const long long slot = loc[b];
            const float2 k01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.x));
            const float2 k23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.y));
            const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.x));
            const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.y));
            float kss = k01.x * k01.x + k01.y * k01.y + k23.x * k23.x + k23.y * k23.y;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) kss += __shfl_xor_sync(0xffffffff, kss, o);
            const float krs = rsqrtf(kss / HD + 1e-5f);
            const uint32_t kq = pack_e4((k01.x * c2.x - k01.y * s2.x) * krs / ksv, (k01.y * c2.x + k01.x * s2.x) * krs / ksv,
                                        (k23.x * c2.y - k23.y * s2.y) * krs / ksv, (k23.y * c2.y + k23.x * s2.y) * krs / ksv);
            const uint32_t vq = pack_e4(v01.x / vsv, v01.y / vsv, v23.x / vsv, v23.y / vsv);
            *reinterpret_cast<uint32_t*>(kc + slot * 128 + hh * HD + d0) = kq;
            *reinterpret_cast<uint32_t*>(vc + slot * 128 + hh * HD + d0) = vq;
            const int c = hh * 4 + (d0 >> 4);
            *reinterpret_cast<uint32_t*>(&sm.knew[(c << 4) + (d0 & 15)]) = kq;
#pragma unroll
            for (int e = 0; e < 4; ++e) sm.vtnew[hh][swz64((d0 + e) * 64)] = (uint8_t)(vq >> (8 * e));
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 2, 128;" ::: "memory");
    float o[64], s[32];
#pragma unroll
    for (int i = 0; i < 64; ++i) o[i] = 0.f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
    uint32_t pa[8], pb[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { pa[i] = 0u; pb[i] = 0u; }
    const bool live = w == 0;
    // Window start relative to key lo: keys t < seq - (win - 1) are out.
    const int thr = win > 0 ? rg.seq - (win - 1) - lo : -(1 << 30);
    bool newdone = false;
    if (last_split) {
        issue_qk2(s, qa, smem_u32(&sm.knew[0]));
        wg_wait<0>();
        fence_regs(s);
        if (live) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = j * 8 + (lane & 3) * 2;
                if (col > 0) { s[j * 4 + 0] = -INFINITY; s[j * 4 + 2] = -INFINITY; }
                s[j * 4 + 1] = -INFINITY; s[j * 4 + 3] = -INFINITY;
            }
            softmax(s, m, l, pa, alpha, scale);
        }
        fence_regs(pa);
        issue_pv2(o, pa, smem_u32(&sm.vtnew[0][0]));
        wg_wait<0>();
        fence_regs(o);
        newdone = true;
    }
    if (n > 0) {
        mb_wait(&sm.ready[0], 0);
        issue_qk2(s, qa, smem_u32(&sm.k[0][0]));
        wg_wait<0>();
        fence_regs(s);
        if (live) {
            if (hi - lo < BN || thr > 0) dmask(s, lane, hi - lo, thr);
            softmax(s, m, l, pa, alpha, scale);
            if (newdone) { drescale(o, alpha); fence_regs(o); }
        }
        auto step = [&](uint32_t (&pin)[8], uint32_t (&pout)[8], int i) {
            const int sp = (i - 1) % NS, sc = i % NS;
            mb_wait(&sm.ready[sc], (i / NS) & 1);
            issue_qk2(s, qa, smem_u32(&sm.k[sc][0]));
            fence_regs(pin);
            issue_pv2(o, pin, smem_u32(&sm.vt[sp][0][0]));
            wg_wait<1>();
            fence_regs(s);
            const int nv = hi - lo - i * BN;
            if (live) {
                if (nv < BN || thr - i * BN > 0) dmask(s, lane, nv, thr - i * BN);
                softmax(s, m, l, pout, alpha, scale);
            }
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
            if (live) drescale(o, alpha);
            fence_regs(o);
        };
        auto lastt = [&](uint32_t (&pin)[8]) {
            const int sp = (n - 1) % NS;
            fence_regs(pin);
            issue_pv2(o, pin, smem_u32(&sm.vt[sp][0][0]));
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
        };
        int i = 1;
        for (; i + 1 < n; i += 2) { step(pa, pb, i); step(pb, pa, i + 1); }
        if (i < n) { step(pa, pb, i); lastt(pb); } else lastt(pa);
    }
    if (!live) return;
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
    const int ra = lane >> 2;
    const long long part0 = (((long long)b * 2 + 0) * S + split) * 8, part1 = (((long long)b * 2 + 1) * S + split) * 8;
    float* pa0 = pacc + (part0 + ra) * HD;
    float* pa1 = pacc + (part1 + ra) * HD;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        *reinterpret_cast<float2*>(pa0 + col) = make_float2(o[j * 4 + 0], o[j * 4 + 1]);
        *reinterpret_cast<float2*>(pa1 + col) = make_float2(o[32 + j * 4 + 2], o[32 + j * 4 + 3]);
    }
    if ((lane & 3) == 0) {
        const float off = NV_LAZY ? 0.f : 8.f;
        *reinterpret_cast<float2*>(pml + (part0 + ra) * 2) = make_float2(m[0] - off, l[0]);
        *reinterpret_cast<float2*>(pml + (part1 + ra) * 2) = make_float2(m[1] - off, l[1]);
    }
}

template <int NS, typename QT>
#ifdef NV_MAXR
__global__ void __maxnreg__(NV_MAXR) nds(
#else
__global__ void __launch_bounds__(256, NV_MINB) nds(
#endif
const QT* __restrict__ qkv, const float* __restrict__ cs,
                                              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                              const float* __restrict__ ks, const float* __restrict__ vsc,
                                              const long long* __restrict__ loc, const int* __restrict__ pt,
                                              const int* __restrict__ seq, float* __restrict__ pacc,
                                              float* __restrict__ pml, float qk_scale, int S, int ch, int maxp,
                                              int qkv_w, int win, __nv_bfloat16* __restrict__ out, int qss,
                                              const uint8_t* __restrict__ pf, long long pfb) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x, wg = tid >> 7;
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mb_init(&sm.ready[i], 4); mb_init(&sm.empty[i], 4); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int split = blockIdx.x % S;
    const Range rg = rng(seq, blockIdx.x / S, split, ch, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS>(sm, kc, vc, pt, rg, maxp);
        if (pfb > 0 && tid < 32) {  // layer s34 (O_PF): after this CTA's last KV load is issued, L2-prefetch its share of the next GEMM's weight
            const long long per = ((pfb / gridDim.x) + 4095) & ~4095LL, end = min(pfb, per * (blockIdx.x + 1));
            for (long long o = per * blockIdx.x + tid * 16384LL; o < end; o += 32 * 16384LL)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pf + o), "r"((int)min(16384LL, end - o)) : "memory");
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        dconsumer<NS, QT>(sm, qkv, cs, rg, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc, loc, ks[0], vsc[0], out, qss);
    }
}
}  // namespace nv

void nds(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vsc,
         torch::Tensor loc, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, torch::Tensor out, double qk_scale, int64_t S, int64_t ch,
         int64_t win, int64_t qss, int64_t pf_ptr, int64_t pf_bytes) {
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    const int B = (int)seq.size(0);
#define NDS_LAUNCH(QT) { \
    auto f = nv::nds<NV_NS, QT>; \
    static bool once = false; \
    if (!once) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; } \
    f<<<dim3((unsigned)(S * B)), 256, smem, c10::cuda::getCurrentCUDAStream()>>>( \
        (const QT*)qkv.data_ptr(), NDS_ARGS); }
#define NDS_ARGS (const float*)cs.data_ptr(), (uint8_t*)kc.data_ptr(), \
        (uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const float*)vsc.data_ptr(), \
        (const long long*)loc.data_ptr(), (const int*)pt.data_ptr(), \
        (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale, (int)S, (int)ch, \
        (int)pt.size(1), (int)qkv.size(-1), (int)win, (__nv_bfloat16*)out.data_ptr(), (int)qss, \
        (const uint8_t*)pf_ptr, (long long)pf_bytes
    if (qss) { TORCH_CHECK(qkv.scalar_type() == at::kFloat); NDS_LAUNCH(nv::qs2) }
    else if (qkv.scalar_type() == at::kFloat) NDS_LAUNCH(float) else NDS_LAUNCH(__nv_bfloat16)
#undef NDS_LAUNCH
#undef NDS_ARGS
}
'''
try:
    _NDS = _load_inline(
        name="muse_nds_a2s10q2_" + __import__("hashlib").sha256(_NDS_SRC.encode()).hexdigest()[:10],
        cpp_sources="void nds(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                    "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
        cuda_sources=_NDS_SRC, functions=["nds"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("nds build failed, using Triton:", _e)
    _NDS = None
# Decode sliding from this batch on nds (agent2 s10: nvs's producer + ONE consumer warpgroup for both KV heads, 8
# splits, 2 CTAs/SM at 96 launch registers): H100 decode 32 sliding 15.2 -> ~14.1 us. Batch 16 stays on csl.
NDS_MIN_BATCH = 32


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
ROPE_POSITIONS = 1 << 21
_ROPE = {}
_SMS = {}
_COUNTERS = {}
# The MMA input precision for P@V (and QK^T without qk8). FP16 holds every E4M3 value exactly and
# converts from it in one instruction (BF16 takes two); P in [0, 1] keeps more bits than in BF16.
# A test hook too: Triton's CPU interpreter can't multiply 16-bit floats.
DOT_DTYPE = tl.float16
# Programmatic dependent launch of the merge kernel (off in the CPU interpreter).
USE_PDL = True
LZ = 10  # _gw_verify softmax: 10 = fixed reference after LZ_PK exact tiles + REDO (s105), 3 = exact per-warp lazy max (s79), 0 = per-tile row max + rescale
MERGE1 = True  # one-pass merges issue the accumulator loads before the max reduction
USE_GDEC = True  # decode full layers at batch * hkv > 2 on the Gluon kernel (_gd_full)
USE_GSLIDE = False  # batched sliding decode on _gd_full too (H100 win, H200 loss)
# s103: plan red (in-grid FP32 atomic merge, s95) on batched decode full too, with BNR replicas per request (H100
# paired vs _combine / _combine_bal: decode 16 full -0.94, decode 32 full -1.18 us, 8/8; BNR 2 -0.02 / -0.70,
# 4 +0.75 / -0.65, 8 +3.4 / +1.1: fewer distinct atomic lines win here, unlike batch 1's 511-way contention).
BRED = True
BNR = 1
USE_GD1 = True  # decode full at batch * hkv == 2 on _gd_full too (511 splits, 8-warp one-pass merge); unchecked on H200
ATT_PLAN = {}  # layer s84: plan overrides for the Gluon full-layer kernels (set per call by the layer's tuner option aplan)
ATT_DIAG = 0  # layer s84: DIAG probe code for full-layer attention (set per call by the layer's tuner; 0 = real kernels)
NT_AFTER = True  # gbal: _new_tokens launched after _gw_verify (it blocked 2nd gw CTAs on its SMs; s125)
USE_GBAL = True  # verify full from 16 KV head pairs: balanced splits over the whole batch (_gw_bal); unchecked on H200
# Split partials' accumulator dtype (plan key pdt). FP32 only: SGLang's FA3 keeps its split partials in
# FP32, and the rules forbid computing at lower precision than SGLang where they don't say otherwise.
_PDT = {"f32": torch.float32}
print(f"env: torch {torch.__version__}, triton {triton.__version__}", flush=True)


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, built as SGLang's RotaryEmbedding (and the reference) do."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        ang = torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device)[:, None] * inv_freq
        cache = _ROPE[theta] = torch.cat((ang.cos(), ang.sin()), dim=-1).contiguous()
    return cache


def _counters(n, device, tag=""):
    """Zeroed int32 arrival counters for the fused merges; each call leaves them zeroed again (the Gluon
    merge's generation words, tag "g", just count up)."""
    buf = _COUNTERS.get((device, tag))
    if buf is None or buf.numel() < n:
        buf = _COUNTERS[(device, tag)] = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
    return buf


def _red_buf(n, device, tag="red"):
    """Zeroed FP32 sums for plan red's in-grid merge; each call leaves them zeroed again."""
    buf = _COUNTERS.get((device, tag))
    if buf is None or buf.numel() < n:
        buf = _COUNTERS[(device, tag)] = torch.zeros(n, dtype=torch.float32, device=device)
    return buf


def _num_sms(device):
    if device.type != "cuda":
        return 132
    if device.index not in _SMS:
        _SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device.index]


# ---- Verify blocks on full layers (Gluon, warp-specialized) ----
# One program per (split, request, KV head pair), FA3 style: a 4-warp producer gathers the pair's
# 128-byte K and V rows (page indices, cp.async) into an NS-slot shared ring and signals `ready`
# (cp.async.mbarrier.arrive); one consumer warpgroup per KV head runs QK^T (FP8 wgmma on its 64-byte
# half of each row), stages its V half as FP16, frees the slot (`empty`), and does the online softmax
# and P@V. The extra split S holds the block's new tokens (normalized, stored as FP8, attended causally).
if HAVE_GLUON:
    @gluon.jit
    def _gw_producer(pt_row, kbase, vbase, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                     PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                     NS: gl.constexpr, PW: gl.constexpr):
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
        pg = gl.load(pt_row + t // PAGE)
        for i in range(ntiles):
            s = i % NS
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            # Next tile's page indices before waiting for the slot.
            t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
            pg = gl.load(pt_row + t // PAGE)
            mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
            cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
            cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
            cp.mbarrier_arrive(ready.index(s), increment_count=False)

    @gluon.jit
    def _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV: gl.constexpr, SP: gl.constexpr,
                  M: gl.constexpr, D: gl.constexpr, R_L: gl.constexpr, A_L: gl.constexpr):
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, A_L))
        co = gl.arange(0, D, layout=gl.SliceLayout(0, A_L))
        part = ((b * HKV + hh) * SP + split) * M + ro
        gl.store(pacc_ptr + part[:, None] * D + co[None, :], acc.to(pacc_ptr.dtype.element_ty))
        rs = gl.arange(0, M, layout=R_L)
        parts = ((b * HKV + hh) * SP + split) * M + rs
        gl.store(pml_ptr + parts * 2, m_i)
        gl.store(pml_ptr + parts * 2 + 1, l_i)

    @gluon.jit
    def _gw_merge_rows(qkv_ptr, b, h, split, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr,
                       L: gl.constexpr, M: gl.constexpr, D: gl.constexpr, RS: gl.constexpr, R_PAD: gl.constexpr):
        """Program `split`'s slice of the merge: rows [split * RS, +RS) of the pair's 2 * M rows (KV heads h,
        h + 1), and their gate logits (loaded before the wait for the other programs)."""
        ML: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        r2 = split * RS + gl.arange(0, R_PAD, layout=gl.SliceLayout(1, ML))
        ok = (r2 < split * RS + RS) & (r2 < 2 * M)
        r2 = gl.minimum(r2, 2 * M - 1)
        hh = h + r2 // M
        r = r2 % M
        c = gl.arange(0, D, layout=gl.SliceLayout(0, ML))
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        tok = b * L + r // G
        col = (hh * G + r % G)[:, None] * D + c[None, :]
        gate = gl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col).to(gl.bfloat16).to(gl.float32)
        return hh, r, c, ok, tok, col, gate

    @gluon.jit
    def _gw_merge(pacc_ptr, pml_ptr, out_ptr, vs, b, hh, r, c, ok, tok, col, gate, HQ: gl.constexpr,
                  HKV: gl.constexpr, S: gl.constexpr, M: gl.constexpr, D: gl.constexpr, R_PAD: gl.constexpr):
        """_combine inside _gw_verify for _gw_merge_rows' slice: merge all S + 1 partials, then the per-head
        RMSNorm and sigmoid(gate)."""
        ML: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        SP: gl.constexpr = S + 1
        base = (b * HKV + hh) * SP * M + r
        m_max = gl.full([R_PAD], -1.0e30, gl.float32, layout=gl.SliceLayout(1, ML))
        for s in gl.static_range(SP):
            m_max = gl.maximum(m_max, gl.load(pml_ptr + (base + s * M) * 2, cache_modifier=".cg"))
        denom = gl.zeros([R_PAD], gl.float32, layout=gl.SliceLayout(1, ML))
        acc = gl.zeros([R_PAD, D], gl.float32, layout=ML)
        for s in gl.static_range(SP):
            w = gl.exp2(gl.load(pml_ptr + (base + s * M) * 2, cache_modifier=".cg") - m_max)
            denom += w * gl.load(pml_ptr + (base + s * M) * 2 + 1, cache_modifier=".cg")
            acc += w[:, None] * gl.load(pacc_ptr + (base + s * M)[:, None] * D + c[None, :],
                                        cache_modifier=".cg").to(gl.float32)
        o = acc / denom[:, None] * vs
        o = o * gl.rsqrt(gl.sum(o * o, 1) / D + 1e-5)[:, None]
        o = o / (1.0 + gl.exp2(gate * -1.4426950408889634))
        gl.store(out_ptr + tok[:, None] * (HQ * D) + col, o.to(gl.bfloat16), mask=ok[:, None])

    @gluon.jit
    def _gw_consumer(q, k_ring, v_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh,
                     split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr,
                     HEAD: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr,
                     S: gl.constexpr, M: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
        acc = gl.zeros([M, D], gl.float32, layout=O_L)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        for i in range(ntiles):
            s = i % NS
            mbarrier.wait(ready.index(s), (i // NS) & 1)
            s_tok = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                  gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            vh.store(v_ring.index(s).slice(HEAD * D, D, 1).load(VL).to(gl.float16))
            fence_async_shared()
            t = lo + i * BN + n_s
            sc = warpgroup_mma_wait(0, deps=[s_tok])
            gl.barrier()
            mbarrier.arrive(empty.index(s))
            sc = gl.where((t < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            a_tok = warpgroup_mma(p16, vh, acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
        _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV, S + 1, M, D, gl.SliceLayout(1, S_L), O_L)

    @gluon.jit
    def _gw_producer_cv(pt_row, kbase, vbase, k_ring, v_ring, vh0, vh1, ready, empty, lo, hi, ntiles,
                        PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                        D: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr, LAG: gl.constexpr):
        """_gw_producer that also converts V to FP16 (per-head halves), LAG tiles behind its gathers."""
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
        pg = gl.load(pt_row + t // PAGE)
        for i in range(ntiles + LAG):
            if i < ntiles:
                s = i % NS
                off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
                t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
                pg = gl.load(pt_row + t // PAGE)
                mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
                cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
                cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
            cp.commit_group()
            if i >= LAG:
                sj = (i - LAG) % NS
                cp.wait_group(LAG)
                gl.barrier()
                vt = v_ring.index(sj)
                vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                fence_async_shared()
                # (mbarrier.arrive syncs the partition before its elected arrive.)
                mbarrier.arrive(ready.index(sj))

    @gluon.jit
    def _gw_consumer_cv(q, k_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh, split,
                        HEAD: gl.constexpr, HKV: gl.constexpr, S: gl.constexpr, M: gl.constexpr, D: gl.constexpr,
                        BN: gl.constexpr, NS: gl.constexpr, PEEL: gl.constexpr, LSUM: gl.constexpr = False):
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
        acc = gl.zeros([M, D], gl.float32, layout=O_L)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        if LSUM:
            # Row sums kept per thread ([M, 8]: the 8 columns of each 8-wide group a quad holds), reduced
            # across the quad once at the end, as FA3 does.
            l8 = gl.sum(gl.reshape(gl.zeros([M, BN], gl.float32, S_L), [M, BN // 8, 8]), 1)
        for i in range(ntiles):
            s = i % NS
            mbarrier.wait(ready.index(s), (i // NS) & 1)
            s_tok = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                  gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            sc = warpgroup_mma_wait(0, deps=[s_tok])
            if PEEL:
                # Only a chunk's last tile can run past its keys.
                if lo + (i + 1) * BN > hi:
                    sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            else:
                sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            if LSUM:
                l8 = l8 * gl.convert_layout(alpha, gl.SliceLayout(1, l8.type.layout))[:, None] + \
                    gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)
            else:
                l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
            mbarrier.arrive(empty.index(s))
        if LSUM:
            l_i = gl.convert_layout(gl.sum(l8, 1), gl.SliceLayout(1, S_L))
        _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV, S + 1, M, D, gl.SliceLayout(1, S_L), O_L)

    # ---- Balanced splits (plan gbal): the batch's key tiles cut into equal ranges, one per program ----
    # Program p (of P per KV head pair) takes tiles [p * T // P, (p + 1) * T // P) of the requests' tiles laid
    # end to end, so a range can span requests: one segment per request it touches, each stored at partial
    # id p + b (unique: moving to the next segment bumps p or b). Ids P + B + b hold the new tokens.
    @gluon.jit
    def _gw_bal(seq_ptr, p, B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, BN: gl.constexpr,
                NW: gl.constexpr):
        # All B_PAD lengths in each warp (replicated across warps): the scan and sums stay inside a warp.
        LY: gl.constexpr = gl.BlockedLayout([max(1, B_PAD // 32)], [32], [NW], [0])
        idx = gl.arange(0, B_PAD, layout=LY)
        seqv = gl.load(seq_ptr + idx, mask=idx < B, other=0).to(gl.int32)
        tiles = (seqv + (BN - 1)) // BN
        incl = gl.associative_scan(tiles, 0, _gw_add)
        tt = gl.maximum(gl.sum(tiles, 0), 1)
        start = p * tt // P
        end = (p + 1) * tt // P
        b_first = gl.sum((incl <= start).to(gl.int32), 0)
        b_end = gl.sum(((incl - tiles < end) & (idx < B)).to(gl.int32), 0)
        nseg = gl.where(end > start, b_end - b_first, 0)
        return idx, seqv, tiles, start, end, b_first, nseg

    # Lazy-max softmax for one [64, 64] S tile of a consumer warpgroup (S_L layout: pack slot j of a thread is row
    # r + 8 * ((j >> 1) & 1) of its quad's two rows; tools/s79/order.py). Operands: out p[32], m_new[32]; in s[32],
    # m[32] (row maxima broadcast), scale[32] (broadcast). Each thread exponentiates against the running maxima
    # (no row max, no rescale); since P >= 0, all its partial row sums <= 2^15 implies every P <= 2^15 (FP16-safe,
    # same relative precision, no lost range). A warp where any thread fails (first tile: maxima -1e30 -> inf; a row
    # grew by > 15 binades; NaN) recomputes its rows with the usual update (row max over the quad by shuffles), which
    # is per-warp legal: every row's elements live in one quad. Exact either way.
    def _lz_asm():
        a = ["{", ".reg .f32 q<32>, sa, sb, na, nb, xa, xb, t;", ".reg .pred ok, all;",
             "neg.f32 na, $96;", "neg.f32 nb, $98;"]
        for j in range(32):
            n = "nb" if j & 2 else "na"
            a += [f"fma.rn.f32 q{j}, ${64 + j}, $128, {n};", f"ex2.approx.ftz.f32 q{j}, q{j};"]
        ra = [j for j in range(32) if not j & 2]
        rb = [j for j in range(32) if j & 2]
        a += [f"add.f32 sa, q{ra[0]}, q{ra[1]};", f"add.f32 sb, q{rb[0]}, q{rb[1]};"]
        a += [f"add.f32 sa, sa, q{j};" for j in ra[2:]] + [f"add.f32 sb, sb, q{j};" for j in rb[2:]]
        a += ["setp.le.f32 ok, sa, 0f47000000;", "setp.le.and.f32 ok, sb, 0f47000000, ok;",
              "vote.sync.all.pred all, ok, 0xffffffff;", "mov.f32 xa, $96;", "mov.f32 xb, $98;", "@all bra LZ_DONE;"]
        a += [f"max.f32 xa, ${64 + ra[0]}, ${64 + ra[1]};", f"max.f32 xb, ${64 + rb[0]}, ${64 + rb[1]};"]
        a += [f"max.f32 xa, xa, ${64 + j};" for j in ra[2:]] + [f"max.f32 xb, xb, ${64 + j};" for j in rb[2:]]
        for o in (1, 2):
            a += [f"shfl.sync.bfly.b32 t, xa, {o}, 0x1f, 0xffffffff;", "max.f32 xa, xa, t;",
                  f"shfl.sync.bfly.b32 t, xb, {o}, 0x1f, 0xffffffff;", "max.f32 xb, xb, t;"]
        a += ["mul.f32 xa, xa, $128;", "mul.f32 xb, xb, $128;", "max.f32 xa, xa, $96;", "max.f32 xb, xb, $98;",
              "neg.f32 na, xa;", "neg.f32 nb, xb;"]
        for j in range(32):
            n = "nb" if j & 2 else "na"
            a += [f"fma.rn.f32 q{j}, ${64 + j}, $128, {n};", f"ex2.approx.ftz.f32 q{j}, q{j};"]
        a += ["LZ_DONE:"] + [f"mov.f32 ${j}, q{j};" for j in range(32)]
        a += [f"mov.f32 ${32 + j}, {'xb' if j & 2 else 'xa'};" for j in range(32)] + ["}"]
        return "\n".join(a)

    def _lz_asm2():
        # LZ 2: decide before exponentiating, from each thread's own row maxima (no shuffles): if every thread of
        # the warp has max(s) * scale - m <= 15 on both rows (so P <= 2^15), exponentiate against the stale maxima;
        # else the usual update for the warp's rows. The scores are dead once P is computed (no spills).
        a = ["{", ".reg .f32 q<32>, na, nb, xa, xb, t;", ".reg .pred ok, all;"]
        ra = [j for j in range(32) if not j & 2]
        rb = [j for j in range(32) if j & 2]
        # (Serial chains: a depth-4 tree measured +0.8 us all-L2.)
        a += [f"max.f32 xa, ${64 + ra[0]}, ${64 + ra[1]};", f"max.f32 xb, ${64 + rb[0]}, ${64 + rb[1]};"]
        a += [f"max.f32 xa, xa, ${64 + j};" for j in ra[2:]] + [f"max.f32 xb, xb, ${64 + j};" for j in rb[2:]]
        a += ["fma.rn.f32 t, xa, $128, 0fC1700000;", "setp.le.f32 ok, t, $96;",
              "fma.rn.f32 t, xb, $128, 0fC1700000;", "setp.le.and.f32 ok, t, $98, ok;",
              "vote.sync.all.pred all, ok, 0xffffffff;", "@all bra LZ_FAST;"]
        for o in (1, 2):
            a += [f"shfl.sync.bfly.b32 t, xa, {o}, 0x1f, 0xffffffff;", "max.f32 xa, xa, t;",
                  f"shfl.sync.bfly.b32 t, xb, {o}, 0x1f, 0xffffffff;", "max.f32 xb, xb, t;"]
        a += ["mul.f32 xa, xa, $128;", "mul.f32 xb, xb, $128;", "max.f32 xa, xa, $96;", "max.f32 xb, xb, $98;",
              "bra LZ_EXP;", "LZ_FAST:", "mov.f32 xa, $96;", "mov.f32 xb, $98;", "LZ_EXP:",
              "neg.f32 na, xa;", "neg.f32 nb, xb;"]
        for j in range(32):
            n = "nb" if j & 2 else "na"
            a += [f"fma.rn.f32 q{j}, ${64 + j}, $128, {n};", f"ex2.approx.ftz.f32 q{j}, q{j};"]
        a += [f"mov.f32 ${j}, q{j};" for j in range(32)]
        a += [f"mov.f32 ${32 + j}, {'xb' if j & 2 else 'xa'};" for j in range(32)] + ["}"]
        return "\n".join(a)

    def _lz_acc_asm():
        # acc *= alpha (per row; same slot pattern in O_L), skipped when the whole warp has alpha == 1 (LZ 3).
        a = ["{", ".reg .pred p, any;", "setp.neu.f32 p, $128, 0f3F800000;", "setp.neu.or.f32 p, $130, 0f3F800000, p;",
             "vote.sync.any.pred any, p, 0xffffffff;"]
        a += [f"mov.f32 ${j}, ${64 + j};" for j in range(64)] + ["@!any bra LZ_SKIP;"]
        a += [f"mul.f32 ${j}, ${j}, {'$130' if j & 2 else '$128'};" for j in range(64)] + ["LZ_SKIP:", "}"]
        return "\n".join(a)

    LZ_PK = gl.constexpr(4)  # LZ 10: exact tiles per segment
    LZ_DELTA = gl.constexpr(14.0)  # LZ 10: FP16 headroom (binades)
    LZ_ACC = gl.constexpr(_lz_acc_asm())
    LZ_ACC_CON = gl.constexpr(",".join(["=f"] * 64 + ["f"] * 128))
    LZ_ASM = gl.constexpr(_lz_asm())
    LZ_ASM2 = gl.constexpr(_lz_asm2())
    LZ_CON = gl.constexpr(",".join(["=f"] * 64 + ["f"] * 96))

    @gluon.jit
    def _gw_pick(a, b):
        return a

    @gluon.jit
    def _gw_add(a, b):
        return a + b

    @gluon.jit
    def _gw_seg(idx, seqv, tiles, start, end, b, BN: gl.constexpr):
        """Segment of request b in tile range [start, end): its keys [lo, hi) and tile count."""
        excl = gl.sum(gl.where(idx < b, tiles, 0), 0)
        nb = gl.sum(gl.where(idx == b, tiles, 0), 0)
        sb = gl.sum(gl.where(idx == b, seqv, 0), 0)
        t0 = gl.maximum(start, excl) - excl
        t1 = gl.minimum(end, excl + nb) - excl
        lo = t0 * BN
        hi = gl.minimum(t1 * BN, sb)
        return lo, hi, gl.maximum(t1 - t0, 0)

    @gluon.jit
    def _gw_producer_bal(seq_ptr, pt_ptr, kbase, vbase, k_ring, v_ring, vh0, vh1, ready, empty, p,
                         B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, MAX_PAGES: gl.constexpr,
                         PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                         D: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr, LAG: gl.constexpr, GX: gl.constexpr = 0):
        """_gw_producer_cv over the program's segments (the ring position carries across them).
        GX (layer s84 DIAG probes, wrong results, tuner timing only): 1 = no V widen once every ring slot holds one,
        2 = every tile re-gathers the segment's first rows (all-L2), 4 = both."""
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        idx, seqv, tiles, start, end, b_first, nseg = _gw_bal(seq_ptr, p, B, B_PAD, P, BN, PW)
        it = 0
        for sg in range(nseg):
            b = b_first + sg
            lo, hi, ntiles = _gw_seg(idx, seqv, tiles, start, end, b, BN)
            pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES
            t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
            pg = gl.load(pt_row + t // PAGE)
            for i in range(ntiles + LAG):
                if i < ntiles:
                    s = (it + i) % NS
                    off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
                    if GX == 2 or GX == 4:
                        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
                    else:
                        t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
                    pg = gl.load(pt_row + t // PAGE)
                    mbarrier.wait(empty.index(s), (((it + i) // NS) & 1) ^ 1)
                    cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
                    cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
                cp.commit_group()
                if i >= LAG:
                    if i - LAG < ntiles:
                        sj = (it + i - LAG) % NS
                        cp.wait_group(LAG)
                        gl.barrier()
                        vt = v_ring.index(sj)
                        if GX == 1 or GX == 4:
                            if it + i - LAG < NS:
                                vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                                vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                        else:
                            vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                            vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                        fence_async_shared()
                        mbarrier.arrive(ready.index(sj))
            it += ntiles

    @gluon.jit
    def _gw_consumer_bal(qkv_ptr, q_s, k_ring, vh, ready, empty, seq_ptr, pid, h, scale, pacc_ptr, pml_ptr,
                         HEAD: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr,
                         B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, NID: gl.constexpr, M: gl.constexpr,
                         D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr, LZ: gl.constexpr = 0,
                         GX: gl.constexpr = 0):
        """(GX 3, layer s84 DIAG: no math after the exact tiles, wrong results.) _gw_consumer_cv (PEEL, LSUM) over the program's segments: each loads its request's queries and stores
        its partial at id p + b."""
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
        cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        idx, seqv, tiles, start, end, b_first, nseg = _gw_bal(seq_ptr, pid, B, B_PAD, P, BN, 4)
        it = 0
        for sg in range(nseg):
            b = b_first + sg
            lo, hi, ntiles = _gw_seg(idx, seqv, tiles, start, end, b, BN)
            x = gl.load(qkv_ptr + (b * L + r // G)[:, None] * QKV_W + ((h + HEAD) * G + r % G)[:, None] * D
                        + cq[None, :]).to(gl.float32)
            q_s.store((x * gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv))
            fence_async_shared()
            gl.barrier()
            m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
            acc = gl.zeros([M, D], gl.float32, layout=O_L)
            l8 = gl.sum(gl.reshape(gl.zeros([M, BN], gl.float32, S_L), [M, BN // 8, 8]), 1)
            if LZ == 10:
                # Fixed-reference softmax (session 105): the first LZ_PK tiles run the exact update against their
                # row maxima + LZ_DELTA binades; later tiles exponentiate against that fixed reference (no row max,
                # no alpha, no acc rescale, no vote: the s78 no-max loop). FP16 P overflows only if a row's scores
                # grow > 16 + LZ_DELTA binades past the first tiles' max (scored inputs: <= 23.2 at LZ_PK 4, 7 seeds,
                # tools/s105/growth.py); such rows end with a non-finite acc, store m = +inf, and _combine_bal REDO
                # recomputes them exactly. Precision: with the max P at 2^-LZ_DELTA the FP16 (subnormal) step is
                # 2^(LZ_DELTA - 25) of the max, still >= 2x finer than E4M3 P against the exact max (FA3, SGLang).
                for i in range(gl.minimum(ntiles, LZ_PK)):
                    s = (it + i) % NS
                    mbarrier.wait(ready.index(s), ((it + i) // NS) & 1)
                    s_tok = warpgroup_mma(q_s, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                          gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
                    sc = warpgroup_mma_wait(0, deps=[s_tok])
                    if lo + (i + 1) * BN > hi:
                        sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
                    m_new = gl.maximum(m_i, gl.max(sc, 1) * scale + LZ_DELTA)
                    p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
                    alpha = gl.exp2(m_i - m_new)
                    l8 = l8 * gl.convert_layout(alpha, gl.SliceLayout(1, l8.type.layout))[:, None] + \
                        gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)
                    m_i = m_new
                    acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
                    p16 = gl.convert_layout(p.to(gl.float16), P_L)
                    a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
                    acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
                    mbarrier.arrive(empty.index(s))
                for i in range(LZ_PK, ntiles if GX != 3 else LZ_PK):
                    s = (it + i) % NS
                    mbarrier.wait(ready.index(s), ((it + i) // NS) & 1)
                    s_tok = warpgroup_mma(q_s, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                          gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
                    sc = warpgroup_mma_wait(0, deps=[s_tok])
                    if lo + (i + 1) * BN > hi:
                        sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
                    p = gl.exp2(gl.fma(sc, scale, -m_i[:, None]))
                    l8 = l8 + gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)
                    p16 = gl.convert_layout(p.to(gl.float16), P_L)
                    a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
                    acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
                    mbarrier.arrive(empty.index(s))
                if GX == 3:
                    for i in range(LZ_PK, ntiles):
                        s = (it + i) % NS
                        mbarrier.wait(ready.index(s), ((it + i) // NS) & 1)
                        mbarrier.arrive(empty.index(s))
                # (x - x != 0 exactly for inf and NaN.)
                asum = gl.convert_layout(gl.sum(acc, 1), gl.SliceLayout(1, S_L))
                m_i = gl.where(asum - asum == 0.0, m_i, float("inf"))
            else:
                for i in range(ntiles):
                    s = (it + i) % NS
                    mbarrier.wait(ready.index(s), ((it + i) // NS) & 1)
                    s_tok = warpgroup_mma(q_s, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                          gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
                    sc = warpgroup_mma_wait(0, deps=[s_tok])
                    if lo + (i + 1) * BN > hi:
                        sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
                    if LZ:
                        p, me = gl.inline_asm_elementwise(
                            LZ_ASM if LZ == 1 else LZ_ASM2, LZ_CON, [sc, gl.zeros([M, BN], gl.float32, S_L) + m_i[:, None],
                                             gl.zeros([M, BN], gl.float32, S_L) + scale],
                            dtype=(gl.float32, gl.float32), is_pure=True, pack=32)
                        m_new = gl.reduce(me, 1, _gw_pick)
                    else:
                        m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
                        p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
                    alpha = gl.exp2(m_i - m_new)
                    l8 = l8 * gl.convert_layout(alpha, gl.SliceLayout(1, l8.type.layout))[:, None] + \
                        gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)
                    m_i = m_new
                    if LZ >= 3:
                        acc = gl.inline_asm_elementwise(
                            LZ_ACC, LZ_ACC_CON, [acc, gl.zeros([M, D], gl.float32, O_L) +
                                                 gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]],
                            dtype=gl.float32, is_pure=True, pack=64)
                    else:
                        acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
                    p16 = gl.convert_layout(p.to(gl.float16), P_L)
                    a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
                    acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
                    mbarrier.arrive(empty.index(s))
            it += ntiles
            l_i = gl.convert_layout(gl.sum(l8, 1), gl.SliceLayout(1, S_L))
            _gw_store(pacc_ptr, pml_ptr, 0, h + HEAD, pid + b, m_i, l_i, acc, HKV, NID, M, D, gl.SliceLayout(1, S_L),
                      O_L)

    # ---- Decode steps (q_len 1) on full layers at batch * hkv > 2 (Gluon, one warp per program) ----
    # Like _attn_split's hpp-2 programs (one split of one request's keys for a KV head pair, block-diagonal
    # queries [2G, 2D] against whole 128-byte rows), but QK^T runs as FP8 mma.sync (Triton's MMAv2 path upcasts
    # every K byte to FP16 on sm_90), V is read from shared memory 16 bytes at a time, and the pair's new token is
    # written and attended by the last split into its own partial slot S (no separate launch). Same bytes, ~15-20%
    # less work per key (H100, all reads from L2: decode 16/32 full 84/157 -> 69/132 us), which pays where
    # per-SM bandwidth is higher (tools/emu.py: 137/261 -> 132/249 us; H100 itself: a tie).
    @gluon.jit
    def _gqld(p, QSS: gl.constexpr):
        # proj s22 (QSS): qkv as 2 FP32 split-K slabs QSS elements apart: add them in FP32 (then the caller rounds to BF16)
        x = gl.load(p)
        if QSS != 0:
            x = x + gl.load(p + QSS)
        return x

    @gluon.jit
    def _gqldm(p, m, QSS: gl.constexpr):
        x = gl.load(p, mask=m, other=0.0)
        if QSS != 0:
            x = x + gl.load(p + QSS, mask=m, other=0.0)
        return x

    @gluon.jit
    def _gd_full_body(seq_in, SEQ_IN: gl.constexpr, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                 cs_ptr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
                 MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
                 TPV: gl.constexpr = False, WINDOW: gl.constexpr = 0, ROPE: gl.constexpr = False,
                 SEP: gl.constexpr = False, Q16: gl.constexpr = True, KST: gl.constexpr = True,
                 BAL: gl.constexpr = False, B: gl.constexpr = 1, B_PAD: gl.constexpr = 32, NP: gl.constexpr = 1,
                 NID: gl.constexpr = 1, RED: gl.constexpr = False, red_ptr=None, rcnt_ptr=None, out_ptr=None,
                 NR: gl.constexpr = 4, epi=True, RCW: gl.constexpr = 0, QSS: gl.constexpr = 0, DX: gl.constexpr = 0):
        # DX (layer s84 DIAG probes, wrong results, tuner timing only): 1 = no math after tile 0 (gathers only),
        # 2 = every tile re-gathers the same rows (all-L2).
        h = gl.program_id(2) * 2
        if BAL:
            # Length-proportional splits (plan dbal, full layers): the NP programs of a KV head pair go to the
            # requests in proportion to their tiles, request b taking ids [c_b, c_b+1) with
            # c_b = b + (NP - B) * excl_b // T (at least one each), so every program gets about T / (NP - B)
            # tiles and none spans requests. Partials by id: key partial p at p + b, new token at NP + B + b
            # (_combine_bal's layout).
            pid = gl.program_id(0)
            LY: gl.constexpr = gl.BlockedLayout([max(1, B_PAD // 32)], [32], [1], [0])
            idx = gl.arange(0, B_PAD, layout=LY)
            tl_v = (gl.load(seq_ptr + idx, mask=idx < B, other=0).to(gl.int32) + (BN - 1)) // BN
            excl_v = gl.associative_scan(tl_v, 0, _gw_add) - tl_v
            tt = gl.maximum(gl.sum(tl_v, 0), 1)
            c_v = idx + (NP - B) * excl_v // tt
            b = gl.sum(((c_v <= pid) & (idx < B)).to(gl.int32), 0) - 1
            c_b = gl.sum(gl.where(idx == b, c_v, 0), 0)
            c_n = gl.sum(gl.where(idx == b + 1, c_v, 0), 0)
            c_n = gl.where(b + 1 < B, c_n, NP)
            split = pid - c_b
            nsp = c_n - c_b
        else:
            split = gl.program_id(0)
            b = gl.program_id(1)
            nsp = S
        # Let the merge kernel launch now (programmatic dependent launch); it waits for this grid.
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False,
                                  pack=1)
        C: gl.constexpr = 2 * D
        M: gl.constexpr = 2 * G
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        ROW_W: gl.constexpr = HKV * D
        MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 1], instr_shape=[16, 8])
        QA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=4)
        KB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=4)
        PA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
        VB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
        VA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
        PB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
        # 16 bytes per thread. (Converting FP8 queries into QA from a [1, 4]-per-thread layout gave wrong values.)
        BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [1, 1], [1, 0])
        SH: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])

        if SEQ_IN:
            seq = seq_in
        else:
            seq = gl.load(seq_ptr + b)
        # Each request's cached keys (sliding layers: the last WINDOW - 1) in S equal tile-aligned chunks.
        if WINDOW > 0:
            base = gl.maximum(seq - (WINDOW - 1), 0)
        else:
            base = seq * 0
        chunk = gl.cdiv(gl.cdiv(seq - base, nsp), BN) * BN
        lo = base + split * chunk
        hi = gl.minimum(lo + chunk, seq)
        ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
        pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES
        kbase = kc_ptr + h * D
        vbase = vc_ptr + h * D
        k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH)
        v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH)
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, BL))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, BL))
        all_c = c_g[None, :] >= 0
        # The first NS - 1 tiles in flight (cp.async zero-fills rows past hi instead of reading them).
        for j in gl.static_range(NS - 1):
            t = lo + j * BN + n_g
            ok = t < hi
            pg = gl.load(pt_row + t // PAGE, mask=ok, other=0)
            if SEQ_IN:
                pg = gl.maximum(pg, 0)  # speculative length: entries past the real one are -1
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            cp.async_copy_global_to_shared(k_ring.index(j), kbase + off, mask=ok[:, None] & all_c)
            cp.async_copy_global_to_shared(v_ring.index(j), vbase + off, mask=ok[:, None] & all_c)
            cp.commit_group()
        t = lo + (NS - 1) * BN + n_g
        pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)
        if SEQ_IN:
            pg = gl.maximum(pg, 0)  # speculative length: entries past the real one are -1

        # Block-diagonal queries: row r is query head (h + r // G) * G + r % G, nonzero only in its own KV head's
        # D columns. RMSNorm, then E4M3.
        r = gl.arange(0, M, layout=gl.SliceLayout(1, BL))
        hp = r // G
        q_ok = (c_g // D)[None, :] == hp[:, None]
        x = _gqldm(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + (c_g % D)[None, :], q_ok,
                   QSS).to(gl.bfloat16).to(gl.float32)
        x = gl.where(q_ok, x, 0.0)
        rs = gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]
        if ROPE:
            # Interleaved RoPE at position seq (the norm commutes with the rotation).
            xp = _gqldm(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + ((c_g % D) ^ 1)[None, :],
                        q_ok, QSS).to(gl.bfloat16).to(gl.float32)
            cs_off = seq * D + (c_g % D) // 2
            cos = gl.load(cs_ptr + cs_off)[None, :]
            sin = (gl.load(cs_ptr + cs_off + D // 2) * gl.where(c_g % 2 == 0, -1.0, 1.0))[None, :]
            x = x * cos + xp * sin
        x8 = (x * rs).to(gl.float8e4nv)
        q = gl.convert_layout(x8, QA)
        if Q16:
            # Queries as FP16 once (exact E4M3 values) and K upcast per tile: on sm_90 ptxas lowers the FP8 mma.sync
            # to FP16 HMMAs (+ FADDs), re-converting the loop-invariant queries every tile. Bit-identical results,
            # 13% fewer instructions (H100 all-L2 decode 16/32 full 69.2/133.8 -> 67.7/129.1 us, real -0.4/-0.7).
            q = q.to(gl.float16)
        k_scale = gl.load(ks_ptr)
        scale = qk_scale * k_scale

        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, MMA))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, MMA))
        if TPV:
            # Transposed P@V per head: O^T [D, G] += V^T [D, BN] @ P^T [BN, G] (only the diagonal blocks: half the
            # MMAs and accumulator registers of the block-diagonal form). V staged once per tile as FP16.
            vh = gl.allocate_shared_memory(gl.float16, [BN, C], gl.SwizzledSharedLayout(8, 1, 8, [1, 0]))
            acc0 = gl.zeros([D, G], gl.float32, layout=MMA)
            acc1 = gl.zeros([D, G], gl.float32, layout=MMA)
        elif KST:
            # Heads stacked along the P@V reduction: acc [M, D] = P_0 @ V_0 + P_1 @ V_1, P_j = P with the other
            # head's rows zeroed. Same MMAs as the block-diagonal form, half the accumulator and alpha multiplies.
            # Bit-identical; H100 all-L2 decode 16/32/64 full 66.7/126.7/253.9 -> 63.4/120.4/241.8 us, real tie.
            acc = gl.zeros([M, D], gl.float32, layout=MMA)
            row0 = (gl.arange(0, M, layout=gl.SliceLayout(1, PA)) < G)[:, None]
        else:
            acc = gl.zeros([M, C], gl.float32, layout=MMA)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA))
        for i in range(ntiles):
            sj = (i + NS - 1) % NS
            ok = t < hi
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            gl.barrier()  # slot sj was read last iteration
            cp.async_copy_global_to_shared(k_ring.index(sj), kbase + off, mask=ok[:, None] & all_c)
            cp.async_copy_global_to_shared(v_ring.index(sj), vbase + off, mask=ok[:, None] & all_c)
            cp.commit_group()
            if DX != 2:
                t = t + BN
            pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)
            if SEQ_IN:
                pg = gl.maximum(pg, 0)  # speculative length: entries past the real one are -1
            cp.wait_group(NS - 1)
            gl.barrier()
            if (i == 0) | (DX != 1):
                s = i % NS
                kb = k_ring.index(s).permute((1, 0)).load(KB)
                if Q16:
                    kb = kb.to(gl.float16)
                sc = mma_v2(q, kb, gl.zeros([M, BN], gl.float32, MMA))
                if lo + (i + 1) * BN > hi:
                    sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
                m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
                alpha = gl.exp2(m_i - m_new)
                p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
                l_i = l_i * alpha + gl.sum(p, 1)
                m_i = m_new
                if TPV:
                    a0, a1 = gl.split(gl.permute(gl.reshape(alpha, [2, G]), (1, 0)))
                    acc0 = acc0 * gl.convert_layout(a0, gl.SliceLayout(0, MMA))[None, :]
                    acc1 = acc1 * gl.convert_layout(a1, gl.SliceLayout(0, MMA))[None, :]
                    p0, p1 = gl.split(gl.permute(gl.reshape(p.to(gl.float16), [2, G, BN]), (2, 1, 0)))
                    vh.store(v_ring.index(s).load(BL).to(gl.float16))
                    gl.barrier()
                    acc0 = mma_v2(vh.slice(0, D, 1).permute((1, 0)).load(VA), gl.convert_layout(p0, PB), acc0)
                    acc1 = mma_v2(vh.slice(D, D, 1).permute((1, 0)).load(VA), gl.convert_layout(p1, PB), acc1)
                elif KST:
                    acc = acc * alpha[:, None]
                    ph = gl.convert_layout(p.to(gl.float16), PA)
                    # The FP16 V tile as one B operand, then split per head in registers (loading the two [BN, D] halves
                    # from smem separately costs register shuffles: all-L2 +3%).
                    v = gl.convert_layout(v_ring.index(s).load(BL).to(gl.float16), VB)
                    v0, v1 = gl.split(gl.permute(gl.reshape(v, [BN, 2, D]), (0, 2, 1)))
                    v0 = gl.convert_layout(v0, VB)
                    v1 = gl.convert_layout(v1, VB)
                    acc = mma_v2(gl.where(row0, ph, 0.0), v0, acc)
                    acc = mma_v2(gl.where(row0, 0.0, ph), v1, acc)
                else:
                    acc = acc * alpha[:, None]
                    v = gl.convert_layout(v_ring.index(s).load(BL).to(gl.float16), VB)
                    acc = mma_v2(gl.convert_layout(p.to(gl.float16), PA), v, acc)
        cp.wait_group(0)

        # Partials [B, HKV, S + 1, G, D]: each row's own head's D columns of acc.
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
        co = gl.arange(0, C, layout=gl.SliceLayout(0, MMA))
        if BAL:
            part = ((h + ro // G) * NID + pid + b) * G + ro % G
        else:
            part = ((b * HKV + h + ro // G) * (S + 1) + split) * G + ro % G
        own = (co // D)[None, :] == (ro // G)[:, None]
        # SEP: an extra program (split S) only handles the new token; it stores no key partial.
        keyp = split < nsp
        own = own & keyp
        if RED:
            # The merge in this grid (plan red, batch 1, one KV head pair): every program adds its partial into
            # FP32 sums relative to the fixed reference max 0 (exact while its max m is in [-60, 100]: no
            # overflow, nothing it contributes falls below FP32's normal range); out-of-range programs store
            # their partial instead and the last program merges those exactly.
            bad = (l_i > 0.0) & ((m_i > 100.0) | (m_i < -60.0))
            flg = gl.max(bad.to(gl.int32), 0) > 0
            keyp = keyp & flg
            # Batched (s103): request b's sums and out-of-range list (RCW words) at its own offsets; listed entries
            # are partial slots (BAL: ids).
            if RCW > 0:
                red_ptr = red_ptr + b * (NR * M * (D + 1))
                rcnt_ptr = rcnt_ptr + b * RCW
            if BAL:
                kid = pid + b
                nid_new = NP + B + b
            else:
                kid = split
                nid_new = S
        if TPV:
            dd = gl.arange(0, D, layout=gl.SliceLayout(1, MMA))
            gg = gl.arange(0, G, layout=gl.SliceLayout(0, MMA))
            pt0 = ((b * HKV + h) * (S + 1) + split) * G + gg
            gl.store(pacc_ptr + pt0[None, :] * D + dd[:, None], acc0.to(pacc_ptr.dtype.element_ty))
            gl.store(pacc_ptr + (pt0 + (S + 1) * G)[None, :] * D + dd[:, None], acc1.to(pacc_ptr.dtype.element_ty))
        elif KST:
            cd = gl.arange(0, D, layout=gl.SliceLayout(0, MMA))
            gl.store(pacc_ptr + part[:, None] * D + cd[None, :], acc.to(pacc_ptr.dtype.element_ty),
                     mask=keyp & (cd[None, :] >= 0))
        else:
            gl.store(pacc_ptr + part[:, None] * D + (co % D)[None, :], acc.to(pacc_ptr.dtype.element_ty), mask=own)
        gl.store(pml_ptr + part * 2, m_i, mask=keyp)
        gl.store(pml_ptr + part * 2 + 1, l_i, mask=keyp)
        if split == (nsp if SEP else nsp - 1):
            # The step's new token for both KV heads: normalize K, store K and V as FP8 at cache_loc (row 0 of
            # each head's block), and write its one-key partial (max = its score, sum = 1, acc = V) to slot S.
            loc = gl.load(loc_ptr + b)
            zero = gl.zeros([M, C], gl.float32, BL)
            kx = gl.where(q_ok, _gqld(qkv_ptr + b * QKV_W + (HQ + h) * D + c_g, QSS)[None, :].to(gl.bfloat16).to(gl.float32) + zero, 0.0)
            krs = gl.rsqrt(gl.sum(kx * kx, 1) / D + 1e-5)[:, None]
            if ROPE:
                kxp = gl.where(q_ok, _gqld(qkv_ptr + b * QKV_W + (HQ + h) * D + (c_g ^ 1), QSS)[None, :].to(gl.bfloat16).to(gl.float32)
                               + zero, 0.0)
                kx = kx * cos + kxp * sin
            kq = (kx * krs / k_scale).to(gl.float8e4nv)
            vx = _gqld(qkv_ptr + b * QKV_W + (HQ + HKV + h) * D + c_g, QSS)[None, :].to(gl.bfloat16).to(gl.float32) + zero
            vq = (gl.where(q_ok, vx, 0.0) / gl.load(vs_ptr)).to(gl.float8e4nv)
            first = q_ok & ((r % G) == 0)[:, None]
            dst = (loc.to(gl.int64) * ROW_W + h * D + c_g)[None, :] + gl.zeros([M, C], gl.int64, BL)
            gl.store(kc_ptr + dst, kq, mask=first)
            gl.store(vc_ptr + dst, vq, mask=first)
            sn = gl.sum(x8.to(gl.float32) * kq.to(gl.float32), 1) * scale
            if BAL:
                pn = ((h + hp) * NID + NP + B + b) * G + r % G
            else:
                pn = ((b * HKV + h + hp) * (S + 1) + S) * G + r % G
            gl.store(pacc_ptr + pn[:, None] * D + (c_g % D)[None, :], vq.to(gl.float32).to(pacc_ptr.dtype.element_ty),
                     mask=q_ok)
            gl.store(pml_ptr + pn * 2, sn)
            gl.store(pml_ptr + pn * 2 + 1, gl.full([M], 1.0, gl.float32, gl.SliceLayout(1, BL)))
            if RED:
                if epi:
                    nbad = gl.max(((sn > 100.0) | (sn < -60.0)).to(gl.int32), 0) > 0
                    if nbad:
                        gl.store(rcnt_ptr + 2 + gl.atomic_add(rcnt_ptr + 1, 1, sem="relaxed", scope="gpu"), nid_new)
                    else:
                        fn = gl.exp2(sn)
                        nrep = (split + 1) % NR
                        gl.atomic_add(red_ptr + ((nrep * M + r) * D)[:, None] + (c_g % D)[None, :],
                                      vq.to(gl.float32) * fn[:, None], mask=q_ok, sem="relaxed", scope="gpu")
                        gl.atomic_add(red_ptr + NR * M * D + nrep * M + r, fn, mask=r >= 0, sem="relaxed",
                                      scope="gpu")
        if RED:
            if epi:
                _gd_red_add(acc, m_i, l_i, flg, split, kid, red_ptr, rcnt_ptr, M, D, NR)

    @gluon.jit
    def _gd_red_add(acc, m_i, l_i, flg, split, kid, red_ptr, rcnt_ptr, M: gl.constexpr, D: gl.constexpr,
                    NR: gl.constexpr):
        """plan red: add this program's partial (acc [M, D], m, l in the mma layout) into replica split % NR of the
        sums at reference max 0, or, if its max is out of range (flg), list it (its partial is stored) for _red_fin."""
        MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 1], instr_shape=[16, 8])
        ML: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [1, 1], [1, 0])
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
        rep = split % NR
        if flg:
            gl.store(rcnt_ptr + 2 + gl.atomic_add(rcnt_ptr + 1, 1, sem="relaxed", scope="gpu"), kid)
        else:
            f = gl.where(l_i > 0.0, gl.exp2(m_i), 0.0)
            # 4 consecutive floats per thread: v4 atomics, half the L2 atomic operations of the mma layout's pairs.
            av = gl.convert_layout(acc * f[:, None], ML)
            r4 = gl.arange(0, M, layout=gl.SliceLayout(1, ML))
            c4 = gl.arange(0, D, layout=gl.SliceLayout(0, ML))
            gl.atomic_add(red_ptr + ((rep * M + r4) * D)[:, None] + c4[None, :], av, sem="relaxed", scope="gpu")
            gl.atomic_add(red_ptr + NR * M * D + rep * M + ro, l_i * f, sem="relaxed", scope="gpu")

    # The rare redo path of _gd_full's SPEC, out of line.
    _gd_full_redo = gluon.jit(noinline=True)(_gd_full_body.fn)

    @gluon.jit
    def _gd_full(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                 cs_ptr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
                 MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
                 TPV: gl.constexpr = False, WINDOW: gl.constexpr = 0, ROPE: gl.constexpr = False,
                 SEP: gl.constexpr = False, Q16: gl.constexpr = True, KST: gl.constexpr = True,
                 BAL: gl.constexpr = False, B: gl.constexpr = 1, B_PAD: gl.constexpr = 32, NP: gl.constexpr = 1,
                 NID: gl.constexpr = 1, SPEC: gl.constexpr = 0, RED: gl.constexpr = False, red_ptr=None,
                 rcnt_ptr=None, out_ptr=None, NR: gl.constexpr = 4, RCW: gl.constexpr = 0, QSS: gl.constexpr = 0,
                 DX: gl.constexpr = 0):
        # SPEC > 0 (batch 1, page 1): seq = table width - 1 known from the shape (SGLang slices the page table to the
        # batch's longest length); start on it, redo with the real length if it differs (as in _attn_split).
        if SPEC > 0:
            real = gl.load(seq_ptr + gl.program_id(1))
            _gd_full_body(SPEC, True, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale, cs_ptr, HQ, HKV, G, D, PAGE, MAX_PAGES, S, BN, NS, TPV, WINDOW, ROPE, SEP, Q16, KST, BAL, B, B_PAD, NP, NID, RED, red_ptr, rcnt_ptr, out_ptr, NR, real == SPEC, RCW, QSS, DX)
            if real != SPEC:
                _gd_full_redo(real, True, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale, cs_ptr, HQ, HKV, G, D, PAGE, MAX_PAGES, S, BN, NS, TPV, WINDOW, ROPE, SEP, Q16, KST, BAL, B, B_PAD, NP, NID, RED, red_ptr, rcnt_ptr, out_ptr, NR, True, RCW, QSS, DX)
        else:
            _gd_full_body(0, False, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale, cs_ptr, HQ, HKV, G, D, PAGE, MAX_PAGES, S, BN, NS, TPV, WINDOW, ROPE, SEP, Q16, KST, BAL, B, B_PAD, NP, NID, RED, red_ptr, rcnt_ptr, out_ptr, NR, True, RCW, QSS, DX)

    @gluon.jit
    def _gw_verify(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                   HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
                   PAGE: gl.constexpr, MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr,
                   NS: gl.constexpr, PW: gl.constexpr, PREGS: gl.constexpr, CREGS: gl.constexpr,
                   LAG: gl.constexpr = 0, PEEL: gl.constexpr = False, LSUM: gl.constexpr = False,
                   FUSE: gl.constexpr = False, cnt_ptr=None, out_ptr=None, RS: gl.constexpr = 1,
                   R_PAD: gl.constexpr = 1, BAL: gl.constexpr = False, B: gl.constexpr = 1,
                   B_PAD: gl.constexpr = 1, NID: gl.constexpr = 1, NP: gl.constexpr = 1, LZ: gl.constexpr = 0,
                   GX: gl.constexpr = 0):
        M: gl.constexpr = L * G
        C: gl.constexpr = 2 * D
        split = gl.program_id(0)
        b = gl.program_id(1)
        h = gl.program_id(2) * 2
        # Let the merge kernel launch now (programmatic dependent launch); it waits for this grid.
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        ROW_W: gl.constexpr = HKV * D
        QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
        SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        VSH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)

        seq = gl.load(seq_ptr + b)
        # Each request's cached keys in S equal tile-aligned chunks.
        chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
        lo = split * chunk
        hi = gl.minimum(lo + chunk, seq)
        ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
        pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES

        k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH8)
        v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], VSH)
        if LAG:
            vh0 = gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)
            vh1 = gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)
        else:
            vh0 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
            vh1 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
        ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        for j in gl.static_range(NS):
            mbarrier.init(ready.index(j), count=1 if LAG else PW * 32)
            mbarrier.init(empty.index(j), count=2)
        r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
        cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
        q_off = (b * L + r // G)[:, None] * QKV_W + (h * G + r % G)[:, None] * D + cq[None, :]
        if BAL:
            # (Each consumer loads its segments' queries.)
            q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
            q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
        else:
            x0 = gl.load(qkv_ptr + q_off).to(gl.float32)
            x0 = (x0 * gl.rsqrt(gl.sum(x0 * x0, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv)
            q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x0)
            x1 = gl.load(qkv_ptr + q_off + G * D).to(gl.float32)
            x1 = (x1 * gl.rsqrt(gl.sum(x1 * x1, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv)
            q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x1)
        scale = qk_scale * gl.load(ks_ptr)
        fence_async_shared()
        gl.barrier()
        if BAL:
            gl.warp_specialize([
                (_gw_consumer_bal, (qkv_ptr, q0, k_ring, vh0, ready, empty, seq_ptr, split, h, scale, pacc_ptr,
                                    pml_ptr, 0, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS, LZ, GX)),
                (_gw_consumer_bal, (qkv_ptr, q1, k_ring, vh1, ready, empty, seq_ptr, split, h, scale, pacc_ptr,
                                    pml_ptr, 1, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS, LZ, GX)),
                (_gw_producer_bal, (seq_ptr, pt_ptr, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, vh0, vh1, ready,
                                    empty, split, B, B_PAD, NP, MAX_PAGES, PAGE, ROW_W, BN, C, D, NS, PW,
                                    LAG, GX)),
            ], [4, PW], [CREGS, PREGS])
        elif LAG:
            gl.warp_specialize([
                (_gw_consumer_cv, (q0, k_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h,
                                   split, 0, HKV, S, M, D, BN, NS, PEEL, LSUM)),
                (_gw_consumer_cv, (q1, k_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h + 1,
                                   split, 1, HKV, S, M, D, BN, NS, PEEL, LSUM)),
                (_gw_producer_cv, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, vh0, vh1, ready, empty,
                                   lo, hi, ntiles, PAGE, ROW_W, BN, C, D, NS, PW, LAG)),
            ], [4, PW], [CREGS, PREGS])
        else:
          gl.warp_specialize([
            (_gw_consumer, (q0, k_ring, v_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h,
                            split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr, 0, HQ, HKV, G, L, S, M, D, BN,
                            NS)),
            (_gw_consumer, (q1, k_ring, v_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b,
                            h + 1, split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr, 1, HQ, HKV, G, L, S, M, D,
                            BN, NS)),
            (_gw_producer, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                            PAGE, ROW_W, BN, C, NS, PW)),
        ], [4, PW], [CREGS, PREGS])
        if FUSE:
            # The merge, in this grid: once all S programs of this (request, KV head pair) have stored their
            # partials, each merges a slice of the rows. One counter word per pair: arrivals in the low 16
            # bits, a generation above; the last arrival clears the arrivals and bumps the generation, which
            # the others wait on (so each call leaves the arrivals at 0). The plan keeps this grid to one
            # wave, so every program waited for is resident. (Fences as cooperative groups' grid sync: the
            # partition barrier, then the acq_rel atomics.)
            hh, r, c, ok, tok, col, gate = _gw_merge_rows(qkv_ptr, b, h, split, HQ, HKV, G, L, M, D, RS, R_PAD)
            vs = gl.load(vs_ptr)
            cnt = cnt_ptr + b * (HKV // 2) + gl.program_id(2)
            gl.barrier()
            old = gl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
            if (old & 0xFFFF) == S - 1:
                gl.atomic_add(cnt, 0x10000 - S, sem="relaxed", scope="gpu")
            else:
                while ((gl.load(cnt, volatile=True) ^ old) >> 16) == 0:
                    pass
            gl.inline_asm_elementwise("fence.acq_rel.gpu;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        # This grid was launched alongside _new_tokens (which needs nothing from it); finish (or merge) only
        # after it, so the merge sees slot S too.
        gl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        if FUSE:
            _gw_merge(pacc_ptr, pml_ptr, out_ptr, vs, b, hh, r, c, ok, tok, col, gate, HQ, HKV, S, M, D, R_PAD)


@triton.jit
def _norm_rope_ld(ptr, off, ok, c, pos, cs_ptr, ROPE: tl.constexpr, D: tl.constexpr):
    """_norm_rope's loads, to issue them before other work."""
    x = tl.load(ptr + off, mask=ok, other=0.0)
    if ROPE:
        xp = tl.load(ptr + (off ^ 1), mask=ok, other=0.0)
        cs_off = pos[:, None] * D + ((c % D) // 2)[None, :]
        cos = tl.load(cs_ptr + cs_off, mask=ok, other=1.0)
        sin = tl.load(cs_ptr + cs_off + D // 2, mask=ok, other=0.0)
    else:
        xp = x
        cos = x.to(tl.float32)
        sin = cos
    return x, xp, cos, sin


@triton.jit
def _norm_rope_fin(x, xp, cos, sin, c, ROPE: tl.constexpr, D: tl.constexpr, HPP: tl.constexpr):
    x = x.to(tl.float32)
    sq = x * x
    if HPP == 1:
        r = tl.rsqrt(tl.sum(sq, 1) / D + 1e-5)[:, None]
    else:
        first = (c < D)[None, :]
        r0 = tl.rsqrt(tl.sum(tl.where(first, sq, 0.0), 1) / D + 1e-5)
        r1 = tl.rsqrt(tl.sum(tl.where(first, 0.0, sq), 1) / D + 1e-5)
        r = tl.where(first, r0[:, None], r1[:, None])
    if ROPE:
        sign = tl.where(c % 2 == 0, -1.0, 1.0)
        x = x * cos + sign[None, :] * xp.to(tl.float32) * sin
    return x * r


@triton.jit
def _norm_rope(ptr, off, ok, c, pos, cs_ptr, ROPE: tl.constexpr, D: tl.constexpr, HPP: tl.constexpr):
    """[R, HPP*D] rows of HPP heads at ptr + off: per-head rmsnorm (no weight), then interleaved
    RoPE at pos [R]. Masked-out entries are 0."""
    x = tl.load(ptr + off, mask=ok, other=0.0).to(tl.float32)
    sq = x * x
    if HPP == 1:
        r = tl.rsqrt(tl.sum(sq, 1) / D + 1e-5)[:, None]
    else:
        first = (c < D)[None, :]
        r0 = tl.rsqrt(tl.sum(tl.where(first, sq, 0.0), 1) / D + 1e-5)
        r1 = tl.rsqrt(tl.sum(tl.where(first, 0.0, sq), 1) / D + 1e-5)
        r = tl.where(first, r0[:, None], r1[:, None])
    if ROPE:
        xp = tl.load(ptr + (off ^ 1), mask=ok, other=0.0).to(tl.float32)
        cs_off = pos[:, None] * D + ((c % D) // 2)[None, :]
        cos = tl.load(cs_ptr + cs_off, mask=ok, other=1.0)
        sin = tl.load(cs_ptr + cs_off + D // 2, mask=ok, other=0.0)
        sign = tl.where(c % 2 == 0, -1.0, 1.0)
        x = x * cos + sign[None, :] * xp * sin
    return x * r


@triton.jit
def _tile(t, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0, MASKED: tl.constexpr,
          PAGE: tl.constexpr, ROW_W: tl.constexpr, D: tl.constexpr, WINDOW: tl.constexpr, QK8: tl.constexpr,
          DOT: tl.constexpr, MIF: tl.constexpr = False, need=False, FIXED: tl.constexpr = False,
          CL: tl.constexpr = False):
    """One tile of cached keys at positions t through the online softmax. MASKED: some t >= hi.
    MIF (with MASKED): the score masks run only if the runtime scalar `need` says this tile has masked keys."""
    if MASKED:
        # Past the end, gather the last valid row again (no load masks); the scores are masked.
        t_ok = t < hi
        t_ld = tl.minimum(t, hi - 1)
    else:
        t_ld = t
    page = tl.load(pt_row + t_ld // PAGE)
    if CL:
        page = tl.maximum(page, 0)  # SPEC's speculative length: entries past the real one are -1
    row = (page.to(tl.int64) * PAGE + t_ld % PAGE) * ROW_W + hp0 * D
    k = tl.load(kc_ptr + row[:, None] + c[None, :])
    v = tl.load(vc_ptr + row[:, None] + c[None, :])
    s = tl.dot(q, tl.trans(k if QK8 else k.to(DOT)))
    if MIF:
        if need:
            ok = t_ok[None, :]
            if WINDOW > 0:
                ok = ok & (t[None, :] > pos[:, None] - WINDOW)
            s = tl.where(ok, s, float("-inf"))
    elif MASKED:
        ok = t_ok[None, :]
        if WINDOW > 0:
            ok = ok & (t[None, :] > pos[:, None] - WINDOW)
        s = tl.where(ok, s, float("-inf"))
    elif WINDOW > 0:
        s = tl.where(t[None, :] > pos[:, None] - WINDOW, s, float("-inf"))
    if FIXED:
        # Fixed reference m_i (set before the loop): no row max, alpha or rescale.
        p = tl.exp2(tl.fma(s, scale, -m_i[:, None]))
        l_i = l_i + tl.sum(p, 1)
        acc = acc + tl.dot(p.to(DOT), v.to(DOT))
        return m_i, l_i, acc
    # The scale (> 0) goes into the row maxima and the exponent's FMA, not every score.
    m_new = tl.maximum(m_i, tl.max(s, 1) * scale)
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(tl.fma(s, scale, -m_new[:, None]))
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(DOT), v.to(DOT))
    return m_new, l_i, acc


@triton.jit
def _new(split, b, seq, j, q, m_i, l_i, acc, hp0, c, scale, k_scale, qkv_ptr, cs_ptr, kc_ptr,
         vc_ptr, vs_ptr, loc_ptr, HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr,
         L: tl.constexpr, L_PAD: tl.constexpr, MG: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
         S: tl.constexpr, QK8: tl.constexpr, DOT: tl.constexpr, SEP: tl.constexpr = False):
    """Split 0 (or with SEP the extra program S, which has no cached keys) attends the block's own tokens (and stores them in the cache). Done before the
    cached keys, so these loads overlap the page-table and cache trips instead of following them."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    if split == (S if SEP else 0):
        # The block's own tokens: normalize, store as FP8, attend causally.
        jn = tl.arange(0, L_PAD)
        jn_ok = jn < L
        tok = b * L + jn
        kn_off = (tok * QKV_W + (HQ + hp0) * D)[:, None] + c[None, :]
        kn = _norm_rope(qkv_ptr, kn_off, jn_ok[:, None], c, seq + jn, cs_ptr, ROPE, D, HPP)
        vn = tl.load(qkv_ptr + kn_off + HKV * D, mask=jn_ok[:, None], other=0.0)
        kq = (kn / k_scale).to(tl.float8e4nv)
        vq = (vn.to(tl.float32) / tl.load(vs_ptr)).to(tl.float8e4nv)
        slot = tl.load(loc_ptr + tok, mask=jn_ok, other=0) * ROW_W + hp0 * D
        tl.store(kc_ptr + slot[:, None] + c[None, :], kq, mask=jn_ok[:, None])
        tl.store(vc_ptr + slot[:, None] + c[None, :], vq, mask=jn_ok[:, None])
        s = tl.dot(q, tl.trans(kq if QK8 else kq.to(DOT))) * scale
        ok = jn_ok[None, :] & (jn[None, :] <= j[:, None])
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(DOT), vq.to(DOT))
        m_i = m_new

    return m_i, l_i, acc


@triton.jit
def _finish(split, b, j, m_i, l_i, acc, hp0, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV: tl.constexpr,
            MG: tl.constexpr, D: tl.constexpr, S: tl.constexpr, SEP: tl.constexpr = False):
    """Every split writes its partial (max, sum, accumulator) for KV heads hp0 .. hp0 + HPP - 1."""
    # Partials [B, HKV, S, MG]: each row's own head's D columns of acc.
    part = ((b * HKV + hp0 + hp) * (S + SEP) + split) * MG + rr
    tl.store(pml_ptr + part * 2, m_i, mask=r_ok)
    tl.store(pml_ptr + part * 2 + 1, l_i, mask=r_ok)
    tl.store(pacc_ptr + part[:, None] * D + (c % D)[None, :], acc.to(pacc_ptr.dtype.element_ty), mask=q_ok)


@triton.jit
def _range(split, b, seq, pt_ptr, WINDOW: tl.constexpr, BALANCE: tl.constexpr, S: tl.constexpr, BLOCK_N: tl.constexpr,
           CHUNK: tl.constexpr, MAX_PAGES: tl.constexpr):
    """_attn_split's cached keys [lo, hi) for this split, its page-table row and the tile offsets."""
    if WINDOW > 0:
        base = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        base = seq * 0
    if BALANCE:
        # Each request's own context in S equal tile-aligned chunks: every program does the same work.
        chunk = tl.cdiv(tl.cdiv(seq - base, S), BLOCK_N) * BLOCK_N
    else:
        chunk = CHUNK
    lo = base + split * chunk
    hi = tl.minimum(lo + chunk, seq)
    pt_row = pt_ptr + b.to(tl.int64) * MAX_PAGES
    n = tl.arange(0, BLOCK_N)
    return lo, hi, pt_row, n


@triton.jit
def _attn_body(seq_in, SEQ_IN: tl.constexpr, qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr,
                pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                MG: tl.constexpr, M_PAD: tl.constexpr, L_PAD: tl.constexpr, D: tl.constexpr,
                PAGE: tl.constexpr, MAX_PAGES: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
                S: tl.constexpr, CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, STAGES: tl.constexpr,
                FUSE: tl.constexpr, S_PAD: tl.constexpr, QK8: tl.constexpr, PDL: tl.constexpr,
                DOT: tl.constexpr, PEEL: tl.constexpr = False, DUAL: tl.constexpr = False,
                BALANCE: tl.constexpr = False, SEP: tl.constexpr = False, PF: tl.constexpr = False,
                PRE: tl.constexpr = 0, MIF: tl.constexpr = False, PREQ: tl.constexpr = 0, FIX: tl.constexpr = 0,
                FDELTA: tl.constexpr = 14.0):
    """One program: a split of request b's context for HPP adjacent KV heads, which share each
    cache row's contiguous HPP*D bytes. Queries are block-diagonal [HPP*MG, HPP*D] (row r is KV
    head hp = r // MG of the pair, and holds values only in that head's D columns), so one
    [rows, HPP*D] x [HPP*D, N] product scores every head against its own keys."""
    split = tl.program_id(0)
    b = tl.program_id(1)
    # DUAL (with HPP 1): the program also runs KV head hp0 + 1 as a second, independent chain.
    hp0 = tl.program_id(2) * (2 if DUAL else HPP)
    if PDL:
        # Let the merge kernel launch now; it waits on-device for this grid to finish.
        gdc_launch_dependents()
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    C: tl.constexpr = HPP * D

    if PF:
        # Full layers: touch the page-table lines of this split's chunk (one entry per 128-byte line) before the
        # query load stalls the warps, so the pipeline prologue's page loads hit L2. The sum is used only at the
        # end. (Also touching the first tile's cache rows ties; on sliding layers, whose chunk starts at the
        # window, it loses: decode 1 sliding +0.26 us.)
        PF_N: tl.constexpr = _np2_jit(CHUNK // PAGE // 32 + 2)
        pf_p = tl.minimum(split * CHUNK // PAGE + tl.arange(0, PF_N) * 32, MAX_PAGES - 1)
        pf = tl.sum(tl.load(pt_ptr + b.to(tl.int64) * MAX_PAGES + pf_p), 0)
    if SEQ_IN:
        seq = seq_in
    else:
        seq = tl.load(seq_ptr + b)
    c = tl.arange(0, C)
    if PRE > 0:
        # L2 prefetch of the K/V rows of the split's first PRE tiles, issued before the query norm (and RoPE), so
        # those trips overlap the qkv/cos/sin ones; the pipelined loop then gathers them from L2. Verify 1 full
        # (with pf): H100 paired PRE 1 -0.45 us (20/20), PRE 2 another -0.48 (19/20), PRE 4 +1.06 vs 2. Batch-1
        # sliding ties (one tile per split); batched sliding loses (+0.2..+0.7), as does a register-held first
        # tile instead of the prefetch (decode sliding +1..+2.5 us).
        lo, hi, pt_row, n = _range(split, b, seq, pt_ptr, WINDOW, BALANCE, S, BLOCK_N, CHUNK, MAX_PAGES)
        t0 = tl.maximum(tl.minimum(lo + tl.arange(0, PRE * BLOCK_N), hi - 1), 0)
        row0 = (tl.maximum(tl.load(pt_row + t0 // PAGE), 0).to(tl.int64) * PAGE + t0 % PAGE) * ROW_W + hp0 * D
        tl.inline_asm_elementwise("prefetch.global.L2 [$1];\n\tprefetch.global.L2 [$2];\n\tmov.u32 $0, 0;", "=r,l,l",
                                  [kc_ptr + row0, vc_ptr + row0], dtype=tl.int32, is_pure=False, pack=1)
    r = tl.arange(0, M_PAD)
    hp = r // MG
    rr = r % MG  # token rr // G, query head (hp0 + hp) * G + rr % G
    r_ok = r < HPP * MG
    j = rr // G
    pos = seq + j
    q_off = (b * L + j)[:, None] * QKV_W + ((hp0 + hp) * G + rr % G)[:, None] * D + (c % D)[None, :]
    q_ok = r_ok[:, None] & ((c // D)[None, :] == hp[:, None])
    if PREQ > 0:
        # PREQ: the L2 prefetch of the first PREQ tiles' K/V rows after the query/cos/sin loads are issued, before
        # the norm and RoPE math, so the K/V trip overlaps it (PRE, above, puts the prefetch first).
        qx, qxp, qcos, qsin = _norm_rope_ld(qkv_ptr, q_off, q_ok, c, pos, cs_ptr, ROPE, D)
        lo, hi, pt_row, n = _range(split, b, seq, pt_ptr, WINDOW, BALANCE, S, BLOCK_N, CHUNK, MAX_PAGES)
        t0 = tl.maximum(tl.minimum(lo + tl.arange(0, PREQ * BLOCK_N), hi - 1), 0)
        row0 = (tl.maximum(tl.load(pt_row + t0 // PAGE), 0).to(tl.int64) * PAGE + t0 % PAGE) * ROW_W + hp0 * D
        tl.inline_asm_elementwise("prefetch.global.L2 [$1];\n\tprefetch.global.L2 [$2];\n\tmov.u32 $0, 0;", "=r,l,l",
                                  [kc_ptr + row0, vc_ptr + row0], dtype=tl.int32, is_pure=False, pack=1)
        q = _norm_rope_fin(qx, qxp, qcos, qsin, c, ROPE, D, HPP)
    else:
        q = _norm_rope(qkv_ptr, q_off, q_ok, c, pos, cs_ptr, ROPE, D, HPP)
    if QK8:
        # Queries rounded to E4M3 (as FA3 does with an FP8 cache): QK^T runs on the FP8 cache as is.
        q = q.to(tl.float8e4nv)
    else:
        q = q.to(DOT)
    q1 = q
    if DUAL:
        q1 = _norm_rope(qkv_ptr, q_off + G * D, q_ok, c, pos, cs_ptr, ROPE, D, HPP)
        q1 = q1.to(tl.float8e4nv) if QK8 else q1.to(DOT)
    k_scale = tl.load(ks_ptr)
    scale = qk_scale * k_scale  # log2 units, applied to q . k_fp8

    m_i = tl.full([M_PAD], -1.0e30, tl.float32)
    l_i = tl.zeros([M_PAD], tl.float32)
    acc = tl.zeros([M_PAD, C], tl.float32)
    m1, l1, acc1 = m_i, l_i, acc
    m_i, l_i, acc = _new(split, b, seq, j, q, m_i, l_i, acc, hp0, c, scale, k_scale, qkv_ptr, cs_ptr, kc_ptr,
                         vc_ptr, vs_ptr, loc_ptr, HQ, HKV, HPP, L, L_PAD, MG, D, ROPE, S, QK8, DOT, SEP)
    if DUAL:
        m1, l1, acc1 = _new(split, b, seq, j, q1, m1, l1, acc1, hp0 + 1, c, scale, k_scale, qkv_ptr, cs_ptr,
                            kc_ptr, vc_ptr, vs_ptr, loc_ptr, HQ, HKV, HPP, L, L_PAD, MG, D, ROPE, S, QK8, DOT, SEP)

    if PRE == 0 and PREQ == 0:
        # (Computed here, after the queries, unless PRE needs it first: moving it up costs decode 1 sliding +0.2 us.)
        lo, hi, pt_row, n = _range(split, b, seq, pt_ptr, WINDOW, BALANCE, S, BLOCK_N, CHUNK, MAX_PAGES)
    # Whole tiles need no bounds masks; full layers need no masks at all there. The tail is masked.
    # Only for single-token decode: with a verify block's 64 rows the second copy costs registers.
    # Cached keys start at base = seq - (WINDOW - 1) (or 0), so a decode query (position seq) needs no window mask on
    # them (H100 paired decode 16 sliding -0.1 us, 11/15 and 13/15 both orders; decode 1/32 tie). A verify row j masks
    # keys below base + j (split 0's first tile only); handling that tile apart lost (+0.2..+1.7 us).
    WIN: tl.constexpr = 0 if L == 1 else WINDOW
    if FIX > 0:
        # Fixed reference: the row maxima of the first FIX tiles' scores (QK^T only) + FDELTA binades; every tile
        # then exponentiates against it (no row max / alpha / rescale in the loop).
        mref = tl.full([M_PAD], -1.0e30, tl.float32)
        for fs in tl.range(lo, tl.minimum(lo + FIX * BLOCK_N, hi), BLOCK_N, num_stages=1):
            ft = fs + n
            fld = tl.minimum(ft, hi - 1)
            frow = (tl.maximum(tl.load(pt_row + fld // PAGE), 0).to(tl.int64) * PAGE + fld % PAGE) * ROW_W + hp0 * D
            fk = tl.load(kc_ptr + frow[:, None] + c[None, :])
            fsc = tl.dot(q, tl.trans(fk if QK8 else fk.to(DOT)))
            fsc = tl.where((ft < hi)[None, :], fsc, float("-inf"))
            mref = tl.maximum(mref, tl.max(fsc, 1) * scale)
        m_fx = tl.maximum(m_i, mref + FDELTA)
        f_al = tl.exp2(m_i - m_fx)
        l_i = l_i * f_al
        acc = acc * f_al[:, None]
        m_i = m_fx
    FXD: tl.constexpr = FIX > 0
    if L == 1 or PEEL:
        full_end = lo + tl.maximum(hi - lo, 0) // BLOCK_N * BLOCK_N
    else:
        full_end = lo
    for start in tl.range(lo, full_end, BLOCK_N, num_stages=STAGES):
        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              False, PAGE, ROW_W, D, WIN, QK8, DOT, FIXED=FXD, CL=SEQ_IN)
        if DUAL:
            m1, l1, acc1 = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q1, pos, c, m1, l1, acc1, scale, hp0 + 1,
                                 False, PAGE, ROW_W, D, WIN, QK8, DOT, CL=SEQ_IN)
    for start in tl.range(full_end, hi, BLOCK_N, num_stages=STAGES):
        # MIF: one loop, masks only on the tiles that need them (window start for verify rows, or past hi).
        need = (start + BLOCK_N > hi) | (start <= seq + (L - 1) - WIN) if WIN > 0 else start + BLOCK_N > hi
        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              True, PAGE, ROW_W, D, WIN, QK8, DOT, MIF, need, FIXED=FXD, CL=SEQ_IN)
        if DUAL:
            m1, l1, acc1 = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q1, pos, c, m1, l1, acc1, scale, hp0 + 1,
                                 True, PAGE, ROW_W, D, WIN, QK8, DOT, MIF, need, CL=SEQ_IN)

    if PF:
        m_i += tl.where(pf == -7, 1.0, 0.0) * 0.0
    if FIX > 0:
        # A row whose scores grew > 16 + FDELTA binades past the reference overflowed FP16 P: flag it (m = +inf,
        # x - x != 0 exactly for inf and NaN) for _combine's exact REDO.
        asum = tl.sum(acc, 1) + l_i
        m_i = tl.where(asum - asum == 0.0, m_i, float("inf"))
    _finish(split, b, j, m_i, l_i, acc, hp0, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S, SEP)
    if DUAL:
        _finish(split, b, j, m1, l1, acc1, hp0 + 1, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S, SEP)

    if FUSE:
        # The last of this (request, head pair)'s S programs to finish merges the partials and
        # writes the output, then resets the counter for the next call.
        cnt = cnt_ptr + b * (HKV // HPP) + hp0 // HPP
        if tl.atomic_add(cnt, 1, sem="acq_rel") == S - 1:
            first = ((b * HKV + hp0 + hp) * S) * MG + rr
            # All splits' maxima and sums at once: [S_PAD, M_PAD].
            si = tl.arange(0, S_PAD)
            at = first[None, :] + si[:, None] * MG
            ok2 = (si < S)[:, None] & r_ok[None, :]
            m = tl.load(pml_ptr + at * 2, mask=ok2, other=-1.0e30, cache_modifier=".cg")
            m_max = tl.max(m, 0)
            denom = tl.sum(tl.exp2(m - m_max[None, :]) * tl.load(pml_ptr + at * 2 + 1, mask=ok2, other=0.0,
                                                                  cache_modifier=".cg"), 0)
            # Unrolled, so every split's accumulator load is in flight at once.
            o = tl.zeros([M_PAD, C], tl.float32)
            for i in tl.static_range(S):
                at1 = first + i * MG
                w = tl.exp2(tl.load(pml_ptr + at1 * 2, mask=r_ok, other=-1.0e30, cache_modifier=".cg") - m_max)
                o += w[:, None] * tl.load(pacc_ptr + at1[:, None] * D + (c % D)[None, :], mask=q_ok, other=0.0,
                                          cache_modifier=".cg").to(tl.float32)
            _norm_gate_store(o / tl.where(r_ok, denom, 1.0)[:, None] * tl.load(vs_ptr), qkv_ptr, out_ptr,
                             b * L + j, (hp0 + hp) * G + rr % G, c, q_ok, HQ, HKV, D)
            tl.atomic_xchg(cnt, 0)


# The rare redo path of _attn_split's SPEC, out of line so it doesn't disturb the hot path's code.
_attn_body_redo = triton.jit(noinline=True)(_attn_body.fn)


@triton.jit
def _attn_split(qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr,
                pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                MG: tl.constexpr, M_PAD: tl.constexpr, L_PAD: tl.constexpr, D: tl.constexpr,
                PAGE: tl.constexpr, MAX_PAGES: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
                S: tl.constexpr, CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, STAGES: tl.constexpr,
                FUSE: tl.constexpr, S_PAD: tl.constexpr, QK8: tl.constexpr, PDL: tl.constexpr,
                DOT: tl.constexpr, PEEL: tl.constexpr = False, DUAL: tl.constexpr = False,
                BALANCE: tl.constexpr = False, SEP: tl.constexpr = False, PF: tl.constexpr = False,
                PRE: tl.constexpr = 0, SPEC: tl.constexpr = 0, MIF: tl.constexpr = False, PREQ: tl.constexpr = 0,
                FIX: tl.constexpr = 0, FDELTA: tl.constexpr = 14.0):
    """Kernel wrapper of _attn_body. SPEC > 0 (batch 1, page 1, shapes only): SGLang slices the page table to the
    batch's longest length, so seq = width - L is known before the seq_lens load lands; run on it at once (the
    page-table and cos/sin loads skip one DRAM trip) and redo the split with the real length if it differs."""
    if SPEC > 0:
        real = tl.load(seq_ptr + tl.program_id(1))
        # One copy of the body in a loop (a second, out-of-line copy for the redo lost ~2 us on the H200 judge, s86):
        # the first pass runs on the constant, a second one only if the real length differs.
        seq = tl.full([], SPEC, tl.int32)
        go = True
        while go:
            _attn_body(seq, True, qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale, HQ, HKV, HPP, G, L, MG, M_PAD, L_PAD, D, PAGE, MAX_PAGES, WINDOW, ROPE, S, CHUNK, BLOCK_N, STAGES, FUSE, S_PAD, QK8, PDL, DOT, PEEL, DUAL, BALANCE, SEP, PF, PRE, MIF, PREQ, FIX, FDELTA)
            go = real != seq
            seq = real
    else:
        _attn_body(0, False, qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale, HQ, HKV, HPP, G, L, MG, M_PAD, L_PAD, D, PAGE, MAX_PAGES, WINDOW, ROPE, S, CHUNK, BLOCK_N, STAGES, FUSE, S_PAD, QK8, PDL, DOT, PEEL, DUAL, BALANCE, SEP, PF, PRE, MIF, PREQ, FIX, FDELTA)


@triton.jit
def _new_tokens(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, MG: tl.constexpr,
                L_PAD: tl.constexpr, D: tl.constexpr, S: tl.constexpr, DOT: tl.constexpr, NID: tl.constexpr = 0,
                NBASE: tl.constexpr = 0, WAIT: tl.constexpr = False):
    """For _gw_verify: one program per (request, KV head) stores the block's new keys and values as FP8
    and attends them causally, into partial slot S (of S + 1)."""
    b = tl.program_id(0)
    hp0 = tl.program_id(1)
    gdc_launch_dependents()
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    seq = tl.load(seq_ptr + b)
    c = tl.arange(0, D)
    rr = tl.arange(0, MG)
    j = rr // G
    q_off = (b * L + j)[:, None] * QKV_W + (hp0 * G + rr % G)[:, None] * D + c[None, :]
    q_ok = rr[:, None] < MG
    q = _norm_rope(qkv_ptr, q_off, q_ok, c, seq + j, qkv_ptr, False, D, 1).to(tl.float8e4nv)
    k_scale = tl.load(ks_ptr)
    m_i = tl.full([MG], -1.0e30, tl.float32)
    l_i = tl.zeros([MG], tl.float32)
    acc = tl.zeros([MG, D], tl.float32)
    m_i, l_i, acc = _new(0, b, seq, j, q, m_i, l_i, acc, hp0, c, qk_scale * k_scale, k_scale, qkv_ptr, qkv_ptr,
                         kc_ptr, vc_ptr, vs_ptr, loc_ptr, HQ, HKV, 1, L, L_PAD, MG, D, False, S, True, DOT)
    if NID:
        # Balanced splits (gbal): partials [HKV, NID, MG, D], this request's new tokens at id NBASE + b.
        _finish(NBASE + b, 0, j, m_i, l_i, acc, hp0, 0, rr, rr < MG, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, NID)
        if WAIT:
            # (NT_AFTER: launched after _gw_verify, which it doesn't read; end only after it, so the merge that
            # waits on this grid sees both.)
            gdc_wait()
    else:
        _finish(S, b, j, m_i, l_i, acc, hp0, 0, rr, rr < MG, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S + 1)


@triton.jit
def _norm_gate_store(o, qkv_ptr, out_ptr, tok, head, c, ok, HQ: tl.constexpr, HKV: tl.constexpr,
                     D: tl.constexpr):
    """out[tok, head] = rmsnorm(o) * sigmoid(gate) for [R, C] rows holding one head's D values each
    (the rest masked out and 0)."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    col = head[:, None] * D + (c % D)[None, :]
    gate = tl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col, mask=ok, other=0.0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + tok[:, None] * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16), mask=ok)


@triton.jit
def _qld(p, QSS: tl.constexpr):
    # proj s22 (QSS): qkv as 2 FP32 split-K slabs QSS elements apart: add them in FP32 (then the caller rounds to BF16)
    x = tl.load(p)
    if QSS != 0:
        x = x + tl.load(p + QSS)
    return x


@triton.jit
def _combine(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
             HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
             D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, PDL: tl.constexpr, ONE: tl.constexpr = True,
             REDO: tl.constexpr = False, seq_ptr=None, kc_ptr=None, vc_ptr=None, ks_ptr=None, pt_ptr=None, qk_scale=1.0,
             MAX_PAGES: tl.constexpr = 1, PAGE: tl.constexpr = 1, CHUNK: tl.constexpr = 1, QSS: tl.constexpr = 0):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    b = tok // L
    h = head // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    gate = _qld(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + head * D + d, QSS).to(tl.bfloat16).to(tl.float32)
    if PDL:
        # Launched early (programmatic dependent launch): wait here for _attn_split's partials.
        gdc_wait()
    # This head's row in each split's partial: [S] maxima and sums, [S, D] accumulators.
    row = (b * HKV + h) * S * M + (tok % L) * G + head % G
    s = tl.arange(0, S_PAD)
    m = tl.load(pml_ptr + (row + s * M) * 2, mask=s < S, other=-1.0e30)
    l = tl.load(pml_ptr + (row + s * M) * 2 + 1, mask=s < S, other=0.0)
    if S_PAD <= SB and ONE:
        # One pass: every load issued before the max reduction (one L2 round trip on the critical path).
        a = tl.load(pacc_ptr + (row + s * M)[:, None] * D + d[None, :], mask=(s < S)[:, None],
                    other=0.0).to(tl.float32)
        m_max = tl.max(m, 0)
        w = tl.exp2(m - m_max)
        denom = tl.sum(w * l, 0)
        acc = tl.sum(w[:, None] * a, 0)
    else:
        m_max = tl.max(m, 0)
        denom = tl.sum(tl.exp2(m - m_max) * l, 0)
        acc = tl.zeros([D], tl.float32)
        sb = tl.arange(0, SB)
        for i in range(0, S, SB):
            si = i + sb
            w = tl.exp2(tl.load(pml_ptr + (row + si * M) * 2, mask=si < S, other=-1.0e30) - m_max)
            a = tl.load(pacc_ptr + (row + si * M)[:, None] * D + d[None, :], mask=(si < S)[:, None],
                        other=0.0).to(tl.float32)
            acc += tl.sum(w[:, None] * a, 0)
    if REDO:
        if tl.max(m, 0) == float("inf"):
            # Cold path: an _attn_split FIX partial overflowed (m = +inf; never seen on the scored data). Merge again,
            # recomputing each flagged split's keys [s * CHUNK, ..) exactly (same FP8 query and scale, FP32 P). The
            # new tokens' slot (tsep) is exact and never flagged.
            ROW_W: tl.constexpr = HKV * D
            BK: tl.constexpr = 8
            x = _qld(qkv_ptr + tok * QKV_W + head * D + d, QSS).to(tl.bfloat16).to(tl.float32)
            q8 = (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-5)).to(tl.float8e4nv).to(tl.float32)
            scale = qk_scale * tl.load(ks_ptr)
            seq = tl.load(seq_ptr + b)
            kk = tl.arange(0, BK)
            m_max = tl.full([], -1.0e30, tl.float32)
            denom = tl.zeros([], tl.float32)
            acc = tl.zeros([D], tl.float32)
            for si in range(0, S):
                ms = tl.load(pml_ptr + (row + si * M) * 2)
                if ms == float("inf"):
                    ms = tl.full([], -1.0e30, tl.float32)
                    ls = tl.zeros([], tl.float32)
                    a_s = tl.zeros([D], tl.float32)
                    for k0 in range(si * CHUNK, tl.minimum(si * CHUNK + CHUNK, seq), BK):
                        t = k0 + kk
                        okk = t < seq
                        pg = tl.load(pt_ptr + b.to(tl.int64) * MAX_PAGES + t // PAGE, mask=okk, other=0)
                        off = ((pg.to(tl.int64) * PAGE + t % PAGE) * ROW_W + h * D)[:, None] + d[None, :]
                        kt = tl.load(kc_ptr + off, mask=okk[:, None], other=0.0).to(tl.float32)
                        vt = tl.load(vc_ptr + off, mask=okk[:, None], other=0.0).to(tl.float32)
                        sc = tl.where(okk, tl.sum(q8[None, :] * kt, 1) * scale, float("-inf"))
                        mn3 = tl.maximum(ms, tl.max(sc, 0))
                        al3 = tl.exp2(ms - mn3)
                        pw = tl.exp2(sc - mn3)
                        ls = ls * al3 + tl.sum(pw, 0)
                        a_s = a_s * al3 + tl.sum(pw[:, None] * vt, 0)
                        ms = mn3
                else:
                    ls = tl.load(pml_ptr + (row + si * M) * 2 + 1)
                    a_s = tl.load(pacc_ptr + (row + si * M) * D + d).to(tl.float32)
                mn2 = tl.maximum(m_max, ms)
                al2 = tl.exp2(m_max - mn2)
                w2 = tl.exp2(ms - mn2)
                denom = denom * al2 + w2 * ls
                acc = acc * al2 + w2 * a_s
                m_max = mn2
    o = acc / denom * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    tl.store(out_ptr + tok * HQ * D + head * D + d, (o * tl.sigmoid(gate)).to(tl.bfloat16))


@triton.jit
def _combine_hb(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
                HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
                D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, HB: tl.constexpr,
                PDL: tl.constexpr, ONE: tl.constexpr = True, QSS: tl.constexpr = 0):
    """_combine for HB adjacent query heads of one KV head per program: their partial rows are
    contiguous in every split, so each split's [HB, D] block is one contiguous read. Faster on
    sliding layers (fewer, fuller programs); full layers keep _combine."""
    tok = tl.program_id(0)
    head0 = tl.program_id(1) * HB
    b = tok // L
    h = head0 // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    hb = tl.arange(0, HB)
    col = (head0 + hb)[:, None] * D + d[None, :]
    gate = _qld(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col, QSS).to(tl.bfloat16).to(tl.float32)
    if PDL:
        gdc_wait()
    row = (b * HKV + h) * S * M + (tok % L) * G + head0 % G + hb  # [HB] rows in split 0
    s = tl.arange(0, S_PAD)
    at = row[None, :] + (s * M)[:, None]
    m = tl.load(pml_ptr + at * 2, mask=(s < S)[:, None], other=-1.0e30)
    l = tl.load(pml_ptr + at * 2 + 1, mask=(s < S)[:, None], other=0.0)
    if S_PAD <= SB and ONE:
        a = tl.load(pacc_ptr + at[:, :, None] * D + d[None, None, :], mask=(s < S)[:, None, None],
                    other=0.0).to(tl.float32)
        m_max = tl.max(m, 0)
        w = tl.exp2(m - m_max[None, :])
        denom = tl.sum(w * l, 0)
        acc = tl.sum(w[:, :, None] * a, 0)
    else:
        m_max = tl.max(m, 0)
        denom = tl.sum(tl.exp2(m - m_max[None, :]) * l, 0)
        acc = tl.zeros([HB, D], tl.float32)
        sb = tl.arange(0, SB)
        for i in range(0, S, SB):
            si = i + sb
            at1 = row[None, :] + (si * M)[:, None]
            ok = (si < S)[:, None]
            w = tl.exp2(tl.load(pml_ptr + at1 * 2, mask=ok, other=-1.0e30) - m_max[None, :])
            a = tl.load(pacc_ptr + at1[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0).to(tl.float32)
            acc += tl.sum(w[:, :, None] * a, 0)
    o = acc / denom[:, None] * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + tok * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))


@triton.jit
def _combine_bal(qkv_ptr, seq_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
                 HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
                 D: tl.constexpr, B: tl.constexpr, B_PAD: tl.constexpr, P: tl.constexpr, NID: tl.constexpr,
                 BN: tl.constexpr, SB: tl.constexpr, HB: tl.constexpr, ONE: tl.constexpr = True,
                 PROP: tl.constexpr = False, REDO: tl.constexpr = False, kc_ptr=None, vc_ptr=None, ks_ptr=None,
                 pt_ptr=None, qk_scale=1.0, MAX_PAGES: tl.constexpr = 1, PAGE: tl.constexpr = 1, QSS: tl.constexpr = 0):
    """_combine_hb for balanced splits (_gw_bal): request b's partials are at ids p + b for the programs p whose
    tile range meets b's tiles, and its new tokens at P + B + b."""
    tok = tl.program_id(0)
    head0 = tl.program_id(1) * HB
    b = tok // L
    h = head0 // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    hb = tl.arange(0, HB)
    col = (head0 + hb)[:, None] * D + d[None, :]
    gate = _qld(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col, QSS).to(tl.bfloat16).to(tl.float32)
    idx = tl.arange(0, B_PAD)
    tiles = (tl.load(seq_ptr + idx, mask=idx < B, other=0).to(tl.int32) + (BN - 1)) // BN
    tt = tl.maximum(tl.sum(tiles, 0), 1)
    excl = tl.sum(tl.where(idx < b, tiles, 0), 0)
    incl = excl + tl.sum(tl.where(idx == b, tiles, 0), 0)
    # Programs p with start(p) < incl, end(p) > excl, end(p) > start(p) (a superset range, then the test).
    p_lo = tl.maximum((excl + 1) * P // tt - 1, 0)
    p_hi = tl.minimum((incl * P) // tt + 1, P - 1)
    if PROP:
        # _gd_full's length-proportional splits (plan dbal): request b owns programs [c_b, c_b+1) exactly.
        p_lo = b + (P - B) * excl // tt
        p_hi = b + (P - B) * incl // tt
    gdc_wait()
    row = (h * NID + b) * M + (tok % L) * G + head0 % G + hb  # [HB] rows at id b (p = 0)
    nrow = row + (P + B) * M
    sb = tl.arange(0, SB)
    if ONE:
        # One online pass: each chunk's maxima, sums and accumulators load together (one L2 round trip per chunk).
        m_max = tl.load(pml_ptr + nrow * 2)
        denom = tl.load(pml_ptr + nrow * 2 + 1)
        acc = tl.load(pacc_ptr + nrow[:, None] * D + d[None, :]).to(tl.float32)
        for i in range(p_lo, p_hi + 1, SB):
            pp = i + sb
            st = pp * tt // P
            en = (pp + 1) * tt // P
            if PROP:
                ok = (pp <= p_hi)[:, None]
            else:
                ok = ((pp <= p_hi) & (st < incl) & (en > excl) & (en > st))[:, None]
            at = row[None, :] + (pp * M)[:, None]
            m = tl.load(pml_ptr + at * 2, mask=ok, other=-1.0e30)
            l = tl.load(pml_ptr + at * 2 + 1, mask=ok, other=0.0)
            a = tl.load(pacc_ptr + at[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0).to(tl.float32)
            m_new = tl.maximum(m_max, tl.max(m, 0))
            alpha = tl.exp2(m_max - m_new)
            w = tl.exp2(m - m_new[None, :])
            denom = denom * alpha + tl.sum(w * l, 0)
            acc = acc * alpha[:, None] + tl.sum(w[:, :, None] * a, 0)
            m_max = m_new
    else:
        m_max = tl.load(pml_ptr + nrow * 2)
        for i in range(p_lo, p_hi + 1, SB):
            pp = i + sb
            st = pp * tt // P
            en = (pp + 1) * tt // P
            if PROP:
                ok = (pp <= p_hi)[:, None]
            else:
                ok = ((pp <= p_hi) & (st < incl) & (en > excl) & (en > st))[:, None]
            at = row[None, :] + (pp * M)[:, None]
            m_max = tl.maximum(m_max, tl.max(tl.load(pml_ptr + at * 2, mask=ok, other=-1.0e30), 0))
        wn = tl.exp2(tl.load(pml_ptr + nrow * 2) - m_max)
        denom = wn * tl.load(pml_ptr + nrow * 2 + 1)
        acc = wn[:, None] * tl.load(pacc_ptr + nrow[:, None] * D + d[None, :]).to(tl.float32)
        for i in range(p_lo, p_hi + 1, SB):
            pp = i + sb
            st = pp * tt // P
            en = (pp + 1) * tt // P
            if PROP:
                ok = (pp <= p_hi)[:, None]
            else:
                ok = ((pp <= p_hi) & (st < incl) & (en > excl) & (en > st))[:, None]
            at = row[None, :] + (pp * M)[:, None]
            w = tl.exp2(tl.load(pml_ptr + at * 2, mask=ok, other=-1.0e30) - m_max[None, :])
            denom += tl.sum(w * tl.load(pml_ptr + at * 2 + 1, mask=ok, other=0.0), 0)
            a = tl.load(pacc_ptr + at[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0).to(tl.float32)
            acc += tl.sum(w[:, :, None] * a, 0)
    if REDO:
        if tl.max(m_max, 0) == float("inf"):
            # Cold path: a _gw_verify LZ 10 partial of these rows overflowed (m = +inf: a row's scores grew by > 30
            # binades past its segment's first tiles; never seen on the scored data). Recompute the rows' attention over all the
            # request's cached keys exactly (same FP8 queries and scale, FP32 P), then add the new tokens' partial.
            ROW_W: tl.constexpr = HKV * D
            BK: tl.constexpr = 8  # (small: wider tiles cost this kernel registers on the hot path)
            x = _qld(qkv_ptr + tok * QKV_W + col, QSS).to(tl.bfloat16).to(tl.float32)
            q8 = (x * tl.rsqrt(tl.sum(x * x, 1) / D + 1e-5)[:, None]).to(tl.float8e4nv).to(tl.float32)
            scale = qk_scale * tl.load(ks_ptr)
            seq = tl.load(seq_ptr + b)
            kk = tl.arange(0, BK)
            m_max = tl.load(pml_ptr + nrow * 2)
            denom = tl.load(pml_ptr + nrow * 2 + 1)
            acc = tl.load(pacc_ptr + nrow[:, None] * D + d[None, :]).to(tl.float32)
            for k0 in range(0, seq, BK):
                t = k0 + kk
                okk = t < seq
                pg = tl.load(pt_ptr + b.to(tl.int64) * MAX_PAGES + t // PAGE, mask=okk, other=0)
                off = ((pg.to(tl.int64) * PAGE + t % PAGE) * ROW_W + h * D)[:, None] + d[None, :]
                kt = tl.load(kc_ptr + off, mask=okk[:, None], other=0.0).to(tl.float32)
                vt = tl.load(vc_ptr + off, mask=okk[:, None], other=0.0).to(tl.float32)
                sc = tl.sum(q8[:, None, :] * kt[None, :, :], 2) * scale
                sc = tl.where(okk[None, :], sc, float("-inf"))
                m_new = tl.maximum(m_max, tl.max(sc, 1))
                alpha = tl.exp2(m_max - m_new)
                pw = tl.exp2(sc - m_new[:, None])
                denom = denom * alpha + tl.sum(pw, 1)
                acc = acc * alpha[:, None] + tl.sum(pw[:, :, None] * vt[None, :, :], 1)
                m_max = m_new
    o = acc / denom[:, None] * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + tok * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))


def _plan(cfg, max_pages, device):
    """Launch parameters from shapes only, tuned on an H100 over the scored shapes: splits per
    (request, KV head group), tile size, warps, pipeline stages, KV heads per program (hpp),
    whether the last program merges the splits (fuse), and whether QK^T runs in FP8 (qk8:
    queries rounded to E4M3, as FA3 does with an FP8 cache)."""
    batch, length, hkv = cfg["batch"], cfg["q_len"], cfg["num_kv_heads"]
    heads = batch * hkv
    pairs = heads // 2
    if cfg["window"] and length == 1 and pairs >= 12 and HAVE_GLUON and USE_GSLIDE and hkv % 2 == 0 \
            and cfg["num_q_heads"] // hkv == 8 and cfg["head_dim"] == 64:
        # Batched sliding decode on the Gluon one-warp kernel (FP8 QK^T mma; a dedicated program per KV head
        # pair for the new token, so the S key splits stay tile-aligned), up to ~800 programs (H100 decode
        # 16/32 sliding 10.6/15.6 -> 9.3/13.9 us, B64 27.7 -> 23.2; at batch <= 8 it loses by 0.2-1.8 us).
        # Off: on the H200 (check f3e099c1) decode 16/32 sliding went 10.4/14.6 -> 13.6/15.4 us.
        splits = 32 if pairs * 32 <= 800 else (16 if pairs * 16 <= 800 else 8)
        return dict(gdec=True, sep=True, splits=splits, block_n=32, stages=2, pdt="f32", hb=2)
    if cfg["window"]:
        # Sliding layers: at most 2,047 cached keys; latency-bound.
        # The merge takes 2 query heads per one-warp program (hb, cw) and every split in one pass (sb):
        # each further pass is another serial L2 round trip on the critical path.
        splits = 32 if heads <= 4 else (16 if heads <= 32 and length == 1 else 8)
        # Partials stay FP32 everywhere (human decision 2026-10-24; BF16 was -0.5 us at verify 16/32 sliding).
        pdt = "f32"
        if heads <= 32:
            # tsep (batch 1-2): the block's new tokens in their own program instead of split 0, the critical path,
            # with 2 stages (one tile per split). H100 paired: verify 1 sliding tsep -0.14 us (8/9 rounds twice),
            # + stages 2 -0.11 (9/15); decode 1 sliding tsep alone and stages 2 alone ties, both -0.35 (15/15).
            # Batched sizes lose with tsep (+0.6-2.7 us).
            tiny = heads <= 4
            return dict(splits=splits, block_n=64, warps=4, stages=2 if tiny else (5 if length > 1 else 3),
                        hpp=1, fuse=False, qk8=True, hb=2, sb=splits, cw=1, pdt=pdt, tsep=tiny,
                        **({"spec": True, "pre": 2} if tiny and batch == 1 and length > 1 else {}),
                        # Verify (agent2 s1): mif (score masks only on the window-start / tail tiles) H100 paired
                        # verify 16 sliding -0.24 us (10/11), + preq 1 (first tile's K/V L2 prefetch between the
                        # query loads and their norm/RoPE) another -0.13 (9/11).
                        **({"mif": True, "preq": 1} if length > 1 and not tiny else {}))
            # (spec + pre 2 for verify 1 sliding, s85: H100 paired -0.30 us, but H200 check f51b8c4b judged 9.55 vs
            # ~7.2 us with unchanged profiled kernels (5.5 us): off. _attn_split keeps the SPEC code for that study.)
        # Large batches: smaller, deeply pipelined programs.
        # (hpp 2 for batched sliding decode, e405425/6fb74be/652aafe, won on the H100 but lost on the H200: scored
        # 56fca423 decode 16/32 sliding 10.3/14.3 -> 11.1/15.4 us vs abd67ae2. Reverted 2026-10-24.)
        if length > 1 and heads >= 64:
            # Verify from batch 32: 4 splits of 64-key tiles instead of 8 of 32, a quarter of the programs (256 at
            # batch 32) with the same per-program chain, and half the FP32 partials for the merge to read. Paired
            # H100 verify 32/40/48/64 sliding -1.6/-5.8/-6.7/-7.5 us (7/7 or 9/9 each); batch 20 +0.9, 24 -0.2.
            return dict(splits=4, block_n=64, warps=4, stages=5, hpp=1, fuse=False, qk8=True, hb=2, sb=4, cw=1,
                        pdt=pdt, mif=True)  # mif (agent2 s1): verify 32 sliding H100 paired -0.61 us (11/11)
        return dict(splits=splits, block_n=32, warps=2 if length == 1 else 4, stages=5, hpp=1, fuse=False,
                    qk8=True, peel=length > 1, hb=2, sb=splits, cw=1, pdt=pdt)
    # Full layers: partials stay FP32 too (BF16 had been -1.0/-1.5 us at decode 16/32 full, -0.4 at verify 1 full).
    if heads == 2 and length == 1 and HAVE_GLUON and USE_GDEC and USE_GD1 and cfg["num_q_heads"] // hkv == 8 \
            and cfg["head_dim"] == 64:
        # Batch-1 decode full on the Gluon one-warp kernel: 511 key splits + the new token's slot (512 BF16
        # partials, about 4 per SM), merged in one pass by 8-warp programs. H100 17.5 -> 15.7 us (attention
        # alone 15.3 -> 12.4, near the 11.5 us gather floor; the 512-way merge costs 3.2 us vs 2.2).
        # red (s95): the merge as FP32 atomic adds at reference max 0 into 4 replicas inside _gd_full, then one
        # 4-warp _red_fin program (reads 16 KB instead of 512 x 2 KB slots): H100 paired 14.55 -> 13.85 us (8/8).
        return dict(gdec=True, splits=511, block_n=32, stages=2, pdt="f32", csb=512, cw=8,
                    spec=True,  # seq from the table width (see _gd_full): H100 paired -0.59 us (13/13)
                    red=True)
    if heads <= (2 if length == 1 else 4):
        # (Session 18's hpp 2 / block_n 64 for decode 1 full won on the H100, 18.0 -> 17.4 us, but lost on the
        # H200: check f3e099c1 decode 1 full 15.2 -> 17.6 us.)
        # Verify 1 full: tsep as on sliding layers (H100 paired -0.11 us, 9/11 rounds), with 127 splits so the merge
        # sees at most 128 slots in one pass (-0.08, 8/11); then block_n 64 with 8 stages (-0.62, 9/11; emulated H200
        # 25.9 -> 20.3). Unchecked on the H200: if verify 1 full is worse there, revert block_n/stages first.
        if length > 1:
            # pf: the split's page-table lines touched before the query load (H100 paired -0.35 us, 10/11 and 8/11).
            # fix 1 (s106): fixed-reference softmax (reference = tile 0's row max + 14 binades; no row max / alpha /
            # rescale in the loop; overflowing rows -> _combine REDO, exact). H100 paired cold -0.19 us (9/9), warm-L2
            # -0.4 (7/7; -1.1 with pre 0).
            # spec (moe s33): seq from the table width (batch 1; redo if the real length differs): H200 DIAG -0.63/-0.70 us
            # standalone (4/4 pairs, checks 4fe17fa9/28658e59), bitwise = before.
            return dict(splits=127, block_n=64, warps=4, stages=8, hpp=1, fuse=False, qk8=True, pdt="f32", tsep=True,
                        pf=True, pre=2, fix=1, spec=True)
        return dict(splits=128, block_n=128, warps=4, stages=6, hpp=1, fuse=False, qk8=True, pdt="f32")
    if length == 1:
        if HAVE_GLUON and USE_GDEC and hkv % 2 == 0 and cfg["num_q_heads"] // hkv == 8 and cfg["head_dim"] == 64:
            # Gluon one-warp programs (FP8 QK^T mma, new token in the last split): 1024 programs (one wave),
            # S + 1 partial slots a power of two for the merge, BF16 partials.
            # Merge in one-warp programs up to 128 slots (H100 batch 8-48 -1.0..-2.1 us, decode 16/32 full
            # -1.7/-1.7; 256-512 slots lose 0.7-1.0 us). Likely residency: 8 attention programs at 242 registers
            # leave an SM 2048 registers for PDL-launched merge programs (capping the merge's registers loses).
            splits = max(4, 2048 // heads - 1)
            # s103: with plan red's merge (no _combine_bal), dbal wins at batch 16 too: H100 paired decode 16 full
            # -1.54 us (10/10) vs equal splits + red (dprogs 768: -0.63 more at 16 but +3.5 at 32; 512/1536 lose).
            if heads >= (32 if BRED else 48) and 2048 // hkv >= 8 * batch:
                # dbal: 2048 / hkv programs per KV head pair shared out in proportion to the requests' lengths (the
                # agentic lengths differ by up to 15%, and equal splits per request leave the longest ones setting
                # the time), merged by _combine_bal (one head, 64 slots per 2-warp program: its 2-head 32-slot form
                # costs +1.3 us of merge tail). H100 paired, seed-7 data, decode 24/32/48/64 full: real
                # -3.5/-3.5/-3.7/+0.5 us, all-L2 -6.6/-7.5/-16.6/-14.5; on the bench's data decode 32 full real +0.4,
                # all-L2 -4.9 (the H200 sits between). Batch 4-16 lose (real +0.8..+3.2).
                return dict(gdec=True, splits=splits, block_n=32, stages=2, pdt="f32", dbal=True, dprogs=2048 // hkv,
                            hb=1, sb=64, cw=2, red=BRED)
            return dict(gdec=True, splits=splits, block_n=32, stages=2, pdt="f32", red=BRED,
                        **({"cw": 1} if splits < 128 else {}))
        # Many small one-warp programs with deep pipelines keep the most scattered row reads in flight.
        return dict(splits=max(8, 2048 // heads), block_n=16, warps=1, stages=5 if heads <= 32 else 6, hpp=2,
                    fuse=False, qk8=True, pdt="f32")
    # Verify blocks: each program runs both KV heads as two independent 64-row chains (dual), which
    # hides latency better than more programs; 3 fit per SM (160 registers), or 4 when capped at 128
    # (no spills). The splits fill exactly one wave, and each request's context is cut into equal
    # chunks (balance) so none idles. 4 per SM wins from 6 pairs up (batch 32: 303 -> 292 us, 64:
    # 593 -> 568 on an H100); below that only where it fills the wave better (batch 3-4 lose 2-4%).
    pairs = batch * max(1, hkv // 2)
    sms = _num_sms(device)
    if HAVE_GLUON and pairs >= 6 and hkv % 2 == 0 and length * cfg["num_q_heads"] // hkv == 64 \
            and cfg["head_dim"] == 64:
        # The warp-specialized Gluon kernel: 2 programs per SM (384 threads). One wave of programs unless
        # two fill the slots clearly better (H100, ctx 80K: batch 16 155 -> 145 us with 16 splits instead
        # of 32, batch 12 124 -> 116; batch 40 keeps two waves, 344 vs 332).
        slots = 2 * sms
        fill = {w: (w * slots // pairs) * pairs / (w * slots) for w in (1, 2)}
        waves = 2 if fill[2] > fill[1] + 0.05 else 1
        splits = max(2, waves * slots // pairs)
        # The merge: with up to 32 partials, one-warp programs over 2 heads (H100 verify 16 full 139 -> 137.5 us).
        merge = dict(hb=2, cw=1) if splits + 1 <= 32 and (cfg["num_q_heads"] // hkv) % 2 == 0 else {}
        # gfuse: the merge runs in the same grid (each program merges a slice of rows once its request's
        # programs have all stored; H100 verify 16/32 full 136.8 -> 136.0 / 257.2 -> 256.2 us). The separate
        # merge (hb/cw) remains for two-wave plans.
        # gbal (from 16 pairs): the programs instead share the batch's key tiles equally (ranges may span
        # requests; the agentic lengths differ by up to 15%), as many as fill the slots; merged by _combine_bal.
        # H100 all-L2 (compute-bound, as the H200 mostly is) verify 16/32 full 110.0/212.4 -> 104.1/195.9 us,
        # real 136.4/255.4 -> 138.8/250.7, batch 40/48/64 real -8.8/-7.1/-8.4. Below 16 pairs the lost in-grid
        # merge (gfuse) costs more than the balance gains (batch 8: +4 us real).
        return dict(gluon=True, splits=splits, block_n=64, stages=3, lag=1, gpeel=True, pw=4,
                    pregs=40, cregs=96, maxnreg=80, lsum=True, gfuse=True, gbal=USE_GBAL and pairs >= 16, **merge)
    fill = {per: max(4, per * sms // pairs) * pairs / (per * sms) for per in (3, 4)}
    per = 4 if pairs >= 6 or fill[4] > fill[3] else 3
    return dict(splits=max(4, per * sms // pairs), block_n=32, warps=4, stages=5, hpp=1, fuse=False,
                qk8=True, peel=True, dual=True, balance=True, maxnreg=128 if per == 4 else None, pdt="f32")


@triton.jit
def _red_fin(qkv_ptr, red_ptr, rcnt_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr, HQ: tl.constexpr, HKV: tl.constexpr,
             G: tl.constexpr, D: tl.constexpr, M: tl.constexpr, S: tl.constexpr, NR: tl.constexpr,
             SL: tl.constexpr = 0, RCW: tl.constexpr = 0, BAL: tl.constexpr = False, TRIG: tl.constexpr = False,
             QSS: tl.constexpr = 0):
    """plan red's merge (one KV head pair, M = 2 G rows = query heads; program b = request b): the NR replicas of
    the sums at reference max 0, merged exactly with any listed out-of-range partials (slot ids in rcnt[2:]); then
    RMSNorm and sigmoid(gate). Leaves the sums and the list count zeroed for the next call. Batched (s103): request
    b's sums at b * NR * M * (D + 1), its list at b * RCW, partial slots per head SL (S + 1, or NID with BAL)."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    b = tl.program_id(0)
    if RCW > 0:
        red_ptr += b * (NR * M * (D + 1))
        rcnt_ptr += b * RCW
        qkv_ptr += b * QKV_W
        out_ptr += b * (HQ * D)
    r = tl.arange(0, M)
    d = tl.arange(0, D)
    col = r[:, None] * D + d[None, :]
    gate = _qld(qkv_ptr + (HQ + 2 * HKV) * D + col, QSS).to(tl.bfloat16).to(tl.float32)
    vs = tl.load(vs_ptr)
    gdc_wait()
    if TRIG:  # layer s27 RF_TRIG: the o GEMV (PDL secondary, W preload before its wait) launches only now
        gdc_launch_dependents()
    a = tl.zeros([M, D], tl.float32)
    l = tl.zeros([M], tl.float32)
    for k in tl.static_range(NR):
        a += tl.load(red_ptr + k * M * D + col)
        l += tl.load(red_ptr + NR * M * D + k * M + r)
    nf = tl.load(rcnt_ptr + 1)
    tl.debug_barrier()  # every thread's copies loaded before one of them zeroes the replicated l and count words
    for k in tl.static_range(NR):
        tl.store(red_ptr + k * M * D + col, tl.zeros([M, D], tl.float32))
        tl.store(red_ptr + NR * M * D + k * M + r, tl.zeros([M], tl.float32))
    tl.store(rcnt_ptr + 1, 0)
    m = tl.where(l > 0.0, 0.0, -1.0e30)
    for i in range(nf):
        s = tl.load(rcnt_ptr + 2 + i)
        if SL == 0:
            prow = ((r // G) * (S + 1) + s) * G + r % G
        elif BAL:
            prow = ((r // G) * SL + s) * G + r % G
        else:
            prow = ((b * HKV + r // G) * SL + s) * G + r % G
        lp = tl.load(pml_ptr + prow * 2 + 1)
        mp = tl.where(lp > 0.0, tl.load(pml_ptr + prow * 2), -1.0e30)
        ap = tl.load(pacc_ptr + prow[:, None] * D + d[None, :]).to(tl.float32)
        mn = tl.maximum(m, mp)
        a1 = tl.exp2(m - mn)
        a2 = tl.where(lp > 0.0, tl.exp2(mp - mn), 0.0)
        l = l * a1 + lp * a2
        a = a * a1[:, None] + ap * a2[:, None]
        m = mn
    o = a / l[:, None] * vs
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))


def _verify_nvs(data, cfg):
    """Verify sliding (L 8, G 8, D 64, HKV 2, page 1) on the native nvs kernel + _combine_hb."""
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv, device = data["qkv"], data["qkv"].device
    if NVS_HALF:  # CTA (request, KV head, split), 2 halves per split merged in the CTA
        S = 2 if batch >= 32 else 4
        chunk = triton.cdiv(triton.cdiv(cfg["window"], S), 128) * 128
    else:
        S = 4 if batch >= 32 else 8  # ~128 CTAs, one per SM
        chunk = triton.cdiv(triton.cdiv(cfg["window"], S), 64) * 64
    rows = length * (hq // hkv)
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device=device)
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    if (data.get("nvs_cl", True) and _NVSCL is not None and NVS_HALF and S == 2 and batch >= NVS_CL_BATCH
            and not data.get("qkv_ss", 0) and not data.get("o_pf", (0, 0))[1] and qkv.dtype == torch.bfloat16):
        _NVSCL.nvs(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], data["k_scale"],
                   data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml,
                   cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, S, chunk, cfg["window"], out, 1)
        return out  # the 2-CTA clusters merged the 2 splits and wrote out
    _NVS.nvs(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], data["k_scale"],
             data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml,
             cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, S, chunk, cfg["window"], *data.get("o_pf", (0, 0)),
             data.get("qkv_ss", 0))
    _combine_hb[(batch * length, hq // 2)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=hq // hkv, L=length, M=rows, D=d, S=S, S_PAD=triton.next_power_of_2(S), SB=S, HB=2,
        ONE=MERGE1, PDL=USE_PDL, QSS=data.get("qkv_ss", 0), num_warps=1, launch_pdl=USE_PDL)
    return out


def _decode_nds(data, cfg):
    """Decode sliding (L 1, G 8, D 64, HKV 2, page 1) on the native nds kernel + _combine_hb."""
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv, device = data["qkv"], data["qkv"].device
    S = 8  # 256 CTAs at batch 32, 2 per SM (s34: 4 and 16 lose on H100)
    chunk = triton.cdiv(triton.cdiv(cfg["window"], S), 64) * 64
    rows = hq // hkv
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device=device)
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    _NDS.nds(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], data["k_scale"],
             data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml, out,
             cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, S, chunk, cfg["window"], data.get("qkv_ss", 0),
             *data.get("o_pf", (0, 0)))
    _combine_hb[(batch, hq // 2)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=rows, L=1, M=rows, D=d, S=S, S_PAD=S, SB=S, HB=2,
        ONE=MERGE1, PDL=USE_PDL, QSS=data.get("qkv_ss", 0), num_warps=1, launch_pdl=USE_PDL)
    return out


def attn_kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    nvs_ok = (_NVS is not None and cfg["window"] == 2048 and length == 8 and batch >= NVS_MIN_BATCH and hq == 16
              and hkv == 2 and d == 64 and cfg["page_size"] == 1 and cfg["rope_theta"])
    if data.get("qkv_ss", 0) and not (length == 1 and batch >= NDS_MIN_BATCH or nvs_ok):  # proj s22: csl/vtk: summed slabs (s72: nvs loads them)
        qkv = data["qkv_slabs"].sum(0).view(qkv.shape)
        data = {**data, "qkv": qkv, "qkv_ss": 0}
    if _CSL is not None and cfg["window"] == 2048 and length == 1 and batch <= CSL_MAX_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] in (1, 64) and cfg["rope_theta"]:
        out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
        _CSL.csl_decode(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], page_table,
                        data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out, cfg["page_size"],
                        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, *data.get("o_pf", (0, 0)), int(data.get("csl_pw", 0)))
        return out
    if _CSL is not None and cfg["window"] == 2048 and length == 8 and batch <= VTK_MAX_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] in (1, 64) and cfg["rope_theta"]:
        out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
        _CSL.vtk_verify(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], page_table,
                        data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out, cfg["page_size"],
                        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, int(data.get("csl_pw", 0)))
        return out
    if _NVS is not None and cfg["window"] == 2048 and length == 8 and batch >= NVS_MIN_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] == 1 and cfg["rope_theta"]:
        return _verify_nvs(data, cfg)
    if _NDS is not None and cfg["window"] == 2048 and length == 1 and batch >= NDS_MIN_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] == 1 and cfg["rope_theta"]:
        return _decode_nds(data, cfg)
    plan = _plan(cfg, max_pages, device)
    if ATT_PLAN and (plan.get("gluon") or plan.get("gdec")):  # layer s84: H200 tuner plan overrides (full layers)
        plan = {**plan, **ATT_PLAN}
    if data.get("qkv_ss", 0) and not (plan.get("gdec") and not plan.get("gluon")):  # proj s22: paths without QSS loads
        qkv = data["qkv_slabs"].sum(0).view(qkv.shape)
        data = {**data, "qkv": qkv, "qkv_ss": 0}
    if plan.get("gluon"):
        return _verify_gluon(data, cfg, plan)
    if plan.get("gdec"):
        return _decode_gluon(data, cfg, plan)
    span = max_pages * cfg["page_size"]
    if cfg["window"]:
        span = min(span, cfg["window"])
    block_n = plan["block_n"]
    chunk = triton.cdiv(triton.cdiv(span, plan["splits"]), block_n) * block_n
    splits = plan["splits"] if plan.get("balance") else triton.cdiv(span, chunk)
    rows = length * group
    hpp = plan["hpp"] if hkv % plan["hpp"] == 0 else 1
    dual = plan.get("dual", False) and hpp == 1 and hkv % 2 == 0
    rope = bool(cfg["rope_theta"])
    cs = _cos_sin(cfg["rope_theta"], d, device) if rope else qkv

    fuse = plan["fuse"] and splits <= 64 and not dual  # the fused merge unrolls over splits
    # tsep: the block's new tokens in an extra program (partial slot S) instead of split 0.
    sep = int(bool(plan.get("tsep")) and not fuse)
    # fix (full layers, merged by _combine with its exact REDO): fixed-reference softmax in the key loop.
    fix = plan.get("fix", 0) if sep and not rope and not cfg["window"] and not dual and not plan.get("balance") \
        and plan.get("hb", 1) == 1 else 0
    pacc = torch.empty((batch, hkv, splits + sep, rows, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((batch, hkv, splits + sep, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    _attn_split[(splits + sep, batch, hkv // (2 if dual else hpp))](
        qkv, cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table,
        data["seq_lens"], data["cache_loc"], pacc, pml, _counters(batch * hkv, device), out,
        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        HQ=hq, HKV=hkv, HPP=hpp, G=group, L=length, MG=rows, M_PAD=max(16, triton.next_power_of_2(hpp * rows)),
        L_PAD=max(16, triton.next_power_of_2(length)), D=d, PAGE=cfg["page_size"], MAX_PAGES=max_pages,
        WINDOW=cfg["window"], ROPE=rope, S=splits, CHUNK=chunk, BLOCK_N=block_n, STAGES=plan["stages"],
        FUSE=fuse, S_PAD=triton.next_power_of_2(splits), QK8=plan["qk8"], PDL=USE_PDL and not fuse, DOT=DOT_DTYPE, PEEL=plan.get("peel", False), DUAL=dual, BALANCE=plan.get("balance", False), SEP=bool(sep),
        PF=bool(plan.get("pf")) and not cfg["window"] and not plan.get("balance"),
        PRE=0 if dual else plan.get("pre", 0), MIF=bool(plan.get("mif")), PREQ=0 if dual else plan.get("preq", 0),
        FIX=fix, FDELTA=plan.get("fdelta", 14.0),
        SPEC=max_pages - length if plan.get("spec") and batch == 1 and cfg["page_size"] == 1 and not fuse else 0,
        num_warps=plan["warps"],
        **({"maxnreg": plan["maxnreg"]} if plan.get("maxnreg") else {}))
    if fuse:
        return out
    splits += sep
    hb = plan.get("hb", 1) if group % plan.get("hb", 1) == 0 else 1
    if hb > 1:
        sb = plan.get("sb", max(1, min(triton.next_power_of_2(splits), 32 // hb)))
        if sep:
            sb = triton.next_power_of_2(max(sb, splits))
        _combine_hb[(batch * length, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=splits, S_PAD=triton.next_power_of_2(splits),
            SB=sb, HB=hb, ONE=MERGE1 and not sep, PDL=USE_PDL, num_warps=plan.get("cw", 4), launch_pdl=USE_PDL)
        return out
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=splits, S_PAD=triton.next_power_of_2(splits),
        # Up to 128 splits in one pass, so every partial's load is in flight at once.
        SB=min(128, triton.next_power_of_2(splits)), ONE=MERGE1, PDL=USE_PDL, num_warps=4 if splits > 32 else 2,
        REDO=fix > 0, seq_ptr=data["seq_lens"], kc_ptr=data["k_cache"], vc_ptr=data["v_cache"], ks_ptr=data["k_scale"],
        pt_ptr=page_table, qk_scale=cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, MAX_PAGES=max_pages,
        PAGE=cfg["page_size"], CHUNK=chunk, launch_pdl=USE_PDL)
    return out


def _verify_gluon(data, cfg, plan):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = length * group
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    splits = plan["splits"]
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    qk_scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    if plan.get("gbal"):
        return _verify_bal(data, cfg, plan, out, qk_scale)
    pacc = torch.empty((batch, hkv, splits + 1, rows, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((batch, hkv, splits + 1, rows, 2), dtype=torch.float32, device=device)
    # The in-grid merge (gfuse) needs every program resident at once (one wave).
    pairs = batch * (hkv // 2)
    fuse = plan.get("gfuse", False) and splits * pairs <= 2 * _num_sms(device) \
        and splits < 0xFFFF
    rs = triton.cdiv(2 * rows, splits)
    _new_tokens[(batch, hkv)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
        pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=max(16, triton.next_power_of_2(length)),
        D=d, S=splits, DOT=DOT_DTYPE, num_warps=4)
    _gw_verify[(splits, batch, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, qk_scale,
        HQ=hq, HKV=hkv, G=group, L=length, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
        BN=plan["block_n"], NS=plan["stages"], PW=plan["pw"], PREGS=plan["pregs"], CREGS=plan["cregs"],
        LAG=plan.get("lag", 0), PEEL=plan.get("gpeel", False), LSUM=plan.get("lsum", False), FUSE=fuse, cnt_ptr=_counters(pairs, device, "g"), out_ptr=out, RS=rs,
        R_PAD=triton.next_power_of_2(rs), num_warps=4, maxnreg=plan["maxnreg"], launch_pdl=True)
    if fuse:
        return out
    s = splits + 1
    hb = plan.get("hb", 1)
    if hb > 1:
        _combine_hb[(batch * length, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=s, S_PAD=triton.next_power_of_2(s),
            SB=triton.next_power_of_2(s), HB=hb, ONE=MERGE1, PDL=USE_PDL, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
        return out
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=min(128, triton.next_power_of_2(s)), ONE=MERGE1, PDL=USE_PDL, num_warps=plan.get("cw", 4 if s > 32 else 2),
        launch_pdl=USE_PDL)
    return out


def _verify_bal(data, cfg, plan, out, qk_scale):
    """_verify_gluon with balanced splits (plan gbal): the programs of each KV head pair index share the batch's
    key tiles equally; partials [HKV, NID, rows, D] by id (see _gw_bal); merged by _combine_bal."""
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = length * group
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    # Programs per KV head pair: as many as fill the slots (2 per SM), or the plan's splits per request.
    progs = 2 * _num_sms(device) // (hkv // 2) if plan.get("fill", True) else plan["splits"] * batch
    nid = progs + 2 * batch
    b_pad = plan.get("bpad", max(32, triton.next_power_of_2(batch)))
    pacc = torch.empty((hkv, nid, rows, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((hkv, nid, rows, 2), dtype=torch.float32, device=device)
    nta = plan.get("nta", NT_AFTER)
    if not nta:
        _new_tokens[(batch, hkv)](
            qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
            pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=max(16, triton.next_power_of_2(length)),
            D=d, S=1, DOT=DOT_DTYPE, NID=nid, NBASE=progs + batch, num_warps=4)
    _gw_verify[(progs, 1, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, qk_scale,
        HQ=hq, HKV=hkv, G=group, L=length, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=progs,
        BN=plan["block_n"], NS=plan["stages"], PW=plan["pw"], PREGS=plan["pregs"], CREGS=plan["cregs"],
        LAG=plan.get("lag", 1), BAL=True, B=batch, B_PAD=b_pad, NID=nid, NP=progs, LZ=plan.get("lz", LZ),
        GX=ATT_DIAG if ATT_DIAG in (1, 2, 3, 4) else 0, num_warps=4, maxnreg=plan["maxnreg"],
        launch_pdl=True)
    if nta:
        _new_tokens[(batch, hkv)](
            qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
            pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=max(16, triton.next_power_of_2(length)),
            D=d, S=1, DOT=DOT_DTYPE, NID=nid, NBASE=progs + batch, num_warps=4,
            WAIT=True, launch_pdl=True)
    hb = plan.get("hb", 2)
    _combine_bal[(batch * length, hq // hb)](
        qkv, data["seq_lens"], pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, B=batch, B_PAD=b_pad, P=progs, NID=nid,
        BN=plan["block_n"], SB=plan.get("sb", triton.next_power_of_2(progs // batch + 3)), HB=hb, ONE=MERGE1,
        REDO=plan.get("lz", LZ) == 10, kc_ptr=data["k_cache"], vc_ptr=data["v_cache"], ks_ptr=data["k_scale"],
        pt_ptr=page_table, qk_scale=qk_scale, MAX_PAGES=page_table.shape[1], PAGE=cfg["page_size"],
        num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
    return out


def _red_bufs(batch, group, d, sl, device):
    """plan red's sums and out-of-range lists (batch 1 keeps its s95 buffers; batched ones are per batch size)."""
    if batch == 1:
        return _red_buf(4 * 2 * group * (d + 1), device), _counters(sl + 3, device, "r"), 0
    rcw = sl + 2
    return (_red_buf(batch * BNR * 2 * group * (d + 1), device, f"red{batch}"),
            _counters(batch * rcw, device, f"r{batch}"), rcw)


def _red_args(batch, group, d, sl, device, out):
    red, rcnt, rcw = _red_bufs(batch, group, d, sl, device)
    return dict(RED=True, red_ptr=red, rcnt_ptr=rcnt, out_ptr=out, NR=4 if batch == 1 else BNR, RCW=rcw)


# _red_fin warps: batch 1 (decode 1 full) 16 instead of 4 (fewer loads per thread on the post-wait L2 trip): H100 paired
# -0.27/-0.34 us (15/15 twice), bit-identical; 8 -0.14..-0.28, 32 tie, 1-2 +0.6..+1.7. Batched decode full: 8/16 tie/lose.
RF_NW1 = 16
RF_TRIG = True  # layer s27 (TUNE option rf_trig at T=1; H100 d1 full -1.5..-3.1 us): batch 1: trigger the dependents right after the wait (s4's +2.5 us was a trigger at entry)
RF_NW = 4


def _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, sl, bal):
    red, rcnt, rcw = _red_bufs(batch, group, d, sl, qkv.device)
    _red_fin[(batch,)](qkv, red, rcnt, pacc, pml, data["v_scale"], out, HQ=hq, HKV=hkv, G=group, D=d, M=2 * group,
                       S=splits, NR=4 if batch == 1 else BNR, SL=0 if batch == 1 else sl, RCW=rcw, BAL=bal, num_warps=RF_NW1 if batch == 1 else RF_NW, launch_pdl=True,
                       TRIG=RF_TRIG and batch == 1, QSS=data.get("qkv_ss", 0))


def _decode_gluon(data, cfg, plan):
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    qkv = data["qkv"]
    qss = data.get("qkv_ss", 0)
    device = qkv.device
    page_table = data["page_table"]
    splits = plan["splits"]
    s = splits + 1
    red = plan.get("red", False) and hkv == 2
    if plan.get("dbal"):
        # Length-proportional splits: the plan's splits * batch programs per KV head pair (or as many as fit, 8 per
        # SM), shared out by request length; merged by _combine_bal.
        progs = plan.get("dprogs", splits * batch)
        nid = progs + 2 * batch
        b_pad = max(32, triton.next_power_of_2(batch))
        pacc = torch.empty((hkv, nid, group, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
        pml = torch.empty((hkv, nid, group, 2), dtype=torch.float32, device=device)
        out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
        _gd_full[(progs, 1, hkv // 2)](
            qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
            data["cache_loc"], pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
            _cos_sin(cfg["rope_theta"], d, device) if cfg["rope_theta"] else qkv,
            HQ=hq, HKV=hkv, G=group, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
            BN=plan["block_n"], NS=plan["stages"], WINDOW=cfg["window"], ROPE=bool(cfg["rope_theta"]),
            Q16=plan.get("q16", True), KST=plan.get("kst", True), BAL=True, B=batch, B_PAD=b_pad, NP=progs, NID=nid,
            **(_red_args(batch, group, d, nid, device, out) if red else {}), QSS=qss, DX=ATT_DIAG - 10 if ATT_DIAG in (11, 12) else 0, num_warps=1)
        if red:
            _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, nid, True)
            return out
        hb = plan.get("hb", 2)
        _combine_bal[(batch, hq // hb)](
            qkv, data["seq_lens"], pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, B=batch, B_PAD=b_pad, P=progs, NID=nid,
            BN=plan["block_n"], SB=plan.get("sb", min(128, triton.next_power_of_2(2 * progs // batch + 1))), HB=hb,
            ONE=MERGE1, PROP=True, QSS=qss, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
        return out
    pacc = torch.empty((batch, hkv, s, group, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((batch, hkv, s, group, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    sep = plan.get("sep", False)
    _gd_full[(splits + sep, batch, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        _cos_sin(cfg["rope_theta"], d, device) if cfg["rope_theta"] else qkv,
        HQ=hq, HKV=hkv, G=group, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
        BN=plan["block_n"], NS=plan["stages"], TPV=plan.get("tpv", False), WINDOW=cfg["window"],
        ROPE=bool(cfg["rope_theta"]), SEP=sep, Q16=plan.get("q16", True),
        KST=plan.get("kst", True),
        SPEC=page_table.shape[1] - 1 if plan.get("spec") and batch == 1 and cfg["page_size"] == 1 else 0,
        **(_red_args(batch, group, d, s, device, out) if red else {}), QSS=qss, DX=ATT_DIAG - 10 if ATT_DIAG in (11, 12) else 0, num_warps=1)
    if red:
        _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, s, False)
        return out
    hb = plan.get("hb", 1)
    if hb > 1:
        _combine_hb[(batch, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, S=s, S_PAD=triton.next_power_of_2(s),
            SB=triton.next_power_of_2(s), HB=hb,
            ONE=MERGE1, PDL=USE_PDL, QSS=qss, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
        return out
    _combine[(batch, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=plan.get("csb", min(128, triton.next_power_of_2(s))), ONE=MERGE1, PDL=USE_PDL,
        QSS=qss, num_warps=plan.get("cw", 4 if s > 32 else 2), launch_pdl=USE_PDL)
    return out

# ---- layer ----
# ---- experts (moe_src.py, names prefixed _moe_) ----
"""Routed experts, NVFP4 W4A16, own CUDA kernels (moe agent).

Four launches per call:
  route : one CTA buckets this GPU's (token, slot) routes by local expert (counts, offsets,
          active-expert list sorted by load, sorted route ids).
  gemm13: persistent grouped GEMM over (active expert, row tile) items. Weights are NVFP4
          repacked in prepare() into mma.sync A-fragment order with the 4-bit codes
          bit-permuted so that dequant is ~1 int op per element (masks/shifts land each code
          straight in a bf16's sign/exponent/mantissa bits, a tiny bf16 x a bias-folded block
          scale gives the exact e2m1 * e4m3 value). bf16 mma m16n8k16, FP32 accumulate, 8 warps
          split K, smem reduce; epilogue applies the global scale and SiLU(gate) * up -> bf16.
  gemm2 : the same kernel on w2, writes y bf16 per route.
  norm  : per token, (1 + w) * y, RMSNorm, weighted sum over its local slots (FP32) -> bf16.
"""

import hashlib
from typing import TypedDict

import torch


class _moe_Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class _moe_Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class _moe_Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8]
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: _moe_Config


_moe_H = 4096
_moe_I = 4096
_moe_TOP_K = 8
_moe_LOCAL = 32
_moe_POST_NORM_EPS = 1e-8

_moe_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cuda.h>
#include <cudaTypedefs.h>

// launch, optionally as a PDL secondary (programmatic stream serialization: CUDA-graph capturable)
static int g_pdl = 4, g_twpf = 0, g_blk = 0;  // g_pdl bits: 1 decode gemm2, 2 decode norm2, 4 big-M kernels
template <typename... KA, typename... AA>
static int launch_k(void (*k)(KA...), int grid, int block, int smem, cudaStream_t st, bool pdl, AA... args) {
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(grid);
  lc.blockDim = dim3(block);
  lc.dynamicSmemBytes = smem;
  lc.stream = st;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = la;
  lc.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&lc, k, ((KA)args)...);
  return (int)cudaGetLastError();
}
int64_t moe_set_pdl(int64_t v, int64_t twpf) { g_pdl = (int)v; g_twpf = (int)twpf; return 0; }
int64_t moe_set_blk(int64_t v) { g_blk = (int)v; return 0; }
static int g_sa = 0;  // s9: moe_twf uses the arithmetic scale decode (set from kernel_fn: prepare() checked the scales)
int64_t moe_set_sa(int64_t v) { g_sa = (int)v; return 0; }
static int g_hot = 0;  // s29: moe_twm skips the 65..128-token experts (moe_hot ran them)
int64_t moe_set_hot(int64_t v) { g_hot = (int)v; return 0; }

#define KD 4096          // K of both GEMMs
#define NOUT 4096        // output rows of both GEMMs (gate/up each I, w2 H)
#define TK2F_RING_SA 0   // s12: SA instantiations of the deeper tk2f rings (208/206/403/108): H200 diag 619ad6ad, all lose
#define TK2F_DIAG 0      // layer s81: tk2f DIAG instantiations (no-math / KPS 4 SQ timing probes, tuner-only; set by TK2F_DIAG below)
#define TK2F_L2H 4       // s21: tk2f weight-copy L2 hints: 1 = evict_last for non-final 8-token groups of an expert, evict_first else; 2 = evict_last non-final only; 3 = evict_first final only
#define NORM_EARLY 1     // n4s1 (layer s53: set here, layer has its own _lib): twm/moe_hot flag each expert's finished gemm2 at meta[100 + e] (>= 1 << 20); moe_norme waits per token on them
#ifndef TK2G_PAR4
#define TK2G_PAR4 1      // n17: tk2g_bound's 4 calls on 4 warps + 32-bit p (bitwise: same bounds)
#endif
#ifndef TK2G_PAR
#define TK2G_PAR 1       // n4s5: GS16 weighted split bounds by warp 0 (tk2g_bound) instead of every thread (tk2g_split)
#endif
#define TK2_EXP 0        // s12: tk2h (TK2H/PSC) and weighted-split (TK2W) instantiations: H200 diags, no gain (off: compile time)

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// 8 e2m1 codes (bit-permuted in prepare) -> 4 bf16x2 A registers (rows g, g+8, g, g+8),
// each = code * 2^-126 (exact, subnormals for 0/0.5) times the block scale * 2^118.
// Output 0 = q & M, outputs 1/2 = rotl(q, 10/13) & M, output 3 = the 8 leftover bits (mask 0x001B001B)
// moved by one multiply (x 2^3 + 2^8 + 2^14: the shifted copies never overlap, so no carries) & M.
__device__ __forceinline__ void dq(uint32_t q, uint32_t s0, uint32_t s1, uint32_t* a) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t o0 = q & M;
  const uint32_t o1 = __funnelshift_l(q, q, 10) & M;
  const uint32_t o2 = __funnelshift_l(q, q, 13) & M;
  const uint32_t o3 = ((q & 0x001B001Bu) * 16648u) & M;
  a[0] = bmul(o0, s0);
  a[1] = bmul(o1, s1);
  a[2] = bmul(o2, s0);
  a[3] = bmul(o3, s1);
}

// s9 SA: e4m3 block scale byte k of w -> bf16x2 (both halves) of value * 2^118 without the LUT: exact for positive
// normal scales (prepare() checks all of them, else SA = 0): bf16 bits = (e << 7 | m << 4) + (238 << 7).
// s27 SCB (r-moe): two scales per decode: scp<P>(w) = bf16 halves (byte 2P, byte 2P + 1) in the format of sca<>; the
// broadcast of one half (blo / bhi) folds into HMUL2's .H0_H0 / .H1_H1 operand selector (free). Bitwise = sca<>.
template <int P>
__device__ __forceinline__ uint32_t scp(uint32_t w) {
  return __byte_perm(w, 0u, P ? 0x4342u : 0x4140u) * 16u + 0x77007700u;
}
__device__ __forceinline__ uint32_t blo(uint32_t p) { uint32_t r; asm("{.reg .b16 l,h; mov.b32 {l,h}, %1; mov.b32 %0, {l,l};}" : "=r"(r) : "r"(p)); return r; }
__device__ __forceinline__ uint32_t bhi(uint32_t p) { uint32_t r; asm("{.reg .b16 l,h; mov.b32 {l,h}, %1; mov.b32 %0, {h,h};}" : "=r"(r) : "r"(p)); return r; }

template <int K>
__device__ __forceinline__ uint32_t sca(uint32_t w) {
  return __byte_perm(w, 0u, 0x4040u | (K << 8) | K) * 16u + 0x77007700u;
}

// layer s73 (moe n20 S6, r-moe 4a87466): 6-bit block-scale codes (prepare _s6_pack: c = e4m3 byte - 80, all scales in
// [80, 143]). Per lane and SQ stage (KPS 2) 8 codes k = 4 h + r: n = the low nibbles (bits 4k), hw = the high 2 bits
// (bits 2k). One PRMT puts the nibble in bits 4-7 and the high bits in 8-9 of each half; | 0x7C00 = (c + 80) * 16 +
// 0x7700: the same bf16x2 as sca<r>() of the original byte, 2 instructions per scale.
__device__ __forceinline__ void s6dec(uint32_t n, uint32_t hw, int h, uint32_t* sc) {
  const uint32_t n4 = n << 4;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const uint32_t b = 2 * h + (r >> 1), c = 4 + h;
    const uint32_t x = __byte_perm((r & 1) ? n : n4, hw >> (2 * r), b | (c << 4) | (b << 8) | (c << 12));
    sc[r] = (x & 0x03F003F0u) | 0x7C007C00u;
  }
}

__device__ __forceinline__ uint32_t u4get(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// meta: [0] nact, [1..33) active experts (by load, descending), [33..66) offsets (33 entries).
template <int R, int G, bool GATED, int MB>
__global__ void __launch_bounds__(256, MB) moe_gemm(
    const uint4* __restrict__ Wq, const uint4* __restrict__ Ws, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
    const int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  constexpr int NR = GATED ? 2 * R : R;
  constexpr int NRT = GATED ? 2 * NOUT / 16 : NOUT / 16;  // packed row tiles per expert
  constexpr int TPE = NOUT / 16 / R;                        // items per expert
  __shared__ uint32_t slut[256];
  __shared__ float red[NR * G * 4 * 32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  slut[tid] = lut[tid];
  const int nact = meta[0];
  const int nitems = nact * TPE;
  __syncthreads();
  for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
    const int a = item / TPE, tg = item - a * TPE;
    const int e = meta[1 + a];
    const int off = meta[33 + e];
    const int n = meta[34 + e] - off;
    const uint4* wb = Wq + (size_t)e * NRT * 64 * 32;
    const uint4* sb = Ws + (size_t)e * NRT * 8 * 32;
    int rts[NR];
#pragma unroll
    for (int i = 0; i < R; ++i) {
      rts[i] = tg * R + i;
      if (GATED) rts[R + i] = NOUT / 16 + tg * R + i;
    }
    uint4 sc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) sc[i] = __ldg(sb + ((size_t)rts[i] * 8 + warp) * 32 + lane);
    const float gsc = gs[e] * 256.0f;
    for (int p0 = 0; p0 < n; p0 += 8 * G) {
      for (int i = tid; i < NR * G * 128; i += 256) red[i] = 0.f;
      float acc[NR][G][4];
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[i][gi][r] = 0.f;
      const uint4* xr[G];
      bool xv[G];
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int idx = p0 + gi * 8 + g;
        xv[gi] = idx < n;
        int row = 0;
        if (xv[gi]) row = GATED ? (sroutes[off + idx] >> 3) : (off + idx);
        xr[gi] = reinterpret_cast<const uint4*>(X + (size_t)row * KD + warp * 512 + 16 * t);
      }
      constexpr int PF = (NR * 8 <= 16) ? 8 : (NR * 4 <= 16) ? 4 : 2;  // k chunks loaded per batch
#pragma unroll
      for (int c0 = 0; c0 < 8; c0 += PF) {
        uint4 w[PF][NR];
        uint4 xb[PF][G][2];
#pragma unroll
        for (int c = 0; c < PF; ++c)
#pragma unroll
          for (int i = 0; i < NR; ++i)
            w[c][i] = ld_stream(wb + ((size_t)rts[i] * 64 + warp * 8 + c0 + c) * 32 + lane);
#pragma unroll
        for (int c = 0; c < PF; ++c)
#pragma unroll
          for (int gi = 0; gi < G; ++gi) {
            if (xv[gi]) {
              xb[c][gi][0] = __ldg(xr[gi] + (c0 + c) * 8);
              xb[c][gi][1] = __ldg(xr[gi] + (c0 + c) * 8 + 1);
            } else {
              xb[c][gi][0] = make_uint4(0, 0, 0, 0);
              xb[c][gi][1] = make_uint4(0, 0, 0, 0);
            }
          }
#pragma unroll
        for (int c = 0; c < PF; ++c) {
          const int kc = c0 + c;
#pragma unroll
          for (int i = 0; i < NR; ++i) {
            const uint32_t sw = u4get(sc[i], kc >> 1) >> ((kc & 1) * 16);
            const uint32_t s0 = slut[sw & 0xFF], s1 = slut[(sw >> 8) & 0xFF];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              uint32_t af[4];
              dq(u4get(w[c][i], j), s0, s1, af);
#pragma unroll
              for (int gi = 0; gi < G; ++gi) {
                const uint32_t b0 = u4get(xb[c][gi][j >> 1], (j & 1) * 2);
                const uint32_t b1 = u4get(xb[c][gi][j >> 1], (j & 1) * 2 + 1);
                mma16816(acc[i][gi], af, b0, b1);
              }
            }
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
#pragma unroll
          for (int r = 0; r < 4; ++r) atomicAdd(&red[((i * G + gi) * 4 + r) * 32 + lane], acc[i][gi][r]);
      __syncthreads();
      constexpr int NE = R * G * 128;
      for (int idx = tid; idx < NE; idx += 256) {
        const int ln = idx & 31, r = (idx >> 5) & 3, gi = (idx >> 7) % G, i = idx / (128 * G);
        const int rowin = (ln >> 2) + ((r & 2) ? 8 : 0);
        const int tok = p0 + gi * 8 + (ln & 3) * 2 + (r & 1);
        if (tok < n) {
          const int col = (tg * R + i) * 16 + rowin;
          const float v = red[idx] * gsc;
          if (GATED) {
            const float u = red[idx + R * G * 128] * gsc;
            const float h = __fdividef(v, 1.0f + __expf(-v)) * u;
            Out[(size_t)(off + tok) * NOUT + col] = __float2bfloat16(h);
          } else {
            Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void __launch_bounds__(1024) moe_route(const int* __restrict__ ids, int nroutes, int start, int count,
                                                  int* __restrict__ meta, int* __restrict__ sroutes) {
  __shared__ int cnt[32], base[32], fill[32];
  const int tid = threadIdx.x;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 32) { cnt[tid] = 0; fill[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < nroutes; i += blockDim.x) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid < 32) {
    const int c = cnt[tid];
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    base[tid] = incl - c;
    meta[33 + tid] = incl - c;
    if (tid == 31) { meta[65] = incl; meta[66] = 0; meta[67] = 0; }
    meta[68 + tid] = 0;  // moe_twf: gemm13 items done per expert
#if NORM_EARLY
    meta[100 + tid] = 0;  // moe_twm/moe_hot: gemm2 items done per expert (+ 1 << 20 when all are)
#endif
    // rank among active experts: heavier first, ties by id
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(0xffffffffu, c, j);
      rank += (cj > c) || (cj == c && j < tid);
    }
    const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
    if (tid == 0) meta[0] = na;
    if (tid >= na) meta[1 + tid] = 0;
    __syncwarp();
    if (c > 0) meta[1 + rank] = tid;
  }
  __syncthreads();
  for (int i = tid; i < nroutes; i += blockDim.x) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) sroutes[base[e] + atomicAdd(&fill[e], 1)] = i;
  }
}

// One CTA (512 threads x 8 values) per token.
__global__ void __launch_bounds__(512) moe_norm(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ ids,
                                                const float* __restrict__ tw, const float* __restrict__ w1,
                                                int start, int count, float eps, __nv_bfloat16* __restrict__ out,
                                                const int* rdy) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  if (rdy) {
    // n4s1 (NORM_EARLY): wait only for the experts of this token's routes (flagged by moe_twm / moe_hot after their
    // last gemm2 item of the expert): CTAs run on the SMs twm's drain frees. Y then read through L2 (ld.cg).
    if (tid < 8) {
      const int e = __ldcg(ids + tok * 8 + tid) - start;
      if (e >= 0 && e < count) {
        int fv;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(fv) : "l"(rdy + e) : "memory");
          if (fv >= (1 << 20)) break;
          __nanosleep(256);
        }
      }
    }
    __syncthreads();
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of gemm2
  }
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = __ldcg(ids + tok * 8 + s) - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      uint4 raw = __ldcg(reinterpret_cast<const uint4*>(Y + ((size_t)(tok * 8 + s)) * NOUT + tid * 8));
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 f = __bfloat1622float2(p[k]);
        v[s][2 * k] = f.x * wv[2 * k];
        v[s][2 * k + 1] = f.y * wv[2 * k + 1];
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) ss[s] += v[s][k] * v[s][k];
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[s][k] = 0.f;
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float x = ss[s];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) ssum[warp][s] = x;
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!loc[s]) continue;
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) tot += ssum[w][s];
    const float f = __ldcg(tw + tok * 8 + s) * rsqrtf(tot / NOUT + eps);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += f * v[s][k];
  }
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  *reinterpret_cast<uint4*>(out + (size_t)tok * NOUT + tid * 8) = o;
}


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }


template <int NS, int NG, int NW, typename F>
__device__ __forceinline__ void big_kloop(float (&acc)[8][4], const F& issue, const uint4* __restrict__ sb, int rt,
                                          const uint4* xs, const uint4* wsm, const uint32_t* slut,
                                          int warp, int lane, int g, int t) {
  constexpr int XS = 64 * 9, WS = NW * 32;
      
#pragma unroll
  for (int gi = 0; gi < 8; ++gi)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[gi][r] = 0.f;
  uint4 sc, scn = __ldg(sb + ((size_t)rt * 8) * 32 + lane);
#pragma unroll 1
  for (int kc0 = 0; kc0 < 64; kc0 += NS) {
    if ((kc0 & 7) == 0) {
      sc = scn;
      if (kc0 < 56) scn = __ldg(sb + ((size_t)rt * 8 + (kc0 >> 3) + 1) * 32 + lane);
    } else {
      sc.x = sc.z; sc.y = sc.w;  // NS == 4, second half of the scale group
    }
#pragma unroll
    for (int u = 0; u < NS; ++u) {
      const int kc = kc0 + u;
      cp_wait<NS - 2>();
      __syncthreads();
      if (kc + NS - 1 < 64) issue(kc + NS - 1);
      cp_commit();
      const uint4 wv = wsm[u * WS + warp * 32 + lane];
      const uint32_t sw = u4get(sc, u >> 1) >> ((u & 1) * 16);
      const uint32_t s0 = slut[sw & 0xFF], s1 = slut[(sw >> 8) & 0xFF];
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        uint32_t af[2][4];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) dq(u4get(wv, jp * 2 + jj), s0, s1, af[jj]);
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
          if (gi < NG) {
            const uint4 b = xs[u * XS + (gi * 8 + g) * 9 + t * 2 + jp];
            mma16816(acc[gi], af[0], b.x, b.y);
            mma16816(acc[gi], af[1], b.z, b.w);
          }
        }
      }
    }
  }
}

// Big-M grouped GEMM. Item = (active expert, 256 packed rows); each of the 16 warps owns one 16-row tile
// over the full K (gated: warps 0-7 gate tiles, 8-15 the matching up tiles), so every weight is
// dequantized once per 64-token pass. All tokens' activations for a 64-wide k chunk are staged in smem
// (shared by the 16 warps) together with each warp's weights for that chunk, in an NS-stage cp.async
// ring. Items are taken from a global counter (heaviest experts first).
template <bool GATED, int NS, int NW, int MB>
__global__ void __launch_bounds__(NW * 32, MB) moe_big(
    const uint4* __restrict__ Wq, const uint4* __restrict__ Ws, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  constexpr int NRT = GATED ? 2 * NOUT / 16 : NOUT / 16;
  constexpr int IPE = GATED ? NOUT / (8 * NW) : NOUT / (16 * NW);  // items per expert
  constexpr int XS = 64 * 9;                             // uint4 per x stage (64 rows, 144 B stride)
  constexpr int WS = NW * 32;                            // uint4 per weight stage
  constexpr int HW = NW / 2;
  static_assert(NS == 4 || NS == 8, "NS");
  extern __shared__ uint4 smem[];
  uint4* xs = smem;
  uint4* wsm = smem + NS * XS;
  __shared__ uint32_t slut[256];
  __shared__ int s_item;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += NW * 32) slut[i] = lut[i];
  const int nitems = meta[0] * IPE;
  int* counter = meta + (GATED ? 66 : 67);
  for (;;) {
    __syncthreads();
    if (tid == 0) s_item = atomicAdd(counter, 1);
    __syncthreads();
    const int item = s_item;
    if (item >= nitems) break;
    const int a = item / IPE, rb = item - a * IPE;
    const int e = meta[1 + a];
    const int off = meta[33 + e];
    const int n = meta[34 + e] - off;
    const uint4* wb = Wq + (size_t)e * NRT * 64 * 32;
    const uint4* sb = Ws + (size_t)e * NRT * 8 * 32;
    const int rt = GATED ? ((warp >= HW) ? NOUT / 16 : 0) + rb * HW + (warp % HW) : rb * NW + warp;
    const float gsc = gs[e] * 256.0f;
    for (int p0 = 0; p0 < n; p0 += 64) {
      const int nv = min(64, n - p0);
      const int ng = (nv + 7) >> 3;
      // x chunks this thread copies: c = tid + h * NW * 32 -> row c >> 3, 16 B segment c & 7
      constexpr int XH = 512 / (NW * 32);
      const uint4* xsrc[XH];
      uint4* xdst[XH];
      int xbytes[XH];
      bool xdo[XH];
#pragma unroll
      for (int h = 0; h < XH; ++h) {
        const int c = tid + h * NW * 32, xrow = c >> 3;
        xdo[h] = xrow < ng * 8;
        xbytes[h] = xrow < nv ? 16 : 0;
        int src = 0;
        if (xrow < nv) src = GATED ? (sroutes[off + p0 + xrow] >> 3) : (off + p0 + xrow);
        xsrc[h] = reinterpret_cast<const uint4*>(X + (size_t)src * KD) + (c & 7);
        xdst[h] = xs + xrow * 9 + (c & 7);
      }
      const uint4* wsrc = wb + (size_t)rt * 64 * 32 + lane;
      uint4* wdst = wsm + warp * 32 + lane;
      auto issue = [&](int kc) {
        const int s = kc % NS;
#pragma unroll
        for (int h = 0; h < XH; ++h)
          if (xdo[h]) cp_async16(xdst[h] + s * XS, xsrc[h] + kc * 8, xbytes[h]);
        cp_async16(wdst + s * WS, wsrc + kc * 32, 16);
      };
#pragma unroll
      for (int s = 0; s < NS - 1; ++s) { issue(s); cp_commit(); }
      float acc[8][4];
      switch (ng) {
        case 1: big_kloop<NS, 1, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
        case 2: big_kloop<NS, 2, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
        case 3: case 4: big_kloop<NS, 4, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
        default: big_kloop<NS, 8, NW>(acc, issue, sb, rt, xs, wsm, slut, warp, lane, g, t); break;
      }
      cp_wait<0>();
      __syncthreads();
      float* red = reinterpret_cast<float*>(smem);  // gated: up tiles hand their values to the gate warps
      if (GATED && warp >= HW) {
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
          if (gi < ng)
#pragma unroll
            for (int r = 0; r < 4; ++r) red[(((warp - HW) * 8 + gi) * 4 + r) * 32 + lane] = acc[gi][r];
      }
      if (GATED) __syncthreads();
      if (!GATED || warp < HW) {
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
          if (gi < ng) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const int col = rt * 16 + g + ((r & 2) ? 8 : 0);
              const int tok = p0 + gi * 8 + t * 2 + (r & 1);
              if (tok < n) {
                const float v = acc[gi][r] * gsc;
                if (GATED) {
                  const float u = red[((warp * 8 + gi) * 4 + r) * 32 + lane] * gsc;
                  Out[(size_t)(off + tok) * NOUT + col] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
                } else {
                  Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
                }
              }
            }
          }
        }
      }
      __syncthreads();  // red / stage buffers are reused by the next pass
    }
  }
}

template <bool GATED, int NS, int NW, int MB>
static int launch_big(const void* wq, const void* ws, const void* gs, const void* lut, const void* x,
                      const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NS * (64 * 9 + NW * 32) * 16;
  if (!grid) {
    int dev, nsm, occ;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_big<GATED, NS, NW, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_big<GATED, NS, NW, MB>, NW * 32, SMEM);
    if (occ < 1) occ = 1;
    grid = nsm * occ;
  }
  moe_big<GATED, NS, NW, MB><<<grid, NW * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint4*)ws, (const float*)gs, (const uint32_t*)lut, (const __nv_bfloat16*)x,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

template <int R, int G, bool GATED, int MB = (R * G == 1) ? 4 : 2>
static int launch_gemm(const void* wq, const void* ws, const void* gs, const void* lut, const void* x,
                       const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  if (!grid) {
    int dev, nsm, occ;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_gemm<R, G, GATED, MB>, 256, 0);
    if (occ < 1) occ = 1;
    grid = nsm * occ;
  }
  moe_gemm<R, G, GATED, MB><<<grid, 256, 0, st>>>(
      (const uint4*)wq, (const uint4*)ws, (const float*)gs, (const uint32_t*)lut, (const __nv_bfloat16*)x,
      (const int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

#define CFG(R, G) (R * 10 + G)
#define DISPATCH(GATED)                                                                     \
  switch (cfg) {                                                                            \
    case 10: return launch_gemm<1, 1, GATED, 3>(wq, ws, gs, lut, x, meta, sr, out, st);    \
    case CFG(1, 1): return launch_gemm<1, 1, GATED>(wq, ws, gs, lut, x, meta, sr, out, st); \
    case CFG(2, 2): return launch_gemm<2, 2, GATED>(wq, ws, gs, lut, x, meta, sr, out, st); \
    case 118: return launch_big<GATED, 8, 8, 2>(wq, ws, gs, lut, x, meta, sr, out, st);     \
  }                                                                                         \
  return -1;


// ---- decode path (M <= 32): stream-K over (active expert, row-tile pair, k chunk) units ----
// Acc13 [pos][8192] / Acc2 [pos][4096] FP32 are zeroed in-call (route2 / act) and receive red.add partials.

// CTA 0: the route (as moe_route) plus inv[route] = sorted position. CTAs 1..: zero Acc13's used rows.
__global__ void __launch_bounds__(256) moe_route2(const int* __restrict__ ids, int nroutes, int start, int count,
                                                  int* __restrict__ meta, int* __restrict__ sroutes,
                                                  int* __restrict__ inv, float4* __restrict__ acc13) {
  const int tid = threadIdx.x;
  if (blockIdx.x > 0) {
    int c = 0;
    for (int i = tid; i < nroutes; i += 256) { const int e = ids[i] - start; c += (e >= 0 && e < count); }
    __shared__ int s_n;
    if (tid == 0) s_n = 0;
    __syncthreads();
    if (c) atomicAdd(&s_n, c);
    __syncthreads();
    const size_t tot = (size_t)s_n * 2048;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)(blockIdx.x - 1) * 256 + tid; i < tot; i += (size_t)(gridDim.x - 1) * 256) acc13[i] = z;
    return;
  }
  __shared__ int cnt[32], base[32], fill[32];
  if (tid < 32) { cnt[tid] = 0; fill[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < nroutes; i += 256) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid < 32) {
    const int c = cnt[tid];
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    base[tid] = incl - c;
    meta[33 + tid] = incl - c;
    if (tid == 31) { meta[65] = incl; meta[66] = 0; meta[67] = 0; }
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(0xffffffffu, c, j);
      rank += (cj > c) || (cj == c && j < tid);
    }
    const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
    if (tid == 0) meta[0] = na;
    if (tid >= na) meta[1 + tid] = 0;
    __syncwarp();
    if (c > 0) meta[1 + rank] = tid;
  }
  __syncthreads();
  for (int i = tid; i < nroutes; i += 256) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) {
      const int pos = base[e] + atomicAdd(&fill[e], 1);
      sroutes[pos] = i;
      inv[i] = pos;
    }
  }
}

// One segment of a warp's unit range, all in expert e. Weights (both row tiles + their 4 scale bytes) are
// loaded D units ahead in a register ring; x fragments of NG token groups one unit ahead.
template <int NRT, int NG, int D, bool XROUTE>
__device__ __forceinline__ void sk_seg(int u, const int send, const uint4* __restrict__ Wq,
                                       const uint32_t* __restrict__ S2, const uint32_t* slut,
                                       const __nv_bfloat16* __restrict__ X, const int* __restrict__ sroutes,
                                       float* __restrict__ Acc, int e, int off, int n, int lane) {
  constexpr int UPE = NRT / 2 * 64;
  static_assert(D % 2 == 0, "D even (x double buffer)");
  const int g = lane >> 2, t = lane & 3;
  const int lu0 = u & (UPE - 1);
  const uint4* wp = Wq + ((size_t)e * NRT * 64 + lu0) * 32 + lane;
  const uint32_t* sp = S2 + ((size_t)e * UPE + lu0) * 32 + lane;
  const uint4* xr[NG];
  bool xv[NG];
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    const int idx = gi * 8 + g;
    xv[gi] = idx < n;
    int row = 0;
    if (xv[gi]) row = XROUTE ? (sroutes[off + idx] >> 3) : (off + idx);
    xr[gi] = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
  }
  int xk = lu0 & 63;  // k chunk of the next x load
  uint4 rw0[D], rw1[D], rx[2][NG][2];
  uint32_t rs[D];
  auto issue = [&](int s) {
    rw0[s] = ld_stream(wp);
    rw1[s] = ld_stream(wp + NRT * 32 * 32);
    rs[s] = __ldg(sp);
    wp += 32;
    sp += 32;
  };
  auto xload = [&](int b) {
#pragma unroll
    for (int gi = 0; gi < NG; ++gi) {
      if (xv[gi]) {
        rx[b][gi][0] = __ldg(xr[gi] + xk * 8);
        rx[b][gi][1] = __ldg(xr[gi] + xk * 8 + 1);
      } else {
        rx[b][gi][0] = make_uint4(0, 0, 0, 0);
        rx[b][gi][1] = make_uint4(0, 0, 0, 0);
      }
    }
    xk = (xk + 1) & 63;
  };
#pragma unroll
  for (int s = 0; s < D; ++s)
    if (u + s < send) issue(s);
  xload(0);
  float acc[2][NG][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int gi = 0; gi < NG; ++gi)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][gi][r] = 0.f;
  for (int base = u; base < send; base += D) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      const int uc = base + s;
      if (uc >= send) break;
      if (uc + 1 < send) xload((s + 1) & 1);
      const uint32_t sw = rs[s];
      const uint32_t sc[4] = {slut[sw & 0xFF], slut[(sw >> 8) & 0xFF], slut[(sw >> 16) & 0xFF], slut[sw >> 24]};
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 wv = i ? rw1[s] : rw0[s];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * i], sc[2 * i + 1], af);
#pragma unroll
          for (int gi = 0; gi < NG; ++gi)
            mma16816(acc[i][gi], af, u4get(rx[s & 1][gi][j >> 1], (j & 1) * 2),
                     u4get(rx[s & 1][gi][j >> 1], (j & 1) * 2 + 1));
        }
      }
      if (uc + D < send) issue(s);
      if (((uc + 1) & 63) == 0 || uc + 1 == send) {
        const int rp = (uc & (UPE - 1)) >> 6;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int gi = 0; gi < NG; ++gi)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              const int tok = gi * 8 + 2 * t + (r & 1);
              const int col = (rp + i * (NRT / 2)) * 16 + g + ((r & 2) ? 8 : 0);
              if (tok < n) atomicAdd(Acc + (size_t)(off + tok) * (NRT * 16) + col, acc[i][gi][r]);
              acc[i][gi][r] = 0.f;
            }
      }
    }
  }
}

// Units: for each active expert a and each group of 8 of its tokens, UPE (row-tile pair, k chunk) units
// (experts with > 8 tokens re-read their weights per group: rare in decode). Each warp takes a contiguous
// range of units: perfect balance whatever the routing, partial sums flushed with red.add.
template <int NRT, int D, int MB, bool XROUTE>
__global__ void __launch_bounds__(256, MB) moe_sk(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2,
                                                  const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
                                                  const int* __restrict__ meta, const int* __restrict__ sroutes,
                                                  float* __restrict__ Acc) {
  constexpr int UPE = NRT / 2 * 64;
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_us[33];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  slut[tid] = lut[tid];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  const int nact = meta[0];
  if (tid < 32) {
    const int e = meta[1 + tid];
    s_e[tid] = e;
    const int ng = tid < nact ? (meta[34 + e] - meta[33 + e] + 7) >> 3 : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_us[tid + 1] = incl * UPE;
    if (tid == 0) s_us[0] = 0;
  }
  __syncthreads();
  const unsigned W = gridDim.x * 8, gw = blockIdx.x * 8 + warp;
  const unsigned U = (unsigned)s_us[nact];
  const int u0 = (int)(((unsigned long long)U * gw) / W), u1 = (int)(((unsigned long long)U * (gw + 1)) / W);
  int u = u0, a = 0;
  while (u < u1) {
    while (s_us[a + 1] <= u) ++a;
    const int gi = (u - s_us[a]) / UPE;
    const int send = min(u1, s_us[a] + (gi + 1) * UPE);
    const int e = s_e[a], off = s_off[e] + gi * 8, n = min(8, s_off[e + 1] - off);
    sk_seg<NRT, 1, D, XROUTE>(u, send, Wq, S2, slut, X, sroutes, Acc, e, off, n, lane);
    u = send;
  }
}

// inter[pos] = bf16(silu(gate) * up) from Acc13 (x global scale); zeroes Acc2[pos].
__global__ void __launch_bounds__(256) moe_act(const float4* __restrict__ acc13, const int* __restrict__ meta,
                                               const int* __restrict__ sroutes, const int* __restrict__ ids,
                                               const float* __restrict__ gs, int start, __nv_bfloat16* __restrict__ inter,
                                               float4* __restrict__ acc2) {
  const int pos = blockIdx.x, tid = threadIdx.x;
  if (pos >= meta[65]) return;
  const int e = ids[sroutes[pos]] - start;
  const float gsc = gs[e] * 256.0f;
  const float4* gp = acc13 + (size_t)pos * 2048 + tid * 4;
  const float4* up = gp + 1024;
  uint32_t o[8];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 a = gp[k], b = up[k];
    const float v[4] = {a.x * gsc, a.y * gsc, a.z * gsc, a.w * gsc}, w[4] = {b.x * gsc, b.y * gsc, b.z * gsc, b.w * gsc};
    float h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) h[q] = __fdividef(v[q], 1.0f + __expf(-v[q])) * w[q];
    __nv_bfloat162 p0 = __floats2bfloat162_rn(h[0], h[1]), p1 = __floats2bfloat162_rn(h[2], h[3]);
    o[2 * k] = *reinterpret_cast<uint32_t*>(&p0);
    o[2 * k + 1] = *reinterpret_cast<uint32_t*>(&p1);
  }
  uint4* dst = reinterpret_cast<uint4*>(inter + (size_t)pos * NOUT + tid * 16);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < 4; ++k) acc2[(size_t)pos * 1024 + tid * 4 + k] = z;
}

// As moe_norm, but y comes from Acc2 (FP32, sorted positions, x global scale).
// SLOT (s13, moe_tk2f PF & 4): Y row = route slot tok * 8 + s, already scaled by gs2 * 256 (no inv / gs loads)
template <bool SLOT = false>
__global__ void __launch_bounds__(512) moe_norm2(const float* __restrict__ Y, const int* __restrict__ inv,
                                                 const int* __restrict__ ids, const float* __restrict__ tw,
                                                 const float* __restrict__ w1, const float* __restrict__ gs, int start,
                                                 int count, float eps, __nv_bfloat16* __restrict__ out, int* zc) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of gemm2
  if (tok == 0 && tid < 128) zc[tid] = 0;  // moe_tk2f's segment counters: zero again for the next call
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = __ldcg(ids + tok * 8 + s) - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      const float4* p = reinterpret_cast<const float4*>(Y + (size_t)(SLOT ? tok * 8 + s : __ldcg(inv + tok * 8 + s)) * NOUT + tid * 8);
      const float4 a = p[0], b = p[1];
      const float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      if (SLOT) {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[s][k] = f[k] * wv[k];
      } else {
        const float gsc = gs[e] * 256.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) v[s][k] = f[k] * gsc * wv[k];
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) ss[s] += v[s][k] * v[s][k];
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) v[s][k] = 0.f;
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float x = ss[s];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) ssum[warp][s] = x;
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (!loc[s]) continue;
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) tot += ssum[w][s];
    const float f = __ldcg(tw + tok * 8 + s) * rsqrtf(tot / NOUT + eps);
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] += f * v[s][k];
  }
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[2 * k], acc[2 * k + 1]);
  *reinterpret_cast<uint4*>(out + (size_t)tok * NOUT + tid * 8) = o;
}

template <int NRT, int D, int MB, bool XROUTE>
static int launch_sk(const void* wq, const void* s2, const void* lut, const void* x, const void* meta,
                     const void* sroutes, void* acc, cudaStream_t st) {
  static int grid = 0;
  if (!grid) {
    int dev, nsm, occ;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, moe_sk<NRT, D, MB, XROUTE>, 256, 0);
    if (occ < 1) occ = 1;
    grid = nsm * occ;
  }
  moe_sk<NRT, D, MB, XROUTE><<<grid, 256, 0, st>>>((const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut,
                                                    (const __nv_bfloat16*)x, (const int*)meta, (const int*)sroutes,
                                                    (float*)acc);
  return (int)cudaGetLastError();
}


// ---- TMA decode GEMM: one CTA per SM, a producer lane streams each block of NC consecutive units (both row
// tiles' weights + scales: 3 contiguous bulk copies) into an NSTG-stage smem ring; consumer warp w takes unit w
// of every block, i.e. k chunks w, w + NC, ... of each row-tile pair, so its x fragments stay in registers.
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* src, int bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(void* dst, const void* src, int bytes, uint64_t* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
}

template <int NRT, int NC, int NSTG, bool XROUTE>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2,
                                                          const uint32_t* __restrict__ lut,
                                                          const __nv_bfloat16* __restrict__ X,
                                                          const int* __restrict__ meta, const int* __restrict__ sroutes,
                                                          float* __restrict__ Acc) {
  constexpr int RPE = NRT / 2;              // row-tile pairs per (expert, token group)
  constexpr int NQ = 64 / NC;               // blocks per row-tile pair
  constexpr int SW = NC * 32;               // uint4 per weight half-stage
  constexpr int SB = NC * 1152;             // bytes per stage
  extern __shared__ __align__(128) uint4 smem[];
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_rs[33];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  const int nact = meta[0];
  if (tid < 32) {
    const int e = meta[1 + tid];
    s_e[tid] = e;
    const int ng = tid < nact ? (meta[34 + e] - meta[33 + e] + 7) >> 3 : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_rs[tid + 1] = incl * RPE;
    if (tid == 0) s_rs[0] = 0;
  }
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NC * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nrp = s_rs[nact];
  const int r0 = (int)((long long)nrp * blockIdx.x / gridDim.x), r1 = (int)((long long)nrp * (blockIdx.x + 1) / gridDim.x);
  if (warp == NC) {
    if (lane == 0) {
      int a = 0, st = 0, ph = 0, i = 0;
      for (int R = r0; R < r1; ++R) {
        while (s_rs[a + 1] <= R) ++a;
        const int e = s_e[a];
        const int lr = (R - s_rs[a]) & (RPE - 1);
        const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
        const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
        for (int q = 0; q < NQ; ++q, ++i) {
          if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
          char* sbase = reinterpret_cast<char*>(smem) + st * SB;
          mbar_expect_tx(&full[st], SB);
          bulk_g2s(sbase, src + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 512, src + NRT * 32 * 32 + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 1024, ssrc + q * NC * 32, NC * 128, &full[st]);
          if (++st == NSTG) { st = 0; ph ^= 1; }
        }
      }
    }
    return;
  }
  // consumer warp
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, st = 0, ph = 0;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {  // new (expert, token group) segment: x fragments for this warp's k chunks
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;  // segment = (expert, token group)
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      const bool v = g < n;
      int row = 0;
      if (v) row = XROUTE ? (sroutes[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
        xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
      }
    }
    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      const uint4 wv0 = sw4[0];
      const uint4 wv1 = sw4[SW];
      const uint32_t sw = reinterpret_cast<const uint32_t*>(sw4 + 2 * SW - warp * 32 - lane)[warp * 32 + lane];
      mbar_arrive(&empty[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
      const uint32_t sc[4] = {slut[sw & 0xFF], slut[(sw >> 8) & 0xFF], slut[(sw >> 16) & 0xFF], slut[sw >> 24]};
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        const uint4 wv = ii ? wv1 : wv0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
          mma16816(acc[ii], af, u4get(xf[q][j >> 1], (j & 1) * 2), u4get(xf[q][j >> 1], (j & 1) * 2 + 1));
        }
      }
    }
    const int rp = (R - seg_start) & (RPE - 1);
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int tok = 2 * t + (r & 1);
        const int col = (rp + ii * RPE) * 16 + g + ((r & 2) ? 8 : 0);
        if (tok < n) atomicAdd(Acc + (size_t)(off + tok) * (NRT * 16) + col, acc[ii][r]);
      }
  }
}

template <int NRT, int NC, int NSTG, bool XROUTE>
static int launch_tk(const void* wq, const void* s2, const void* lut, const void* x, const void* meta,
                     const void* sroutes, void* acc, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * NC * 1152;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tk<NRT, NC, NSTG, XROUTE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tk<NRT, NC, NSTG, XROUTE><<<grid, (NC + 1) * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const __nv_bfloat16*)x, (const int*)meta,
      (const int*)sroutes, (float*)acc);
  return (int)cudaGetLastError();
}


// ---- moe_tk2: as moe_tk, but each row-tile pair is reduced across the consumer warps in smem and written with
// plain stores (no atomics, no zeroing). GATED (gemm13): every CTA computes the routing itself (deterministic,
// identical in all CTAs; CTA 0 publishes meta / sroutes / inv for the later kernels), epilogue silu(gate) * up
// -> bf16 inter[pos]. !GATED (gemm2): routing from meta, epilogue writes FP32 y[pos] (x global scale later).
// row R (local index in its expert's rows, ng token groups) -> row pair rp, group gi. blk = 0 or ng = 1: groups one
// after the other (rp fastest); else blocks of blk rps cycle through the groups, so a CTA re-reads the same weights
// for the next group right away (from L2)
__device__ __forceinline__ int tk2_map(int local, int ng, int blk, int rpe, int& gi) {
  if (ng == 1 || !blk) {
    gi = local / rpe;
    return local & (rpe - 1);
  }
  const int b = local / (blk * ng), w = local - b * blk * ng;
  gi = w / blk;
  return b * blk + (w - gi * blk);
}

template <int NRT, int NC, int NSTG, bool GATED>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2(
    const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, const uint32_t* __restrict__ lut,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids, int nroutes,
    int start, int count, int* __restrict__ meta, int* __restrict__ sroutes, int* __restrict__ inv,
    void* __restrict__ Out, int flags) {
  static_assert(NC == 8, "routing uses 8 warps x 32 ids");
  constexpr int RPE = NRT / 2;
  constexpr int NQ = 64 / NC;
  constexpr int SW = NC * 32;
  constexpr int SB = NC * 1152;
  extern __shared__ __align__(128) uint4 smem[];
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_rs[33], s_srt[256], s_wc[8][32], s_nact;
  __shared__ float s_red[2][NC * 256];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NC * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  const int blk = flags & 0xFFFF;
  // PDL: let the next kernel's CTAs launch as soon as SMs free up (they wait for us before reading our output)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (GATED || (flags >> 16)) {
    // routing (M * 8 <= 256 routes): warp w takes routes [32 w, 32 w + 32); gemm2 recomputes it (same result)
    // instead of reading gemm13's meta, so its producer can stream w2 before gemm13 has finished
    int e = -1, rank = 0;
    if (warp < 8) {
      s_wc[warp][lane] = 0;
      const int i = warp * 32 + lane;
      if (i < nroutes) {
        const int v = ids[i] - start;
        if (v >= 0 && v < count) e = v;
      }
      const unsigned m = __match_any_sync(0xffffffffu, e);
      rank = __popc(m & ((1u << lane) - 1));
      __syncwarp();
      if (e >= 0 && rank == 0) s_wc[warp][e] = __popc(m);
    }
    __syncthreads();
    if (tid < 32) {
      int c = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) c += s_wc[w][tid];
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      int run = incl - c;
      s_off[tid] = run;
      if (tid == 31) s_off[32] = incl;
#pragma unroll
      for (int w = 0; w < 8; ++w) { const int cw = s_wc[w][tid]; s_wc[w][tid] = run; run += cw; }
      int rk = 0;
      for (int j = 0; j < 32; ++j) {
        const int cj = __shfl_sync(0xffffffffu, c, j);
        rk += (cj > c) || (cj == c && j < tid);
      }
      const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
      if (tid == 0) s_nact = na;
      if (tid >= na) s_e[tid] = 0;
      __syncwarp();
      if (c > 0) s_e[rk] = tid;
    }
    __syncthreads();
    if (e >= 0) {
      const int i = warp * 32 + lane, pos = s_wc[warp][e] + rank;
      s_srt[pos] = i;
      if (GATED && blockIdx.x == 0) { sroutes[pos] = i; inv[i] = pos; }
    }
    if (GATED && blockIdx.x == 0 && tid < 33) {
      meta[33 + tid] = s_off[tid];
      if (tid < 32) meta[1 + tid] = s_e[tid];
      if (tid == 0) meta[0] = s_nact;
    }
  } else {  // gemm2 not a PDL secondary: gemm13's routing
    if (tid < 33) s_off[tid] = meta[33 + tid];
    if (tid < 32) s_e[tid] = meta[1 + tid];
    if (tid == 0) s_nact = meta[0];
  }
  __syncthreads();
  const int nact = s_nact;
  if (tid < 32) {
    const int e = s_e[tid];
    const int ng = tid < nact ? (s_off[e + 1] - s_off[e] + 7) >> 3 : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_rs[tid + 1] = incl * RPE;
    if (tid == 0) s_rs[0] = 0;
  }
  __syncthreads();
  const int nrp = s_rs[nact];
  const int r0 = (int)((long long)nrp * blockIdx.x / gridDim.x), r1 = (int)((long long)nrp * (blockIdx.x + 1) / gridDim.x);
  if (warp == NC) {
    if (lane == 0) {
      int a = 0, st = 0, ph = 0, i = 0;
      for (int R = r0; R < r1; ++R) {
        while (s_rs[a + 1] <= R) ++a;
        const int e = s_e[a];
        int gi;
        const int lr = tk2_map(R - s_rs[a], (s_rs[a + 1] - s_rs[a]) / RPE, blk, RPE, gi);
        const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
        const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
        for (int q = 0; q < NQ; ++q, ++i) {
          if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
          char* sbase = reinterpret_cast<char*>(smem) + st * SB;
          mbar_expect_tx(&full[st], SB);
          bulk_g2s(sbase, src + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 512, src + NRT * 32 * 32 + q * NC * 32, NC * 512, &full[st]);
          bulk_g2s(sbase + NC * 1024, ssrc + q * NC * 32, NC * 128, &full[st]);
          if (++st == NSTG) { st = 0; ph ^= 1; }
        }
      }
    }
    return;
  }
  if (!GATED) asm volatile("griddepcontrol.wait;" ::: "memory");  // inter is gemm13's output
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, rpb = 0, off = 0, n = 0, st = 0, ph = 0, rb = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      // segment = rows of one (expert, token group) in a row: the whole expert, or blk rps when it has > 1 group
      const int ng = (s_rs[a + 1] - s_rs[a]) / RPE;
      int gi;
      const int lr = tk2_map(R - s_rs[a], ng, blk, RPE, gi);
      if (ng == 1 || !blk) {
        rpb = 0;
        seg_start = s_rs[a] + gi * RPE;
        seg_end = seg_start + RPE;
      } else {
        rpb = lr - lr % blk;
        seg_start = R - lr % blk;
        seg_end = seg_start + blk;
      }
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      const bool v = g < n;
      int row = 0;
      if (v) row = GATED ? (s_srt[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
        xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
      }
    }
    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      const uint4 wv0 = sw4[0];
      const uint4 wv1 = sw4[SW];
      const uint32_t sw = reinterpret_cast<const uint32_t*>(sw4 + 2 * SW - warp * 32 - lane)[warp * 32 + lane];
      mbar_arrive(&empty[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
      const uint32_t sc[4] = {slut[sw & 0xFF], slut[(sw >> 8) & 0xFF], slut[(sw >> 16) & 0xFF], slut[sw >> 24]};
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        const uint4 wv = ii ? wv1 : wv0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
          mma16816(acc[ii], af, u4get(xf[q][j >> 1], (j & 1) * 2), u4get(xf[q][j >> 1], (j & 1) * 2 + 1));
        }
      }
    }
    // cross-warp reduce (double-buffered: one consumer barrier per pair)
    float* red = s_red[rb];
    rb ^= 1;
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[((warp * 2 + ii) * 4 + r) * 32 + lane] = acc[ii][r];
    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
    const int rp = rpb + (R - seg_start);
    const int ct = tid, l = ct & 31, gg = l >> 2, tt = l & 3;
    if (GATED) {
      if (ct < 128) {
        const int r = ct >> 5;
        float v = 0.f, u = 0.f;
#pragma unroll
        for (int w = 0; w < NC; ++w) { v += red[((w * 2) * 4 + r) * 32 + l]; u += red[((w * 2 + 1) * 4 + r) * 32 + l]; }
        const int tok = 2 * tt + (r & 1);
        if (tok < n) {
          v *= gsc;
          u *= gsc;
          const int col = rp * 16 + gg + ((r & 2) ? 8 : 0);
          reinterpret_cast<__nv_bfloat16*>(Out)[(size_t)(off + tok) * NOUT + col] =
              __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        }
      }
    } else {
      const int ii = ct >> 7, r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED>
static int launch_tk2(const void* wq, const void* s2, const void* lut, const void* gs, const void* x, const void* ids,
                      int nroutes, int start, int count, void* meta, void* sroutes, void* inv, void* out,
                      cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * NC * 1152;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tk2<NRT, NC, NSTG, GATED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  return launch_k(moe_tk2<NRT, NC, NSTG, GATED>, grid, (NC + 1) * 32, SMEM, st, !GATED && (g_pdl & 1),
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const float*)gs, (const __nv_bfloat16*)x,
      (const int*)ids, nroutes, start, count, (int*)meta, (int*)sroutes, (int*)inv, out, g_blk | ((!GATED && (g_pdl & 1)) << 16));
}

#define SK_DISPATCH(NRT, XR)                                                                 \
  switch (cfg) {                                                                             \
    case 242: return launch_sk<NRT, 4, 2, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 223: return launch_sk<NRT, 2, 3, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 243: return launch_sk<NRT, 4, 3, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 224: return launch_sk<NRT, 2, 4, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 262: return launch_sk<NRT, 6, 2, XR>(wq, s2, lut, x, meta, sr, acc, st);           \
    case 3168: return launch_tk<NRT, 16, 8, XR>(wq, s2, lut, x, meta, sr, acc, st);         \
    case 3166: return launch_tk<NRT, 16, 6, XR>(wq, s2, lut, x, meta, sr, acc, st);         \
    case 3088: return launch_tk<NRT, 8, 8, XR>(wq, s2, lut, x, meta, sr, acc, st);          \
    case 3812: return launch_tk<NRT, 8, 12, XR>(wq, s2, lut, x, meta, sr, acc, st);         \
  }                                                                                          \
  return -1;

// ---- moe_tb: big-M (M > 32) grouped GEMM on a TMA ring, 1 CTA/SM, 8 consumer warps + 1 producer warp.
// Item = (active expert, 64-token pass, row block of 8 warps x RT 16-row tiles), taken by the producer from a
// global counter (meta[66] / meta[67], zeroed by moe_route), heaviest experts first. A stage = one 64-wide k
// chunk: the item's weights (one bulk copy; prepare packs [e][rb][kc][warp][RT][lane]), their scales
// ([e][rb][kc][warp][lane][RT] u16) and the pass's activation rows (one 128 B copy per token, 144 B smem stride).
// Gated: a warp's RT tiles are RT/2 gate tiles + the matching up tiles -> silu(gate) * up in registers.
template <int RT, bool GATED>
struct TBC {
  static constexpr int WB = 8 * RT * 512;
  static constexpr int SB = 8 * RT * 64;
  static constexpr int XB = 64 * 144;
  static constexpr int STG = WB + SB + XB;
  static constexpr int NRB = (GATED ? 2 * NOUT : NOUT) / 16 / (8 * RT);
};

template <int RT, bool GATED>
__device__ __forceinline__ void tb_ldw(const char* sb, int warp, int lane, uint4 (&wv)[RT], uint32_t (&sc)[RT / 2]) {
  using C = TBC<RT, GATED>;
#pragma unroll
  for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb)[(warp * RT + r) * 32 + lane];
  if (RT == 4) {
    const uint2 v = *reinterpret_cast<const uint2*>(sb + C::WB + (warp * 32 + lane) * RT * 2);
    sc[0] = v.x; sc[RT / 2 - 1] = v.y;
  } else {
    sc[0] = *reinterpret_cast<const uint32_t*>(sb + C::WB + (warp * 32 + lane) * RT * 2);
  }
}

template <int RT, int NG>
__device__ __forceinline__ void tb_half(float (&acc)[RT][8][4], const uint4 (&wv)[RT], const uint32_t (&sc)[RT / 2],
                                        const uint4 (&b)[NG], int jp, const uint32_t* slut) {
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);
    const uint32_t s0 = slut[sw & 0xFF], s1 = slut[(sw >> 8) & 0xFF];
    uint32_t af[2][4];
    dq(u4get(wv[r], jp * 2), s0, s1, af[0]);
    dq(u4get(wv[r], jp * 2 + 1), s0, s1, af[1]);
#pragma unroll
    for (int gi = 0; gi < NG; ++gi) {
      mma16816(acc[r][gi], af[0], b[gi].x, b[gi].y);
      mma16816(acc[r][gi], af[1], b[gi].z, b[gi].w);
    }
  }
}

// The caller has waited for full[st]. Software-pipelined: the next stage's weights and scales are fetched
// before the second half of this stage's MMAs.
template <int RT, int NG, int NSTG, bool GATED>
__device__ __forceinline__ void tb_kloop(float (&acc)[RT][8][4], const char* smem, uint64_t* full, uint64_t* empty,
                                         int& st, int& ph, const uint32_t* slut, int warp, int lane, int g, int t) {
  using C = TBC<RT, GATED>;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int gi = 0; gi < 8; ++gi)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][gi][q] = 0.f;
  uint4 wv[RT];
  uint32_t sc[RT / 2];
  tb_ldw<RT, GATED>(smem + st * C::STG, warp, lane, wv, sc);
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const uint4* xs = reinterpret_cast<const uint4*>(smem + st * C::STG + C::WB + C::SB);
    {
      uint4 b[NG];
#pragma unroll
      for (int gi = 0; gi < NG; ++gi) b[gi] = xs[(gi * 8 + g) * 9 + t * 2];
      tb_half<RT, NG>(acc, wv, sc, b, 0, slut);
    }
    uint4 b[NG];
#pragma unroll
    for (int gi = 0; gi < NG; ++gi) b[gi] = xs[(gi * 8 + g) * 9 + t * 2 + 1];
    uint4 wn[RT];
    uint32_t sn[RT / 2];
    const int cst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
    if (kc < 63) {
      mbar_wait(&full[st], ph);
      tb_ldw<RT, GATED>(smem + st * C::STG, warp, lane, wn, sn);
    }
    mbar_arrive(&empty[cst]);
    tb_half<RT, NG>(acc, wv, sc, b, 1, slut);
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = wn[r];
#pragma unroll
    for (int r = 0; r < RT / 2; ++r) sc[r] = sn[r];
  }
}

template <int RT, int NSTG, bool GATED>
__global__ void __launch_bounds__(384, 1) moe_tb(
    const char* __restrict__ Wb, const char* __restrict__ Sb, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __nv_bfloat16* __restrict__ X,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  using C = TBC<RT, GATED>;
  constexpr int NRB = C::NRB;
  extern __shared__ __align__(128) uint4 smem_tb[];
  char* smem = reinterpret_cast<char*>(smem_tb);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[33], s_item[NSTG];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int np = tid < nact ? ((s_off[e + 1] - s_off[e] + 63) >> 6) * NRB : 0;
    int incl = np;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_cum[tid + 1] = incl;
    if (tid == 0) s_cum[0] = 0;
  }
  __syncthreads();
  const int nitems = s_cum[32];
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8) return;
    int* counter = meta + (GATED ? 66 : 67);
    int st = 0, ph = 0, i = 0;
    for (;;) {
      int item = 0;
      if (lane == 0) item = atomicAdd(counter, 1);
      item = __shfl_sync(0xffffffffu, item, 0);
      if (item >= nitems) {
        if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        if (lane == 0) { s_item[st] = -1; mbar_arrive(&full[st]); }
        break;
      }
      int a = 0;
      while (s_cum[a + 1] <= item) ++a;
      const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
      const int e = s_e[a];
      const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
      const char* wsrc = Wb + ((size_t)e * NRB + rb) * 64 * C::WB;
      const char* ssrc = Sb + ((size_t)e * NRB + rb) * 64 * C::SB;
      const char* xsrc[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = lane + 32 * h;
        int src = 0;
        if (row < nv) src = GATED ? (sroutes[off + row] >> 3) : (off + row);
        xsrc[h] = reinterpret_cast<const char*>(X + (size_t)src * KD);
      }
      const int bytes = C::WB + C::SB + nv * 128;
      for (int kc = 0; kc < 64; ++kc, ++i) {
        if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * C::STG;
        if (lane == 0) {
          if (kc == 0) s_item[st] = item;
          mbar_expect_tx(&full[st], bytes);
        }
        __syncwarp();
        if (lane == 0) bulk_g2s(sb, wsrc + (size_t)kc * C::WB, C::WB, &full[st]);
        if (lane == 1) bulk_g2s(sb + C::WB, ssrc + (size_t)kc * C::SB, C::SB, &full[st]);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if (lane + 32 * h < nv) bulk_g2s(sb + C::WB + C::SB + (lane + 32 * h) * 144, xsrc[h] + kc * 128, 128, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    int a = 0;
    while (s_cum[a + 1] <= item) ++a;
    const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
    const int e = s_e[a];
    const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
    const float gsc = gs[e] * 256.0f;
    float acc[RT][8][4];
    switch ((nv + 7) >> 3) {
      case 1: tb_kloop<RT, 1, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 2: tb_kloop<RT, 2, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 3: tb_kloop<RT, 3, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 4: tb_kloop<RT, 4, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 5: tb_kloop<RT, 5, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 6: tb_kloop<RT, 6, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      case 7: tb_kloop<RT, 7, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
      default: tb_kloop<RT, 8, NSTG, GATED>(acc, smem, full, empty, st, ph, slut, warp, lane, g, t); break;
    }
    constexpr int H = GATED ? RT / 2 : RT;
#pragma unroll
    for (int r = 0; r < H; ++r) {
      const int tile = rb * 8 * H + warp * H + r;
#pragma unroll
      for (int gi = 0; gi < 8; ++gi) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tok = gi * 8 + t * 2 + (q & 1);
          if (tok < nv) {
            const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
            const float v = acc[r][gi][q] * gsc;
            if (GATED) {
              const float u = acc[r + H][gi][q] * gsc;
              Out[(size_t)(off + tok) * NOUT + col] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
            } else {
              Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
            }
          }
        }
      }
    }
  }
}

template <int RT, int NSTG, bool GATED>
static int launch_tb(const void* wb, const void* sb, const void* gs, const void* lut, const void* x,
                     const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * TBC<RT, GATED>::STG;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tb<RT, NSTG, GATED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tb<RT, NSTG, GATED><<<grid, 384, SMEM, st>>>(
      (const char*)wb, (const char*)sb, (const float*)gs, (const uint32_t*)lut, (const __nv_bfloat16*)x,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

// ---- moe_tw: big-M (M > 32) grouped GEMM with wgmma (A = dequantized weights from registers, B = activations
// from smem via a TMA tensor map, SW128). Same items / producer / weight packing as moe_tb, but activations come
// from a gathered, K-permuted copy (moe_gather: row pos = route in expert order; within each 64-chunk logical
// k = j*16 + h*8 + 2t + b holds physical k = 16t + 4j + 2h + b, the order _pack_weight uses), and gemm13 writes
// inter in that order too. 2 consumer warpgroups (warp w: RT 16-row tiles), 1 producer warp, 1 CTA/SM.
struct TMaps { CUtensorMap m[6]; };  // box rows 8, 16, 32, 64, 96 (s17: moe_twm hot class), 128 (s29: moe_hot)

// NW = consumer warps: 8 (RT 2, 2 consumer WGs + a producer WG, setmaxnreg) or 16 (RT 1, 4 consumer WGs + a
// producer warp; same weight/scale packing: warp 2w+r of NW 16 = (warp w, tile r) of NW 8; gated: even warps gate,
// odd warps up, joined through smem in the epilogue).
template <int RT, bool GATED, int NW = 8>
struct TWC {
  static constexpr int XB = 64 * 128;
  static constexpr int WB = NW * RT * 512;
  static constexpr int SB = NW * RT * 64;
  static constexpr int STG = XB + WB + SB;
  static constexpr int NRB = (GATED ? 2 * NOUT : NOUT) / 16 / (NW * RT);
  static constexpr int XCH = (NW == 16 && GATED) ? 8 * 32 * 32 * 4 : 0;  // gate/up exchange
  static constexpr int THREADS = NW == 16 ? 16 * 32 + 32 : 384;
  static constexpr int PS = RT == 4 ? 32 : 64;  // tokens per pass
};

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_r(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint64_t sw128_desc(const void* p) {
  return (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 62);
}

//@@WGMMA@@

template <int RT, int N, bool GATED, int NW>
__device__ __forceinline__ void tw_epi(float (&acc)[RT][N / 2], int lane, int warp, int nv, int off, int rb, float gsc,
                                       const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int H = GATED ? RT / 2 : RT;
#pragma unroll
  for (int r = 0; r < H; ++r) {
    const int tile = rb * NW * H + warp * H + r;
#pragma unroll
    for (int gi = 0; gi < N / 8; ++gi) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = gi * 8 + t * 2 + (q & 1);
        if (tok < nv) {
          const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
          const float v = acc[r][gi * 4 + q] * gsc;
          if (GATED) {
            const float u = acc[r + H][gi * 4 + q] * gsc;
            // logical (permuted) position of physical k = col for gemm2's B
            const int c = col & 63;
            const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
            Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
          } else {
            Out[(size_t)sroutes[off + tok] * NOUT + col] = __float2bfloat16(v);
          }
        }
      }
    }
  }
}

// DYN: one main loop for both GEMMs (moe_twf), epilogue picked by g2 at run time
template <int RT, int KG, int N, int NSTG, bool GATED, int NW, bool DYN = false, bool SA = false, int STRIDE = 0,
          int XBO = 0>  // s29: XBO = x tile bytes (weights' offset in the stage) when != the TWC default
__device__ __forceinline__ void tw_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                        const uint32_t* slut, int warp, int lane, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out,
                                        bool g2 = false, int nkc = 64, float* part = nullptr, int* tick = nullptr,
                                        int* s_flag = nullptr, int pofs = 0) {
  using C = TWC<RT, GATED, NW>;
  constexpr int XBv = XBO ? XBO : C::XB;
  constexpr int NA = N / 2;
  float acc[RT][NA];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[KG][RT][4], afB[KG][RT][4];
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * (STRIDE ? STRIDE : C::STG);
    if (kc) mbar_wait(&full[st], ph);
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + XBv)[(warp * RT + r) * 32 + lane];
    uint32_t sc[(RT + 1) / 2];
    if (NW == 16) {
      sc[0] = *reinterpret_cast<const uint16_t*>(sb + XBv + C::WB + (((warp >> 1) * 32 + lane) * 2 + (warp & 1)) * 2);
    } else if (RT == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(sb + XBv + C::WB + (warp * 32 + lane) * RT * 2);
      sc[0] = v.x; sc[RT / 2 - 1] = v.y;
    } else {
      sc[0] = *reinterpret_cast<const uint32_t*>(sb + XBv + C::WB + (warp * 32 + lane) * RT * 2);
    }
    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);
      if (SA) {
        const uint32_t pr = (r & 1) ? scp<1>(sc[r >> 1]) : scp<0>(sc[r >> 1]);
        s0[r] = blo(pr);
        s1[r] = bhi(pr);
      } else {
        s0[r] = slut[sw & 0xFF];
        s1[r] = slut[(sw >> 8) & 0xFF];
      }
    }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; j0 += KG) {
      // ring slot = group parity (KG == 4: stage parity, tracked by kc)
      const bool odd = (KG == 4) ? (kc & 1) : ((j0 / KG) & 1);
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (odd) dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afB[jj][r]);
          else dq(u4get(wv[r], j0 + jj), s0[r], s1[r], afA[jj][r]);
        }
      wg_fence();
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r) wgmma_bf16<N>(acc[r], odd ? afB[jj][r] : afA[jj][r], d0 + (j0 + jj) * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int jj = 0; jj < KG; ++jj)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (odd) fence_r(afA[jj][r][i]);
            else fence_r(afB[jj][r][i]);
          }
      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);
    }
    pst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
#pragma unroll
  for (int r = 0; r < RT; ++r) {
#pragma unroll
    for (int jj = 0; jj < KG; ++jj)
#pragma unroll
      for (int i = 0; i < 4; ++i) { fence_r(afA[jj][r][i]); fence_r(afB[jj][r][i]); }
#pragma unroll
    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
  const int g = lane >> 2, t = lane & 3;
  if (NW == 16 && GATED) {
    float* xb = reinterpret_cast<float*>(const_cast<char*>(smem) + NSTG * C::STG) + (warp >> 1) * 32 * 32;
    if (warp & 1) {
#pragma unroll
      for (int i = 0; i < NA; ++i) xb[i * 32 + lane] = acc[0][i];
    }
    asm volatile("bar.sync %0, 64;" ::"r"(1 + (warp >> 1)) : "memory");
    if (!(warp & 1)) {
      const int tile = rb * 8 + (warp >> 1);
#pragma unroll
      for (int gi = 0; gi < N / 8; ++gi)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tok = gi * 8 + t * 2 + (q & 1);
          if (tok < nv) {
            const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
            const float v = acc[0][gi * 4 + q] * gsc;
            const float u = xb[(gi * 4 + q) * 32 + lane] * gsc;
            const int c = col & 63;
            const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
            Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
          }
        }
    }
    asm volatile("bar.sync %0, 64;" ::"r"(1 + (warp >> 1)) : "memory");
    return;
  }
  if constexpr (DYN) {
    if (part) {  // s11 K-split tail item: the first half to finish stores its FP32 partial, the second adds it
      const int ct = warp * 32 + lane;
      asm volatile("bar.sync 1, 256;" ::: "memory");  // s_flag is free (previous split item done with it)
      if (ct == 0) *s_flag = atomicAdd(tick, 1);
      asm volatile("bar.sync 1, 256;" ::: "memory");
      const int o = *s_flag;
      if (!o) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int i = 0; i < NA; ++i) __stcg(part + (r * NA + i) * 256 + ct, acc[r][i]);
        __threadfence();
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (ct == 0) asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(tick + 512), "r"(1) : "memory");
        return;
      }
      if (ct == 0) {  // the other half took its ticket first: wait for its partial
        int v;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(tick + 512) : "memory");
        } while (!v);
        tick[0] = 0;  // last user of both words: zero for the next call
        tick[512] = 0;
        __threadfence();
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      const float* ot = part + pofs;
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int i = 0; i < NA; ++i) acc[r][i] += __ldcg(ot + (r * NA + i) * 256 + ct);
    }
    if (g2) tw_epi<RT, N, false, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
    else tw_epi<RT, N, true, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
  } else {
    tw_epi<RT, N, GATED, NW>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
  }
}

template <int RT, int KG, int NSTG, bool GATED, int NW>
__global__ void __launch_bounds__(TWC<RT, GATED, NW>::THREADS, 1) moe_tw(
    const char* __restrict__ Wb, const char* __restrict__ Sb, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out, int pfk) {
  using C = TWC<RT, GATED, NW>;
  constexpr int NRB = C::NRB;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[33], s_item[NSTG];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += C::THREADS) slut[i] = lut[i];
  // PDL: gemm13 waits for moe_gather (and so moe_route) before reading meta, then lets gemm2 launch; gemm2 reads
  // meta (complete by then) and only waits (producer) before its first copy of gemm13's output
  if (GATED) asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NW * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int np = tid < nact ? ((s_off[e + 1] - s_off[e] + C::PS - 1) / C::PS) * NRB : 0;
    int incl = np;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_cum[tid + 1] = incl;
    if (tid == 0) s_cum[0] = 0;
  }
  __syncthreads();
  const int nitems = s_cum[32];
  if (warp >= NW) {
    if (NW == 8) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > NW || lane != 0) return;
    int* counter = meta + (GATED ? 66 : 67);
    int st = 0, ph = 0, i = 0;
    int item = atomicAdd(counter, 1);
    if (!GATED) {
      if (item < nitems && pfk) {  // L2-prefetch the first item's first pfk k chunks while gemm13 finishes
        int a = 0;
        while (s_cum[a + 1] <= item) ++a;
        const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
        const int e = s_e[a];
        const char* wsrc = Wb + ((size_t)e * NRB + rb) * 64 * C::WB;
        const char* ssrc = Sb + ((size_t)e * NRB + rb) * 64 * C::SB;
        prefetch_l2(wsrc, pfk * C::WB);
        prefetch_l2(ssrc, pfk * C::SB);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    for (;; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int a = 0;
      while (s_cum[a + 1] <= item) ++a;
      const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
      const int e = s_e[a];
      const int off = s_off[e] + p * C::PS, nv = min(C::PS, s_off[e + 1] - off);
      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const char* wsrc = Wb + ((size_t)e * NRB + rb) * 64 * C::WB;
      const char* ssrc = Sb + ((size_t)e * NRB + rb) * 64 * C::SB;
      const int bytes = C::WB + C::SB + (1024 << mi);
      for (int kc = 0; kc < 64; ++kc, ++i) {
        if (kc && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * C::STG;
        if (kc == 0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, &maps.m[mi], kc * 64, off, &full[st]);
        bulk_g2s(sb + C::XB, wsrc + (size_t)kc * C::WB, C::WB, &full[st]);
        bulk_g2s(sb + C::XB + C::WB, ssrc + (size_t)kc * C::SB, C::SB, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  if (NW == 8) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    int a = 0;
    while (s_cum[a + 1] <= item) ++a;
    const int rem = item - s_cum[a], p = rem / NRB, rb = rem - p * NRB;
    const int e = s_e[a];
    const int off = s_off[e] + p * C::PS, nv = min(C::PS, s_off[e + 1] - off);
    const float gsc = gs[e] * 256.0f;
#define TWI(NN) tw_item<RT, KG, NN, NSTG, GATED, NW>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, Out)
    if (C::PS == 32) {  // N <= 32 only: ptxas sizes the wgmma pipeline for the largest variant in the function
    switch ((nv + 7) >> 3) {
      case 1: TWI(8); break;
      case 2: TWI(16); break;
      case 3: TWI(24); break;
      default: TWI(32); break;
    }
    } else {
    switch ((nv + 7) >> 3) {
      case 1: TWI(8); break;
      case 2: TWI(16); break;
      case 3: TWI(24); break;
      case 4: TWI(32); break;
      case 5: TWI(40); break;
      case 6: TWI(48); break;
      case 7: TWI(56); break;
      default: TWI(64); break;
    }
    }
#undef TWI
  }
}

template <int RT, int KG, int NSTG, bool GATED, int NW = 8>
static int launch_tw(const void* wb, const void* sb, const void* gs, const void* lut, const TMaps& maps,
                     const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  using C = TWC<RT, GATED, NW>;
  constexpr int SMEM = NSTG * C::STG + C::XCH + 1024;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tw<RT, KG, NSTG, GATED, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  return launch_k(moe_tw<RT, KG, NSTG, GATED, NW>, grid, C::THREADS, SMEM, st, (g_pdl & 4) != 0,
      (const char*)wb, (const char*)sb, (const float*)gs, (const uint32_t*)lut, maps,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out, g_twpf);
}

// xg[pos] = x[sroutes[pos] >> 3] with each 64-chunk K-permuted (logical word (j*2+h)*4+t <- physical word 8t+2j+h).
__global__ void __launch_bounds__(256) moe_gather(const uint32_t* __restrict__ X, const int* __restrict__ meta,
                                                  const int* __restrict__ sroutes, uint32_t* __restrict__ XG) {
  const int pos = blockIdx.x;
  asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL secondary of moe_route
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (pos >= meta[65]) return;
  const uint32_t* src = X + (size_t)(sroutes[pos] >> 3) * (KD / 2);
  uint32_t* dst = XG + (size_t)pos * (KD / 2);
#pragma unroll
  for (int i = 0; i < KD / 2 / 256; ++i) {
    const int o = threadIdx.x + 256 * i, w = o & 31, t = w & 3, jh = w >> 2;
    dst[o] = src[(o & ~31) | (8 * t + jh)];
  }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define PTR(t) ((const void*)(t).data_ptr())
int64_t moe_gemm13(int64_t cfg, torch::Tensor wq_, torch::Tensor ws_, torch::Tensor gs_, torch::Tensor lut_,
                   torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *ws = PTR(ws_), *gs = PTR(gs_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_),
             *sr = PTR(sr_);
  void* out = out_.data_ptr();
  DISPATCH(true)
}
int64_t moe_gemm2(int64_t cfg, torch::Tensor wq_, torch::Tensor ws_, torch::Tensor gs_, torch::Tensor lut_,
                  torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *ws = PTR(ws_), *gs = PTR(gs_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_),
             *sr = PTR(sr_);
  void* out = out_.data_ptr();
  DISPATCH(false)
}
int64_t moe_route_launch(torch::Tensor ids, int64_t start, int64_t count, torch::Tensor meta, torch::Tensor sroutes) {
  moe_route<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const int*)ids.data_ptr(), (int)ids.numel(), (int)start,
                                                               (int)count, (int*)meta.data_ptr(), (int*)sroutes.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_norm_launch(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start,
                        int64_t count, double eps, torch::Tensor out) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const __nv_bfloat16*)y.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (__nv_bfloat16*)out.data_ptr(), (const int*)nullptr);
}
// n4s1: moe_norm after moe_twm (+ moe_hot) built with NORM_EARLY 1: per-token readiness from meta[100 + e]
int64_t moe_norme_launch(torch::Tensor y, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start,
                         int64_t count, double eps, torch::Tensor out, torch::Tensor meta) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const __nv_bfloat16*)y.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (__nv_bfloat16*)out.data_ptr(),
      (const int*)meta.data_ptr() + 100);
}
int64_t moe_tb_launch(int64_t gated, int64_t cfg, torch::Tensor wb_, torch::Tensor sb_, torch::Tensor gs_,
                      torch::Tensor lut_, torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wb = PTR(wb_), *sb = PTR(sb_), *gs = PTR(gs_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_),
             *sr = PTR(sr_);
  void* out = out_.data_ptr();
#define TB(RT, NS)                                                                       \
  if (cfg == RT * 10 + NS) {                                                             \
    if (gated) return launch_tb<RT, NS, true>(wb, sb, gs, lut, x, meta, sr, out, st);    \
    return launch_tb<RT, NS, false>(wb, sb, gs, lut, x, meta, sr, out, st);              \
  }
  TB(4, 8) TB(2, 8)
  return -1;
}
static PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
// 4 tensor maps (box rows 8/16/32/64, box cols 64, SW128) over a bf16 [rows, KD] buffer. Encoded at every launch
// from the buffer actually passed (under graph capture they are baked into the kernel node with that buffer).
static void tw_maps(TMaps& maps, void* ptr, int64_t rows) {
  if (!g_encode) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_encode, 12000, cudaEnableDefault, &q);
  }
  for (int i = 0; i < 6; ++i) {
    cuuint64_t dims[2] = {(cuuint64_t)KD, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)KD * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)(i < 4 ? 8 << i : i == 4 ? 96 : 128)};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g_encode(&maps.m[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  }
}
int64_t moe_gather_launch(torch::Tensor x, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor xg) {
  return launch_k(moe_gather, (int)sroutes.numel(), 256, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const uint32_t*)x.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(), (uint32_t*)xg.data_ptr());
}
int64_t moe_tw_launch(int64_t gated, int64_t cfg, torch::Tensor wb_, torch::Tensor sb_, torch::Tensor gs_,
                      torch::Tensor lut_, torch::Tensor xb_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wb = PTR(wb_), *sb = PTR(sb_), *gs = PTR(gs_), *lut = PTR(lut_), *meta = PTR(meta_), *sr = PTR(sr_);
  void* out = out_.data_ptr();
  TMaps maps;
  tw_maps(maps, xb_.data_ptr(), xb_.size(0));
#define TW(RT, KG, NS)                                                                       \
  if (cfg == RT * 100 + KG * 10 + NS) {                                                      \
    if (gated) return launch_tw<RT, KG, NS, true>(wb, sb, gs, lut, maps, meta, sr, out, st);  \
    return launch_tw<RT, KG, NS, false>(wb, sb, gs, lut, maps, meta, sr, out, st);            \
  }
  TW(2, 4, 8) TW(2, 2, 8) TW(2, 1, 8) TW(4, 1, 8)
  if (cfg == 1118) {  // NW 16, RT 1 (uses the RT 2 packing)
    if (gated) return launch_tw<1, 1, 8, true, 16>(wb, sb, gs, lut, maps, meta, sr, out, st);
    return launch_tw<1, 1, 8, false, 16>(wb, sb, gs, lut, maps, meta, sr, out, st);
  }
  return -1;
}
// ---- moe_twf: gemm13 and gemm2 of the big-M path in ONE persistent launch (RT 4, 32-token passes). The work
// counter hands out all gemm13 items (heaviest experts first), then all gemm2 items in the same expert order. A
// gemm13 item's consumers bump done[e] after writing their inter rows; a gemm2 item's producer waits for done[e] ==
// (gemm13 items of e) before its first copy of inter. All gemm13 items are handed out before any gemm2 item and
// never wait, and the grid is 1 CTA per SM (all resident), so this cannot deadlock. CTAs that finish gemm13 early
// start on gemm2 instead of idling through gemm13's tail.
#ifndef NEEDK
#define NEEDK 1
#endif
template <int RT, int NSTG, bool SA = false>
__global__ void __launch_bounds__(384, 1) moe_twf(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit) {
  using CG = TWC<RT, true, 8>;
  using CN = TWC<RT, false, 8>;
  static_assert(CG::STG == CN::STG, "stage layout");
  constexpr int PS = CG::PS;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_flag;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    const int npass = tid < nact ? (s_off[e + 1] - s_off[e] + PS - 1) / PS : 0;
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      int incl = npass * (gq ? CN::NRB : CG::NRB);
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      s_cum[gq][tid + 1] = incl;
      if (tid == 0) s_cum[gq][0] = 0;
    }
  }
  __syncthreads();
  // s11: the last ns gemm2 items (the lightest: the tail) run as 2 K halves each (items n13 + n2f + 2 j + h)
  // The split starts at an expert boundary (moe_route orders an expert's tokens nondeterministically: a token's
  // result must not depend on whether its pass is split), so ns may come out a bit below nsplit.
  const int n13 = s_cum[0][32], n2 = s_cum[1][32];
  int n2f = n2;
  if (nsplit > 0)
    for (int a = 32; a >= 0; --a)
      if (s_cum[1][a] >= n2 - nsplit) n2f = s_cum[1][a];
  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
  // item -> (gemm2?, active slot a, pass p, row block rb, K half h (-1: whole K), split slot j)
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb, int& h, int& j) {
    g2 = item >= n13;
    h = -1;
    j = 0;
    int it = g2 ? item - n13 : item;
    if (g2 && it >= n2f) {
      j = (it - n2f) >> 1;
      h = (it - n2f) & 1;
      it = n2f + j;
    }
    const int* cum = s_cum[g2];
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = g2 ? CN::NRB : CG::NRB;
    const int rem = it - cum[a];
    p = rem / nrb;
    rb = rem - p * nrb;
  };
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8 || lane != 0) return;
    int* counter = meta + 66;
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1);; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int g2, a, p, rb, h, j;
      decode(item, g2, a, p, rb, h, j);
      const int e = s_e[a];
      const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
      const int k0 = h < 0 ? 0 : h * 32, k1 = h < 0 ? 64 : k0 + 32;
      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const int nrb = g2 ? CN::NRB : CG::NRB;
      const char* wsrc = (g2 ? W2 : W13) + ((size_t)e * nrb + rb) * 64 * CG::WB;
      const char* ssrc = (g2 ? S2 : S13) + ((size_t)e * nrb + rb) * 64 * CG::SB;
      const CUtensorMap* map = g2 ? &maps2.m[mi] : &maps13.m[mi];
      const int bytes = CG::WB + CG::SB + (1024 << mi);
      if (g2) {
        const int need = ((s_off[e + 1] - s_off[e] + PS - 1) / PS) * CG::NRB;
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= need * NEEDK) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * CG::STG;
        if (kc == k0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s(sb + CG::XB, wsrc + (size_t)kc * CG::WB, CG::WB, &full[st]);
        bulk_g2s(sb + CG::XB + CG::WB, ssrc + (size_t)kc * CG::SB, CG::SB, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);
    const int e = s_e[a];
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    float* pt = h < 0 ? nullptr : part + (size_t)(2 * j + h) * 16384;
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
#define TWI(NN) tw_item<RT, 1, NN, NSTG, true, 8, true, SA>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \
                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384)
    if constexpr (RT == 4) {
      switch ((nv + 7) >> 3) {
        case 1: TWI(8); break;
        case 2: TWI(16); break;
        case 3: TWI(24); break;
        default: TWI(32); break;
      }
    } else {  // s11 RT 2: 64-token passes (each dequantized weight serves up to 64 tokens)
      switch ((nv + 7) >> 3) {
        case 1: TWI(8); break;
        case 2: TWI(16); break;
        case 3: case 4: TWI(32); break;
        default: TWI(64); break;
      }
    }
#undef TWI
    if (!g2) {
      // release this item's inter rows to gemm2 producers on other SMs (their TMA reads them: async proxy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
  }
}

template <bool SA, int RT>
static int64_t twf_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_,
                       torch::Tensor s2_, torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_,
                       torch::Tensor sr_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_,
                       int64_t ns) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<RT, true, 8>::STG + 1024;
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_twf<RT, NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_twf<RT, NSTG, SA>, grid, 384, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(), (int*)tk_.data_ptr(),
      (float*)part_.data_ptr(), (int)std::min<int64_t>(ns, std::min<int64_t>(tk_.numel() / 2, part_.numel() / 32768)));
}
int64_t moe_twf_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_,
                       torch::Tensor s2_, torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_,
                       torch::Tensor sr_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_,
                       int64_t ns) {
#define TWFA w13_, s13_, gs13_, w2_, s2_, gs2_, lut_, xg_, meta_, sr_, inter_, y_, tk_, part_, ns
  if (g_sa & 2) return (g_sa & 1) ? twf_go<true, 2>(TWFA) : twf_go<false, 2>(TWFA);  // s11: RT 2, 64-token passes
  return (g_sa & 1) ? twf_go<true, 4>(TWFA) : twf_go<false, 4>(TWFA);
#undef TWFA
}
// s11 moe_twm = moe_twf with the item shape chosen PER EXPERT: experts with > 32 tokens run RT 2 items with 64-token
// passes (each dequantized weight serves 64 tokens; packing w*b2), the others RT 4 items with 32-token passes (w*b4).
// Stage stride = the RT 4 stage (the RT 2 stage is smaller and uses the same x / weight / scale offsets scheme).
#ifndef TWM_TL
#define TWM_TL 0
#endif
#ifndef HOT_EMU
#define HOT_EMU 0  // s29: H200-proxy emulation in moe_twm/moe_hot (tests only; see moe_twm)
#endif
#ifndef TWQ_PREG
#define TWQ_PREG 32  // s17 moe_twq: producer warpgroup / consumer registers after setmaxnreg (640 threads: 96 at launch)
#define TWQ_CREG 112
#endif
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
// s17 moe_twm hot class (cl 3, -nsplit bit 23 = TWM_HOT): an expert with 65..96 tokens runs ONE pass (instead of two
// 64-token RT 2 passes = two dequantizations of every weight) with one 16-row tile per warp and N <= 96 (48
// accumulator registers: within twm's budget, unlike RT 2 x N 96). Item = half an RT 2 row block of the w*b2 packing
// (slots w8 4h..4h+3 are contiguous: 4 KB of weights + 512 B of scales per k chunk); stage = x (96-row box) at 0,
// weights at 12 KB, scales at 16 KB. Gated: warp 2i = gate tile, 2i+1 = its up tile; the up warp hands its FP32
// accumulators to the gate warp through the item's LAST ring slot (24 KB <= one stage; after a consumer barrier:
// the other warpgroup may still be reading that slot), released only afterwards.
template <int N, int NSTG, bool SA, int STRIDE>
__device__ __forceinline__ void tw_hot(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                       const uint32_t* slut, int warp, int lane, int nv, int off, int rb2, float gsc,
                                       const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out, bool g2) {
  constexpr int NA = N / 2, XB = 96 * 128, WB = 4096;
  float acc[1][NA];
#pragma unroll
  for (int i = 0; i < NA; ++i) acc[0][i] = 0.f;
  uint32_t afA[4], afB[4];
  const int so = XB + WB + ((warp >> 1) * 32 + lane) * 4 + (warp & 1) * 2;
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const char* sb = smem + st * STRIDE;
    if (kc) mbar_wait(&full[st], ph);
    const uint4 wv = reinterpret_cast<const uint4*>(sb + XB)[warp * 32 + lane];
    const uint32_t word = *reinterpret_cast<const uint16_t*>(sb + so);
    uint32_t s0, s1;
    if (SA) {
      const uint32_t pr = scp<0>(word); s0 = blo(pr); s1 = bhi(pr);
    } else {
      s0 = slut[word & 0xFF]; s1 = slut[word >> 8];
    }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const bool odd = j0 & 1;
      if (odd) dq(u4get(wv, j0), s0, s1, afB);
      else dq(u4get(wv, j0), s0, s1, afA);
      wg_fence();
      wgmma_bf16<N>(acc[0], odd ? afB : afA, d0 + j0 * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (odd) fence_r(afA[i]);
        else fence_r(afB[i]);
      }
      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);
    }
    pst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 4; ++i) { fence_r(afA[i]); fence_r(afB[i]); }
#pragma unroll
  for (int i = 0; i < NA; ++i) fence_f(acc[0][i]);
  if (g2) {
    mbar_arrive(&empty[pst]);
    tw_epi<1, N, false, 8>(acc, lane, warp, nv, off, rb2, gsc, sroutes, Out);
    return;
  }
  float* x = reinterpret_cast<float*>(const_cast<char*>(smem) + pst * STRIDE) + (warp >> 1) * 32 * NA;
  asm volatile("bar.sync 1, 256;" ::: "memory");  // the other warpgroup's MMAs may still read this slot's x tile
  if (warp & 1) {
#pragma unroll
    for (int i = 0; i < NA; ++i) x[i * 32 + lane] = acc[0][i];
  }
  asm volatile("bar.sync %0, 64;" ::"r"(2 + (warp >> 1)) : "memory");
  if (!(warp & 1)) {
    const int g = lane >> 2, t = lane & 3, tile = rb2 * 4 + (warp >> 1);
#pragma unroll
    for (int gi = 0; gi < N / 8; ++gi)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int tok = gi * 8 + t * 2 + (q & 1);
        if (tok < nv) {
          const int col = tile * 16 + g + ((q & 2) ? 8 : 0);
          const float v = acc[0][gi * 4 + q] * gsc;
          const float u = x[(gi * 4 + q) * 32 + lane] * gsc;
          const int c = col & 63;
          const int lc = (col & ~63) | (((c >> 2) & 3) << 4) | (((c >> 1) & 1) << 3) | ((c >> 4) << 1) | (c & 1);
          Out[(size_t)(off + tok) * NOUT + lc] = __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        }
      }
  }
  asm volatile("bar.sync %0, 64;" ::"r"(2 + (warp >> 1)) : "memory");
  mbar_arrive(&empty[pst]);  // the slot held the exchange: release it only now
}

template <int NSTG, bool SA = false>
__global__ void __launch_bounds__(384, 1) moe_twm(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const char* __restrict__ W13h, const char* __restrict__ S13h, const char* __restrict__ W2h,
    const char* __restrict__ S2h,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, int* __restrict__ tick, float* __restrict__ part, int nsplit, int hx = 0) {
  // s29 hx: moe_hot (the PDL primary) runs the experts with 65..128 tokens: skip them here, do not wait for the
  // primary at entry (it runs concurrently; moe_gather completed before moe_hot's own wait returned), wait at the end
  // (so that moe_norm, this kernel's PDL secondary, sees moe_hot's Y)
  using CG = TWC<4, true, 8>;
  using CN = TWC<4, false, 8>;
  using HG = TWC<2, true, 8>;
  using HN = TWC<2, false, 8>;
  static_assert(CG::STG == CN::STG && HG::STG <= CG::STG && HG::XB == CG::XB, "stage layout");
  constexpr int STRIDE = CG::STG;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_flag, s_big[2][32], s_ord[4], s_cost;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  // s14: nsplit < 0 = no K split; instead the lightest (last) non-big experts whose RT 4 gemm2 items add up to
  // <= -nsplit run their gemm2 at RT 2 with 32-token passes (class 2: half-size items, shorter drain)
  // (-nsplit) >> 12 = the same for gemm13 (RT 4 gemm13 items of the lightest experts)
  // (-nsplit) >> 24 = item order (s14): bit 0 alternating heavy / light experts, bit 1 timeline stamps, bit 2
  // pass-minor items (rb-major: all passes of a row block adjacent: L2 reuse), bits 3-4 = merge the big (> 32 token)
  // experts' items evenly into the first W = phi * N positions of each phase (phi 1, .85, .7), light ones between;
  // bit 5: merge in gemm13 only; bit 6: merged set = multi-pass (> 64 token) experts only
  // s19: bit 22 set = bits 12-21 are a K-split count (TWF_SPLIT, combinable with order / hot; no tail classes then)
  const int sx = nsplit < 0 ? ((-nsplit) >> 22) & 1 : 0;
  const int tail = nsplit < 0 && !sx ? (-nsplit) & 4095 : 0, tail13 = nsplit < 0 && !sx ? ((-nsplit) >> 12) & 1023 : 0;
  const int order = nsplit < 0 ? (-nsplit) >> 24 : 0, hot = nsplit < 0 ? ((-nsplit) >> 23) & 1 : 0;
  nsplit = nsplit < 0 ? (sx ? ((-nsplit) >> 12) & 1023 : 0) : nsplit;
  // s20: split count 1022 / 1023 = AUTO: K-split the last r light gemm2 items, r = (total item cost mod gridDim) when
  // that last wave is at most half full (halves then fill it; else no split). 1022: every item costs 1; 1023: light
  // RT 4 items weighted by their pass size (N8 1, N16 1.25, N32 1.5). Shape/routing-derived on device, no host sync.
  const int autos = nsplit >= 1022 ? nsplit - 1021 : 0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  if (!hx) asm volatile("griddepcontrol.wait;" ::: "memory");
#if HOT_EMU
  // s29 H200-proxy only (HOT_EMU = moe_hot's grid): the last HOT_EMU CTAs start only as moe_hot CTAs exit, like the
  // twm CTAs queued behind moe_hot on a full device
  if (hx && tid == 0 && (int)blockIdx.x >= (int)gridDim.x - HOT_EMU) {
    const int k = blockIdx.x - (gridDim.x - HOT_EMU);
    int v;
    do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(meta + 67) : "memory"); } while ((v >> 20) <= k);
  }
#endif
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int sl = (order & 1) ? ((tid & 1) ? nact - 1 - (tid >> 1) : tid >> 1) : tid;
    const int e = tid < nact ? meta[1 + sl] : 0;
    s_e[tid] = e;
    int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    if (hx && nt > 64 && nt <= 128) nt = 0;
    const int big = nt > 32;
    const int hc = hot && nt > 64 && nt <= 96;  // s17 hot class: one pass, 1 tile per warp
    const int npass = hc ? 1 : big ? (nt + 63) / 64 : (nt + 31) / 32;
    int c2 = hc ? 3 : big, c13 = hc ? 3 : big;
    if (tail | tail13) {
      int sfx = big ? 1 << 20 : npass;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_down_sync(0xffffffffu, sfx, o);
        if (tid + o < 32) sfx += v;
      }
      if (!big && nt > 0 && sfx * CN::NRB <= tail) c2 = 2;
      if (!big && nt > 0 && sfx * CG::NRB <= tail13) c13 = 2;
    }
    s_big[0][tid] = c13;
    s_big[1][tid] = c2;
    if (autos) {
      const int w = (autos == 2 && !c2 && nt > 8) ? (nt > 16 ? 6 : 5) : 4;
      const int n13i = npass * (c13 ? HG::NRB * (c13 == 3 ? 2 : 1) : CG::NRB), n2i = npass * (c2 ? HN::NRB * (c2 == 3 ? 2 : 1) : CN::NRB);
      int cst = tid < nact ? w * (n13i + n2i) : 0;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) cst += __shfl_xor_sync(0xffffffffu, cst, o);
      if (tid == 0) s_cost = cst;
    }
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      const int c = gq ? c2 : c13;
      int incl = npass * (c ? (gq ? HN::NRB : HG::NRB) * (c == 3 ? 2 : 1) : (gq ? CN::NRB : CG::NRB));
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      s_cum[gq][tid + 1] = incl;
      if (tid == 0) s_cum[gq][0] = 0;
    }
  }
  __syncthreads();
  // s11: the last ns gemm2 items (the lightest: the tail) run as 2 K halves each (items n13 + n2f + 2 j + h)
  // The split starts at an expert boundary (moe_route orders an expert's tokens nondeterministically: a token's
  // result must not depend on whether its pass is split), so ns may come out a bit below nsplit.
  const int n13 = s_cum[0][32], n2 = s_cum[1][32];
  int n2f = n2;
  if (autos) {
    const int GU = 4 * gridDim.x, r = s_cost % GU;
    nsplit = (r > 0 && 2 * r <= GU) ? min((r + 3) / 4, 66) : 0;
  }
  if (nsplit > 0)
    for (int a = 32; a >= 0; --a) {
      if (a < 32 && s_big[1][a] != 0) break;  // s19: only light (RT 4) experts split (hot/big items take no K half)
      if (s_cum[1][a] >= n2 - nsplit) n2f = s_cum[1][a];
    }
  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
#if NORM_EARLY
  __shared__ int s_nd2[32];
  if (tid < 32) {
    const int c0 = s_cum[1][tid], c1 = s_cum[1][tid + 1];
    s_nd2[tid] = (c1 - c0) + max(0, c1 - max(c0, n2f));  // gemm2 item ends of slot tid (K halves count twice)
  }
#endif
  if (tid == 0) {
    int nbig = 0;
    while (nbig < 32 && s_big[0][nbig] == 1 && (!(order & 64) || s_cum[0][nbig + 1] - s_cum[0][nbig] > HG::NRB)) ++nbig;
    const int mrg = (order >> 3) & 3;
    s_ord[0] = nbig;
    s_ord[1] = mrg == 0 ? 0 : mrg == 1 ? 100 : mrg == 2 ? 85 : 70;
    s_ord[2] = order & 36;
  }
  __syncthreads();
  // item -> (gemm2?, active slot a, pass p, row block rb, K half h (-1: whole K), split slot j)
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb, int& h, int& j) {
    g2 = item >= n13;
    h = -1;
    j = 0;
    int it = g2 ? item - n13 : item;
    if (g2 && it >= n2f) {
      j = (it - n2f) >> 1;
      h = (it - n2f) & 1;
      it = n2f + j;
    }
    const int* cum = s_cum[g2];
    const int phx = s_ord[1], ob = s_ord[2];
    if (phx && h < 0 && !(g2 && (ob & 32))) {
      const int nA = cum[s_ord[0]], N = g2 ? n2f : cum[32];  // s19: K-split items stay at the end
      if (nA > 0 && nA < N) {
        const int W = max(nA, N * phx / 100);
        const int ca = it < W ? it * nA / W : nA, ca1 = it + 1 <= W ? (it + 1) * nA / W : nA;
        it = ca1 > ca ? ca : nA + it - ca;
      }
    }
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = s_big[g2][a] ? (g2 ? HN::NRB : HG::NRB) * (s_big[g2][a] == 3 ? 2 : 1) : (g2 ? CN::NRB : CG::NRB);
    const int rem = it - cum[a];
    if (ob & 4) {
      const int np = (cum[a + 1] - cum[a]) / nrb;
      rb = rem / np;
      p = rem - rb * np;
    } else {
      p = rem / nrb;
      rb = rem - p * nrb;
    }
  };
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8 || lane != 0) return;
    int* counter = meta + 66;
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1);; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int g2, a, p, rb, h, j;
      decode(item, g2, a, p, rb, h, j);
      const int e = s_e[a], cl = s_big[g2][a], big = cl != 0, PS = cl == 3 ? 128 : cl == 1 ? 64 : 32;
      const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
      const int k0 = h < 0 ? 0 : h * 32, k1 = h < 0 ? 64 : k0 + 32;
      const int mi = cl == 3 ? 4 : nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const int nrb = big ? (g2 ? HN::NRB : HG::NRB) : (g2 ? CN::NRB : CG::NRB);
      const int WB = big ? HG::WB : CG::WB, SB = big ? HG::SB : CG::SB;
      // hot class: item rb = 2 * (RT 2 row block) + half; the half's slots are the 2nd half of each k chunk's block
      const int rbp = cl == 3 ? rb >> 1 : rb, hh = cl == 3 ? rb & 1 : 0;
      const char* wsrc = (big ? (g2 ? W2h : W13h) : (g2 ? W2 : W13)) + ((size_t)e * nrb + rbp) * 64 * WB + hh * (WB / 2);
      const char* ssrc = (big ? (g2 ? S2h : S13h) : (g2 ? S2 : S13)) + ((size_t)e * nrb + rbp) * 64 * SB + hh * (SB / 2);
      const CUtensorMap* map = g2 ? &maps2.m[mi] : &maps13.m[mi];
      const int xbytes = cl == 3 ? 96 * 128 : 1024 << mi, wbytes = cl == 3 ? WB / 2 : WB, sbytes = cl == 3 ? SB / 2 : SB;
      const int xo = cl == 3 ? 96 * 128 : CG::XB;  // weights' offset in the stage
      const int bytes = wbytes + sbytes + xbytes;
      if (g2) {
        const int b13 = s_big[0][a], PS13 = b13 == 3 ? 128 : b13 == 1 ? 64 : 32;
        const int need = ((s_off[e + 1] - s_off[e] + PS13 - 1) / PS13) * (b13 ? HG::NRB * (b13 == 3 ? 2 : 1) : CG::NRB);
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= need * NEEDK) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (kc == k0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
        {  // moe s21 (12d95a9): evict_last if a later pass of this expert re-reads these weights, else evict_first
          uint64_t pol;
          if (off + PS < s_off[e + 1]) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
          else asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
          bulk_g2s_h(sb + xo, wsrc + (size_t)kc * WB, wbytes, &full[st], pol);
          bulk_g2s_h(sb + xo + wbytes, ssrc + (size_t)kc * SB, sbytes, &full[st], pol);
        }
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
#if TWM_TL
  // s14 timeline (build with TWM_TL 1, order & 2): per CTA 64 u64 in part: [0] start, [1] items, 3 per item
  // (start, end, item), [63] exit
  unsigned long long* tl = (order & 2) ? reinterpret_cast<unsigned long long*>(part) + blockIdx.x * 64 : nullptr;
  int ntl = 0;
  if (tl && tid == 0) tl[0] = gtimer();
#endif
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
#if TWM_TL
    if (tl && tid == 0) {
      if (item < 0) { tl[1] = ntl; tl[63] = gtimer(); }
      else if (ntl < 20) { tl[2 + 3 * ntl] = gtimer(); tl[4 + 3 * ntl] = item; }
    }
#endif
    if (item < 0) break;
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);
    const int e = s_e[a], cl = s_big[g2][a], big = cl != 0, PS = cl == 3 ? 128 : cl == 1 ? 64 : 32;
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    float* pt = h < 0 ? nullptr : part + (size_t)(2 * j + h) * 16384;
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
#define TWI(RT, NN) tw_item<RT, 1, NN, NSTG, true, 8, true, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \
                                                        h < 0 ? 64 : 32, pt, tick + j, &s_flag, h ? -16384 : 16384)
    if (cl == 3) {
      if (nv <= 80) tw_hot<80, NSTG, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2);
      else tw_hot<96, NSTG, SA, STRIDE>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2);
    } else if (!big) {
      switch ((nv + 7) >> 3) {
        case 1: TWI(4, 8); break;
        case 2: TWI(4, 16); break;
        case 3: TWI(4, 24); break;
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
#undef TWI
    if (!g2) {
      // release this item's inter rows to gemm2 producers on other SMs (their TMA reads them: async proxy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
#if NORM_EARLY
    else {  // n4s1: Y rows of expert e complete after its last gemm2 item (a K-split's 2nd half writes Y before it counts)
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        if (atomicAdd(done + 32 + e, 1) == s_nd2[a] - 1) atomicAdd(done + 32 + e, 1 << 20);
      }
    }
#endif
#if TWM_TL
    if (tl && tid == 0 && ntl < 20) tl[3 + 3 * ntl] = gtimer();
    ++ntl;
#endif
  }
  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// ---- s29 moe_hot: the experts with 65..128 tokens (layer T = 256: 1-3 per call) in ONE pass of RT 2 items with
// N 128 (w*b2 packing: the same item rows as moe_twm's big class) instead of twm's two 64-token passes (= every weight
// dequantized twice). Own launch: 128 accumulators exceed twm's register budget, and a 2nd wgmma N in one main loop
// serializes the HGMMAs (s28). moe_twm runs as its PDL secondary (hx) on the SMs this grid leaves free and skips these
// experts. Items: all gemm13 items, then all gemm2 items (they wait on done[e] as in twm; deadlock-free the same way).
// Stage = x 128-row box (16 KB, SW128) + weights 8 KB + scales 1 KB.
// s23 NH 64: the experts with 33..64 tokens in one N 64 pass (layer T = 128), with moe_twq (flags bit 20) skipping them.
template <int NSTG, bool SA, int NH = 128>
__global__ void __launch_bounds__(384, 1) moe_hot(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y) {
  using HG = TWC<2, true, 8>;
  using HN = TWC<2, false, 8>;
  constexpr int XB = NH * 128, STRIDE = XB + HG::WB + HG::SB;
  static_assert(STRIDE % 1024 == 0 && HG::WB == HN::WB && HG::SB == HN::SB, "stage layout");
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_item[NSTG], s_nh;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 256; i += 384) slut[i] = lut[i];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 8 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    const int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    const bool h = nt > NH / 2 && nt <= NH;
    const unsigned b = __ballot_sync(0xffffffffu, h);
    if (h) s_e[__popc(b & ((1u << tid) - 1))] = e;
    if (tid == 0) s_nh = __popc(b);
  }
  __syncthreads();
  const int nh = s_nh, n13 = nh * HG::NRB, nitems = n13 + nh * HN::NRB;
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp > 8 || lane != 0) return;
    int* counter = meta + 67;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1) & 0xFFFFF;; item = atomicAdd(counter, 1) & 0xFFFFF) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      const int g2 = item >= n13, it = g2 ? item - n13 : item, nrb = g2 ? HN::NRB : HG::NRB;
      const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e];
      const char* wsrc = (g2 ? W2 : W13) + ((size_t)e * nrb + rb) * 64 * HG::WB;
      const char* ssrc = (g2 ? S2 : S13) + ((size_t)e * nrb + rb) * 64 * HG::SB;
      const CUtensorMap* map = g2 ? &maps2.m[NH == 128 ? 5 : 3] : &maps13.m[NH == 128 ? 5 : 3];
      if (g2) {
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= HG::NRB * NEEDK) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = 0; kc < 64; ++kc, ++i) {
        if (kc && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (!kc) s_item[st] = item;
        mbar_expect_tx(&full[st], STRIDE);
        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s_h(sb + XB, wsrc + (size_t)kc * HG::WB, HG::WB, &full[st], pol);
        bulk_g2s_h(sb + XB + HG::WB, ssrc + (size_t)kc * HG::SB, HG::SB, &full[st], pol);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
  int st = 0, ph = 0;
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
    if (item < 0) break;
    const int g2 = item >= n13, it = g2 ? item - n13 : item, nrb = g2 ? HN::NRB : HG::NRB;
    const int a = it / nrb, rb = it - a * nrb, e = s_e[a], off = s_off[e], nv = s_off[e + 1] - off;
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    tw_item<2, 1, NH, NSTG, true, 8, true, SA, STRIDE, XB>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb,
                                                           gsc, sroutes, g2 ? Y : Inter, g2);
    if (!g2) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
#if NORM_EARLY
    else {
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        if (atomicAdd(done + 32 + e, 1) == HN::NRB - 1) atomicAdd(done + 32 + e, 1 << 20);
      }
    }
#endif
  }
#if HOT_EMU
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (tid == 0) atomicAdd(meta + 67, 1 << 20);
#endif
}
template <bool SA, int NH>
static int64_t hot_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                      torch::Tensor inter_, torch::Tensor y_, int grid) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * (NH * 128 + TWC<2, true, 8>::WB + TWC<2, true, 8>::SB) + 1024;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(moe_hot<NSTG, SA, NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    init = true;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_hot<NSTG, SA, NH>, grid, 384, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr());
}
int64_t moe_hot_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_, int64_t grid) {
#define HOTA w13_, s13_, gs13_, w2_, s2_, gs2_, lut_, xg_, meta_, sr_, inter_, y_, (int)grid
  if (g_hot & 4) return (g_sa & 1) ? hot_go<true, 64>(HOTA) : hot_go<false, 64>(HOTA);  // s23: 33..64-token experts
  return (g_sa & 1) ? hot_go<true, 128>(HOTA) : hot_go<false, 128>(HOTA);
#undef HOTA
}
template <bool SA>
static int64_t twm_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor w13h_, torch::Tensor s13h_, torch::Tensor w2h_, torch::Tensor s2h_,
                      torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor inter_,
                      torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_twm<NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_twm<NSTG, SA>, grid, 384, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const char*)PTR(w13h_), (const char*)PTR(s13h_),
      (const char*)PTR(w2h_), (const char*)PTR(s2h_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(), (int*)tk_.data_ptr(),
      (float*)part_.data_ptr(), (int)std::min<int64_t>(ns, std::min<int64_t>(tk_.numel() / 2, part_.numel() / 32768)),
      g_hot);
}
int64_t moe_twm_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor w13h_, torch::Tensor s13h_, torch::Tensor w2h_,
                       torch::Tensor s2h_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_, torch::Tensor tk_, torch::Tensor part_, int64_t ns) {
#define TWMA w13_, s13_, gs13_, w2_, s2_, gs2_, w13h_, s13h_, w2h_, s2h_, lut_, xg_, meta_, sr_, inter_, y_, tk_, part_, ns
  return (g_sa & 1) ? twm_go<true>(TWMA) : twm_go<false>(TWMA);
#undef TWMA
}
// ---- s17 moe_twq: moe_twm with 16 consumer warps (4 warpgroups) x RT 2 instead of 8 x RT 4. An item covers the same
// 32 row tiles (it reads the w*b4 / w*sb4 packing: warp w16 takes slots of 8-warp warp w16 / 2), so the tensor work
// and the bytes per stage are unchanged, but each scheduler has 4 consumer warps instead of 2 (s17 ncu: twm items are
// latency-bound, 2 warps per scheduler, issue 43-62% busy). 640 threads: 4 consumer warpgroups (setmaxnreg 112; ptxas
// compiles the whole kernel at 96 registers) + a producer warpgroup (one thread works; setmaxnreg 32). Gated warp w16: gate tile rb*16 + w16 (slot r = w16 & 1), up = slot 2 + (w16 & 1);
// gemm2: tiles rb*32 + 2*w16 + {0, 1} (slots 2*(w16 & 1) + {0, 1}). All experts run 32-token passes (64-token
// passes need > 96 registers at 640 threads: wgmma serialized; an inline producer in warp 0 to get 128 registers was
// 2x slower, s17 NOTES): s17 proxy wins at T=128 (v16) only; T=256 (v32, hot experts of 90+ tokens) stays on twm.
template <int N, int NSTG, bool SA>
__device__ __forceinline__ void tq_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                        const uint32_t* slut, int warp, int lane, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out, bool g2,
                                        int nkc = 64, float* part = nullptr, int* tick = nullptr, int* s_flag = nullptr,
                                        int h = 0, int np = 1) {
  using C = TWC<4, true, 8>;
  constexpr int NA = N / 2, STRIDE = C::STG;
  float acc[2][NA];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[2][4], afB[2][4];
  const int w8 = warp >> 1, sub = warp & 1;
  const int wo0 = (w8 * 4 + (g2 ? 2 * sub : sub)) * 32 + lane, wo1 = wo0 + (g2 ? 32 : 64);
  const uint32_t sel = g2 ? (sub ? 0x7654u : 0x3210u) : (sub ? 0x7632u : 0x5410u);
  const int so = C::XB + C::WB + (w8 * 32 + lane) * 8;
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < nkc; ++kc) {
    const char* sb = smem + st * STRIDE;
    if (kc) mbar_wait(&full[st], ph);
    const uint4* wp = reinterpret_cast<const uint4*>(sb + C::XB);
    uint4 wv[2];
    wv[0] = wp[wo0];
    wv[1] = wp[wo1];
    const uint2 v = *reinterpret_cast<const uint2*>(sb + so);
    const uint32_t word = __byte_perm(v.x, v.y, sel);
    uint32_t s0[2], s1[2];
    if (SA) {
      const uint32_t p0 = scp<0>(word), p1 = scp<1>(word);
      s0[0] = blo(p0); s1[0] = bhi(p0); s0[1] = blo(p1); s1[1] = bhi(p1);
    } else {
      s0[0] = slut[word & 0xFF]; s1[0] = slut[(word >> 8) & 0xFF];
      s0[1] = slut[(word >> 16) & 0xFF]; s1[1] = slut[word >> 24];
    }
    const uint64_t d0 = sw128_desc(sb);
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
      const bool odd = j0 & 1;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (odd) dq(u4get(wv[r], j0), s0[r], s1[r], afB[r]);
        else dq(u4get(wv[r], j0), s0[r], s1[r], afA[r]);
      }
      wg_fence();
#pragma unroll
      for (int r = 0; r < 2; ++r) wgmma_bf16<N>(acc[r], odd ? afB[r] : afA[r], d0 + j0 * 2);
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (odd) fence_r(afA[r][i]);
          else fence_r(afB[r][i]);
        }
      if (j0 == 0 && pst >= 0) mbar_arrive(&empty[pst]);
    }
    pst = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
  }
  wg_wait<0>();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { fence_r(afA[r][i]); fence_r(afB[r][i]); }
#pragma unroll
    for (int i = 0; i < NA; ++i) fence_f(acc[r][i]);
  }
  mbar_arrive(&empty[pst]);
  if (part) {  // n21 K-split tail item (np parts of 64 / np k chunks, gemm2 only): every part stores its FP32 partial,
    // then takes a ticket; the last one sums the np partials in part order (deterministic) and runs the epilogue
    const int ct = warp * 32 + lane;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < NA; ++i) __stcg(part + (size_t)h * 16384 + (r * NA + i) * 512 + ct, acc[r][i]);
    __threadfence();
    asm volatile("bar.sync 1, 512;" ::: "memory");  // s_flag is free (the previous split item is done with it)
    if (ct == 0) {
      const int o = atomicAdd(tick, 1);
      if (o == np - 1) { *tick = 0; __threadfence(); }  // last user: zero for the next call
      *s_flag = o;
    }
    asm volatile("bar.sync 1, 512;" ::: "memory");
    if (*s_flag != np - 1) return;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < NA; ++i) {
        float v = __ldcg(part + (r * NA + i) * 512 + ct);
        for (int q = 1; q < np; ++q) v += __ldcg(part + (size_t)q * 16384 + (r * NA + i) * 512 + ct);
        acc[r][i] = v;
      }
  }
  if (g2) tw_epi<2, N, false, 16>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
  else tw_epi<2, N, true, 16>(acc, lane, warp, nv, off, rb, gsc, sroutes, Out);
}

template <int NSTG, bool SA>
__global__ void __launch_bounds__(640, 1) moe_twq(
    const char* __restrict__ W13, const char* __restrict__ S13, const float* __restrict__ gs13,
    const char* __restrict__ W2, const char* __restrict__ S2, const float* __restrict__ gs2,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps13, const __grid_constant__ TMaps maps2,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Inter,
    __nv_bfloat16* __restrict__ Y, float* __restrict__ part, int flags, int* __restrict__ tick) {
  using C = TWC<4, true, 8>;
  constexpr int STRIDE = C::STG, NR13 = TWC<4, true, 8>::NRB, NR2 = TWC<4, false, 8>::NRB;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[2][33], s_item[NSTG], s_big[32], s_flag;
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  int* done = meta + 68;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // n21 K split: the last ~(flags >> 8 & 255) gemm2 items (from an expert boundary) run as 2^(flags >> 16 & 3) K parts
  const int kq = (flags >> 16) & 3, npt = 1 << kq;
  int nsplit = kq ? (flags >> 8) & 255 : 0;
  for (int i = tid; i < 256; i += 640) slut[i] = lut[i];
  // s23 hx (flags bit 20): moe_hot<NH 64> (the PDL primary) runs the experts with 33..64 tokens: skip them, no wait at
  // entry (moe_hot triggers after its own wait: the gather is complete), wait at the end (moe_norm sees moe_hot's Y)
  const int hx = (flags >> 20) & 1;
  if (!hx) asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (tid < 33) s_off[tid] = meta[33 + tid];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 16 * 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 32) {
    const int nact = meta[0];
    const int e = tid < nact ? meta[1 + tid] : 0;
    s_e[tid] = e;
    int nt = tid < nact ? s_off[e + 1] - s_off[e] : 0;
    if (hx && nt > 32 && nt <= 64) nt = 0;
    const int big = 0;  // s17: 64-token passes do not fit (registers)
    const int npass = big ? (nt + 63) / 64 : (nt + 31) / 32;
    s_big[tid] = big;
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) {
      int incl = npass * (gq ? NR2 : NR13);
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      s_cum[gq][tid + 1] = incl;
      if (tid == 0) s_cum[gq][0] = 0;
    }
  }
  __syncthreads();
  const int n13 = s_cum[0][32], n2 = s_cum[1][32];
  int n2f = n2;
  if (nsplit > 0)
    for (int a = 32; a >= 0; --a)
      if (s_cum[1][a] >= n2 - nsplit) n2f = s_cum[1][a];
  const int ns = n2 - n2f, nitems = n13 + n2f + 2 * ns;
  // item -> (gemm2?, active slot a, pass p, row block rb, K half h (-1: whole K), split slot j)
  auto decode = [&](int item, int& g2, int& a, int& p, int& rb, int& h, int& j) {
    g2 = item >= n13;
    h = -1;
    j = 0;
    int it = g2 ? item - n13 : item;
    if (g2 && it >= n2f) {
      j = (it - n2f) >> 1;
      h = (it - n2f) & 1;
      it = n2f + j;
    }
    const int* cum = s_cum[g2];
    a = 0;
    while (cum[a + 1] <= it) ++a;
    const int nrb = g2 ? NR2 : NR13, rem = it - cum[a];
    if (flags & 8) {  // pass-minor: all passes of a row block adjacent (later passes read the weights from L2)
      const int np = (cum[a + 1] - cum[a]) / nrb;
      rb = rem / np;
      p = rem - rb * np;
    } else {
      p = rem / nrb;
      rb = rem - p * nrb;
    }
  };
  if (warp >= 16) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(TWQ_PREG) : "memory");
    if (warp > 16 || lane != 0) return;
    int* counter = meta + 66;
    int st = 0, ph = 0, i = 0;
    for (int item = atomicAdd(counter, 1);; item = atomicAdd(counter, 1)) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      if (item >= nitems) {
        s_item[st] = -1;
        mbar_arrive(&full[st]);
        break;
      }
      int g2, a, p, rb, h, j;
      decode(item, g2, a, p, rb, h, j);
      const int e = s_e[a], PS = s_big[a] ? 64 : 32;
      const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
      const int k0 = h < 0 ? 0 : h * (64 >> kq), k1 = h < 0 ? 64 : k0 + (64 >> kq);
      const int mi = nv <= 8 ? 0 : nv <= 16 ? 1 : nv <= 32 ? 2 : 3;
      const int nrb = g2 ? NR2 : NR13;
      const char* wsrc = (g2 ? W2 : W13) + ((size_t)e * nrb + rb) * 64 * C::WB;
      const char* ssrc = (g2 ? S2 : S13) + ((size_t)e * nrb + rb) * 64 * C::SB;
      const CUtensorMap* map = g2 ? &maps2.m[mi] : &maps13.m[mi];
      const int bytes = C::WB + C::SB + (1024 << mi);
      if (g2) {
        const int need = ((s_off[e + 1] - s_off[e] + PS - 1) / PS) * NR13;
        int v;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(done + e) : "memory");
          if (v >= need) break;
          __nanosleep(64);
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      for (int kc = k0; kc < k1; ++kc, ++i) {
        if (kc != k0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
        char* sb = smem + st * STRIDE;
        if (kc == k0) s_item[st] = item;
        mbar_expect_tx(&full[st], bytes);
        tma2d(sb, map, kc * 64, off, &full[st]);
        bulk_g2s(sb + C::XB, wsrc + (size_t)kc * C::WB, C::WB, &full[st]);
        bulk_g2s(sb + C::XB + C::WB, ssrc + (size_t)kc * C::SB, C::SB, &full[st]);
        if (++st == NSTG) { st = 0; ph ^= 1; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(TWQ_CREG) : "memory");
  int st = 0, ph = 0;
#if TWM_TL
  unsigned long long* tl = (flags & 2) ? reinterpret_cast<unsigned long long*>(part) + blockIdx.x * 64 : nullptr;
  int ntl = 0;
  if (tl && tid == 0) tl[0] = gtimer();
#endif
  for (;;) {
    mbar_wait(&full[st], ph);
    const int item = s_item[st];
#if TWM_TL
    if (tl && tid == 0) {
      if (item < 0) { tl[1] = ntl; tl[63] = gtimer(); }
      else if (ntl < 20) { tl[2 + 3 * ntl] = gtimer(); tl[4 + 3 * ntl] = item; }
    }
#endif
    if (item < 0) break;
    int g2, a, p, rb, h, j;
    decode(item, g2, a, p, rb, h, j);
    const int e = s_e[a], PS = s_big[a] ? 64 : 32;
    const int off = s_off[e] + p * PS, nv = min(PS, s_off[e + 1] - off);
    const float gsc = (g2 ? gs2[e] : gs13[e]) * 256.0f;
    __nv_bfloat16* out = g2 ? Y : Inter;
    float* pt = h < 0 ? nullptr : part + (size_t)j * npt * 16384;
#define TQI(NN) tq_item<NN, NSTG, SA>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, out, g2, \
                                      h < 0 ? 64 : 64 >> kq, pt, tick + j, &s_flag, h, npt)
    switch ((nv + 7) >> 3) {
      case 1: TQI(8); break;
      case 2: TQI(16); break;
      case 3: TQI(24); break;
      default: TQI(32); break;
    }
#undef TQI
    if (!g2) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("bar.sync 1, 512;" ::: "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(done + e, 1);
      }
    }
#if TWM_TL
    if (tl && tid == 0 && ntl < 20) tl[3 + 3 * ntl] = gtimer();
    ++ntl;
#endif
  }
  if (hx && tid == 0) asm volatile("griddepcontrol.wait;" ::: "memory");
}

// n21: clamp the K-split item count to the partial buffer (16384 floats per part) and the ticket words
static int64_t twq_flags(int64_t f, int64_t npart, int64_t ntk) {
  const int64_t kq = (f >> 16) & 3, ns = (f >> 8) & 255;
  if (!kq || !ns) return f & ~((int64_t)0x3FF00);
  const int64_t m = std::min<int64_t>(ns, std::min<int64_t>(npart / (16384 << kq), ntk));
  return (f & ~((int64_t)0x3FF00)) | (m << 8) | (m ? kq << 16 : 0);
}
template <bool SA>
static int64_t twq_go(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                      torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                      torch::Tensor inter_, torch::Tensor y_, torch::Tensor part_, torch::Tensor tk_, int64_t flags) {
  constexpr int NSTG = 8;
  constexpr int SMEM = NSTG * TWC<4, true, 8>::STG + 1024;
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_twq<NSTG, SA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  TMaps m13, m2;
  tw_maps(m13, xg_.data_ptr(), xg_.size(0));
  tw_maps(m2, inter_.data_ptr(), inter_.size(0));
  return launch_k(moe_twq<NSTG, SA>, grid, 640, SMEM, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const char*)PTR(w13_), (const char*)PTR(s13_), (const float*)PTR(gs13_), (const char*)PTR(w2_),
      (const char*)PTR(s2_), (const float*)PTR(gs2_), (const uint32_t*)PTR(lut_), m13, m2, (int*)meta_.data_ptr(),
      (const int*)PTR(sr_), (__nv_bfloat16*)inter_.data_ptr(), (__nv_bfloat16*)y_.data_ptr(),
      (float*)part_.data_ptr(), (int)twq_flags(flags, part_.numel(), tk_.numel()), (int*)tk_.data_ptr());
}
int64_t moe_twq_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor gs13_, torch::Tensor w2_, torch::Tensor s2_,
                       torch::Tensor gs2_, torch::Tensor lut_, torch::Tensor xg_, torch::Tensor meta_, torch::Tensor sr_,
                       torch::Tensor inter_, torch::Tensor y_, torch::Tensor part_, torch::Tensor tk_, int64_t flags) {
#define TWQA w13_, s13_, gs13_, w2_, s2_, gs2_, lut_, xg_, meta_, sr_, inter_, y_, part_, tk_, flags
  return (g_sa & 1) ? twq_go<true>(TWQA) : twq_go<false>(TWQA);
#undef TWQA
}
// ---- moe_tk2f: the decode gemm13 and gemm2 in ONE launch (moe_tk2 rows, one routing, one ring). Phase 1 = gemm13
// rows, phase 2 = gemm2 rows (each split over the CTAs as in moe_tk2); the producer streams phase-2 weights as soon
// as its phase-1 copies are issued. A CTA publishes the phase-1 rows it finished of each (expert, group) segment in
// cnt[a * 4 + gi] (bar + threadfence + atomicAdd); a phase-2 segment waits (ld.acquire) for cnt == 256 before it
// loads its x (= inter) fragments. Deadlock-free: phase 1 never waits, and every CTA (grid = #SMs) becomes resident
// once earlier kernels free their SMs. cnt lives in the state, zero between calls: moe_norm2 (after this kernel) zeroes it again (a memset node cost ~2.5 us).
// layer s68 (moe n14 TK2F_SQ, r-moe 01326b4): SQ = Wq is the single-copy repack (prepare _sq_pack: [E][RPE][NQ][SB bytes]
// = {tile lr chunks, tile lr + RPE chunks, scales}, exactly the stage's smem image): ONE bulk copy per stage instead of 3.
template <int NRT, int NC, int NSTG, int KPS, bool SQ = false, bool S6 = false>
__device__ __forceinline__ void tk2f_produce(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, int r0,
                                             int r1, const int* s_rs, const int* s_e, uint4* smem, uint64_t* full,
                                             uint64_t* empty, int& st, int& ph, int& i) {
  constexpr int RPE = NRT / 2, NQ = 64 / (NC * KPS), SB = NC * KPS * (S6 ? 1120 : 1152), KB = NC * KPS;
  static_assert(!S6 || (SQ && KPS == 2), "S6: SQ stages, KPS 2");
  int a = 0;
#if TK2F_L2H
  // s21: an expert with > 8 tokens is streamed once per 8-token group; the groups' rows sit 256 rows apart (other
  // CTAs, other times), so without hints the re-read often comes from HBM again (H100 ncu d16 s7: +12% DRAM bytes).
  uint64_t pol_last, pol_first;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_last));
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_first));
#endif
  for (int R = r0; R < r1; ++R) {
    while (s_rs[a + 1] <= R) ++a;
    const int e = s_e[a];
    const int lr = (R - s_rs[a]) & (RPE - 1);
    const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
    const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
    const char* qsrc = reinterpret_cast<const char*>(Wq) + ((size_t)e * RPE + lr) * NQ * SB;  // SQ
#if TK2F_L2H
    // mode 4: every group of a multi-group expert (the other group's CTA may read the same row first)
    const bool nfin = TK2F_L2H == 4 ? s_rs[a + 1] - s_rs[a] > RPE : R + RPE < s_rs[a + 1];
    const int hk = nfin ? (TK2F_L2H == 3 ? 0 : 1) : (TK2F_L2H == 2 ? 0 : 2);  // 0 none, 1 evict_last, 2 evict_first
    const uint64_t pol = hk == 1 ? pol_last : pol_first;
#endif
    for (int q = 0; q < NQ; ++q, ++i) {
      if (i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      char* sbase = reinterpret_cast<char*>(smem) + st * SB;
      mbar_expect_tx(&full[st], SB);
#if TK2F_L2H
      if (SQ) {
        if (hk) bulk_g2s_h(sbase, qsrc + q * SB, SB, &full[st], pol);
        else bulk_g2s(sbase, qsrc + q * SB, SB, &full[st]);
      } else if (hk) {
        bulk_g2s_h(sbase, src + q * KB * 32, KB * 512, &full[st], pol);
        bulk_g2s_h(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st], pol);
        bulk_g2s_h(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st], pol);
      } else
#else
      if (SQ) bulk_g2s(sbase, qsrc + q * SB, SB, &full[st]);
      else
#endif
      {
        bulk_g2s(sbase, src + q * KB * 32, KB * 512, &full[st]);
        bulk_g2s(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st]);
        bulk_g2s(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st]);
      }
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA = false, int PF = 0>
__device__ __forceinline__ void tk2f_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int need, int r0, int r1,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint64_t* full, uint64_t* empty, int& st,
                                             int& ph, int& rb) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * ((PF & 1024) ? 1120 : 1152);
  constexpr bool S6 = (PF & 1024) != 0;  // layer s73 (moe n20): 6-bit scale codes
  constexpr bool BAL = (PF & 32) != 0;  // moe n8 BAL: per-segment counters cnt[64 + j]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, ci = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      ci = a * 4 + gi;
      if (BAL && !GATED) {
        const int* cp = cnt + 64 + seg_start / RPE;
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cp) : "memory");
        } while (c < 2 * RPE);
      } else if (!GATED && !((PF & 2) && R != r0)) {  // all gemm13 rows written (by any CTA): see moe_tk2f. PF & 2 (s13): only
        int c;                                    // at the CTA's first gemm2 row (cnt never decreases within a call)
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);
      }
      const bool v = g < n;
      int row = 0;
      if (v) row = GATED ? (s_srt[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        if (GATED) {
          xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
          xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
        } else {  // written in this kernel: L2 only
          xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
          xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
        }
      }
    }
    if ((PF & 1) && R + 1 == seg_end && R + 1 < r1) {  // s13: L1 prefetch of the next segment's x during this row
      const int a2 = seg_end >= s_rs[a + 1] ? a + 1 : a;
      const int e2 = s_e[a2], off2 = s_off[e2] + ((seg_end - s_rs[a2]) / RPE) * 8;
      if (g < s_off[e2 + 1] - off2) {
        const int row2 = GATED ? (s_srt[off2 + g] >> 3) : (off2 + g);
        const char* p = reinterpret_cast<const char*>(X + (size_t)row2 * KD + 16 * t);
#pragma unroll
        for (int qq = 0; qq < NQ; ++qq) asm volatile("prefetch.global.L1 [%0];" ::"l"(p + (qq * NC + warp) * 128));
      }
    }
    constexpr int NA = (PF & 8) ? 2 : 1;  // s13 PF & 8: 2 accumulator chains per tile (j even / odd), summed at the end
    float acc[2][4], acc2[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = acc2[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      uint4 wv0[KPS], wv1[KPS];
      uint32_t sw[KPS], s6n = 0, s6h = 0;
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        wv0[h] = sw4[h * SW];
        wv1[h] = sw4[(KPS + h) * SW];
        if (!S6) sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];
      }
      if (S6) {
        const uint32_t* sb = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane);
        s6n = sb[warp * 32 + lane];
        s6h = reinterpret_cast<const unsigned short*>(sb + NC * 32)[warp * 32 + lane];
      }
      const int cs = st;  // s10: arrive on empty only AFTER the stage's registers are used (see below)
      if (++st == NSTG) { st = 0; ph ^= 1; }
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        uint32_t sc[4];
        if (S6) {
          s6dec(s6n, s6h, h, sc);
        } else if (SA) {
          { const uint32_t p0 = scp<0>(sw[h]), p1 = scp<1>(sw[h]); sc[0] = blo(p0); sc[1] = bhi(p0); sc[2] = blo(p1); sc[3] = bhi(p1); }
        } else {
          sc[0] = slut[sw[h] & 0xFF]; sc[1] = slut[(sw[h] >> 8) & 0xFF]; sc[2] = slut[(sw[h] >> 16) & 0xFF];
          sc[3] = slut[sw[h] >> 24];
        }
        const int qq = q * KPS + h;
        if (PF & 256) {  // layer s81 (moe n10 probe): timing only (wrong output), never selectable: no math, keep the loads
          acc[0][0] += __uint_as_float(sc[0] ^ wv0[h].x ^ wv1[h].y ^ xf[qq][0].x);
          continue;
        }
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
          const uint4 wv = ii ? wv1[h] : wv0[h];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t af[4];
            dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
            mma16816((NA == 2 && (j & 1)) ? acc2[ii] : acc[ii], af, u4get(xf[qq][j >> 1], (j & 1) * 2),
                   u4get(xf[qq][j >> 1], (j & 1) * 2 + 1));
          }
        }
      }
      // s10 RACE FIX: arriving right after ISSUING the ring LDS (as before) let the producer's next bulk copy
      // overwrite the stage before some LDS had read it (seen as wrong weights in the first row after an x reload,
      // when 16 LDGs per thread are in flight: the s7 "first gemm2 row" race). Arrive once the data is consumed.
      mbar_arrive(&empty[cs]);
    }
    if (NA == 2) {
#pragma unroll
      for (int ii = 0; ii < 2; ++ii)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[ii][r] += acc2[ii][r];
    }
    float* red = s_red + rb * (NC * 256);
    rb ^= 1;
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[((warp * 2 + ii) * 4 + r) * 32 + lane] = acc[ii][r];
    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
    const int rp = R - seg_start;
    const int ct = tid, l = ct & 31, gg = l >> 2, tt = l & 3;
    if (GATED) {
      if (ct < 128) {
        const int r = ct >> 5;
        float v = 0.f, u = 0.f;
#pragma unroll
        for (int w = 0; w < NC; ++w) { v += red[((w * 2) * 4 + r) * 32 + l]; u += red[((w * 2 + 1) * 4 + r) * 32 + l]; }
        const int tok = 2 * tt + (r & 1);
        if (tok < n) {
          v *= gsc;
          u *= gsc;
          const int col = rp * 16 + gg + ((r & 2) ? 8 : 0);
          reinterpret_cast<__nv_bfloat16*>(Out)[(size_t)(off + tok) * NOUT + col] =
              __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        }
      }
      if (R + 1 == seg_end || R + 1 == r1) {  // publish this CTA's rows of the segment
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(BAL ? cnt + 64 + seg_start / RPE : cnt, R + 1 - max(seg_start, r0));
        }
      }
    } else if (NC == 8 || ct < 256) {
      const int ii = ct >> 7, r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        if (PF & 4) reinterpret_cast<float*>(Out)[(size_t)s_srt[off + tok] * NOUT + col] = v * gsc;  // route slot order
        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

// s10 GS16: 16-token segments. One row of a segment with n > 8 tokens dequantizes every A fragment ONCE and runs
// it against both 8-token groups (TWO): group 0's x fragments in registers (xf), group 1's in smem (xs, lane-private,
// written by the same lane at the segment change). Dequant is ~70% of a row's instructions, so a 2-group row costs
// ~1.15x a 1-group row instead of 2x. s_red holds 2 groups x 2 tiles (double-buffered).
template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false, bool SLOT = false,
          bool S6 = false>
__device__ __forceinline__ void tk2g_row(const uint32_t* slut, const uint4 (&xf)[64 / NC][2], const uint4* xs, void* Out,
                                         int rp, int off, int n, float gsc, float* s_red, uint4* smem, uint64_t* full,
                                         uint64_t* empty, int& st, int& ph, int& rb, bool gated_rt = false,
                                         const int* s_srt = nullptr) {
  // SLOT (n6, = tk2f PF & 4): gemm2 Y row = route slot s_srt[off + tok], pre-scaled by gsc (= gs2 * 256): moe_norm3
  // RTG (s12): gated is a run-time flag (gemm13 and gemm2 share ONE main-loop body: i-cache), RPE from it
  const bool GATED = RTG ? gated_rt : GATED_;
  const int RPE = RTG ? (gated_rt ? 256 : 128) : NRT / 2;
  constexpr int NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * (S6 ? 1120 : 1152), NG = TWO ? 2 : 1;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint4* wsm = smem + warp * 32 + lane;
  float acc[NG][2][4];
#pragma unroll
  for (int h = 0; h < NG; ++h)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[h][i][r] = 0.f;
#pragma unroll
  for (int q = 0; q < NS; ++q) {
    uint4 xb[KPS][2];
    if (TWO) {
#pragma unroll
      for (int h = 0; h < KPS; ++h) { xb[h][0] = xs[((q * KPS + h) * 2) * 32]; xb[h][1] = xs[((q * KPS + h) * 2 + 1) * 32]; }
    }
    mbar_wait(&full[st], ph);
    const uint4* sw4 = wsm + st * (SB / 16);
    uint4 wv0[KPS], wv1[KPS];
    uint32_t sw[KPS], s6n = 0, s6h = 0;
#pragma unroll
    for (int h = 0; h < KPS; ++h) {
      wv0[h] = sw4[h * SW];
      wv1[h] = sw4[(KPS + h) * SW];
      if (!S6) sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];
    }
    if (S6) {  // layer s73 (moe n20)
      const uint32_t* sb = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane);
      s6n = sb[warp * 32 + lane];
      s6h = reinterpret_cast<const unsigned short*>(sb + NC * 32)[warp * 32 + lane];
    }
    const int cs = st;
    if (++st == NSTG) { st = 0; ph ^= 1; }
#pragma unroll
    for (int h = 0; h < KPS; ++h) {
      uint32_t sc[4];
      if (S6) {
        s6dec(s6n, s6h, h, sc);
      } else if (SA) {
        { const uint32_t p0 = scp<0>(sw[h]), p1 = scp<1>(sw[h]); sc[0] = blo(p0); sc[1] = bhi(p0); sc[2] = blo(p1); sc[3] = bhi(p1); }
      } else {
        sc[0] = slut[sw[h] & 0xFF]; sc[1] = slut[(sw[h] >> 8) & 0xFF]; sc[2] = slut[(sw[h] >> 16) & 0xFF];
        sc[3] = slut[sw[h] >> 24];
      }
      const int qq = q * KPS + h;
#pragma unroll
      for (int ii = 0; ii < 2; ++ii) {
        const uint4 wv = ii ? wv1[h] : wv0[h];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
          mma16816(acc[0][ii], af, u4get(xf[qq][j >> 1], (j & 1) * 2), u4get(xf[qq][j >> 1], (j & 1) * 2 + 1));
          if (TWO) mma16816(acc[NG - 1][ii], af, u4get(xb[h][j >> 1], (j & 1) * 2), u4get(xb[h][j >> 1], (j & 1) * 2 + 1));
        }
      }
    }
    mbar_arrive(&empty[cs]);  // only after the stage is consumed: see tk2f_consume (s10 race)
  }
  float* red = s_red + rb * (2 * NC * 256);
  rb ^= 1;
#pragma unroll
  for (int h = 0; h < NG; ++h)
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[(((h * NC + warp) * 2 + ii) * 4 + r) * 32 + lane] = acc[h][ii][r];
  asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
  const int ct = tid, l = ct & 31, gg = l >> 2, tt = l & 3;
  if (GATED) {
    const int h = ct >> 7;  // token group (threads 128-255 only for TWO)
    if (TWO || h == 0) {
      const int r = (ct >> 5) & 3;
      float v = 0.f, u = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) {
        v += red[(((h * NC + w) * 2) * 4 + r) * 32 + l];
        u += red[(((h * NC + w) * 2 + 1) * 4 + r) * 32 + l];
      }
      const int tok = h * 8 + 2 * tt + (r & 1);
      if (tok < n) {
        v *= gsc;
        u *= gsc;
        const int col = rp * 16 + gg + ((r & 2) ? 8 : 0);
        reinterpret_cast<__nv_bfloat16*>(Out)[(size_t)(off + tok) * NOUT + col] =
            __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
      }
    }
  } else {
    const int ii = ct >> 7, r = (ct >> 5) & 3;
#pragma unroll
    for (int h = 0; h < NG; ++h) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[(((h * NC + w) * 2 + ii) * 4 + r) * 32 + l];
      const int tok = h * 8 + 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        if (SLOT) reinterpret_cast<float*>(Out)[(size_t)s_srt[off + tok] * NOUT + col] = v * gsc;
        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA>
__device__ __forceinline__ void tk2g_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int need, int r0, int r1,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint4* s_x2, uint64_t* full, uint64_t* empty,
                                             int& st, int& ph, int& rb) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  uint4* xs = s_x2 + warp * (NQ * 2 * 32) + lane;
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0;
  float gsc = 0.f;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 16;
      n = min(16, s_off[e + 1] - off);
      if (!GATED) {  // all gemm13 rows written (by any CTA): see moe_tk2f
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);

      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (h && n <= 8) break;
        const bool v = h * 8 + g < n;
        int row = 0;
        if (v) row = GATED ? (s_srt[off + h * 8 + g] >> 3) : (off + h * 8 + g);
        const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
        for (int qq = 0; qq < NQ; ++qq) {
          const int kc = qq * NC + warp;
          uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
          if (v) {
            if (GATED) { x0 = __ldg(xr + kc * 8); x1 = __ldg(xr + kc * 8 + 1); }
            else { x0 = __ldg(xr + kc * 8); x1 = __ldg(xr + kc * 8 + 1); }
          }
          if (h) { xs[(qq * 2) * 32] = x0; xs[(qq * 2 + 1) * 32] = x1; }
          else { xf[qq][0] = x0; xf[qq][1] = x1; }
        }
      }
    }
    if (n > 8) tk2g_row<NRT, NC, NSTG, GATED, KPS, SA, true>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb);
    else tk2g_row<NRT, NC, NSTG, GATED, KPS, SA, false>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb);
    if (GATED && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
      asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
      if (tid == 0) {
        __threadfence();
        atomicAdd(cnt, R + 1 - max(seg_start, r0));
      }
    }
  }
}

// s12 MERGED consumer (tk2h): both phases in ONE run-time loop, so the kernel holds one main-loop body per row kind
// (GS 8: 1 body; GS 16: 2) instead of 2 / 4. GS16's loss was instruction-cache misses: ncu stall_no_instruction at
// layer d1 923 -> 9418 (118 KB kernel vs 65 KB) with no 2-group row at all.
template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false, bool S6 = false>
__device__ __forceinline__ void tk2h_consume(const uint32_t* slut, const float* __restrict__ gs13,
                                             const float* __restrict__ gs2, const __nv_bfloat16* X,
                                             __nv_bfloat16* inter, float* Y, int* cnt, int need, int a0, int a1, int b0,
                                             int b1, const int* s_rs13, const int* s_rs2, const int* s_e,
                                             const int* s_off, const int* s_srt, float* s_red, uint4* smem, uint4* s_x2,
                                             uint64_t* full, uint64_t* empty, int& st, int& ph, int& rb) {
  constexpr int NQ = 64 / NC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  uint4* xs = GS == 16 ? s_x2 + warp * (NQ * 2 * 32) + lane : nullptr;
#pragma unroll 1
  for (int p = 0; p < 2; ++p) {
    const bool gated = p == 0;
    if (!gated) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // layer s58: as TK2F_LATE_TRIG 3
    const int RPE = gated ? 256 : 128;
    const int* s_rs = gated ? s_rs13 : s_rs2;
    const float* gs = gated ? gs13 : gs2;
    const __nv_bfloat16* Xp = gated ? X : inter;
    void* Out = gated ? (void*)inter : (void*)Y;
    const int r0 = gated ? a0 : b0, r1 = gated ? a1 : b1;
    int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0;
    float gsc = 0.f;
#pragma unroll 1
    for (int R = r0; R < r1; ++R) {
      if (R >= seg_end) {
        while (s_rs[a + 1] <= R) ++a;
        const int gi = (R - s_rs[a]) / RPE;
        seg_start = s_rs[a] + gi * RPE;
        seg_end = seg_start + RPE;
        const int e = s_e[a];
        gsc = gs[e] * 256.0f;
        off = s_off[e] + gi * GS;
        n = min(GS, s_off[e + 1] - off);
        if (!gated) {  // all gemm13 rows written (by any CTA): see moe_tk2f. PSC (s12): only this segment's 256 rows
          // (segment j = gemm13 rows [256 j, 256 j + 256) = gemm2 rows [128 j, 128 j + 128); counter cnt[64 + j], j < 64)
          const int* cp = PSC ? cnt + 64 + seg_start / 128 : cnt;
          const int nd = PSC ? 256 : need;
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cp) : "memory");
          } while (c < nd);
        }
#pragma unroll
        for (int h = 0; h < GS / 8; ++h) {
          if (h && n <= 8) break;
          const bool v = h * 8 + g < n;
          int row = 0;
          if (v) row = gated ? (s_srt[off + h * 8 + g] >> 3) : (off + h * 8 + g);
          const uint4* xr = reinterpret_cast<const uint4*>(Xp + (size_t)row * KD + 16 * t);
#pragma unroll
          for (int qq = 0; qq < NQ; ++qq) {
            const int kc = qq * NC + warp;
            uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
            if (v) { x0 = __ldcg(xr + kc * 8); x1 = __ldcg(xr + kc * 8 + 1); }  // layer s58: __ldcg (PDL / in-kernel producer)
            if (h) { xs[(qq * 2) * 32] = x0; xs[(qq * 2 + 1) * 32] = x1; }
            else { xf[qq][0] = x0; xf[qq][1] = x1; }
          }
        }
      }
      if (GS == 16 && n > 8)
        tk2g_row<0, NC, NSTG, false, KPS, SA, true, true, SLOT, S6>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated, s_srt);
      else
        tk2g_row<0, NC, NSTG, false, KPS, SA, false, true, SLOT, S6>(slut, xf, xs, Out, R - seg_start, off, n, gsc, s_red, smem, full, empty, st, ph, rb, gated, s_srt);
      if (gated && (R + 1 == seg_end || R + 1 == r1)) {  // publish this CTA's rows of the segment
        asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
        if (tid == 0) {
          __threadfence();
          atomicAdd(PSC ? cnt + 64 + seg_start / 256 : cnt, R + 1 - max(seg_start, r0));
        }
      }
    }
  }
}

// DYN variant (s8): CTA b first takes the static rows [b*S, (b+1)*S) of a phase (S = rows/grid - D), then single
// rows from the tail [grid*S, N) via a global counter (ctr, pulled by the producer one row ahead). The producer
// tells the consumers each row's index through s_rid (written before the row's first full-barrier arrive, which
// releases it); -1 = end of the phase (a stage with no data: plain arrive). Rows of a CTA stay increasing.
template <int NRT, int NC, int NSTG, int KPS, bool HARD>
__device__ __forceinline__ void tk2d_produce(const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, int S,
                                             int N, int* ctr, const int* s_rs, const int* s_e, int* s_rid, uint64_t* ridb,
                                             uint4* smem, uint64_t* full, uint64_t* empty, int& st, int& ph, int& i,
                                             int& k) {
  constexpr int RPE = NRT / 2, NQ = 64 / (NC * KPS), SB = NC * KPS * 1152, KB = NC * KPS;
  const int G0 = gridDim.x * S;
  int a = 0, R = blockIdx.x * S;
  const int sEnd = R + S;
  int nxt = S == 0 ? atomicAdd(ctr, 1) : 0;  // next tail row, fetched one row early
  while (true) {
    int Rn;
    if (R < sEnd) {
      Rn = R++;
      if (R == sEnd) nxt = atomicAdd(ctr, 1);
    } else {
      Rn = G0 + nxt;
      if (Rn >= N) Rn = -1;
      else nxt = atomicAdd(ctr, 1);
    }
    if (Rn >= 0 && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
    s_rid[k & 3] = Rn;  // row id (-1: end of the phase) on its own arrive-only barrier: no wait for data
    mbar_arrive(&ridb[k & 3]);
    ++k;
    if (Rn < 0) return;
    while (s_rs[a + 1] <= Rn) ++a;
    const int e = s_e[a];
    const int lr = (Rn - s_rs[a]) & (RPE - 1);
    const uint4* src = Wq + ((size_t)e * NRT * 64 + lr * 64) * 32;
    const uint32_t* ssrc = S2 + ((size_t)e * RPE * 64 + lr * 64) * 32;
    for (int q = 0; q < NQ; ++q, ++i) {
      if (q && i >= NSTG) mbar_wait(&empty[st], ph ^ 1);
      char* sbase = reinterpret_cast<char*>(smem) + st * SB;
      mbar_expect_tx(&full[st], SB);
      bulk_g2s(sbase, src + q * KB * 32, KB * 512, &full[st]);
      bulk_g2s(sbase + KB * 512, src + NRT * 32 * 32 + q * KB * 32, KB * 512, &full[st]);
      bulk_g2s(sbase + KB * 1024, ssrc + q * KB * 32, KB * 128, &full[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
    }
  }
}

template <int NRT, int NC, int NSTG, bool GATED, int KPS>
__device__ __forceinline__ void tk2d_consume(const uint32_t* slut, const float* __restrict__ gs,
                                             const __nv_bfloat16* X, void* Out, int* cnt, int need, int S, const int* s_rid, uint64_t* ridb,
                                             const int* s_rs, const int* s_e, const int* s_off, const int* s_srt,
                                             float* s_red, uint4* smem, uint64_t* full, uint64_t* empty, int& st,
                                             int& ph, int& rb, int& k) {
  constexpr int RPE = NRT / 2, NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * 1152;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, nd = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  const int R0 = blockIdx.x * S;
  for (int j = 0;; ++j, ++k) {
    int R = R0 + j;
    if (j >= S) {  // tail row: its index comes from the producer
      mbar_wait(&ridb[k & 3], (k >> 2) & 1);
      R = s_rid[k & 3];
      if (R < 0) { ++k; break; }
    }
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;
      seg_end = seg_start + RPE;
      const int e = s_e[a];
      gsc = gs[e] * 256.0f;
      off = s_off[e] + gi * 8;
      n = min(8, s_off[e + 1] - off);
      if (!GATED && nd == 0) {  // all gemm13 rows written (by any CTA), once per CTA
        int c;
        do {
          asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt) : "memory");
        } while (c < need);
      }
      const bool v = g < n;
      int row = 0;
      if (v) row = GATED ? (s_srt[off + g] >> 3) : (off + g);
      const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)row * KD + 16 * t);
#pragma unroll
      for (int qq = 0; qq < NQ; ++qq) {
        const int kc = qq * NC + warp;
        xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
        xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
      }
    }
    ++nd;
    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][r] = 0.f;
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      mbar_wait(&full[st], ph);
      const uint4* sw4 = wsm + st * (SB / 16);
      uint4 wv0[KPS], wv1[KPS];
      uint32_t sw[KPS];
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        wv0[h] = sw4[h * SW];
        wv1[h] = sw4[(KPS + h) * SW];
        sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];
      }
      mbar_arrive(&empty[st]);
      if (++st == NSTG) { st = 0; ph ^= 1; }
#pragma unroll
      for (int h = 0; h < KPS; ++h) {
        const uint32_t sc[4] = {slut[sw[h] & 0xFF], slut[(sw[h] >> 8) & 0xFF], slut[(sw[h] >> 16) & 0xFF], slut[sw[h] >> 24]};
        const int qq = q * KPS + h;
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
          const uint4 wv = ii ? wv1[h] : wv0[h];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint32_t af[4];
            dq(u4get(wv, j), sc[2 * ii], sc[2 * ii + 1], af);
            mma16816(acc[ii], af, u4get(xf[qq][j >> 1], (j & 1) * 2), u4get(xf[qq][j >> 1], (j & 1) * 2 + 1));
          }
        }
      }
    }
    float* red = s_red + rb * (NC * 256);
    rb ^= 1;
#pragma unroll
    for (int ii = 0; ii < 2; ++ii)
#pragma unroll
      for (int r = 0; r < 4; ++r) red[((warp * 2 + ii) * 4 + r) * 32 + lane] = acc[ii][r];
    asm volatile("bar.sync 1, %0;" ::"n"(NC * 32) : "memory");
    const int rp = R - seg_start;
    const int ct = tid, l = ct & 31, gg = l >> 2, tt = l & 3;
    if (GATED) {
      if (ct < 128) {
        const int r = ct >> 5;
        float v = 0.f, u = 0.f;
#pragma unroll
        for (int w = 0; w < NC; ++w) { v += red[((w * 2) * 4 + r) * 32 + l]; u += red[((w * 2 + 1) * 4 + r) * 32 + l]; }
        const int tok = 2 * tt + (r & 1);
        if (tok < n) {
          v *= gsc;
          u *= gsc;
          const int col = rp * 16 + gg + ((r & 2) ? 8 : 0);
          reinterpret_cast<__nv_bfloat16*>(Out)[(size_t)(off + tok) * NOUT + col] =
              __float2bfloat16(__fdividef(v, 1.0f + __expf(-v)) * u);
        }
      }
    } else {
      const int ii = ct >> 7, r = (ct >> 5) & 3;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NC; ++w) v += red[((w * 2 + ii) * 4 + r) * 32 + l];
      const int tok = 2 * tt + (r & 1);
      if (tok < n) {
        const int col = (rp + ii * RPE) * 16 + gg + ((r & 2) ? 8 : 0);
        reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;
      }
    }
  }
  if (GATED) {  // publish this CTA's gemm13 rows
    asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
    if (tid == 0 && nd) {
      __threadfence();
      atomicAdd(cnt, nd);
    }
  }
}

// s9 fused post-expert norm (FN): every CTA takes a ticket (cnt[4], monotonic over calls: generation g = cnt[7], read
// at kernel start, before any ticket of the call) after its gemm2 rows; the last M take one token each, wait until all
// G CTAs have ticked, then do moe_norm2's work for it in one pass (y rows from L2, all loads in flight at once) and
// write out. The last ticket zeroes cnt[0] (every CTA is past its gemm2 wait) and bumps cnt[7]. Replaces moe_norm2
// and its launch (moe_norm2, used by the other paths, zeroes all counters: consistent with this too).
// s_rt[i] / s_rg[i]: route i's topk weight and global scale * 256 (0 when not local), s_inv[i]: its y row (-1).
template <int NC>
__device__ __forceinline__ void tk2f_norm(int* cnt, int M, const int* s_inv, const float* s_rt, const float* s_rg,
                                          const float* __restrict__ w1, const float* Y, float eps,
                                          __nv_bfloat16* __restrict__ out, float* s_ss, int* s_tk, int gen) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, G = gridDim.x;
  float wv[16];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float4 a = __ldg(wp + h * 512 + tid * 2), b = __ldg(wp + h * 512 + tid * 2 + 1);
      wv[h * 8 + 0] = a.x; wv[h * 8 + 1] = a.y; wv[h * 8 + 2] = a.z; wv[h * 8 + 3] = a.w;
      wv[h * 8 + 4] = b.x; wv[h * 8 + 5] = b.y; wv[h * 8 + 6] = b.z; wv[h * 8 + 7] = b.w;
    }
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  if (tid == 0) {
    __threadfence();
    const int t = atomicAdd(cnt + 4, 1) - gen * G;
    if (t == G - 1) {  // last ticket: all CTAs are done with cnt[0]
      cnt[0] = 0;
      __threadfence();
      atomicExch(cnt + 7, gen + 1);
    }
    s_tk[0] = t;
    s_tk[1] = gen;
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  const int tok = s_tk[0] - (G - M);
  if (tok < 0) return;
  gen = s_tk[1];
  int pos[8];
  float gsc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    pos[s] = max(s_inv[tok * 8 + s], 0);  // not local: row 0 (valid memory), scaled by 0
    gsc[s] = s_rg[tok * 8 + s];
  }
  {
    const int need = (gen + 1) * G;
    int c;
    do {
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c) : "l"(cnt + 4) : "memory");
    } while (c < need);
  }
  float v[8][16], ss[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const float4* p = reinterpret_cast<const float4*>(Y + (size_t)pos[s] * NOUT);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float4 a = __ldg(p + h * 512 + tid * 2), b = __ldg(p + h * 512 + tid * 2 + 1);
      v[s][h * 8 + 0] = a.x; v[s][h * 8 + 1] = a.y; v[s][h * 8 + 2] = a.z; v[s][h * 8 + 3] = a.w;
      v[s][h * 8 + 4] = b.x; v[s][h * 8 + 5] = b.y; v[s][h * 8 + 6] = b.z; v[s][h * 8 + 7] = b.w;
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    ss[s] = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) { v[s][k] *= gsc[s] * wv[k]; ss[s] += v[s][k] * v[s][k]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[s] += __shfl_xor_sync(0xffffffffu, ss[s], o);
    if (lane == 0) s_ss[warp * 8 + s] = ss[s];
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC * 32) : "memory");
  float acc[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) acc[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < NC; ++w) tot += s_ss[w * 8 + s];
    const float fs = gsc[s] != 0.f ? s_rt[tok * 8 + s] * rsqrtf(tot / NOUT + eps) : 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] += fs * v[s][k];
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint4 o;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int k = 0; k < 4; ++k) op[k] = __floats2bfloat162_rn(acc[h * 8 + 2 * k], acc[h * 8 + 2 * k + 1]);
    reinterpret_cast<uint4*>(out + (size_t)tok * NOUT)[h * 256 + tid] = o;
  }
}

// GS16 weighted static split: first row of CTA b when rows of 2-group segments (n > 8) weigh wt/8 and others 1.
// Monotone in b, and b = gridDim.x gives the total row count, so CTA ranges tile the rows exactly.
__device__ __forceinline__ int tk2g_split(int b, int nact, int wt, int rps, const int* s_e, const int* s_off) {
  long long W = 0;
  for (int a = 0; a < nact; ++a) {
    const int c = s_off[s_e[a] + 1] - s_off[s_e[a]];
    for (int o = 0; o < c; o += 16) W += (long long)rps * (c - o > 8 ? wt : 8);
  }
  const long long p = W * b / gridDim.x;
  long long cum = 0;
  int row = 0;
  for (int a = 0; a < nact; ++a) {
    const int c = s_off[s_e[a] + 1] - s_off[s_e[a]];
    for (int o = 0; o < c; o += 16) {
      const int w = c - o > 8 ? wt : 8;
      if (p < cum + (long long)rps * w) return row + (int)((p - cum + w - 1) / w);
      cum += (long long)rps * w;
      row += rps;
    }
  }
  return row;
}

// n4s5: tk2g_split by warp 0 in 32-bit lane-parallel math (one lane per active expert; same bounds as tk2g_split).
// tk2g_split ran 64-bit loops in every thread: consumers started 1.8 (median) / 4.1 (max) us late at d16 (G88).
__device__ __forceinline__ int tk2g_bound(int b, int nact, int wt, int rps, const int* s_e, const int* s_off) {
  const int lane = threadIdx.x & 31;
  const int c = lane < nact ? s_off[s_e[lane] + 1] - s_off[s_e[lane]] : 0;
  int w = 0;  // this expert's weight / rps
  for (int o = 0; o < c; o += 16) w += c - o > 8 ? wt : 8;
  const int ng = (c + 15) >> 4;
  int incl = w * rps, rows = ng * rps;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o), r = __shfl_up_sync(0xffffffffu, rows, o);
    if (lane >= o) { incl += v; rows += r; }
  }
  const int W = __shfl_sync(0xffffffffu, incl, 31);
#if TK2G_PAR4
  const int p = (int)((unsigned)W * (unsigned)b / gridDim.x);  // n17: 32-bit (W * b < 2^28)
#else
  const int p = (int)((long long)W * b / gridDim.x);
#endif
  int cum = incl - w * rps, row = rows - ng * rps;
  int res = 0x7fffffff;
  if (ng && p >= cum && p < incl) {
    for (int o = 0; o < c; o += 16) {
      const int sw = c - o > 8 ? wt : 8;
      if (p < cum + rps * sw) { res = row + (p - cum + sw - 1) / sw; break; }
      cum += rps * sw;
      row += rps;
    }
  }
  if (p >= W) res = __shfl_sync(0xffffffffu, rows, 31);
  return __reduce_min_sync(0xffffffffu, res);
}

// s12 WEIGHTED static split (GS 8): a row of an expert's first 8-token group streams its weights from HBM, rows of the
// extra groups (hot experts) mostly hit L2 and cost ~wE/8 of it (H200 fit on layer routings: 4.4 / 6.97 us = 0.63).
// Warp 0 computes CTA b's row bounds [r(b), r(b+1)) of a phase with RPS rows per group; lane = active expert index.
// 32-bit math: total weight <= 64 groups x 256 rows x 8.
__device__ __forceinline__ int tk2w_bound(int b, int nact, int wE, int rps, const int* s_e, const int* s_off) {
  const int lane = threadIdx.x & 31;
  const int c = lane < nact ? s_off[s_e[lane] + 1] - s_off[s_e[lane]] : 0;
  const int ng = (c + 7) >> 3;
  const int w = ng ? rps * (8 + (ng - 1) * wE) : 0;  // this expert's weight
  int incl = w, rows = ng * rps;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o), r = __shfl_up_sync(0xffffffffu, rows, o);
    if (lane >= o) { incl += v; rows += r; }
  }
  const int W = __shfl_sync(0xffffffffu, incl, 31);
  const int p = (int)((long long)W * b / gridDim.x);
  const int cum = incl - w, row0 = rows - ng * rps;
  int res = 0x7fffffff;
  if (ng && p >= cum && p < incl) {  // p falls in this expert: group 0 (weight 8 per row), then groups of weight wE
    const int q = p - cum;
    if (q < rps * 8) res = row0 + (q + 7) / 8;
    else res = row0 + rps + (q - rps * 8 + wE - 1) / wE;
  }
  if (p >= W) res = __shfl_sync(0xffffffffu, rows, 31);
  res = __reduce_min_sync(0xffffffffu, res);
  return res;
}

template <int NC, int NSTG, int KPS, bool DYN, bool FN = false, bool SA = false, int GS = 8, int MRG = 0, int PF = 0>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2f(
    const uint4* __restrict__ Wq13, const uint32_t* __restrict__ S13, const uint4* __restrict__ Wq2,
    const uint32_t* __restrict__ S2, const uint32_t* __restrict__ lut, const float* __restrict__ gs13,
    const float* __restrict__ gs2, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids, int nroutes,
    int start, int count, int* __restrict__ meta, int* __restrict__ sroutes, int* __restrict__ inv,
    __nv_bfloat16* inter, float* __restrict__ Y, int* cnt, int D, const float* __restrict__ tw,
    const float* __restrict__ w1, __nv_bfloat16* __restrict__ out, int M, float eps) {
  static_assert(NC % 8 == 0, "routing uses warps 0..7 x 32 ids");
  extern __shared__ __align__(128) uint4 smem[];
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_rs13[33], s_rs2[33], s_srt[256], s_wc[8][32], s_nact;
  __shared__ float s_red[(GS == 16 || MRG ? 4 : 2) * NC * 256];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  __shared__ int s_rid[4];
  __shared__ __align__(8) uint64_t ridb[4];
  __shared__ int s_inv[FN ? 256 : 1], s_tk[2];
  __shared__ float s_rt[FN ? 256 : 1], s_rg[FN ? 256 : 1];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int gen = 0;  // FN: this call's generation (see tk2f_norm); thread 0 only
  if (FN && tid == 0) asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(gen) : "l"(cnt + 7) : "memory");
  for (int i = tid; i < 256; i += (NC + 1) * 32) slut[i] = lut[i];
  if (tid == 0) {
    for (int i = 0; i < NSTG; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NC * 32); }
    for (int i = 0; i < 4; ++i) mbar_init(&ridb[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  {  // routing, as moe_tk2 GATED
    int e = -1, rank = 0;
    if (warp < 8) {
      s_wc[warp][lane] = 0;
      const int i = warp * 32 + lane;
      if (i < nroutes) {
        const int v = ids[i] - start;
        if (v >= 0 && v < count) e = v;
      }
      const unsigned m = __match_any_sync(0xffffffffu, e);
      rank = __popc(m & ((1u << lane) - 1));
      __syncwarp();
      if (e >= 0 && rank == 0) s_wc[warp][e] = __popc(m);
    }
    __syncthreads();
    if (tid < 32) {
      int c = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) c += s_wc[w][tid];
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (tid >= o) incl += v;
      }
      int run = incl - c;
      s_off[tid] = run;
      if (tid == 31) s_off[32] = incl;
#pragma unroll
      for (int w = 0; w < 8; ++w) { const int cw = s_wc[w][tid]; s_wc[w][tid] = run; run += cw; }
      int rk = 0;
      for (int j = 0; j < 32; ++j) {
        const int cj = __shfl_sync(0xffffffffu, c, j);
        rk += (cj > c) || (cj == c && j < tid);
      }
      const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
      if (tid == 0) s_nact = na;
      if (tid >= na) s_e[tid] = 0;
      __syncwarp();
      if (c > 0) s_e[rk] = tid;
    }
    __syncthreads();
    if (FN && warp < 8) s_inv[warp * 32 + lane] = e >= 0 ? s_wc[warp][e] + rank : -1;
    if (e >= 0) {
      const int i = warp * 32 + lane, pos = s_wc[warp][e] + rank;
      s_srt[pos] = i;
      if (blockIdx.x == 0) { sroutes[pos] = i; inv[i] = pos; }
    }
    if (blockIdx.x == 0 && tid < 33) {
      meta[33 + tid] = s_off[tid];
      if (tid < 32) meta[1 + tid] = s_e[tid];
      if (tid == 0) meta[0] = s_nact;
    }
  }
  __syncthreads();
  const int nact = s_nact;
  if (tid < 32) {
    const int e = s_e[tid];
    const int ng = tid < nact ? (s_off[e + 1] - s_off[e] + GS - 1) / GS : 0;
    int incl = ng;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    s_rs13[tid + 1] = incl * 256;
    s_rs2[tid + 1] = incl * 128;
    if (tid == 0) { s_rs13[0] = 0; s_rs2[0] = 0; }
  }
  __syncthreads();
  const int n13 = s_rs13[nact], n2 = s_rs2[nact];
  if constexpr (DYN) {
    const int G = gridDim.x, ns13 = max(0, n13 / G - D), ns2 = max(0, n2 / G - D);
    int st = 0, ph = 0, k = 0;
    if (warp == NC) {
      if (lane == 0) {
        int i = 0;
        tk2d_produce<512, NC, NSTG, KPS, false>(Wq13, S13, ns13, n13, cnt + 1, s_rs13, s_e, s_rid, ridb, smem, full, empty, st, ph, i, k);
        tk2d_produce<256, NC, NSTG, KPS, true>(Wq2, S2, ns2, n2, cnt + 2, s_rs2, s_e, s_rid, ridb, smem, full, empty, st, ph, i, k);
      }
      return;
    }
    int rb = 0;
    tk2d_consume<512, NC, NSTG, true, KPS>(slut, gs13, X, inter, cnt, n13, ns13, s_rid, ridb, s_rs13, s_e, s_off, s_srt, s_red, smem,
                                          full, empty, st, ph, rb, k);
    tk2d_consume<256, NC, NSTG, false, KPS>(slut, gs2, inter, Y, cnt, n13, ns2, s_rid, ridb, s_rs2, s_e, s_off, s_srt, s_red, smem,
                                           full, empty, st, ph, rb, k);
    return;
  }
  int a0 = (int)((long long)n13 * blockIdx.x / gridDim.x), a1 = (int)((long long)n13 * (blockIdx.x + 1) / gridDim.x);
  int b0 = (int)((long long)n2 * blockIdx.x / gridDim.x), b1 = (int)((long long)n2 * (blockIdx.x + 1) / gridDim.x);
  if (GS == 8 && !DYN && D > 0) {  // s12 weighted split (wE = D / 8 for extra-group rows), see tk2w_bound
    __shared__ int s_ab[4];
    if (warp == 0) {
      const int v0 = tk2w_bound(blockIdx.x, nact, D, 256, s_e, s_off), v1 = tk2w_bound(blockIdx.x + 1, nact, D, 256, s_e, s_off);
      const int v2 = tk2w_bound(blockIdx.x, nact, D, 128, s_e, s_off), v3 = tk2w_bound(blockIdx.x + 1, nact, D, 128, s_e, s_off);
      if (lane == 0) { s_ab[0] = v0; s_ab[1] = v1; s_ab[2] = v2; s_ab[3] = v3; }
    }
    __syncthreads();
    a0 = s_ab[0]; a1 = s_ab[1]; b0 = s_ab[2]; b1 = s_ab[3];
  }
  if (GS == 16 && D > 8) {  // GS16: 2-group rows cost D/8 of a 1-group row: split the rows by weight, not count
    __shared__ int s_gb[4];
    if (TK2G_PAR) {  // n4s5: warp 0, lane-parallel (tk2g_bound)
#if TK2G_PAR4
      if (warp < 4) {  // n17: the 4 bounds on 4 warps
        const int v = tk2g_bound(blockIdx.x + (warp & 1), nact, D, (warp & 2) ? 128 : 256, s_e, s_off);
        if (lane == 0) s_gb[warp] = v;
      }
#else
      if (warp == 0) {
        const int v0 = tk2g_bound(blockIdx.x, nact, D, 256, s_e, s_off), v1 = tk2g_bound(blockIdx.x + 1, nact, D, 256, s_e, s_off);
        const int v2 = tk2g_bound(blockIdx.x, nact, D, 128, s_e, s_off), v3 = tk2g_bound(blockIdx.x + 1, nact, D, 128, s_e, s_off);
        if (lane == 0) { s_gb[0] = v0; s_gb[1] = v1; s_gb[2] = v2; s_gb[3] = v3; }
      }
#endif
      __syncthreads();
      a0 = s_gb[0]; a1 = s_gb[1]; b0 = s_gb[2]; b1 = s_gb[3];
    } else {
      a0 = tk2g_split(blockIdx.x, nact, D, 256, s_e, s_off);
      a1 = tk2g_split(blockIdx.x + 1, nact, D, 256, s_e, s_off);
      b0 = tk2g_split(blockIdx.x, nact, D, 128, s_e, s_off);
      b1 = tk2g_split(blockIdx.x + 1, nact, D, 128, s_e, s_off);
    }
  }
  if (PF & 32) {  // moe n8 BAL: gemm13's extra rows (n13 % G) on the LAST CTAs, gemm2's (n2 % G) on the FIRST ones
    const int G = gridDim.x, b = blockIdx.x, q1 = n13 / G, e1 = G - n13 % G, q2 = n2 / G, e2 = n2 % G;
    a0 = b * q1 + max(0, b - e1); a1 = a0 + q1 + (b >= e1);
    b0 = b * q2 + min(b, e2); b1 = b0 + q2 + (b < e2);
  }
  int st = 0, ph = 0;
  if (warp == NC) {
    if (lane == 0) {
      int i = 0;
      constexpr bool SQ = (PF & 512) != 0;  // layer s68 (moe n14): single-copy stages (Wq13 / Wq2 = the _sq_pack buffers)
      constexpr bool S6 = (PF & 1024) != 0;  // layer s73 (moe n20): 6-bit scale codes (the _s6_pack buffers)
      tk2f_produce<512, NC, NSTG, KPS, SQ, S6>(Wq13, S13, a0, a1, s_rs13, s_e, smem, full, empty, st, ph, i);
      tk2f_produce<256, NC, NSTG, KPS, SQ, S6>(Wq2, S2, b0, b1, s_rs2, s_e, smem, full, empty, st, ph, i);
    }
    return;
  }
  float rt = 0.f, rg = 0.f;  // FN: route tid's topk weight and gs2 * 256 (0: not local), used by tk2f_norm
  if (FN && tid < nroutes) {
    const int e = ids[tid] - start;
    if (e >= 0 && e < count) {
      rt = tw[tid];
      rg = gs2[e] * 256.0f;
    }
  }
  int rb = 0;
  if constexpr (MRG) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // layer s58: x is the PDL predecessor's output (TK2FX/TK2FJ)
    tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0, (PF & 1024) != 0>(slut, gs13, gs2, X, inter, Y, cnt, n13, a0, a1, b0, b1, s_rs13, s_rs2, s_e, s_off,
                                        s_srt, s_red, smem, smem + NSTG * NC * KPS * 72, full, empty, st, ph, rb);
    return;
  }
  if constexpr (GS == 16) {
    uint4* s_x2 = smem + NSTG * NC * KPS * 72;  // after the ring: 8 KB per consumer warp
    tk2g_consume<512, NC, NSTG, true, KPS, SA>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem,
                                               s_x2, full, empty, st, ph, rb);
    tk2g_consume<256, NC, NSTG, false, KPS, SA>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem,
                                                s_x2, full, empty, st, ph, rb);
    return;
  }
  tk2f_consume<512, NC, NSTG, true, KPS, SA, PF>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem, full,
                                    empty, st, ph, rb);
  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, s_red, smem, full,
                                     empty, st, ph, rb);
  if constexpr (FN) {
    s_rt[tid] = rt;
    s_rg[tid] = rg;
    tk2f_norm<NC>(cnt, M, s_inv, s_rt, s_rg, w1, Y, eps, out, s_red, s_tk, gen);
  }
}

template <int NC, int NSTG, int KPS, bool DYN = false, bool FN = false, bool SA = false, int GS = 8, int MRG = 0, int PF = 0>
static int64_t tk2f_go(int D, int nr, const void* w13, const void* s13, const void* w2, const void* s2, const void* lut,
                       const void* gs13, const void* gs2, const void* x, const void* ids, int start, int count,
                       void* meta, void* sr, void* inv, void* inter, void* y, void* cnt, const void* tw = nullptr,
                       const void* w1 = nullptr, void* out = nullptr, float eps = 0.f) {
  constexpr int SMEM = NSTG * NC * KPS * 1152 + (GS == 16 ? NC * 8 * 2 * 32 * 16 : 0);
  static int grid = 0;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  return launch_k(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, grid, (NC + 1) * 32, SMEM, st, false,
      (const uint4*)w13, (const uint32_t*)s13, (const uint4*)w2, (const uint32_t*)s2, (const uint32_t*)lut,
      (const float*)gs13, (const float*)gs2, (const __nv_bfloat16*)x, (const int*)ids, nr, start, count, (int*)meta,
      (int*)sr, (int*)inv, (__nv_bfloat16*)inter, (float*)y, (int*)cnt, D, (const float*)tw, (const float*)w1,
      (__nv_bfloat16*)out, nr / 8, eps);
}

// s9: moe_tk2f with the post-expert norm fused (FN): writes out [M, H] bf16 directly; no moe_norm2 after it.
int64_t moe_tk2fn_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor w2_, torch::Tensor s2_,
                         torch::Tensor lut_, torch::Tensor gs13_, torch::Tensor gs2_, torch::Tensor x_,
                         torch::Tensor ids_, int64_t start, int64_t count, torch::Tensor meta_, torch::Tensor sr_,
                         torch::Tensor inv_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor cnt_, int64_t cfg,
                         torch::Tensor tw_, torch::Tensor w1_, torch::Tensor out_, double eps) {
  const int nr = (int)ids_.numel();
  if (nr > 256) return -2;
#define TK2FN_ARGS 0, nr, PTR(w13_), PTR(s13_), PTR(w2_), PTR(s2_), PTR(lut_), PTR(gs13_), PTR(gs2_), PTR(x_), \
    PTR(ids_), (int)start, (int)count, meta_.data_ptr(), sr_.data_ptr(), inv_.data_ptr(), inter_.data_ptr(), \
    y_.data_ptr(), cnt_.data_ptr(), PTR(tw_), PTR(w1_), out_.data_ptr(), (float)eps
  if (cfg == 402) return tk2f_go<8, 2, 4, false, true>(TK2FN_ARGS);
  return tk2f_go<8, 4, 2, false, true>(TK2FN_ARGS);
#undef TK2FN_ARGS
}


// cfg = KPS * 100 + NSTG: 108 (s7 default), 204 (s8), 208, 402, 403, 206; + 1000 * (D + 1): DYN tail (204/402 only)
int64_t moe_tk2f_launch(torch::Tensor w13_, torch::Tensor s13_, torch::Tensor w2_, torch::Tensor s2_,
                        torch::Tensor lut_, torch::Tensor gs13_, torch::Tensor gs2_, torch::Tensor x_,
                        torch::Tensor ids_, int64_t start, int64_t count, torch::Tensor meta_, torch::Tensor sr_,
                        torch::Tensor inv_, torch::Tensor inter_, torch::Tensor y_, torch::Tensor cnt_, int64_t cfg) {
  const int nr = (int)ids_.numel();
  if (nr > 256) return -2;
#define TK2F_ARGS0 nr, PTR(w13_), PTR(s13_), PTR(w2_), PTR(s2_), PTR(lut_), PTR(gs13_), PTR(gs2_), PTR(x_), PTR(ids_), \
    (int)start, (int)count, meta_.data_ptr(), sr_.data_ptr(), inv_.data_ptr(), inter_.data_ptr(), y_.data_ptr(), \
    cnt_.data_ptr()
  const bool s6 = cfg >= 20000000;  // layer s73 (moe n20 S6): + 20000000 = SQ with 6-bit scale codes (_s6_pack buffers)
  const bool sq = cfg >= 10000000;  // layer s68 (moe n14 SQ): + 10000000 = single-copy stages (w13_ / w2_ = _sq_pack buffers)
  cfg %= 10000000;
  const bool bal = cfg >= 1000000;  // moe n8: + 1000000 = BAL (PF | 32; SA pf 4 / 5 only)
  cfg %= 1000000;
  if (s6) {  // SA 84204 (+ BAL), tk2h GS16 + PF4
    if (cfg >= 140000 && cfg < 150000 && cfg % 1000 == 4)
      return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4 | 512 | 1024>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
    TORCH_CHECK(cfg == 184204, "moe_tk2f S6: cfg ", cfg);
    if (bal) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36 | 512 | 1024>(0, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 512 | 1024>(0, TK2F_ARGS0);
  }
  if (sq) {  // SA 84204 only (+ BAL): d1 slot, T = 16 / 32
    if (cfg >= 140000 && cfg < 150000 && cfg % 1000 == 4)  // moe n16: tk2h GS16 + PF4 on SQ stages
      return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4 | 512>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
#if TK2F_DIAG
    // layer s81 DIAG (tuner timing only, never selectable): 184402/184403 = KPS 4 SQ stages (36 KB, _sq_pack nq 2 buffers),
    // NSTG 2/3; 184214 / 184412 / 184413 = the same (and 204) with no consumer math (PF 256: wrong output)
    if (cfg == 184402) return tk2f_go<8, 2, 4, false, false, true, 8, 0, 4 | 512>(0, TK2F_ARGS0);
    if (cfg == 184403) return tk2f_go<8, 3, 4, false, false, true, 8, 0, 4 | 512>(0, TK2F_ARGS0);
    if (cfg == 184214) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 512 | 256>(0, TK2F_ARGS0);
    if (cfg == 184412) return tk2f_go<8, 2, 4, false, false, true, 8, 0, 4 | 512 | 256>(0, TK2F_ARGS0);
    if (cfg == 184413) return tk2f_go<8, 3, 4, false, false, true, 8, 0, 4 | 512 | 256>(0, TK2F_ARGS0);
#endif
    TORCH_CHECK(cfg == 184204, "moe_tk2f SQ: cfg ", cfg);
    if (bal) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36 | 512>(0, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 512>(0, TK2F_ARGS0);
  }
  const bool sa = cfg >= 100000;  // + 100000: SA (arithmetic scale decode; prepare() checked the scales)
  cfg %= 100000;
  if (cfg > 80000 && cfg < 100000 && cfg % 1000 == 204) {  // s13 PF = cfg / 1000 % 10: 1 = L1 prefetch of the next segment's
    // x one row ahead, 2 = gemm2's counter spin only at the CTA's first gemm2 row, 4 = y32 in route slot order, pre-scaled
    // by gs2 * 256 (then moe_norm3 instead of moe_norm2), 8 = 2 accumulator chains per tile; built: 1, 2, 3, 4, 7, 12
    const int pf = (int)(cfg - 80000) / 1000;
    if (bal && sa && pf == 4) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36>(0, TK2F_ARGS0);
#ifdef MOE_BAL_NU8
    if (bal && sa && pf == 5) {
      TORCH_CHECK(nr == 64 && y_.size(0) >= 65 && g_ssw1 != nullptr, "tk2f PF 52: shape");
      return tk2f_go<8, 4, 2, false, false, true, 8, 0, 52>(0, TK2F_ARGS0, nullptr, g_ssw1);
    }
#endif
    if (pf == 12) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 12>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 12>(0, TK2F_ARGS0);
    if (pf == 4) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 4>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 4>(0, TK2F_ARGS0);
    if (pf == 7) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 7>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 7>(0, TK2F_ARGS0);
    if (pf == 1) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 1>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 1>(0, TK2F_ARGS0);
    if (pf == 2) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 2>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 2>(0, TK2F_ARGS0);
    return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 3>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 0, 3>(0, TK2F_ARGS0);
  }
  if (sa && cfg >= 40000 && cfg < 50000 && cfg % 1000 == 4)  // layer s58: moe n6 tk2h GS16 + PF4 (h16w*p4), the only tk2h built
    return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
#if TK2_EXP
  if (cfg >= 30000) {  // s12 merged-phase consumer (tk2h): + 30000 GS 8, + 40000 GS 16 (cfg / 1000 % 10 = 2-group weight),
    // + 50000 / + 60000: the same with per-segment gemm13 -> gemm2 counters (PSC)
    const int kind = (int)(cfg / 10000);
    const int wt = 8 + (int)(cfg / 1000) % 10;
    if (cfg % 1000 == 4) {  // n6: + 4 = slot-ordered pre-scaled y32 (tk2f PF 4) -> moe_norm3
      if (kind == 3) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 1, 4>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 1, 4>(0, TK2F_ARGS0);
      return sa ? tk2f_go<8, 4, 2, false, false, true, 16, 1, 4>(wt, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 16, 1, 4>(wt, TK2F_ARGS0);
    }
    if (kind == 3) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 1>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 1>(0, TK2F_ARGS0);
    if (kind == 4) return sa ? tk2f_go<8, 4, 2, false, false, true, 16, 1>(wt, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 16, 1>(wt, TK2F_ARGS0);
    if (kind == 7) {  // + 70000: main consumer (tk2f_consume), weighted static split, wE = cfg / 1000 % 10
      const int wE = (int)(cfg / 1000) % 10;
      return sa ? tk2f_go<8, 4, 2, false, false, true>(wE, TK2F_ARGS0) : tk2f_go<8, 4, 2>(wE, TK2F_ARGS0);
    }
    if (kind == 5) return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 2>(0, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 8, 2>(0, TK2F_ARGS0);
    return sa ? tk2f_go<8, 4, 2, false, false, true, 16, 2>(wt, TK2F_ARGS0) : tk2f_go<8, 4, 2, false, false, false, 16, 2>(wt, TK2F_ARGS0);
  }
#endif
  if (cfg >= 20000) {  // + 20000: 16 consumer warps (NC 16), cfg % 1000 = KPS * 100 + NSTG (static rows)
    const int c = (int)(cfg % 1000);
    if (sa) {
      if (c == 202) return tk2f_go<16, 2, 2, false, false, true>(0, TK2F_ARGS0);
      if (c == 104) return tk2f_go<16, 4, 1, false, false, true>(0, TK2F_ARGS0);
      if (c == 402) return tk2f_go<16, 2, 4, false, false, true>(0, TK2F_ARGS0);
      if (c == 103) return tk2f_go<16, 3, 1, false, false, true>(0, TK2F_ARGS0);
      return tk2f_go<16, 4, 2, false, false, true>(0, TK2F_ARGS0);
    }
    return tk2f_go<16, 4, 2>(0, TK2F_ARGS0);
  }
  const bool g16 = cfg >= 10000;  // + 10000: GS16 (16-token segments; static rows, KPS 2, NSTG 4)
  cfg %= 10000;
  if (g16) {  // (cfg / 1000) % 10 = extra weight of a 2-group row in 1/8 (see tk2g_split)
    const int wt = 8 + (int)(cfg / 1000) % 10;
    if (sa) return tk2f_go<8, 4, 2, false, false, true, 16>(wt, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, false, 16>(wt, TK2F_ARGS0);
  }
  const int D = (int)(cfg / 1000) - 1;  // cfg >= 1000: DYN with D = cfg / 1000 - 1
  cfg %= 1000;
  if (D >= 0) {
    if (cfg == 204) return tk2f_go<8, 4, 2, true>(D, TK2F_ARGS0);
    return tk2f_go<8, 2, 4, true>(D, TK2F_ARGS0);
  }
#define TK2F_ARGS 0, nr, PTR(w13_), PTR(s13_), PTR(w2_), PTR(s2_), PTR(lut_), PTR(gs13_), PTR(gs2_), PTR(x_), PTR(ids_), \
    (int)start, (int)count, meta_.data_ptr(), sr_.data_ptr(), inv_.data_ptr(), inter_.data_ptr(), y_.data_ptr(), \
    cnt_.data_ptr()
  if (sa && D < 0) {
    if (cfg == 402) return tk2f_go<8, 2, 4, false, false, true>(TK2F_ARGS);
    if (TK2F_RING_SA) {  // s12: deeper rings with SA, for the H200 diag (MOE_TK2F_RING_SA build flag)
      if (cfg == 208) return tk2f_go<8, 8, 2, false, false, true>(TK2F_ARGS);
      if (cfg == 206) return tk2f_go<8, 6, 2, false, false, true>(TK2F_ARGS);
      if (cfg == 403) return tk2f_go<8, 3, 4, false, false, true>(TK2F_ARGS);
      if (cfg == 108) return tk2f_go<8, 8, 1, false, false, true>(TK2F_ARGS);
    }
    return tk2f_go<8, 4, 2, false, false, true>(TK2F_ARGS);
  }
  if (cfg == 204) return tk2f_go<8, 4, 2>(TK2F_ARGS);
  if (cfg == 208) return tk2f_go<8, 8, 2>(TK2F_ARGS);
  if (cfg == 402) return tk2f_go<8, 2, 4>(TK2F_ARGS);
  if (cfg == 403) return tk2f_go<8, 3, 4>(TK2F_ARGS);
  if (cfg == 206) return tk2f_go<8, 6, 2>(TK2F_ARGS);
  return tk2f_go<8, 8, 1>(TK2F_ARGS);
#undef TK2F_ARGS
#undef TK2F_ARGS0
}

int64_t moe_tk2_launch(int64_t gated, int64_t nstg, torch::Tensor wq_, torch::Tensor s2_, torch::Tensor lut_,
                       torch::Tensor gs_, torch::Tensor x_, torch::Tensor ids_, int64_t start, int64_t count,
                       torch::Tensor meta_, torch::Tensor sr_, torch::Tensor inv_, torch::Tensor out_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *s2 = PTR(s2_), *lut = PTR(lut_), *gs = PTR(gs_), *x = PTR(x_), *ids = PTR(ids_);
  void *meta = meta_.data_ptr(), *sr = sr_.data_ptr(), *inv = inv_.data_ptr(), *out = out_.data_ptr();
  const int nr = (int)ids_.numel();
  if (nr > 256) return -2;
#define TK2(NS)                                                                                              \
  if (nstg == NS) {                                                                                          \
    if (gated) return launch_tk2<512, 8, NS, true>(wq, s2, lut, gs, x, ids, nr, (int)start, (int)count, meta, sr, inv, out, st); \
    return launch_tk2<256, 8, NS, false>(wq, s2, lut, gs, x, ids, nr, (int)start, (int)count, meta, sr, inv, out, st); \
  }
  TK2(8) TK2(10) TK2(12)
  return -1;
}
int64_t moe_sk_launch(int64_t gated, int64_t cfg, torch::Tensor wq_, torch::Tensor s2_, torch::Tensor lut_,
                      torch::Tensor x_, torch::Tensor meta_, torch::Tensor sr_, torch::Tensor acc_) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const void *wq = PTR(wq_), *s2 = PTR(s2_), *lut = PTR(lut_), *x = PTR(x_), *meta = PTR(meta_), *sr = PTR(sr_);
  void* acc = acc_.data_ptr();
  if (gated) { SK_DISPATCH(512, true) } else { SK_DISPATCH(256, false) }
}
int64_t moe_route2_launch(torch::Tensor ids, int64_t start, int64_t count, torch::Tensor meta, torch::Tensor sroutes,
                          torch::Tensor inv, torch::Tensor acc13, int64_t nzero) {
  moe_route2<<<1 + nzero, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int*)ids.data_ptr(), (int)ids.numel(), (int)start, (int)count, (int*)meta.data_ptr(),
      (int*)sroutes.data_ptr(), (int*)inv.data_ptr(), (float4*)acc13.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_act_launch(torch::Tensor acc13, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor ids, torch::Tensor gs,
                       int64_t start, torch::Tensor inter, torch::Tensor acc2) {
  moe_act<<<(int)inter.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const float4*)acc13.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(),
      (const int*)ids.data_ptr(), (const float*)gs.data_ptr(), (int)start, (__nv_bfloat16*)inter.data_ptr(),
      (float4*)acc2.data_ptr());
  return (int64_t)cudaGetLastError();
}
int64_t moe_norm2_launch(torch::Tensor acc2, torch::Tensor inv, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1,
                         torch::Tensor gs, int64_t start, int64_t count, double eps, torch::Tensor out, torch::Tensor zc) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm2<false>, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 2) != 0,
      (const float*)acc2.data_ptr(), (const int*)inv.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (const float*)gs.data_ptr(), (int)start, (int)count, (float)eps,
      (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr());
}
// s13: after moe_tk2f with PF & 4 (y32 rows = route slots, pre-scaled by gs2 * 256)
int64_t moe_norm3_launch(torch::Tensor y32, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start,
                         int64_t count, double eps, torch::Tensor out, torch::Tensor zc) {
  const int M = (int)out.size(0);
  return launch_k(moe_norm2<true>, M, 512, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 2) != 0,
      (const float*)y32.data_ptr(), (const int*)nullptr, (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (const float*)nullptr, (int)start, (int)count, (float)eps,
      (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr());
}
"""


def _moe__gen_wgmma():
    out = ["template <int N> __device__ __forceinline__ void wgmma_bf16(float* d, const uint32_t* a, uint64_t desc);"]
    for n in list(range(8, 104, 8)) + [128]:
        na = n // 2
        dl = ", ".join(f"%{i}" for i in range(na))
        al = ", ".join(f"%{na + i}" for i in range(4))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(na))
        out.append(
            f"template <> __device__ __forceinline__ void wgmma_bf16<{n}>(float* d, const uint32_t* a, uint64_t desc) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n"\n'
            f'    "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{{al}}}, %{na + 4}, p, 1, 1, 0;\\n}}\\n"\n'
            f'    : {outs}\n'
            f'    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));\n}}')
    return "\n".join(out)


_moe_CUDA_SRC = _moe_CUDA_SRC.replace("//@@WGMMA@@", _moe__gen_wgmma())

_moe__LIB = None
_moe__CPP = ("int64_t moe_gemm13(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gemm2(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor);"
        "int64_t moe_norme_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, torch::Tensor, torch::Tensor);"
        "int64_t moe_norm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor);"
        "int64_t moe_sk_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tk2_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tb_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gather_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_twf_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_twm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_tw_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route2_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t);"
        "int64_t moe_act_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
        "torch::Tensor, torch::Tensor);"
        "int64_t moe_norm2_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, double, torch::Tensor, torch::Tensor);"
        "int64_t moe_norm3_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor, torch::Tensor);"
        "int64_t moe_tk2f_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_tk2fn_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor, "
        "torch::Tensor, double);"
        "int64_t moe_twq_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
        "int64_t moe_set_pdl(int64_t, int64_t);"
        "int64_t moe_set_blk(int64_t);"
        "int64_t moe_set_sa(int64_t);"
        "int64_t moe_set_hot(int64_t);"
        "int64_t moe_hot_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);")


def _moe__lib():
    global _moe__LIB
    if _moe__LIB is None:
        from torch.utils.cpp_extension import load_inline
        h = hashlib.sha256(_moe_CUDA_SRC.encode()).hexdigest()[:12]
        _moe__LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_moe__CPP, cuda_sources=_moe_CUDA_SRC,
                           functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch", "moe_sk_launch", "moe_tk2_launch", "moe_tk2f_launch", "moe_tk2fn_launch", "moe_tb_launch",
                                      "moe_gather_launch", "moe_tw_launch", "moe_twf_launch", "moe_twm_launch", "moe_twq_launch",
                                      "moe_route2_launch", "moe_act_launch", "moe_norm2_launch", "moe_norm3_launch", "moe_set_pdl", "moe_set_blk", "moe_set_sa", "moe_set_hot", "moe_hot_launch"],
                           extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"],
                           extra_cflags=["-O3"], verbose=False)
    return _moe__LIB


# ---- prepare-time repacking (lossless: same bits, new order) ----

def _moe__unpack_codes(packed):
    """uint8 [..., K/2] -> uint8 codes [..., K] (element 2i = low nibble)."""
    lo = packed & 0x0F
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# bit positions (lo element, hi element) of (sign, e1, e0, m) for the 4 outputs, see dq().
_moe__POS = [
    ((15, 8, 7, 6), (31, 24, 23, 22)),
    ((5, 30, 29, 28), (21, 14, 13, 12)),  # rotl 10
    ((2, 27, 26, 25), (18, 11, 10, 9)),   # rotl 13
    ((1, 0, 4, 3), (17, 16, 20, 19)),     # (q & 0x001B001B) * (2^3 + 2^8 + 2^14)
]


def _moe__pack_weight(packed):
    """One expert's uint8 [N, K/2] -> int32 [N/16, K/64, 32 lanes, 4] in A-fragment order."""
    N = packed.shape[0]
    K = packed.shape[1] * 2
    c = _moe__unpack_codes(packed).to(torch.int32)  # [N, K]
    # row = rt*16 + h*8 + g ; k = kc*64 + t*16 + j*4 + sb*2 + lohi
    c = c.view(N // 16, 2, 8, K // 64, 4, 4, 2, 2)  # rt h g kc t j sb lohi
    c = c.permute(0, 3, 2, 4, 5, 6, 1, 7)          # rt kc g t j sb h lohi
    c = c.reshape(N // 16, K // 64, 32, 4, 8)       # slot = (sb*2 + h)*2 + lohi = r*2 + lohi
    word = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for r in range(4):
        for lohi in range(2):
            code = c[..., r * 2 + lohi].to(torch.int64)
            ps, pe1, pe0, pm = _moe__POS[r][lohi]
            word |= ((code >> 3) & 1) << ps
            word |= ((code >> 2) & 1) << pe1
            word |= ((code >> 1) & 1) << pe0
            word |= (code & 1) << pm
    return (word - ((word >> 31) << 32)).to(torch.int32)  # same 32 bits, as int32


def _moe__pack_scales(sf):
    """One expert's e4m3 [N, K/16] -> uint8 [N/16, K/512, 32 lanes, 16]."""
    N = sf.shape[0]
    K = sf.shape[1] * 16
    b = sf.view(torch.uint8).view(N // 16, 2, 8, K // 512, 8, 4)  # rt h g kg kcl t
    b = b.permute(0, 3, 2, 5, 4, 1)                                # rt kg g t kcl h
    return b.reshape(N // 16, K // 512, 32, 16).contiguous()


def _moe__sq_pack(wq, s2, nq=4):
    """layer s68 (moe n14 SQ): wq int32 [E, NRT, 64, 32, 4] + s2 int32 [E, NRT/2, 64, 32] -> int32 [E, NRT/2, nq, 2 * W + S]: per
    (row-tile pair lr, k quarter q) the tk2f stage image {tile lr kc q*16.., tile lr + NRT/2 same kc, scales}."""
    E, nrt = wq.shape[0], wq.shape[1]
    w = wq.reshape(E, 2, nrt // 2, nq, -1)
    sc = s2.reshape(E, nrt // 2, nq, -1)
    return torch.cat((w[:, 0], w[:, 1], sc), dim=-1).contiguous()


def _moe__s6_pack(wq, s2, nq=4):
    """layer s73 (moe n20 S6): as _sq_pack, but the stage's scales as 6-bit codes c = e4m3 byte - 80 (caller checked
    80 <= byte <= 0x7E): per (lr, q) {tile lr, tile lr + NRT/2, nibbles int32 [8 warps][32 lanes], high bits int16
    [8][32]}; lane word bits 4k / 2k hold code k = 4 h + r. 17920 B/stage (lossless)."""
    E, nrt = wq.shape[0], wq.shape[1]
    w = wq.reshape(E, 2, nrt // 2, nq, -1)
    c = s2.contiguous().view(torch.uint8).reshape(E, nrt // 2, nq, 2, 8, 32, 4).long() - 80  # E lr q h warp lane r
    c = c.permute(0, 1, 2, 4, 5, 3, 6).reshape(E, nrt // 2, nq, 8, 32, 8)                     # ... warp lane k
    k = torch.arange(8, device=c.device)
    nib = ((c & 15) << (4 * k)).sum(-1)
    hi = ((c >> 4) << (2 * k)).sum(-1)
    nib = (nib - ((nib >> 31) << 32)).to(torch.int32).reshape(E, nrt // 2, nq, 256)
    hi = (hi - ((hi >> 15) << 16)).to(torch.int16).reshape(E, nrt // 2, nq, 256).view(torch.int32)
    return torch.cat((w[:, 0], w[:, 1], nib, hi), dim=-1).contiguous()


def _moe__pair_scales(ps):
    """_pack_scales output [NRT, K/512, 32, 16] -> uint8 [NRT/2, K/64, 32 lanes, 4]: per (row-tile pair rp, k chunk)
    each lane's 4 bytes (tile rp rows g, g+8, tile rp + NRT/2 rows g, g+8)."""
    n = ps.shape[0]
    v = ps.view(n, ps.shape[1], 32, 8, 2).permute(0, 1, 3, 2, 4).reshape(n, ps.shape[1] * 8, 32, 2)
    return torch.cat((v[: n // 2], v[n // 2:]), dim=-1).contiguous()


def _moe__tb_pack(wq, ws, gated, rt):
    """moe_tb layout: weights [E][rb][kc][warp][RT][lane] uint4, scales [E][rb][kc][warp][lane][RT] u16.
    Gated: warp w of block rb owns gate tiles rb*8*H + w*H + (0..H) and the matching up tiles (H = RT/2)."""
    E, nrt = wq.shape[0], wq.shape[1]
    nrb = nrt // (8 * rt)
    rb = torch.arange(nrb).view(-1, 1, 1)
    w = torch.arange(8).view(1, -1, 1)
    r = torch.arange(rt).view(1, 1, -1)
    if gated:
        h = rt // 2
        tile = (r >= h).long() * (nrt // 2) + rb * 8 * h + w * h + r % h
    else:
        tile = rb * 8 * rt + w * rt + r
    tile = tile.to(wq.device)
    W, S = [], []
    for e in range(E):
        we = wq[e].reshape(nrt, 64, 128)[tile]                                   # rb w r kc lane*4
        W.append(we.permute(0, 3, 1, 2, 4).contiguous())
        se = ws[e].view(torch.int16).reshape(nrt, 8, 32, 8).permute(0, 1, 3, 2).reshape(nrt, 64, 32)[tile]
        S.append(se.permute(0, 3, 1, 4, 2).contiguous())                         # rb kc w lane r
    return torch.stack(W), torch.stack(S)


def _moe__scale_lut(device):
    """e4m3 byte -> bf16x2 of (value * 2^118), both halves."""
    v = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float() * (2.0 ** 118)
    bits = v.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    word = bits | (bits << 16)
    return word.to(device)


def _moe_prepare(weights):
    lib = _moe__lib()
    dev = weights["w13"].device
    assert weights["w13"].shape[1:] == (2 * _moe_I, _moe_H // 2) and weights["w2"].shape[1:] == (_moe_H, _moe_I // 2)
    st = {"lib": lib, "start": 0, "count": _moe_LOCAL}
    w13q, w13s, w2q, w2s = [], [], [], []
    for e in range(_moe_LOCAL):
        w13q.append(_moe__pack_weight(weights["w13"][e]))
        w13s.append(_moe__pack_scales(weights["w13_sf"][e]))
        w2q.append(_moe__pack_weight(weights["w2"][e]))
        w2s.append(_moe__pack_scales(weights["w2_sf"][e]))
    st["w13q"] = torch.stack(w13q)
    st["w13s"] = torch.stack(w13s)
    st["w13s2"] = torch.stack([_moe__pair_scales(x) for x in w13s]).view(torch.int32)
    st["w2s2"] = torch.stack([_moe__pair_scales(x) for x in w2s]).view(torch.int32)
    st["w2q"] = torch.stack(w2q)
    st["w2s"] = torch.stack(w2s)
    del w13q, w13s, w2q, w2s
    # layer s73 (moe n20 S6): needs every block scale byte in [80, 0x7E] (positive normal, value >= 8), else 8-bit SQ
    s6 = _moe_TK2F_SQ and _moe_TK2F_S6 and all(bool(((b >= 80) & (b <= 0x7E)).all()) for b in (
        st["w13s2"].view(torch.uint8), st["w2s2"].view(torch.uint8)))
    if s6:
        st["w13s6"] = _moe__s6_pack(st["w13q"], st["w13s2"])
        st["w2s6"] = _moe__s6_pack(st["w2q"], st["w2s2"])
        for a_, b_ in (("w13s6", "w13s2"), ("w2s6", "w2s2")):  # lossless round trip
            assert torch.equal(_moe__s6_unpack(st[a_]), st[b_].reshape(st[b_].shape[:4])), a_
    if _moe_TK2F_SQ and (not s6 or _moe_TK2F_S6_KEEPSQ):  # layer s68 (moe n14): tk2f single-copy stage images (lossless repack of w13q/w2q + w13s2/w2s2)
        st["w13sq"] = _moe__sq_pack(st["w13q"], st["w13s2"])
        st["w2sq"] = _moe__sq_pack(st["w2q"], st["w2s2"])
    if _moe_TK2F_DIAG:  # layer s81: KPS 4 (36 KB) single-copy stage images for the DIAG probes
        st["w13sq4"] = _moe__sq_pack(st["w13q"], st["w13s2"], 2)
        st["w2sq4"] = _moe__sq_pack(st["w2q"], st["w2s2"], 2)
    if _moe_TW_CFG:
        rt = 2 if _moe_TW_CFG >= 1000 else _moe_TW_CFG // 100
        if rt not in _moe_TB_PACK:
            st[f"w13b{rt}"], st[f"w13sb{rt}"] = _moe__tb_pack(st["w13q"], st["w13s"], True, rt)
            st[f"w2b{rt}"], st[f"w2sb{rt}"] = _moe__tb_pack(st["w2q"], st["w2s"], False, rt)
    for rt in sorted(set(_moe_TB_PACK) | ({2} if _moe_TWF_MIX or _moe_TWF_RT == 2 or 2 in _moe_TWF_RT_M.values() else set())):
        st[f"w13b{rt}"], st[f"w13sb{rt}"] = _moe__tb_pack(st["w13q"], st["w13s"], True, rt)
        st[f"w2b{rt}"], st[f"w2sb{rt}"] = _moe__tb_pack(st["w2q"], st["w2s"], False, rt)
    st["gs13"] = weights["w13_gs"][:_moe_LOCAL].float().contiguous()
    st["gs2"] = weights["w2_gs"][:_moe_LOCAL].float().contiguous()
    st["lut"] = _moe__scale_lut(dev)
    # SA decode (s9) needs every block scale positive and normal (e4m3 exponent field 1..15, not NaN)
    sfb = [weights[k][:_moe_LOCAL].contiguous().view(torch.uint8) for k in ("w13_sf", "w2_sf")]
    st["sa"] = all(bool((((b & 0x80) == 0) & ((b & 0x78) != 0) & (b != 0x7F)).all()) for b in sfb)
    del sfb
    st["cnt"] = torch.zeros(128, dtype=torch.int32, device=dev)  # moe_tk2f counters (zero between calls)
    st["tk"] = torch.zeros(1024, dtype=torch.int32, device=dev)  # moe_twf split-item tickets (self-resetting)
    st["w1"] = (1.0 + weights["post_norm_w"].float()).contiguous()
    torch.cuda.synchronize()
    global _moe__DIAG_DONE
    if _moe_DIAG and not _moe__DIAG_DONE:
        _moe__DIAG_DONE = True
        _moe__diag(st)
    return st


_moe__DIAG_DONE = False


# (R, G) per batch size for gemm13 / gemm2: R row tiles (x16 rows) per item, G token groups of 8 per pass.
def _moe__plan(M):
    if M == 1:
        return 10, 10
    if M <= 32:
        return 11, 11
    return 118, 118


_moe_PLAN_OVERRIDE = None
# PDL / row order are set once in the library (g_pdl = 4: big-M kernels only, g_blk = 0); moe_set_pdl / moe_set_blk
# change them for experiments. Judge s5: decode PDL (gemm2 or norm2) and tk2_map blk 2 at bs32 lose on the H200.
_moe_TB_RT = 0  # big-M moe_tb on (1) or off (0: old moe_big path)
_moe_TB_CFG = 48  # RT*10 + stages
_moe_TW_CFG = 418  # big-M wgmma kernel moe_tw: RT*100 + KG*10 + stages (0: off)
_moe_TWF_RT = 4  # s11: 2 = moe_twf with RT 2 / 64-token passes (prepare then packs w13b2/w2b2): judge 7dda218c bs128 224.5,
# bs256 267.0 us (RT 4: ~200 / ~247): worse when experts have <= 32 tokens; H100 layer v16 -5, v32 -7 us (H100 misleads)
_moe_TWF_MIX = 1  # s11: 1 = moe_twm (RT 2 / 64-token passes for experts with > 32 tokens, RT 4 for the rest)
_moe_TWF_RT_M = {}  # s11: per-M override, e.g. {256: 2} for the layer's verify 32 (hot experts with 90+ tokens: 34 vs 24 passes)
_moe_TWM_TAIL = 0  # s14: moe_twm runs the gemm2 of the lightest experts (up to this many RT 4 items) at RT 2 / 32-token passes
_moe_TWM_TAIL_M = {}  # s14: per-M override of TWM_TAIL
_moe_TWM_TAIL13 = 0  # s14: same for the gemm13 items of the lightest experts (RT 4 gemm13 item budget)
_moe_TWM_TAIL13_M = {}
_moe_TWM_ORDER = 0  # s14: moe_twm item order (bits, see moe_twm): 76 = pass-minor + multi-pass experts' items merged evenly
#   over both phases (H100 layer v32 248.8 -> 239.9 us without TWM_TAIL (with it: 247), v16 = 0); 2 = timeline stamps
_moe_TWM_ORDER_M = {256: 76}  # s14: per-M override
_moe_HOT_M = {}  # s29: {M: grid}: moe_hot (65..128-token experts, one N 128 pass) on that many CTAs before moe_twm
_moe_TWM_HOT = 0  # s17: moe_twm hot class (experts with 65..96 tokens: one pass, 1 tile per warp, N <= 96)
_moe_TWM_HOT_M = {}  # s17: per-M override
_moe_TWQ = 0  # s17: big-M on moe_twq (16 consumer warps x RT 2, 32-token passes) if nonzero (value = flags; bit 1 = timeline,
#   bit 3 = pass-minor item order; use 4 for plain on)
_moe_TWQ_M = {}  # s17: per-M override, e.g. {128: 4} (proxy: v16 -6.5 proxy-us, v32 +4.6)
_moe_TWF = 1  # big-M (TW_CFG 418): gemm13 + gemm2 in one persistent launch (moe_twf)
_moe_TB_PACK = ()  # RT values packed in prepare for moe_tb (and moe_twf RT 2)
_moe__CFGS = (10, 11, 22, 118)
_moe_SK_CFG = 3088  # decode kernel cfg: 2DM = moe_sk (D, MB); 3[NC][NSTG] = moe_tk
_moe_TK2_MAX = 32  # M above this: big-M path
_moe_TK2F_DYN = 84204  # M > 1 cfg. s13: 84204 = 204 + y32 in route-slot order pre-scaled + moe_norm3 (bitwise = 204; H200 diag adfb897e d16 -0.45, d32 -0.75 us). M = 1 (TK2F) stays 204: the layer's n2u reads the old y32 order. 5402 = DYN row tail (D = 4): judge no gain (b890d69b)
_moe_TWSA = 1  # s9: same for moe_twf (big-M)
_moe_TK2SA = 1  # s9: arithmetic e4m3 scale decode in moe_tk2f (when prepare() found all scales positive normal)
_moe_TK2FN = 0  # s9: 1 = post-expert norm fused into moe_tk2f (moe_tk2fn_launch): judge bs1 +2 us (check 168fa04c), dead
_moe_TK2G_W = 0  # s10: GS16 2-group row weight = (8 + TK2G_W) / 8 in the static row split
_moe_TK2G_MIN = 17  # s10: GS16 only from this M (H100: GS16 is slower on the layer routings; A/B on the judge)
_moe_TK2G = 0  # s10: 1 = 16-token segments in moe_tk2f (2 token groups share each dequantized weight fragment): judge bs32 +7 us (check b25182e8), H100 layer routings +6..18 us: off
_moe_TWF_DBG = 0
_moe_TWF_SPLIT_M = {}  # s11 per-M override of TWF_SPLIT; judge c1659443 {128: 132, 256: 66}: tie (tail -12 us, twf +10 us)
_moe_NORM_EARLY = 1  # n4s1: moe_norm after moe_twm waits per token on its experts' gemm2 flags (needs #define NORM_EARLY 1)
_moe_TWF_SPLIT = 0  # s11: moe_twf runs its last TWF_SPLIT gemm2 items as 2 K halves (FP32 partials, ticket): shorter tail
_moe_TK2F = 204  # decode: gemm13 + gemm2 in one launch (moe_tk2f, cfg KPS*100+NSTG); 0 = 3-launch moe_tk2 path
_moe_TK2_NSTG = 8  # decode path: moe_tk2 stages (0: the 5-launch SK_CFG path)
_moe_SK_ZERO = 64  # CTAs zeroing Acc13 in route2


# s12 DIAG (print only; never submit with DIAG on): H200 timing of expert variants on the LAYER's routings
# (r-layer reference front + top-k, 5 seeds per load; /tmp/moe10/routes.pt), cold L2, CUDA-graph replay medians.
_moe_DIAG = 0
_moe__DIAG_HDR = [('decode 1', 7, 1), ('decode 1', 11, 1), ('decode 1', 13, 1), ('decode 1', 17, 1), ('decode 1', 99, 1), ('decode 16', 7, 16), ('decode 16', 11, 16), ('decode 16', 13, 16), ('decode 16', 17, 16), ('decode 16', 99, 16), ('decode 32', 7, 32), ('decode 32', 11, 32), ('decode 32', 13, 32), ('decode 32', 17, 32), ('decode 32', 99, 32), ('verify 1', 7, 8), ('verify 1', 11, 8), ('verify 1', 13, 8), ('verify 1', 17, 8), ('verify 1', 99, 8), ('verify 16', 7, 128), ('verify 16', 11, 128), ('verify 16', 13, 128), ('verify 16', 17, 128), ('verify 16', 99, 128), ('verify 32', 7, 256), ('verify 32', 11, 256), ('verify 32', 13, 256), ('verify 32', 17, 256), ('verify 32', 99, 256)]
_moe__DIAG_R = "eNqVXIt22zgOLc02laBVxDpQorWm/v+/5IpvAASdWZ8zjSVRJAjicXFJj0cfPmC9X66/7+HC+a940//zy9+vP+fD589b/jc/f/L71/XbW7z7T338ljosLUjL699P2qdnnZHr55/wb5Buy//5/Hfx/ee6b1uzMJ9w+f5gt/22pakm4ZZHfcA6In/a59ejjbvlx0sR75f3jzDQdv0J9+zmtW62dqONvbVpRbFdnUGQHUK3J5Yeyheg3eabc3rgvMX03IkpnL71A/Haso4c8kHit0uyP0L8J2lhpZ7se3nwWFRNvodb12sumpezVbj0Vm7O1xhDd67czCM8sqoeW7+CmHtbiJCPjUmzcfU9sOjrbI/o7H7FRw8i2Pb+S5mgLT1es/zjiDarUUS548UfFNIslwFVleQ21ULCajw80RebEBXjGfrxuBE9Vjd4T3PZ4oToZ0mmvHRzt6z3M+leeMwmNBVuvNcGT6LpoIZ7bbjprif850lXs3VcvD4tRnlkPVn8rci8EL+N7dsMHBMgrl9aZpcWxD6T2i0WPYXLLU0q+v2SXl2ez63zibMp+EF96+yn/i5eoM3lx8SnkFtAnYUr75l8v/PQue9axhfTX7ApIaCUVGt3je/qeJcshkQuaIOa6yuWu45KvGalhEiFZe2t78w+3/TvinsLQ/b2Mbbcak+a3s/O2avBpuZoy92lBZ4lW0m8ezkXjVZbZ3yhp0eV9X2jcgAN/DPRV/zSWh4ip8rP33D7M7vgNdr+VR9dmfkrdv6b6+6enD7k93vUw80L9W7nnek6NPqntPrKDvv3tyLO1+Xdf6+/v5O4X2F4Y7KkX8yqMkwxv4m81+e/faf3T2bcCzc5U2cQb9ySWp3/WZfxBlXnWD3J/+AGNVM1YMxldRWm+mAfOHBLtqfmp1qGr8O7OFwexDSdH8SgoMSD9RIUOjgABV+okabiDOxCUhgkmR1IOUsUuiS6h/tLeu8mBwBPTSjIPxPhPsMVSi0BN/MEds66qCeXBUkwxE6d6dYHuX2IGKZow5LbRg4EqQtEEKuI/XKTqZ8lanA/j1dmdlyUs5dsDs2AWOVMMchk6LqffYTGFpzDBV+nWcz/1uaN10PrcWStrBtso8KwcRF/Ap/AAWSJMdsUComkhOkFdGrCMw1mcsCbRONaMUHNezHvml/7hOD0ROGa8xttukAAeOvAscQyl9SpDwFxAItrjwik35AMy/NFSK7P2xyXFYnwwsIdm4BNygV93bVlpR4JuoGEr2u6uLebLiEtBy2gYrPl90tFU+rkYGPNbNyls8W4kDxCOtTj3nkkOW5eACna3l2NLuELWozPQDcYrGWz/zClGyzrXq0JWgy4lyura+6q9GgGqrjY8nWO9t+tjqG9xTQXE8smIPH1bVbWMiKz2MqxBAcsVYCmsUvKo93HIour8566uGA6J4CJY3fg2bbgSSAh52Qhg+W9fZgAkUwo6JDmUdfbve30kPoxpR4PRSYLC3tejIXkizfuS1aJrfAKdFdHWcVs1uvLPWeSHf9FmWF7m4GcLWEQF3ei5T09tzTulW6vNpMkKPznQI5bTCNzfDG89BdikD0tC5QyO9z04IQpd9WMcFMn7L4pwGzyq0/q9HgWF8aILkoHZx7+TeJy5EH/4PK62q3z33ygY3ssyzb7oIJUjMfYnUYiJzTiOGUQVoH2VKiNo0zP+mHYHz5Jzv2x54kRUA2OIZZa57ZyMVgduRXzrGkNapx1bORY6c/Q8WvQgW+UrMFSGhy6ehn8zP7yR3FqL6Y/zKdTWW8sdjV/Zx4V/8CrsiSE5sYcOu4GwNeSKMWghgBWP/ZN010CntyOgQ+R/GDVJtuiy/zSPyRdkaA3qupAQZ2Y+mev5pNHM/MrOAgZpKBm/nvB5RsCf7ZKSBk9yd0KTgcdBMf3kM2vAczBKky5s0nXG7ZYsrZYYr0+T2/1oOIowzyJoIwvcWNgA5rCsMPRA54jp3UFt6/e1mpzfoVd1fszc+nDxS2UbhITtyOx9MaTsngrSC6B77mlPC+lXDX2IHvYLAc5kfKEfWY4qAId5/xvnTJJFYcfAzCqBvhZpo1r9CvOwqk0n9UMATFhDzIIipeLwSNZKaOtJKSciDlw717zkrMkEEfhw67tw3SG8iG9GE5hHzcbZcTYPVaY+1/rXwE+zcyGdV9DbY6KM7N9ilaOup64QE8Beh5wsv1AqT6De1PHvWMAXB+HjIr5Dhk70uo6A9/CiNpi3ULvQONy0qybR/6eQItJi5UgYnwf7TeAJQxnOgAuZXxWy+jzxC74JNsqRHXUFkgzDQAvsZVImUugSqJ6Uky2rBwTAcH1Bn7z345W8Cca5nSOaxtpploo8hUCJUGmxukBawzfIcww0d0TFOMSdIVJr152Hq0BB1FcxOhVwB+ejjX0Y19rsvVg2SrMaf3L3FfVPCWbGdYb7/TWWax+h5yeQTE3I3P5PmURkOXwxlbsfVFIUQNIn5yR2c08kdSU39g5iwGyI4YO2teqmSn2jPPATNB0KSBlOmTRH1LOhlmHepZ2cvRJhWzfDILKmtXV3JwkMEkvb1NlJhrbn+313hHhGnWK/jVfU9PAUYSDQHhx+NUWuVZ8qkmfvNtpGE1rIVbqh2ZPwJkN1zKhcZHvmZXRiTJbV7b09paUfIVnCzTs7LkesUWiAvWB69Ni22whLIkn0Q4alXrWplbqfveqCccCiXtgC5Ib5QsiPzKz/dQRYLAj3HuT2cR1Kmym2rZdIrRfG1PGz5IAjrO1axU5sAHMnbTjFmxQIhzsaZ5qSE5SnvN3OILgOxJ3b76Hpr1f5Y0YZJAY/WsAx9UV1GoNjeFiB7Dt+6YlcAPoPCpkUBqUih9Nozvw1CjBblvmIBXV+hIvRaO+Auxz1wOA3G3rduYYr0K52LU8NoRPwKQprgbzmlebid296ZotVkf42PiChQ5JN9mfOKjIU5PJ66xrejyxPtn+y6yts9isXL2gB/4dOwc7Schz4Imwpu098KlmXDSnZ06nvKDZ45FdjJAYli4vMv9TT8KtSDf3K/+PrPpvrKzghGRc56+CHy2bmK9bi1uizqdsCycFzKgeMSJuGV7obTruG3KAtDh1g0g8ZdLCCQ2APBsIqqIKvgYZZXg+sv1Wfb5l2lSRxQHUAmU583QChU/gvX+xVK1e+IEa30vjD+YUoFETN7Xuto1dBgbCuf3X8G1rcLHXsmjISwlMH/kUhlImIGPWavwLGPYoIXNt/D3QebvqEfNglUHFka6/iTp/Hjr/zXFtlXpl893GUSlRWxNKGCDi5cKeHyXcg+bfHU/NDTecEDq6l49xwJy/42VAxASnzvZGCkkveCjmlgqv/4vCRUjbXk4Mv8iiXDZoEZGfbGL70tNIbcpCN35OxaFHOpfslKZWSzFLY1sCZfKDByjUVB8Pca0gyNnqPvYbHrjbMMhbMoASt3TbnuLvqtAB1g9LsGE0ONamZqpuZIFKXwtgS2qr5WKfOl6kmtuQP69j9IdgwOnOw0KK4haW2H1BZTX+Tx1+Y/Vx4RUnhS0j9f9cUwWyJRrPsj80hcN3LtnXth9I6XHr1W7MS4pNDww12prINzqe/6zw95vSM/qTnWjJfpZO2SDZcDb/H/ZgcWUp0zWzLDKbQ2WK7Bw4pdUKKsfKkiJuefzGqW01XyH8hwBtpCi8/mbBpjBgqrpRLacm48mpDNT9G4p5HpUdcR/dvk/DtROvQ2DgReM9tp+J5c0s8sLkmYV9SjfZdaxqYZAQ23k6b90RG995Ac2A5YRkT4HnI8GLOO7/TpKuTo/i7duPXAqIjX1yrDqhhsO2h0+p0gop04683RsfZEHZBHKtAFXP+UKa5ylpMxobul1BpvlSrwtiHGOZ+Iz8I0jcw6GXspAfHUMo+ZZY/HC2T+7fWZoTQU1rt+rL0AHn0PaoYR+VzTUm35gRsTx4AM84DhT2qzdp5L8ikvBJzRgfr89rQF1VltR5hxT9r2T+kM/LsbM7u6KYU2rrGDBarN2c7Q5eMMbKxofx9ugZM2TVnaHHytLCmfYdi3u9AQGKcf267d1ZWuAq5YNBKj+0Thi/Ho7eGq9Xz0nhuyBsHeNsxYElJvi80Sh4sEVypK5zCs7u9O/4OtDtXcqkQasV+DnzLixBy2Tl3o1Y/G3MHa2DWlzkrFqcEZbjI294mKmrm0WEBwWRQVNcf/7nxvSk/sDS/u394Wy4AQn650fpFX44jPNZTMuy+OA4ydWZ6dkZJWp2ioUC1OpDJ36wRd3lGXcqioYc2ZSbXEcseLqZL8FJJL78LpZDHtM/m0tN4F7CaWVX22lpjZxXLAHqlADAarL49KuZpbwP/Mj7mbb/QxJfSydA92+Qkj4SqHjPCaAsnlPr+XVKmwbl9WOQHW4Kj3pmRGt/oOCvQHp6+ZWS06BEOoDvBHDubcbW7VK2fpFfQKs4oxv9zsh/VKC6vqqtgHtt5oAmBVV4ti/i+e9rjtfVW/628v2mQjVapXXx8FlYqNMyTDOEE6gdr7yAcS92mOpB7vQSysdhf8C20yTsTC2WLz9v7X7k6o50mrq2fc5ZWwaBRVNfdzAbAUTzb95QWf03p0MWFZ7PBO+psPg2XDs0r8BU3bzCgkVhkudGtrmdTb9mOA/XriZV/lNMuekxSfO3GbS5UdHc9v4+6rqj8PMSb6BhCm2vmzGwqIA40HAa89xZhjTs8/osVfTmxa+ebOsind3XkVO8fygIm+9Bv13LfKCWjBLM3erPPySfkc71oFqT5pEj7HA6QrGy8i/4s56aQfa/LLDgfnUra/Lbqd1uSZweni+D0QH63r0qc4lt2+jkOwJiH1rYtPXphxadNc9KFVoTcwvstz6UNQTC9msO4s/YfvzCp/jRR3V3NPl/km4/FJo9ffsf7xy7jA=="
_moe_DIAG_VARIANTS = {  # s14 queued: H200 A/B of the twm item orders (H100: o76 v32 -9 us vs main, v16 0)
    "dec": [("gs8", {})],
    "big": [("main", {})] + [(f"o{o}", {"TWM_ORDER": o, "TWM_TAIL_M": {}}) for o in (0, 76, 12, 28, 92)]
           + [("o76t", {"TWM_ORDER": 76}), ("mainb", {})],
}


def _moe__diag(st):
    import zlib, base64
    g = globals()
    raw = zlib.decompress(base64.b64decode(_moe__DIAG_R))
    routes, p = [], 0
    for load, seed, T in _moe__DIAG_HDR:
        loc = torch.frombuffer(bytearray(raw[p:p + T * 8]), dtype=torch.int8).view(T, 8).int().cuda()
        p += T * 8
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids = torch.where(loc >= 0, loc + st["start"], (st["start"] + st["count"] + 5) % 256).int()
        tw = torch.rand(T, 8, device="cuda", generator=gen)
        data = {"hidden_states": torch.randn(T, _moe_H, device="cuda", generator=gen).bfloat16(), "topk_ids": ids,
                "topk_weights": tw / tw.sum(-1, keepdim=True),
                "config": {"bs": T, "d_hidden": _moe_H, "local_start": st["start"], "local_count": st["count"]}}
        routes.append((load, seed, T, data))
    flush = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda").view(torch.int64)
    sink = torch.empty((), dtype=torch.int64, device="cuda")
    res = {}
    for load, seed, T, data in routes:
        vs = _moe_DIAG_VARIANTS["dec" if T <= _moe_TK2_MAX else "big"]
        graphs = []
        for name, ov in vs:
            old = {k: g[k] for k in ov}
            g.update(ov)
            try:
                _moe_kernel_fn(data, st)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    o = _moe_kernel_fn(data, st)
                gr.replay()
                torch.cuda.synchronize()
                graphs.append((name, gr, o))
            finally:
                g.update(old)
        base = graphs[0][2].float()
        ts = {n: [] for n, _, _ in graphs}
        for _ in range(30):
            for n, gr, _ in graphs:
                torch.sum(flush, 0, out=sink)
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); gr.replay(); b.record()
                torch.cuda.synchronize()
                ts[n].append(a.elapsed_time(b) * 1e3)
        for n, _, o in graphs:
            t = sorted(ts[n])[len(ts[n]) // 2]
            err = (o.float() - base).abs().max().item()
            res.setdefault((load, n), []).append(t)
            print(f"diag12 {load} s{seed} {n}: {t:.1f} us (maxdiff vs {graphs[0][0]} {err:.3g})", flush=True)
        del graphs
    for (load, n), v in res.items():
        print(f"diag12 MEAN {load} {n}: {sum(v) / len(v):.1f} us", flush=True)
    if _moe_DIAG_TL:
        for load, seed, T, data in routes:
            if T > _moe_TK2_MAX:
                for o in _moe_DIAG_TL:
                    _moe__diag_tl(st, data, f"{load} s{seed}", o)
    del flush
    torch.cuda.synchronize()


_moe_DIAG_TL = ()  # s14: twm item orders to timeline in the diag (needs TWM_TL 1 in the CUDA source: diag copies only)


def _moe__diag_tl(st, data, name, order):
    """s14: per-CTA moe_twm timeline (TWM_TL build): exit spread, HBM-rate estimate per 10 us, item durations by kind."""
    import collections
    g = globals()
    old = (g["TWM_ORDER"], g["TWM_TAIL_M"], g["TWM_ORDER_M"])
    g["TWM_ORDER"], g["TWM_TAIL_M"], g["TWM_ORDER_M"] = order | 2, {}, {}
    try:
        for _ in range(3):
            _moe_kernel_fn(data, st)
        torch.cuda.synchronize()
    finally:
        g["TWM_ORDER"], g["TWM_TAIL_M"], g["TWM_ORDER_M"] = old
    part, meta = st.pop("_tl")
    tl = part.view(torch.int64)[:132 * 64].view(132, 64).cpu()
    meta = meta.cpu()
    t0 = tl[:, 0].min().item()
    us = lambda v: (v - t0) / 1000
    n = tl[:, 1]
    ex = sorted(us(tl[c, 63].item()) for c in range(132))
    rows = sorted((us(tl[c, 2 + 3 * k].item()), us(tl[c, 3 + 3 * k].item()), tl[c, 4 + 3 * k].item())
                  for c in range(132) for k in range(min(n[c].item(), 20)))
    na = meta[0].item(); off = meta[33:66].tolist()
    E = [meta[1 + a].item() for a in range(na)]
    nt = [off[e + 1] - off[e] for e in E]; big = [x > 32 for x in nt]
    PS = [64 if b else 32 for b in big]; npass = [(x + q - 1) // q for x, q in zip(nt, PS)]
    nbig = 0
    while nbig < na and big[nbig] and (not (order & 64) or npass[nbig] > 1): nbig += 1
    items = []
    for g2, nrbl in ((0, [32 if b else 16 for b in big]), (1, [16 if b else 8 for b in big])):
        lst = []
        for a in range(na):
            if order & 4:
                lst += [(g2, a, q) for rb in range(nrbl[a]) for q in range(npass[a])]
            else:
                lst += [(g2, a, q) for q in range(npass[a]) for rb in range(nrbl[a])]
        mrg = (order >> 3) & 3
        if mrg and not (g2 and order & 32):
            phx = {1: 100, 2: 85, 3: 70}[mrg]
            nA = sum(npass[a] * nrbl[a] for a in range(nbig)); N = len(lst)
            if 0 < nA < N:
                W = max(nA, N * phx // 100); out = []
                for it in range(N):
                    ca = it * nA // W if it < W else nA; ca1 = (it + 1) * nA // W if it + 1 <= W else nA
                    out.append(lst[ca] if ca1 > ca else lst[nA + it - ca])
                lst = out
        items += lst
    T = ex[-1]; hb = [0.0] * (int(T / 10) + 2)
    kinds = collections.defaultdict(list)
    for a_, b_, it in rows:
        g2, a, q = items[it]
        nv = min(PS[a], nt[a] - q * PS[a])
        kinds[(g2, int(big[a]), q > 0, (nv + 7) // 8 * 8)].append(b_ - a_)
        if q: continue
        mb = 0.55 if big[a] else 1.09
        d = max(b_ - a_, 0.1)
        for z in range(int(a_ / 10), int(b_ / 10) + 1):
            lo, hi = max(a_, z * 10), min(b_, z * 10 + 10)
            if hi > lo: hb[z] += mb * (hi - lo) / d
    print(f"diagtl {name} o{order}: experts {na} exit min/med/max {ex[0]:.1f}/{ex[66]:.1f}/{ex[-1]:.1f} "
          f"first g2 start {min(a for a, b, it in rows if items[it][0]):.1f} last g13 end "
          f"{max(b for a, b, it in rows if not items[it][0]):.1f} tokens {nt}", flush=True)
    print(f"diagtl {name} o{order} HBM TB/s per 10us: " + " ".join(f"{v / 10:.2f}" for v in hb), flush=True)
    print(f"diagtl {name} o{order} kinds (g2 big reuse N: n mean): " + "; ".join(
        f"{k[0]}{k[1]}{int(k[2])} {k[3]}: {len(v)} {sum(v) / len(v):.1f}" for k, v in sorted(kinds.items())), flush=True)


_moe_TK2H = 0  # s12: merged-phase tk2f consumer (one main-loop body, i-cache): 8 = GS 8, 16 = GS 16 (from M >= TK2H_MIN16)
_moe_TK2H_MIN16 = 1
_moe_TK2H_MIN = 1  # n4s5: TK2H applies from this M (below: the main tk2f cfg); 9 = only calls that can have 2-group rows
_moe_TK2W = 0  # s12: weighted static row split for M > 1 (extra-group rows weigh TK2W / 8; 0 = by count), cfg + 70000
_moe_TK2H_PSC = 0  # s12: 1 = per-segment gemm13 -> gemm2 counters (cnt[64 + seg]) instead of the global one
_moe_TK2H_PF4 = 0  # n6: 1 = tk2h writes y32 like tk2f PF 4 (slot order, pre-scaled) -> moe_norm3 (cfg + 4; kinds 3/4 only)
_moe_TK2H_W = 0  # GS16 2-group row weight (8 + W) / 8 in the static split (tk2g_split)
_moe_TK2H_M = {}  # layer s58: {M: W + 1}: tk2h GS16 + PF4 (SA only) at that M, 2-group weight (8 + W) / 8


_moe_TK2F_BAL_M = set()  # moe n8 BAL: M values whose tk2f (84204 / NU8 85204, SA) runs balanced (cfg + 1000000)


def _moe__tk2f_cfg(M, st):
    return _moe__tk2f_cfg0(M, st) + (1000000 if M in _moe_TK2F_BAL_M and _moe_TK2SA and st["sa"] else 0)


_moe_TK2F_S6 = 0  # layer s73 (moe n20, r-moe 4a87466): 6-bit scale-code SQ stages when every scale byte is in [80, 0x7E]. OFF: H200 moe diag 13b528a8 tk2f d16 +0.7..1.0, d32 +2.9, tk2h W3 d32 +1.6 us (H100 -1..-3 = DRAM-bound artefact)
_moe_TK2F_S6_ON = True  # run tk2f / tk2h on them (False: the 8-bit SQ stages; needs TK2F_S6_KEEPSQ)
_moe_TK2F_S6_KEEPSQ = False  # also build the 8-bit SQ images when S6 is on (A/B tests)


def _moe__s6_unpack(p):
    """inverse of _s6_pack's scale part -> int32 view of the paired scales (prepare-time round-trip check)."""
    E, rpe, nq = p.shape[0], p.shape[1], p.shape[2]
    t = p[..., -384:]
    nib = t[..., :256].reshape(E, rpe, nq, 8, 32, 1).long() & 0xFFFFFFFF
    hi = t[..., 256:].contiguous().view(torch.int16).reshape(E, rpe, nq, 8, 32, 1).long() & 0xFFFF
    k = torch.arange(8, device=p.device)
    c = ((nib >> (4 * k)) & 15) | (((hi >> (2 * k)) & 3) << 4)                 # E lr q warp lane k(= 4h + r)
    c = (c + 80).reshape(E, rpe, nq, 8, 32, 2, 4).permute(0, 1, 2, 5, 3, 4, 6)  # E lr q h warp lane r
    return c.to(torch.uint8).contiguous().view(torch.int32).reshape(E, rpe, nq * 16, 32)


_moe_TK2F_SQ = 1  # layer s68 (moe n14, H200 d32 -1.5, d16 -0.55 us, bitwise = off): 184204 (+ BAL) on the _sq_pack buffers


def _moe__tk2f_w(c, st):
    """(cfg, w13, w2) for moe_tk2f_launch: SA 84204 (+ BAL) -> + 10000000 on the single-copy buffers (SQ)."""
    sq = c % 1000000 == 184204 or (140000 <= c % 1000000 < 150000 and c % 1000 == 4)  # 84204 (+ BAL) / moe n16 tk2h
    if _moe_TK2F_SQ and sq and _moe_TK2F_S6_ON and "w13s6" in st:  # layer s73 (moe n20 S6)
        return c + 20000000, st["w13s6"], st["w2s6"]
    if _moe_TK2F_SQ and sq and "w13sq" in st:
        return c + 10000000, st["w13sq"], st["w2sq"]
    return c, st["w13q"], st["w2q"]


_moe_TK2F_DIAG = False  # layer s81: build the tk2f DIAG probes (DIAG check copies only; sed to True)
_moe_TK2X_M = {}  # layer s81 DIAG: M -> probe code (tuner timing only): 1 nm204, 2 sq402, 3 sq403, 4 nm402, 5 nm403, 6 plain SQ 204


def _moe__tk2x(c, st, code):
    """layer s81 DIAG: (cfg, w13, w2) of probe `code` (c = the SQ 184204 cfg) -- wrong outputs for the nm codes."""
    assert _moe_TK2F_DIAG and c == 10184204, c
    if code == 6:
        return c, st["w13sq"], st["w2sq"]
    if code == 1:
        return 10184214, st["w13sq"], st["w2sq"]
    return 10000000 + (184402, 184403, 184412, 184413)[code - 2], st["w13sq4"], st["w2sq4"]


def _moe__tk2f_cfg0(M, st):
    sa = 100000 if _moe_TK2SA and st["sa"] else 0
    if _moe_TK2H_M.get(M) and sa:
        return sa + 40000 + (_moe_TK2H_M[M] - 1) * 1000 + 4
    if _moe_TK2W and M > 1:
        return sa + 70000 + _moe_TK2W * 1000
    if _moe_TK2H and M >= _moe_TK2H_MIN:
        return sa + (40000 + _moe_TK2H_W * 1000 if _moe_TK2H == 16 and M >= _moe_TK2H_MIN16 else 30000) + 20000 * _moe_TK2H_PSC + 4 * (_moe_TK2H_PF4 and not _moe_TK2H_PSC)
    return (_moe_TK2F_DYN if M > 1 else _moe_TK2F) + sa + (10000 + _moe_TK2G_W * 1000 if _moe_TK2G and M >= _moe_TK2G_MIN else 0)


def _moe_kernel_fn(data, st):
    cfg = data["config"]
    assert cfg["local_start"] == st["start"] and cfg["local_count"] == st["count"]
    lib = st["lib"]
    x = data["hidden_states"] if data.get("gather_norm") is not None else data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].float().contiguous()
    M = x.shape[0]
    dev = x.device
    meta = torch.empty(132, dtype=torch.int32, device=dev)  # [100, 132): NORM_EARLY gemm2 flags
    sroutes = torch.empty(M * _moe_TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((M * _moe_TOP_K, _moe_I), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, _moe_H), dtype=torch.bfloat16, device=dev)
    if M <= _moe_TK2_MAX and _moe_TK2_NSTG:
        inv = torch.empty(M * _moe_TOP_K, dtype=torch.int32, device=dev)
        y32 = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.float32, device=dev)
        if _moe_TK2F and _moe_TK2FN:
            lib.moe_tk2fn_launch(st["w13q"], st["w13s2"], st["w2q"], st["w2s2"], st["lut"], st["gs13"], st["gs2"], x,
                                 ids, st["start"], st["count"], meta, sroutes, inv, inter, y32, st["cnt"], _moe_TK2F, tw,
                                 st["w1"], out, _moe_POST_NORM_EPS)
            return out
        if _moe_TK2F:
            c, w13_, w2_ = _moe__tk2f_w(_moe__tk2f_cfg(M, st), st)
            if _moe_TK2X_M.get(M):  # layer s81 DIAG
                c, w13_, w2_ = _moe__tk2x(c, st, _moe_TK2X_M[M])
            lib.moe_tk2f_launch(w13_, st["w13s2"], w2_, st["w2s2"], st["lut"], st["gs13"], st["gs2"], x, ids,
                                st["start"], st["count"], meta, sroutes, inv, inter, y32, st["cnt"], c)
            c5 = c % 1000000 % 100000
            if (80000 < c5 < 100000 and (c5 - 80000) // 1000 & 4) or (30000 <= c5 < 50000 and c5 % 1000 == 4):  # s13 slot-ordered y32 (n6: tk2h + 4)
                lib.moe_norm3_launch(y32, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out, st["cnt"])
                return out
            lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], _moe_POST_NORM_EPS, out,
                                 st["cnt"])
            return out
        lib.moe_tk2_launch(1, _moe_TK2_NSTG, st["w13q"], st["w13s2"], st["lut"], st["gs13"], x, ids, st["start"], st["count"],
                           meta, sroutes, inv, inter)
        lib.moe_tk2_launch(0, _moe_TK2_NSTG, st["w2q"], st["w2s2"], st["lut"], st["gs2"], inter, ids, st["start"],
                           st["count"], meta, sroutes, inv, y32)
        lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], _moe_POST_NORM_EPS, out, st["cnt"])
        return out
    if M <= 32:
        sk = _moe_PLAN_OVERRIDE(M) if _moe_PLAN_OVERRIDE else (_moe_SK_CFG, _moe_SK_CFG)
        inv = torch.empty(M * _moe_TOP_K, dtype=torch.int32, device=dev)
        acc13 = torch.empty((M * _moe_TOP_K, 2 * _moe_I), dtype=torch.float32, device=dev)
        acc2 = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.float32, device=dev)
        lib.moe_route2_launch(ids, st["start"], st["count"], meta, sroutes, inv, acc13, _moe_SK_ZERO)
        lib.moe_sk_launch(1, sk[0], st["w13q"], st["w13s2"], st["lut"], x, meta, sroutes, acc13)
        lib.moe_act_launch(acc13, meta, sroutes, ids, st["gs13"], st["start"], inter, acc2)
        lib.moe_sk_launch(0, sk[1], st["w2q"], st["w2s2"], st["lut"], inter, meta, sroutes, acc2)
        lib.moe_norm2_launch(acc2, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], _moe_POST_NORM_EPS, out, st["cnt"])
        return out
    p13, p2 = _moe_PLAN_OVERRIDE(M) if _moe_PLAN_OVERRIDE else _moe__plan(M)
    assert p13 in _moe__CFGS and p2 in _moe__CFGS, (p13, p2)
    rg = bool(data.get("rg")) and _moe_TW_CFG and data.get("gather_norm") is not None and data.get("bucket") is None
    if rg:  # layer s52: route + gathern in one launch below
        pass
    elif data.get("bucket") is None:
        lib.moe_route_launch(ids, st["start"], st["count"], meta, sroutes)
    else:  # layer s27: bucketed on the layer's router stream (meta, sroutes of moe_route_launch)
        meta, sroutes = data["bucket"]
    if _moe_TW_CFG:
        # inter here holds gemm2's B in the K-permuted order (see moe_tw)
        rt = 2 if _moe_TW_CFG >= 1000 else _moe_TW_CFG // 100
        xg = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.bfloat16, device=dev)
        if rg:
            assert st["start"] == 0
            lib.moe_routegathern_launch(ids, st["count"], x, *data["gather_norm"], meta, sroutes, xg, data["gather_eps"])
        elif data.get("gather_norm") is None:
            lib.moe_gather_launch(x, meta, sroutes, xg)
        else:  # layer s27 GNORM: x = the pooler output; the pre-expert rmsnorm happens in the gather
            lib.moe_gathern_launch(x, *data["gather_norm"], meta, sroutes, xg, data["gather_eps"])
        if _moe_TWF and rt == 4:
            trt = _moe_TWF_RT_M.get(M, _moe_TWF_RT)
            trt = trt if f"w2b{trt}" in st else 4
            lib.moe_set_sa((1 if _moe_TWSA and st["sa"] else 0) | (2 if trt == 2 else 0))
            ns = _moe_TWF_SPLIT_M.get(M, _moe_TWF_SPLIT)
            part = (torch.full if _moe_TWF_DBG else torch.empty)((max(66 if ns >= 1022 else ns, 1) * 32768,), *((float("nan"),) if _moe_TWF_DBG else ()), dtype=torch.float32, device=dev)
            if _moe_TWQ_M.get(M, _moe_TWQ):
                q = _moe_TWQ_M.get(M, _moe_TWQ)
                qs, qk = (q >> 8) & 255, (q >> 16) & 3  # n21: K-split tail in the flags: q | items << 8 | log2 parts << 16
                if qs and qk and not _moe_DIAG_TL:
                    part = torch.empty((qs << qk) * 16384, dtype=torch.float32, device=dev)
                if _moe_DIAG_TL and _moe_TWM_ORDER_M.get(M, _moe_TWM_ORDER) & 2: st["_tl"] = (part, meta); q |= 2
                hq = (q >> 24) & 127  # moe s23: q | grid << 24 = moe_hot<NH 64> on grid CTAs first; moe_twq skips its experts
                q &= (1 << 24) - 1
                if hq:
                    lib.moe_set_hot(4)
                    lib.moe_hot_launch(st["w13b2"], st["w13sb2"], st["gs13"], st["w2b2"], st["w2sb2"], st["gs2"], st["lut"],
                                       xg, meta, sroutes, inter, y, hq)
                    q |= 1 << 20
                lib.moe_twq_launch(st["w13b4"], st["w13sb4"], st["gs13"], st["w2b4"], st["w2sb4"], st["gs2"], st["lut"], xg,
                                   meta, sroutes, inter, y, part, st["tk"], q)
            elif _moe_TWF_MIX and "w2b2" in st:
                tl = _moe_TWM_TAIL_M.get(M, _moe_TWM_TAIL) + 4096 * _moe_TWM_TAIL13_M.get(M, _moe_TWM_TAIL13) + (_moe_TWM_ORDER_M.get(M, _moe_TWM_ORDER) << 24)
                tl += (1 << 23) if _moe_TWM_HOT_M.get(M, _moe_TWM_HOT) else 0
                if tl and ns:  # s19: K-split tail together with order / hot (tail classes off)
                    tl = (tl & ~((1 << 23) - 1)) + (1 << 22) + 4096 * min(ns, 1023)
                ns = -tl if tl else ns
                if _moe_DIAG_TL and tl >> 24 & 2: st["_tl"] = (part, meta)
                hg = _moe_HOT_M.get(M, 0)
                lib.moe_set_hot(1 if hg else 0)
                if hg:  # s29: experts with 65..128 tokens in one N 128 pass, moe_twm (PDL secondary) skips them
                    lib.moe_hot_launch(st["w13b2"], st["w13sb2"], st["gs13"], st["w2b2"], st["w2sb2"], st["gs2"], st["lut"],
                                       xg, meta, sroutes, inter, y, hg)
                lib.moe_twm_launch(st["w13b4"], st["w13sb4"], st["gs13"], st["w2b4"], st["w2sb4"], st["gs2"], st["w13b2"],
                                   st["w13sb2"], st["w2b2"], st["w2sb2"], st["lut"], xg, meta, sroutes, inter, y, st["tk"],
                                   part, ns)
                if _moe_NORM_EARLY:
                    lib.moe_norme_launch(y, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out, meta)
                    return out
            else:
              lib.moe_twf_launch(st[f"w13b{trt}"], st[f"w13sb{trt}"], st["gs13"], st[f"w2b{trt}"], st[f"w2sb{trt}"], st["gs2"], st["lut"], xg,
                               meta, sroutes, inter, y, st["tk"], part, ns)
            lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out)
            if _moe_TWF_DBG: st["dbg"] = (y, inter, meta, sroutes)
            return out
        lib.moe_tw_launch(1, _moe_TW_CFG, st[f"w13b{rt}"], st[f"w13sb{rt}"], st["gs13"], st["lut"], xg, meta, sroutes, inter)
        lib.moe_tw_launch(0, _moe_TW_CFG, st[f"w2b{rt}"], st[f"w2sb{rt}"], st["gs2"], st["lut"], inter, meta, sroutes, y)
        lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out)
        return out
    if _moe_TB_RT:
        rt = _moe_TB_CFG // 10
        lib.moe_tb_launch(1, _moe_TB_CFG, st[f"w13b{rt}"], st[f"w13sb{rt}"], st["gs13"], st["lut"], x, meta, sroutes, inter)
        lib.moe_tb_launch(0, _moe_TB_CFG, st[f"w2b{rt}"], st[f"w2sb{rt}"], st["gs2"], st["lut"], inter, meta, sroutes, y)
        lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out)
        return out
    lib.moe_gemm13(p13, st["w13q"], st["w13s"], st["gs13"], st["lut"], x, meta, sroutes, inter)
    lib.moe_gemm2(p2, st["w2q"], st["w2s"], st["gs2"], st["lut"], inter, meta, sroutes, y)
    lib.moe_norm_launch(y, ids, tw, st["w1"], st["start"], st["count"], _moe_POST_NORM_EPS, out)
    return out

# ---- layer ----


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it)
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases, 64 in some tests
    window: int           # 2048 on sliding-window layers; 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192]
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T]
    cache_loc: torch.Tensor   # int64 [T]
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1]
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor     # float32 []
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages]
    seq_lens: torch.Tensor    # int32 [B]
    local_start: torch.Tensor # int32 [1]
    config: Config


HIDDEN = 8192
D_EXPERT = 4096
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
ROUTER_MULTIPLIER = 0.1767766952966369
ROUTE_EPS = 1e-15
ROUTER_FP32_MAX_TOKENS = 8  # MuseSpark1xRouter: FP32 F.linear up to here, BF16 hi/lo GEMMs above
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
RESIDUAL_TEMPERATURE = 0.3
RESIDUAL_GATE_EPS = 1e-3
# SGLang's fused_post_expert_norm_weighted_sum runs ~14 torch kernels up to 8 tokens (judge diag f5fdc570):
# our one-kernel version up to here (H100: 4.8 us at T<=8, 5.2 at 32 vs SGLang's Triton kernel 4.1 on H200 at 32).
OWN_WSUM_MAX_TOKENS = 16
GLUE_PDL = False
GEMV_PDL = True  # gemv10/12 trigger their dependents at start; glue after them launches with PDL (PDL_TOKENS)
PDL_TOKENS = (1, 8)
# At T=1 the unpool GEMV launches as a PDL secondary of moe_norm2 (patched to trigger at its start), so its first
# weight loads overlap norm2 (norm2 reads ~128 KB: little HBM contention).
UNPOOL_PDL = True
# GEMVs launched as PDL secondaries (they load W, then griddepcontrol.wait), per (tokens, sliding layer). H100:
# T=1 o (after csl, patched to trigger) -1..-3 us, unpool (after norm2) -3; T=8 o after sliding attention -1.9,
# after full attention +1.3; T=8 unpool tie.
# tokens -> split-K slabs: the pooler as a strided batched GEMM with FP32 out, summed in the rmsnorm (no cuBLAS
# split-K reduce kernel). H100 standalone m16/m32: 25.8/26.1 vs F.linear 28.0/28.1 us.
POOLER_BMM = {16: 2, 32: 2}
GEMV_PDL_SECONDARY = {(1, True): ("unpool_slice", "o"), (1, False): ("unpool_slice", "o"), (8, True): ("o", "unpool_slice"),
                      (8, False): ("unpool_slice",)}
BND_SPLIT = {(1, 1): 4, (1, 2): 4, (8, 1): 2, (8, 2): 2, (16, 1): 4, (16, 2): 4, (32, 1): 2, (32, 2): 2}  # s33: (tokens, 1 | 2 = first | second boundary) -> programs per row (each stores N / SPL columns)
LOCAL_FALLBACK = False  # no SGLang: also build the dequantizing reference experts (tests of the Marlin-side glue)  # PDL after a cuBLAS producer: no gain (H100 o GEMM + boundary 12.64 vs 12.74 us)


# ---- boundaries: post-norm of the branch, residual gate, pre-norm of the new stream ----

@triton.jit
def _boundary(br_ptr, s_ptr, alpha_ptr, beta_ptr, bgain_ptr, gain_ptr, s_out_ptr, h_ptr, h32_ptr,
              NB: tl.constexpr, N: tl.constexpr, BGAIN: tl.constexpr, H32: tl.constexpr,
              EPS_POST: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr, pf_ptr=None, PF: tl.constexpr = 0,
              PF_NT: tl.constexpr = 512, RB: tl.constexpr = False, TRIG: tl.constexpr = False,
              NS: tl.constexpr = 1, SS: tl.constexpr = 0, SPL: tl.constexpr = 1, SEP: tl.constexpr = False,
              sc_ptr=None, ids_ptr=None, tw_ptr=None, SC: tl.constexpr = False, SC_LOCAL: tl.constexpr = 0,
              SC_K: tl.constexpr = 4096, SC_EPS: tl.constexpr = 0.0):
    """One row. b = branch (its first NB columns; the rest are zero), times (1 + w) with BGAIN;
    s' = alpha s + beta rmsnorm(b, EPS_POST); h = rmsnorm(s', EPS) (1 + w) in BF16 (and FP32 with H32).
    RB: the branch is FP32 (gemv19 output), rounded to BF16 first.
    PF > 0: the last PF programs instead L2-prefetch PF x PF_NT x 1 KB from pf_ptr (the router weight) and exit.
    TRIG: let the PDL dependent (the gemv19 pooler, which streams W before its griddepcontrol.wait) launch now.
    NS > 1 (RB): the branch is the sum of NS FP32 split-K slabs SS elements apart (nu8).
    SPL > 1 (s33 BND_SPLIT): SPL programs per row, each computes the row's norms redundantly and stores its N / SPL slice.
    SC (s59 NU8D, with SEP): the branch is sum over the row's 8 route slots j (local: 0 <= ids < SC_LOCAL) of
    tw_j rsqrt(sc[j] / SC_K + SC_EPS) br[j] (br = nu8 D's per-slot unpool rows [T * 8][NB], sc its sums of squares)."""
    if TRIG:
        gdc_launch_dependents()
    if PF > 0:
        if tl.program_id(0) >= tl.num_programs(0) - PF:
            i = (tl.program_id(0) - (tl.num_programs(0) - PF)) * PF_NT + tl.arange(0, PF_NT)
            tl.inline_asm_elementwise("cp.async.bulk.prefetch.L2.global [$1], 1024; mov.u32 $0, 0;", "=r,l",
                                      [pf_ptr + i.to(tl.int64) * 256], dtype=tl.int32, is_pure=False, pack=1)
            return
    row = (tl.program_id(0) // SPL).to(tl.int64)
    cols = tl.arange(0, N)
    in_b = cols < NB
    mine = (cols // (N // SPL)) == tl.program_id(0) % SPL
    # Weights and the incoming stream first: with PDL they load while the branch's producer finishes.
    alpha = tl.load(alpha_ptr + cols)
    beta = tl.load(beta_ptr + cols)
    gain = tl.load(gain_ptr + cols, mask=mine, other=0.0)
    if BGAIN:
        bgain = tl.load(bgain_ptr + cols, mask=in_b, other=0.0)
    s = tl.load(s_ptr + row * N + cols)
    if SEP:  # s57: the sliced branch (NB < N) as its own [NS, NB] tile over all threads (the N-wide masked load ran on the
        jb = tl.arange(0, NB)  # ~2 warps covering cols < NB: 8 slabs there cost ~2 us at T=8)
        mine_b = (jb // (N // SPL)) == tl.program_id(0) % SPL
        alpha_b = tl.load(alpha_ptr + jb)
        beta_b = tl.load(beta_ptr + jb)
        gain_b = tl.load(gain_ptr + jb, mask=mine_b, other=0.0)
        s_b = tl.load(s_ptr + row * N + jb)
        if BGAIN:
            bgain_b = tl.load(bgain_ptr + jb)
        if SC:  # the routing (pr8's, complete long before nu8 ran) before the wait; [8, NB] tiles of per-slot values
            sl8 = row * 8 + tl.arange(0, 8)[:, None] + jb[None, :] * 0
            e8 = tl.load(ids_ptr + sl8)
            loc8 = (e8 >= 0) & (e8 < SC_LOCAL)
            tw8 = tl.load(tw_ptr + sl8)
        if PDL:
            gdc_wait()
        ks = tl.arange(0, NS)
        if SC:  # [8 slots, NB] tiles (masked to the local slots); per-slot scalars as broadcast tiles (no layout change)
            ssq = tl.load(sc_ptr + sl8, mask=loc8, other=1.0)
            sf = tl.where(loc8, tw8 * tl.rsqrt(ssq / SC_K + SC_EPS), 0.0)
            bb = tl.sum(tl.load(br_ptr + sl8 * NB + jb[None, :], mask=loc8, other=0.0) * sf, 0)
        else:
            bb = tl.sum(tl.load(br_ptr + ks[:, None] * SS + row * NB + jb[None, :]), 0) if NS > 1 else tl.load(br_ptr + row * NB + jb)
        if RB:
            bb = bb.to(tl.bfloat16)
        bb = bb.to(tl.float32)
        if BGAIN:
            bb = bb * bgain_b
        rb = tl.rsqrt(tl.sum(bb * bb, 0) / N + EPS_POST)
        s_b = alpha_b * s_b + beta_b * (bb * rb)
        s = tl.where(in_b, 0.0, alpha * s)
        tl.store(s_out_ptr + row * N + cols, s, mask=mine & (cols >= NB))
        tl.store(s_out_ptr + row * N + jb, s_b, mask=mine_b)
        rs = tl.rsqrt((tl.sum(s * s, 0) + tl.sum(s_b * s_b, 0)) / N + EPS)
        tl.store(h_ptr + row * N + cols, (s * rs * gain).to(tl.bfloat16), mask=mine & (cols >= NB))
        tl.store(h_ptr + row * N + jb, (s_b * rs * gain_b).to(tl.bfloat16), mask=mine_b)
        if H32:
            tl.store(h32_ptr + row * N + cols, s * rs * gain, mask=mine & (cols >= NB))
            tl.store(h32_ptr + row * N + jb, s_b * rs * gain_b, mask=mine_b)
        return
    if PDL:
        gdc_wait()
    b = tl.load(br_ptr + row * NB + cols, mask=in_b, other=0.0)
    for i in tl.static_range(1, NS):
        b += tl.load(br_ptr + i * SS + row * NB + cols, mask=in_b, other=0.0)
    if RB:  # FP32 branch (gemv19): round to the BF16 the GEMM would return
        b = b.to(tl.bfloat16)
    b = b.to(tl.float32)
    if BGAIN:
        b = b * bgain
    rb = tl.rsqrt(tl.sum(b * b, 0) / N + EPS_POST)
    s = alpha * s + beta * (b * rb)
    tl.store(s_out_ptr + row * N + cols, s, mask=mine)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    h = s * rs * gain
    tl.store(h_ptr + row * N + cols, h.to(tl.bfloat16), mask=mine)
    if H32:
        tl.store(h32_ptr + row * N + cols, h, mask=mine)


@triton.jit
def _rmsnorm_gain(x_ptr, gain_ptr, out_ptr, N: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr,
                  TRIG: tl.constexpr, XS: tl.constexpr = 0, ROUND: tl.constexpr = False, NS: tl.constexpr = 1,
                  SS: tl.constexpr = 0, z_ptr=None, ZN: tl.constexpr = 0, TRIG2: tl.constexpr = False,
                  zl_ptr=None, ids_ptr=None, ZL: tl.constexpr = 0, ZW: tl.constexpr = 1024):
    """One row: rmsnorm(x) (1 + w), BF16 in and out. ZN > 0: program 0 also zeroes z_ptr[:ZN] after its wait (nu8's
    sums of squares, accumulated by the following moe_tk2f). TRIG: let the dependent (moe_tk2 gemm13) launch now.
    XS: x row stride (0: N). ROUND: x is FP32, rounded to BF16 first (the pooler's output dtype).
    NS > 1: x is the sum of NS FP32 split-K partials SS elements apart (then rounded with ROUND)."""
    if TRIG:
        gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    gain = tl.load(gain_ptr + cols)
    if PDL:
        gdc_wait()
    if TRIG2:  # s27 TK2FX: the dependent (moe_tk2f) reads pr1's routing before its own wait: trigger only after ours
        gdc_launch_dependents()
    if ZN > 0:
        if tl.program_id(0) == 0:
            tl.store(z_ptr + tl.arange(0, ZN), tl.zeros((ZN,), tl.float32))
    if ZL > 0:  # s59 NU8D: zero this token's local slots' rows of nu8 D's red.add output (ids: the routing, complete)
        sl8 = row * 8 + tl.arange(0, 8)
        e8 = tl.load(ids_ptr + sl8)
        loc8 = (e8 >= 0) & (e8 < ZL)
        tl.store(zl_ptr + sl8[:, None] * ZW + tl.arange(0, ZW)[None, :], tl.zeros((8, ZW), tl.float32), mask=loc8[:, None])
        tl.store(z_ptr + sl8, tl.zeros((8,), tl.float32), mask=loc8)  # and their sums of squares
    x = tl.load(x_ptr + row * (XS if XS else N) + cols)
    for i in tl.static_range(1, NS):
        x += tl.load(x_ptr + i * SS + row * (XS if XS else N) + cols)
    if ROUND:
        x = x.to(tl.bfloat16)
    x = x.to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(out_ptr + row * N + cols, (x * r * gain).to(tl.bfloat16))


# ---- router: split-K logits, then routing ----

@triton.jit
def _router_split(h_ptr, w_ptr, wlo_ptr, part_ptr, T, K: tl.constexpr, E: tl.constexpr,
                  BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr, HILO: tl.constexpr,
                  TRIG: tl.constexpr = False):
    """part[ks, t, e] = sum over this split's K of h[t, k] w[e, k]. HILO: h BF16, w = hi + lo (BF16 pair, both
    [E, K]), FP32 accumulation; else h and w FP32 in FP32 FMA (ieee). TRIG (s39): let the PDL dependent (_routing) launch."""
    if TRIG:
        gdc_launch_dependents()
    pe = tl.program_id(0)
    ks = tl.program_id(1)
    pt = tl.program_id(2)
    e = pe * BE + tl.arange(0, BE)
    t = pt * BT + tl.arange(0, BT)
    t_ok = t < T
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    k_per = K // KS
    for k0 in tl.range(ks * k_per, ks * k_per + k_per, BK):
        k = k0 + tl.arange(0, BK)
        h = tl.load(h_ptr + t[:, None] * K + k[None, :], mask=t_ok[:, None], other=0.0)
        w = tl.load(w_ptr + e[None, :] * K + k[:, None])
        if HILO:
            wl = tl.load(wlo_ptr + e[None, :] * K + k[:, None])
            acc = tl.dot(h, w, acc)
            acc = tl.dot(h, wl, acc)
        else:
            acc = tl.dot(h, w, acc, input_precision="ieee")
    tl.store(part_ptr + (ks * T + t[:, None]) * E + e[None, :], acc, mask=t_ok[:, None])


@triton.jit
def _router_fma(h_ptr, w_ptr, part_ptr, T, K: tl.constexpr, E: tl.constexpr, BT: tl.constexpr, BE: tl.constexpr,
                BK: tl.constexpr, KS: tl.constexpr):
    """Up to BT tokens, FP32: part[ks, t, e] = sum over this split's K of h[t, k] w[e, k], in FP32 FMAs."""
    pe = tl.program_id(0)
    ks = tl.program_id(1)
    e = pe * BE + tl.arange(0, BE)
    t = tl.arange(0, BT)
    t_ok = t < T
    acc = tl.zeros((BT, BE), dtype=tl.float32)
    k_per = K // KS
    for k0 in tl.range(ks * k_per, ks * k_per + k_per, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + e[:, None] * K + k[None, :])
        h = tl.load(h_ptr + t[:, None] * K + k[None, :], mask=t_ok[:, None], other=0.0)
        acc += tl.sum(h[:, None, :] * w[None, :, :], 2)
    tl.store(part_ptr + (ks * T + t[:, None]) * E + e[None, :], acc, mask=t_ok[:, None])


@triton.jit
def _routing(part_ptr, bias_ptr, ls_ptr, ids_ptr, lids_ptr, wts_ptr, T, E: tl.constexpr, KS: tl.constexpr,
             TOPK: tl.constexpr, LOCAL: tl.constexpr, MULT: tl.constexpr, REPS: tl.constexpr,
             SKS: tl.constexpr = 0, ST: tl.constexpr = 0, TRIG: tl.constexpr = False, PDL: tl.constexpr = False):
    """One token: scores = sigmoid(MULT logits); the top TOPK of scores + bias (lowest index on ties);
    weights = picked scores / (their sum + REPS); local ids = (id - local_start) mod E, -1 off this GPU.
    TRIG (s39 ROUTE_PDL): let the PDL dependent (moe_route) launch now. PDL: launched as a PDL secondary (of pr3)."""
    if TRIG:
        gdc_launch_dependents()
    if PDL:
        gdc_wait()
    t = tl.program_id(0)
    e = tl.arange(0, E)
    logit = tl.zeros((E,), dtype=tl.float32)
    for ks in tl.static_range(KS):
        if ST:  # partial ks of token t at ks * SKS + t * ST (the fused pooler/router GEMM's FP32 output)
            logit += tl.load(part_ptr + ks * SKS + t * ST + e)
        else:
            logit += tl.load(part_ptr + (ks * T + t) * E + e)
    score = 1.0 / (1.0 + tl.exp(-logit * MULT))
    key = score + tl.load(bias_ptr + e)
    j = tl.arange(0, TOPK)
    ids = tl.zeros((TOPK,), dtype=tl.int32)
    picked = tl.zeros((TOPK,), dtype=tl.float32)
    for i in tl.static_range(TOPK):
        best = tl.max(key, 0)
        idx = tl.min(tl.where(key == best, e, E), 0)
        sc = tl.sum(tl.where(e == idx, score, 0.0), 0)
        ids = tl.where(j == i, idx, ids)
        picked = tl.where(j == i, sc, picked)
        key = tl.where(e == idx, float("-inf"), key)
    wts = picked / (tl.sum(picked, 0) + REPS)
    ls = tl.load(ls_ptr)
    loc = (ids - ls + E) % E
    loc = tl.where(loc < LOCAL, loc, -1)
    tl.store(ids_ptr + t * TOPK + j, ids)
    tl.store(lids_ptr + t * TOPK + j, loc)
    tl.store(wts_ptr + t * TOPK + j, wts)


def _router_plan(tokens):
    """(BT, BE, BK, KS, warps) for the split-K router."""
    if tokens <= ROUTER_FP32_MAX_TOKENS:  # H100 T=1: 5.8 us (cuBLAS FP32 8.1, 23.6 at T=8)
        return triton.next_power_of_2(tokens), 2, 1024, 8, 4
    if tokens <= 64:
        return 16 if tokens <= 16 else 32 if tokens <= 32 else 64, 32, 64, 16, 4
    return 64, 64, 64, 8, 4


def _route(state, h, h32, local_start, tokens, router_pdl=False, trig=False):
    bt, be, bk, ks, warps = _router_plan(tokens)
    dev = h.device
    fp32 = tokens <= ROUTER_FP32_MAX_TOKENS
    lib = _G10.get("lib") if fp32 and tokens in ROUTER3_CFG else None
    pr3 = _G10.get("lib") if not fp32 and tokens in PR3_CFG else None
    if lib is not None or pr3 is not None:
        ks = 1  # no split K: part = the logits
    rpdl = ((pr3 is not None or not fp32 and lib is None and tokens in ROUTE_PDL_SPLIT_TOKENS)
            and _opt(state, tokens, "route_pdl", ROUTE_PDL))
    part = torch.empty((ks, tokens, N_EXPERTS), dtype=torch.float32, device=dev)
    if lib is not None:
        lib.r3_router(state["router"], h32, part, *ROUTER3_CFG[tokens])
    elif pr3 is not None:
        pr3.pj_router3(state["router_hl2"], h, part.view(tokens, N_EXPERTS), *PR3_CFG[tokens])
    elif fp32:
        _router_fma[(N_EXPERTS // be, ks)](h32, state["router"], part, tokens, K=HIDDEN, E=N_EXPERTS, BT=bt, BE=be,
                                           BK=bk, KS=ks, num_warps=warps, launch_pdl=router_pdl)
    else:
        _router_split[(N_EXPERTS // be, ks, triton.cdiv(tokens, bt))](
            h, state["router_hi"], state["router_lo"], part, tokens, K=HIDDEN, E=N_EXPERTS, BT=bt, BE=be, BK=bk,
            KS=ks, HILO=True, TRIG=rpdl, num_warps=warps, launch_pdl=router_pdl)
    ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    wts = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    _routing[(tokens,)](part, state["router_bias"], local_start, ids, lids, wts, tokens, E=N_EXPERTS, KS=ks,
                        TOPK=TOP_K, LOCAL=LOCAL_EXPERTS, MULT=ROUTER_MULTIPLIER, REPS=ROUTE_EPS, TRIG=trig, PDL=rpdl,
                        num_warps=1 if tokens >= ROUTING_WARPS_1_MIN_TOKENS else 2, launch_pdl=rpdl)
    return ids, lids, wts


# ---- experts ----

@triton.jit
def _post_expert_wsum(y_ptr, wts_ptr, lids_ptr, nw_ptr, out_ptr, D: tl.constexpr, TOPK: tl.constexpr,
                      EPS: tl.constexpr):
    """One token: out = sum over its local slots k of wts[k] rmsnorm(bf16(y[k] (1 + w))), BF16 (SGLang's
    fused_post_expert_norm_weighted_sum with weight_first; slots off this GPU (local id -1) are skipped)."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, D)
    k = tl.arange(0, TOPK)
    nw = tl.load(nw_ptr + cols).to(tl.float32)
    lid = tl.load(lids_ptr + t * TOPK + k)
    wt = tl.where(lid >= 0, tl.load(wts_ptr + t * TOPK + k), 0.0)
    y = tl.load(y_ptr + (t * TOPK + k[:, None]) * D + cols[None, :], mask=(lid >= 0)[:, None], other=0.0)
    z = (y.to(tl.float32) * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    r = tl.rsqrt(tl.sum(z * z, 1) / D + EPS)
    acc = tl.sum(z * (wt * r)[:, None], 0)
    tl.store(out_ptr + t * D + cols, acc.to(tl.bfloat16))


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    import types
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _experts_marlin(x, topk_weights, local_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize; local_ids -1 off this GPU."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if num_tokens > OWN_WSUM_MAX_TOKENS:
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
            slot_expert_ids=local_ids)
    out = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _post_expert_wsum[(num_tokens,)](y, topk_weights, local_ids, state["post_expert_w"], out, D=size_k, TOPK=TOP_K,
                                     EPS=POST_NORM_EPS, num_warps=16)
    return out


# The moe agent's NVFP4 experts (moe_src.py) at these token counts, SGLang's Marlin path elsewhere. Their board
# (H200, us, SGLang incl. its torch post-norm at small T / theirs): bs1 77.3/42.5, bs8 143/106, bs16 196/159,
# bs32 204/211, bs128 466/340, bs256 617/471.
GNORM_DEFAULT = True  # s27 GNORM (H100 ab_files 4 rounds: v16 309.2/457.9 -> 309.3/455.1, v32 400.5/649.6 -> 398.4/647.6): T > 32 pre-expert rmsnorm fused into moe_gather (moe_gathern)
ROUTING_WARPS_1_MIN_TOKENS = 33  # s39: _routing on 1 warp (shuffle-only reductions) above this T: bit-identical, H100 -0.4 us per call (2.84 vs 3.26 us
# standalone); in the layer d16/d32 sliding lost 1.3/2.2 us with 1 warp (side stream, beside the g19 pooler): 2 warps there
ROUTE_PDL = True  # s39: moe_route as a PDL secondary of _routing at T > 32 (see the moe_route patch); _routing as a PDL
# secondary of the router (pr3 at T=128; _router_split at ROUTE_PDL_SPLIT_TOKENS)
ROUTE_PDL_SPLIT_TOKENS = (16, 32)
RG_DEFAULT = False  # s52: T > 32 with GNORM: route + gathern in one launch (moe_routegathern; moe s33 RG); TUNE option "rg" at
# T=256 (H100 ab v32 sl/full with twf_split 1022: -4.6/-7.1, with 66: -5.5/-8.3 us; alone ~tie). T=128: +1.8 us (the
# bucketing is hidden on the side stream there; RG puts it on the main path)
BKT_DEFAULT = True  # s27 (H100 ab_files 4 rounds: v32 404.2/652.6 -> 401.2/649.5, v16 tie; H200 router ends ~14 us before the pooler): T > 32: moe_route on the router's stream (before the join); moe_src kernel_fn takes data["bucket"]
BUCKET_SIDE_MAX_TOKENS = 16  # side-stream bucketing won at T<=8 on the judge (c17d1581); T=128/256 unproven
TK2_MAX_TOKENS = 32  # moe's TMA decode kernel (moe_tk2: route fused into gemm13, 3 launches) at T <= this
# T > 32: moe's kernel_fn = route, gather, wgmma moe_tw gemm13/gemm2, norm (r-moe c39a176)
TW_TOKENS = ()  # moe_tw at decode T: dead (s5 H100 d16 +14, d32 +29 us vs tk2)
MARLIN_TOKENS = ()  # moe 6a1cda0 at T=32 beat Marlin on the judge: decode 32 sliding 218.3 (fc3e60d0) vs 225.0 (440361a9)


def _use_moe(tokens):
    return not HAVE_SGL or tokens not in MARLIN_TOKENS


def _moe_bucket(state, local_ids, pdl=False):
    """moe_src's route kernel: this GPU's routes bucketed by expert (meta, sorted route ids). pdl (s39 ROUTE_PDL): as a
    PDL secondary of the routing kernel just before it (which triggers at its start)."""
    dev = local_ids.device
    meta = torch.empty(132, dtype=torch.int32, device=dev)
    sroutes = torch.empty(local_ids.numel(), dtype=torch.int32, device=dev)
    (state["moe"]["lib"].moe_route_pdl_launch if pdl else state["moe"]["lib"].moe_route_launch)(local_ids, 0, LOCAL_EXPERTS, meta, sroutes)
    return meta, sroutes


def _experts_moe(x, topk_weights, local_ids, state, bucket=None, gnorm=False):
    """moe_src's route -> gemm13 -> gemm2 -> norm, on local ids (-1 off this GPU: skipped as out of range)."""
    st = state["moe"]
    lib = st["lib"]
    m = x.shape[0]
    if m in TW_TOKENS:
        rt = _moe_TW_CFG // 100
        meta, sroutes = _moe_bucket(state, local_ids)
        xg = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        inter = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        y = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        out = torch.empty((m, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        lib.moe_gather_launch(x, meta, sroutes, xg)
        lib.moe_tw_launch(1, _moe_TW_CFG, st[f"w13b{rt}"], st[f"w13sb{rt}"], st["gs13"], st["lut"], xg, meta, sroutes, inter)
        lib.moe_tw_launch(0, _moe_TW_CFG, st[f"w2b{rt}"], st[f"w2sb{rt}"], st["gs2"], st["lut"], inter, meta, sroutes, y)
        lib.moe_norm_launch(y, local_ids, topk_weights, st["w1"], 0, LOCAL_EXPERTS, POST_NORM_EPS, out)
        return out
    if (m <= TK2_MAX_TOKENS and _moe_TK2_NSTG) or (m > 32 and _moe_TW_CFG):
        upf = UPF3.get(m) if m <= TK2_MAX_TOKENS else None
        if upf and upf[0] == "unpool":  # the first upf[1] bytes of every row of the BF16 unpooler (cuBLAS reads k ascending)
            u = state["unpool"]
            lib.moe_set_upf(u.data_ptr(), u.shape[0], u.shape[1] * u.element_size(), upf[1])
        elif upf:  # "g19": the first upf[1] bytes of gemv19's stage-major unpool pack (k slab outermost)
            e = state.get("g19", {}).get(("unpool", m))
            if e is None:
                upf = None
            else:
                lib.moe_set_upf(e[0].data_ptr(), min(upf[1], e[0].numel()) >> 16, 1 << 16, 1 << 16)
        _moe_TWM_ORDER_M.clear()  # s25: moe_twm item order as a TUNE option (H200 judge v43 vs moe's proxy disagree at T=256)
        if _opt(state, m, "twm_order", 0):
            _moe_TWM_ORDER_M[m] = _opt(state, m, "twm_order", 0)
        _moe_TWM_TAIL_M.clear()
        if _opt(state, m, "twm_tail", 0):  # moe s14: H200 diag 20d0b9c5 tail 32: v32 -3 us (clashes with order 76)
            _moe_TWM_TAIL_M[m] = _opt(state, m, "twm_tail", 0)
        _moe_TWQ_M.clear()
        if _opt(state, m, "twq", 0):  # moe s17: moe_twq (16 consumer warps x RT 2) instead of moe_twm; proxy v16 -4.7 H200 us
            _moe_TWQ_M[m] = _opt(state, m, "twq", 0)
        _moe_TWM_HOT_M.clear()
        if _opt(state, m, "twm_hot", 0):  # moe s17: moe_twm one-pass class for 65..96-token experts; proxy v32 -3..5 H200 us
            _moe_TWM_HOT_M[m] = 1
        _moe_HOT_M.clear()
        if _opt(state, m, "hot", 0):  # moe s29: moe_hot (65..128-token experts in one N128 pass) on that many CTAs before moe_twm
            _moe_HOT_M[m] = _opt(state, m, "hot", 0)
        _moe_TK2F_BAL_M.clear()
        if _opt(state, m, "bal", TK2F_BAL_DEFAULT):  # moe n8 BAL (balanced tk2f phases, per-segment counters) at T=8/16/32
            _moe_TK2F_BAL_M.add(m)
        _moe_TK2X_M.clear()
        if _opt(state, m, "tk2x", 0):  # s81 DIAG probes (tuner timing only, never selected)
            _moe_TK2X_M[m] = _opt(state, m, "tk2x", 0)
        _moe_TK2H_M.clear()
        if _opt(state, m, "tk2h", 0):  # s58: moe n6 tk2h (GS16 merged consumer + PF4) at T=16/32, value = 2-group weight W + 1
            _moe_TK2H_M[m] = _opt(state, m, "tk2h", 0)
        _moe_TWF_SPLIT_M.clear()
        if _opt(state, m, "twf_split", 0):  # moe s19: K-split tail of moe_twm (light experts only; combinable with order)
            _moe_TWF_SPLIT_M[m] = _opt(state, m, "twf_split", 0)
        out = _moe_kernel_fn({"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights,
                              "config": {"local_start": 0, "local_count": LOCAL_EXPERTS},
                              "bucket": bucket if m > 32 else None,
                              "gather_norm": (state["pre_expert"],) if gnorm else None, "gather_eps": RMS_EPS,
                              "rg": gnorm and bucket is None and _opt(state, m, "rg", RG_DEFAULT)}, st)
        if upf:
            lib.moe_set_upf(0, 0, 0, 0)
        return out
    dev = x.device
    inter = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((m * TOP_K, D_EXPERT), dtype=torch.bfloat16, device=dev)
    out = torch.empty((m, D_EXPERT), dtype=torch.bfloat16, device=dev)
    p13, p2 = _moe__plan(m)
    if bucket is None:
        meta, sroutes = _moe_bucket(state, local_ids)
    else:
        meta, sroutes = bucket
    lib.moe_gemm13(p13, st["w13q"], st["w13s"], st["gs13"], st["lut"], x, meta, sroutes, inter)
    lib.moe_gemm2(p2, st["w2q"], st["w2s"], st["gs2"], st["lut"], inter, meta, sroutes, y)
    lib.moe_norm_launch(y, local_ids, topk_weights, st["w1"], 0, LOCAL_EXPERTS, POST_NORM_EPS, out)
    return out


def _experts_local(x, topk_weights, local_ids, state):
    """Local test fallback (no SGLang): every local expert on every token from BF16 copies made in prepare,
    each output post-normed (scale, round to BF16, RMSNorm) and kept by its routing weight."""
    tokens = x.shape[0]
    y = torch.zeros((tokens, TOP_K, D_EXPERT), dtype=torch.bfloat16, device=x.device)
    for e, (w13, w2) in enumerate(state["experts_bf16"]):
        h = x @ w13.T
        ye = (F.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]) @ w2.T
        y += (local_ids == e)[:, :, None] * ye[:, None, :]
    out = torch.empty_like(x)
    _post_expert_wsum[(tokens,)](y, topk_weights, local_ids, state["post_expert_w"], out, D=D_EXPERT, TOPK=TOP_K,
                                 EPS=POST_NORM_EPS, num_warps=8)
    return out


# ---- prepare / kernel_fn ----

def _residual(gate):
    w = gate.float() / RESIDUAL_TEMPERATURE
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=RESIDUAL_GATE_EPS))
    return alpha.contiguous(), beta.contiguous()


def _join_builds():
    """Wait for the import-time JIT builds (s68); resolve the attention extensions."""
    for f in _PREBUILD:
        try:
            f.result()
        except Exception as e:  # noqa: BLE001  (retried serially below / at first use)
            print("layer prebuild failed (retrying serially):", e)
    for d in (_G10, _LT, _G22, _G24, _G27):
        if d.get("lib", 0) is None:
            d.pop("lib")
    g = globals()
    for n in ("_CSL", "_NVS", "_NDS", "_NVSCL"):
        if isinstance(g.get(n), _LazyBuild):
            g[n] = g[n].get()


def prepare(weights):
    router = weights["router"].contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    gain = lambda name: (1.0 + weights[name].float()).contiguous()  # noqa: E731
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler")},
        "unpool_slice": weights["unpooler"][:UNPOOL_SLICE].contiguous(),
        "unpool": weights["unpooler"].contiguous(),
        "router": router, "router_hi": hi.contiguous(), "router_lo": lo.contiguous(),
        "router_hl2": torch.stack([hi, lo]).contiguous(),  # [2, 256, k] for pj_router3
        # [pooler; router hi; router lo]: one GEMM of h for both above 8 tokens (FUSED_ROUTER_TOKENS)
        "pool_router": torch.cat([weights["pooler"], hi, lo]).contiguous() if FUSED_ROUTER_TOKENS or FR_TUNE else None,
        "router_bias": weights["router_bias"].float().contiguous(),
        "res1": _residual(weights["attn_residual_gate"]), "res2": _residual(weights["ffn_residual_gate"]),
        "pre_ffn": gain("pre_ffn_norm"), "post_ffn": gain("post_ffn_norm"), "next_pre_attn": gain("next_pre_attn_norm"),
        "pre_expert": gain("pre_expert_norm"),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight in BF16.
        "post_expert_w": (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16),
        "side_stream": torch.cuda.Stream(weights["qkv_gate"].device),
        "hi_stream": torch.cuda.Stream(weights["qkv_gate"].device, priority=-1),
    }
    _join_builds()
    state["moe"] = _moe_prepare({"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
                                 "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
                                 "post_norm_w": weights["post_expert_norm"]})
    if HAVE_SGL:
        state["bank"] = _marlin_bank(weights)
    elif LOCAL_FALLBACK:
        from nvfp4 import dequantize
        state["experts_bf16"] = [
            (dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16),
             dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16))
            for e in range(LOCAL_EXPERTS)]
    _g10_lib()
    _g12_prepare(state)
    _g19_prepare(state)
    _lt_prepare(state)
    _g22_prepare(state)
    torch.cuda.synchronize()
    _tune(state)
    return state


# ---- BF16 GEMV at 1 token: the proj agent's gemv10 (g10_src.cu = r-proj d894f4e, inlined by build.py) ----

_G10_SRC = r'''// r-proj d894f4e muse_bf16_proj/submission.py _G10_SRC (proj agent's gemv10), copied verbatim
// gemv10: grid-stride GEMV for small m, W plain row-major BF16 [n, KK], KK <= U*2048.
// The flat W is cut into chunks of U*256*8 elements (U*4 KB), each a whole number of rows. A block is NV groups
// of 256 threads sharing one copy of x in smem; group g of block b is "virtual block" v = b*NV + g and takes chunks
// v, v+G, v+2G, ... (G = grid*NV): at any time the grid reads one contiguous window of W, like a pure grid-stride
// reader. Thread t loads the U uint4 at u*2048 + t*8 of the chunk (PF chunks in flight), dots them with x in
// FP32, warps reduce by shuffles, the group sums each row's warps in smem (named barrier) and writes BF16.
// No split-K, no atomics, no workspace. Big blocks keep the per-block x reads (k*2*m bytes each) small.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g10 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ void dot8(float& a, const uint4& w, const uint4& x) {
    a = fmaf(lo(w.x), lo(x.x), a); a = fmaf(hi(w.x), hi(x.x), a);
    a = fmaf(lo(w.y), lo(x.y), a); a = fmaf(hi(w.y), hi(x.y), a);
    a = fmaf(lo(w.z), lo(x.z), a); a = fmaf(hi(w.z), hi(x.z), a);
    a = fmaf(lo(w.w), lo(x.w), a); a = fmaf(hi(w.w), hi(x.w), a);
}
template <int M, int U, int NV, int MINB, int KK, int PF>
__global__ void __launch_bounds__(256 * NV, MINB) gemv(const uint4* __restrict__ W, const uint4* __restrict__ x,
                                                       __nv_bfloat16* __restrict__ out, int n) {
    constexpr int CE = U * 2048, RPC = CE / KK, K8 = KK / 8;
    constexpr int NA = KK >= 2048 ? CE / KK : U;      // accumulators per thread (distinct rows among its u)
    constexpr int EPR = KK >= 2048 ? 8 : KK / 256;    // red entries per row
    static_assert(CE % KK == 0, "chunk must hold whole rows");
    const int t = threadIdx.x & 255, vg = threadIdx.x >> 8, lane = t & 31, wp = t >> 5;
    const int nch = n / RPC;
    extern __shared__ uint4 xs[];  // x[M][KK]
    __shared__ float red_[NV][2][NA * 8][M];
    float (*red)[NA * 8][M] = red_[vg];
    const int G = gridDim.x * NV;
    int c = blockIdx.x * NV + vg;
    uint4 w[PF][U];
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) w[p][u] = ldw(W + (long)(c + p * G) * (CE / 8) + u * 256 + t);
        }
    for (int i = threadIdx.x; i < M * K8; i += 256 * NV) xs[i] = __ldg(x + i);
    __syncthreads();
    int par = 0;
    for (; c < nch; c += PF * G) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cc = c + p * G;
            if (cc < nch) {
                float acc[NA][M];
#pragma unroll
                for (int a = 0; a < NA; ++a)
#pragma unroll
                    for (int i = 0; i < M; ++i) acc[a][i] = 0.f;
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int kx = (u * 256 + t) % K8, a = KK >= 2048 ? u * 2048 / KK : u;
#pragma unroll
                    for (int i = 0; i < M; ++i) dot8(acc[a][i], w[p][u], xs[i * K8 + kx]);
                }
                const int cn = cc + PF * G;
                if (cn < nch) {
#pragma unroll
                    for (int u = 0; u < U; ++u) w[p][u] = ldw(W + (long)cn * (CE / 8) + u * 256 + t);
                }
#pragma unroll
                for (int a = 0; a < NA; ++a)
#pragma unroll
                    for (int i = 0; i < M; ++i) {
                        float v = acc[a][i];
#pragma unroll
                        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                        if (lane == 0) red[par][a * 8 + wp][i] = v;
                    }
                asm volatile("bar.sync %0, 256;" :: "r"(vg + 1));
                if (t < RPC * M) {
                    const int r = t / M, i = t % M;
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < EPR; ++q) s += red[par][r * EPR + q][i];
                    out[(long)i * n + (long)cc * RPC + r] = __float2bfloat16(s);
                }
                par ^= 1;
            }
        }
    }
}
}  // namespace g10
void g10_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t u, int64_t nv, int64_t minb, int64_t pf,
              int64_t blocks) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int sm = m * k * 2;
#define G10_CASE(M_, K_, U_, V_, B_, P_) \
    if (m == M_ && k == K_ && u == U_ && nv == V_ && minb == B_ && pf == P_) { \
        if (sm > 48 * 1024) cudaFuncSetAttribute(g10::gemv<M_, U_, V_, B_, K_, P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g10::gemv<M_, U_, V_, B_, K_, P_><<<blocks, 256 * V_, sm, s>>>((const uint4*)w.data_ptr(), (const uint4*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), n); return; }
    G10_CONFIGS
    TORCH_CHECK(false, "g10: no config");
}
'''
if UNPOOL_PDL:
    _i = _moe_CUDA_SRC.index("{\n", _moe_CUDA_SRC.index("__global__ void __launch_bounds__(512) moe_norm2(")) + 2
    _moe_CUDA_SRC = _moe_CUDA_SRC[:_i] + '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n' + _moe_CUDA_SRC[_i:]
# moe_tk2 (decode experts, T <= 32) as PDL secondaries: gemm13 (TK2_PDL13) launches while _rmsnorm_gain (patched
# to trigger at its start) runs; its producer warp streams the first weight stages and does the in-kernel routing
# before the consumers' griddepcontrol.wait (right before their first x read). gemm2 (TK2_PDL2) waits at entry.
POOLER_HI_PRIORITY = ()  # s5 dead: pooler first, but the router starves beside it (tie/worse)
# boundary 1 L2-prefetches the FP32 router weight (8 MB) from extra programs at these token counts
ROUTER_L2_PF_TOKENS = ()  # judge f8e06892: d1 -2% (noise?); off. Was (1,): H100: d1 -0.8 us; T=8 +0.9 (the T=8 router is issue-bound, not HBM)
ROUTER_AFTER_POOLER = ()  # s5: tie/worse (router starved beside the pooler); token counts: pooler first, router its PDL secondary (see kernel_fn)
# above 8 tokens: pooler + SGLang's hi/lo BF16 router in one cuBLAS GEMM (FP32 out), no side stream. s7 H100: tie
# (d32 GEMM 30.8 + splitK reduce 2.1 + routing 3.4 vs pooler || router 38.3 us; v32 -2..-10, v16 full +5): the fused
# GEMM is the pooler + 3 us (router bytes at HBM rate), same as the overlap. Needs a better GEMM config to win (lt FP32 out).
FUSED_ROUTER_TOKENS = (256,)  # s37: + moe_gathern on its strided FP32 rows; H100 ab_files v32 sl/full -2.4/-2.5, -5.4/-4.4 us (2 runs); T=128 +3.5/0
FR_TUNE = (128, 256)  # s37: T where TUNE option "fr" (fused pooler+router GEMM, default T in FUSED_ROUTER_TOKENS) exists
TK2_PDL13 = False  # s5: H100 tie (d1 +1, v1 -1, d32 -1 us): rmsnorm is only 1.7 us and the safe join loses its own PDL
TK2_PDL2 = False
if TK2_PDL13 or TK2_PDL2:
    _i0 = _moe_CUDA_SRC.index("__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2(")
    _i1 = _moe_CUDA_SRC.index("static int launch_tk2(", _i0)
    _k = _moe_CUDA_SRC[_i0:_i1]  # the moe_tk2 kernel
    _a = "  uint4 xf[NQ][2];\n"
    assert _k.count(_a) == 1
    _k = _k.replace(_a, _a + '  if (GATED) asm volatile("griddepcontrol.wait;" ::: "memory");\n')
    _a = "  } else {\n    if (tid < 33) s_off[tid] = meta[33 + tid];\n"
    assert _k.count(_a) == 1
    _k = _k.replace(_a, _a.replace("{\n", '{\n    asm volatile("griddepcontrol.wait;" ::: "memory");\n'))
    _moe_CUDA_SRC = _moe_CUDA_SRC[:_i0] + _k + _moe_CUDA_SRC[_i1:]
    _a = """  moe_tk2<NRT, NC, NSTG, GATED><<<grid, (NC + 1) * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const float*)gs, (const __nv_bfloat16*)x,
      (const int*)ids, nroutes, start, count, (int*)meta, (int*)sroutes, (int*)inv, out);
"""
    assert _moe_CUDA_SRC.count(_a) == 1
    _moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, """  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(grid); lc.blockDim = dim3((NC + 1) * 32); lc.dynamicSmemBytes = SMEM; lc.stream = st;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la;
  lc.numAttrs = GATED ? %d : %d;
  cudaLaunchKernelEx(&lc, moe_tk2<NRT, NC, NSTG, GATED>,
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const float*)gs, (const __nv_bfloat16*)x,
      (const int*)ids, nroutes, start, count, (int*)meta, (int*)sroutes, (int*)inv, out);
""" % (int(TK2_PDL13), int(TK2_PDL2)))
_G12_SRC = r'''// r-proj e71d015 muse_bf16_proj/dev/gemv12.cu (proj agent's gemv12, m=8 tensor-core streaming GEMM), copied verbatim
// gemv12: tensor-core streaming GEMM for m = 8..32 (out[m, n] = x[m, k] @ W[n, k]^T, BF16 in, FP32 accumulate).
// mma.sync m16n8k16 with x as A (tokens) and W as B. W is repacked in prepare() into 512-B atoms = 8 rows x 32 k,
// lane l holding row l/4, k (l%4)*8..+8 (one uint4 = the B fragments of two k16 mmas; x's A fragments use the same
// k permutation); layout [ks][rg][KC/32 atoms][lane] (ks = k-chunk of KC = 1024*KCU, rg = 8-row group).
// Block b = (ks = b / bk, a run of NG rgs); it is NG groups of WG warps, group gi streams rg0+gi's KC/32 atoms
// (warp wi takes atoms wi, wi+WG, ...; PF in flight in registers). x[:, chunk] is staged in smem first (cp.async;
// it must not queue behind the W stream). Each warp's FP32 C fragments go to its own smem slot; WG slots summed.
// One k-chunk: BF16 written directly. Else FP32 partials [S][m][n] in a workspace, a per-rg counter, and the last
// of the S blocks of a rg sums them in FP32 and writes BF16 (counters re-zeroed).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g12 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int M, int KCU, int WG, int PF>
__global__ void __launch_bounds__(1024, 1) gemm(const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                int* __restrict__ cnt, int n, int nrg_all, int k, int bk) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16, XS = KC * 2 + 64;  // x row stride in smem (bytes), padded
    constexpr int NT = M < 16 ? M : 16, NA = KC / 32 / WG;               // atoms per warp
    static_assert(NA % PF == 0, "PF must divide the atoms per warp");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x[M][KC] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [warps][TT][32 lanes] C fragments
    __shared__ int flag[64];
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x / bk, rg0 = (blockIdx.x % bk) * NG, rg = rg0 + gi;
    for (int i = t; i < M * (KC / 8); i += blockDim.x) {
        const int r = i / (KC / 8), q = i % (KC / 8);
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + q * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"((const uint4*)(x + (long)r * k + (long)ks * KC) + q));
    }
    asm volatile("cp.async.commit_group;");
    const uint4* wb = W + (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wb + j*WG*32
    uint4 w[PF];
#ifndef G12_LATE_W
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
#endif
    asm volatile("cp.async.wait_all;");
    __syncthreads();
#ifdef G12_LATE_W
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
#endif
    float acc[TT][4];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
    for (int a = 0; a < NA; a += PF) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                uint4 xa = *(const uint4*)(xs + (tt * 16 + g) * XS + kb * 2);
                uint4 xb = make_uint4(0, 0, 0, 0);
                if (NT > 8) xb = *(const uint4*)(xs + (tt * 16 + g + 8) * XS + kb * 2);
                mma(acc[tt], xa.x, xb.x, xa.y, xb.y, w[p].x, w[p].y);
                mma(acc[tt], xa.z, xb.z, xa.w, xb.w, w[p].z, w[p].w);
            }
            if (a + p + PF < NA) w[p] = ldw(wb + (a + p + PF) * WG * 32);
        }
    }
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) accp[(wp * TT + tt) * 32 + lane] = make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]);
    __syncthreads();
    // outputs i = ((rgl * M + tok) * 8 + row8): sum the WG warp slots of group rgl
    const int ne = NG * M * 8;
    auto total = [&](int i) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        const int tt = tok >> 4, h = (tok >> 3) & 1, l = (tok & 7) * 4 + (row8 >> 1), e = h * 2 + (row8 & 1);
        const float* base = (const float*)accp + ((rgl * WG) * TT + tt) * 128 + l * 4 + e;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[q * TT * 128];
        return s;
    };
    if (S == 1) {
        for (int i = t; i < ne; i += blockDim.x) {
            const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(total(i));
        }
        return;
    }
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        __stcg(ws + ((long)ks * M + tok) * n + (rg0 + rgl) * 8 + row8, total(i));
    }
    __threadfence();
    __syncthreads();
    if (t < NG) {
        const int last = atomicAdd(cnt + rg0 + t, 1) == S - 1;
        flag[t] = last;
        if (last) cnt[rg0 + t] = 0;
    }
    __syncthreads();
    __threadfence();
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        if (flag[rgl]) {
            float s = 0.f;
            for (int q = 0; q < S; ++q) s += __ldcg(ws + ((long)q * M + tok) * n + (rg0 + rgl) * 8 + row8);
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(s);
        }
    }
}
}  // namespace g12
// nrg_all: 8-row groups of the packed weight; bk: blocks per k-chunk; each block takes n/8/bk rgs with WG warps each.
void g12_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t nrg_all,
              int64_t kcu, int64_t wg, int64_t pf, int64_t bk) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int S = k / (1024 * kcu), ng = n / 8 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 64) + threads / 32 * ((m + 15) / 16) * 512;
    TORCH_CHECK(n / 8 % bk == 0 && threads <= 1024 && ng <= 64 && sm <= 227 * 1024 && (S == 1 || m * n * S <= 8 * 32 * 8192), "g12: shape");
#define G12_CASE(M_, U_, G_, P_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_) { \
        cudaFuncSetAttribute(g12::gemm<M_, U_, G_, P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g12::gemm<M_, U_, G_, P_><<<S * bk, threads, sm, s>>>((const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk); return; }
    G12_CONFIGS
    TORCH_CHECK(false, "g12: no config");
}
'''
# proj's gemv13 (m = 1) / gemv14 (m = 8) over losslessly compressed 12-bit BF16 weights (g13_src.cu / g14_src.cu =
# r-proj 3f173eb: bit-exact escape headers, inlined by build.py; _compress/_pack13/_pack14 below = proj's). Only where the weight compresses
# (else gemv10/12 as before). H200 (proj judge): pooler m1 17.8 -> 15.5, qkv m1 11.5 -> 11.0, pooler m8 18.8 -> 16.3,
# qkv m8 13.9 -> 13.0 us.
_G13_SRC = r'''// gemv13: gemv10 (grid-stride GEMV, m = 1) over losslessly compressed BF16 weights: 12 bits per weight.
// prepare() splits W [n, KK] into a sign|mantissa byte plane (flat, row-major) and a 4-bit exponent code plane:
// code c = e - base (base = max exponent - 15), so the decoded exponent is 112 | c (exact power-of-2 rescale of the
// sum by 2^(base - 112)). Weights with e < base (zeros, tiny values: ~1 per 8K) are escapes: code 0, byte 0 (decodes
// to D = 2^(base - 127) in true scale) and a per-row header of ESC (col, true / scale) pairs (exact) fixes the sum: + x[col] * (true / scale - 2^-15).
// Nibble layout: 32-bit word f/8 holds weights f..f+7 (f % 8 == 0); weight 2p + h at bits 4p + 16h, so pair p's
// exponents land at BF16 bits 7..10 / 23..26 with one shift. Decode per pair: prmt (sign replicate), shift, 2 lop3.
// Chunk = U * 4096 weights (whole rows); thread t takes the 16 weights at u * 4096 + t * 16 (one uint4 of bytes and
// one uint2 of codes per u). Its k positions are fixed, so x lives in registers as FP32.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g13 {
#define XI(k) (((k) >> 4) * 20 + ((k) & 15))
constexpr int ESC = 8, MAXC = 16;  // escapes per row; chunks per virtual block
__device__ __forceinline__ uint4 ld4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld2(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
// pair p (0..3) of a nibble word: exponent bits for BF16 pair at bits 7..10 and 23..26, OR 112 << 7
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
// 4 weights (bytes of sw) -> 2 BF16 pairs; nibble word nw, pairs P0, P0 + 1
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = decw<P0>(sw, nw, 0x9180u, 0x4140u);
    const unsigned q1 = decw<P0 + 1>(sw, nw, 0xB3A2u, 0x4342u);
    f[0] = __uint_as_float(q0 << 16); f[1] = __uint_as_float(q0 & 0xffff0000u);
    f[2] = __uint_as_float(q1 << 16); f[3] = __uint_as_float(q1 & 0xffff0000u);
}
__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* xp) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) *reinterpret_cast<float4*>(x + i) = *reinterpret_cast<const float4*>(xp + i);
    float f[16];
    dec4<0>(s.x, nb.x, f); dec4<2>(s.y, nb.x, f + 4); dec4<0>(s.z, nb.y, f + 8); dec4<2>(s.w, nb.y, f + 12);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) { a0 = fmaf(f[i], x[i], a0); a1 = fmaf(f[i + 1], x[i + 1], a1); }
    return a0 + a1;
}

template <int U, int NV, int MINB, int KK, int PF>
__global__ void __launch_bounds__(256 * NV, MINB) gemv(const uint4* __restrict__ S, const uint2* __restrict__ N,
                                                       const float2* __restrict__ H, const __nv_bfloat16* __restrict__ x,
                                                       __nv_bfloat16* __restrict__ out, int n, float scale) {
    constexpr int CE = U * 4096, RPC = CE / KK;
    static_assert(CE % KK == 0 && KK % 512 == 0, "chunk must hold whole rows");
    const int t = threadIdx.x & 255, vg = threadIdx.x >> 8, lane = t & 31, wp = t >> 5;
    const int nch = n / RPC;
    __shared__ float red_[NV][2][U * 8];
    __shared__ __align__(16) float xs[KK / 16 * 20];  // 16-float groups padded to 20: no bank conflicts
    __shared__ float2 hs_[NV][MAXC * RPC * ESC];  // this vb's escape headers: chunk j = j-th of c, c + G, ...
    float (*red)[U * 8] = red_[vg];
    const int G = gridDim.x * NV;
    int c = blockIdx.x * NV + vg;
    uint4 s[PF][U];
    uint2 nb[PF][U];
    // escapes: the row's reducing thread adds x[col] * (value - 2^-15) per header entry to the row's sum
    auto load = [&](int p, int ch) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long i = (long)ch * (CE / 16) + u * 256 + t;
            s[p][u] = ld4(S + i);
            nb[p][u] = ld2(N + i);
        }
    };
    // x and the escape headers first: they must not queue behind the W prologue
    constexpr int XV = (KK / 8 + 256 * NV - 1) / (256 * NV), HV = (MAXC * RPC * ESC + 255) / 256;
    uint4 xv[XV];
    float2 hv[HV];
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * 256 * NV;
        if (i < KK / 8) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);
    }
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * 256, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hv[q] = __ldg(H + (long)ch * RPC * ESC + i % (RPC * ESC));
    }
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * 256, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hs_[vg][i] = hv[q];
    }
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * 256 * NV;
        if (i < KK / 8) {
            const unsigned v[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) { const int kk = XI(8 * i + 2 * j); xs[kk] = __uint_as_float(v[j] << 16); xs[kk + 1] = __uint_as_float(v[j] & 0xffff0000u); }
        }
    }
    __syncthreads();
    int par = 0, j = 0;
    for (; c < nch; c += PF * G) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cc = c + p * G;
            if (cc < nch) {
                float acc[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    acc[u] = dot16(s[p][u], nb[p][u], xs + XI((u * 4096 + t * 16) % KK));
                }
                const int cn = cc + PF * G;
                if (cn < nch) load(p, cn);
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    float v = acc[u];
#pragma unroll
                    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                    if (lane == 0) red[par][u * 8 + wp] = v;
                }
                asm volatile("bar.sync %0, 256;" :: "r"(vg + 1));
                if (t < RPC) {
                    float sum = 0.f;
#pragma unroll
                    for (int u = 0; u < U; ++u)
#pragma unroll
                        for (int w = 0; w < 8; ++w)
                            if ((u * 4096 + w * 512) / KK == t) sum += red[par][u * 8 + w];
#ifndef NOESC
                    // escapes on the row's full sum, two fmas each: exact when the row holds only that weight (probes)
                    const float2* he = hs_[vg] + ((j + p) * RPC + t) * ESC;
                    for (int e = 0; e < ESC; ++e) {
                        const float2 h = he[e];
                        if (h.y == 0x1p-15f) break;  // slots fill in order; empty = 2^-15 (real: |value| < 2^-15)
                        const float xe = xs[XI(__float_as_int(h.x))];
                        sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
                    }
#endif
                    out[(long)cc * RPC + t] = __float2bfloat16(sum * scale);
                }
                par ^= 1;
            }
        }
        j += PF;
    }
}
}  // namespace g13

void g13_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
              int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks) {
    const int k = x.size(1), n = out.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(x.size(0) == 1, "g13: m == 1");
    TORCH_CHECK((long)n * k <= (long)g13::MAXC * blocks * nv * u * 4096, "g13: too many chunks per virtual block");
#define G13_CASE(K_, U_, V_, B_, P_) \
    if (k == K_ && u == U_ && nv == V_ && minb == B_ && pf == P_) { \
        g13::gemv<U_, V_, B_, K_, P_><<<blocks, 256 * V_, 0, st>>>((const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), \
        (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }
    G13_CONFIGS
    TORCH_CHECK(false, "g13: no config");
}
'''
_G14_SRC = r'''// gemv14: gemv12 (tensor-core streaming GEMM, m = 8..32) over losslessly compressed BF16 weights, 12 bits each
// (the format of gemv13: sign|mantissa bytes + 4-bit exponent codes c = e - base, decoded exponent 112 | c, the sum
// rescaled by 2^(base - 112); weights below the window are escapes fixed up from a per-row header).
// Per 512-B atom of gemv12 (8 rows x 32 k), lane l's 8 weights are one uint2 of bytes (S plane) and one 32-bit code
// word (N plane), [ks][rg][atom][lane]; they decode (prmt sign-replicate + 3 logic ops per BF16 pair) straight into
// the uint4 of B fragments. Escapes: the block's rows' headers (ESC x (col, delta / scale)) are cp.async'd to smem
// up front; in the epilogue each output adds x[tok, col] * delta for the escapes whose col is in the block's
// k-chunk (x[:, chunk] is fully staged: XR == XP). Partials stay in the scaled domain; the final sum is rescaled.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g14 {
constexpr int ESC = 8;
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld2(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned ld1(const unsigned* p) {
    unsigned v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(uint2 s, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s.x, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s.x, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s.y, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s.y, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int M, int KCU, int WG, int PF, int XP, int XR>
__global__ void __launch_bounds__(1024, 1) gemm(const uint2* __restrict__ WS, const unsigned* __restrict__ WN,
                                                const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                int* __restrict__ cnt, int n, int nrg_all, int k, int bk) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16;
    constexpr int NT = M < 16 ? M : 16, NA = KC / 32 / WG;               // atoms per warp
    static_assert(NA % (PF * XP) == 0 && XR == XP && XP <= 4, "x pieces: fully staged");
    constexpr int PA = NA / XP, PB = KC / XP * 2;  // atoms per x piece (per warp), bytes per x piece row
    constexpr int XS = XR * PB + 64;                // x ring row stride in smem (bytes), padded: XR pieces of PB
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x ring [M][XR pieces] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [warps][TT][32 lanes] C fragments
    float2* hs = (float2*)(smem + M * XS + blockDim.x / 32 * TT * 512);  // [NG * 8 rows][ESC] escape headers
    __shared__ int flag[64];
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x / bk, rg0 = (blockIdx.x % bk) * NG, rg = rg0 + gi;
    // x[:, chunk] in XP pieces along k through a ring of XR piece buffers (cp.async, one commit group per piece),
    // the first XR issued before any W load: the block starts on piece 0 while the rest lands.
    auto stage = [&](int p) {
        for (int i = t; i < M * (PB / 16); i += blockDim.x) {
            const int r = i / (PB / 16), q = i % (PB / 16);
            const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + (p % XR) * PB + q * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa),
                         "l"((const uint4*)(x + (long)r * k + (long)ks * KC + p * (KC / XP)) + q));
        }
    };
    // escape headers ride in x piece 0's commit group
    for (int i = t; i < NG * 8 * ESC / 2; i += blockDim.x) {
        const unsigned sa = (unsigned)__cvta_generic_to_shared(hs + 2 * i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(H + (long)rg0 * 8 * ESC + 2 * i));
    }
#pragma unroll
    for (int p = 0; p < XR; ++p) { stage(p); asm volatile("cp.async.commit_group;"); }
    const long wo = (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wo + j*WG*32
    const uint2* wbs = WS + wo;
    const unsigned* wbn = WN + wo;
    uint2 ws_[PF];
    unsigned wn_[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) { ws_[p] = ld2(wbs + p * WG * 32); wn_[p] = ld1(wbn + p * WG * 32); }
    float acc[TT][4];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
    for (int a = 0; a < NA; a += PF) {
        if (a % PA == 0) {  // piece pc = a / PA must have landed (groups complete in order)
            const int pc = a / PA;
            if (XR == XP) {     // everything staged up front
                const int left = XP - 1 - pc;
                if (left >= 3) asm volatile("cp.async.wait_group 3;");
                else if (left == 2) asm volatile("cp.async.wait_group 2;");
                else if (left == 1) asm volatile("cp.async.wait_group 1;");
                else asm volatile("cp.async.wait_group 0;");
                __syncthreads();
            } else {            // ring: committed XR + pc - 1 groups (one per boundary), piece pc is group pc
                if (pc == 0) {
                    if (XR == 4) asm volatile("cp.async.wait_group 3;");
                    else if (XR == 3) asm volatile("cp.async.wait_group 2;");
                    else asm volatile("cp.async.wait_group 1;");
                } else {
                    if (XR == 4) asm volatile("cp.async.wait_group 2;");
                    else if (XR == 3) asm volatile("cp.async.wait_group 1;");
                    else asm volatile("cp.async.wait_group 0;");
                }
                __syncthreads();  // piece pc visible to all; everyone is done with piece pc - 1
                if (pc > 0 && pc - 1 + XR < XP) stage(pc - 1 + XR);
                if (pc > 0) asm volatile("cp.async.commit_group;");
            }
        }
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint4 wv = dec8(ws_[p], wn_[p]);
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
#pragma unroll
            for (int tt = 0; tt < TT; ++tt) {
                const int kx = (kb - (a / PA) * (KC / XP) + ((a / PA) % XR) * (KC / XP)) * 2;
                uint4 xa = *(const uint4*)(xs + (tt * 16 + g) * XS + kx);
                uint4 xb = make_uint4(0, 0, 0, 0);
                if (NT > 8) xb = *(const uint4*)(xs + (tt * 16 + g + 8) * XS + kx);
                mma(acc[tt], xa.x, xb.x, xa.y, xb.y, wv.x, wv.y);
                mma(acc[tt], xa.z, xb.z, xa.w, xb.w, wv.z, wv.w);
            }
            if (a + p + PF < NA) { ws_[p] = ld2(wbs + (a + p + PF) * WG * 32); wn_[p] = ld1(wbn + (a + p + PF) * WG * 32); }
        }
    }
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) accp[(wp * TT + tt) * 32 + lane] = make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]);
    __syncthreads();
    // outputs i = ((rgl * M + tok) * 8 + row8): sum the WG warp slots of group rgl
    const int ne = NG * M * 8;
    auto total = [&](int i) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        const int tt = tok >> 4, h = (tok >> 3) & 1, l = (tok & 7) * 4 + (row8 >> 1), e = h * 2 + (row8 & 1);
        const float* base = (const float*)accp + ((rgl * WG) * TT + tt) * 128 + l * 4 + e;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[q * TT * 128];
        const float2* he = hs + (rgl * 8 + row8) * ESC;
#pragma unroll
        for (int e = 0; e < ESC; ++e) {
            const int col = __float_as_int(he[e].x) - ks * KC;
            if (col >= 0 && col < KC) {  // two fmas: exact when the dot holds only this weight (probe rows)
                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2));
                s = fmaf(xv, he[e].y, fmaf(xv, -0x1p-15f, s));
            }
        }
        return s;
    };
    if (S == 1) {
        for (int i = t; i < ne; i += blockDim.x) {
            const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(total(i) * scale);
        }
        return;
    }
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        __stcg(ws + ((long)ks * M + tok) * n + (rg0 + rgl) * 8 + row8, total(i));
    }
    __threadfence();
    __syncthreads();
    if (t < NG) {
        const int last = atomicAdd(cnt + rg0 + t, 1) == S - 1;
        flag[t] = last;
        if (last) cnt[rg0 + t] = 0;
    }
    __syncthreads();
    __threadfence();
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        if (flag[rgl]) {
            float s = 0.f;
            for (int q = 0; q < S; ++q) s += __ldcg(ws + ((long)q * M + tok) * n + (rg0 + rgl) * 8 + row8);
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(s * scale);
        }
    }
}
}  // namespace g14
// nrg_all: 8-row groups of the packed weight; bk: blocks per k-chunk; each block takes n/8/bk rgs with WG warps each.
void g14_gemm(torch::Tensor ws_, torch::Tensor wn, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              torch::Tensor ws, torch::Tensor cnt, int64_t nrg_all, int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int S = k / (1024 * kcu), ng = n / 8 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 64) + threads / 32 * ((m + 15) / 16) * 512 + ng * 8 * g14::ESC * 8;
    TORCH_CHECK(n / 8 % bk == 0 && threads <= 1024 && ng <= 64 && sm <= 227 * 1024 && (S == 1 || m * n * S <= 8 * 32 * 8192), "g14: shape");
#define G14_CASE(M_, U_, G_, P_, X_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_ && xp == X_) { \
        cudaFuncSetAttribute(g14::gemm<M_, U_, G_, P_, X_, X_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g14::gemm<M_, U_, G_, P_, X_, X_><<<S * bk, threads, sm, s>>>((const uint2*)ws_.data_ptr(), (const unsigned*)wn.data_ptr(), \
        (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk); return; }
    G14_CONFIGS
    TORCH_CHECK(false, "g14: no config");
}
'''
_G19_SRC = r'''// gemv19: gemv18's TMA-ring streamer over lossless 12-bit weights (m = 16/32, same _pack18 stage layout), FP32 output
// only, with load balancing. Per-SM HBM bandwidth varies by ~10% (gemv18 blocks ended 21-27 us at pooler m16 with
// equal static ranges), so: each block first streams a static stream-K range of the first U - TAIL stages, then grabs
// tail items of IS stages from a global counter until none are left. The producer warp decides everything: it tags
// every ring slot with its stage u (+ x buffer, + "run ends here" flag); consumers flush partials at run ends; x is
// double-buffered and loaded by the producer when a stage needs another k-chunk. All partials go to the FP32 output
// by red.global.add (finisher warp); a tiny zero kernel (PDL primary) zeroes the output and the counter first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g19 {
constexpr int ESC = 8, NWC = 8;
#ifdef G19_TRACE
__device__ unsigned long long trace[132 * 64];
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#define TR(slot) trace[blockIdx.x * 64 + (slot)] = gt()
#else
#define TR(slot)
#endif
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void ldsm2(unsigned addr, unsigned& d0, unsigned& d1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(d0), "=r"(d1) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW19_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W19_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// M tokens (8/16/32), KC k per chunk, AP atoms (32 k) per warp per stage, NS ring stages, IS stages per tail item.
// out FP32 [M, n] (zeroed by the primary), ctr int (zeroed by the primary). tail: stages left for dynamic items.
// smem: ring [NS][SB] | x [2][M][XS] | pbuf [2][128 rows][M] f32 | barriers | tag [NS] int | ptag [2] int
template <int M, int KC, int AP, int NS, int IS, int PRE, int XB>
__global__ void __launch_bounds__((NWC + 2) * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H,
                                                         float scale, const __nv_bfloat16* __restrict__ x,
                                                         float* __restrict__ out, int* __restrict__ ctr, int n, int k, int tail, int npre) {
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = NWC * AP * 768, XS = KC * 2 + 16, PBF = NWC * 16 * M;
    static_assert(SPK <= 32 && NA % AP == 0 && SPK % IS == 0, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), xs = ring + NS * SB;
    float* pbuf = (float*)(smem + NS * SB + XB * M * XS);
    const unsigned bars = (unsigned)__cvta_generic_to_shared(pbuf + 2 * PBF);
    int* tag = (int*)(pbuf + 2 * PBF) + 2 * (2 * NS + 8);  // after the barriers. per slot: u | xb << 29 | last x use << 30 | run end << 31 (-1: no more work)
    int* ptag = tag + NS;                                 // per pbuf: tile (-1: consumers done)
    int* tag2 = ptag + 2;  // proj s51 TAG2: per slot ks | st << 8 | tr << 13, written by the producer (no runtime div/mod in the consumers)
    // full[NS], empty[NS], xfull[2], xempty[2], pfull[2], pempty[2]
    const unsigned xfull = bars + 16 * NS, xempty = xfull + 16, pfull = xfull + 32, pempty = xfull + 48;
    // s57 (layer): npre bit 16 = trigger the PDL dependent at start (the boundary after the unpool: its pre-wait loads
    // never touch out, and it can only launch once every CTA here is resident, so the epoch barrier cannot starve)
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const int nrg = n / 16, ntr = nrg / NWC, S = k / KC, KA = k / 32;
    const int U = S * ntr * SPK, US = U - tail;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NWC); }
        for (int i = 0; i < 2; ++i) {
            bar_init(xfull + 8 * i, 1); bar_init(xempty + 8 * i, NWC); bar_init(pfull + 8 * i, NWC); bar_init(pempty + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
#ifdef G19_TRACE
    if (t == 0) { unsigned sm_; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_)); trace[blockIdx.x * 64 + 10] = sm_; }
#endif
    if (w == NWC) {  // producer: one lane walks the static range, then the tail items
        if (lane == 0) {
            int ra = (int)((long)US * blockIdx.x / gridDim.x), rb = (int)((long)US * (blockIdx.x + 1) / gridDim.x);
            int xks[2] = {-1, -1}, xuse[2] = {0, 0}, xb = 1, i = 0;
            if (PRE && npre > 0 && ra < rb) {  // out zeroed long before: stream the first stages' W while the previous kernel runs
                const int ks0 = ra / (ntr * SPK);
                for (; i < npre && ra < rb && ra / (ntr * SPK) == ks0; ++i, ++ra) {
                    const int st = ra % SPK;
                    const bool xl = ra + 1 == rb || (ra + 1) / (ntr * SPK) != ks0;
                    tag[i] = ra | (xl ? 1 << 30 : 0) | (ra + 1 == rb || st + 1 == SPK ? 1 << 31 : 0);  // x buffer 0
                    tag2[i] = ks0 | st << 8 | ((ra % (ntr * SPK)) / SPK) << 13;
                    bar_expect(bars + 8 * i, SB);
                    bulk(ring + i * SB, WP + (long)ra * SB, SB, bars + 8 * i);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                xb = 0; xks[0] = ks0; xuse[0] = 1;
                bar_expect(xfull, M * KC * 2);
                for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k + (long)ks0 * KC, KC * 2, xfull);
            } else {
                asm volatile("griddepcontrol.wait;" ::: "memory");  // x, out, ctr ready
            }
            for (;;) {
                if (ra >= rb) {  // next run: a tail item
                    if (tail == 0) break;
                    const int it = atomicAdd(ctr, 1);  // grabbed when needed: the ring's NS stages hide its latency
                    if (it * IS >= tail) break;
                    ra = US + it * IS; rb = ra + IS;
                }
                for (int u = ra; u < rb; ++u, ++i) {
                    const int ks = u / (ntr * SPK), st = u % SPK;
                    if (ks != xks[xb]) {  // next x buffer, load chunk ks (consumers release a buffer after its last use)
                        xb = XB == 2 ? xb ^ 1 : 0;
                        if (xuse[xb]) bar_wait(xempty + 8 * xb, (xuse[xb] - 1) & 1);
                        ++xuse[xb];
                        xks[xb] = ks;
                        bar_expect(xfull + 8 * xb, M * KC * 2);
                        for (int qq = 0; qq < M; ++qq) bulk(xs + xb * M * XS + qq * XS, x + (long)qq * k + (long)ks * KC, KC * 2, xfull + 8 * xb);
                    }
                    const int slot = i % NS;
                    if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                    const bool end = u + 1 == rb || st + 1 == SPK;
                    const bool xl = u + 1 == rb || (u + 1) / (ntr * SPK) != ks;
                    if (xl) xks[xb] = -1;  // released by the consumers after this stage: reload even if the next run has this ks
                    tag[slot] = u | xb << 29 | (xl ? 1 << 30 : 0) | (end ? 1 << 31 : 0);
                    tag2[slot] = ks | st << 8 | ((u % (ntr * SPK)) / SPK) << 13;
                    const unsigned full = bars + 8 * slot;
                    bar_expect(full, SB);
                    bulk(ring + slot * SB, WP + (long)u * SB, SB, full);
                }
                ra = rb;
            }
            const int slot = i % NS;  // end marker
            if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
            tag[slot] = -1;
            bar_arrive(bars + 8 * slot);
            TR(1);
        }
        return;
    }
    if (w == NWC + 1) {  // finisher: pbuf [128 rows][M] -> out [M][n] by red.global.add.v4 (lane = token, 4 rows)
        asm volatile("griddepcontrol.wait;" ::: "memory");
        unsigned target = 0;
        if (PRE == 2) {  // zero a slice of out, then arrive on the epoch counter ctr[1] (+gridDim.x per launch, never reset)
            float4* o4 = (float4*)out;
            for (int i = blockIdx.x * 32 + lane; i < M * n / 4; i += gridDim.x * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            __syncwarp();
            if (lane == 0) {
                __threadfence();
                const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
                target = old - old % gridDim.x + gridDim.x;
            }
        }
        for (int q = 0;; ++q) {
            const int pb = q & 1;
            bar_wait(pfull + 8 * pb, (q >> 1) & 1);
            const int tr = ptag[pb];
            if (tr < 0) break;
            if (PRE == 2 && q == 0) {  // every block has zeroed its slice
                if (lane == 0) {
                    unsigned v;
                    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - target) < 0);
                }
                __syncwarp();
            }
            const float* src = pbuf + pb * PBF;
            float* wt = out + (long)tr * NWC * 16;
#pragma unroll 4
            for (int i = lane; i < PBF / 4; i += 32) {
                const int tok = i % M, r4 = (i / M) * 4;
                const float a0 = src[r4 * M + tok] * scale, a1 = src[(r4 + 1) * M + tok] * scale;
                const float a2 = src[(r4 + 2) * M + tok] * scale, a3 = src[(r4 + 3) * M + tok] * scale;
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(wt + (long)tok * n + r4), "f"(a0), "f"(a1), "f"(a2), "f"(a3) : "memory");
            }
            __syncwarp();
            if (lane == 0) bar_arrive(pempty + 8 * pb);
        }
        return;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = acc2[j][0] = acc2[j][1] = acc2[j][2] = acc2[j][3] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    int q = 0, st0 = 0, prev_end = 1, prev_xl = 1;
    unsigned xph[2] = {0, 0};
    float4 hd0, hd1;
    for (int i = 0;; ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const int tg = tag[slot];
        if (tg == -1) break;
        const int u = tg & 0x1FFFFFFF, xb = (tg >> 29) & 1;
        const bool end = tg < 0, xl = (tg >> 30) & 1;
        const int t2 = tag2[slot], ks = t2 & 0xFF, st = (t2 >> 8) & 31, tr = t2 >> 13;
        if (prev_xl) {  // first stage of an x use: wait for it
            bar_wait(xfull + 8 * xb, xph[xb]);
            xph[xb] ^= 1;
        }
        prev_xl = xl;
        if (prev_end) {  // run start: its escape headers (rows g, g + 8 of the warp's row group; lane c: slots 2c, 2c + 1)
            st0 = st;
            const float2* hp0 = H + (long)((tr * NWC + w) * 16 + g) * ESC;
            hd0 = __ldg((const float4*)hp0 + c);
            hd1 = __ldg((const float4*)(hp0 + 8 * ESC) + c);
        }
        prev_end = end;
        const unsigned xbase = xs + xb * M * XS + xrow;
        const unsigned sb = ring + slot * SB + w * AP * 512 + lane * 16, nb = ring + slot * SB + NWC * AP * 512 + w * AP * 256 + lane * 8;
#pragma unroll
        for (int j = 0; j < AP; ++j) {
            uint4 sv;
            uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
            const unsigned kb = (st * AP + j) * 64;  // byte offset of the atom's k within the chunk
#ifdef G19_NOMMA
            acc[0][j & 3] += __uint_as_float(sv.x ^ sv.y ^ sv.z ^ sv.w ^ nv.x ^ nv.y); continue;
#endif
#ifdef G19_NOX
            { const uint4 af = dec8(sv.x, sv.y, nv.x), af2 = dec8(sv.z, sv.w, nv.y);
              acc[0][j & 3] += __uint_as_float(af.x ^ af.y ^ af.z ^ af.w ^ af2.x ^ af2.y ^ af2.z ^ af2.w); continue; }
#endif
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint4 af = dec8(s ? sv.z : sv.x, s ? sv.w : sv.y, s ? nv.y : nv.x);
#pragma unroll
                for (int jj = 0; jj + 1 < NT; jj += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(xbase + jj * 8 * XS + kb + s * 32, b0, b1, b2, b3);
                    mma(s ? acc2[jj] : acc[jj], af, b0, b1);
                    mma(s ? acc2[jj + 1] : acc[jj + 1], af, b2, b3);
                }
                if constexpr (NT & 1) {  // layer s38: M = 8 (one n8 token group; lanes 0-15 address it)
                    unsigned b0, b1;
                    ldsm2(xbase + (NT - 1) * 8 * XS + kb + s * 32, b0, b1);
                    mma(s ? acc2[NT - 1] : acc[NT - 1], af, b0, b1);
                }
            }
        }
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
        if (!end) continue;  // (a last x use always ends a run: released below, after the escape fixups read x)
        // run end: rows rg*16 + g (+8), tokens jj*8 + 2c (+1), k in [klo, khi)
        const int rg = tr * NWC + w, nat = (st - st0 + 1) * AP;
        const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + (st + 1) * AP * 32;
        const bool sfix = !(nat == KA && PRE != 2);  // pbuf path: escape fixups applied in smem after the store (below)
        float v[NT][4];
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) { v[jj][e] = acc[jj][e] + acc2[jj][e]; acc[jj][e] = acc2[jj][e] = 0.f; }
        if (!sfix) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float4 hq = h ? hd1 : hd0;
#pragma unroll
                for (int e2 = 0; e2 < ESC; ++e2) {  // empty slot = (col 0, 2^-15): skipped (real escapes: |value| < 2^-15)
                    const int src = (lane & ~3) | (e2 >> 1);
                    const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                    const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                    const int col = __float_as_int(hx);
                    if (hy != 0x1p-15f && col >= klo && col < khi) {  // two fmas: exact when the dot holds only this weight (probe rows)
                        const unsigned char* xc = smem + NS * SB + xb * M * XS + (col - ks * KC) * 2;
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                                v[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, v[jj][2 * h + b]));
                            }
                    }
                }
            }
        }
        if (xl && !sfix) {  // done with this x buffer (the fixups above were its last reads)
            __syncwarp();
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
        const int row0 = rg * 16 + g;
        if (nat == KA && PRE != 2) {  // the whole dot: plain stores (PRE 2: out is being zeroed by other blocks)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) out[(long)(jj * 8 + 2 * c + (e & 1)) * n + row0 + 8 * (e >> 1)] = v[jj][e] * scale;
            continue;
        }
        const int pb = q & 1;  // hand the partials to the finisher: pbuf[pb][w * 16 + row16][tok]
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
        float* pw = pbuf + pb * PBF + (w * 16 + g) * M + 2 * c;
#pragma unroll
        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h) *(float2*)(pw + 8 * h * M + jj * 8) = make_float2(v[jj][2 * h], v[jj][2 * h + 1]);
        // escape fixups on the stored partials, one escape at a time over the warp (lane = token): the in-range slots
        // (lane c holds slots 2c, 2c + 1 of rows g, g + 8) are found by ballots and walked in lane order, warp-uniformly
        // (r-proj gemv19.cu s15: the old per-slot shfl/branch chain cost ~1 us per run end)
        {
            __syncwarp();
            const unsigned char* xt = smem + NS * SB + xb * M * XS + lane * XS - ks * KC * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int pr = 0; pr < 2; ++pr) {
                    const float4 hq = h ? hd1 : hd0;
                    const float hx = pr ? hq.z : hq.x, hy = pr ? hq.w : hq.y;
                    unsigned b = __ballot_sync(0xffffffffu, hy != 0x1p-15f && __float_as_int(hx) >= klo && __float_as_int(hx) < khi);
                    while (b) {
                        const int src = __ffs(b) - 1;
                        b &= b - 1;
                        const int col = __float_as_int(__shfl_sync(0xffffffffu, hx, src));
                        const float hv = __shfl_sync(0xffffffffu, hy, src);
                        if (lane < M) {
                            float* pp = pbuf + pb * PBF + (w * 16 + (src >> 2) + 8 * h) * M + lane;
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xt + col * 2));
                            *pp = fmaf(xv, hv, fmaf(xv, -0x1p-15f, *pp));
                        }
                    }
                }
            __syncwarp();
        }
        if (xl) {  // done with this x buffer (the fixups above were its last reads)
            if (lane == 0) bar_arrive(xempty + 8 * xb);
        }
        if (t == 0) ptag[pb] = tr;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
        ++q;
    }
    {  // done: tell the finisher
        const int pb = q & 1;
        if (q >= 2) bar_wait(pempty + 8 * pb, ((q >> 1) - 1) & 1);
        if (t == 0) ptag[pb] = -1;
        __syncwarp();
        if (lane == 0) bar_arrive(pfull + 8 * pb);
    }
    if (t == 0) TR(4);
}
// PDL primary: lets the gemm launch at once, zeroes its output and counter (the gemm waits before touching them)
__global__ void zero(float4* __restrict__ p, int n4, int* __restrict__ ctr) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (blockIdx.x == 0 && threadIdx.x == 0) *ctr = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
// test predecessor: lets its dependent launch, then writes x = src (n8 uint4) and spins ~ns_ nanoseconds
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
// diagnostics: a bare TMA ring stream (no math) of `total` bytes, NS stages of SB bytes, one 288-thread block per SM
__global__ void __launch_bounds__(288, 1) stream(const unsigned char* W, long total, int SB, int NS, float* out) {
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), bars = ring + NS * SB;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31;
    const long U = total / SB, u0 = U * blockIdx.x / gridDim.x, u1 = U * (blockIdx.x + 1) / gridDim.x;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (w == 8) {
        if (lane == 0)
            for (long u = u0, i = 0; u < u1; ++u, ++i) {
                const int slot = i % NS;
                if (i >= NS) bar_wait(bars + 8 * (NS + slot), ((i / NS) - 1) & 1);
                bar_expect(bars + 8 * slot, SB);
                bulk(ring + slot * SB, W + u * SB, SB, bars + 8 * slot);
            }
        return;
    }
    float acc = 0.f;
    for (long u = u0, i = 0; u < u1; ++u, ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const float4* p = (const float4*)(smem + slot * SB);
        for (int j = t; j < SB / 16; j += 256) { const float4 v = p[j]; acc += v.x + v.w; }
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
    }
    if (acc == 1.2345f) out[0] = acc;
}
// BF16 copy of a gemm's FP32 output, as its PDL dependent (for consumers that need BF16)
__global__ void cvt(const float4* __restrict__ src, uint2* __restrict__ dst, int n4) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const float4 v = src[i];
        const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
        dst[i] = make_uint2(*(const unsigned*)&a, *(const unsigned*)&b);
    }
}
}  // namespace g19
void g19_cvt(torch::Tensor src, torch::Tensor dst) {
    const int n4 = src.numel() / 4;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((n4 + 255) / 256 < 132 ? (n4 + 255) / 256 : 132); cfg.blockDim = dim3(256);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, g19::cvt, (const float4*)src.data_ptr(), (uint2*)dst.data_ptr(), n4);
}
void g19_stream(torch::Tensor w, int64_t sb, int64_t ns, torch::Tensor out) {
    const int sm = sb * ns + 16 * ns;
    cudaFuncSetAttribute(g19::stream, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    g19::stream<<<132, 288, sm, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)w.data_ptr(), w.numel(), sb, ns, out.data_ptr<float>());
}
void g19_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g19::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g19_zero(torch::Tensor out, torch::Tensor ctr) {
    g19::zero<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((float4*)out.data_ptr(), out.numel() / 4, ctr.data_ptr<int>());
}
#ifdef G19_TRACE
torch::Tensor g19_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g19::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
// mode 0: a zero kernel (out, ctr) first, the gemm its PDL dependent. mode 1: out (and ctr if tail) already zero
// (stream-ordered before the previous kernel), the gemm a PDL dependent of the previous kernel, streaming W early.
// xb: x buffers (2: the next k-chunk's x loads while the current one is in use; 1: less smem, the load waits).
// npre (modes 1, 2): weight stages streamed before griddepcontrol.wait (0: wait first, x first; use 0 when the
// previous kernel leaves nothing to overlap). mode 2: as 1, but the gemm zeroes out itself (grid barrier on the epoch counter ctr[1]: one int per call site
// and grid size, any initial multiple of the grid, e.g. 0; tail must be 0).
// out: FP32 [m, n] (overwritten), ctr: int32 [>= 1] scratch. grid: blocks (<= #SMs, 1 per SM). tail: dynamic stages.
void g19_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t is, int64_t grid, int64_t tail, int64_t mode, int64_t npre, int64_t xb) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = ns * g19::NWC * ap * 768 + xb * m * (kc * 2 + 16) + 2 * g19::NWC * 16 * m * 4 + 4 * (ns + 2) + 8 * (2 * ns + 8) + 8 + 4 * ns;
    TORCH_CHECK(n % (16 * g19::NWC) == 0 && k % kc == 0 && sm <= 227 * 1024 && tail % is == 0 && (mode < 2 || tail == 0) && (npre & 0xffff) <= ns, "g19: shape");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g19::zero<<<32, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4, ctr.data_ptr<int>());
#ifdef G19_ONLY2
    TORCH_CHECK(mode == 2, "g19: built for mode 2 only");
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_) g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_>
#else
#define G19_KERN(M_, KC_, AP_, NS_, IS_, XB_) (mode == 2 ? g19::gemm<M_, KC_, AP_, NS_, IS_, 2, XB_> : mode ? g19::gemm<M_, KC_, AP_, NS_, IS_, 1, XB_> : g19::gemm<M_, KC_, AP_, NS_, IS_, 0, XB_>)
#endif
#define G19_CASE(M_, KC_, AP_, NS_, IS_, XB_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && is == IS_ && xb == XB_) { \
        auto kern = G19_KERN(M_, KC_, AP_, NS_, IS_, XB_); \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        /* layer s62: npre bit 17 = the whole SM's shared memory as carveout (room for a PDL dependent's CTAs) */ \
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, (npre >> 17) & 1 ? 100 : -1); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3((g19::NWC + 2) * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)tail, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G19_CONFIGS
    TORCH_CHECK(false, "g19: no config");
}
'''
_G24_SRC = r'''// gemv24 (s53): m = 16/32 GEMM over lossless 12-bit weights (DEC3 layout) with PER-WARP TMA rings: no producer warp,
// no block-wide stage barriers. Every warp streams its own split-K tasks (16 rows x KC k = KC/32 atoms, contiguous:
// pack19 with rg 1) through its own NS-slot ring (lane 0 issues one bulk copy per slot, refills a slot right after
// consuming it), so a slow warp never gates the others (g19's per-stage producer/consumer handoff is the target).
// Tasks: t = ks * (n / 16) + row group; block b owns tasks [T b / G, T (b + 1) / G) (static), warp w every WPB-th;
// the block's x chunks (<= XB) are loaded once after griddepcontrol.wait. out FP32 [M, n] zeroed in the kernel
// (epoch-counter grid barrier before the first red), partials by red.global.add.v4 through a per-warp smem tile.
// SK (s54, template RR = R | 256): stream-K over stages instead (each warp an equal contiguous stage range, partial
// flushes at task boundaries): H200 proxy wins at m32 / qkv m16, loses at m16 pool/unp/o (NOTES-proj s54).
// Escape fixups at each task end (g19's register path); G24_NOFIX skips them (x @ W'^T, escaped weights = 2^-15 * scale).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g24 {
#ifdef G24_TRACE
__device__ unsigned long long trace[4096 * 3];
#endif
constexpr int WB = 768, ESC = 8;  // bytes per 16 x 32 atom
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x4140u);
    r.y = decw<1>(s0, nw, 0x4342u);
    r.z = decw<2>(s1, nw, 0x4140u);
    r.w = decw<3>(s1, nw, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count)); }
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW24_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W24_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// smem: ring [WPB][NS][SB] | x [XB][M][XS] | stage [WPB][M][STR] f32 | full [WPB][NS] | xfull [XB] | target
// R: row groups per task (one x fragment load feeds R atoms); stage (ks, tile of R row groups, st) =
// [R][AP][32 lanes x 16 B sign|mantissa] [R][AP][32 x 8 B codes] (pack19 with rg = R).
template <int M, int KC, int AP, int NS, int WPB, int XB, int RR>
__global__ void __launch_bounds__(WPB * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int* __restrict__ ctr, int n, int k, int npre, float* __restrict__ zo, int zn4) {
    constexpr int R = RR & 0xff;
    constexpr bool SK = RR >> 8;
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = R * AP * WB, XS = KC * 2 + 16, STR = 16 * R + 4;
    static_assert(NA % AP == 0 && NT % 2 == 0 && M <= 32, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    // npre bit 16: trigger the PDL dependent at start (the layer's g19 convention, s57: dependents that only touch out
    // after their own griddepcontrol.wait)
    if (npre & 0x10000) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // npre bit 17 (proj s56 EZ): out arrives zeroed by an earlier g24 of the same layer call (its zo), complete before
    // our griddepcontrol.wait returns (every kernel in between waits on its predecessor): no zero/fence/epoch barrier
    const bool ez = (npre >> 17) & 1, ezpf = ez && ((npre >> 18) & 1);
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem) + w * NS * SB, xs = (unsigned)__cvta_generic_to_shared(smem) + WPB * NS * SB;
    float* stg = (float*)(smem + WPB * NS * SB + XB * M * XS) + w * M * STR;
    const unsigned fb = (unsigned)__cvta_generic_to_shared(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4) + w * NS * 8;
    const unsigned xfb = (unsigned)__cvta_generic_to_shared(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4) + WPB * NS * 8;
    int* tgt = (int*)(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4 + WPB * NS * 8 + XB * 8);
    const int nrt = n / (16 * R), S = k / KC, T = S * nrt;
    // SK (s54, RR bit 8) = stream-K over stages: block b owns global stages [SG b / G, SG (b + 1) / G) (stage g = task
    // g / SPK, st g % SPK), warp w a contiguous 1/WPB of them, flushing its partials at every task boundary it crosses
    const long SG = (long)T * SPK;
    const int sa = SK ? (int)(SG * blockIdx.x / gridDim.x) : 0, sz = SK ? (int)(SG * (blockIdx.x + 1) / gridDim.x) : 0;
    const int wa = sa + (int)((long)(sz - sa) * w / WPB);
    const int ta = SK ? sa / SPK : (int)((long)T * blockIdx.x / gridDim.x);
    const int tb = SK ? (sz + SPK - 1) / SPK : (int)((long)T * (blockIdx.x + 1) / gridDim.x);
    const int ks0 = ta / nrt;
    const int nq = SK ? 0 : tb - ta > w ? (tb - ta - w + WPB - 1) / WPB : 0;
    const int total = SK ? sa + (int)((long)(sz - sa) * (w + 1) / WPB) - wa : nq * SPK;
    auto issue = [&](int j) {  // lane 0: this warp's sequence element j -> slot j % NS
        const int tq = SK ? (wa + j) / SPK : ta + w + (j / SPK) * WPB, st = SK ? (wa + j) % SPK : j % SPK, sl = j % NS;
#ifdef G24_NOW
        bar_arrive(fb + 8 * sl);
#else
        bar_expect(fb + 8 * sl, SB);
        bulk(ring + sl * SB, WP + ((long)tq * SPK + st) * SB, SB, fb + 8 * sl);
#endif
    };
    int jn = 0;  // next sequence element to issue
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) for (int b = 0; b < XB; ++b) bar_init(xfb + 8 * b, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (; jn < total && jn < npre; ++jn) issue(jn);  // W only, before the predecessor is done
    }
    if (ezpf)  // proj s56 (npre bit 18): out (zeroed long before, maybe evicted) back into L2 for the red.adds; prefetch only
        for (int i = (blockIdx.x * WPB * 32 + t) * 32; i < M * n; i += gridDim.x * WPB * 32 * 32)  // (no data dependence)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(out + i));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (t == 0) {  // the block's x chunks
        const int ksl = (tb - 1) / nrt;
        for (int b = 0; b < XB && ks0 + b <= ksl; ++b) {
            bar_expect(xfb + 8 * b, M * KC * 2);
            for (int qq = 0; qq < M; ++qq) bulk(xs + b * M * XS + qq * XS, x + (long)qq * k + (long)(ks0 + b) * KC, KC * 2, xfb + 8 * b);
        }
    }
    if (lane == 0) for (; jn < total && jn < NS; ++jn) issue(jn);
    jn = __shfl_sync(0xffffffffu, jn, 0);
    if (!ez) {  // zero this block's slice of out, then arrive on the epoch counter ctr[1] (+gridDim.x per launch, never reset)
        float4* o4 = (float4*)out;
        for (int i = blockIdx.x * WPB * 32 + t; i < M * n / 4; i += gridDim.x * WPB * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        __threadfence();
        __syncthreads();
        if (t == 0) {
            const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
            *tgt = (int)(old - old % gridDim.x + gridDim.x);
        }
        __syncthreads();
    }
    const int target = ez ? 0 : *tgt;
    bool passed = ez;
#ifdef G24_TRACE
    unsigned long long tr0;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr0));
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[R][NT][4], acc2[R][NT][4];
    int j = 0;
    for (int q = 0; SK ? j < total : q < nq; ++q) {
        const int tq = SK ? (wa + j) / SPK : ta + w + q * WPB;
        const int st0 = SK ? (wa + j) % SPK : 0, st1 = SK ? min(SPK, st0 + total - j) : SPK;
        const int ks = tq / nrt, tr = tq % nrt, xb = ks - ks0;
#ifndef G24_NOFIX
        // escape headers of rows g, g + 8 of each row group (lane c: slots 2c, 2c + 1), loaded early: used at the task end
        float4 hd[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) hd[r][h] = __ldg((const float4*)(H + (long)((tr * R + r) * 16 + g + 8 * h) * ESC) + c);
#endif
        bar_wait(xfb + 8 * xb, 0);
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[r][jj][e] = acc2[r][jj][e] = 0.f;
        const unsigned xbase = xs + xb * M * XS + xrow;
        for (int st = st0; st < st1; ++st, ++j) {
            const int sl = j % NS;
            bar_wait(fb + 8 * sl, (j / NS) & 1);
            const unsigned sb = ring + sl * SB + lane * 16, nb = ring + sl * SB + R * AP * 512 + lane * 8;
            constexpr int G = R > 1 ? 1 : AP < 2 ? AP : 2;
#pragma unroll
            for (int j0 = 0; j0 < AP; j0 += G) {
                uint4 sv[G][R];
                uint2 nv[G][R];
#pragma unroll
                for (int a = 0; a < G; ++a)
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[a][r].x), "=r"(sv[a][r].y), "=r"(sv[a][r].z), "=r"(sv[a][r].w) : "r"(sb + (r * AP + j0 + a) * 512));
                        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[a][r].x), "=r"(nv[a][r].y) : "r"(nb + (r * AP + j0 + a) * 256));
                    }
                unsigned bx[G][2][NT][2];
#pragma unroll
                for (int a = 0; a < G; ++a)
#pragma unroll
                    for (int s = 0; s < 2; ++s)
#pragma unroll
                        for (int jj = 0; jj < NT; jj += 2)
                            ldsm4(xbase + jj * 8 * XS + (st * AP + j0 + a) * 64 + s * 32, bx[a][s][jj][0], bx[a][s][jj][1], bx[a][s][jj + 1][0], bx[a][s][jj + 1][1]);
#pragma unroll
                for (int a = 0; a < G; ++a)
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const uint4 af = dec8(s ? sv[a][r].z : sv[a][r].x, s ? sv[a][r].w : sv[a][r].y, s ? nv[a][r].y : nv[a][r].x);
#pragma unroll
                            for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[r][jj] : acc[r][jj], af, bx[a][s][jj][0], bx[a][s][jj][1]);
                        }
            }
            __syncwarp();
            if (jn < total) {  // refill the slot just read (generic-proxy reads before the async-proxy write)
                if (lane == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    issue(jn);
                }
                ++jn;
            }
        }
        // task end: partials (16 R rows x M tokens) -> stage tile [tok][row] -> red.add.v4 into out [tok][n]
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[r][jj][e] += acc2[r][jj][e];
#ifndef G24_NOFIX
        {  // escape fixups (g19's register path): escaped weights were decoded as 2^-15 (scale units); add x * (true - 2^-15)
            const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + st1 * AP * 32;
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float4 hq = hd[r][h];
                unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
                unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
                if (!(f0 | f1)) continue;
                f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
                f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#pragma unroll
                for (int e2 = 0; e2 < ESC; ++e2) {
                    if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
                    const int src = (lane & ~3) | (e2 >> 1);
                    const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                    const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                    const int col = __float_as_int(hx);
                    if (hy != 0x1p-15f && col >= klo && col < khi) {
                        const unsigned char* xc = smem + WPB * NS * SB + xb * M * XS + (col - ks * KC) * 2;
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                                acc[r][jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[r][jj][2 * h + b]));
                            }
                    }
                }
            }
        }
#endif
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + r * 16 + g + 8 * (e >> 1)] = acc[r][jj][e] * scale;
        if (!passed) {  // every block has zeroed its slice of out
            if (lane == 0) {
                unsigned v;
                do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - (unsigned)target) < 0);
            }
            passed = true;
        }
        __syncwarp();
#pragma unroll
        for (int i = lane; i < M * 4 * R; i += 32) {
            const int tok = i / (4 * R), r4 = (i % (4 * R)) * 4;
            const float4 a = *(const float4*)(stg + tok * STR + r4);
            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(out + (long)tok * n + tr * 16 * R + r4), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w) : "memory");
        }
        __syncwarp();
    }
    // proj s56 ZN: zero the next g24's out (zo, zn4 float4s) as each warp finishes; plain stores, no fence: the next g24
    // reads it only after its griddepcontrol.wait, i.e. after this grid has completed
    for (int i = (blockIdx.x * WPB + w) * 32 + lane; i < zn4; i += gridDim.x * WPB * 32) ((float4*)zo)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef G24_TRACE
    if (lane == 0) {
        unsigned long long tr1;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr1));
        const int id = blockIdx.x * WPB + w;
        unsigned smid_;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_));
        trace[3 * id] = tr0; trace[3 * id + 1] = tr1; trace[3 * id + 2] = (unsigned long long)total | ((unsigned long long)smid_ << 32);
    }
#endif
}
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
}  // namespace g24
#ifdef G24_TRACE
torch::Tensor g24_trace() {
    auto t = torch::empty({4096 * 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(t.data_ptr(), g24::trace, sizeof(unsigned long long) * 4096 * 3, 0, cudaMemcpyDeviceToDevice);
    return t;
}
#else
torch::Tensor g24_trace() { return torch::empty({0}); }
#endif
void g24_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g24::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g24_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t wpb, int64_t xb, int64_t grid, int64_t npre, int64_t rr, torch::Tensor zo) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sk = rr >> 8;
    rr &= 0xff;
    const int sm = wpb * ns * rr * ap * g24::WB + xb * m * (kc * 2 + 16) + wpb * m * (16 * rr + 4) * 4 + wpb * ns * 8 + xb * 8 + 16;
    const int nrg = n / (16 * rr), T = (k / kc) * nrg;
    for (int b = 0; b < grid; ++b) {  // x chunks per block <= xb
        const long spk = kc / 32 / ap, SG = (long)T * spk;
        const int ta = sk ? (int)(SG * b / grid / spk) : (int)((long)T * b / grid);
        const int tb = sk ? (int)((SG * (b + 1) / grid + spk - 1) / spk) : (int)((long)T * (b + 1) / grid);
        TORCH_CHECK(tb == ta || (tb - 1) / nrg - ta / nrg < xb, "g24: block spans more than xb x chunks");
    }
    TORCH_CHECK(n % (16 * rr) == 0 && k % kc == 0 && sm <= 227 * 1024, "g24: shape/smem ", sm);
    TORCH_CHECK(zo.numel() % 4 == 0 && (!zo.numel() || (zo.is_contiguous() && zo.scalar_type() == torch::kFloat32)), "g24: zo");
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G24_CASE(M_, KC_, AP_, NS_, WPB_, XB_, R_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && wpb == WPB_ && xb == XB_ && (rr | sk << 8) == R_) { \
        auto kern = g24::gemm<M_, KC_, AP_, NS_, WPB_, XB_, R_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(WPB_ * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
                           out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)npre, zo.numel() ? zo.data_ptr<float>() : nullptr, (int)(zo.numel() / 4)); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G24_CONFIGS
    TORCH_CHECK(false, "g24: no config");
}
'''
_G27_SRC = r'''#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g27 {
#ifdef G27_TRACE
__device__ unsigned long long trace[1024 * 6];
#define TR(i) if (t == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v_)); trace[blockIdx.x * 6 + (i)] = v_; }
#define TRW(i) if (lane == 0 && w == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v_)); trace[blockIdx.x * 6 + (i)] = v_; }
#else
#define TR(i)
#define TRW(i)
#endif
#ifdef G24_TRACE
__device__ unsigned long long trace[4096 * 3];
#endif
constexpr int WB = 768, ESC = 8;  // bytes per 16 x 32 atom
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x4140u);
    r.y = decw<1>(s0, nw, 0x4342u);
    r.z = decw<2>(s1, nw, 0x4140u);
    r.w = decw<3>(s1, nw, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count)); }
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW27_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W27_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}

#ifndef G27_MC
#define G27_MC 1  // s58: 1 = x by bulk copies, one per (row, k slice), spread over threads (layer d16 -1.8 sl); 0 = cp.async xload; >1 cluster multicast
#endif
// s58 MC: x rows multicast into all G27_MC CTAs of the cluster (same smem offsets / barrier offset in each CTA)
__device__ __forceinline__ void bulk_mc(unsigned dst, const void* src, int bytes, unsigned b) {
#if G27_MC > 1
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b), "h"((unsigned short)((1u << G27_MC) - 1)) : "memory");
#else
    bulk(dst, src, bytes, b);
#endif
}
__device__ __forceinline__ unsigned cl_rank() {
#if G27_MC > 1
    unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
#else
    return 0;
#endif
}
__device__ __forceinline__ void cl_sync() {
#if G27_MC > 1
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}
// x [m, k] -> smem rows of XS bytes by all threads: 16 B cp.async (L2, .cg), then wait + block barrier
__device__ __forceinline__ void xload(unsigned xs, const __nv_bfloat16* x, int m, int k, int XS, int t, int nt) {
    const int r16 = k / 8;
#ifdef G27_XHALF
    m /= 2;  // timing probe only
#endif
    for (int i = t; i < m * r16; i += nt) {
        const int row = i / r16, col = i % r16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(xs + row * XS + col * 16), "l"(x + (long)row * k + col * 8) : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}
// gemv27 (s57): no cross-block split-K. Block b owns RGB whole row groups (16 rows each, all of k); its WPB = RGB * KQ
// warps each stream one (row group, k / KQ slice) through a per-warp NS-slot TMA ring (pack24 with kc = k / KQ, r 1),
// partials go through smem (aliasing the warp's ring), the block sums its KQ partials in a fixed order and writes
// out (FP32 or BF16) with plain stores: no out zeroing, no fence, no grid epoch barrier, no atomics; deterministic.
// x [M, k] whole per block (after griddepcontrol.wait); W stages issued before the wait (npre, PDL secondary).
template <int M, int KQ, int AP, int NS, int RGB, bool OBF>
__global__ void __launch_bounds__(RGB * KQ * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                         void* __restrict__ outp, int n, int k, int npre, float* __restrict__ zo, int zn4) {
    constexpr int WPB = RGB * KQ, NT = M / 8, SB = AP * WB, STR = 16 + 4;
    static_assert(NT % 2 == 0 && M <= 32 && NS * SB >= M * STR * 4, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    TR(0)
    const int rgl = w % RGB, kq = w / RGB, rg = blockIdx.x * RGB + rgl, nrt = n / 16;
    const int KW = k / KQ, SPK = KW / 32 / AP, XS = k * 2 + 16;
    const unsigned s0 = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned ring = s0 + w * NS * SB, xs = s0 + WPB * NS * SB;
    const unsigned fb = xs + M * XS + w * NS * 8, xfb = xs + M * XS + WPB * NS * 8;
    const unsigned char* wsrc = WP + ((long)kq * nrt + rg) * SPK * SB;
    auto issue = [&](int j) {
        bar_expect(fb + 8 * (j % NS), SB);
        bulk(ring + (j % NS) * SB, wsrc + (long)j * SB, SB, fb + 8 * (j % NS));
    };
    int jn = 0;
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) bar_init(xfb, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (; jn < SPK && jn < npre && jn < NS; ++jn) issue(jn);  // W only, before the predecessor is done
#if G27_MC
        if (w == 0) bar_expect(xfb, M * k * 2);
#endif
    }
    __syncthreads();
#if G27_MC
    cl_sync();  // peers' x barriers initialized + armed before any multicast lands
#endif
    asm volatile("griddepcontrol.wait;" ::: "memory");
    TR(1)
#if G27_MC
    {
        const int cs = G27_MC > 1 ? G27_MC : 1, rk = cl_rank();
        for (int qq = rk + cs * t; qq < M; qq += cs * WPB * 32) bulk_mc(xs + qq * XS, x + (long)qq * k, k * 2, xfb);
    }
#endif
#ifdef G27_XBULK
    if (t == 0) {
        bar_expect(xfb, M * k * 2);
        for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k, k * 2, xfb);
    }
#endif
    if (lane == 0) for (; jn < SPK && jn < NS; ++jn) issue(jn);
    jn = __shfl_sync(0xffffffffu, jn, 0);
#ifndef G27_NOFIX
    float4 hd[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) hd[h] = __ldg((const float4*)(H + (long)(rg * 16 + g + 8 * h) * ESC) + c);
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] = acc2[jj][e] = 0.f;
    const unsigned xbase = xs + xrow + kq * (KW / 32) * 64;
#if defined(G27_XBULK) || G27_MC
    bar_wait(xfb, 0);
#else
#ifdef G27_XREP
    xload(xs, x + (long)blockIdx.x * M * k, M, k, XS, t, WPB * 32);  // timing probe only: per-block x copy
#else
    xload(xs, x, M, k, XS, t, WPB * 32);
#endif
#endif
    TRW(2)
    for (int st = 0; st < SPK; ++st) {
        const int sl = st % NS;
        bar_wait(fb + 8 * sl, (st / NS) & 1);
        const unsigned sb = ring + sl * SB + lane * 16, nb = ring + sl * SB + AP * 512 + lane * 8;
        constexpr int G = AP < 2 ? AP : 2;
#pragma unroll
        for (int j0 = 0; j0 < AP; j0 += G) {
            uint4 sv[G];
            uint2 nv[G];
#pragma unroll
            for (int a = 0; a < G; ++a) {
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[a].x), "=r"(sv[a].y), "=r"(sv[a].z), "=r"(sv[a].w) : "r"(sb + (j0 + a) * 512));
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[a].x), "=r"(nv[a].y) : "r"(nb + (j0 + a) * 256));
            }
            unsigned bx[G][2][NT][2];
#pragma unroll
            for (int a = 0; a < G; ++a)
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int jj = 0; jj < NT; jj += 2)
                        ldsm4(xbase + jj * 8 * XS + (st * AP + j0 + a) * 64 + s * 32, bx[a][s][jj][0], bx[a][s][jj][1], bx[a][s][jj + 1][0], bx[a][s][jj + 1][1]);
#pragma unroll
            for (int a = 0; a < G; ++a)
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const uint4 af = dec8(s ? sv[a].z : sv[a].x, s ? sv[a].w : sv[a].y, s ? nv[a].y : nv[a].x);
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[jj] : acc[jj], af, bx[a][s][jj][0], bx[a][s][jj][1]);
                }
        }
        __syncwarp();
        if (jn < SPK) {
            if (lane == 0) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                issue(jn);
            }
            ++jn;
        }
    }
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] += acc2[jj][e];
#ifndef G27_NOFIX
    {  // escape fixups (g24's): escaped weights decoded as 2^-15 (scale units); add x * (true - 2^-15)
        const int klo = kq * KW, khi = klo + KW;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = hd[h];
            unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
            unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
            if (!(f0 | f1)) continue;
            f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
            f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {
                if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
                if (hy != 0x1p-15f && col >= klo && col < khi) {
                    const unsigned char* xc = smem + WPB * NS * SB + col * 2;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                            acc[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[jj][2 * h + b]));
                        }
                }
            }
        }
    }
#endif
    // partials -> [tok][row] tile in this warp's own ring (all its stages are consumed), then a fixed-order block sum
    TRW(3)
    float* stg = (float*)(smem + w * NS * SB);
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + g + 8 * (e >> 1)] = acc[jj][e] * scale;
    __syncthreads();
    TR(4)
    for (int i = t; i < M * RGB * 16; i += WPB * 32) {
        const int tok = i / (RGB * 16), rr = i % (RGB * 16);
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < KQ; ++q) s += ((const float*)(smem + (q * RGB + rr / 16) * NS * SB))[tok * STR + rr % 16];
        const long o = (long)tok * n + blockIdx.x * RGB * 16 + rr;
        if (OBF) ((__nv_bfloat16*)outp)[o] = __float2bfloat16_rn(s);
        else ((float*)outp)[o] = s;
    }
    // ZN (as g24's, layer proj s56): zero the next g24's FP32 out (zo, zn4 float4s); plain stores, no fence
    for (int i = blockIdx.x * WPB * 32 + t; i < zn4; i += gridDim.x * WPB * 32) ((float4*)zo)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef G27_TRACE
    __syncthreads();
#endif
    TR(5)
}
// s57 PD variant: NAW = k / KQ / 32 atoms per warp, all held in registers: every W stage is waited for and DECODED
// before griddepcontrol.wait (W does not depend on the predecessor); after the wait only x (ldmatrix) + mma + sum.
// Needs NS >= NAW / AP (whole slice in the ring, issued pre-wait). XQ: x arrives in KQ slices (one barrier each).
template <int M, int KQ, int AP, int NAW, int RGB, bool OBF>
__global__ void __launch_bounds__(RGB * KQ * 32, 1) gemm_pd(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                            void* __restrict__ outp, int n, int k, int npre, float* __restrict__ zo, int zn4) {
    constexpr int WPB = RGB * KQ, NT = M / 8, SB = AP * WB, STR = 16 + 4, NS = NAW / AP, KW = NAW * 32;
    static_assert(NT % 2 == 0 && M <= 32 && NS * SB >= M * STR * 4 && NAW % AP == 0, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    TR(0)
    const int rgl = w % RGB, kq = w / RGB, rg = blockIdx.x * RGB + rgl, nrt = n / 16;
    const int XS = k * 2 + 16;
    const unsigned s0 = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned ring = s0 + w * NS * SB, xs = s0 + WPB * NS * SB;
    const unsigned fb = xs + M * XS + w * NS * 8, xfb = xs + M * XS + WPB * NS * 8;
    const unsigned char* wsrc = WP + ((long)kq * nrt + rg) * NS * SB;
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) for (int i = 0; i < KQ; ++i) bar_init(xfb + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int j = 0; j < NS; ++j) {
            bar_expect(fb + 8 * j, SB);
            bulk(ring + j * SB, wsrc + (long)j * SB, SB, fb + 8 * j);
        }
#if G27_MC
        if (w == 0) for (int q = 0; q < KQ; ++q) bar_expect(xfb + 8 * q, M * KW * 2);
#endif
    }
#ifndef G27_NOFIX
    float4 hd[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) hd[h] = __ldg((const float4*)(H + (long)(rg * 16 + g + 8 * h) * ESC) + c);
#endif
#if G27_MC
    cl_sync();  // early: peers' x barriers armed; hidden under the W stream
#else
    __syncthreads();
#endif
    uint4 af[NAW][2];
#pragma unroll
    for (int st = 0; st < NS; ++st) {
        bar_wait(fb + 8 * st, 0);
        const unsigned sb = ring + st * SB + lane * 16, nb = ring + st * SB + AP * 512 + lane * 8;
#pragma unroll
        for (int a = 0; a < AP; ++a) {
            uint4 sv; uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + a * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + a * 256));
            af[st * AP + a][0] = dec8(sv.x, sv.y, nv.x);
            af[st * AP + a][1] = dec8(sv.z, sv.w, nv.y);
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    TR(1)
#if G27_MC
    {  // slice q of row qq by thread i = q * M + qq of rank qq % cs
        const int cs = G27_MC > 1 ? G27_MC : 1, rk = cl_rank();
        for (int i = t; i < KQ * M; i += WPB * 32) {
            const int q = i / M, qq = i % M;
            if (qq % cs == rk) bulk_mc(xs + qq * XS + q * KW * 2, x + (long)qq * k + q * KW, KW * 2, xfb + 8 * q);
        }
    }
#elif defined(G27_XBULK)
    if (t == 0) {
        for (int q = 0; q < KQ; ++q) {
            bar_expect(xfb + 8 * q, M * KW * 2);
            for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS + q * KW * 2, x + (long)qq * k + q * KW, KW * 2, xfb + 8 * q);
        }
    }
#else
#ifdef G27_XREP
    xload(xs, x + (long)blockIdx.x * M * k, M, k, XS, t, WPB * 32);  // timing probe only: per-block x copy
#else
    xload(xs, x, M, k, XS, t, WPB * 32);
#endif
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] = acc2[jj][e] = 0.f;
    const unsigned xbase = xs + xrow + kq * NAW * 64;
#if defined(G27_XBULK) || G27_MC
    bar_wait(xfb + 8 * kq, 0);
#endif
    TRW(2)
#pragma unroll
    for (int a = 0; a < NAW; ++a) {
        unsigned bx[2][NT][2];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int jj = 0; jj < NT; jj += 2)
                ldsm4(xbase + jj * 8 * XS + a * 64 + s * 32, bx[s][jj][0], bx[s][jj][1], bx[s][jj + 1][0], bx[s][jj + 1][1]);
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[jj] : acc[jj], af[a][s], bx[s][jj][0], bx[s][jj][1]);
    }
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] += acc2[jj][e];
#ifndef G27_NOFIX
    {
        const int klo = kq * KW, khi = klo + KW;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = hd[h];
            unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
            unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
            if (!(f0 | f1)) continue;
            f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
            f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {
                if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
                if (hy != 0x1p-15f && col >= klo && col < khi) {
                    const unsigned char* xc = smem + WPB * NS * SB + col * 2;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                            acc[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[jj][2 * h + b]));
                        }
                }
            }
        }
    }
#endif
    TRW(3)
    float* stg = (float*)(smem + w * NS * SB);
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + g + 8 * (e >> 1)] = acc[jj][e] * scale;
    __syncthreads();
    TR(4)
    for (int i = t; i < M * RGB * 16; i += WPB * 32) {
        const int tok = i / (RGB * 16), rr = i % (RGB * 16);
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < KQ; ++q) s += ((const float*)(smem + (q * RGB + rr / 16) * NS * SB))[tok * STR + rr % 16];
        const long o = (long)tok * n + blockIdx.x * RGB * 16 + rr;
        if (OBF) ((__nv_bfloat16*)outp)[o] = __float2bfloat16_rn(s);
        else ((float*)outp)[o] = s;
    }
    // ZN (as g24's, layer proj s56): zero the next g24's FP32 out (zo, zn4 float4s); plain stores, no fence
    for (int i = blockIdx.x * WPB * 32 + t; i < zn4; i += gridDim.x * WPB * 32) ((float4*)zo)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef G27_TRACE
    __syncthreads();
#endif
    TR(5)
}
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
}  // namespace g27
#ifdef G27_TRACE
torch::Tensor g27_trace() {
    auto t = torch::empty({1024 * 6}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(t.data_ptr(), g27::trace, sizeof(unsigned long long) * 1024 * 6, 0, cudaMemcpyDeviceToDevice);
    return t;
}
#else
torch::Tensor g27_trace() { return torch::empty({0}); }
#endif
void g27_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g27::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
template <int M_, int KQ_, int AP_, int NS_, int RGB_, bool OBF_>
static void launch27(torch::Tensor& wp, torch::Tensor& h, double scale, torch::Tensor& x, torch::Tensor& out, int sm, int n, int k, int64_t npre, torch::Tensor& zo, cudaStream_t stream) {
    auto kern = [] { if constexpr (NS_ >= 100) return g27::gemm_pd<M_, KQ_, AP_, NS_ - 100, RGB_, OBF_>; else return g27::gemm<M_, KQ_, AP_, NS_, RGB_, OBF_>; }();
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n / (16 * RGB_)); cfg.blockDim = dim3(RGB_ * KQ_ * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
#if G27_MC > 1
    attr[1].id = cudaLaunchAttributeClusterDimension; attr[1].val.clusterDim.x = G27_MC; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1;
    cfg.numAttrs = 2;
    TORCH_CHECK((n / (16 * RGB_)) % G27_MC == 0, "g27 mc: grid");
#endif
    cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(),
                       out.data_ptr(), n, k, (int)npre, zo.numel() ? zo.data_ptr<float>() : nullptr, (int)(zo.numel() / 4));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void g27_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              int64_t kq, int64_t ap, int64_t ns, int64_t rgb, int64_t npre, torch::Tensor zo) {
    TORCH_CHECK(zo.numel() % 4 == 0 && (!zo.numel() || (zo.is_contiguous() && zo.scalar_type() == torch::kFloat32)), "g27: zo");
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const bool obf = out.scalar_type() == torch::kBFloat16;
    const bool pd = ns >= 100;  // PD variant: ns = 100 + NAW (atoms per warp); ring = NAW / AP slots
    const int nsl = pd ? (ns - 100) / ap : ns;
    TORCH_CHECK(!pd || (ns - 100) * 32 * kq == k, "g27 pd: NAW");
    const int sm = rgb * kq * nsl * ap * g27::WB + m * (k * 2 + 16) + rgb * kq * nsl * 8 + kq * 8 + 16;
    TORCH_CHECK(n % (16 * rgb) == 0 && k % (kq * 32 * ap) == 0 && sm <= 227 * 1024, "g27: shape/smem ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G27_CASE(M_, KQ_, AP_, NS_, RGB_, OBF_) \
    if (m == M_ && kq == KQ_ && ap == AP_ && ns == NS_ && rgb == RGB_ && obf == OBF_) { \
        launch27<M_, KQ_, AP_, NS_, RGB_, OBF_>(wp, h, scale, x, out, sm, n, k, npre, zo, stream); return; }
    G27_CONFIGS
    TORCH_CHECK(false, "g27: no config");
}
'''
# proj s57 gemv27 (g27_src.cu = r-proj muse_bf16_proj/dev/s57/gemv27.cu): o at T=16/32 without cross-block split-K: a
# block owns RGB whole 16-row groups, its warps split k (KQ), partials summed in smem, plain FP32 stores (no out zeroing,
# fence, epoch barrier or atomics; deterministic); ns >= 100 = PD: each warp's whole W slice (ns - 100 atoms) is streamed
# and DECODED into registers before griddepcontrol.wait, so after the wait only x + mma + the block sum remain.
# Config (m, KQ, AP, ns, RGB, bf16 out). Same 12-bit pack as g24 (_pack24 with kc = k / KQ). Takes ZN like g24 (zeroes the
# next g24's out) and never needs a zeroed out itself. proj H100 standalone after a 3 us PDL pred (us): o m16 PD 4.8-5.4
# vs g24 6.2-6.9 vs cuBLAS 7.8; m32 PD 6.2-7.0 vs 6.9-7.2. TUNE option g27:<key> (bool); default on where G27_ON.
# H200 (proj check c7b58683, after a 3 us PDL pred): o m16 PD 5.25 vs g24 6.43 vs cuBLAS 6.75 vs lt 7.20; m32 6.69 vs
# cuBLAS 6.43 (and H100 layer d32 +1.1..1.8): T=16 only. H100 layer ab_flag (TUNE {}, 4 rounds x seeds 7,11,99): d16 sl
# -0.6..-0.9, full -0.1..-0.3 us vs g24 o.
# proj s58: g27_src.cu = r-proj dev/s58/gemv27.cu, G27_MC 1 (x by spread bulk copies): H100 layer d16 sl -1.8 full -1.05 us
# vs s57; H200 (proj check a4962d70): o m16 4.42 (s57 5.28, g24 6.59); m32 6.27 vs g24 7.33 / lt 6.72 / cuBLAS 6.98 ->
# T=32 as TUNE option g27:o (H100 tie).
G27_CFG = {("o", 16): (16, 4, 4, 108, 4, 0), ("o", 32): (32, 4, 4, 108, 4, 0)}
G27_ON = {("o", 16)}
_G27 = {}
_GPR1_SRC = r'''// pr1: the layer's pooler + router + routing at 1 token in ONE launch (PDL secondary of the boundary).
// Blocks [0, RB) = router (FP32 W [256, 8192] x FP32 h32: R whole rows per group, x in registers, the next group
// bulk-prefetched into L2), blocks [RB, grid) = pooler (gemv13 body over lossless 12-bit BF16 W, out BF16).
// The last router block (ticket) does the routing: sigmoid(MULT logit), top-8 of score + bias (lowest index on ties),
// weights = picked / (sum + REPS), local ids (-1 off this GPU). W loads (and the gemv13 escape headers) are issued
// before griddepcontrol.wait; x (h / h32, written by the boundary) after it. Triggers its dependents at start.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace pr1 {
#ifndef BPRE
#define BPRE 1
#endif
#define XI(k) (((k) >> 4) * 20 + ((k) & 15))
constexpr int ESC = 8, MAXC = 16, E = 256, K = 8192, TOPK = 8;
__device__ __forceinline__ uint4 ld4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld2(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = decw<P0>(sw, nw, 0x9180u, 0x4140u);
    const unsigned q1 = decw<P0 + 1>(sw, nw, 0xB3A2u, 0x4342u);
    f[0] = __uint_as_float(q0 << 16); f[1] = __uint_as_float(q0 & 0xffff0000u);
    f[2] = __uint_as_float(q1 << 16); f[3] = __uint_as_float(q1 & 0xffff0000u);
}
__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* xp) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) *reinterpret_cast<float4*>(x + i) = *reinterpret_cast<const float4*>(xp + i);
    float f[16];
    dec4<0>(s.x, nb.x, f); dec4<2>(s.y, nb.x, f + 4); dec4<0>(s.z, nb.y, f + 8); dec4<2>(s.w, nb.y, f + 12);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) { a0 = fmaf(f[i], x[i], a0); a1 = fmaf(f[i + 1], x[i + 1], a1); }
    return a0 + a1;
}

struct Args {
    const uint4* S; const uint2* N; const float2* H; const __nv_bfloat16* x; __nv_bfloat16* out; int n; float scale;
    const float4* rw; const float4* x32; float* logits; int* cnt; const float* bias; const int* ls;
    int* ids; int* lids; float* wts; float mult, reps; int local, npf;
    // BND (layer s11): the boundary before the FFN fused in as a prologue of every block: from the o GEMV's BF16 out
    // br and the FP32 stream s: s' = alpha s + beta rmsnorm(br, eps_post) (written by block 0 with h in BF16),
    // h = rmsnorm(s', eps) gain, computed redundantly per block in registers (8 elements per thread).
    const __nv_bfloat16* br; const float* s; const float* alpha; const float* beta; const float* gain;
    float* s_out; __nv_bfloat16* h_out; float eps_post, eps; int bnd;
};

__device__ __forceinline__ float bsum(float v, float* sm) {  // block sum, 1024 threads, sm >= 32 floats
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sm[wp] = v;
    __syncthreads();
    v = sm[lane];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
struct BPre { float as[8], be[8], gn[8]; };  // alpha s, beta, gain (independent of the producer: loaded before the wait)
__device__ __forceinline__ void bpre(const Args& a, int g0, int g1, BPre& p) {
    const int g[2] = {g0, g1};
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const float4 s4 = __ldg(reinterpret_cast<const float4*>(a.s) + g[q]);
        const float4 al = __ldg(reinterpret_cast<const float4*>(a.alpha) + g[q]);
        const float4 be = __ldg(reinterpret_cast<const float4*>(a.beta) + g[q]);
        const float4 gn = __ldg(reinterpret_cast<const float4*>(a.gain) + g[q]);
        p.as[4 * q] = al.x * s4.x; p.as[4 * q + 1] = al.y * s4.y; p.as[4 * q + 2] = al.z * s4.z; p.as[4 * q + 3] = al.w * s4.w;
        p.be[4 * q] = be.x; p.be[4 * q + 1] = be.y; p.be[4 * q + 2] = be.z; p.be[4 * q + 3] = be.w;
        p.gn[4 * q] = gn.x; p.gn[4 * q + 1] = gn.y; p.gn[4 * q + 2] = gn.z; p.gn[4 * q + 3] = gn.w;
    }
}
// h at float4 groups g0, g1 (elements 4 g .. 4 g + 3) -> h[0..7]; call after griddepcontrol.wait, all 1024 threads
__device__ __forceinline__ void bnd8(const Args& a, int g0, int g1, float* h, float* sm, const BPre* pp = nullptr) {
    const int g[2] = {g0, g1};
    BPre p;
    if (pp) p = *pp; else bpre(a, g0, g1, p);
    float b[8], sv[8];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const uint2 bv = __ldcg(reinterpret_cast<const uint2*>(a.br) + g[q]);
        b[4 * q] = __uint_as_float(bv.x << 16); b[4 * q + 1] = __uint_as_float(bv.x & 0xffff0000u);
        b[4 * q + 2] = __uint_as_float(bv.y << 16); b[4 * q + 3] = __uint_as_float(bv.y & 0xffff0000u);
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(b[i], b[i], ss);
    const float rb = rsqrtf(bsum(ss, sm) / K + a.eps_post);
    float s2 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sv[i] = p.as[i] + p.be[i] * (b[i] * rb);
#pragma unroll
    for (int i = 0; i < 8; ++i) s2 = fmaf(sv[i], sv[i], s2);
    const float rs = rsqrtf(bsum(s2, sm + 32) / K + a.eps);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int i = 4 * q; i < 4 * q + 4; ++i) h[i] = sv[i] * rs * p.gn[i];
        if (blockIdx.x == 0) {
            reinterpret_cast<float4*>(a.s_out)[g[q]] = make_float4(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3]);
            __nv_bfloat162 h01 = __floats2bfloat162_rn(h[4 * q], h[4 * q + 1]), h23 = __floats2bfloat162_rn(h[4 * q + 2], h[4 * q + 3]);
            uint2 hv; hv.x = *reinterpret_cast<unsigned*>(&h01); hv.y = *reinterpret_cast<unsigned*>(&h23);
            reinterpret_cast<uint2*>(a.h_out)[g[q]] = hv;
        }
    }
}

// router role: block rb of RB, groups of R consecutive rows, 1024 threads (2 float4 of each row per thread)
template <int R, int T>
__device__ __forceinline__ void router(const Args& a, int rb, int RB) {
    constexpr int J = K / (4 * T), NG = E / R, NW = T / 32;
    static_assert(K % (4 * T) == 0 && E % R == 0, "router split");
    __shared__ float red[2][NW][R];
    __shared__ int last;
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    float4 w[R][J];
    int g = rb;
    auto load = [&](int gg) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < J; ++j) w[r][j] = __ldcs(a.rw + (size_t)(gg * R + r) * (K / 4) + t + T * j);
    };
    auto pf = [&](int gg) {
        if (t == 0 && gg < NG)
            asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }"
                         :: "l"(a.rw + (size_t)gg * R * (K / 4)), "r"(R * K * 4) : "memory");
    };
    if (g < NG) load(g);
    pf(g + RB);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float4 xr[J];
    if (a.bnd) {
        static_assert(J == 2 && T == 1024, "BND: 1024 threads, 8 elements each");
        __shared__ float bsm[64];
        float hh[8];
        bnd8(a, t, t + T, hh, bsm);
        xr[0] = make_float4(hh[0], hh[1], hh[2], hh[3]); xr[1] = make_float4(hh[4], hh[5], hh[6], hh[7]);
    } else {
#pragma unroll
    for (int j = 0; j < J; ++j) xr[j] = __ldcg(a.x32 + t + T * j);
    }
    int par = 0;
    for (; g < NG; g += RB) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float s0 = 0.f, s1 = 0.f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                s0 = fmaf(w[r][j].x, xr[j].x, s0); s1 = fmaf(w[r][j].y, xr[j].y, s1);
                s0 = fmaf(w[r][j].z, xr[j].z, s0); s1 = fmaf(w[r][j].w, xr[j].w, s1);
            }
            acc[r] = s0 + s1;
        }
        if (g + RB < NG) { load(g + RB); pf(g + 2 * RB); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[r];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[par][wp][r] = v;
        }
        __syncthreads();
        if (t < R) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < NW; ++q) s += red[par][q][t];
            a.logits[g * R + t] = s;
        }
        par ^= 1;
    }
    // ticket: the last router block does the routing
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(a.cnt, 1) == RB - 1;
    __syncthreads();
    if (!last || wp) return;
    __threadfence();
    float sc[E / 32], key[E / 32];
#pragma unroll
    for (int i = 0; i < E / 32; ++i) {
        const int e = lane + 32 * i;
        const float lg = __ldcg(a.logits + e);
        sc[i] = 1.f / (1.f + expf(-lg * a.mult));
        key[i] = sc[i] + __ldg(a.bias + e);
    }
    int myid = 0; float mysc = 0.f, sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float bk = key[0]; int bi = lane; float bs = sc[0];
#pragma unroll
        for (int i = 1; i < E / 32; ++i)
            if (key[i] > bk) { bk = key[i]; bi = lane + 32 * i; bs = sc[i]; }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; bs = os; }
        }
#pragma unroll
        for (int i = 0; i < E / 32; ++i)
            if (lane + 32 * i == bi) key[i] = -INFINITY;
        if (lane == k) { myid = bi; mysc = bs; }
        sum += bs;
    }
    if (lane < TOPK) {
        a.ids[lane] = myid;
        int loc = (myid - __ldg(a.ls) + E) % E;
        a.lids[lane] = loc < a.local ? loc : -1;
        a.wts[lane] = mysc / (sum + a.reps);
    }
    if (lane == 0) *a.cnt = 0;
}

// pooler role: gemv13 (m = 1) on virtual blocks of the pooler part of the grid
template <int U, int NV, int KK, int PF>
__device__ __forceinline__ void pooler(const Args& a, int pb, int PB) {
    constexpr int CE = U * 4096, RPC = CE / KK;
    static_assert(CE % KK == 0 && KK % 512 == 0, "chunk must hold whole rows");
    const int t = threadIdx.x & 255, vg = threadIdx.x >> 8, lane = t & 31, wp = t >> 5;
    const int nch = a.n / RPC;
    __shared__ float red_[NV][2][U * 8];
    __shared__ __align__(16) float xs[KK / 16 * 20];
    __shared__ float2 hs_[NV][MAXC * RPC * ESC];
    float (*red)[U * 8] = red_[vg];
    const int G = PB * NV;
    int c = pb * NV + vg;
    uint4 s[PF][U];
    uint2 nb[PF][U];
    auto load = [&](int p, int ch) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long i = (long)ch * (CE / 16) + u * 256 + t;
            s[p][u] = ld4(a.S + i);
            nb[p][u] = ld2(a.N + i);
        }
    };
    constexpr int XV = (KK / 8 + 256 * NV - 1) / (256 * NV), HV = (MAXC * RPC * ESC + 255) / 256;
    float2 hv[HV];
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * 256, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hv[q] = __ldg(a.H + (long)ch * RPC * ESC + i % (RPC * ESC));
    }
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
    // L2 prefetch of the next npf chunks (bytes + codes) while the boundary still runs
    if (t == 0)
        for (int p = PF; p < PF + a.npf; ++p) {
            const int ch = c + p * G;
            if (ch < nch) {
                asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }" :: "l"(a.S + (long)ch * (CE / 16)), "r"(CE) : "memory");
                asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }" :: "l"(a.N + (long)ch * (CE / 16)), "r"(CE / 2) : "memory");
            }
        }
    BPre bp;
    if (BPRE && a.bnd) bpre(a, 2 * threadIdx.x, 2 * threadIdx.x + 1, bp);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    uint4 xv[XV];
    if (a.bnd) {
        static_assert(XV == 1 && NV == 4 && KK == 8192, "BND: 1024 threads, 8 elements each");
        __shared__ float bsm[64];
        float hh[8];
        bnd8(a, 2 * threadIdx.x, 2 * threadIdx.x + 1, hh, bsm, BPRE ? &bp : nullptr);
        __nv_bfloat162 v0 = __floats2bfloat162_rn(hh[0], hh[1]), v1 = __floats2bfloat162_rn(hh[2], hh[3]),
                       v2 = __floats2bfloat162_rn(hh[4], hh[5]), v3 = __floats2bfloat162_rn(hh[6], hh[7]);
        xv[0].x = *reinterpret_cast<unsigned*>(&v0); xv[0].y = *reinterpret_cast<unsigned*>(&v1);
        xv[0].z = *reinterpret_cast<unsigned*>(&v2); xv[0].w = *reinterpret_cast<unsigned*>(&v3);
    } else {
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * 256 * NV;
        if (i < KK / 8) xv[q] = __ldcg(reinterpret_cast<const uint4*>(a.x) + i);
    }
    }
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * 256, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hs_[vg][i] = hv[q];
    }
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * 256 * NV;
        if (i < KK / 8) {
            const unsigned v[4] = {xv[q].x, xv[q].y, xv[q].z, xv[q].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) { const int kk = XI(8 * i + 2 * j); xs[kk] = __uint_as_float(v[j] << 16); xs[kk + 1] = __uint_as_float(v[j] & 0xffff0000u); }
        }
    }
    __syncthreads();
    int par = 0, j = 0;
    for (; c < nch; c += PF * G) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cc = c + p * G;
            if (cc < nch) {
                float acc[U];
#pragma unroll
                for (int u = 0; u < U; ++u) acc[u] = dot16(s[p][u], nb[p][u], xs + XI((u * 4096 + t * 16) % KK));
                const int cn = cc + PF * G;
                if (cn < nch) load(p, cn);
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    float v = acc[u];
#pragma unroll
                    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                    if (lane == 0) red[par][u * 8 + wp] = v;
                }
                asm volatile("bar.sync %0, 256;" :: "r"(vg + 1));
                if (t < RPC) {
                    float sum = 0.f;
#pragma unroll
                    for (int u = 0; u < U; ++u)
#pragma unroll
                        for (int w = 0; w < 8; ++w)
                            if ((u * 4096 + w * 512) / KK == t) sum += red[par][u * 8 + w];
                    const float2* he = hs_[vg] + ((j + p) * RPC + t) * ESC;
                    for (int e = 0; e < ESC; ++e) {
                        const float2 h = he[e];
                        if (h.y == 0x1p-15f) break;
                        const float xe = xs[XI(__float_as_int(h.x))];
                        sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
                    }
                    a.out[(long)cc * RPC + t] = __float2bfloat16(sum * a.scale);
                }
                par ^= 1;
            }
        }
        j += PF;
    }
}

template <int U, int NV, int PF, int R>
__global__ void __launch_bounds__(256 * NV, 1) fused(const Args a, int RB) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if ((int)blockIdx.x < RB) router<R, 256 * NV>(a, blockIdx.x, RB);
    else pooler<U, NV, 8192, PF>(a, blockIdx.x - RB, gridDim.x - RB);
}
}  // namespace pr1

// s, nb, h: _pack13(pooler W); x bf16 [1, 8192]; out bf16 [1, n]; rw fp32 [256, 8192]; x32 fp32 [1, 8192];
// logits fp32 [256] scratch; cnt int32 [>=1] zeros (self-reset); bias fp32 [256]; ls int32 [1];
// ids/lids int32 [1, 8], wts fp32 [1, 8]. cfg: u, nv, pf, r (rows per router group), rb (router blocks), blocks,
// npf (pooler chunks per virtual block L2-prefetched before griddepcontrol.wait), pdl.
void pr1_launch(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
                torch::Tensor rw, torch::Tensor x32, torch::Tensor logits, torch::Tensor cnt, torch::Tensor bias,
                torch::Tensor ls, torch::Tensor ids, torch::Tensor lids, torch::Tensor wts, double mult, double reps,
                int64_t local, int64_t u, int64_t nv, int64_t pf, int64_t r, int64_t rb, int64_t blocks, int64_t npf, int64_t pdl,
                std::vector<torch::Tensor> bt, double eps_post, double eps) {
    // bt = {} or {br bf16 [1, 8192], s fp32, alpha, beta, gain fp32 [8192], s_out fp32, h_out bf16}: fused boundary (BND)
    TORCH_CHECK(x.size(0) == 1 && x.size(1) == 8192 && rw.size(0) == 256 && rw.size(1) == 8192, "pr1: shapes");
    const int n = out.size(1);
    TORCH_CHECK((long)n * 8192 <= (long)pr1::MAXC * (blocks - rb) * nv * u * 4096, "pr1: too many chunks per virtual block");
    pr1::Args a{(const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), (const float2*)h.data_ptr(),
                (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale,
                (const float4*)rw.data_ptr(), (const float4*)x32.data_ptr(), logits.data_ptr<float>(), cnt.data_ptr<int>(),
                bias.data_ptr<float>(), ls.data_ptr<int>(), ids.data_ptr<int>(), lids.data_ptr<int>(), wts.data_ptr<float>(),
                (float)mult, (float)reps, (int)local, (int)npf};
    if (!bt.empty()) {
        TORCH_CHECK(bt.size() == 7 && bt[0].scalar_type() == at::kBFloat16 && bt[1].scalar_type() == at::kFloat &&
                    bt[6].scalar_type() == at::kBFloat16 && bt[0].numel() == 8192 && bt[1].numel() == 8192 && nv == 4, "pr1: BND args");
        a.br = (const __nv_bfloat16*)bt[0].data_ptr(); a.s = bt[1].data_ptr<float>(); a.alpha = bt[2].data_ptr<float>();
        a.beta = bt[3].data_ptr<float>(); a.gain = bt[4].data_ptr<float>(); a.s_out = bt[5].data_ptr<float>();
        a.h_out = (__nv_bfloat16*)bt[6].data_ptr(); a.eps_post = (float)eps_post; a.eps = (float)eps; a.bnd = 1;
    }
    cudaLaunchConfig_t lc = {}; lc.gridDim = blocks; lc.dynamicSmemBytes = 0;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0;
#define PR1_CASE(U_, V_, P_, R_) \
    if (u == U_ && nv == V_ && pf == P_ && r == R_) { lc.blockDim = 256 * V_; \
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, pr1::fused<U_, V_, P_, R_>, a, (int)rb)); return; }
    PR1_CONFIGS
    TORCH_CHECK(false, "pr1: no config");
}
'''
_GPR8_SRC = r'''// pr8: the layer's T=8 pooler (gemv14 body, S = 1: whole-K rows, x[8, 8192] staged in smem) + FP32 router
// (h32 FP32 x FP32 W) + routing in ONE launch, a PDL secondary of the boundary. Every block (128 = n / 32 rows of
// pooler each) also takes one router item: 16 rows x 1024 k (rows (b / 8) * 16.., k slice b % 8), W in registers loaded
// before griddepcontrol.wait, x32 slice cp.async'd to smem first after it. Item partials [8 slices][8 tok][256] go to
// scratch; the last of a row group's 8 slices (ticket) sums them in slice order (deterministic) into logits; the last
// row group (ticket) does the routing (warp w = token w): sigmoid(MULT logit), top-8 of score + bias (lowest index on
// ties), weights = picked / (sum + REPS), local ids. Counters self-reset. Triggers its dependents at start.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace pr8 {
constexpr int ESC = 8;
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld2(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned ld1(const unsigned* p) {
    unsigned v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(uint2 s, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s.x, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s.x, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s.y, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s.y, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
constexpr int E = 256, TOPK = 8, MT = 8, RSL = 8, RK = 8192 / RSL;  // s88: router rows per item = 4 x row groups per block
struct Args {
    const uint2* WS; const unsigned* WN; const float2* H; float scale; const __nv_bfloat16* x; __nv_bfloat16* out;
    int n, nrg_all, bk;
    const float4* rw; const float* x32; float* part; float* logits; int* cnt; const float* bias; const int* ls;
    int* ids; int* lids; float* wts; float mult, reps; int local, npf;
};

__device__ __forceinline__ void route_token(const Args& a, const float* lgs, const float* bs_, int ls, int tok, int lane) {
    float sc[E / 32], key[E / 32];
#pragma unroll
    for (int i = 0; i < E / 32; ++i) {
        const int e = lane + 32 * i;
        const float lg = lgs[tok * E + e];
        sc[i] = 1.f / (1.f + expf(-lg * a.mult));
        key[i] = sc[i] + bs_[e];
    }
    int myid = 0; float mysc = 0.f, sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float bk = key[0]; int bi = lane; float bs = sc[0];
#pragma unroll
        for (int i = 1; i < E / 32; ++i)
            if (key[i] > bk) { bk = key[i]; bi = lane + 32 * i; bs = sc[i]; }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ok = __shfl_xor_sync(0xffffffffu, bk, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; bs = os; }
        }
#pragma unroll
        for (int i = 0; i < E / 32; ++i)
            if (lane + 32 * i == bi) key[i] = -INFINITY;
        if (lane == k) { myid = bi; mysc = bs; }
        sum += bs;
    }
    if (lane < TOPK) {
        a.ids[tok * TOPK + lane] = myid;
        const int loc = (myid - ls + E) % E;
        a.lids[tok * TOPK + lane] = loc < a.local ? loc : -1;
        a.wts[tok * TOPK + lane] = mysc / (sum + a.reps);
    }
}

// the extra block: waits for the 128 router items, sums their K slices in order (deterministic), routes the 8 tokens
__device__ __forceinline__ void finish(const Args& a) {
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    __shared__ float lgs[MT * E], bs_[E];
    __shared__ int done, ls_;
    if (t < E) bs_[t] = __ldg(a.bias + t);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (t == 0) ls_ = __ldg(a.ls);
    if (t == 0) {
        int v;
        do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(a.cnt) : "memory");
            if (v < a.bk) __nanosleep(256);
        } while (v < a.bk);
        done = 1;
    }
    __syncthreads();
    for (int i = t; i < MT * E; i += blockDim.x) {
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < RSL; ++q) s += __ldcg(a.part + (long)q * MT * E + i);
        lgs[i] = s;
        a.logits[i] = s;
    }
    __syncthreads();
    if (wp < MT) route_token(a, lgs, bs_, ls_, wp, lane);
    if (t == 0) a.cnt[0] = 0;
}

// s88: NGT row groups per block (4: 1024 thr, 1 block/SM; 2: 512 thr, 2 blocks/SM = 'light', co-resides with o), x staged
// as an XR-piece ring of XP pieces (XR0 0 = all XP pieces); escape fixups then read x from global (L2).
template <int KCU, int WG, int PF, int XP, int XR0, int NGT>
__global__ void __launch_bounds__(NGT * WG * 32, NGT == 4 ? 1 : 2) fused(const Args a) {
    constexpr int M = MT, XR = XR0 ? XR0 : XP, KC = 1024 * KCU, TT = 1, RROWS = NGT * 4;
    static_assert(NGT * WG == 32 || NGT * WG == 16, "pr8: 64 threads per router row");
    static_assert(KC == 8192, "pr8: S = 1 (whole-K rows)");
    constexpr int NT = 8, NA = KC / 32 / WG;
    static_assert(NA % (PF * XP) == 0 && XR <= 4 && XR <= XP && XP <= 8, "x pieces");
    constexpr int PA = NA / XP, PB = KC / XP * 2;
    constexpr int XS = XR * PB + 64;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;
    float4* accp = (float4*)(smem + M * XS);
    float2* hs = (float2*)(smem + M * XS + blockDim.x / 32 * TT * 512);
    __shared__ int flag;
    __shared__ float rred[RROWS][2][MT];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if ((int)blockIdx.x == a.bk) {
        if (a.npf != -1 && a.npf != -2) finish(a);
        else asm volatile("griddepcontrol.wait;" ::: "memory");
        return;
    }
#ifdef PR8_TRACE
    unsigned long long* tr = (unsigned long long*)(a.part + 8 * MT * E) + blockIdx.x * 8;
#define TR(i) if (threadIdx.x == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v_)); tr[i] = v_; }
#else
#define TR(i)
#endif
    TR(0)
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int rg0 = blockIdx.x * NG, rg = rg0 + gi;
    const int n = a.n;
    float* xr = (float*)(smem + M * XS + blockDim.x / 32 * TT * 512 + NG * 8 * ESC * 8);  // [8][RK] FP32 x slice
    // router item: rows ri * 16 + (t >> 6), k = sl * RK + (t & 63) * 4 + 256 j
    const int ri = blockIdx.x / RSL, sl = blockIdx.x % RSL, rrow = ri * RROWS + (t >> 6), rq = t & 63;
    const bool has_r = ri < E / RROWS && a.npf != -1;  // npf -1: no router item, -2: no tickets (timing probes)
    float4 rwv[RK / 256];
    const bool rlate = a.npf == 1;  // npf 1: router W loads after griddepcontrol.wait
    if (has_r && !rlate) {
#pragma unroll
        for (int j = 0; j < RK / 256; ++j) rwv[j] = __ldcs(a.rw + ((size_t)rrow * 8192 + sl * RK + rq * 4 + 256 * j) / 4);
    }
    const long wo = (((long)rg) * (KC / 32) + wi) * 32 + lane;
    const uint2* wbs = a.WS + wo;
    const unsigned* wbn = a.WN + wo;
    uint2 ws_[PF];
    unsigned wn_[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) { ws_[p] = ld2(wbs + p * WG * 32); wn_[p] = ld1(wbn + p * WG * 32); }
    auto bulk_pf = [&](int nb) {  // s36: bulk L2 prefetch of atoms [PF, PF + nb) of each row group (WS, WN)
        if (t >= 2 * NG) return;
        const int gi2 = t >> 1;
        nb = min(nb, NA - PF);
        const long o0 = ((long)(rg0 + gi2) * (KC / 32) + PF * WG) * 32;
        const char* src = (t & 1) ? (const char*)(a.WN + o0) : (const char*)(a.WS + o0);
        const unsigned bytes = (unsigned)nb * WG * 32 * ((t & 1) ? 4 : 8);
        if (nb > 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
    };
    if (a.npf >= 100 && a.npf < 200) bulk_pf(a.npf - 100);  // before the wait
    if (a.npf > 1 && a.npf < 100 && t < NG * WG) {  // L2 prefetch of this warp-slot's next atoms (gemv14 order: atom stride WG * 32 lanes)
        const int gi2 = t / WG, wi2 = t % WG;
        const long wo2 = (((long)(rg0 + gi2)) * (KC / 32) + wi2) * 32;
        for (int p = PF; p < PF + a.npf && p < NA; ++p) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(a.WS + wo2 + (long)p * WG * 32));
            if (p % 2 == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.WN + wo2 + (long)p * WG * 32));
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (has_r && rlate) {
#pragma unroll
        for (int j = 0; j < RK / 256; ++j) rwv[j] = __ldcs(a.rw + ((size_t)rrow * 8192 + sl * RK + rq * 4 + 256 * j) / 4);
    }
    // x32 slice first (its own commit group), then the headers + x pieces (gemv14)
    for (int i = t; i < M * (RK / 4); i += blockDim.x) {
        const int r = i / (RK / 4), q = i % (RK / 4);
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xr + r * RK + q * 4);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(a.x32 + (long)r * 8192 + sl * RK + q * 4));
    }
    asm volatile("cp.async.commit_group;");
    auto stage = [&](int p) {
        for (int i = t; i < M * (PB / 16); i += blockDim.x) {
            const int r = i / (PB / 16), q = i % (PB / 16);
            const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + (p % XR) * PB + q * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa),
                         "l"((const uint4*)(a.x + (long)r * KC + p * (KC / XP)) + q));
        }
    };
    for (int i = t; i < NG * 8 * ESC / 2; i += blockDim.x) {
        const unsigned sa = (unsigned)__cvta_generic_to_shared(hs + 2 * i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(a.H + (long)rg0 * 8 * ESC + 2 * i));
    }
#pragma unroll
    for (int p = 0; p < XR; ++p) { stage(p); asm volatile("cp.async.commit_group;"); }
    if (a.npf >= 200) bulk_pf(a.npf - 200);  // after the wait and the x / x32 / header copies are issued
    auto router_item = [&]() {
    // router item
    if (XR >= 4) asm volatile("cp.async.wait_group 4;");
    else if (XR == 3) asm volatile("cp.async.wait_group 3;");
    else if (XR == 2) asm volatile("cp.async.wait_group 2;");
    else asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    TR(1)
    if (has_r) {
        float acc[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            float s0 = 0.f, s1 = 0.f;
#pragma unroll
            for (int j = 0; j < RK / 256; ++j) {
                const float4 xv = *(const float4*)(xr + i * RK + rq * 4 + 256 * j);
                s0 = fmaf(rwv[j].x, xv.x, s0); s1 = fmaf(rwv[j].y, xv.y, s1);
                s0 = fmaf(rwv[j].z, xv.z, s0); s1 = fmaf(rwv[j].w, xv.w, s1);
            }
            float v = s0 + s1;
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[i] = v;
        }
        if (lane < MT) {
            float v = acc[0];
#pragma unroll
            for (int i = 1; i < MT; ++i) if (lane == i) v = acc[i];
            rred[t >> 6][(t >> 5) & 1][lane] = v;
        }
    }
    __syncthreads();
    if (has_r && t < RROWS * MT) {
        const int r = t / MT, tok = t % MT;
        __stcg(a.part + ((long)sl * MT + tok) * E + ri * RROWS + r, rred[r][0][tok] + rred[r][1][tok]);
    }
    TR(2)
    if (has_r && a.npf != -2) {  // signal: partial stores happen-before thread 0's release add (cumulative)
        __syncthreads();
        if (t == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(a.cnt) : "memory");
    }
    };
#if !defined(PR8_RLAST) && !defined(PR8_RMID)
    router_item();
#endif
    TR(3)
    // pooler (gemv14, S = 1)
    float acc[TT][4];
    acc[0][0] = acc[0][1] = acc[0][2] = acc[0][3] = 0.f;
    for (int at = 0; at < NA; at += PF) {
        if (at % PA == 0) {
            const int pc = at / PA;
            if (XR < XP && pc > 0) {  // ring: piece pc - 1 is consumed by every warp; refill its slot with piece pc - 1 + XR
                __syncthreads();
                if (pc - 1 + XR < XP) stage(pc - 1 + XR);
                asm volatile("cp.async.commit_group;");  // (possibly empty: keeps the group count uniform)
            }
            const int left = XR < XP ? (XR - 1 < XP - 1 - pc ? XR - 1 : XP - 1 - pc) : XP - 1 - pc;
            if (left >= 3) asm volatile("cp.async.wait_group 3;");
            else if (left == 2) asm volatile("cp.async.wait_group 2;");
            else if (left == 1) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
#ifdef PR8_RMID  // s36: router item in the middle of the pooler loop (at == PR8_RMID)
        if (at == PR8_RMID) router_item();
#endif
#pragma unroll
        for (int p = 0; p < PF; ++p) {
#ifdef PR8_NOMATH  // timing probe: loads only
            acc[0][p & 3] += __int_as_float(ws_[p].x ^ ws_[p].y ^ wn_[p]);
#else
            const uint4 wv = dec8(ws_[p], wn_[p]);
            const int kb = ((at + p) * WG + wi) * 32 + c * 8;
            const int kx = (kb - (at / PA) * (KC / XP) + ((at / PA) % XR) * (KC / XP)) * 2;
            uint4 xa = *(const uint4*)(xs + g * XS + kx);
            mma(acc[0], xa.x, 0u, xa.y, 0u, wv.x, wv.y);
            mma(acc[0], xa.z, 0u, xa.w, 0u, wv.z, wv.w);
#endif
#ifdef PR8_NOW  // timing probe: W re-read from the first atoms (L1/L2 hits)
            if (at + p + PF < NA) { ws_[p] = ld2(wbs + ((at + p + PF) % PF) * WG * 32 + (at & 1)); wn_[p] = ld1(wbn + ((at + p + PF) % PF) * WG * 32 + (at & 1)); }
#else
            if (at + p + PF < NA) { ws_[p] = ld2(wbs + (at + p + PF) * WG * 32); wn_[p] = ld1(wbn + (at + p + PF) * WG * 32); }
#endif
        }
    }
    TR(4)
#ifdef PR8_RLAST  // s36: router item after the pooler loop (W stream not stalled by x32 / router math)
    router_item();
#endif
    accp[wp * 32 + lane] = make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
    __syncthreads();
    const int ne = NG * M * 8;
    for (int i = t; i < ne; i += blockDim.x) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        const int h = (tok >> 3) & 1, l = (tok & 7) * 4 + (row8 >> 1), e = h * 2 + (row8 & 1);
        const float* base = (const float*)accp + (rgl * WG) * 128 + l * 4 + e;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[q * 128];
        const float2* he = hs + (rgl * 8 + row8) * ESC;
#pragma unroll
        for (int e2 = 0; e2 < ESC; ++e2) {
            const int col = __float_as_int(he[e2].x);
            if (col >= 0 && col < KC) {
                const float xv = XR < XP ? __bfloat162float(__ldcg(a.x + (long)tok * KC + col))
                                         : __bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2));
                s = fmaf(xv, he[e2].y, fmaf(xv, -0x1p-15f, s));
            }
        }
        a.out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(s * a.scale);
    }
    TR(5)
}
// s36 timing probe (from gemv19): writes dst, triggers PDL at start, spins ns
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
}  // namespace pr8
void pr8_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    pr8::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}

// ws_/wn/h/scale: _pack14(pooler W, 8192) (S = 1); x bf16 [8, 8192]; out bf16 [8, n]; rw fp32 [256, 8192];
// x32 fp32 [8, 8192]; part fp32 [8 * 8 * 256] scratch; logits fp32 [8, 256] scratch; cnt int32 [1] zero
// (self-reset by the finisher block, blockIdx bk); bias fp32 [256]; ls int32 [1]; ids/lids int32 [8, 8], wts fp32 [8, 8].
// cfg: kcu (8), wg, pf, bk (blocks = bk = n / 8 / (rows groups per block); must be 128), xp, npf, pdl.
void pr8_launch(torch::Tensor ws_, torch::Tensor wn, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
                torch::Tensor rw, torch::Tensor x32, torch::Tensor part, torch::Tensor logits, torch::Tensor cnt,
                torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor lids, torch::Tensor wts,
                double mult, double reps, int64_t local, int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp,
                int64_t npf, int64_t pdl) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int ng = n / 8 / bk, threads = ng * wg * 32;
    const int xp_ = xp & 15, xr_ = (xp >> 4) ? (xp >> 4) : xp_;  // s88: xp = XP | XR << 4 (XR 0 = all pieces staged)
    const int sm = m * (1024 * kcu * 2 / xp_ * xr_ + 64) + threads / 32 * 512 + ng * 8 * pr8::ESC * 8 + 8 * pr8::RK * 4;
    TORCH_CHECK(m == 8 && k == 8192 && kcu == 8 && (bk == 128 && threads == 1024 || bk == 256 && threads == 512)
                && n / 8 % bk == 0 && 512 / ng == bk && sm <= 227 * 1024 && rw.size(0) == 256, "pr8: shape");
    pr8::Args a{(const uint2*)ws_.data_ptr(), (const unsigned*)wn.data_ptr(), (const float2*)h.data_ptr(), (float)scale,
                (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, n / 8, (int)bk,
                (const float4*)rw.data_ptr(), x32.data_ptr<float>(), part.data_ptr<float>(), logits.data_ptr<float>(),
                cnt.data_ptr<int>(), bias.data_ptr<float>(), ls.data_ptr<int>(), ids.data_ptr<int>(), lids.data_ptr<int>(),
                wts.data_ptr<float>(), (float)mult, (float)reps, (int)local, (int)npf};
    cudaLaunchConfig_t lc = {}; lc.gridDim = bk + 1; lc.blockDim = threads; lc.dynamicSmemBytes = sm;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0;
#define PR8_CASE(U_, G_, P_, X_, R_, N_) \
    if (kcu == U_ && wg == G_ && pf == P_ && xp == (X_ | R_ << 4) && ng == N_) { \
        C10_CUDA_CHECK(cudaFuncSetAttribute(pr8::fused<U_, G_, P_, X_, R_, N_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm)); \
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, pr8::fused<U_, G_, P_, X_, R_, N_>, a)); return; }
    PR8_CONFIGS
    TORCH_CHECK(false, "pr8: no config");
}
'''
_GOB8_SRC = r'''// layer s86 OB8: the T=8 o GEMM (proj's gemv12 body, m = 8, one k-chunk) with the first boundary fused into its
// epilogue, so pr8 (its PDL secondary) starts right after it (H100 oracle proj s80: no boundary = o->pr8 end -4.2 us).
// Boundary (the Triton _boundary, RB off): b = bf16(o out); s' = alpha s + beta b rb, rb = rsqrt(sum b^2 / N + eps_post);
// h = s' rs gain, rs = rsqrt(sum s'^2 / N + eps). Each thread owns one (token, column) of its block's 64 columns.
// Per block and token, four FP32 partial sums over its columns (sum b^2, (alpha s)^2, (alpha s)(beta b), (beta b)^2;
// sum s'^2 = A + 2 rb C + rb^2 D) go to part[block][token][4]; a grid barrier (all blocks are resident: the PDL
// dependent launches only after every block triggered); then every block sums the partials in a fixed order
// (deterministic) and writes s', h (BF16) and h32 (FP32) for its own columns.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace ob8 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
struct Bnd {
    const float* s; const float* alpha; const float* beta; const float* gain;
    float* s_out; __nv_bfloat16* h; float* h32; float* part; unsigned* bar; float eps_post, eps; int pdl;
};
constexpr int M = 8, WG = 2, NT = 8;
template <int PF>
__global__ void __launch_bounds__(512, 1) gemm(const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                               int n, int nrg_all, int k, const Bnd B) {
    if (!(B.pdl & 2)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // pdl & 2: after the W loop
    constexpr int KC = 1024, XS = KC * 2 + 64, NA = KC / 32 / WG;
    static_assert(NA % PF == 0, "PF must divide the atoms per warp");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;
    float4* accp = (float4*)(smem + M * XS);
    __shared__ float red[16][4][4];   // [warp][token in the warp][quantity]
    __shared__ float red2[16][32];
    __shared__ float fin[32];         // [token][quantity] grid totals
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int rg0 = blockIdx.x * NG, rg = rg0 + gi;
    // this thread's boundary element (independent of the producer: before the wait)
    const int row8 = t & 7, tok = (t >> 3) % M, rgl = (t >> 3) / M, col = (rg0 + rgl) * 8 + row8;
    const long ei = (long)tok * n + col;
    const float sv = __ldg(B.s + ei), al = __ldg(B.alpha + col), be = __ldg(B.beta + col), gn = __ldg(B.gain + col);
    const uint4* wb = W + ((long)rg * (KC / 32) + wi) * 32 + lane;
    uint4 w[PF];
    if (B.pdl & 1) {
        const char* pb = (const char*)(W + (long)rg0 * KC);
        for (long o = (long)t * 128; o < (long)NG * KC * 16; o += (long)blockDim.x * 128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pb + o));
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    for (int i = t; i < M * (KC / 8); i += blockDim.x) {
        const int r = i / (KC / 8), q = i % (KC / 8);
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + q * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"((const uint4*)(x + (long)r * k) + q));
    }
    asm volatile("cp.async.commit_group;");
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
    asm volatile("cp.async.wait_all;");
    __syncthreads();
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int a = 0; a < NA; a += PF) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
            uint4 xa = *(const uint4*)(xs + g * XS + kb * 2);
            mma(acc, xa.x, 0u, xa.y, 0u, w[p].x, w[p].y);
            mma(acc, xa.z, 0u, xa.w, 0u, w[p].z, w[p].w);
            if (a + p + PF < NA) w[p] = ldw(wb + (a + p + PF) * WG * 32);
        }
    }
    if (B.pdl & 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    accp[wp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    // gemv12's output i = (rgl * M + tok) * 8 + row8 = t: sum the WG warp slots of group rgl
    float bt;
    {
        const int l = tok * 4 + (row8 >> 1), e = row8 & 1;
        const float* base = (const float*)accp + (rgl * WG) * 128 + l * 4 + e;
        bt = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) bt += base[q * 128];
    }
    const float bf = __bfloat162float(__float2bfloat16(bt));  // the BF16 the GEMM returns
    const float xa = al * sv, yb = be * bf;
    float v[4] = {bf * bf, xa * xa, xa * yb, yb * yb};
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) v[q] += __shfl_xor_sync(0xffffffffu, v[q], o);
    if (row8 == 0)
#pragma unroll
        for (int q = 0; q < 4; ++q) red[wp][lane >> 3][q] = v[q];
    __syncthreads();
    // warp wp holds tokens (wp & 1) * 4 + 0..3 (row groups wp / 2): block total per (token, quantity), fixed order
    if (t < 32) {
        const int tk = t >> 2, q = t & 3;
        float s = 0.f;
        for (int w2 = tk >> 2; w2 < 16; w2 += 2) s += red[w2][tk & 3][q];
        __stcg(B.part + ((long)blockIdx.x * 32 + t), s);
    }
    __syncthreads();
    if (t == 0) {  // grid barrier (epoch counter: never reset; gridDim divides 2^32)
        __threadfence();
        const unsigned tk = atomicAdd(B.bar, 1u), target = (tk / gridDim.x + 1u) * gridDim.x;
        unsigned cur;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(B.bar) : "memory");
        } while ((int)(cur - target) < 0);
    }
    __syncthreads();
    {   // grid totals: 16 chunks of gridDim / 16 blocks per (token, quantity), then the 16 chunk sums in order
        const int p = t & 31, j = t >> 5, per = gridDim.x / 16;
        if (j < 16) {
            float s = 0.f;
            for (int b = j * per; b < (j + 1) * per; ++b) s += __ldcg(B.part + (long)b * 32 + p);
            red2[j][p] = s;
        }
    }
    __syncthreads();
    if (t < 32) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) s += red2[j][t];
        fin[t] = s;
    }
    __syncthreads();
    const float rb = rsqrtf(fin[tok * 4] / n + B.eps_post);
    const float s2 = fin[tok * 4 + 1] + 2.f * rb * fin[tok * 4 + 2] + rb * rb * fin[tok * 4 + 3];
    const float rs = rsqrtf(s2 / n + B.eps);
    const float sp = xa + be * (bf * rb);
    const float hv = sp * rs * gn;
    B.s_out[ei] = sp;
    B.h[ei] = __float2bfloat16(hv);
    if (B.h32) B.h32[ei] = hv;
}
}  // namespace ob8

// w: _pack12(o W, 1024) [1][n/8][32][32][8] bf16; x bf16 [8, 1024]; s fp32 [8, n] (the stream); alpha/beta/gain fp32 [n];
// s_out fp32 [8, n]; h bf16 [8, n]; h32 fp32 [8, n] or empty; part fp32 [blocks * 32]; bar int32 [1] (zeros once).
void ob8_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor gain,
              torch::Tensor s_out, torch::Tensor h, torch::Tensor h32, torch::Tensor part, torch::Tensor bar,
              int64_t nrg_all, int64_t pf, int64_t bk, double eps_post, double eps, int64_t pdl) {
    const int m = x.size(0), k = x.size(1), n = h.size(1);
    constexpr int WG = ob8::WG;
    const int ng = n / 8 / bk, threads = ng * WG * 32;
    const int sm = m * (1024 * 2 + 64) + threads / 32 * 512;
    TORCH_CHECK(m == 8 && k == 1024 && threads == 512 && n / 8 % bk == 0 && bk % 16 == 0 && part.numel() >= bk * 32
                && s.size(1) == n, "ob8: shape");
    ob8::Bnd B{s.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(), gain.data_ptr<float>(),
               s_out.data_ptr<float>(), (__nv_bfloat16*)h.data_ptr(), h32.numel() ? h32.data_ptr<float>() : nullptr,
               part.data_ptr<float>(), (unsigned*)bar.data_ptr(), (float)eps_post, (float)eps, (int)pdl};
    cudaLaunchConfig_t lc = {}; lc.gridDim = bk; lc.blockDim = threads; lc.dynamicSmemBytes = sm;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = (pdl & 1) ? 1 : 0;
#define OB8_CASE(P_) \
    if (pf == P_) { cudaFuncSetAttribute(ob8::gemm<P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ob8::gemm<P_>, (const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
                                          n, (int)nrg_all, k, B)); return; }
    OB8_CONFIGS
    TORCH_CHECK(false, "ob8: no config");
}
'''
# proj's gemv19 (g19_src.cu = r-proj dev/gemv19.cu @6996d0c): m = 16/32 GEMM over the 12-bit weights, FP32 out zeroed
# in-kernel (epoch grid barrier on ctr[1]) + red.add split-K partials, launched as a PDL dependent of the previous
# kernel: its producer streams the first npre weight stages before griddepcontrol.wait. (key, T) -> (m, kc, ap, ns,
# npre, xb), proj's H100 picks (r-proj BF16_G19). Consumers round the FP32 out to BF16 (= the GEMM's BF16 output).
G19_CFG = {("pooler", 16): (16, 1024, 8, 3, 3, 1), ("unpool", 16): (16, 1024, 4, 4, 2, 2),
           # s15 (proj s15 handoff, with its escape-fixup rework): T=32 pooler/unpool, T=16 qkv (FP32 out + cast)
           ("pooler", 32): (32, 1024, 8, 2, 2, 1), ("unpool", 32): (32, 1024, 8, 2, 2, 1),
           ("qkv_gate", 16): (16, 1024, 4, 4, 4, 2),
           # s19: H200-tuned in prepare (TUNE): qkv T=32 and o T=16/32 on g19 (configs already built for other picks)
           ("qkv_gate", 32): (32, 1024, 8, 2, 2, 1), ("o", 16): (16, 1024, 4, 4, 2, 2), ("o", 32): (32, 1024, 8, 2, 2, 1),
           # s38: M = 8 (g19 with one n8 token group) for T=8 (verify 1) qkv / o / pooler, TUNE options g19:<key>
           ("qkv_gate", 8): (8, 1024, 4, 4, 4, 2), ("o", 8): (8, 1024, 4, 4, 2, 2), ("pooler", 8): (8, 1024, 8, 3, 3, 1)}
# proj's H200 diag (after a 3 us PDL predecessor): m16 pooler AP8 15.7 vs 19.0, unpool tie, o +0.8, every m32 loses
# (+1.5..5): m32 stays on cuBLAS / the bmm pooler. Was (H100 picks, s10 v35): pooler m16 (16, 1024, 4, 4, 2, 2),
# m32 (32, 1024, 4, 4, 2, 1), unpool the same, o m16 (16, 512, 4, 3, 2, 2).
G19_BUILD = [(16, 1024, 4, 4, 2, 2)]  # also compiled (A/B)
# proj s53 gemv24 (g24_src.cu = r-proj dev/s53/gemv24.cu): g19's interface (FP32 out zeroed in the kernel, epoch counter,
# PDL dependent streaming W before its wait, npre bit 16 = trigger at start) with per-warp TMA rings (no producer warp /
# stage barriers). TUNE option g24:<key> (replaces g19 / cuBLAS at that call site). (m, kc, ap, ns, wpb, xb, r).
# proj H200 proxy (H100 G72, after a 3 us pred; g24 | g19): m32 pool 24.8|26.6, unp 23.9|24.7, o 8.7|9.9, qkv 16.1|16.4.
G24_CFG = {("pooler", 16): (16, 1024, 4, 3, 12, 2, 1), ("unpool", 16): (16, 1024, 2, 4, 16, 2, 1),
           ("qkv_gate", 16): (16, 1024, 2, 4, 16, 2, 1), ("o", 16): (16, 512, 2, 4, 16, 2, 1),
           ("pooler", 32): (32, 512, 4, 3, 12, 2, 1), ("unpool", 32): (32, 512, 4, 3, 12, 2, 1),
           ("qkv_gate", 32): (32, 512, 4, 3, 12, 2, 1), ("o", 32): (32, 512, 4, 3, 12, 2, 1)}
# proj s54: stream-K g24 (r | 256: each warp an equal contiguous range of ring stages, partial flushes at task boundaries);
# TUNE option g24sk:<key> (bool) runs these configs instead (overrides g24/g19/cuBLAS at that call site). proj H200 proxy
# (H100 G88/G108 standalone, after a 3 us pred; SK | static): qkv m16 11.4 | 11.8, m32 13.9 | 14.9; pool m32 21.1 | 21.5;
# unp m32 21.3 | 22.1; pool m16 at G108 19.5 | 21.5 (static 4,3,12 19.1); o loses (+0.4..0.9).
# proj s56 H200 diag (check 636c89fc, standalone after a 3 us PDL pred, us): pooler m16 G108 (4,3,12) 15.36 vs (2,4,16)
# 18.34; pooler m32 G108 SK (4,3,12) 20.00 vs SK (2,4,16) 21.15; qkv m16 SK 9.89 vs static 10.53;
# qkv m32 SK 13.15 vs lt 13.92 (first op at T=32: tuner only); unpool/o: g24 configs in use are best or tie.
# H100 layer ab_files A B B A: pooler T=16 (4,3,12): d16 sl -0.9, full -0.4 (default); pooler T=32 SK (4,3,12) d32 +1.2..1.7
# and qkv T=16 SK +0..0.6 on the H100: not default (g24sk:qkv_gate stays a TUNE[16] candidate).
G24_SKCFG = {("pooler", 16): (16, 1024, 2, 4, 16, 2, 257), ("unpool", 16): (16, 1024, 2, 4, 16, 2, 257),
             ("qkv_gate", 16): (16, 1024, 2, 4, 16, 2, 257), ("pooler", 32): (32, 512, 2, 4, 16, 2, 257),
             ("unpool", 32): (32, 512, 2, 4, 16, 2, 257), ("qkv_gate", 32): (32, 512, 2, 4, 16, 2, 257)}
# proj s69: stream-K g24 with (4,3,12) at T=32 pooler/unpool (TUNE option g24sk2:<key>, checked before g24sk). proj H100
# G88 H200 proxy (after a 3 us pred, 2 reps): pool m32 23.9-24.1 vs SK (2,4,16) 24.7-24.9, unp 22.0 vs 22.8-22.9; consumer-only
# (G24_NOW, G108) pool 17.6 vs 18.4-18.7, unp 17.2 vs 17.9-18.2. proj s56 H200 diag: pool m32 SK (4,3,12) 20.00 vs (2,4,16) 21.15.
# H100 in-layer it lost (+1.2..1.7 us d32, W-bound there): H200 tuner only.
G24_SK2CFG = {("pooler", 32): (32, 512, 4, 3, 12, 2, 257), ("unpool", 32): (32, 512, 4, 3, 12, 2, 257)}
_G24 = {}
# proj s56 EZ/ZN: within one layer call, each gemv24 zeroes the NEXT gemv24's FP32 out (preallocated, kept in _G24_ZN until
# that call) as its warps finish; the next one then skips its own zeroing + fence + grid epoch barrier (npre bit 17).
# Needs every kernel launched in between to execute griddepcontrol.wait (or no PDL) so completion is transitive.
# H100 standalone (graph pred + gemm, us/call): qkv/o/pool/unp m16 -0.4..-1.4; proxy G88 -0.5..-1.3; ZN side +0..0.3.
G24_ZCHAIN = {"qkv_gate": "o", "o": "pooler", "pooler": "unpool"}
G24Z_TOKENS = {16}  # T=32: H100 in-layer d32 +1..2.5 us (pooler m32 ZN stores +0.6..1.5, EZ unpool +0.5..1): off
G24_EZUSE = True  # A/B knob: False = the ZN stores happen but the consumer still zeroes itself
G24_EZPF = False  # True: the zeroed-ahead out is L2-prefetched before the consumer's griddepcontrol.wait (H100 d16/d32 tie: off)
_G24_ZN = {}  # (key, m) -> zeroed out for that call; cleared at every kernel_fn start
# proj s59 G24X (cross-call EZ for the graph's first op): qkv's FP32 out at T in G24X_TOKENS is a persistent buffer
# (state["g24x"][T], zeros from prepare) that the qkv g24 consumes as EZ (no zero + fence + grid epoch barrier); the
# call's g24 unpool zeroes it again as its ZN (qkv's out is dead after attention). If unpool did not (other path),
# kernel_fn zeroes it at its end: invariant = the buffer is zero between calls. H100 standalone first-op qkv m16
# 18.5 -> 17.6 us, m32 21.1 -> 19.2. H100 layer ab_files (TUNE {}, A B B A, 8 rounds) d16 sl -1.56, full -0.68 us;
# T=32 (only when the tuner picks g24:qkv_gate, a TUNE[32] option): g24 qkv -0.5/-0.9 us vs without G24X (6 rounds).
G24X_TOKENS = {16, 32}
_G24X = []  # the pending buffer of this call (unpool's ZN target)
# proj s63 G24P (TUNE option "g24p", T in G24P_TOKENS): the same cross-call EZ for the POOLER's FP32 out (persistent
# state["g24p"][T], zero between calls): the pooler g24 runs EZ on it (no zero + fence + grid epoch barrier; H200 diag61
# EZ bound pool m32 G108 -1.0..-1.5 us), and the call's g24 unpool zeroes it again as its ZN (pooled is dead after
# the rmsnorm) when G24X does not use that slot. Fallback: kernel_fn zeroes it at its end.
G24P_TOKENS = {32}
_G24P = []
BOUNDARY_TRIG = True  # the first boundary triggers its dependents at start when the gemv19 pooler follows
G19_ON = {"pooler", "unpool", "qkv_gate", "o"}  # o: H100 tie (cuBLAS after attention)
G19_DEFAULT_OFF = {("qkv_gate", 32), ("o", 16), ("o", 32), ("qkv_gate", 8), ("o", 8), ("pooler", 8)}  # prepared, used only if TUNE picks them (H100: tie/+3 us)
G19_GRID = {("pooler", 16): 108, ("pooler", 32): 108, ("pooler", 8): 108}  # (key, T) -> blocks (default: all SMs): the pooler leaves SMs to the side-stream router (H100 d16, AP8 + BOUNDARY_TRIG: 104 213.2, 112 214.9, 120 221.4, 132 220.2; bmm 219.3 us)
_G19_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count if torch.cuda.is_available() else 132
_GPR3_SRC = r'''// Router at m >= 9 as SGLang's BF16 hi + lo split with FP32 accumulation, in ONE GEMM kernel: split-K tensor-core
// GEMM (mma.sync m16n8k16, x = A, W rows = B), hi and lo MMAs into the same FP32 accumulators (sharing x fragments).
// Block = MT tokens x NT experts x KC of k; cp.async ring of 64-wide k stages (swizzled 128-B rows, ldmatrix).
// Split-K partials: staged in smem, then float4 FP32 atomics (red at L2) straight into out, which a tiny kernel
// zeroes first; the GEMM is its programmatic dependent (PDL): it loads and multiplies while the zeroing runs and
// waits (griddepcontrol.wait) only before its atomics. No workspace, no counters.

namespace pr3 {
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void ldm4(unsigned addr, unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// byte offset of 16-B chunk ch (0..7) of a 128-B row
__device__ __forceinline__ unsigned swz(int row, int ch) { return row * 128 + ((ch ^ (row & 7)) << 4); }

// MT tokens, NT experts per block, WM x WN warps, KC = k per block, NS stages of 64 k.
template <int MT, int NT, int WM, int WN, int KC, int NS>
__global__ void __launch_bounds__(WM * WN * 32) router(const __nv_bfloat16* __restrict__ w,  // [2][256][K] hi, lo
                                                       const __nv_bfloat16* __restrict__ x,  // [m][K]
                                                       float* __restrict__ out, int m, int K) {
    constexpr int NTH = WM * WN * 32, TM = MT / WM, TN = NT / WN, FM = TM / 16, FN = TN / 8;
    constexpr int XB = MT * 128, WB = NT * 128, SB = XB + 2 * WB, NK = KC / 64;
    static_assert(TM % 16 == 0 && TN % 16 == 0, "warp tile");
    extern __shared__ __align__(128) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, wm = warp / WN, wn = warp % WN;
    const int s = blockIdx.x, n0 = blockIdx.y * NT, m0 = blockIdx.z * MT, k0 = s * KC;
    const unsigned sb = smem_u32(sm);

    auto load = [&](int it) {
        if (it < NK) {
            const unsigned st = sb + (it % NS) * SB;
            const int kk = k0 + it * 64;
            for (int i = t; i < MT * 8; i += NTH) {
                const int r = i >> 3, ch = i & 7, row = m0 + r;
                const bool ok = row < m;
                cp16(st + swz(r, ch), x + (size_t)(ok ? row : 0) * K + kk + ch * 8, ok ? 16 : 0);
            }
            for (int i = t; i < 2 * NT * 8; i += NTH) {
                const int h = i / (NT * 8), r = (i >> 3) % NT, ch = i & 7;
                cp16(st + XB + h * WB + swz(r, ch), w + ((size_t)h * 256 + n0 + r) * K + kk + ch * 8, 16);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    float acc[FM][FN][4];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
    for (int i = 0; i < NS - 1; ++i) load(i);
    for (int it = 0; it < NK; ++it) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
        __syncthreads();
        load(it + NS - 1);
        const unsigned st = sb + (it % NS) * SB;
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {  // k16 steps: chunks 2kq, 2kq+1
            unsigned a[FM][4];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                const int r = wm * TM + i * 16 + (lane & 15);
                ldm4(st + swz(r, 2 * kq + (lane >> 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < FN; j += 2) {
                    const int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                    unsigned b0, b1, b2, b3;
                    ldm4(st + XB + h * WB + swz(r, 2 * kq + ((lane >> 3) & 1)), b0, b1, b2, b3);
#pragma unroll
                    for (int i = 0; i < FM; ++i) {
                        mma(acc[i][j], a[i], b0, b1);
                        mma(acc[i][j + 1], a[i], b2, b3);
                    }
                }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    // partial tile -> smem [MT][NT + 8] (conflict-free float2 stores), then float4 atomics into out
    constexpr int RS = NT + 8;
    static_assert(MT * RS * 4 <= NS * SB, "reduction tile");
    float* red = reinterpret_cast<float*>(sm);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            const int row = wm * TM + i * 16 + (lane >> 2), col = wn * TN + j * 8 + 2 * (lane & 3);
            *reinterpret_cast<float2*>(red + row * RS + col) = make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2*>(red + (row + 8) * RS + col) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
    asm volatile("griddepcontrol.wait;\n" ::: "memory");  // out zeroed
    __syncthreads();
    constexpr int TOT = MT * (NT / 4);
    const int rot = (int)(((long)s * TOT) / gridDim.x);  // splits of a tile start at different rows
    for (int i0 = t; i0 < TOT; i0 += NTH) {
        const int i = (i0 + rot) % TOT;
        const int r = i / (NT / 4), c = (i % (NT / 4)) * 4;
        if (m0 + r < m)
            atomicAdd(reinterpret_cast<float4*>(out + (size_t)(m0 + r) * 256 + n0 + c),
                      *reinterpret_cast<const float4*>(red + r * RS + c));
    }
}

__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;\n");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace pr3

#define PR3_CASE(MT, NT, WM, WN, KC, NS)                                                                           \
    if (mt == MT && nt == NT && wmn == WM && wnn == WN && kc == KC && ns == NS) {                                  \
        auto f = pr3::router<MT, NT, WM, WN, KC, NS>;                                                              \
        constexpr int SMEM = NS * (MT * 128 + 2 * NT * 128);                                                       \
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                                \
        pr3::zero<<<(m * 64 + 255) / 256, 256, 0, stream>>>(reinterpret_cast<float4*>(out.data_ptr<float>()),      \
                                                            m * 64);                                               \
        cudaLaunchConfig_t lc = {};                                                                                \
        lc.gridDim = dim3(K / KC, 256 / NT, (m + MT - 1) / MT);                                                    \
        lc.blockDim = dim3(WM * WN * 32);                                                                          \
        lc.dynamicSmemBytes = SMEM;                                                                                \
        lc.stream = stream;                                                                                        \
        cudaLaunchAttribute at[1];                                                                                 \
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                             \
        at[0].val.programmaticStreamSerializationAllowed = 1;                                                      \
        lc.attrs = at; lc.numAttrs = 1;                                                                            \
        cudaLaunchKernelEx(&lc, f, reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),                           \
                           reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), m, K);     \
        return;                                                                                                    \
    }

// w bf16 [2, 256, K] (hi, lo), x bf16 [m, K] contiguous, out fp32 [m, 256] contiguous (overwritten)
void pj_router3(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mt, int64_t nt, int64_t wmn, int64_t wnn,
                int64_t kc, int64_t ns) {
    const int m = x.size(0), K = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    PR3_CONFIGS
    TORCH_CHECK(false, "pj_router3: no config");
}
'''
KV_PF = True  # s9: the qkv gemv13 L2-prefetches the window's KV rows for csl (d1 sliding; H100 csl 6.1-7.3 -> ~4 us)
G13_CFG = {"qkv_gate": (2, 4, 1, 1, 132), "pooler": (2, 4, 1, 1, 132)}  # (u, nv, minb, pf, blocks), k = 8192
# s33 (proj): o at T=1 on gemv13 (12-bit W, 12.6 vs 16.8 MB) as a PDL secondary: W prologue (.nc) before
# griddepcontrol.wait, x (__ldcg) after. k = 1024. TUNE option "g13o" at T=1 (H100 d1 full -1.0..-1.3 us, sliding
# +-0.3: attention and o share HBM, so W bytes moved before the wait slow csl about as much as they save; NOTES-proj s33).
G13_O = (2, 1, 4, 1, 528)  # (u, nv, minb, pf, blocks); None = no o pack
G13_K = {"o": 1024}
# s21/s22: alternative GEMV configs, used only where TUNE picks them (option "<kind>alt:<key>" = index into the list, 1-based)
G13_ALT = {"qkv_gate": [(2, 4, 1, 2, 132)]}  # s21: T=1 PF 2 (2 rows in flight per virtual block; H100 +1.3/+2 us)
G10_ALT = {"o": [(1024, 4, 4, 1, 1, 132)]}  # s22: T=1 o (PDL secondary of csl): U 4 on 132 blocks (PF 2 on 264: H100 +11 us, dropped)
# s22: T=8 qkv: proj s16 (r-proj 2480c53 BF16_G14) without split-K: kcu 8 (own pack), 96 blocks of 3 row groups (H100
# standalone 15.2-15.6 vs 16.8 us). (2, 2, 8, 32, 1) / (2, 2, 4, 48, 1): H100 in-layer ties, dropped.
G14_ALT = {"qkv_gate": [(8, 8, 4, 96, 1), (8, 8, 4, 96, 2), (8, 8, 8, 96, 1)]}
_ALTS = {"g10": G10_ALT, "g13": G13_ALT, "g14": G14_ALT}


def _cfg(state, tokens, kind, key, cfg):
    """key's config for that GEMV kind, or the TUNE-picked alternative."""
    i = int(_opt(state, tokens, kind + "alt:" + key, 0)) if key in _ALTS[kind] else 0
    return _ALTS[kind][key][i - 1] if i else cfg[key]
G14_CFG = {"qkv_gate": (2, 2, 4, 32, 1), "pooler": (8, 8, 4, 128, 1)}  # (kcu, wg, pf, bk, xp), m = 8
# s11: proj pr1 (r-proj dev/pr1.cu@149d115): T=1 pooler (gemv13 body) + FP32 router + routing top-8 in ONE launch,
# PDL secondary of the boundary (which triggers at start). (u, nv, minb, r, rb, blocks, npf). H100 d1 sliding 95.1 -> 92.5.
PR1_TOKENS = (1, 8)  # 8: proj pr8 (r-proj dev/pr8.cu@3be546b): gemv14 pooler blocks also take a router item, an extra
# finisher block sums the K slices + routes. H100 v1 158.1/172.4 -> 157.0/171.0 us (proj). cfg (kcu, wg, pf, bk, xp, npf)
PR8_CFG = {8: (8, 8, 4, 128, 1, 212)}
# s88: light pr8 (TUNE pr8l): 2 row groups per block = 512 thr, 256 blocks, 2/SM (~107 KB smem: x as a 2-of-4-piece ring,
# xp = XP | XR << 4), so blocks co-reside with the o GEMM's tail and start their W ramp earlier.
PR8L_ALTS = []  # s88 DEAD (H100): light/ring alts, e.g. (8, 8, 4, 256, 4 | 1 << 4, 212) = 512 thr 2/SM (v1 sl tie, +0.6..0.9 us), (8, 8, 4, 128, 4 | 2 << 4, 212) heavy ring +1.3; with OB8 +1.1..1.6. TUNE pr8l: 1-based index
PR1_CFG = {1: (2, 4, 1, 4, 26, 132, 0)}  # s35: npf 4 -> 0 (H200 tuner picked npf 0 at T=1 sliding in both logged checks: -0.8/-2.3 us)
PR1_PF_ALT = (2,)  # s23: pr1 pooler PF (rows in flight per virtual block) alternatives, TUNE option "pr1_pf" (H200 GEMVs ~2.5 TB/s)
OB8_LATE = False  # s86: ob8 triggers pr8 after its W loop (else at start)
OB8 = False  # s86 DEAD: T=8 o GEMM (gemv12 body) with the first boundary fused into its epilogue (ob8_src.cu, grid barrier), pr8 its PDL secondary. H100 ab_flag v1 sl +2.2..2.5, full +0.3..0.7 us: pr8 (1024 thr x 64 regs = a whole SM) cannot co-reside with o blocks; the barrier holds all 128 SMs to the last block, so pr8 loses the pre-wait W ramp it gets behind the 16-program boundary
PR1_BND = False  # s11: the boundary fused into pr1 as a per-block prologue (each block re-reads 144 KB: H100 d1 +1.2/+0.3 us even with s/alpha/beta/gain loaded before the wait): off
G13_ESC = 8  # escapes per row (gemv13.cu / gemv14.cu ESC)


def _compress(w):
    """Lossless 12-bit BF16 (gemv13/gemv14): sign|mantissa byte and 4-bit exponent code c = e - base per weight
    (base = max exponent - 15; decoded exponent 112 | c, so sums are rescaled by scale = 2^(base - 112)); weights below
    the window (zeros, tiny values) are escapes: code 0, byte 0 (decodes to 2^(base - 127)) plus a per-row header of
    G13_ESC (col, true / scale) entries (exact; kernels subtract 2^-15 = 2^(base - 127) / scale). None if a row has more than G13_ESC escapes."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    base = int(e.max()) - 15
    if base < 1:
        return None
    code = e - base
    esc = code < 0
    if int(esc.sum(1).max()) > G13_ESC:
        return None
    code = torch.where(esc, 0, code)
    sm = torch.where(esc, 0, ((b >> 8) & 0x80) | (b & 0x7F))
    scale = 2.0 ** (base - 112)
    # empty slots: col 0, value 2^-15 (the kernels add x[col] * (value - 2^-15): exactly 0)
    hdr = torch.zeros(n, G13_ESC, 2, dtype=torch.float32, device=w.device)
    hdr[:, :, 1] = 2.0 ** -15
    rows, cols = esc.nonzero(as_tuple=True)  # row-major order
    if len(rows):
        slot = torch.arange(len(rows), device=w.device) - torch.searchsorted(rows, rows)
        hdr[rows, slot, 0] = cols.int().view(torch.float32)
        hdr[rows, slot, 1] = w[rows, cols].float() / scale  # BF16 over a power of two: exact in FP32
    # round trip: every weight decodes to its original bits (else no compression for this weight)
    dec = torch.where(esc, 0, (((sm & 0x80) << 8) | ((code + base) << 7) | (sm & 0x7F)))
    dec = dec.to(torch.int32).where(dec < 32768, dec - 65536).to(torch.int16).view(torch.bfloat16)
    if len(rows):
        fix = (hdr[rows, slot, 1] * scale).to(torch.bfloat16)
        if not torch.equal(fix.view(torch.int16), w[rows, cols].contiguous().view(torch.int16)):
            return None
        dec[rows, cols] = fix
        if not torch.equal(hdr[rows, slot, 1] * scale, w[rows, cols].float()):
            return None
    if not torch.equal(dec.view(torch.int16), w.contiguous().view(torch.int16)):
        return None
    return code, sm, hdr, scale


def _nib(code):
    """Codes [..., 8] -> int32 words: weight 2p + h at bits 4p + 16h."""
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=code.device)
    v = (code.long() << sh).sum(-1)
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).int()


def _dec3(sm, code):
    """s35 DEC3 layout over groups of 8 weights (last dim = the kernels' dec8 unit: 8 bytes + 1 word): byte = c0 << 7 |
    mantissa; word: weight 2P + h's sign, c1, c2, c3 at bits (15, 8, 9, 10) + 16h + (0, -4, -8, 4)[P] (mod 32). Kernels
    (decw in dev/gemv*.cu): prmt with 0x38 for the exponent's top bits + one lop3 per BF16 pair. Checked lossless here."""
    sm, code = sm.long(), code.long()
    i = torch.arange(8, device=sm.device)
    off = 16 * (i % 2) + torch.tensor([0, -4, -8, 4], device=sm.device)[i // 2]
    pos = [(b + off) % 32 for b in (15, 8, 9, 10)]
    bits = [(sm >> 7) & 1, (code >> 1) & 1, (code >> 2) & 1, (code >> 3) & 1]
    v = sum(b << q for b, q in zip(bits, pos)).sum(-1)
    b3 = ((code & 1) << 7) | (sm & 0x7F)
    back = [(v.unsqueeze(-1) >> q) & 1 for q in pos]  # round trip: every bit lands in its own place
    if not (torch.equal(back[0], bits[0]) and torch.equal(back[1] | back[2] << 1 | back[3] << 2, code >> 1)
            and torch.equal(b3 >> 7, code & 1) and torch.equal(b3 & 0x7F, sm & 0x7F)):
        raise RuntimeError("DEC3 packing round trip failed")
    return b3.to(torch.uint8), torch.where(v >= 2 ** 31, v - 2 ** 32, v).int()


def _pack13(w):
    """gemv13 layout: flat byte plane, code words over 8 consecutive weights."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    s3, n3 = _dec3(sm.flatten().view(-1, 8), code.flatten().view(-1, 8))
    return s3.flatten().contiguous(), n3.contiguous(), hdr, scale


def _pack14(w, kc):
    """gemv14 layout: gemv12's atom order [ks][rg][atom][lane][8] for bytes and codes."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    perm = lambda t: t.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
    s3, n3 = _dec3(perm(sm), perm(code))
    return s3, n3.contiguous(), hdr, scale
# (k, u, nv, minb, pf, blocks) per weight, m = 1, chosen by proj on the H100 (dev/g10.py)
G10_CFG = {"qkv_gate": (8192, 4, 4, 1, 1, 132), "pooler": (8192, 4, 4, 1, 1, 132), "o": (1024, 2, 4, 2, 1, 264),
           "unpool_slice": (4096, 2, 4, 1, 1, 132)}
_G10 = {}
G10_PFL2 = {}  # DEAD on H200 too (diag b1ee542c: d1 sliding no gain); s15: {sliding: keys} whose gemv10 (PDL secondary) bulk-prefetches its W into L2 before griddepcontrol.wait. H100 d1 {True: ("o",), False: ("o",)}: +1.8/+1.8 us (csl slowed); H200 untested


# (kcu, wg, pf, bk) per weight, m = 8 (verify 1): the proj agent's gemv12 (g12_src.cu = r-proj e71d015), its H100 picks
G12_CFG = {"qkv_gate": (2, 2, 4, 32), "pooler": (8, 8, 4, 128), "o": (1, 2, 8, 128), "unpool_slice": (4, 4, 4, 32)}
G12_TOKENS = 8


def _pack12(w, kc):
    """proj's gemv12 layout: [k-chunk][8-row group][kc/32 atoms][lane][8] (atom = 8 rows x 32 k), lossless."""
    n, k = w.shape
    return w.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()


def _g10_lib():
    if "lib" not in _G10:
        _G10["lib"] = None
        try:
            import hashlib
            from torch.utils.cpp_extension import load_inline
            cases = " ".join("G10_CASE(1,%d,%d,%d,%d,%d)" % c[:5] for c in sorted(set(G10_CFG.values()) | {a for v in G10_ALT.values() for a in v}))
            cases12 = " ".join("G12_CASE(%d,%d,%d,%d)" % ((G12_TOKENS,) + c[:3]) for c in sorted(set(G12_CFG.values())))
            src = _G10_SRC.replace("G10_CONFIGS", cases) + "\n" + _G12_SRC.replace("G12_CONFIGS", cases12)
            if UNPOOL_PDL:  # gemv10 may launch as a PDL secondary: its first W loads issue before griddepcontrol.wait
                anchor = "    for (int i = threadIdx.x; i < M * K8; i += 256 * NV) xs[i] = __ldg(x + i);"
                launch = ("        g10::gemv<M_, U_, V_, B_, K_, P_><<<blocks, 256 * V_, sm, s>>>((const uint4*)w.data_ptr(), "
                          "(const uint4*)x.data_ptr(), \\\n        (__nv_bfloat16*)out.data_ptr(), n); return; }")
                sig = "int64_t blocks) {"
                for x_ in (anchor, launch, sig):
                    assert src.count(x_) == 1, x_
                # s15: pdl == 2: first bulk-prefetch this thread group's W chunks into L2 (ptxas sinks the register W
                # loads above below the wait: SASS showed only the first stage before ACQBULK); x by __ldcg (no .nc)
                src = src.replace(anchor, '    if (pfl2)\n'
                                  '        for (long j = t; c + j * G < nch; j += 256)\n'
                                  '            asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }" :: "l"(W + (c + j * G) * (CE / 8)), "r"(CE * 2) : "memory");\n'
                                  '    asm volatile("griddepcontrol.wait;" ::: "memory");\n' + anchor.replace("__ldg(", "__ldcg("))
                gsig = "__nv_bfloat16* __restrict__ out, int n) {"
                assert src.count(gsig) == 1
                src = src.replace(gsig, "__nv_bfloat16* __restrict__ out, int n, int pfl2) {")
                src = src.replace(sig, "int64_t blocks, int64_t pdl) {")
                src = src.replace(launch, (
                    "        cudaLaunchConfig_t lc = {}; lc.gridDim = blocks; lc.blockDim = 256 * V_; lc.dynamicSmemBytes = sm; \\\n"
                    "        lc.stream = s; cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \\\n"
                    "        la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0; \\\n"
                    "        cudaLaunchKernelEx(&lc, g10::gemv<M_, U_, V_, B_, K_, P_>, (const uint4*)w.data_ptr(), \\\n"
                    "                           (const uint4*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)n, (int)(pdl >= 2)); return; }"))
            if UNPOOL_PDL:  # gemv12 too: as a PDL secondary it L2-prefetches its W slice, then griddepcontrol.wait
                reps = [
                    ("int* __restrict__ cnt, int n, int nrg_all, int k, int bk) {",
                     "int* __restrict__ cnt, int n, int nrg_all, int k, int bk, int pdl) {"),
                    ("    for (int i = t; i < M * (KC / 8); i += blockDim.x) {\n        const int r = i / (KC / 8), q = i % (KC / 8);",
                     "    if (pdl) {\n"
                     "        const char* pb = (const char*)(W + ((long)ks * nrg_all + rg0) * KC);\n"
                     "        for (long o = (long)t * 128; o < (long)NG * KC * 16; o += (long)blockDim.x * 128)\n"
                     "            asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pb + o));\n"
                     "        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n    }\n"
                     "    for (int i = t; i < M * (KC / 8); i += blockDim.x) {\n        const int r = i / (KC / 8), q = i % (KC / 8);"),
                    ("int64_t kcu, int64_t wg, int64_t pf, int64_t bk) {", "int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t pdl) {"),
                    ("        g12::gemm<M_, U_, G_, P_><<<S * bk, threads, sm, s>>>((const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \\\n"
                     "        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk); return; }",
                     "        cudaLaunchConfig_t lc = {}; lc.gridDim = S * bk; lc.blockDim = threads; lc.dynamicSmemBytes = sm; \\\n"
                     "        lc.stream = s; cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \\\n"
                     "        la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0; \\\n"
                     "        cudaLaunchKernelEx(&lc, g12::gemm<M_, U_, G_, P_>, (const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \\\n"
                     "        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, (int)nrg_all, k, (int)bk, (int)pdl); return; }"),
                ]
                for x_, y_ in reps:
                    assert src.count(x_) == 1, x_
                    src = src.replace(x_, y_)
            if GEMV_PDL:  # let the next kernel (boundary / rmsnorm, launched with PDL) start its prologue early
                for anchor in ("    const int t = threadIdx.x & 255, vg", "    const int t = threadIdx.x, lane"):
                    assert src.count(anchor) == 1, anchor
                    src = src.replace(anchor, '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n' + anchor)
            src13 = (_G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(8192,%d,%d,%d,%d)" % c[:4] for c in sorted(set(G13_CFG.values()) | {a for v in G13_ALT.values() for a in v}))
                                      + " " + " ".join("G13_CASE(1024,%d,%d,%d,%d)" % c[:4] for c in [G13_O] if c))
                     + "\n" + _G14_SRC.replace("G14_CONFIGS", " ".join("G14_CASE(%d,%d,%d,%d,%d)" % (G12_TOKENS, c[0], c[1], c[2], c[4])
                                                                      for c in sorted(set(G14_CFG.values()) | {a for v in G14_ALT.values() for a in v}))))
            if GEMV_PDL:
                for anchor in ("    const int t = threadIdx.x & 255, vg", "    const int t = threadIdx.x, lane"):
                    assert src13.count(anchor) == 1, anchor
                    src13 = src13.replace(anchor, '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n' + anchor)
            # KV_PF: gemv13 (the qkv GEMV at 1 token) also L2-prefetches the attention's KV rows: thread i of the grid
            # reads page-table entry pft[i] (i < pfn) after issuing its W prologue and prefetches that K and V row
            # (pfrb bytes apart per page). csl (d1 sliding) then gathers from L2 instead of HBM.
            reps = [
                ("__nv_bfloat16* __restrict__ out, int n, float scale) {",
                 "__nv_bfloat16* __restrict__ out, int n, float scale,\n"
                 "                                                       const int* __restrict__ pft, const char* __restrict__ pfk,\n"
                 "                                                       const char* __restrict__ pfv, int pfn, int pfrb, int pdl) {"),
                ("        if (i < KK / 8) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);\n",
                 "        if (i < KK / 8 && !pdl) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);\n"),
                ("        if (c + p * G < nch) load(p, c + p * G);\n",
                 "        if (c + p * G < nch) load(p, c + p * G);\n"
                 "    if (pfn > 0) {\n"
                 "        const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                 "        if (i < pfn) {\n"
                 "            const int pg = __ldg(pft + i);\n"
                 "            if (pg >= 0) {\n"
                 "                asm volatile(\"prefetch.global.L2::evict_last [%0];\" :: \"l\"(pfk + (long)pg * pfrb));\n"
                 "                asm volatile(\"prefetch.global.L2::evict_last [%0];\" :: \"l\"(pfv + (long)pg * pfrb));\n"
                 "            }\n        }\n    }\n"
                 "    if (pdl) {  // s33: PDL secondary: W prologue above, x (the predecessor's output) after the wait\n"
                 "        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
                 "#pragma unroll\n"
                 "        for (int q = 0; q < XV; ++q) {\n"
                 "            const int i = threadIdx.x + q * 256 * NV;\n"
                 "            if (i < KK / 8) xv[q] = __ldcg(reinterpret_cast<const uint4*>(x) + i);\n"
                 "        }\n    }\n"),
                ("int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks) {",
                 "int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks,\n"
                 "              torch::Tensor pft, torch::Tensor pfk, torch::Tensor pfv, int64_t pfn, int64_t pfrb, int64_t pdl) {"),
                ("        g13::gemv<U_, V_, B_, K_, P_><<<blocks, 256 * V_, 0, st>>>(",
                 "        cudaLaunchConfig_t lc = {}; lc.gridDim = blocks; lc.blockDim = 256 * V_; lc.dynamicSmemBytes = 0; \\\n"
                 "        lc.stream = st; cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \\\n"
                 "        la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0; \\\n"
                 "        cudaLaunchKernelEx(&lc, g13::gemv<U_, V_, B_, K_, P_>, "),
                ("(__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }",
                 "(__nv_bfloat16*)out.data_ptr(), n, (float)scale, \\\n"
                 "        (const int*)pft.data_ptr(), (const char*)pfk.data_ptr(), (const char*)pfv.data_ptr(), (int)pfn, (int)pfrb, (int)pdl); return; }"),
            ]
            for x_, y_ in reps:
                assert src13.count(x_) == 1, x_
                src13 = src13.replace(x_, y_)
            src = src + "\n" + src13 + "\n" + _R3_SRC.replace("R3_CONFIGS", " ".join("R3_CASE(%d,%d,%d)" % c for c in sorted(set(ROUTER3_CFG.values()))))
            # s39 ROUTE_PDL: pr3::router lets its PDL dependent (_routing, which waits before reading the logits) launch
            # once its MMA part is done (the wait for the zero kernel, before the atomics)
            _a = '    asm volatile("griddepcontrol.wait;\\n" ::: "memory");  // out zeroed\n'
            assert _GPR3_SRC.count(_a) == 1
            src = src + "\n" + _GPR3_SRC.replace(_a, '    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n' + _a).replace("PR3_CONFIGS", " ".join("PR3_CASE(%d,%d,%d,%d,%d,%d)" % c for c in sorted(set(PR3_CFG.values()) | set(PR3_BUILD))))
            if G19_CFG and _G19_SRC is not None:
                src = src + "\n#define G19_ONLY2\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
                    "G19_CASE(%d,%d,%d,%d,1,%d)" % (c[0], c[1], c[2], c[3], c[5]) for c in sorted(set(G19_CFG.values()) | set(G19_BUILD))))
            else:
                src = src + "\nvoid g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}\n"
            src = src + "\n" + _G17_SRC.replace("G17_CONFIGS", " ".join("G17_CASE(%d,%d,%d)" % c for c in sorted(set(G17_CFG.values())))) + "\n" + _N2U_SRC
            if _GPR1_SRC is not None:
                src = src + "\n" + _GPR1_SRC.replace("PR1_CONFIGS", " ".join("PR1_CASE(%d,%d,%d,%d)" % c for c in sorted({(*v[:2], pf, v[3]) for v in PR1_CFG.values() for pf in (v[2], *PR1_PF_ALT)})))
                if _GPR8_SRC is not None:
                    src = src + "\n" + _GPR8_SRC.replace("PR8_CONFIGS", " ".join("PR8_CASE(%d,%d,%d,%d,%d,%d)" % (c[0], c[1], c[2], c[4] & 15, c[4] >> 4, 512 // c[3]) for c in sorted(set(PR8_CFG.values()) | set(PR8L_ALTS))))
            else:
                src = src + "\nvoid pr1_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::vector<torch::Tensor>, double, double) {}\n"
            if _GOB8_SRC is not None and OB8:  # s86: dead (H100), built only when forced on
                src = src + "\n" + _GOB8_SRC.replace("OB8_CONFIGS", "OB8_CASE(%d)" % G12_CFG["o"][2])
            else:
                src = src + "\nvoid ob8_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, double, double, int64_t) {}\n"
            _G10["lib"] = load_inline(
                name="layer_g10_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                cpp_sources="void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, "
                            "int64_t, int64_t" + (", int64_t" if UNPOOL_PDL else "") + ");\n"
                            "void g12_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t" + (", int64_t" if UNPOOL_PDL else "") + ");\n"
                            "void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                            "void g14_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                            "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                            "void r3_router(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                            "void g17_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);\n"
                            "void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                            "void n2u_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "int64_t, int64_t, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, std::vector<torch::Tensor>, double, double);\n"
                            "void nu8_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
                            "torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);\n"
                            "void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                            "void pr1_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::vector<torch::Tensor>, double, double);\n"
                            + ("void pr8_launch(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, "
                               "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                               "torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);" if _GPR1_SRC is not None and _GPR8_SRC is not None else "")
                            + "void ob8_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, double, double, int64_t);\n",
                cuda_sources=src, functions=["ob8_gemm", "g10_gemv", "g12_gemm", "g13_gemv", "g14_gemm", "r3_router", "g17_gemv", "n2u_launch", "nu8_launch", "pj_router3", "g19_gemm", "pr1_launch"]
                + (["pr8_launch"] if _GPR1_SRC is not None and _GPR8_SRC is not None else []),
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
        except Exception as e:  # noqa: BLE001
            print("layer: gemv10/12 build failed, cuBLAS:", e)
    return _G10["lib"]


# FP32 router up to 8 tokens without split K (r3_router; proj's router2 r-proj e605717 with FP32 x, tokens split over
# blockIdx.y in groups of M so x fits in registers): tokens -> (M, rows per block, threads). H100 cold L2 m8: 8.74 us vs
# _router_fma 11.52 (+ its 8 split partials); m1 6.91 vs 7.10 (tie: not used).
ROUTER3_CFG = {8: (4, 2, 512)}
# Router above 8 tokens = SGLang's BF16 hi/lo split in ONE split-K mma kernel (proj's router3, r-proj a1f0af7 dev/router3.cu
# = pr3_src.cu): FP32 atomics into its output (zeroed by a tiny PDL-primary kernel). tokens -> (MT, NT, WM, WN, KC, NS).
# proj H200 standalone: T16 7.38 -> 5.55, T32 8.66 -> 6.36, T128 10.24 -> 9.00 us.
# Layer H100: T=128 (v16) -4.5 us; T=16/32 +1..3 us (its 72 KB-smem CTAs crowd the pooler bmm beside it): off there.
PR3_CFG = {128: (128, 64, 4, 4, 256, 4)}
PR3_BUILD = [(16, 64, 1, 4, 256, 4)]  # also compiled (A/B at T=16/32)
_R3_SRC = r'''// router3: proj's router2 (no split K, x in registers, W float4 loads issued up front) with FP32 x; tokens split over
// blockIdx.y in groups of M so that x fits in registers.
namespace r3 {
template <int M, int R, int T, int K>
__global__ void __launch_bounds__(T) router(const float4* __restrict__ w, const float4* __restrict__ x,
                                            float* __restrict__ out, int m, int n) {
    constexpr int J = K / (4 * T), NW = T / 32;
    __shared__ float red[NW][R * M];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r0 = blockIdx.x * R, i0 = blockIdx.y * M;
    float4 a[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = __ldcs(w + (size_t)(r0 + r) * (K / 4) + t + T * j);
    float4 xr[M][J];
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
            xr[i][j] = i0 + i < m ? __ldg(x + (size_t)(i0 + i) * (K / 4) + t + T * j) : make_float4(0, 0, 0, 0);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r][i] = fmaf(a[r][j].x, xr[i][j].x, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].y, xr[i][j].y, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].z, xr[i][j].z, acc[r][i]);
                acc[r][i] = fmaf(a[r][j].w, xr[i][j].w, acc[r][i]);
            }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            float v = acc[r][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][r * M + i] = v;
        }
    __syncthreads();
    if (t < R * M) {
        const int r = t / M, i = t % M;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) s += red[q][t];
        if (i0 + i < m) out[(size_t)(i0 + i) * n + r0 + r] = s;
    }
}
}  // namespace r3
#define R3_CASE(M, R, T) \
    if (mp == M && rr == R && tt == T) { \
        r3::router<M, R, T, 8192><<<dim3(n / R, (m + M - 1) / M), T, 0, c10::cuda::getCurrentCUDAStream()>>>( \
            (const float4*)w.data_ptr(), (const float4*)x.data_ptr(), out.data_ptr<float>(), m, n); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
void r3_router(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mp, int64_t rr, int64_t tt) {
    const int m = (int)x.size(0), n = (int)w.size(0);
    R3_CONFIGS
    TORCH_CHECK(false, "r3: no config");
}
'''



# BF16 unpool slice (n = 1024, K = 4096) at 1 and 8 tokens: proj's gemv17 (r-proj dev/gemv17.cu @405439e: R whole rows
# per block, x in registers, no split K; judge H200 m1 4.67 -> 3.75 us, m8 6.50 -> 5.99 vs gemv10/gemv12). Layer
# changes: W loads are issued before griddepcontrol.wait (PDL secondary of moe_norm2), and it triggers its dependents.
# tokens -> (M, rows per block, threads)
G17_CFG = {1: (1, 4, 512), 8: (8, 4, 256)}
G17_KEYS = ("unpool_slice",)
_G17_SRC = r'''namespace g17 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

template <int M, int R, int T, int K>
__global__ void __launch_bounds__(T) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ out, int m, int n) {
    constexpr int J = K / (8 * T), NW = T / 32;
    static_assert(K % (8 * T) == 0, "k split");
    __shared__ float red[NW][R * M];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, r0 = blockIdx.x * R;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // s14: ptxas may sink the W register loads below griddepcontrol.wait (an acquire), so the block's R contiguous W
    // rows are also bulk-prefetched into L2 first (kept before ACQBULK in SASS).
    if (t == 0)
        asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }" :: "l"(w + (size_t)r0 * (K / 8)), "r"(R * K * 2) : "memory");
    uint4 a[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) a[r][j] = ldw(w + (size_t)(r0 + r) * (K / 8) + t + T * j);
    asm volatile("griddepcontrol.wait;" ::: "memory");  // no-op unless launched as a PDL secondary
    uint4 xr[M][J];
#pragma unroll
    for (int i = 0; i < M; ++i)
#pragma unroll
        for (int j = 0; j < J; ++j)
            // __ldcg, not __ldg: ptxas hoists .nc loads above griddepcontrol.wait (s14: SASS showed x loaded before
            // ACQBULK; as a PDL secondary that read stale x: the s8 'race')
            xr[i][j] = i < m ? __ldcg(reinterpret_cast<const uint4*>(x + (size_t)i * K) + t + T * j) : make_uint4(0, 0, 0, 0);
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const unsigned xv[4] = {xr[i][j].x, xr[i][j].y, xr[i][j].z, xr[i][j].w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const unsigned wv[4] = {a[r][j].x, a[r][j].y, a[r][j].z, a[r][j].w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc[r][i] = fmaf(lo(wv[q]), lo(xv[q]), acc[r][i]);
                    acc[r][i] = fmaf(hi(wv[q]), hi(xv[q]), acc[r][i]);
                }
            }
        }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            float v = acc[r][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][r * M + i] = v;
        }
    __syncthreads();
    for (int e = t; e < R * M; e += T) {
        const int r = e / M, i = e % M;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) s += red[q][e];
        if (i < m) out[(size_t)i * n + r0 + r] = __float2bfloat16(s);
    }
}
}  // namespace g17
#define G17_CASE(M, R, T) \
    if (mp == M && rr == R && tt == T && k == 4096) { \
        cudaLaunchConfig_t lc = {}; lc.gridDim = n / R; lc.blockDim = T; lc.dynamicSmemBytes = 0; \
        lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1]; \
        la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \
        la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0; \
        cudaLaunchKernelEx(&lc, g17::gemv<M, R, T, 4096>, (const uint4*)w.data_ptr(), \
                           (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), m, n); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
void g17_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mp, int64_t rr, int64_t tt, int64_t pdl) {
    const int m = (int)x.size(0), k = (int)x.size(1), n = (int)out.size(1);
    TORCH_CHECK(m <= mp && n % rr == 0, "g17: shape");
    G17_CONFIGS
    TORCH_CHECK(false, "g17: no config");
}
'''


# T = 1: moe_norm2 (post-expert norm + weighted sum, moe_src) fused into the unpool-slice GEMV (gemv17 structure,
# 512 threads, thread t owns k = 8t..8t+7, which is also moe_norm2's layout): every block recomputes the norm of the
# token's local expert outputs (y32 FP32 from moe_tk2f, <= 8 x 16 KB, L2) while its W rows are in flight, rounds it to
# BF16 (as moe_norm2's output) and dots it with its R rows. Bitwise the same as moe_norm2 -> gemv17. Block 0 zeroes
# moe_tk2f's segment counters (moe_norm2's other job). Saves the norm2 launch + its serial latency.
N2U_TOKENS = (1,)
N2U_PFW = True  # s14: n2u bulk-prefetches its W rows into L2 before griddepcontrol.wait (see n2u)
N2U_SLOT = True  # s14: tk2f at T=1 with PF & 4 (84204: y32 in slot order, pre-scaled; bitwise = 204) + n2u<8, SLOT>
N2U_CFG = (8, 1)  # rows per block, PDL secondary of moe_tk2f (s13: which triggers at its CTAs' gemm2 rows, TK2F_LATE_TRIG)
N2U_BOUNDARY = False  # the last block also does the second boundary: H100 d1 +2.7 us vs the PDL _boundary (dead, s8)
# s13: TK2F_LATE_TRIG (T = 1 tk2f instantiation only, PF == 0): 0 = trigger PDL dependents at the kernel's start (moe's),
# 1 = at each CTA's end (after its last gemm2 row), 2 = when a CTA starts its gemm2 rows, 3 = as 2 in every tk2f
# instantiation (T <= 32) + moe_norm2/3 as PDL secondaries (g_pdl 6). With N2U_CFG[1] = 1 n2u then
# launches onto the SMs freed by drained tk2f CTAs and streams its W rows before griddepcontrol.wait.
TK2F_LATE_TRIG = 3
if TK2F_LATE_TRIG:
    _a = '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  {  // routing, as moe_tk2 GATED'
    assert _moe_CUDA_SRC.count(_a) == 1
    _c = "(PF != 0)" if TK2F_LATE_TRIG < 3 else "(false)"
    _moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "  if " + _c + " " + _a[2:])
    _t = ("  if (PF == 0) " if TK2F_LATE_TRIG < 3 else "  ") + 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
    _a = ("  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, "
          "s_red, smem, full,\n                                     empty, st, ph, rb);\n")
    assert _moe_CUDA_SRC.count(_a) == 1
    _moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + _t if TK2F_LATE_TRIG == 1 else _t + _a)
    if TK2F_LATE_TRIG == 3:  # 3: all instantiations trigger at their gemm2 rows, and moe_norm2/3 launch as PDL secondaries
        _a = "static int g_pdl = 4,"
        assert _moe_CUDA_SRC.count(_a) == 1
        _moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "static int g_pdl = 6,")
# s15 UPF3: at T in UPF3, every moe_tk2f CTA bulk-prefetches (default L2 policy) the first UPF3[T] bytes of each of its
# rows of the full unpooler W into L2 after its last gemm2 row (the cuBLAS unpool after moe_norm2/3 then starts with
# that part in L2; HBM is mostly idle in the CTA drain + norm). Pointer via moe_set_upf() before the launch (a host
# static read at launch/capture time; reset to null after).
UPF3 = {}  # DEAD on H200 too (diag b1ee542c: d32 -4%, d16 -1.5%); s15: {T: ("unpool", bytes per row) or ("g19", bytes)}, e.g. {32: ("unpool", 3072), 16: ("g19", 16 << 20)}; H100 d32: AT 0 tie (unpool -4, tk2f +7.8), AT 1 +1..2, AT 2 +1..2 (H200 untested)
UPF3_AT = 0  # 0: tk2f CTA end (H100 d32: unpool -4 us but tk2f +7.8: the stragglers lose bandwidth), 1: moe_norm2/3 after its wait
_a = "int64_t moe_set_sa(int64_t v) { g_sa = (int)v; return 0; }"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + "\nstatic const char* g_upf = nullptr;\nstatic int g_upf_rows = 0, g_upf_rb = 0, g_upf_seg = 0;\n"
                                      "int64_t moe_set_upf(int64_t p, int64_t rows, int64_t rb, int64_t seg) {\n"
                                      "  g_upf = (const char*)p; g_upf_rows = (int)rows; g_upf_rb = (int)rb; g_upf_seg = (int)seg; return 0;\n}")
_a = "    const float* __restrict__ w1, __nv_bfloat16* __restrict__ out, int M, float eps) {\n  static_assert(NC % 8 == 0, \"routing uses warps 0..7 x 32 ids\");"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a.replace("int M, float eps) {", "int M, float eps,\n    const char* upf, int upf_rows, int upf_rb, int upf_seg) {"))
_a = "      (__nv_bfloat16*)out, nr / 8, eps);\n}"
assert _moe_CUDA_SRC.count(_a) == 1, _moe_CUDA_SRC.count(_a)
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "      (__nv_bfloat16*)out, nr / 8, eps, g_upf, g_upf_rows, g_upf_rb, g_upf_seg);\n}")
_a = ("  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, "
      "s_red, smem, full,\n                                     empty, st, ph, rb);\n")
assert _moe_CUDA_SRC.count(_a) == 1
if UPF3_AT == 0:
    _moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + "  if (upf != nullptr && tid < 32)\n"
                                          "    for (int r = blockIdx.x + gridDim.x * tid; r < upf_rows; r += gridDim.x * 32)\n"
                                          "      asm volatile(\"{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }\" :: \"l\"(upf + (long)r * upf_rb), \"r\"(upf_seg) : \"memory\");\n")
else:  # in moe_norm2/3 right after its griddepcontrol.wait (moe_tk2f complete: no competition with its stream)
    _i = _moe_CUDA_SRC.index("__global__ void __launch_bounds__(512) moe_norm2(const float* __restrict__ Y")
    _a = "int count, float eps, __nv_bfloat16* __restrict__ out, int* zc) {"
    _j = _moe_CUDA_SRC.index(_a, _i)
    _moe_CUDA_SRC = (_moe_CUDA_SRC[:_j] + _a.replace("int* zc) {", "int* zc,\n    const char* upf = nullptr, int upf_rows = 0, int upf_rb = 0, int upf_seg = 0) {")
                     + _moe_CUDA_SRC[_j + len(_a):])
    _a = "  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");  // PDL secondary of gemm2\n"
    _j = _moe_CUDA_SRC.index(_a, _i)
    assert _j - _i < 2000
    if UPF3_AT == 1:  # bulk (the CTA retires only after its bulk prefetches complete: H100 norm +5.6, unpool -4)
        _pf = ("  if (upf != nullptr)\n"
               "    for (int r = blockIdx.x + gridDim.x * tid; r < upf_rows; r += gridDim.x * 512)\n"
               "      asm volatile(\"{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }\" :: \"l\"(upf + (long)r * upf_rb), \"r\"(upf_seg) : \"memory\");\n")
    else:  # 2: per-line prefetch.global.L2 (fire and forget), lines spread over the grid
        _pf = ("  if (upf != nullptr) {\n"
               "    const int lpr = upf_seg >> 7, nl = upf_rows * lpr;\n"
               "    for (int l = blockIdx.x * 512 + tid; l < nl; l += gridDim.x * 512)\n"
               "      asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(upf + (long)(l / lpr) * upf_rb + (l % lpr) * 128));\n"
               "  }\n")
    _moe_CUDA_SRC = _moe_CUDA_SRC[:_j + len(_a)] + _pf + _moe_CUDA_SRC[_j + len(_a):]
    _a = "      (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr());\n}"
    assert _moe_CUDA_SRC.count(_a) == 2, _moe_CUDA_SRC.count(_a)
    _moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "      (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr(), g_upf, g_upf_rows, g_upf_rb, g_upf_seg);\n}")
_moe__CPP = _moe__CPP + "int64_t moe_set_upf(int64_t, int64_t, int64_t, int64_t);"
# s26 NU8 (T = 8): moe_tk2f PF 20 (= 4 | 16; cfg 85204) also accumulates each route slot's post-norm sum of squares
# sum_k (y w1)^2 (moe_norm3's tot) by atomics into float row 64 of y32 (zeroed by the pre-expert _rmsnorm_gain), so the
# layer's nu8 kernel can do norm3 + the unpool slice split over k (no norm3 launch, W in smem before its wait).
# w1 via moe_set_ss() (a host static read at launch / capture time, as moe_set_upf).
_a = "                                             int& ph, int& rb) {\n  constexpr int RPE = NRT / 2, NQ = 64 / NC, NS = NQ / KPS, SW = NC * 32, SB = NC * KPS * ((PF & 1024) ? 1120 : 1152);"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a.replace("int& ph, int& rb) {", "int& ph, int& rb, const float* w1x = nullptr, float* ssx = nullptr) {")
                                      + "\n  float ssa = 0.f;  // s26 NU8 (PF & 16)")
_a = ("        if (PF & 4) reinterpret_cast<float*>(Out)[(size_t)s_srt[off + tok] * NOUT + col] = v * gsc;  // route slot order\n"
      "        else reinterpret_cast<float*>(Out)[(size_t)(off + tok) * NOUT + col] = v;\n      }\n")
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a + """      if constexpr ((PF & 16) != 0) {  // s26 NU8: slot sum of squares of y w1, kept per thread over the segment's rows
        const float q = tok < n ? (v * gsc) * w1c : 0.f;
        ssa = fmaf(q, q, ssa);
        if (R + 1 == seg_end || R + 1 == r1) {  // the segment's (slot set's) last row in this CTA: one atomic per 8 columns
          float x = ssa;
          x += __shfl_xor_sync(0xffffffffu, x, 4);
          x += __shfl_xor_sync(0xffffffffu, x, 8);
          x += __shfl_xor_sync(0xffffffffu, x, 16);
          if (gg == 0 && tok < n) atomicAdd(ssx + s_srt[off + tok], x);
          ssa = 0.f;
        }
      }
""")
_a = ("  tk2f_consume<256, NC, NSTG, false, KPS, SA, PF>(slut, gs2, inter, Y, cnt, n13, b0, b1, s_rs2, s_e, s_off, s_srt, "
      "s_red, smem, full,\n                                     empty, st, ph, rb);\n")
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a.replace("empty, st, ph, rb);", "empty, st, ph, rb, w1, (PF & 16) ? Y + 64 * 4096 : nullptr);"))
_a = "    constexpr int NA = (PF & 8) ? 2 : 1;  // s13 PF & 8"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, """    float w1c = 0.f;  // s26 NU8: this thread's epilogue column's w1, loaded before the row's MMA loop
    if constexpr ((PF & 16) != 0 && !GATED) {
      const int r_ = (tid >> 5) & 3;
      w1c = __ldg(w1x + ((R - seg_start) + (tid >> 7) * RPE) * 16 + ((tid & 31) >> 2) + ((r_ & 2) ? 8 : 0));
    }
""" + _a)
_a = "    if (pf == 12) return sa ?"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "    if (pf == 5) {  // s26 NU8: PF 20 (T = 8 only: sums at y32 row 64)\n"
                                      "      TORCH_CHECK(nr == 64 && y_.size(0) >= 65 && g_ssw1 != nullptr, \"tk2f PF 20: shape\");\n"
                                      "      return sa ? tk2f_go<8, 4, 2, false, false, true, 8, 0, 20>(0, TK2F_ARGS0, nullptr, g_ssw1) : "
                                      "tk2f_go<8, 4, 2, false, false, false, 8, 0, 20>(0, TK2F_ARGS0, nullptr, g_ssw1);\n    }\n" + _a)
_a = "\nstatic const char* g_upf = nullptr;"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "\nstatic const float* g_ssw1 = nullptr;\nint64_t moe_set_ss(int64_t p) { g_ssw1 = (const float*)p; return 0; }" + _a)
_moe__CPP = _moe__CPP + "int64_t moe_set_ss(int64_t);"
# s27 TK2FX (T = 1 / 8, pr1 paths: no side stream): moe_tk2f as a PDL secondary of the pre-expert _rmsnorm_gain, which
# triggers right after its own griddepcontrol.wait (so pr1, which wrote ids, has completed): tk2f's routing and its
# producer's first weight stages overlap the rmsnorm + launch gap; the consumers wait before their first x read (x =
# the rmsnorm's output, read with __ldcg: __ldg/.nc gets hoisted above griddepcontrol.wait, NOTES s14). Without the
# PDL launch attribute the wait is a no-op. Host static g_tk2fpdl (moe_set_tk2fpdl) read by tk2f_go at launch/capture.
_a = ("  tk2f_consume<512, NC, NSTG, true, KPS, SA, PF>(slut, gs13, X, inter, cnt, n13, a0, a1, s_rs13, s_e, s_off, s_srt, s_red, smem, full,\n")
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, '  asm volatile("griddepcontrol.wait;" ::: "memory");  // s27 TK2FX: x is the PDL predecessor\'s output\n' + _a)
_a = """        if (GATED) {
          xf[qq][0] = v ? __ldg(xr + kc * 8) : make_uint4(0, 0, 0, 0);
          xf[qq][1] = v ? __ldg(xr + kc * 8 + 1) : make_uint4(0, 0, 0, 0);
"""
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a.replace("__ldg", "__ldcg"))
_a = "  return launch_k(moe_tk2f<NC, NSTG, KPS, DYN, FN, SA, GS, MRG, PF>, grid, (NC + 1) * 32, SMEM, st, false,"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "  const bool pdlx = g_tk2fpdl != 0;  // one-shot\n  g_tk2fpdl = 0;\n" + _a.replace("st, false,", "st, pdlx,"))
_a = "\nstatic const char* g_upf = nullptr;"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, "\nstatic int g_tk2fpdl = 0;\nint64_t moe_set_tk2fpdl(int64_t v) { g_tk2fpdl = (int)v; return 0; }" + _a)
_moe__CPP = _moe__CPP + "int64_t moe_set_tk2fpdl(int64_t);"
# s27 GNORM (T > 32): the pre-expert rmsnorm fused into moe_gather: each gathered row (expert order, K-permuted as
# moe_gather) normalizes its source token from the pooler output itself (FP32 input rounded to BF16 first, as
# _rmsnorm_gain ROUND; r = rsqrt(mean(x^2) + eps), out = BF16(x * r * gain)): one launch and the p round trip fewer.
_a = "int64_t moe_gather_launch(torch::Tensor x, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor xg) {"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, r"""template <bool F32>
__global__ void __launch_bounds__(256) moe_gathern(const void* __restrict__ X, const float* __restrict__ gain,
                                                   const int* __restrict__ meta, const int* __restrict__ sroutes,
                                                   uint32_t* __restrict__ XG, float eps, long xs) {
  const int pos = blockIdx.x, tid = threadIdx.x;
  __shared__ float s_r[8];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (pos >= __ldcg(meta + 65)) return;
  const size_t row = (size_t)(__ldcg(sroutes + pos) >> 3);
  float v[KD / 2 / 256][2];
  int sw[KD / 2 / 256];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < KD / 2 / 256; ++i) {
    const int o = tid + 256 * i, w = o & 31, t = w & 3, jh = w >> 2;
    sw[i] = (o & ~31) | (8 * t + jh);
    float2 f;
    if constexpr (F32) {
      f = __ldcg(reinterpret_cast<const float2*>(X) + row * xs + sw[i]);
      f = __bfloat1622float2(__float22bfloat162_rn(f));
    } else {
      const uint32_t u = __ldcg(reinterpret_cast<const unsigned int*>(X) + row * xs + sw[i]);
      f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
    }
    v[i][0] = f.x; v[i][1] = f.y;
    ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) s_r[tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < 8; ++w) tot += s_r[w];
  const float r = rsqrtf(tot / KD + eps);
  uint32_t* dst = XG + (size_t)pos * (KD / 2);
#pragma unroll
  for (int i = 0; i < KD / 2 / 256; ++i) {
    const float2 g = __ldg(reinterpret_cast<const float2*>(gain) + sw[i]);
    const __nv_bfloat162 b = __floats2bfloat162_rn(v[i][0] * r * g.x, v[i][1] * r * g.y);
    dst[tid + 256 * i] = *reinterpret_cast<const uint32_t*>(&b);
  }
}
int64_t moe_gathern_launch(torch::Tensor x, torch::Tensor gain, torch::Tensor meta, torch::Tensor sroutes,
                           torch::Tensor xg, double eps) {
  TORCH_CHECK(x.size(1) == KD && x.stride(1) == 1 && x.stride(0) % 2 == 0 && gain.numel() == KD, "moe_gathern: shape");
  const long xs = (long)(x.stride(0) / 2);  // row stride in float2 (FP32) / bf16x2 (BF16) units
  if (x.scalar_type() == at::kFloat)
    return launch_k(moe_gathern<true>, (int)sroutes.numel(), 256, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
        (const void*)x.data_ptr(), (const float*)gain.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(),
        (uint32_t*)xg.data_ptr(), (float)eps, xs);
  return launch_k(moe_gathern<false>, (int)sroutes.numel(), 256, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const void*)x.data_ptr(), (const float*)gain.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(),
      (uint32_t*)xg.data_ptr(), (float)eps, xs);
}
""" + _a)
_moe__CPP = _moe__CPP + "int64_t moe_gathern_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);"
# s52 RG (moe s33 moe_routegather, r-moe 7ac1069) merged with GNORM: moe_route + moe_gathern in ONE launch, one CTA per
# token. Every CTA counts all routes per local expert (and those of earlier tokens) itself, so a route's sorted position
# = base[e] + #earlier routes to e (stable (expert, route) order, no cross-CTA sync); CTA 0 writes meta (= moe_route's
# fields); each CTA normalizes its token's row ONCE (moe_gathern's numerics) and writes it to each of its routes' slots.
# ids / X are PDL-producer data: __ldcg only (NOTES s14: .nc loads get hoisted above griddepcontrol.wait).
_a = "int64_t moe_gathern_launch(torch::Tensor x, torch::Tensor gain, torch::Tensor meta, torch::Tensor sroutes,"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, r"""template <bool F32>
__global__ void __launch_bounds__(256) moe_routegathern(const int* ids, int nroutes, int count, const void* X,
                                                        const float* __restrict__ gain, int* __restrict__ meta,
                                                        int* __restrict__ sroutes, uint32_t* __restrict__ XG, float eps, long xs) {
  __shared__ int cnt[32], pre[32], s_pos[8];
  __shared__ float s_r[8];
  const int tid = threadIdx.x, tok = blockIdx.x, lane = tid & 31;
  if (tid < 32) { cnt[tid] = 0; pre[tid] = 0; }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();
  const int lim = tok * 8;
  for (int i = tid; i < ((nroutes + 255) & ~255); i += 256) {  // uniform trip count: __match_any_sync below
    const int e = i < nroutes ? __ldcg(ids + i) : -1;
    const bool v = e >= 0 && e < count;
    const unsigned m = __match_any_sync(0xffffffffu, v ? e : -1);
    const unsigned mp = __match_any_sync(0xffffffffu, (v && i < lim) ? e : -1);
    if (v && (__ffs(m) - 1) == lane) atomicAdd(&cnt[e], __popc(m));
    if (v && i < lim && (__ffs(mp) - 1) == lane) atomicAdd(&pre[e], __popc(mp));
  }
  // this token's row: load + sum of squares while the counts settle
  const size_t row = (size_t)tok;
  float v[KD / 2 / 256][2];
  int sw[KD / 2 / 256];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < KD / 2 / 256; ++i) {
    const int o = tid + 256 * i, w = o & 31, t = w & 3, jh = w >> 2;
    sw[i] = (o & ~31) | (8 * t + jh);
    float2 f;
    if constexpr (F32) {
      f = __ldcg(reinterpret_cast<const float2*>(X) + row * xs + sw[i]);
      f = __bfloat1622float2(__float22bfloat162_rn(f));
    } else {
      const uint32_t u = __ldcg(reinterpret_cast<const unsigned int*>(X) + row * xs + sw[i]);
      f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
    }
    v[i][0] = f.x; v[i][1] = f.y;
    ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) s_r[tid >> 5] = ss;
  __syncthreads();
  if (tid < 32) {
    const int c = cnt[tid];
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += u;
    }
    const int b = incl - c;
    pre[tid] += b;
    if (tok == 0) {
      meta[33 + tid] = b;
      if (tid == 31) { meta[65] = incl; meta[66] = 0; meta[67] = 0; }
      meta[68 + tid] = 0;
      meta[100 + tid] = 0;  // moe n4s1 NORM_EARLY gemm2 flags
      int rank = 0;
      for (int j = 0; j < 32; ++j) {
        const int cj = __shfl_sync(0xffffffffu, c, j);
        rank += (cj > c) || (cj == c && j < tid);
      }
      const int na = __popc(__ballot_sync(0xffffffffu, c > 0));
      if (tid == 0) meta[0] = na;
      if (tid >= na) meta[1 + tid] = 0;
      __syncwarp();
      if (c > 0) meta[1 + rank] = tid;
    }
  }
  __syncthreads();
  if (tid < 8) {
    const int e = __ldcg(ids + lim + tid);
    int pos = -1;
    if (e >= 0 && e < count) {
      pos = pre[e];
      for (int j = 0; j < tid; ++j) pos += __ldcg(ids + lim + j) == e;  // repeated ids in a token (not from topk)
      sroutes[pos] = lim + tid;
    }
    s_pos[tid] = pos;
  }
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < 8; ++w) tot += s_r[w];
  const float r = rsqrtf(tot / KD + eps);
  uint32_t o32[KD / 2 / 256];
#pragma unroll
  for (int i = 0; i < KD / 2 / 256; ++i) {
    const float2 g = __ldg(reinterpret_cast<const float2*>(gain) + sw[i]);
    const __nv_bfloat162 b = __floats2bfloat162_rn(v[i][0] * r * g.x, v[i][1] * r * g.y);
    o32[i] = *reinterpret_cast<const uint32_t*>(&b);
  }
  __syncthreads();
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int pos = s_pos[s];
    if (pos < 0) continue;
    uint32_t* dst = XG + (size_t)pos * (KD / 2);
#pragma unroll
    for (int i = 0; i < KD / 2 / 256; ++i) dst[tid + 256 * i] = o32[i];
  }
}
int64_t moe_routegathern_launch(torch::Tensor ids, int64_t count, torch::Tensor x, torch::Tensor gain, torch::Tensor meta,
                                torch::Tensor sroutes, torch::Tensor xg, double eps) {
  TORCH_CHECK(x.size(1) == KD && x.stride(1) == 1 && x.stride(0) % 2 == 0 && gain.numel() == KD && ids.numel() == 8 * x.size(0),
              "moe_routegathern: shape");
  const long xs = (long)(x.stride(0) / 2);
  auto k = x.scalar_type() == at::kFloat ? moe_routegathern<true> : moe_routegathern<false>;
  return launch_k(k, (int)x.size(0), 256, 0, at::cuda::getCurrentCUDAStream(), (g_pdl & 4) != 0,
      (const int*)ids.data_ptr(), (int)ids.numel(), (int)count, (const void*)x.data_ptr(), (const float*)gain.data_ptr(),
      (int*)meta.data_ptr(), (int*)sroutes.data_ptr(), (uint32_t*)xg.data_ptr(), (float)eps, xs);
}
""" + _a)
_moe__CPP = _moe__CPP + "int64_t moe_routegathern_launch(torch::Tensor, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);"
# s39 ROUTE_PDL: moe_route as a PDL secondary of the routing (which triggers at its start): its launch and smem setup
# overlap the routing. It waits before reading ids, read with __ldcg (an __ldg/.nc load gets hoisted above the wait,
# NOTES s14). Without the PDL attribute (moe_src's own paths) the wait is a no-op.
_a = """  if (tid < 32) { cnt[tid] = 0; fill[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < nroutes; i += blockDim.x) {
    const int e = ids[i] - start;
    if (e >= 0 && e < count) atomicAdd(&cnt[e], 1);"""
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, _a.replace("  __syncthreads();\n", '  __syncthreads();\n  asm volatile("griddepcontrol.wait;" ::: "memory");\n').replace("ids[i]", "__ldcg(ids + i)"))
_a = """    if (e >= 0 && e < count) sroutes[base[e] + atomicAdd(&fill[e], 1)] = i;"""
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a.replace("    if", "    const int e = ids[i] - start;\n    if"),
                                      _a.replace("    if", "    const int e = __ldcg(ids + i) - start;\n    if"))
assert _moe_CUDA_SRC.count("__ldcg(ids + i) - start;") == 2 and "griddepcontrol.wait;\" ::: \"memory\");\n  for (int i = tid; i < nroutes; i += blockDim.x) {\n    const int e = __ldcg(" in _moe_CUDA_SRC
_a = "int64_t moe_route_launch(torch::Tensor ids, int64_t start, int64_t count, torch::Tensor meta, torch::Tensor sroutes) {"
assert _moe_CUDA_SRC.count(_a) == 1
_moe_CUDA_SRC = _moe_CUDA_SRC.replace(_a, """int64_t moe_route_pdl_launch(torch::Tensor ids, int64_t start, int64_t count, torch::Tensor meta, torch::Tensor sroutes) {
  return launch_k(moe_route, 1, 1024, 0, at::cuda::getCurrentCUDAStream(), true, (const int*)ids.data_ptr(), (int)ids.numel(),
                  (int)start, (int)count, (int*)meta.data_ptr(), (int*)sroutes.data_ptr());
}
""" + _a)
_moe__CPP = _moe__CPP + "int64_t moe_route_pdl_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor);"


def _moe__lib():  # = moe_src's _lib() + moe_set_upf
    global _moe__LIB
    if _moe__LIB is None:
        from torch.utils.cpp_extension import load_inline
        src = _moe_CUDA_SRC.replace("#define TK2F_DIAG 0", "#define TK2F_DIAG 1") if _moe_TK2F_DIAG else _moe_CUDA_SRC  # s81
        h = hashlib.sha256(src.encode()).hexdigest()[:12]
        _moe__LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_moe__CPP, cuda_sources=src,
                                functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch", "moe_norme_launch", "moe_sk_launch", "moe_tk2_launch", "moe_tk2f_launch", "moe_tk2fn_launch", "moe_tb_launch",
                                           "moe_gather_launch", "moe_tw_launch", "moe_twf_launch", "moe_twm_launch", "moe_twq_launch",
                                           "moe_route2_launch", "moe_act_launch", "moe_norm2_launch", "moe_norm3_launch", "moe_set_pdl", "moe_set_blk", "moe_set_sa",
                                           "moe_set_upf", "moe_set_ss", "moe_set_tk2fpdl", "moe_gathern_launch", "moe_route_pdl_launch", "moe_routegathern_launch",
                                           "moe_set_hot", "moe_hot_launch"],
                                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"],
                                extra_cflags=["-O3"], verbose=False)
    return _moe__LIB


_N2U_SRC = r'''namespace n2u {
// SLOT (s14, moe_tk2f PF & 4): y32 row = route slot s, pre-scaled by gs2 * 256 (no inv / gs loads); the routing (ids, tw:
// pr1's, complete before tk2f passed its own griddepcontrol.wait) is read before griddepcontrol.wait, so only the y32
// loads stay behind it (H100 stamps: 3 dependent L2 round trips after the wait cost ~2.5-3 us).
template <int R, bool SLOT = false, bool PFW = false>
__global__ void __launch_bounds__(512) n2u(const float* __restrict__ Y, const int* __restrict__ inv,
                                           const int* __restrict__ ids, const float* __restrict__ tw,
                                           const float* __restrict__ w1, const float* __restrict__ gs, int start,
                                           int count, float eps, const uint4* __restrict__ w,
                                           __nv_bfloat16* __restrict__ out, int* zc, int n, const float* __restrict__ s1,
                                           const float* __restrict__ alpha, const float* __restrict__ beta,
                                           const float* __restrict__ bgain, const float* __restrict__ gain,
                                           float* __restrict__ s2, __nv_bfloat16* __restrict__ hid, int* bc,
                                           float eps_post, float eps_h) {
    constexpr int K = 4096, NW = 16;
    __shared__ float ssum[NW][8];
    __shared__ float red[NW][R];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, r0 = blockIdx.x * R;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // s14: ptxas sinks the W register loads below griddepcontrol.wait (an acquire: earlier loads may move after it;
    // SASS: every LDG of W after ACQBULK), so the block's R contiguous W rows are bulk-prefetched into L2 first.
    if (PFW && tid == 0)
        asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }" :: "l"(w + (size_t)r0 * (K / 8)), "r"(R * K * 2) : "memory");
    uint4 a[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a[r] = g17::ldw(w + (size_t)(r0 + r) * (K / 8) + tid);
    float wv[8];
    {
        const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
        float4 p = wp[0], q = wp[1];
        wv[0] = p.x; wv[1] = p.y; wv[2] = p.z; wv[3] = p.w; wv[4] = q.x; wv[5] = q.y; wv[6] = q.z; wv[7] = q.w;
    }
    if (bc != nullptr && tid < 64) {  // L2-prefetch the boundary's FP32 vectors (5 x 32 KB) for the last block
        const float* vecs[5] = {s1, alpha, beta, bgain, gain};
        const int per = (256 + gridDim.x - 1) / gridDim.x, a_ = tid / per, l = blockIdx.x * per + tid % per;
        if (a_ < 5 && l < 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(vecs[a_] + l * 32));
    }
    bool loc[8];
    float twv[8];
    if (SLOT) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int e = ids[s] - start;
            loc[s] = e >= 0 && e < count;
            twv[s] = tw[s];
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (blockIdx.x == 0 && tid < 128) zc[tid] = 0;
    float v[8][8], ss[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int e = 0;
        if (!SLOT) {
            e = ids[s] - start;
            loc[s] = e >= 0 && e < count;
            twv[s] = tw[s];
        }
        ss[s] = 0.f;
        if (loc[s]) {
            const float gsc = SLOT ? 1.0f : gs[e] * 256.0f;
            const float4* p = reinterpret_cast<const float4*>(Y + (size_t)(SLOT ? s : inv[s]) * K + tid * 8);
            const float4 c = p[0], d = p[1];
            const float f[8] = {c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
            if (SLOT) {
#pragma unroll
                for (int k = 0; k < 8; ++k) v[s][k] = f[k] * wv[k];
            } else {
#pragma unroll
                for (int k = 0; k < 8; ++k) v[s][k] = f[k] * gsc * wv[k];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) ss[s] += v[s][k] * v[s][k];
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) v[s][k] = 0.f;
        }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        float x = ss[s];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
        if (lane == 0) ssum[warp][s] = x;
    }
    __syncthreads();
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (!loc[s]) continue;
        float tot = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) tot += ssum[q][s];
        const float f = twv[s] * rsqrtf(tot / K + eps);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += f * v[s][k];
    }
    float xv[8];  // the BF16 rounding of moe_norm2's output
#pragma unroll
    for (int k = 0; k < 8; ++k) xv[k] = __bfloat162float(__float2bfloat16(acc[k]));
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const unsigned wq[4] = {a[r].x, a[r].y, a[r].z, a[r].w};
        float t = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            t = fmaf(g17::lo(wq[q]), xv[2 * q], t);
            t = fmaf(g17::hi(wq[q]), xv[2 * q + 1], t);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (lane == 0) red[warp][r] = t;
    }
    __syncthreads();
    if (tid < R) {
        float t = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) t += red[q][tid];
        out[r0 + tid] = __float2bfloat16(t);
    }
    if (bc == nullptr) return;
    // The last block to finish: the second boundary (as layer.py _boundary, NB = n of N = 8192 columns nonzero):
    // s2 = alpha s1 + beta rmsnorm(f (1 + w)), hid = rmsnorm(s2) (1 + w') in BF16.
    __shared__ int s_last;
    __shared__ float bred[2][NW];
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(bc, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    constexpr int N = 8192;
    float b[16], sv[16];
    float q1 = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int c = h * 4096 + tid * 8 + k;
            b[h * 8 + k] = c < n ? __bfloat162float(__ldcg(out + c)) * bgain[c] : 0.f;
            q1 += b[h * 8 + k] * b[h * 8 + k];
        }
#pragma unroll
    for (int o = 16; o; o >>= 1) q1 += __shfl_xor_sync(0xffffffffu, q1, o);
    if (lane == 0) bred[0][warp] = q1;
    __syncthreads();
    float t1 = 0.f;
#pragma unroll
    for (int q = 0; q < NW; ++q) t1 += bred[0][q];
    const float rb = rsqrtf(t1 / N + eps_post);
    float q2 = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int c0 = h * 4096 + tid * 8;
        const float4 sa = *reinterpret_cast<const float4*>(s1 + c0), sb = *reinterpret_cast<const float4*>(s1 + c0 + 4);
        const float4 aa = *reinterpret_cast<const float4*>(alpha + c0), ab = *reinterpret_cast<const float4*>(alpha + c0 + 4);
        const float4 ba = *reinterpret_cast<const float4*>(beta + c0), bb = *reinterpret_cast<const float4*>(beta + c0 + 4);
        const float sx[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
        const float ax[8] = {aa.x, aa.y, aa.z, aa.w, ab.x, ab.y, ab.z, ab.w};
        const float bx[8] = {ba.x, ba.y, ba.z, ba.w, bb.x, bb.y, bb.z, bb.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sv[h * 8 + k] = ax[k] * sx[k] + bx[k] * (b[h * 8 + k] * rb);
            q2 += sv[h * 8 + k] * sv[h * 8 + k];
        }
        *reinterpret_cast<float4*>(s2 + c0) = make_float4(sv[h * 8], sv[h * 8 + 1], sv[h * 8 + 2], sv[h * 8 + 3]);
        *reinterpret_cast<float4*>(s2 + c0 + 4) = make_float4(sv[h * 8 + 4], sv[h * 8 + 5], sv[h * 8 + 6], sv[h * 8 + 7]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) q2 += __shfl_xor_sync(0xffffffffu, q2, o);
    if (lane == 0) bred[1][warp] = q2;
    __syncthreads();
    float t2 = 0.f;
#pragma unroll
    for (int q = 0; q < NW; ++q) t2 += bred[1][q];
    const float rs = rsqrtf(t2 / N + eps_h);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int c0 = h * 4096 + tid * 8;
        const float4 ga = *reinterpret_cast<const float4*>(gain + c0), gb = *reinterpret_cast<const float4*>(gain + c0 + 4);
        const float gx[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
        uint4 o4;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            op[k] = __floats2bfloat162_rn(sv[h * 8 + 2 * k] * rs * gx[2 * k], sv[h * 8 + 2 * k + 1] * rs * gx[2 * k + 1]);
        *reinterpret_cast<uint4*>(hid + c0) = o4;
    }
    if (tid == 0) *bc = 0;  // for the next call
}
}  // namespace n2u
void n2u_launch(torch::Tensor y32, torch::Tensor inv, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1,
                torch::Tensor gs, int64_t start, int64_t count, double eps, torch::Tensor w, torch::Tensor out,
                torch::Tensor zc, int64_t rr, int64_t pdl, std::vector<torch::Tensor> bnd, double eps_post, double eps_h) {
    // bnd: {} (no boundary) or {s1, alpha, beta, bgain, gain, s2, hid, cnt}
    const bool B = bnd.size() == 8;
    auto P = [&](int i) { return B ? bnd[i].data_ptr() : nullptr; };
    const int n = (int)out.size(1);
    const int R_ = (int)(rr % 1000);  // rr + 1000: SLOT, + 3000: SLOT + PFW
    TORCH_CHECK(out.size(0) == 1 && w.size(1) == 4096 && n % R_ == 0, "n2u: shape");
    cudaLaunchConfig_t lc = {}; lc.gridDim = n / R_; lc.blockDim = 512; lc.dynamicSmemBytes = 0;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0;
#define N2U_ARGS (const float*)y32.data_ptr(), (const int*)inv.data_ptr(), (const int*)ids.data_ptr(), \
        (const float*)tw.data_ptr(), (const float*)w1.data_ptr(), (const float*)gs.data_ptr(), (int)start, (int)count, \
        (float)eps, (const uint4*)w.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int*)zc.data_ptr(), n, \
        (const float*)P(0), (const float*)P(1), (const float*)P(2), (const float*)P(3), (const float*)P(4), (float*)P(5), \
        (__nv_bfloat16*)P(6), (int*)P(7), (float)eps_post, (float)eps_h
    if (rr == 3008) cudaLaunchKernelEx(&lc, n2u::n2u<8, true, true>, N2U_ARGS);
    else if (rr == 1008) cudaLaunchKernelEx(&lc, n2u::n2u<8, true>, N2U_ARGS);
    else if (rr == 4) cudaLaunchKernelEx(&lc, n2u::n2u<4>, N2U_ARGS);
    else if (rr == 8) cudaLaunchKernelEx(&lc, n2u::n2u<8>, N2U_ARGS);
    else if (rr == 16) cudaLaunchKernelEx(&lc, n2u::n2u<16>, N2U_ARGS);
    else TORCH_CHECK(false, "n2u: rows");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#ifndef NU8_PF
#define NU8_PF 1
#endif
namespace nu8 {
// s26 NU8 (T = 8): moe_norm3 (y32 rows = route slots, pre-scaled; per-slot sums of squares at y32 row 64 from moe_tk2f
// PF 20) + the unpool slice (1024 x 4096 BF16) split over k. Block (rg, kc): rows rg * 64.. x k kc * 512..: the W tile
// goes to smem by LDGSTS before griddepcontrol.wait; after it the local slots' y chunks (LDGSTS, one round trip), then
// h = BF16(sum_s f_s y_s w1) (moe_norm3's math, slot order and rounding; thread = k) for the 8 tokens, then
// mma.m16n8k16 (W rows as M, tokens as N, FP32 accumulate): warp = (m tile, k quarter), quarters summed in smem.
// FP32 partials to slab kc (out [8][8][1024]); the boundary adds the slabs and rounds to BF16 as the GEMM would.
// X (s26): no moe_tk2f change (cfg 84204): blocks 0..7 (row group 0, k chunk = blockIdx) add their chunk's partial
// sums of squares of the local slots to y32 row 64 (zeroed with the counter at row 64 col 128 by the pre-expert
// _rmsnorm_gain), then bump the counter; every block waits for 8 (blocks 0..7 are dispatched first and wait only on
// each other: no deadlock while 8 SMs are free).
// D (s59 NU8D): no exchange and no norm in nu8: per local slot i the block's k chunk of BF16(y_i w1) is the mma B
// operand (slots as N), its FP32 partials red.add into out[slot][1024] (local slots' rows zeroed by the pre-expert rmsnorm), blocks of row
// group 0 red.add their chunk's per-slot sums of squares to Y row 64 [slot]; the T=8 boundary (SC) forms
// sum_i f_i rsqrt(ss_i / K + eps) P_i over the token's local slots (masked loads) and rounds it to BF16.
template <int KB, bool X = false, bool D = false>
__global__ void __launch_bounds__(512) nu8(const float* __restrict__ Y, const int* __restrict__ ids,
                                           const float* __restrict__ tw, const float* __restrict__ w1, int start,
                                           int count, float eps, const __nv_bfloat16* __restrict__ w,
                                           float* __restrict__ out, int* zc) {
    static_assert(KB == 512, "thread = k");
    constexpr int K = 4096, RB = 64, CPR = KB / 8, NKC = K / KB, LD = KB + 8;  // LD: padded smem row (bf16)
    constexpr int YS = D ? KB + 8 : KB;  // ys row stride (floats; D: padded for the B fragment loads)
    extern __shared__ __align__(16) unsigned char nu8_smem[];
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(nu8_smem);              // [RB][LD]
    __nv_bfloat16* sh = sw + RB * LD;                                            // [8][LD]
    float* ys = reinterpret_cast<float*>(sh + 8 * LD);                           // [nl][KB], later the k-quarter sums
    __shared__ float sf[64], s_f0[64];  // sf: f by local list position
    __shared__ int sl[64], s_list[64], s_tb[9];
    __shared__ unsigned s_m[2];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kc = blockIdx.x % NKC, rg = blockIdx.x / NKC, k0 = kc * KB, r0 = rg * RB;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int c = tid; c < RB * CPR; c += 512) {
        const int r = c / CPR, q = c % CPR;
        const unsigned sa = (unsigned)__cvta_generic_to_shared(sw + r * LD + q * 8);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(w + (size_t)(r0 + r) * K + k0 + q * 8) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (NU8_PF && blockIdx.x < gridDim.x / 2 && tid >= 64 && tid < 64 + RB) {  // the late half's W tiles into L2
        const int b2 = blockIdx.x + gridDim.x / 2, kc2 = b2 % NKC, rg2 = b2 / NKC;
        asm volatile("{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0; cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, pl; }" :: "l"(w + (size_t)(rg2 * RB + tid - 64) * K + kc2 * KB), "r"(KB * 2) : "memory");
    }
    if (tid < 64) {  // the routing (pr8's, complete before moe_tk2f passed its own griddepcontrol.wait)
        const int e = ids[tid] - start;
        const bool loc = e >= 0 && e < count;
        const unsigned m = __ballot_sync(0xffffffffu, loc);
        if (lane == 0) s_m[warp] = m;
        s_f0[tid] = tw[tid];
        sl[tid] = loc;
    }
    __syncthreads();
    const unsigned long long m64 = (unsigned long long)s_m[0] | ((unsigned long long)s_m[1] << 32);
    if (tid < 64 && ((m64 >> tid) & 1)) s_list[__popcll(m64 & ((1ull << tid) - 1))] = tid;  // local slots, ascending
    if (tid <= 8) s_tb[tid] = tid == 8 ? __popcll(m64) : __popcll(m64 & ((1ull << (tid * 8)) - 1));  // token t: s_tb[t]..
    __syncthreads();
    const int nl = __popcll(m64);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (blockIdx.x == 0 && tid < 128) zc[tid] = 0;  // moe_tk2f's segment counters (moe_norm3's other job)
    constexpr int CPS = KB / 4;  // 16-byte chunks per slot chunk
    for (int c = tid; c < nl * CPS; c += 512) {
        const int i = c / CPS, q = c % CPS;
        const unsigned sa = (unsigned)__cvta_generic_to_shared(ys + i * YS + q * 4);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(Y + (size_t)s_list[i] * K + k0 + q * 4) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    const float wv = __ldg(w1 + k0 + tid);
    int tb[9];
#pragma unroll
    for (int t = 0; t <= 8; ++t) tb[t] = s_tb[t];
    if constexpr (D) {
        __shared__ float s_q[16][64];
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        for (int i = 0; i < nl; ++i) ys[i * YS + tid] *= wv;  // y w1 (FP32, as moe_norm3)
        if (rg == 0) {
            for (int i = 0; i < nl; ++i) {
                float q = ys[i * YS + tid];
                q *= q;
#pragma unroll
                for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
                if (lane == 0) s_q[warp][i] = q;
            }
        }
        __syncthreads();
        if (rg == 0 && tid < nl) {
            float q = 0.f;
#pragma unroll
            for (int ww = 0; ww < 16; ++ww) q += s_q[ww][tid];
            atomicAdd(const_cast<float*>(Y) + 64 * K + s_list[tid], q);  // zeroed by the rmsnorm
        }
        const int mt = warp & 3, kq = warp >> 2, g = lane >> 2, tq = lane & 3, nt = (nl + 7) >> 3;
        float c[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
        const __nv_bfloat16* wa = sw + (mt * 16 + g) * LD + kq * (KB / 4) + 2 * tq;
        const float* yb = ys + g * YS + kq * (KB / 4) + 2 * tq;
#pragma unroll
        for (int ks = 0; ks < KB / 64; ++ks) {
            const int o = ks * 16;
            const unsigned a0 = *reinterpret_cast<const unsigned*>(wa + o), a1 = *reinterpret_cast<const unsigned*>(wa + 8 * LD + o);
            const unsigned a2 = *reinterpret_cast<const unsigned*>(wa + o + 8), a3 = *reinterpret_cast<const unsigned*>(wa + 8 * LD + o + 8);
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                if (n < nt) {
                    const bool ok = n * 8 + g < nl;
                    const float2 p0 = *reinterpret_cast<const float2*>(yb + n * 8 * YS + o);
                    const float2 p1 = *reinterpret_cast<const float2*>(yb + n * 8 * YS + o + 8);
                    __nv_bfloat162 q0 = __floats2bfloat162_rn(p0.x, p0.y), q1 = __floats2bfloat162_rn(p1.x, p1.y);
                    const unsigned b0 = ok ? *reinterpret_cast<unsigned*>(&q0) : 0u, b1 = ok ? *reinterpret_cast<unsigned*>(&q1) : 0u;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(c[n][0]), "+f"(c[n][1]), "+f"(c[n][2]), "+f"(c[n][3])
                                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
        __syncthreads();  // ys consumed: reuse as red [4 quarters][64 slots][64 rows]
        float* red = ys;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            if (n < nt) {
                const int r = mt * 16 + g, col = n * 8 + 2 * tq;
                red[(kq * 64 + col) * 64 + r] = c[n][0];
                red[(kq * 64 + col + 1) * 64 + r] = c[n][1];
                red[(kq * 64 + col) * 64 + r + 8] = c[n][2];
                red[(kq * 64 + col + 1) * 64 + r + 8] = c[n][3];
            }
        }
        __syncthreads();
        for (int idx = tid; idx < nl * 64; idx += 512) {
            const int i = idx >> 6, r = idx & 63;
            const float v = red[i * 64 + r] + red[(64 + i) * 64 + r] + red[(128 + i) * 64 + r] + red[(192 + i) * 64 + r];
            atomicAdd(out + (size_t)s_list[i] * 1024 + r0 + r, v);  // red.add over the 8 k chunks (zeroed by the rmsnorm)
        }
        return;
    }
    if constexpr (X) {
        __shared__ float s_q[16][64];
        int* ctr = reinterpret_cast<int*>(const_cast<float*>(Y) + 64 * K + 128);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        if (blockIdx.x < NKC) {
            for (int i = 0; i < nl; ++i) {
                float q = ys[i * KB + tid] * wv;
                q *= q;
#pragma unroll
                for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
                if (lane == 0) s_q[warp][i] = q;
            }
            __syncthreads();
            if (tid < nl) {
                float q = 0.f;
#pragma unroll
                for (int ww = 0; ww < 16; ++ww) q += s_q[ww][tid];
                atomicAdd(const_cast<float*>(Y) + 64 * K + s_list[tid], q);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) atomicAdd(ctr, 1);
        }
        if (tid < 64) {
            int c_;
            do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(c_) : "l"(ctr) : "memory"); } while (c_ < NKC);
            if (sl[tid]) sf[__popcll(m64 & ((1ull << tid) - 1))] = s_f0[tid] * rsqrtf(__ldcg(Y + 64 * K + tid) / K + eps);
        }
        __syncthreads();
    } else {
        if (tid < 64 && sl[tid])  // f of local slot tid, at its list position
            sf[__popcll(m64 & ((1ull << tid) - 1))] = s_f0[tid] * rsqrtf(__ldcg(Y + 64 * K + tid) / K + eps);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        float acc = 0.f;
        for (int i = tb[t]; i < tb[t + 1]; ++i) acc += sf[i] * (ys[i * KB + tid] * wv);
        sh[t * LD + tid] = __float2bfloat16(acc);
    }
    __syncthreads();
    const int mt = warp & 3, kq = warp >> 2, g = lane >> 2, tq = lane & 3;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* wa = sw + (mt * 16 + g) * LD + kq * (KB / 4) + 2 * tq;
    const __nv_bfloat16* hb = sh + g * LD + kq * (KB / 4) + 2 * tq;
#pragma unroll
    for (int ks = 0; ks < KB / 64; ++ks) {
        const int o = ks * 16;
        const unsigned a0 = *reinterpret_cast<const unsigned*>(wa + o), a1 = *reinterpret_cast<const unsigned*>(wa + 8 * LD + o);
        const unsigned a2 = *reinterpret_cast<const unsigned*>(wa + o + 8), a3 = *reinterpret_cast<const unsigned*>(wa + 8 * LD + o + 8);
        const unsigned b0 = *reinterpret_cast<const unsigned*>(hb + o), b1 = *reinterpret_cast<const unsigned*>(hb + o + 8);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    float* red = ys;  // [4 quarters][64 rows][8 tokens] (the y chunks are consumed)
    red[(kq * 64 + mt * 16 + g) * 8 + 2 * tq] = c[0];
    red[(kq * 64 + mt * 16 + g) * 8 + 2 * tq + 1] = c[1];
    red[(kq * 64 + mt * 16 + g + 8) * 8 + 2 * tq] = c[2];
    red[(kq * 64 + mt * 16 + g + 8) * 8 + 2 * tq + 1] = c[3];
    __syncthreads();
    {
        const int row = tid >> 3, t = tid & 7;
        const float v = red[row * 8 + t] + red[(64 + row) * 8 + t] + red[(128 + row) * 8 + t] + red[(192 + row) * 8 + t];
        out[(size_t)kc * 8 * 1024 + t * 1024 + r0 + row] = v;
    }
}
}  // namespace nu8
void nu8_launch(torch::Tensor y32, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, int64_t start, int64_t count,
                double eps, torch::Tensor w, torch::Tensor out, torch::Tensor zc, int64_t kb, int64_t pdl) {
    TORCH_CHECK(y32.size(0) >= 65 && ids.numel() == 64 && w.size(0) >= 1024 && w.size(1) == 4096 &&
                out.numel() == (kb >= 2000 ? 64 : (4096 / (kb % 1000)) * 8) * 1024, "nu8: shape");
    const int kb_ = (int)(kb % 1000);  // + 1000: X, + 2000: D
    cudaLaunchConfig_t lc = {}; lc.gridDim = (1024 / 64) * (4096 / kb_); lc.blockDim = 512;
    lc.dynamicSmemBytes = 72 * (kb_ + 8) * 2 + 64 * (kb_ + (kb >= 2000 ? 8 : 0)) * 4;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0;
#define NU8_ARGS (const float*)y32.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(), \
        (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (const __nv_bfloat16*)w.data_ptr(), \
        (float*)out.data_ptr(), (int*)zc.data_ptr()
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(nu8::nu8<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 520 * 2 + 64 * 512 * 4);
        cudaFuncSetAttribute(nu8::nu8<512, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 520 * 2 + 64 * 512 * 4);
        cudaFuncSetAttribute(nu8::nu8<512, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 520 * 2 + 64 * 520 * 4);
        init = true;
    }
    if (kb == 512) cudaLaunchKernelEx(&lc, nu8::nu8<512>, NU8_ARGS);
    else if (kb == 1512) cudaLaunchKernelEx(&lc, nu8::nu8<512, true>, NU8_ARGS);
    else if (kb == 2512) cudaLaunchKernelEx(&lc, nu8::nu8<512, false, true>, NU8_ARGS);
    else TORCH_CHECK(false, "nu8: kb");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

# ---- cuBLASLt with explicit algo configs (lt_src.cu = r-proj dev/lt.cu @f7b1c8b) at T >= 16 ----
# (layer, tokens): cfg (algo, tile, stages, split k, reduction, swizzle, custom option, cluster), proj's prepare-time
# picks on the judge's H200 (r-proj profiles/submission.22109e72563f4cc3.log), us vs F.linear: pooler m16 22.0/24.3,
# m32 22.2/24.6, m128 26.0/29.7, m256 28.2/34.4; qkv m128 18.9/20.0, m256 21.0/22.9. Not pooler m256: beside the
# router (side stream) in the layer it is 60.7 vs F.linear 49.2 us (H100, verify 32 sliding).
# Judge check f8e06892 (pooler m16/m32 too): d16 +0.7%, d32 -2.6% vs v15 (noise ~2%): m16/m32 not used.
LT_CFG = {("pooler", 128): (66, 143, 35, 1, 0, 0, 0, 2),
          ("qkv_gate", 128): (66, 92, 35, 1, 0, 0, 0, 5), ("qkv_gate", 256): (66, 124, 35, 1, 0, 0, 0, 2),
          # s23: proj's H200 tuner winners (r-proj judge logs, median of 30: us F.linear -> lt, standalone incl. launch):
          # qkv m16 17.48 -> 16.83, m32 17.82 -> 17.25, o m16 10.04 -> 9.77, m32 10.30 -> 9.98, m128 10.94 -> 10.73,
          # m256 12.80 -> 12.54, unpooler m128 23.21 -> 22.96. Used only where TUNE picks them (LT_DEFAULT_OFF).
          ("qkv_gate", 16): (66, 394, 35, 3, 2, 0, 5, 4), ("qkv_gate", 32): (66, 16, 35, 3, 2, 0, 7, 4),
          ("o", 16): (66, 394, 35, 1, 0, 0, 7, 4), ("o", 32): (66, 16, 35, 1, 0, 0, 5, 4),
          ("o", 128): (66, 17, 35, 1, 0, 0, 4, 2), ("o", 256): (66, 20, 35, 1, 0, 0, 5, 2),
          ("unpool", 128): (66, 18, 35, 1, 0, 0, 6, 5),
          # pooler/unpool at T=16/32 instead of g19 (proj diag19, H200 after a 3 us PDL predecessor: pooler m32 lt 20.3 vs
          # g19 21.6, unpool m32 19.3 vs 20.3; m16 g19 wins standalone)
          ("pooler", 16): (66, 312, 35, 1, 0, 0, 1, 4), ("pooler", 32): (66, 13, 35, 1, 0, 0, 1, 4),
          ("unpool", 16): (66, 394, 35, 1, 0, 0, 7, 4), ("unpool", 32): (66, 397, 35, 1, 0, 0, 5, 4),
          # s24: proj s16 H200 tuner (Handoffs 10:10Z, standalone, tuner us lt vs F.linear): pooler m256 28.6-28.9 vs
          # 32.9-36.0 (other algo than the s5 m256 pick that lost beside the router), unpooler m256 -0.2
          ("pooler", 256): (66, 18, 35, 1, 0, 0, 0, 2), ("unpool", 256): (66, 20, 35, 1, 0, 0, 6, 5)}
LT_DEFAULT_OFF = {("qkv_gate", 16), ("qkv_gate", 32), ("o", 16), ("o", 32), ("o", 128), ("o", 256), ("unpool", 128),
                  ("pooler", 16), ("pooler", 32), ("unpool", 16), ("unpool", 32), ("pooler", 256), ("unpool", 256)}
# s24: alternative cuBLASLt configs, TUNE option "ltalt:<key>" = 1-based index (implies lt): proj s16 H200 qkv m32
# 17.2-17.3 vs F.linear 17.7-18.1 tuner us.
LT_ALT = {("qkv_gate", 32): [(66, 27, 35, 3, 2, 0, 2, 4)]}
_LT_SRC = r'''// cuBLASLt BF16 GEMM with an explicit algo config: out[m,n] = x[m,k] @ W[n,k]^T, FP32 compute and accumulation.
// Same library and kernel family SGLang's F.linear reaches (algo 66 on sm90), but with the tile / split-K / cluster
// config picked per shape in prepare() (timed on the judge's own GPU) instead of cuBLAS's heuristic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

static cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t h = nullptr;
  if (!h) cublasLtCreate(&h);
  return h;
}

struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
};

// cfg = (algo id, tile, stages, split k, reduction scheme, swizzle, custom option, cluster shape).
// Returns 0 if the config is valid for the shape and fits the workspace.
static int lt_make(LtPlan& p, int64_t m, int64_t n, int64_t k, const std::vector<int64_t>& cfg, int64_t wsbytes) {
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, k, n, k);  // W, column-major [k, n]
  cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);  // x, column-major [k, m]
  cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n);  // out, column-major [n, m]
  if (cublasLtMatmulAlgoInit(lt_handle(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                             CUDA_R_16BF, (int)cfg[0], &p.algo) != CUBLAS_STATUS_SUCCESS) return 1;
  uint32_t tile = cfg[1], stages = cfg[2], red = cfg[4], swz = cfg[5], cust = cfg[6];
  int32_t sk = cfg[3];
  uint16_t cl = cfg[7];
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4);
  if (sk > 1) cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, 4);
  cublasLtMatmulAlgoConfigSetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cl, 2);
  cublasLtMatmulHeuristicResult_t r;
  if (cublasLtMatmulAlgoCheck(lt_handle(), p.op, p.a, p.b, p.c, p.c, &p.algo, &r) != CUBLAS_STATUS_SUCCESS) return 2;
  if (r.workspaceSize > (size_t)wsbytes) return 3;
  return 0;
}

static void lt_free(LtPlan& p) {
  cublasLtMatrixLayoutDestroy(p.a);
  cublasLtMatrixLayoutDestroy(p.b);
  cublasLtMatrixLayoutDestroy(p.c);
  cublasLtMatmulDescDestroy(p.op);
}

// Returns the cuBLAS status (0 = launched).
int64_t lt_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, std::vector<int64_t> cfg) {
  int64_t m = x.size(0), k = x.size(1), n = out.size(1);
  LtPlan p;
  int64_t st = lt_make(p, m, n, k, cfg, ws.numel());
  if (st == 0) {
    float one = 1.f, zero = 0.f;
    st = 100 + (int64_t)cublasLtMatmul(lt_handle(), p.op, &one, w.data_ptr(), p.a, x.data_ptr(), p.b, &zero,
                                       out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo, ws.data_ptr(), ws.numel(),
                                       at::cuda::getCurrentCUDAStream());
    if (st == 100) st = 0;
  }
  lt_free(p);
  return st;
}
'''
_LT = {}


def _lt_build():
    if not LT_CFG or _LT_SRC is None:
        return
    if "lib" not in _LT:
        try:
            from torch.utils.cpp_extension import load_inline
            _LT["lib"] = load_inline(
                name="muse_layer_lt1",
                cpp_sources="int64_t lt_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int64_t>);",
                cuda_sources=_LT_SRC, functions=["lt_gemm"], extra_ldflags=["-lcublasLt"], extra_cuda_cflags=["-O3"],
                verbose=False)
        except Exception as e:  # noqa: BLE001
            print("layer cuBLASLt build failed, F.linear instead:", e)
            _LT["lib"] = None


def _lt_prepare(state):
    """Build the cuBLASLt wrapper; keep the configs that launch on this GPU and match F.linear."""
    state["lt_ok"] = {}
    if not LT_CFG or _LT_SRC is None:
        return
    _lt_build()
    lib = _LT["lib"]
    if lib is None:
        return
    w = {**state["dense"], "unpool": state["unpool"]}
    dev = w["pooler"].device
    state["lt_ws"] = torch.empty(4 << 20, dtype=torch.uint8, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    for (key, m), cfg in LT_CFG.items():
        x = torch.randn((m, w[key].shape[1]), generator=g, device=dev).to(torch.bfloat16)
        out = torch.empty((m, w[key].shape[0]), dtype=torch.bfloat16, device=dev)
        st = lib.lt_gemm(w[key], x, out, state["lt_ws"], list(cfg))
        torch.cuda.synchronize()
        ref = F.linear(x, w[key]).float()
        if st == 0 and ((out.float() - ref).norm() / ref.norm()).item() < 1e-2:
            state["lt_ok"][(key, m)] = cfg
        else:
            print("layer cuBLASLt config dropped:", key, m, cfg, st)
    state["lt_alt_ok"] = {}
    for (key, m), cfgs in LT_ALT.items():
        for i, cfg in enumerate(cfgs, 1):
            x = torch.randn((m, w[key].shape[1]), generator=g, device=dev).to(torch.bfloat16)
            out = torch.empty((m, w[key].shape[0]), dtype=torch.bfloat16, device=dev)
            st = lib.lt_gemm(w[key], x, out, state["lt_ws"], list(cfg))
            torch.cuda.synchronize()
            ref = F.linear(x, w[key]).float()
            if st == 0 and ((out.float() - ref).norm() / ref.norm()).item() < 1e-2:
                state["lt_alt_ok"][(key, m, i)] = cfg
            else:
                print("layer cuBLASLt alt config dropped:", key, m, cfg, st)


# s20: proj's gemv22 (g22_src.cu = r-proj dev/gemv22.cu @a17318c): m = 128/256 swap-AB wgmma GEMM over the 12-bit
# weights, tile-major stream-K, escapes packed per stage; FP32 out (zero kernel + red.add; consumers round to BF16).
# (key, T) -> (m, ap, ns, E). H100 standalone (cold L2): qkv m128 21.9 vs cuBLAS 21.1, pooler 30.9/27.5, unpool
# 28.8/27.3, o 14.9/10.2; m256 all +10 us (compute-bound there: not built). Used only where TUNE picks it (H200).
G22_CFG = {}  # s22: off (proj H200 check fb2fe4ee: g22 loses every m128/m256 shape); was {("qkv_gate", 128): (128, 4, 3, 32), ("pooler", 128): ..., ("unpool", 128): ...}
_G22_SRC = r'''// gemv22: GEMM for m = 128 / 256 tokens (the layer's T=128/256: verify 16 / 32) over lossless 12-bit BF16 weights
// (gemv19's packed atoms, 4-bit exponent codes + per-row escape headers), FP32 output [m, n].
// Swap-AB wgmma: W rows = the wgmma M (64 per warpgroup, A decoded into registers), tokens = N (128 or 256, B = x by a
// SW128 tensor map). Tile-major stream-K: stage u = (128-row tile, 64-k step), each block streams a contiguous range of
// stages; W (12 KB) and the stage's x box (M x 64 k) share one ring slot. Consumers (8 warps = 2 warpgroups) flush their
// partials at run ends (tile change or range end) by red.global.add into out, which a zero kernel cleared first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#ifdef G22_NOFIX
#define G22_NOFIXV 1
#else
#define G22_NOFIXV 0
#endif
namespace g22 {
constexpr int ESC = 8;
#ifdef G22_TRACE
__device__ unsigned long long trace[132 * 64];
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#define TR(slot) trace[blockIdx.x * 64 + (slot)] = gt()
#else
#define TR(slot)
#endif
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    return (s & 0x07800780u) | 0x38003800u;
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = (prmt(s0, 0x9180u) & 0x807f807fu) | expo<0>(nw);
    r.y = (prmt(s0, 0xB3A2u) & 0x807f807fu) | expo<1>(nw);
    r.z = (prmt(s1, 0x9180u) & 0x807f807fu) | expo<2>(nw);
    r.w = (prmt(s1, 0xB3A2u) & 0x807f807fu) | expo<3>(nw);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW22_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W22_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
#ifdef G22_NOWG
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
#else
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
#endif
__device__ __forceinline__ void fence_r(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// K-major SW128 B descriptor (8 token rows x 128 B core groups, 1024 B apart)
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int M> __device__ __forceinline__ void wgmma(float* d, const unsigned* a, unsigned long long desc);
template <> __device__ __forceinline__ void wgmma<128>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<256>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

// one ring stage of a consumer warp: AP atoms (32 k) of its 16-row group decoded into af (one A fragment per k16),
// then 2 * AP wgmma m64nMk16 of its warpgroup over the slot's x box(es) ([M tokens][64 k] SW128 per 64 k)
template <int M, int AP>
__device__ __forceinline__ void stage(float* acc, unsigned (&af)[2 * AP][4], unsigned sb, unsigned nb, unsigned xbuf) {
#pragma unroll
    for (int j = 0; j < AP; ++j) {
        uint4 sv;
        uint2 nv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
        const uint4 a0 = dec8(sv.x, sv.y, nv.x), a1 = dec8(sv.z, sv.w, nv.y);
        af[2 * j][0] = a0.x; af[2 * j][1] = a0.y; af[2 * j][2] = a0.z; af[2 * j][3] = a0.w;
        af[2 * j + 1][0] = a1.x; af[2 * j + 1][1] = a1.y; af[2 * j + 1][2] = a1.z; af[2 * j + 1][3] = a1.w;
    }
    wg_fence();
#pragma unroll
    for (int j = 0; j < 2 * AP; ++j) {
        const int kk = j * 16;
#ifdef G22_NOMMA
        acc[j & 7] += __uint_as_float(af[j][0] ^ af[j][1] ^ af[j][2] ^ af[j][3] ^ (unsigned)kk ^ xbuf);
#else
        wgmma<M>(acc, af[j], sw128_desc(xbuf + (kk >> 6) * M * 128 + (kk & 63) * 2));
#endif
    }
    wg_commit();
}
// M tokens (128/256), AP atoms (32 k) per warp per stage (AP * 32 = k per stage, a multiple of 64), NS ring stages.
// WP: packed stages, tile-major (u = tile * S + ks, S = k / (32 AP)). out FP32 [M, n], zeroed before (PDL primary).
// packed stage (PB bytes): W atoms (SB) | E escapes (row in tile | k in stage << 8, true / scale; row -1: none)
// smem (1024-aligned): ring [NS][stage PB | pad to XO | XS x] | barriers | flush scratch
template <int M, int AP, int NS, int E>
__global__ void __launch_bounds__(384, 1) gemm(const unsigned char* __restrict__ WP,                                                float scale, const __grid_constant__ CUtensorMap xmap,
                                               const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int n, int k, int npre) {
    constexpr int NCW = 8, NT = M / 8, SB = NCW * AP * 768, XS = AP / 2 * M * 128, PB = SB + 8 * E, XO = SB + 1024, SL = XO + XS;
    static_assert(AP % 2 == 0 && SB % 1024 == 0 && E % 32 == 0 && E <= 128, "config");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned bars = ring + NS * SL;
    const int t = threadIdx.x, lane = t & 31, w = __shfl_sync(0xffffffffu, t >> 5, 0), g = lane >> 2, c = lane & 3;
    const int S = k / (32 * AP), ntr = n / (16 * NCW);
    // tile-aligned split: tile t gets nb = G / ntr (+1 for the first G % ntr tiles) blocks: one run per block
    int ra0, rb0;
    {
        const int G = gridDim.x, b = blockIdx.x, base = G / ntr, ex = G % ntr;
        int tt, j, nb;
        if (b < ex * (base + 1)) { tt = b / (base + 1); j = b % (base + 1); nb = base + 1; }
        else { const int b2 = b - ex * (base + 1); tt = ex + b2 / base; j = b2 % base; nb = base; }
        ra0 = tt * S + S * j / nb; rb0 = tt * S + S * (j + 1) / nb;
    }
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), NCW + (G22_NOFIXV ? 0 : 1)); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) TR(0);
    if (w >= NCW) {  // producer warpgroup: one lane streams the block's stages
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (w == NCW && lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
            int i = 0;
            // W of the first stages may stream before the previous kernel ends (x must wait)
            for (; i < npre && i < NS && ra0 + i < rb0; ++i) {
                bar_expect(bars + 8 * i, PB + XS);
                bulk(ring + i * SL, WP + (long)(ra0 + i) * PB, PB, bars + 8 * i);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            TR(1);
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS, ks = u % S;
                if (j >= i) {
                    if (j >= NS) bar_wait(bars + 8 * (NS + slot), ((j / NS) - 1) & 1);
#ifdef G22_NOX
                    bar_expect(bars + 8 * slot, u == ra0 ? PB + XS : PB);
#else
                    bar_expect(bars + 8 * slot, PB + XS);
#endif
                    bulk(ring + slot * SL, WP + (long)u * PB, PB, bars + 8 * slot);
                }
#ifdef G22_NOX
                if (u == ra0) {
#else
                {
#endif
#pragma unroll
                for (int b = 0; b < AP / 2; ++b) tma2d(ring + slot * SL + XO + b * M * 128, &xmap, ks * AP * 32 + b * 64, 0, bars + 8 * slot);
                }
            }
        } else if (w == NCW + 1 && !G22_NOFIXV) {  // fixup warp: escapes, x from the ring slot, red.add into out
            // escapes of stage u: EV[EST[u] .. EST[u + 1]) = (row | (k % 32AP) << 16, true / scale); the block's are contiguous
            asm volatile("griddepcontrol.wait;" ::: "memory");
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS;
                bar_wait(bars + 8 * slot, (j / NS) & 1);
                const unsigned xb = ring + slot * SL + XO;
                const int row0 = (u / S) * 128;
#pragma unroll 1
                for (int e0 = 0; e0 < E; e0 += 32) {
                    int2 ev;
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(ev.x), "=r"(ev.y) : "r"(ring + slot * SL + SB + (e0 + lane) * 8));
                    const unsigned full = __ballot_sync(0xffffffffu, ev.x >= 0);
                    unsigned bal = full;
                    while (bal) {
                        const int src = __ffs(bal) - 1;
                        bal &= bal - 1;
                        const int ex = __shfl_sync(0xffffffffu, ev.x, src), row = row0 + (ex & 0xFF), kk = ex >> 8;
                        const float hy = __int_as_float(__shfl_sync(0xffffffffu, ev.y, src));
#pragma unroll 4
                        for (int tok = lane; tok < M; tok += 32) {
                            const unsigned ad = xb + (kk >> 6) * M * 128 + tok * 128 + ((((kk & 63) >> 3) ^ (tok & 7)) << 4) + (kk & 7) * 2;
                            unsigned short xr;
                            asm volatile("ld.shared.u16 %0, [%1];" : "=h"(xr) : "r"(ad));
                            const float xv = __uint_as_float((unsigned)xr << 16);
                            asm volatile("red.global.add.f32 [%0], %1;" :: "l"(out + (long)tok * n + row), "f"(fmaf(xv, hy, -xv * 0x1p-15f) * scale));
                        }
                    }
                    if (full != 0xffffffffu) break;
                }
                __syncwarp();
                if (lane == 0) bar_arrive(bars + 8 * (NS + slot));
            }
            if (lane == 0) TR(7);
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT * 4];  // wgmma m64nM accumulator: per n8 tile jj, [4] = mma.sync C layout (rows g, g + 8; tokens 2c, 2c + 1)
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
    unsigned afA[2 * AP][4], afB[2 * AP][4];
    int i = 0, pslot = -1, nrun = 0;
    // one stage with A set af (ao: the previous stage's set, maybe in flight)
#define G22_STEP(af, ao) { \
        const int slot = i % NS; \
        bar_wait(bars + 8 * slot, (i / NS) & 1); \
        stage<M, AP>(acc, af, ring + slot * SL + w * AP * 512 + lane * 16, ring + slot * SL + NCW * AP * 512 + w * AP * 256 + lane * 8, ring + slot * SL + XO); \
        wg_wait<1>(); \
        _Pragma("unroll") for (int j = 0; j < 2 * AP; ++j) _Pragma("unroll") for (int e = 0; e < 4; ++e) fence_r(ao[j][e]); \
        __syncwarp(); \
        if (lane == 0 && pslot >= 0) bar_arrive(bars + 8 * (NS + pslot)); \
        pslot = slot; ++i; }
#pragma unroll 1
    for (int ra = ra0; ra < rb0;) {  // runs: the block's range split at tile boundaries
        const int tr = ra / S, rb = min(rb0, (tr + 1) * S), nst = rb - ra;
        int s = 0;
        if (t == 0 && nrun == 0) { bar_wait(bars, 0); TR(2); }
#pragma unroll 1
        for (; s + 1 < nst; s += 2) {
            G22_STEP(afA, afB)
            G22_STEP(afB, afA)
        }
        if (s < nst) G22_STEP(afA, afB)
        wg_wait<0>();
        if (t == 0) TR(3 + 2 * nrun);
#pragma unroll
        for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { fence_r(afA[j][e]); fence_r(afB[j][e]); }
#pragma unroll
        for (int e = 0; e < NT * 4; ++e) fence_f(acc[e]);
        __syncwarp();
        if (lane == 0) bar_arrive(bars + 8 * (NS + pslot));
        pslot = -1;
        // flush: 32-token groups through smem ([tok][16 rows], stride 20 floats), red.add.v4 along rows
        {
            float* sc = (float*)(smem + NS * SL + 16 * NS + 64) + w * 32 * 20;
            float* ob = out + (tr * NCW + w) * 16;
#pragma unroll
            for (int t0 = 0; t0 < NT; t0 += 4) {
#pragma unroll
                for (int jj = t0; jj < t0 + 4; ++jj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) sc[((jj - t0) * 8 + 2 * c + (e & 1)) * 20 + g + 8 * (e >> 1)] = acc[jj * 4 + e] * scale;
                __syncwarp();
#pragma unroll
                for (int i2 = 0; i2 < 4; ++i2) {
                    const int f = lane + 32 * i2, tl = f >> 2, q4 = f & 3;
                    const float4 v = *(const float4*)(sc + tl * 20 + 4 * q4);
#ifdef G22_NORED
                    if (v.x == 1.2345f) ob[0] = v.y;
#else
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ob + (long)(t0 * 8 + tl) * n + 4 * q4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#endif
                }
                __syncwarp();
            }
#pragma unroll
            for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
        }
        if (t == 0) TR(4 + 2 * nrun);
        ++nrun;
        ra = rb;
    }
    if (t == 0) TR(9);
#undef G22_STEP
}
// PDL primary: lets the gemm launch at once and zeroes its output (the gemm waits before touching it)
__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace g22
#ifdef G22_TRACE
torch::Tensor g22_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g22::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
static PFN_cuTensorMapEncodeTiled_v12000 g22_encode = nullptr;
static void g22_xmap(CUtensorMap& map, void* ptr, int m, int k) {
    if (!g22_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g22_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g22_encode, "g22: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)m};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g22_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g22: cuTensorMapEncodeTiled failed ", (int)r);
}
// mode 0: zero kernel then the gemm (PDL dependent). mode 1: out already zero, the gemm a PDL dependent of the
// previous kernel (npre W stages stream before griddepcontrol.wait).
void g22_gemm(torch::Tensor wp, double scale, torch::Tensor x, torch::Tensor out,
              int64_t ap, int64_t ns, int64_t e, int64_t grid, int64_t mode, int64_t npre) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = 1024 + ns * (8 * ap * 768 + 1024 + ap / 2 * m * 128) + 16 * ns + 64 + 8 * 32 * 20 * 4;
    CUtensorMap xmap;
    g22_xmap(xmap, x.data_ptr(), m, k);
    TORCH_CHECK(n % 128 == 0 && k % (32 * ap) == 0 && sm <= 227 * 1024 && n / 128 <= grid && grid <= (n / 128) * (k / (32 * ap)), "g22: shape/smem/grid ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g22::zero<<<132, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4);
#define G22_CASE(M_, AP_, NS_, E_) \
    if (m == M_ && ap == AP_ && ns == NS_ && e == E_) { \
        auto kern = g22::gemm<M_, AP_, NS_, E_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (float)scale, xmap, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), n, k, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G22_CONFIGS
    TORCH_CHECK(false, "g22: no config");
}
'''
_G22 = {}


def _pack22(c, ap, E):
    """proj's _pack22: tile-major gemv22 stages + E escapes per stage (int32 row | k << 8, true / scale bits); None
    if a stage has more than E escapes."""
    code, sm, hdr, scale = c
    n, k = code.shape
    kc = 32 * ap
    P = _pack19(c, kc, ap, tm=1, dec3=False)
    ks, ntr = k // kc, n // 128
    EV = torch.full((ks * ntr, E, 2), -1, dtype=torch.int32, device=code.device)
    EV[:, :, 1] = 0
    rows, slots = (hdr[:, :, 1] != 2.0 ** -15).nonzero(as_tuple=True)
    if len(rows):
        col = hdr[rows, slots, 0].view(torch.int32).long()
        u = (rows // 128) * ks + col // kc
        o = torch.argsort(u * (1 << 20) + rows * 8 + slots)
        rows, col, u, hy = rows[o], col[o], u[o], hdr[rows[o], slots[o], 1]
        rank = torch.arange(len(u), device=code.device) - torch.searchsorted(u, u)
        if int(rank.max()) >= E:
            return None
        EV[u, rank, 0] = ((rows % 128) | ((col % kc) << 8)).int()
        EV[u, rank, 1] = hy.view(torch.int32)
    return torch.cat([P, EV.view(torch.uint8).view(ks * ntr, -1)], 1).contiguous()


def _g22_build():
    if not G22_CFG or _G22_SRC is None:
        return
    if "lib" not in _G22:
        try:
            import hashlib
            from torch.utils.cpp_extension import load_inline
            src = _G22_SRC.replace("G22_CONFIGS", " ".join("G22_CASE(%d,%d,%d,%d)" % c for c in sorted(set(G22_CFG.values()))))
            _G22["lib"] = load_inline(
                name="layer_g22_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                cpp_sources="void g22_gemm(torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources=src, functions=["g22_gemm"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
        except Exception as e:  # noqa: BLE001
            print("layer gemv22 build failed:", e)
            _G22["lib"] = None


def _g22_prepare(state):
    """gemv22 packs per (key, T), each checked once against F.linear (dropped -> the default path)."""
    state["g22"] = {}
    if not G22_CFG or _G22_SRC is None:
        return
    _g22_build()
    lib = _G22["lib"]
    if lib is None:
        return
    w = {**state["dense"], "unpool": state["unpool"]}
    dev = state["unpool"].device
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    comp = {}
    for (key, m), c in G22_CFG.items():
        if key not in comp:
            comp[key] = _compress(w[key])
        if comp[key] is None:
            continue
        pk = _pack22(comp[key], c[1], c[3])
        if pk is None:
            print("layer gemv22 escapes overflow:", key, m, c)
            continue
        x = torch.randn((m, w[key].shape[1]), generator=g, device=dev).to(torch.bfloat16)
        out = torch.full((m, w[key].shape[0]), float("nan"), dtype=torch.float32, device=dev)
        try:
            lib.g22_gemm(pk, comp[key][3], x, out, c[1], c[2], c[3], _G19_SMS, 0, c[2])
            torch.cuda.synchronize()
            ref = F.linear(x, w[key]).float()
            ok = ((out.bfloat16().float() - ref).norm() / ref.norm()).item() < 1e-2
        except Exception as ex:  # noqa: BLE001
            print("layer gemv22 failed:", key, m, ex)
            ok = False
        if ok:
            state["g22"][(key, m)] = (pk, comp[key][3], w[key].shape[0], c)
        else:
            print("layer gemv22 config dropped:", key, m, c)
    del comp


def _pack19(c, kc, ap, tm=0, dec3=True):
    """proj's pack19 (r-proj dev/g19_lib.py): _compress output -> stage-major uint8 [stages, 8 * ap * 768]."""
    code, sm, hdr, scale = c
    n, k = code.shape
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()  # noqa: E731
    if dec3:  # s43: proj s35 DEC3 layout (g19); gemv22 keeps the old one
        S, N = _dec3(perm(sm).reshape(k // kc, n // 16, kc // 32, 32, 2, 8), perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8))
        S, N = S.reshape(k // kc, n // 16, kc // 32, 32, 16), N.contiguous()
    else:
        S = perm(sm).to(torch.uint8).reshape(k // kc, n // 16, kc // 32, 32, 16)
        sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=code.device)
        N = (perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8).long() << sh).sum(-1)
        N = torch.where(N >= 2 ** 31, N - 2 ** 32, N).int().contiguous()
    ks, ntr, spk = k // kc, n // 128, kc // 32 // ap
    pm = (1, 0, 3, 2, 4) if tm else (0, 1, 3, 2, 4)  # tm: tile-major stages (tile, ks, st) for gemv22
    S = S.view(ks, ntr, 8, spk, ap * 512).permute(*pm).reshape(ks * ntr * spk, 8 * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, 8, spk, ap * 256).permute(*pm).reshape(ks * ntr * spk, 8 * ap * 256)
    return torch.cat([S, N], 1).contiguous()


def _g19_prepare(state):
    """gemv19 packs per (key, T) (shared when kc/ap match) + one epoch counter per call site; each pick is checked
    once against F.linear (dropped -> cuBLAS)."""
    state["g19"] = {}
    lib = _G10.get("lib")
    if lib is None or not G19_CFG or _G19_SRC is None:
        return
    w = {**state["dense"], "unpool": state["unpool"]}
    comp, packs = {}, {}
    dev = state["unpool"].device
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    for (key, m), c in G19_CFG.items():
        if key not in comp:
            comp[key] = _compress(w[key])
        if comp[key] is None:
            continue
        pk = packs.get((key, c[1], c[2]))
        if pk is None:
            pk = packs[(key, c[1], c[2])] = _pack19(comp[key], c[1], c[2])
        grid = G19_GRID.get((key, m), _G19_SMS)
        e = (pk, comp[key][2], comp[key][3], {grid: torch.zeros(2, dtype=torch.int32, device=dev)}, c)
        x = torch.randn((m, w[key].shape[1]), generator=g, device=dev).to(torch.bfloat16)
        out = torch.full((m, w[key].shape[0]), float("nan"), dtype=torch.float32, device=dev)
        try:
            lib.g19_gemm(e[0], e[1], e[2], x, out, e[3][grid], c[1], c[2], c[3], 1, grid, 0, 2, c[4], c[5])
            torch.cuda.synchronize()
            ref = F.linear(x, w[key]).float()
            ok = ((out.bfloat16().float() - ref).norm() / ref.norm()).item() < 1e-2
        except Exception as ex:  # noqa: BLE001
            print("layer gemv19 failed:", key, m, ex)
            ok = False
        if ok:
            state["g19"][(key, m)] = e
        else:
            print("layer gemv19 config dropped:", key, m, c)
    _g24_prepare(state, w, comp, g)
    del comp


def _pack24(c, kc, ap):
    """gemv24 stages (pack19 with one 16-row group per stage): stage (ks, row group, st) = [ap atoms x 32 lanes x 16 B
    sign|mantissa][ap x 32 x 8 B codes], DEC3."""
    code, sm, hdr, scale = c
    n, k = code.shape
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()  # noqa: E731
    S, N = _dec3(perm(sm).reshape(k // kc, n // 16, kc // 32, 32, 2, 8), perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8))
    U = (k // kc) * (n // 16) * (kc // 32 // ap)
    S = S.reshape(U, ap * 512)
    N = N.contiguous().view(torch.uint8).reshape(U, ap * 256)
    return torch.cat([S, N], 1).contiguous()


def _g24_build():
    if not G24_CFG or _G24_SRC is None:
        return
    if "lib" not in _G24:
        try:
            import hashlib
            from torch.utils.cpp_extension import load_inline
            _G24["lib"] = load_inline(
                name="layer_g24_" + hashlib.sha256(_G24_SRC.encode()).hexdigest()[:10] + "_" + str(len(G24_CFG) + len(G24_SKCFG) + len(G24_SK2CFG)),
                cpp_sources="void g24_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor);",
                cuda_sources=_G24_SRC.replace("G24_CONFIGS", " ".join("G24_CASE(%d,%d,%d,%d,%d,%d,%d)" % c for c in sorted(set(G24_CFG.values()) | set(G24_SKCFG.values()) | set(G24_SK2CFG.values())))),
                functions=["g24_gemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
        except Exception as ex:  # noqa: BLE001
            print("layer gemv24 build failed:", ex)
            _G24["lib"] = None


def _g24_prepare(state, w, comp, g):
    state["g24"] = {}
    state["g24sk"] = {}
    state["g24sk2"] = {}
    if not G24_CFG or _G24_SRC is None:
        return
    _g24_build()
    lib = _G24["lib"]
    if lib is None:
        return
    dev = state["unpool"].device
    packs = {}
    for sd, (key, m), c in [("g24", k_, v_) for k_, v_ in G24_CFG.items()] + [("g24sk", k_, v_) for k_, v_ in G24_SKCFG.items()] + [("g24sk2", k_, v_) for k_, v_ in G24_SK2CFG.items()]:
        if key not in comp:
            comp[key] = _compress(w[key])
        if comp[key] is None:
            continue
        pk = packs.get((key, c[1], c[2]))
        if pk is None:
            pk = packs[(key, c[1], c[2])] = _pack24(comp[key], c[1], c[2])
        grid = G19_GRID.get((key, m), _G19_SMS)
        e = (pk, comp[key][2], comp[key][3], {grid: torch.zeros(2, dtype=torch.int32, device=dev)}, c)
        x = torch.randn((m, w[key].shape[1]), generator=g, device=dev).to(torch.bfloat16)
        out = torch.full((m, w[key].shape[0]), float("nan"), dtype=torch.float32, device=dev)
        try:
            lib.g24_gemm(e[0], e[1], e[2], x, out, e[3][grid], c[1], c[2], c[3], c[4], c[5], grid, c[3], c[6], _NOZ(dev))
            torch.cuda.synchronize()
            ref = F.linear(x, w[key]).float()
            ok = ((out.bfloat16().float() - ref).norm() / ref.norm()).item() < 1e-2
        except Exception as ex:  # noqa: BLE001
            print("layer gemv24 failed:", key, m, ex)
            ok = False
        if ok:
            state[sd][(key, m)] = e
            if key == "qkv_gate" and sd == "g24":
                state.setdefault("g24x", {})[m] = torch.zeros((m, w[key].shape[0]), dtype=torch.float32, device=dev)
            if key == "pooler" and m in G24P_TOKENS and m not in state.get("g24p", {}):
                state.setdefault("g24p", {})[m] = torch.zeros((m, w[key].shape[0]), dtype=torch.float32, device=dev)
        else:
            print("layer gemv24 config dropped:", key, m, c)
    _g27_prepare(state, w, comp, g, packs)


def _g27_build():
    if not G27_CFG or _G27_SRC is None:
        return
    if "lib" not in _G27:
        try:
            import hashlib
            from torch.utils.cpp_extension import load_inline
            _G27["lib"] = load_inline(
                name="layer_g27_" + hashlib.sha256(_G27_SRC.encode()).hexdigest()[:10] + "_" + str(len(G27_CFG)),
                cpp_sources="void g27_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor);",
                cuda_sources=_G27_SRC.replace("G27_CONFIGS", " ".join("G27_CASE(%d,%d,%d,%d,%d,%s)" % (c[:5] + ("true" if c[5] else "false",)) for c in sorted(set(G27_CFG.values())))),
                functions=["g27_gemm"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
        except Exception as ex:  # noqa: BLE001
            print("layer gemv27 build failed:", ex)
            _G27["lib"] = None


def _g27_prepare(state, w, comp, g, packs):
    state["g27"] = {}
    if not G27_CFG or _G27_SRC is None:
        return
    _g27_build()
    lib = _G27["lib"]
    if lib is None:
        return
    dev = state["unpool"].device
    for (key, m), c in G27_CFG.items():
        if key not in comp:
            comp[key] = _compress(w[key])
        if comp[key] is None:
            continue
        k = w[key].shape[1]
        if k % (c[1] * 32 * c[2]) or (c[3] >= 100 and (c[3] - 100) * 32 * c[1] != k):
            print("layer gemv27 config dropped (shape):", key, m, c)
            continue
        kc = k // c[1]
        pk = packs.get((key, kc, c[2]))
        if pk is None:
            pk = packs[(key, kc, c[2])] = _pack24(comp[key], kc, c[2])
        e = (pk, comp[key][2], comp[key][3], c)
        x = torch.randn((m, k), generator=g, device=dev).to(torch.bfloat16)
        out = torch.full((m, w[key].shape[0]), float("nan"), dtype=torch.bfloat16 if c[5] else torch.float32, device=dev)
        try:
            lib.g27_gemm(e[0], e[1], e[2], x, out, c[1], c[2], c[3], c[4], 99, _NOZ(dev))
            torch.cuda.synchronize()
            ref = F.linear(x, w[key]).float()
            ok = ((out.float() - ref).norm() / ref.norm()).item() < 1e-2
        except Exception as ex:  # noqa: BLE001
            print("layer gemv27 failed:", key, m, ex)
            ok = False
        if ok:
            state["g27"][(key, m)] = e
        else:
            print("layer gemv27 config dropped:", key, m, c)


# (s18/s19) Choices where the H100 and the judge's H200 may disagree, tuned in prepare(): the whole layer is timed
# (CUDA graph replays, cold L2, interleaved, median) on synthetic decode/verify inputs (shapes only, random data) with
# each candidate option set; the fastest set per (T, sliding) is kept in state["tune"][(T, sliding)] (full layers at
# ctx 80000 like the scored ones). Options (defaults = no entry):
# g19:<key> (bool) gemv19 for that GEMM; pr1_npf / pr1_rb (int) pr1's L2-prefetched pooler chunks / router blocks;
# kv_pf (bool) KV_PF; tk2fx / tk2fj (bool) TK2FX / TK2FJ (s27); rf_trig (bool) attention's _red_fin early trigger (s27); qf32 (bool) QF32 (s28); bkt / gnorm (bool) BKT / GNORM (s27); pr1 (bool) pr1/pr8 at that T; twm_order / twm_tail (int) moe_twm item order / tail items (s25); twq (int) moe_twq flags, twm_hot (int) moe_twm hot class (moe s17); twf_split (int) moe_twm K-split tail items (moe s19); rg (bool) RG (s52); tk2h (int) moe n6 tk2h GS16+PF4 at T=16/32, 2-group weight W + 1 (s58); nu8d (bool) NU8_D (s59); pr1bnd (bool) PR1_BND at T=1 (s74); bspl1 / bspl2 (int) BND_SPLIT of the first / second boundary (s74); bal (bool) moe n8 BAL tk2f (s61); fr (bool) fused pooler+router GEMM above 8 tokens (s37, FR_TUNE); lt:<key> (bool) the cuBLASLt pick (else cuBLAS heuristics). A candidate that fails to run is skipped.
# s18: the T=16/32 pooler grid (108..132): the H200 picks 108 too (check ea9421fa), so it is fixed now.
TUNE = {1: ({}, {"pr1_npf": 4}, {"pr1_npf": 8}, {"kv_pf": False}, {"pr1_rb": 20}, {"pr1_rb": 32}, {"g13alt:qkv_gate": 1},
            {"g10alt:o": 1}, {"pr1_pf": 2}, {"pr1_pf": 2, "pr1_rb": 32}, {"tk2fx": False},
            {"rf_trig": False}, {"g13o": False}, {"g13o": False, "pr1_rb": 20}, {"g13o": False, "kv_pf": False}),  # s77: base {g13o, pr1_rb 32}  # s74: pr1bnd / bspl1 / bspl2 options removed (H200 check 4bf626fa: pr1bnd +0.6..1.4 us, bspl noise)
        16: ({}, {"g24z": False}, {"g24x": False}, {"g19:qkv_gate": False}, {"g19:o": True}, {"g19:o": False}, {"g19:qkv_gate": False, "lt:qkv_gate": True}, {"lt:o": True},
             {"g19:qkv_gate": False, "lt:qkv_gate": True, "lt:o": True}, {"g19:pooler": False, "lt:pooler": True},
             {"g19:unpool": False, "lt:unpool": True}, {"tk2fj": False}, {"qf32": False}, {"g19trig": False},
             {"bal": True}, {"bal": False}, {"cslpw": False}, {"cslcarve": True},
             {"g24:pooler": False, "g24:unpool": False, "g24:o": False, "g24:qkv_gate": False}, {"g24:pooler": False, "g24:unpool": False},
             {"g24:o": False}, {"g24:qkv_gate": False},
             {"g24sk:pooler": True, "g24sk:unpool": True}, {"g24sk:qkv_gate": True}, {"g24sk:pooler": True, "g24sk:unpool": True, "g24sk:qkv_gate": True},
             {"g27:o": False},  # s84: attention plan (full layers only)
             ),
        32: ({}, {"g19:qkv_gate": True}, {"g19:o": True}, {"g19:qkv_gate": True, "g19:o": True}, {"lt:qkv_gate": True},
             {"lt:o": True}, {"lt:qkv_gate": True, "lt:o": True}, {"g19:pooler": False, "lt:pooler": True},
             {"g19:unpool": False, "lt:unpool": True}, {"g19:pooler": False, "lt:pooler": True, "g19:unpool": False, "lt:unpool": True},
             {"ltalt:qkv_gate": 1}, {"ltalt:qkv_gate": 1, "lt:o": True}, {"tk2fj": False},
             {"ltalt:qkv_gate": 1, "lt:o": True, "tk2fj": False}, {"qss": False}, {"qss": False, "lt:qkv_gate": True},
             {"o_pf": 0}, {"g19trig": False}, {"tk2h": 3}, {"tk2h": 4}, {"tk2h": 4, "lt:o": True}, {"bal": True}, {"bal": True, "lt:o": True},
             {"g24:pooler": False, "g24:unpool": False}, {"g24:pooler": False}, {"g24:unpool": False}, {"g24:qkv_gate": True}, {"g24:o": True},
             {"g24sk:pooler": False, "g24sk:unpool": False}, {"g24sk:pooler": False}, {"g24sk:unpool": False}, {"g24sk:qkv_gate": True},
             {"g27:o": True}, {"tk2h": 4, "g24:qkv_gate": True}, {"tk2h": 4, "lt:o": True, "g24:qkv_gate": True},
             {"g24sk2:pooler": True}, {"g24sk2:unpool": True}, {"g24sk2:pooler": True, "g24sk2:unpool": True},
             {"g24sk2:pooler": True, "g24sk2:unpool": True, "tk2h": 4, "g24:qkv_gate": True},
             {"g24sk2:pooler": True, "g24sk2:unpool": True, "tk2h": 4, "lt:o": True, "g24:qkv_gate": True}, {"tk2h": 0, "g24:qkv_gate": False},),  # s84: aplan  # s76: old base  # proj s69: g24sk2 = SK (4,3,12)  # s73: g24p options removed (H200 check c755dcc3: +0.6..2 us)
        8: ({}, {"g14alt:qkv_gate": 0}, {"g14alt:qkv_gate": 1}, {"g14alt:qkv_gate": 2}, {"g14alt:qkv_gate": 3},
            {"nu8": False}, {"nu8d": False}, {"nu8d": False, "nu8x": False}, {"nu8": False, "g14alt:qkv_gate": 0}, {"tk2fx": False},
            {"tk2fx": False, "g14alt:qkv_gate": 1}, {"g19:qkv_gate": True},
            {"cslpw": True},
            {"cslpw": True, "g14alt:qkv_gate": 3}, {"cslpw": True, "g14alt:qkv_gate": 2}),  # s87: all {pr1: False} options removed (never picked on 22 H200 logs, +1.7..5.8 us; latent nu8d race path, lead 11:50Z)  # s77: sliding base g14alt 3
        128: ({}, {"lt:pooler": False}, {"lt:qkv_gate": False}, {"lt:o": True}, {"lt:unpool": True}, {"lt:o": True, "lt:unpool": True},
              {"twm_order": 76}, {"bkt": False}, {"gnorm": False}, {"twf_split": 66}, {"twf_split": 1022}, {"o_pf": 0}, {"fr": False},
              {"twf_split": 66, "lt:o": True, "lt:unpool": True}, {"gnorm": False, "lt:unpool": True},),  # s84: aplan  # s77: fr in the base (tuner 10/10 both types)  # s76: every twq option removed at T=128 (moe 07:25Z: twq ~+5 us vs twm in-layer on 16 judge runs; HOTQ check baae0e41 sl +3..9)
              # s75: twq split options 73988/73996 removed (H200 checks a0bcbbf4/2f7349ee: in-layer loses)
        256: ({}, {"nvscl": False, "sl": True}, {"lt:qkv_gate": False}, {"lt:o": True}, {"lt:pooler": True}, {"lt:unpool": True},
              {"lt:pooler": True, "lt:unpool": True, "lt:o": True}, {"twm_order": 76}, {"twm_order": 76, "lt:unpool": True},
              {"twm_order": 4}, {"twm_tail": 32}, {"twq": 4}, {"twq": 12}, {"twm_hot": 1}, {"twm_hot": 1, "twm_order": 76},
              {"twm_hot": 1, "twm_order": 76, "lt:unpool": True}, {"bkt": False}, {"bkt": False, "twm_order": 76, "lt:unpool": True},
              {"gnorm": False}, {"gnorm": False, "twm_order": 76, "lt:unpool": True}, {"twf_split": 66},
              {"twf_split": 66, "twm_order": 76}, {"twf_split": 1022}, {"twf_split": 1022, "twm_order": 76}, {"fr": False},
              {"fr": False, "twm_order": 76}, {"fr": False, "twm_order": 76, "lt:unpool": True}, {"hot": 0}, {"hot": 24}, {"hot": 40},
              {"rg": True}, {"rg": True, "twf_split": 1022}, {"rg": True, "twf_split": 66},
              {"rg": True, "twf_split": 66, "lt:o": True}, {"rg": True, "twf_split": 66, "lt:unpool": True},
              {"rg": True, "twf_split": 1022, "lt:pooler": True, "lt:unpool": True, "lt:o": True},)}  # s84: aplan  # s77 stacks
# s35: per-(T, sliding) defaults under the tuner (the tune pick overrides them; {} = these): the picks both logged H200
# tuners of 909010f (0f065691, 703d3a7e) agreed on, so a noisy judge tuner can't fall back to the slower setting.
# T=8 full {nu8, g14alt 1}: -2.0/-2.1 us both; T=8 sliding nu8 -1.7 / g14alt 1 -1.3; T=16 sliding g19:o -0.4/-0.7.
_G24_ALL = {"g24:pooler": True, "g24:unpool": True, "g24:o": True, "g24:qkv_gate": True}
TUNE_BASE = {(1, True): {"g13o": True, "pr1_rb": 32}, (1, False): {"g13o": True, "pr1_rb": 32},
             (8, True): {"nu8": True, "g14alt:qkv_gate": 3}, (8, False): {"nu8": True, "g14alt:qkv_gate": 1},
             (16, True): {"g19:o": True, "bal": True, "tk2fj": False, **_G24_ALL},
             (128, True): {"fr": True}, (128, False): {"fr": True},
             (16, False): dict(_G24_ALL), (32, True): {"g24:pooler": True, "g24:unpool": True, "g24sk:pooler": True, "g24sk:unpool": True, "tk2h": 4, "g24:qkv_gate": True},
             (32, False): {"g24:pooler": True, "g24:unpool": True, "g24sk:pooler": True, "g24sk:unpool": True, "tk2h": 4, "g24:qkv_gate": True},
             (256, True): {"hot": 32}, (256, False): {"hot": 32}}  # s65: proj s53 gemv24 at T=16 (all 4 GEMMs) / T=32 (pooler+unpool):
# s76: T=32 {tk2h 4, g24:qkv_gate} (proj s66 combo) in the base: both H200 tuners of checks baae0e41 / 1b9a63fb picked it (+-lt:o):
# sl -1.4 / -4.5, full -3.8 / -5.3 us vs the old base. Options that set tk2h / g24:qkv_gate override it.
# proj s54: T=32 pooler+unpool on stream-K g24 (g24sk): H100 ab_flag 5 rounds x 2 seeds d32 sl -1.3/-1.9, full -1.5/-1.1 us;
# g24sk at T=16 +0.4..0.9 and qkv at T=32 +1..1.5 on the H100 (W-bound): TUNE candidates only.
# H100 ab_flag 4 rounds x 3 seeds, TUNE {}: d16 sl/full -2.9/-3.0 us, d32 sl/full -1.5/-1.8 (o on g24 at T=32 sl +2.3; qkv tie/-1)
# s69: T=32 tk2h 4 (W3, now on SQ stages) stays a tuner option: as TUNE_BASE (check 71f451bb) the H200 tuner measured
# {tk2h 0} sl 169.4 vs 172.4, full 348.2 vs 348.3 us in the layer (moe-only said -1.7); {tk2h 3} +8..10.
# tk2h dropped from TUNE[16] (moe n13 H200: every tk2h loses +3..5 us at T=16).
# s61: T=16 sliding moe n8 BAL (H100 real routings 4 seeds -2.6 us mean); s44: moe s29 moe_hot G32 at T=256 (H100 v32 sl -8 us)
# s62: tokens where decode sliding csl is a PDL secondary of the qkv GEMM (TUNE option cslpw). H100 ab_flag 5-6 rounds x
# 3 seeds: d16 sliding -0.7..-1.0 us (csl CTAs still wait for g19's SMs: launch + prologue overlap only); d1 +1.3..1.7 (off);
# T=8 (vtk after g14, TUNE[8] option): v1 sliding -0.3..+0.0 (6 rounds x 3 seeds).
# SAFETY (s63 audit, NOTES s63): the pre-wait loads read only inputs (page_table, seq_lens, cache_loc, k/v scales, K/V cache
# rows at positions < seq_lens) and the config-derived cos/sin table; the only cache writer is csl/vtk itself (the new
# rows, at positions >= seq_lens, taken from smem). This holds ONLY while the qkv GEMM is the graph's first kernel: never
# put a kernel that writes the K/V cache (or triggers early after writing an input) before it. tools/pw_race.py: 0 diffs.
CSL_PW_TOKENS = (16,)
CSL_PW_CARVE = False  # s62 (TUNE cslcarve): the g19 qkv asks for the full smem carveout so 1 csl CTA/SM fits beside it from
# the start (H100: csl starts at 2.8 us, but its gathers slow g19 17.3 -> 20.6 us: d16 sl +0.3..0.6 net; xb 1 qkv: same)
if _moe_TK2F_DIAG:  # s81: DIAG check copies only: the tk2f probes (+ the defaults) at T = 1/16/32, no other tuning
    _TK2X = [{"tk2x": k, "diag": True} for k in (6, 1, 2, 3, 4, 5)]
    TUNE = {1: ({}, *_TK2X), 16: ({}, *[dict(o, bal=False) for o in _TK2X]), 32: ({}, {"tk2h": 0, "diag": True}, *[dict(o, tk2h=0) for o in _TK2X[1:]])}
ATT_DIAG_BUILD = False  # s84: DIAG check copies only (sed to True): full-attention probes (attn_src ATT_DIAG) + defaults
if ATT_DIAG_BUILD:  # gd: 11 = gathers only (no math after tile 0), 12 = all-L2; gw: 1 = no V widen, 2 = all-L2, 3 = no consumer math, 4 = 1+2
    TUNE = {16: ({}, *[{"attx": k, "diag": True} for k in (11, 12)]), 32: ({}, *[{"attx": k, "diag": True} for k in (11, 12)]),
            128: ({}, *[{"attx": k, "diag": True} for k in (2, 3)]), 256: ({}, *[{"attx": k, "diag": True} for k in (1, 2, 3, 4)])}
TUNE_REPS = 40
TUNE_SKEW = True  # s25: skewed synthetic routing (_synth_stream)
TUNE_MARGIN = 0.001  # a non-default pick must beat the defaults by this fraction in a fresh 2-way interleaved run
TUNE_SHAPE = {1: (1, 1), 8: (1, 8), 16: (16, 1), 32: (32, 1), 128: (16, 8), 256: (32, 8)}  # T -> (batch, q_len)


_SLIDING = [True]  # the current kernel_fn call's layer type (set at its start; read by _opt at trace time)


def _opt(state, tokens, name, default):
    key = (tokens, _SLIDING[0])
    t = state.get("tune", {}).get(key, {})
    return t[name] if name in t else TUNE_BASE.get(key, {}).get(name, default)


# s25: skewed synthetic routing for the tuner (expert kernels' timing depends on the per-expert load, e.g. moe_twm's
# item order). Local expert j (0..19) is one of a token's 8 experts with probability SYNTH_P[j] (the scored loads'
# measured mean shares: T=128/256 top local expert ~37% of tokens, ~1.3 local routes per token); the rest of the 8 are
# off-GPU experts. The stream is the sum of the 8 picked experts' unit router rows plus a little noise, so the router
# picks them. Built from the weights only (prepare-time); the outputs are discarded.
SYNTH_P = (.37, .29, .16, .11, .08, .065, .05, .047, .035, .027, .02, .016, .012, .012, .01, .008, .008, .004, .004, .004)


def _synth_stream(router, tokens, g):
    dev = router.device
    p = torch.zeros(N_EXPERTS, device=dev)
    p[:len(SYNTH_P)] = torch.tensor(SYNTH_P, device=dev)
    pick = torch.rand((tokens, N_EXPERTS), generator=g, device=dev) < p  # local picks
    pick &= (torch.cumsum(pick.int(), 1) <= TOP_K)
    off = torch.rand((tokens, N_EXPERTS - LOCAL_EXPERTS), generator=g, device=dev).argsort(1)  # off-GPU fill, random order
    need = TOP_K - pick.sum(1, keepdim=True)
    fill = torch.zeros_like(pick)
    fill[:, LOCAL_EXPERTS:] = off < need
    u = router / router.norm(dim=1, keepdim=True)
    st = (pick | fill).float() @ u * (HIDDEN / TOP_K) ** 0.5
    return st + 0.3 * torch.randn((tokens, HIDDEN), generator=g, device=dev)


def _synth_input(batch, dev, ctx=3000, q_len=1, sliding=True, router=None):
    """A decode step / verify block (page 1) of random data with the scored inputs' shapes and dtypes."""
    g = torch.Generator(device=dev).manual_seed(1234)
    tokens = batch * q_len
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
    maxp = ctx + q_len
    pt = torch.randperm(batch * maxp, generator=g, device=dev).to(torch.int32).view(batch, maxp)
    pos = (seq.long()[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    stream = torch.randn((tokens, HIDDEN), generator=g, device=dev) if router is None else _synth_stream(router, tokens, g)
    kc = (torch.randn((batch * maxp, 1, 2, 64), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    vc = (torch.randn((batch * maxp, 1, 2, 64), generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    return {"stream": stream, "attn_input": (stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + RMS_EPS)).to(torch.bfloat16),
            "k_cache": kc, "v_cache": vc, "positions": pos,
            "cache_loc": torch.gather(pt.long(), 1, pos.view(batch, q_len)).view(-1),
            "cache_seqlens": seq + q_len, "cu_seqlens_q": torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=dev),
            "scheduler_metadata": None, "k_scale": one, "v_scale": one.clone(), "page_table": pt, "seq_lens": seq,
            "local_start": torch.zeros(1, dtype=torch.int32, device=dev),
            "config": {"batch": batch, "q_len": q_len, "tokens": tokens, "hidden": HIDDEN, "num_q_heads": 16,
                       "num_kv_heads": 2, "head_dim": 64, "page_size": 1, "window": 2048 if sliding else 0,
                       "rope_theta": 500000.0 if sliding else 0.0,
                       "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0,
                       "n_experts": N_EXPERTS, "top_k": TOP_K, "local_experts": LOCAL_EXPERTS}}


def _tune(state):
    state["tune"] = {}
    if not TUNE:
        return
    dev = state["unpool_slice"].device
    try:
        flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        for (tokens, cands), sliding in ((tc, sl) for tc in TUNE.items() for sl in (True, False)):
            key = (tokens, sliding)
            data = _synth_input(TUNE_SHAPE[tokens][0], dev, 3000 if sliding else 80000, TUNE_SHAPE[tokens][1], sliding,
                                state["router"] if TUNE_SKEW else None)
            graphs, ok = [], []
            cands = [o for o in cands if not (sliding and o.get("full") or not sliding and o.get("sl"))]  # s84/s85: full-/sliding-only options
            for opts in cands:
                state["tune"][key] = opts
                fn = lambda: kernel_fn(data, state)  # noqa: E731
                try:
                    fn()
                except Exception as ex:  # noqa: BLE001  (host-side checks only: a CUDA fault would raise below)
                    print(f"layer tune {key} {opts} skipped ({type(ex).__name__}: {ex})", flush=True)
                    continue
                torch.cuda.synchronize()
                ok.append(opts)
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream())
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s):
                    fn()
                    with torch.cuda.graph(gr, stream=s):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                graphs.append(gr)
            cands = ok
            idx = list(range(len(cands)))
            ts = [[] for _ in idx]
            for r in range(TUNE_REPS + 3):
                for i in (idx if r % 2 == 0 else idx[::-1]):
                    flush.zero_()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    graphs[i].replay()
                    b.record()
                    b.synchronize()
                    if r >= 3:
                        ts[i].append(a.elapsed_time(b))
            med = [sorted(v)[len(v) // 2] for v in ts]
            best = min((i for i in idx if not cands[i].get("diag")), key=lambda i: med[i])  # s81: DIAG probes never picked
            conf = ""
            if best != 0 and cands[0] == {}:  # s24: confirm against the defaults (winner's curse of a min over many noisy medians)
                c2 = [[], []]
                for r in range(TUNE_REPS):
                    for j, i in enumerate((0, best) if r % 2 == 0 else (best, 0)):
                        flush.zero_()
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        graphs[i].replay()
                        b.record()
                        b.synchronize()
                        c2[i != 0].append(a.elapsed_time(b))
                m0, mb = (sorted(v)[len(v) // 2] for v in c2)
                conf = f" (confirm default {m0 * 1e3:.1f} vs {mb * 1e3:.1f}us)"
                if mb > m0 * (1 - TUNE_MARGIN):
                    best = 0
            state["tune"][key] = cands[best]
            print(f"layer tune T={tokens} {'sliding' if sliding else 'full'}: " + " | ".join(f"{cands[i] or 'default'} {med[i] * 1e3:.1f}us" for i in idx)
                  + f"{conf} -> {cands[best] or 'default'}", flush=True)
            del graphs, data
        del flush
    except Exception as ex:  # keep the defaults
        print(f"layer tune failed ({type(ex).__name__}: {ex}); using defaults", flush=True)
        state["tune"] = {}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _g27_sel(state, key, m):
    """The gemv27 entry _g19 runs for (key, m), or None."""
    if key is None or not _opt(state, m, "g27:" + key, (key, m) in G27_ON):
        return None
    return state.get("g27", {}).get((key, m))


def _g24_sel(state, key, m):
    """The gemv24 entry _g19 runs for (key, m), or None (g22 or g19/cuBLAS take it)."""
    if _g27_sel(state, key, m) is not None:
        return None
    if _opt(state, m, "g22:" + key, False) and state.get("g22", {}).get((key, m)) is not None:
        return None
    e = state.get("g24sk2", {}).get((key, m)) if _opt(state, m, "g24sk2:" + key, False) else None
    if e is None:
        e = state.get("g24sk", {}).get((key, m)) if _opt(state, m, "g24sk:" + key, False) else None
    if e is None:
        e = state.get("g24", {}).get((key, m)) if _opt(state, m, "g24:" + key, False) else None
    return e


def _NOZ(dev):
    z = _NOZ_T.get(dev)
    if z is None:
        z = _NOZ_T[dev] = torch.empty(0, dtype=torch.float32, device=dev)
    return z


_NOZ_T = {}


def _g19(x, key, state, trig=False):  # trig: 1 = trigger dependents at start, 3 = also the full smem carveout
    """x @ W.T (key's weight) in FP32 by gemv19 (a PDL dependent of the previous kernel), or None."""
    m = x.shape[0]
    e = state.get("g22", {}).get((key, m)) if _opt(state, m, "g22:" + key, False) else None
    if e is not None:  # s20: gemv22 (T=128), where TUNE picks it
        c = e[3]
        out = torch.empty((m, e[2]), dtype=torch.float32, device=x.device)
        _G22["lib"].g22_gemm(e[0], e[1], x.contiguous(), out, c[1], c[2], c[3], _G19_SMS, 0, c[2])
        return out
    e = _g27_sel(state, key, m)
    if e is not None:  # proj s57: gemv27 (no zeroed out needed; ZN for the next g24 like g24)
        c = e[3]
        _G24_ZN.pop((key, m), None)
        out = torch.empty((m, e[1].shape[0]), dtype=torch.float32, device=x.device)
        nk = G24_ZCHAIN.get(key)
        en = _g24_sel(state, nk, m) if nk is not None and _opt(state, m, "g24z", m in G24Z_TOKENS) else None
        zo = _NOZ(x.device)
        if en is not None:
            zo = _G24_ZN[(nk, m)] = torch.empty((m, en[1].shape[0]), dtype=torch.float32, device=x.device)
        _G27["lib"].g27_gemm(e[0], e[1], e[2], x.contiguous(), out, c[1], c[2], c[3], c[4], 99 | ((int(trig) & 1) << 16), zo)
        return out
    e = _g24_sel(state, key, m)
    if e is not None:  # proj s53: gemv24, where TUNE picks it
        c = e[4]
        out = _G24_ZN.pop((key, m), None)  # proj s56: zeroed by the previous g24 of this layer call (EZ)
        ez = out is not None and G24_EZUSE
        if out is None:
            out = torch.empty((m, e[1].shape[0]), dtype=torch.float32, device=x.device)
        nk = G24_ZCHAIN.get(key)
        en = _g24_sel(state, nk, m) if nk is not None and _opt(state, m, "g24z", m in G24Z_TOKENS) else None
        zo = _NOZ(x.device)
        if en is not None:  # this g24 zeroes the next one's out after its own work (ZN)
            zo = _G24_ZN[(nk, m)] = torch.empty((m, en[1].shape[0]), dtype=torch.float32, device=x.device)
        elif key == "unpool" and _G24X and _G24X[0].shape[0] == m:  # s59 G24X: zero the next call's qkv out
            zo = _G24X.pop()
        elif key == "unpool" and _G24P and _G24P[0].shape[0] == m:  # s63 G24P: zero the next call's pooler out
            zo = _G24P.pop()
        grid = G19_GRID.get((key, m), _G19_SMS)
        ctr = e[3].get(grid)
        if ctr is None:
            ctr = e[3][grid] = torch.zeros(2, dtype=torch.int32, device=x.device)
        _G24["lib"].g24_gemm(e[0], e[1], e[2], x.contiguous(), out, ctr, c[1], c[2], c[3], c[4], c[5], grid,
                             c[3] | ((int(trig) & 1) << 16) | (int(ez) << 17) | (int(G24_EZPF) << 18), c[6], zo)
        return out
    on = key in G19_ON and _opt(state, m, "g19:" + key, (key, m) not in G19_DEFAULT_OFF)
    e = state.get("g19", {}).get((key, m)) if on else None
    if e is None:
        return None
    c = e[4]
    out = torch.empty((x.shape[0], e[1].shape[0]), dtype=torch.float32, device=x.device)
    grid = G19_GRID.get((key, m), _G19_SMS)
    ctr = e[3].get(grid)  # the epoch barrier needs one counter per grid size
    if ctr is None:
        ctr = e[3][grid] = torch.zeros(2, dtype=torch.int32, device=x.device)
    _G10["lib"].g19_gemm(e[0], e[1], e[2], x.contiguous(), out, ctr, c[1], c[2], c[3], 1, grid, 0, 2, c[4] | (trig << 16), c[5])
    return out


def _g12_prepare(state):
    """gemv12's packed weights, FP32 split-K workspace (overwritten each call) and per-rg counters (left zeroed).
    All gemv12 calls run on the main stream, one at a time, so they share the workspace and counters."""
    if _G10.get("lib") is None:
        return
    dev = state["unpool_slice"].device
    w = {**state["dense"], "unpool_slice": state["unpool_slice"]}
    state["g12"] = {key: _pack12(w[key], 1024 * c[0]) for key, c in G12_CFG.items()}
    state["g12_ws"] = torch.empty(8 * 32 * 8192, dtype=torch.float32, device=dev)
    state["g12_cnt"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    state["ob8_part"] = torch.empty(G12_CFG["o"][3] * 32, dtype=torch.float32, device=dev)  # s86 OB8 per-block partials
    state["ob8_bar"] = torch.zeros(1, dtype=torch.int32, device=dev)  # s86 OB8 grid barrier (epoch counter)
    state["pr1_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)  # pr1's router ticket (self-resetting)
    state["n2u_cnt"] = torch.zeros(1, dtype=torch.int32, device=dev)  # n2u's last-block counter (self-resetting)
    state["g13"] = {key: pk for key in list(G13_CFG) + (["o"] if G13_O else []) if (pk := _pack13(w[key])) is not None}
    state["g14"] = {key: pk for key, c in G14_CFG.items() if (pk := _pack14(w[key], 1024 * c[0])) is not None}
    state["g14k"] = {(key, kc): pk for key, v in G14_ALT.items() if key in state["g14"]  # alt packs (other kcu)
                     for kc in sorted({a[0] for a in v} - {G14_CFG[key][0]}) if (pk := _pack14(w[key], 1024 * kc)) is not None}


def _linear(x, w, key, state=None, sliding=False, kv_pf=None):
    """x @ w.T in BF16: gemv10 at 1 token (FP32 accumulate, no split-K reduce), gemv12 at 8 (FP32 accumulate and
    split-K partials), else cuBLAS."""
    lib = _G10.get("lib")
    if state is not None and x.shape[0] in (1, G12_TOKENS):
        pk = state.get("g13" if x.shape[0] == 1 else "g14", {}).get(key)
        o13 = x.shape[0] == 1 and key == "o" and bool(_opt(state, 1, "g13o", False))
        if pk is not None and (key not in GEMV_PDL_SECONDARY.get((x.shape[0], sliding), ()) or o13) and (key != "o" or o13):
            out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
            if x.shape[0] == 1:
                pf = kv_pf if kv_pf is not None else (x, x, x, 0, 0)
                lib.g13_gemv(pk[0], pk[1], pk[2], x.contiguous(), out, pk[3],
                              *(G13_O if o13 else _cfg(state, 1, "g13", key, G13_CFG)), *pf,
                              int(o13 and UNPOOL_PDL and key in GEMV_PDL_SECONDARY.get((1, sliding), ())))
            else:
                c = _cfg(state, G12_TOKENS, "g14", key, G14_CFG)
                pk = pk if c[0] == G14_CFG[key][0] else state["g14k"][(key, c[0])]
                lib.g14_gemm(pk[0], pk[1], pk[2], pk[3], x.contiguous(), out, state["g12_ws"], state["g12_cnt"],
                             w.shape[0] // 8, *c)
            return out
    if lib is not None and key in G17_KEYS and x.shape[0] in G17_CFG:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        lib.g17_gemv(w, x.contiguous(), out, *G17_CFG[x.shape[0]], int(key in GEMV_PDL_SECONDARY.get((x.shape[0], sliding), ())))
        return out
    if x.shape[0] == 1 and lib is not None and key in G10_CFG and x.shape[1] == G10_CFG[key][0]:
        out = torch.empty((1, w.shape[0]), dtype=torch.bfloat16, device=x.device)
        pdl = ((key in GEMV_PDL_SECONDARY.get((1, sliding), ())) * (1 + (key in G10_PFL2.get(sliding, ()))),) if UNPOOL_PDL else ()
        lib.g10_gemv(w, x.contiguous(), out, *(_cfg(state, 1, "g10", key, G10_CFG) if state is not None else G10_CFG[key])[1:], *pdl)
        return out
    if x.shape[0] == G12_TOKENS and state is not None and "g12" in state and key in G12_CFG:
        out = torch.empty((G12_TOKENS, w.shape[0]), dtype=torch.bfloat16, device=x.device)
        pdl = (int(key in GEMV_PDL_SECONDARY.get((G12_TOKENS, sliding), ())),) if UNPOOL_PDL else ()
        lib.g12_gemm(state["g12"][key], x.contiguous(), out, state["g12_ws"], state["g12_cnt"], w.shape[0] // 8,
                     *G12_CFG[key], *pdl)
        return out
    alt = _opt(state, x.shape[0], "ltalt:" + key, 0) if state is not None else 0
    cfg = state.get("lt_alt_ok", {}).get((key, x.shape[0], alt)) if alt else None
    if cfg is not None:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        _LT["lib"].lt_gemm(w, x.contiguous(), out, state["lt_ws"], list(cfg))
        return out
    cfg = state.get("lt_ok", {}).get((key, x.shape[0])) if state is not None else None
    if cfg is not None and _opt(state, x.shape[0], "lt:" + key, (key, x.shape[0]) not in LT_DEFAULT_OFF):
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        _LT["lib"].lt_gemm(w, x.contiguous(), out, state["lt_ws"], list(cfg))
        return out
    return F.linear(x, w)


TK2F_SQ_T1 = False


def _experts_n2u(x, topk_weights, local_ids, state, bnd=()):
    """T = 1: moe_tk2f (as moe_src's kernel_fn) -> n2u (moe_norm2 + unpool slice): the unpool slice output."""
    st = state["moe"]
    lib = st["lib"]
    dev = x.device
    meta = torch.empty(132, dtype=torch.int32, device=dev)
    sroutes = torch.empty(TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((TOP_K, _moe_I), dtype=torch.bfloat16, device=dev)
    inv = torch.empty(TOP_K, dtype=torch.int32, device=dev)
    y32 = torch.empty((TOP_K, D_EXPERT), dtype=torch.float32, device=dev)
    ids = local_ids.contiguous()
    slot = N2U_SLOT and _moe_TK2F == 204 and N2U_CFG[0] == 8
    c = (84204 if slot else _moe_TK2F) + (100000 if _moe_TK2SA and st["sa"] else 0)
    # s68: moe n14 SQ off here (H100 ab_flag 4 rounds x 3 seeds d1 +0.4..0.9 us; H200 moe diag d1 0)
    c, w13_, w2_ = _moe__tk2f_w(c, st) if TK2F_SQ_T1 else (c, st["w13q"], st["w2q"])
    if _opt(state, 1, "tk2x", 0):  # s81 DIAG probes (tuner timing only, never selected)
        c, w13_, w2_ = _moe__tk2x(_moe__tk2f_w(c, st)[0], st, _opt(state, 1, "tk2x", 0))
    lib.moe_tk2f_launch(w13_, st["w13s2"], w2_, st["w2s2"], st["lut"], st["gs13"], st["gs2"],
                        x.contiguous(), ids, 0, LOCAL_EXPERTS, meta, sroutes, inv, inter, y32, st["cnt"], c)
    f = torch.empty((1, UNPOOL_SLICE), dtype=torch.bfloat16, device=dev)
    _G10["lib"].n2u_launch(y32, inv, ids, topk_weights.float().contiguous(), st["w1"], st["gs2"], 0, LOCAL_EXPERTS,
                           POST_NORM_EPS, state["unpool_slice"], f, st["cnt"], N2U_CFG[0] + (1000 + 2000 * N2U_PFW) * slot, N2U_CFG[1], list(bnd), POST_NORM_EPS, RMS_EPS)
    return f


# s26 NU8 (T = 8): moe_tk2f PF 20 (its gemm2 epilogue also sums each slot's (y w1)^2 into y32 row 64) -> nu8 (moe_norm3 +
# the unpool slice split over k into NU8_KB-wide FP32 slabs, PDL secondary of moe_tk2f) -> the boundary adds the slabs.
# Replaces moe_norm3 + gemv17 (H100 v1 tail tk2f end -> norm3 3.7 -> g17 4.6 -> boundary 3.3 us). TUNE option "nu8".
NU8_DEFAULT = False
# s28 QF32 (proj s19 request): at T=16 decode the g19 qkv stays FP32 into attention (csl sliding, _gd_full + merges full),
# rounded to BF16 (RN) at each load: bit-identical to the torch cast it drops (H200 ea9421fa: 1.2-1.3 us per layer).
QF32 = True
QSS_NS = 2  # proj s22: H200 (proj check ccf41026, tuner us) qkv T=32 bmm 2 slabs 16.03 vs cuBLASLt split-K + reduce 17.50
QSS_VERIFY = True  # s72: QSS also at verify sliding (nvs + _combine_hb load the slabs) when the qss option is on (default off at T=128/256)
# (H100 ab_flag 3 rounds x seeds 7,11: v16 sl -0.9/+0.0, v32 sl +2.0/+1.5 us; nvs's FP32 slab loads +1.6 us vs bmm -1.0)
# H200 tuner (check 0f12113f): T=128 sl {qss: True} 306.4 vs default 306.0, full 368.6 vs 368.3: DEAD, candidate removed
QSS_DEFAULT = {32: True}  # tokens -> default of TUNE option qss (H100 layer, options pinned: d32 sliding -1.5, full -1.5 us)
# s34 O_PF {tokens: fraction}: sliding attention (csl/nds/nvs) L2-prefetches the first fraction of o's weight once each
# CTA's K/V loads are out (HBM idles in the latency-bound attention; o then starts L2-warm). H100 ab_flag: d32 sliding
# 0.5: -1.0/-1.3 us (1.0 -0.2, prefetch at nds START: o -2.2 but nds +3 = zero-sum), v16 sliding 0.5: -1.6/-1.7;
# losers: d16 (csl, o on g19) 0.5 +0.6, 1.0 +1.6; v32 tie. TUNE option o_pf (0 = off).
O_PF = {32: 0.5, 128: 0.5}
VTK_QF32 = True  # s38: the T=8 sliding attention (vtk) takes the g19 m8 qkv in FP32 too (TUNE option g19:qkv_gate at T=8)
NVS_CL = True  # s85: verify-32 sliding on moe s31's 2-CTA cluster nvs (attn_src _NVSCL; H200 standalone -1.1..-1.4 us). TUNE nvscl
QF32_NDS = True  # s29: also at T=32 (nds sliding templated on the qkv type; full = the same Gluon path as T=16)
RF_TRIG_DEFAULT = True  # s27 (attn_src RF_TRIG): H100 ab_files d1 full 113.0 -> 109.9 (8 rounds), 112.3 -> 110.7 (4)
TK2FX_TOKENS = (1, 8)  # s27 TK2FX: moe_tk2f PDL secondary of the pre-expert rmsnorm (pr1 paths only). TUNE option "tk2fx"
TK2FJ_TOKENS = (16, 32)  # s27: the same at T = 16/32 after joining the router's side stream before the rmsnorm
TK2FJ_DEFAULT = True  # H100 ab_files 5 rounds: d16 212.4/313.3 -> 209.9/313.8, d32 205.7/433.8 -> 203.7/432.0 us
TK2FX_DEFAULT = True  # H100 ab_files 6 rounds: d1 90.4/112.8 -> 89.9/111.8, v1 154.0/167.7 -> 152.1/167.8 us
NU8_KB = 512
NU8_X = True  # sums of squares in nu8 (blocks 0..7 + a counter) instead of moe_tk2f PF 20 (H100: its epilogue +2.4 us)


def _nu8_on(state, tokens):
    return (tokens == 8 and _use_moe(tokens) and _moe_TK2F and _G10.get("lib") is not None and tokens <= SLICED_UNPOOL_MAX_TOKENS
            and _moe__tk2f_cfg(tokens, state["moe"]) % 100000 == 84204 and _opt(state, tokens, "nu8", NU8_DEFAULT))


def _nu8_x(state, tokens):
    return _opt(state, tokens, "nu8x", NU8_X)


NU8_D = True  # s59 NU8D: per-slot partials, the post-expert norm scales applied in the boundary (no exchange). TUNE "nu8d"
# H100 ab_flag 6 rounds ABBA x 3 seeds (tuner off) vs X: v1 sliding -0.6..0.9, full -0.4..0.6 us (nu8 -0.7, boundary +0.6)


TK2F_BAL_DEFAULT = False  # moe n8 BAL at T = 8/16/32 (TUNE option "bal")


def _nu8_d(state, tokens):
    return _opt(state, tokens, "nu8d", NU8_D)


def _experts_nu8(x, topk_weights, local_ids, state, y32):
    """T = 8: moe_tk2f (PF 20) -> nu8: the unpool slice as FP32 split-K slabs [D_EXPERT / NU8_KB, 8, 1024]. y32 [65, 4096]:
    row 64 (the slots' sums of squares) was zeroed by the pre-expert _rmsnorm_gain."""
    st = state["moe"]
    _moe_TK2F_BAL_M.discard(8)
    if _opt(state, 8, "bal", TK2F_BAL_DEFAULT):  # moe n8 BAL
        _moe_TK2F_BAL_M.add(8)
    lib = st["lib"]
    dev = x.device
    meta = torch.empty(132, dtype=torch.int32, device=dev)
    sroutes = torch.empty(8 * TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((8 * TOP_K, _moe_I), dtype=torch.bfloat16, device=dev)
    inv = torch.empty(8 * TOP_K, dtype=torch.int32, device=dev)
    ids = local_ids.contiguous()
    tw = topk_weights.float().contiguous()
    d = y32.shape[0] == 81  # NU8D (kernel_fn allocated the per-slot rows)
    lib.moe_set_ss(0 if d else st["w1"].data_ptr())
    lib.moe_tk2f_launch(st["w13q"], st["w13s2"], st["w2q"], st["w2s2"], st["lut"], st["gs13"], st["gs2"],
                        x.contiguous(), ids, 0, LOCAL_EXPERTS, meta, sroutes, inv, inter, y32, st["cnt"],
                        _moe__tk2f_cfg(8, st) + (0 if d or _nu8_x(state, 8) else 1000))
    lib.moe_set_ss(0)
    f = y32[65:81].view(8 * TOP_K, UNPOOL_SLICE) if d else torch.empty((D_EXPERT // NU8_KB, 8, UNPOOL_SLICE), dtype=torch.float32, device=dev)
    _G10["lib"].nu8_launch(y32, ids, tw, st["w1"], 0, LOCAL_EXPERTS, POST_NORM_EPS, state["unpool_slice"], f, st["cnt"],
                           NU8_KB + (2000 if d else 1000 * _nu8_x(state, 8)), 1)
    return f, ((y32[64], ids, tw) if d else None)


_moe_TWM_ORDER_M = {}  # s15: moe's H200 proxy (s16): order 76 at T=256 (r-moe 499bde6 default) loses ~3.6 us at v32


G19_TRIG_TOKENS = (16, 32)  # s57: the g19 unpool triggers at start, the second boundary is its PDL secondary
G19_TRIG_O_TOKENS = (16, 32)  # s57: the same for the g19 o GEMM (T=16 sliding pick) and the first boundary
BND_SEP_TOKENS = (8,)  # s57 SEP: the T=8 second boundary loads nu8's 8 slabs as an [8, 1024] tile (H100 ab_flag v1 -1.0/-1.1 us; T=1 tie)


def _tail(moe, stream1, h, tokens, state, pdl, f=None, sc=None):
    """unpool -> second boundary: (next attention input, stream)."""
    sliced = tokens <= SLICED_UNPOOL_MAX_TOKENS
    if f is None:
        trig = not sliced and _opt(state, tokens, "g19trig", tokens in G19_TRIG_TOKENS)
        f = _linear(moe, state["unpool_slice"], "unpool_slice", state) if sliced else _g19(moe, "unpool", state, trig)
        pdl = pdl or trig and f is not None
        if f is None:
            f = _linear(moe, state["unpool"], "unpool", state)  # all-gather: identity (zeros)
    stream2 = torch.empty_like(stream1)
    hidden = torch.empty_like(h)
    alpha, beta = state["res2"]
    spl = _opt(state, tokens, "bspl2", BND_SPLIT.get((tokens, 2), 1))  # s74: TUNE option
    _boundary[(tokens * spl,)](f, stream1, alpha, beta, state["post_ffn"], state["next_pre_attn"], stream2, hidden, hidden,
                         NB=UNPOOL_SLICE if sliced else HIDDEN, N=HIDDEN, BGAIN=True, H32=False,
                         EPS_POST=POST_NORM_EPS, EPS=RMS_EPS, PDL=pdl, RB=f.dtype == torch.float32, num_warps=16,
                         launch_pdl=pdl, NS=f.shape[0] if f.dim() == 3 else 1, SS=f.shape[1] * UNPOOL_SLICE if f.dim() == 3 else 0,
                         SPL=spl, SEP=sliced and tokens in BND_SEP_TOKENS or sc is not None,
                         **({} if sc is None else dict(sc_ptr=sc[0], ids_ptr=sc[1], tw_ptr=sc[2], SC=True,
                                                     SC_LOCAL=LOCAL_EXPERTS, SC_K=D_EXPERT, SC_EPS=POST_NORM_EPS)))
    return hidden, stream2


def kernel_fn(data, state):
    _G24X.clear()
    _G24P.clear()
    r = _kernel_fn(data, state)
    if _G24X:  # s59 G24X: unpool did not zero qkv's persistent out: restore the zero invariant
        _G24X.pop().zero_()
    if _G24P:  # s63 G24P: same for the pooler's persistent out
        _G24P.pop().zero_()
    return r


def _kernel_fn(data, state):
    cfg = data["config"]
    _G24_ZN.clear()
    _SLIDING[0] = cfg["window"] > 0  # s73: before the G24X/G24P _opt reads (they read the previous call's flag before)
    if cfg["tokens"] in G24X_TOKENS and _opt(state, cfg["tokens"], "g24x", True) and _g24_sel(state, "qkv_gate", cfg["tokens"]) is not None:
        xb = state.get("g24x", {}).get(cfg["tokens"])
        if xb is not None and _g24_sel(state, "qkv_gate", cfg["tokens"])[4][0] == cfg["tokens"]:
            _G24_ZN[("qkv_gate", cfg["tokens"])] = xb  # zero (invariant): the qkv g24 runs EZ on it
            _G24X.append(xb)
    if cfg["tokens"] in G24P_TOKENS and _opt(state, cfg["tokens"], "g24p", False) and _g24_sel(state, "pooler", cfg["tokens"]) is not None:
        pb = state.get("g24p", {}).get(cfg["tokens"])
        if pb is not None and _g24_sel(state, "pooler", cfg["tokens"])[4][0] == cfg["tokens"]:
            _G24_ZN[("pooler", cfg["tokens"])] = pb  # zero (invariant): the pooler g24 runs EZ on it
            _G24P.append(pb)
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    dense = state["dense"]
    dev = data["stream"].device
    kc, vc, pt = data["k_cache"], data["v_cache"], data["page_table"]
    kv_pf = None
    if _opt(state, tokens, "kv_pf", KV_PF) and tokens == 1 and cfg["window"] > 0 and kc.shape[1] == 1 and kc.is_contiguous() and vc.is_contiguous():
        # the window's page-table entries (csl's speculative range: the table's last W + 1 columns)
        n_pf = min(pt.shape[1], cfg["window"] + 1)
        kv_pf = (pt[0, pt.shape[1] - n_pf:], kc, vc, n_pf, kc.stride(0) * kc.element_size())
    global RF_TRIG, ATT_DIAG, ATT_PLAN
    ATT_DIAG = 0 if cfg["window"] > 0 else _opt(state, tokens, "attx", 0)  # s84 DIAG probes (tuner timing only)
    ATT_PLAN = {} if cfg["window"] > 0 else _opt(state, tokens, "aplan", {})  # s84: attention plan picked by the H200 tuner
    RF_TRIG = _opt(state, tokens, "rf_trig", RF_TRIG_DEFAULT)  # s27: attention's _red_fin (decode 1 full) triggers o after its wait
    # s62 CSL_PW: decode sliding csl as a PDL secondary of the g19 qkv (g19 triggers at start; csl's K/V gathers
    # go out before its griddepcontrol.wait, overlapping the GEMM's weight stream)
    csl_pw = bool(cfg["window"] > 0 and length in (1, 8) and _opt(state, tokens, "cslpw", tokens in CSL_PW_TOKENS))
    qkv = _g19(data["attn_input"], "qkv_gate", state,  # s15: T=16 (FP32 out, rounded to the GEMM's BF16)
               trig=csl_pw and (1 + 2 * bool(_opt(state, tokens, "cslcarve", CSL_PW_CARVE))))
    csl_pw = csl_pw and (qkv is not None or tokens in (1, 8))  # T=1/8: the qkv gemv13/14 trigger at start (GEMV_PDL)
    slabs, qss = None, 0
    if (qkv is None and (length == 1 and batch >= NDS_MIN_BATCH and HAVE_GLUON and USE_GDEC and QF32_NDS
                         and (_NDS is not None or cfg["window"] == 0)
                         # s72: verify sliding on nvs (it sums the 2 slabs at its loads, as nds)
                         or length == 8 and batch >= NVS_MIN_BATCH and _NVS is not None and cfg["window"] == 2048
                         and cfg["page_size"] == 1 and QSS_VERIFY)
            and _opt(state, tokens, "qss", QSS_DEFAULT.get(tokens, False))):
        # proj s22 (QSS): qkv split over k into QSS_NS FP32 slabs by one strided batched GEMM (no split-K reduce
        # kernel); attention adds the slabs in FP32 and rounds to BF16 at each load (= cuBLAS split-K + BF16 out).
        x, w = data["attn_input"], dense["qkv_gate"]
        kc = x.shape[1] // QSS_NS
        slabs = torch.bmm(x.view(tokens, QSS_NS, kc).permute(1, 0, 2), w.view(w.shape[0], QSS_NS, kc).permute(1, 2, 0),
                          out_dtype=torch.float32)
        qkv, qss = slabs[0], slabs[0].numel()
    if qkv is None:
        qkv = _linear(data["attn_input"], dense["qkv_gate"], "qkv_gate", state, kv_pf=kv_pf)
    elif not qss and not (length == 1 and HAVE_GLUON and USE_GDEC and _opt(state, tokens, "qf32", QF32)
              and (batch <= CSL_MAX_BATCH and _CSL is not None
                   or batch >= NDS_MIN_BATCH and (_NDS is not None or cfg["window"] == 0) and QF32_NDS)
              or length == 8 and VTK_QF32 and _CSL is not None and cfg["window"] == 2048 and batch <= VTK_MAX_BATCH
              and cfg["page_size"] in (1, 64) and cfg["rope_theta"]):
        qkv = qkv.to(torch.bfloat16)
    o_pf = (0, 0)
    pf_frac = _opt(state, tokens, "o_pf", O_PF.get(tokens, 0)) if cfg["window"] > 0 else 0
    if pf_frac:  # s34: decode sliding attention (csl / nds) L2-prefetches the first pf_frac of o's weight once its K/V are in
        e = state.get("g19", {}).get(("o", tokens)) if "o" in G19_ON and _opt(state, tokens, "g19:o", ("o", tokens) not in G19_DEFAULT_OFF) else None
        e27 = _g27_sel(state, "o", tokens)  # proj s57: gemv27's pack when it runs o
        w = e27[0] if e27 is not None else e[0] if e is not None else dense["o"]
        o_pf = (w.data_ptr(), int(w.numel() * w.element_size() * pf_frac) & ~15)
    o = attn_kernel_fn({
        "o_pf": o_pf, "csl_pw": csl_pw, "nvs_cl": _opt(state, tokens, "nvscl", NVS_CL),
        "qkv": qkv.view(batch, length, -1), "positions": data["positions"], "cache_loc": data["cache_loc"],
        "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
        "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
        "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": data["page_table"],
        "seq_lens": data["seq_lens"], "config": cfg, "qkv_ss": qss, "qkv_slabs": slabs}).view(tokens, hq * d)
    fp32_router = tokens <= ROUTER_FP32_MAX_TOKENS
    pdl = GLUE_PDL or (GEMV_PDL and tokens in PDL_TOKENS and _G10.get("lib") is not None)
    stream1 = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    h32 = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev) if fp32_router else stream1
    alpha, beta = state["res1"]
    npf = (state["router"].numel() * 4) // (512 * 1024) if tokens in ROUTER_L2_PF_TOKENS and fp32_router else 0
    pr1 = (tokens in PR1_TOKENS and _opt(state, tokens, "pr1", True) and pdl and fp32_router and _GPR1_SRC is not None and not npf
           and (tokens == 1 and "pooler" in state.get("g13", {}) or tokens == 8 and _GPR8_SRC is not None
                and "pooler" in state.get("g14", {}) and G14_CFG["pooler"][0] == 8))
    # s86 OB8: the T=8 o GEMM writes the boundary's stream1 / h / h32 itself (pr8 = its PDL secondary)
    ob8 = (pr1 and tokens == 8 and _GOB8_SRC is not None and _opt(state, tokens, "ob8", OB8) and "g12" in state
           and not (("o", 8) in state.get("g19", {}) and "o" in G19_ON and _opt(state, 8, "g19:o", ("o", 8) not in G19_DEFAULT_OFF))
           and _g27_sel(state, "o", 8) is None and o.shape[1] == 1024)
    otrig = _opt(state, tokens, "g19trig", tokens in G19_TRIG_O_TOKENS)
    if ob8:
        otrig, a = False, None
        _G10["lib"].ob8_gemm(state["g12"]["o"], o.contiguous(), data["stream"], alpha, beta, state["pre_ffn"], stream1, h, h32,
                             state["ob8_part"], state["ob8_bar"], HIDDEN // 8, G12_CFG["o"][2], G12_CFG["o"][3],
                             POST_NORM_EPS, RMS_EPS, int(UNPOOL_PDL and "o" in GEMV_PDL_SECONDARY.get((8, cfg["window"] > 0), ()))
                             | 2 * _opt(state, tokens, "ob8late", OB8_LATE))
    else:
        a = _g19(o, "o", state, otrig)
        otrig = otrig and a is not None  # s57: the first boundary is then the o GEMM's PDL secondary
        if a is None:
            a = _linear(o, dense["o"], "o", state, cfg["window"] > 0)  # all-reduce: identity
    bnd = ob8 or pr1 and tokens == 1 and _opt(state, tokens, "pr1bnd", PR1_BND) and a.dtype == torch.bfloat16  # s74: TUNE option
    if not bnd:
      spl = _opt(state, tokens, "bspl1", BND_SPLIT.get((tokens, 1), 1))  # s74: TUNE option
      _boundary[(tokens * spl + npf,)](a, data["stream"], alpha, beta, alpha, state["pre_ffn"], stream1, h, h32,
                               NB=HIDDEN, N=HIDDEN, BGAIN=False, H32=fp32_router, EPS_POST=POST_NORM_EPS, EPS=RMS_EPS,
                               PDL=pdl or otrig, pf_ptr=state["router"], PF=npf, RB=a.dtype == torch.float32,
                               TRIG=pr1 or (BOUNDARY_TRIG and ("pooler", tokens) in state.get("g19", {}) and "pooler" in G19_ON), num_warps=16,
                               launch_pdl=pdl or otrig, SPL=spl)
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    if tokens in ROUTER_AFTER_POOLER and pdl:
        # The pooler GEMV (triggers its dependents at start) first; the router is its PDL secondary on the same
        # stream (reads only h: no griddepcontrol.wait), so it fills SMs beside the pooler instead of delaying it;
        # the rmsnorm waits for the pooler through an event (full edge) on the side stream.
        pooled = _linear(h, dense["pooler"], "pooler", state)
        pooled_done = torch.cuda.Event()
        pooled_done.record(main_stream)
        # p before the router: the side stream writes it, so it must not reuse a block _route frees (its partials)
        p = torch.empty_like(pooled)
        topk_ids, local_ids, topk_weights = _route(state, h, h32, data["local_start"], tokens, router_pdl=True)
        side_stream.wait_event(pooled_done)
        with torch.cuda.stream(side_stream):
            _rmsnorm_gain[(tokens,)](pooled, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=False, TRIG=False,
                                     num_warps=8)
        main_stream.wait_stream(side_stream)
        if not torch.cuda.is_current_stream_capturing():
            for t in (pooled, p):
                t.record_stream(side_stream)
        moe = _experts_moe(p, topk_weights, local_ids, state)
        return _tail(moe, stream1, h, tokens, state, pdl) + (topk_ids,)
    if (not fp32_router and state.get("pool_router") is not None
            and _opt(state, tokens, "fr", tokens in FUSED_ROUTER_TOKENS)):
        # SGLang's router above 8 tokens = BF16 h against the BF16 hi/lo split of the FP32 weight (FP32 accumulation):
        # the same h as the pooler, so one GEMM [pooler; hi; lo] with FP32 output; the routing sums the hi/lo logits,
        # the rmsnorm rounds the pooler part to BF16 (what the BF16-out pooler GEMM returns). No side stream.
        pr = torch.mm(h, state["pool_router"].t(), out_dtype=torch.float32)
        ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        lids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        wts = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
        # s37: T > 32: moe_route right after the routing, the pre-expert rmsnorm inside moe_gathern (strided FP32 rows)
        gnorm = (tokens > 32 and _use_moe(tokens) and _moe_TW_CFG and _opt(state, tokens, "gnorm", GNORM_DEFAULT))
        rg = gnorm and _opt(state, tokens, "rg", RG_DEFAULT)
        bkt = gnorm and not rg and _opt(state, tokens, "bkt", BKT_DEFAULT)
        rpdl = (bkt or rg) and _opt(state, tokens, "route_pdl", ROUTE_PDL)
        _routing[(tokens,)](pr[:, D_EXPERT:], state["router_bias"], data["local_start"], ids, lids, wts, tokens,
                            E=N_EXPERTS, KS=2, TOPK=TOP_K, LOCAL=LOCAL_EXPERTS, MULT=ROUTER_MULTIPLIER, REPS=ROUTE_EPS,
                            SKS=N_EXPERTS, ST=D_EXPERT + 2 * N_EXPERTS, TRIG=rpdl, num_warps=1 if tokens >= ROUTING_WARPS_1_MIN_TOKENS else 2)
        bucket = _moe_bucket(state, lids, rpdl) if bkt else None
        if gnorm:
            p = pr[:, :D_EXPERT]
        else:
            p = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
            _rmsnorm_gain[(tokens,)](pr, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=False, TRIG=False,
                                     XS=D_EXPERT + 2 * N_EXPERTS, ROUND=True, num_warps=8)
        moe = _experts_moe(p, wts, lids, state, bucket, gnorm) if _use_moe(tokens) else (
            _experts_marlin if HAVE_SGL else _experts_local)(p, wts, lids, state)
        return _tail(moe, stream1, h, tokens, state, pdl) + (ids,)
    if pr1:  # pooler + router + routing in one launch (proj pr1), PDL secondary of the boundary: no side stream
        pooled = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
        logits = torch.empty((tokens, N_EXPERTS), dtype=torch.float32, device=dev)
        topk_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        topk_weights = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
        bucket = None
        if tokens == 1:
            pk = state["g13"]["pooler"]
            _G10["lib"].pr1_launch(pk[0], pk[1], pk[2], h, pooled, pk[3], state["router"], h32, logits, state["pr1_cnt"],
                                   state["router_bias"], data["local_start"], topk_ids, local_ids, topk_weights,
                                   ROUTER_MULTIPLIER, ROUTE_EPS, LOCAL_EXPERTS, *PR1_CFG[tokens][:2], _opt(state, tokens, "pr1_pf", PR1_CFG[tokens][2]),
                                   PR1_CFG[tokens][3], _opt(state, tokens, "pr1_rb", PR1_CFG[tokens][4]),
                                   PR1_CFG[tokens][5], _opt(state, tokens, "pr1_npf", PR1_CFG[tokens][6]), 1,
                                   [a, data["stream"], alpha, beta, state["pre_ffn"], stream1, h] if bnd else [],
                                   POST_NORM_EPS, RMS_EPS)
        else:
            pk = state["g14"]["pooler"]
            part = torch.empty(8 * tokens * N_EXPERTS, dtype=torch.float32, device=dev)
            _G10["lib"].pr8_launch(pk[0], pk[1], pk[2], pk[3], h, pooled, state["router"], h32, part, logits, state["pr1_cnt"],
                                   state["router_bias"], data["local_start"], topk_ids, local_ids, topk_weights,
                                   ROUTER_MULTIPLIER, ROUTE_EPS, LOCAL_EXPERTS, *(PR8L_ALTS[_opt(state, tokens, "pr8l", 0) - 1] if _opt(state, tokens, "pr8l", 0) else PR8_CFG[tokens]), 1)
    else:
        side_stream.wait_stream(main_stream)
        with torch.cuda.stream(side_stream):
            bkt = _use_moe(tokens) and (TK2_MAX_TOKENS < tokens <= BUCKET_SIDE_MAX_TOKENS
                                        or tokens > 32 and _moe_TW_CFG and _opt(state, tokens, "bkt", BKT_DEFAULT)
                                        and not _opt(state, tokens, "rg", RG_DEFAULT))
            rpdl = bkt and _opt(state, tokens, "route_pdl", ROUTE_PDL)
            topk_ids, local_ids, topk_weights = _route(state, h, h32, data["local_start"], tokens, trig=rpdl)
            # The experts' bucketing too, off the main stream's path.
            bucket = _moe_bucket(state, local_ids, rpdl) if bkt else None
    if pr1:
        pass
    elif tokens in POOLER_HI_PRIORITY:
        # the pooler on a high-priority stream: its CTAs go ahead of the router's remaining waves
        hi = state["hi_stream"]
        hi.wait_stream(main_stream)
        with torch.cuda.stream(hi):
            pooled = _linear(h, dense["pooler"], "pooler", state)
        main_stream.wait_stream(hi)
        if not torch.cuda.is_current_stream_capturing():
            h.record_stream(hi)
            pooled.record_stream(main_stream)
    elif (pooled := _g19(h, "pooler", state)) is not None:
        pass
    elif tokens in POOLER_BMM and not _opt(state, tokens, "lt:pooler", False):
        # split K in NS slabs as one strided batched GEMM with FP32 out; the rmsnorm sums the slabs (no reduce kernel)
        ns = POOLER_BMM[tokens]
        kc = HIDDEN // ns
        wp = dense["pooler"]
        pooled = torch.bmm(h.view(tokens, ns, kc).permute(1, 0, 2), wp.view(D_EXPERT, ns, kc).permute(1, 2, 0),
                           out_dtype=torch.float32)
    else:
        pooled = _linear(h, dense["pooler"], "pooler", state)
    p = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tk2_pdl = TK2_PDL13 and tokens <= TK2_MAX_TOKENS and _use_moe(tokens)
    if tk2_pdl:
        # moe_tk2 gemm13 reads the routing before its griddepcontrol.wait: join first, so that its only
        # (programmatic) predecessor is this rmsnorm, which has full edges from the pooler and the routing.
        main_stream.wait_stream(side_stream)
    nsp = pooled.shape[0] if pooled.dim() == 3 else 1  # bmm slabs [ns, T, n]
    nu8 = _nu8_on(state, tokens)
    tk2fx = (pr1 and pdl and not tk2_pdl and tokens in TK2FX_TOKENS and _use_moe(tokens) and _moe_TK2F
             and _opt(state, tokens, "tk2fx", TK2FX_DEFAULT))
    tk2fj = (not pr1 and not tk2_pdl and tokens in TK2FJ_TOKENS and tokens <= TK2_MAX_TOKENS and _use_moe(tokens)
             and _moe_TK2F and _opt(state, tokens, "tk2fj", TK2FJ_DEFAULT))
    if tk2fj:  # s27: join the side stream (routing) first: the rmsnorm loses its PDL, moe_tk2f becomes its PDL secondary
        main_stream.wait_stream(side_stream)
    rpdl = pdl and not tk2_pdl and not tk2fj
    # s86: the rmsnorm below reads local_ids (nu8d ZL): the routing must be on this stream (pr1) or joined (tk2fj).
    # Without pr1 it is still running on the side stream there: opt test verify 1 sliding {pr1: False} failed 7/30 seeds.
    nu8d = nu8 and _nu8_d(state, tokens) and local_ids.is_contiguous() and (pr1 or tk2fj)
    y32 = torch.empty((81 if nu8d else 65, D_EXPERT), dtype=torch.float32, device=dev) if nu8 else None
    gnorm = (not pr1 and not nu8 and tokens > 32 and _use_moe(tokens) and _moe_TW_CFG and pooled.dim() == 2
             and pooled.is_contiguous() and _opt(state, tokens, "gnorm", GNORM_DEFAULT))
    if gnorm:
        p = pooled  # s27 GNORM: the rmsnorm runs inside the experts' gather
    else:
      _rmsnorm_gain[(tokens,)](pooled, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=rpdl,
                             TRIG=tk2_pdl, ROUND=pooled.dtype == torch.float32, NS=nsp, SS=tokens * D_EXPERT, num_warps=8,
                             launch_pdl=rpdl, z_ptr=y32[64] if nu8 else None, ZN=256 if nu8 and not nu8d else 0, TRIG2=tk2fx or tk2fj,
                             zl_ptr=y32[65] if nu8d else None, ids_ptr=local_ids if nu8d else None, ZL=LOCAL_EXPERTS if nu8d else 0)
    if tk2fx or tk2fj:
        state["moe"]["lib"].moe_set_tk2fpdl(1)  # the next moe_tk2f launch (one-shot) is this rmsnorm's PDL secondary
    if not pr1:
        main_stream.wait_stream(side_stream)
    if not pr1 and not torch.cuda.is_current_stream_capturing():
        for t in (topk_ids, local_ids, topk_weights) + (bucket or ()):
            t.record_stream(main_stream)
        for t in (h, h32):
            t.record_stream(side_stream)
    if (tokens in N2U_TOKENS and _use_moe(tokens) and _moe_TK2F and _G10.get("lib") is not None
            and tokens <= SLICED_UNPOOL_MAX_TOKENS):
        if N2U_BOUNDARY:
            stream2 = torch.empty_like(stream1)
            hidden = torch.empty_like(h)
            alpha, beta = state["res2"]
            _experts_n2u(p, topk_weights, local_ids, state, (stream1, alpha, beta, state["post_ffn"], state["next_pre_attn"],
                                                             stream2, hidden, state["n2u_cnt"]))
            return hidden, stream2, topk_ids
        f = _experts_n2u(p, topk_weights, local_ids, state)
        return _tail(None, stream1, h, tokens, state, pdl, f=f) + (topk_ids,)
    if nu8:
        f, sc = _experts_nu8(p, topk_weights, local_ids, state, y32)
        return _tail(None, stream1, h, tokens, state, pdl, f=f, sc=sc) + (topk_ids,)
    if _use_moe(tokens):
        moe = _experts_moe(p, topk_weights, local_ids, state, bucket, gnorm)  # all-reduce: identity
    else:
        moe = (_experts_marlin if HAVE_SGL else _experts_local)(p, topk_weights, local_ids, state)
    return _tail(moe, stream1, h, tokens, state, pdl) + (topk_ids,)


_PREBUILD = [_BUILD_POOL.submit(f) for f in (_moe__lib, _g10_lib, _lt_build, _g22_build, _g24_build, _g27_build)]
