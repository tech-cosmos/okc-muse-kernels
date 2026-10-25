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
from triton.language.extra.cuda import gdc_wait

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
from torch.utils.cpp_extension import load_inline as _load_inline
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

template <int PAGE>
__global__ void __launch_bounds__(128) decode(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec) {
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
    const unsigned short* qp = qkv + (long)b * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const unsigned short* kp = qkv + (long)b * QKV_W + (HQ + h) * D + 2 * lane;
    const unsigned kw = *(const unsigned*)kp, vw = *(const unsigned*)(kp + HKV * D);
    const unsigned gw = *(const unsigned*)(qkv + (long)b * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);
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
                double qk_scale) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && CSL_SPEC) ? mp - 1 : -1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(32 * batch);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define CSL_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, spec
    if (page == 1) {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(csl::decode<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
        cudaLaunchKernelEx(&cfg, csl::decode<1>, CSL_ARGS);
    } else {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(csl::decode<64>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
        TORCH_CHECK(page == 64, "csl: page 1 or 64");
        cudaLaunchKernelEx(&cfg, csl::decode<64>, CSL_ARGS);
    }
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

template <int PAGE>
__global__ void __launch_bounds__(128) verify(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
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
    const unsigned short* qp = qkv + (long)tok * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    // The block's new tokens 0..jt (only the new-token warp needs them).
    const bool newer = rank == 15 && wk == 3;
    const unsigned gw = *(const unsigned*)(qkv + (long)tok * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);
    const long nloc = loc[tok];
    const float kscale = *ks, vscale = *vs;
    const float rks = 1.f / kscale, rvs = 1.f / vscale;
    // spec >= 0 (batch 1, page 1): SGLang slices the page table to the batch's longest length, so the window's
    // page-table range is known from the table width: gather from it without waiting for seq_lens, then compare
    // with the real length and redo the loads if it differs (never on SGLang's inputs; correct either way).
    int seq = spec >= 0 ? spec : seqs[b];
    // The block's new tokens 0..jt (only the new-token warp needs them), cos/sin rows and cache rows, all issued
    // before the gathers (clamped indices so the loads are unconditional).
    unsigned nkw[L], nvw[L];
    float nco[L], nsi[L];
#pragma unroll
    for (int n = 0; n < L; ++n) {
        // Predicated (not branched) so they stay in the first basic block, issued with the gathers.
        const int nn = min(n, jt);
        const unsigned short* kp = qkv + (long)(b * L + nn) * QKV_W + (HQ + h) * D + 2 * lane;
        nkw[n] = newer ? *(const unsigned*)kp : 0u; nvw[n] = newer ? *(const unsigned*)(kp + HKV * D) : 0u;
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
                double qk_scale) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && CSL_SPEC) ? mp - vtk::L : -1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(32 * batch * vtk::L);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define VTK_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, spec
    if (page == 1) {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(vtk::verify<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
        cudaLaunchKernelEx(&cfg, vtk::verify<1>, VTK_ARGS);
    } else {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(vtk::verify<64>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
        TORCH_CHECK(page == 64, "vtk: page 1 or 64");
        cudaLaunchKernelEx(&cfg, vtk::verify<64>, VTK_ARGS);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
try:
    _CSL = _load_inline(
        name="muse_csl_s96",
        cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);"
                    "void vtk_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);",
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

template <int NS>
__device__ __forceinline__ void consumer(Smem<NS>& sm, int h, const __nv_bfloat16* __restrict__ qkv,
                                         const float* __restrict__ cs, Range rg, int split, int S, int qkv_w,
                                         float scale, int win, float* __restrict__ pacc, float* __restrict__ pml,
                                         uint8_t* __restrict__ kc,
                                         uint8_t* __restrict__ vc, const long long* __restrict__ loc, float ksv,
                                         float vsv, int H, int total) {
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

template <int NS, int LAG>
__global__ void __launch_bounds__(384, NV_MINB) nvs(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cs,
                                              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                              const float* __restrict__ ks, const float* __restrict__ vsc,
                                              const long long* __restrict__ loc, const int* __restrict__ pt,
                                              const int* __restrict__ seq, float* __restrict__ pacc,
                                              float* __restrict__ pml, float qk_scale, int S, int ch, int maxp,
                                              int qkv_w, int win) {
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
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS>(sm, wg - 1, qkv, cs, wg == 1 ? r0 : r1, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc,
                     loc, ks[0], vsc[0], H, r0.n);
    }
#else
    const Range rg = rng(seq, blockIdx.x / S, split, ch, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS, LAG>(sm, kc, vc, pt, rg, maxp);
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS>(sm, wg - 1, qkv, cs, rg, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc, loc, ks[0], vsc[0], 0, 0);
    }
#endif
}
}  // namespace nv

void nvs(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vsc,
         torch::Tensor loc, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, double qk_scale, int64_t S, int64_t ch, int64_t win) {
    // Producer lag NV_LAG (1) at every S: LAG 2 at 4 splits was H100 -0.18 us but H200 +0.2 (check 9249146d, verify 32).
    auto f = nv::nvs<NV_NS, NV_LAG>;
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    static bool once = false;
    if (!once) {
        cudaFuncSetAttribute(nv::nvs<NV_NS, NV_LAG>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        once = true;
    }
    const int B = (int)seq.size(0);
    f<<<dim3((unsigned)(S * B * (1 + NV_HALF))), 384, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)qkv.data_ptr(), (const float*)cs.data_ptr(), (uint8_t*)kc.data_ptr(),
        (uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const float*)vsc.data_ptr(),
        (const long long*)loc.data_ptr(), (const int*)pt.data_ptr(),
        (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale, (int)S, (int)ch,
        (int)pt.size(1), (int)qkv.size(-1), (int)win);
}
'''
try:
    _NVS = _load_inline(
        name="muse_nvs_a2s13h",
        cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, "
                    "int64_t);",
        cuda_sources=_NVS_SRC, functions=["nvs"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"], verbose=False)
except Exception as _e:  # noqa: BLE001
    print("nvs build failed, using Triton:", _e)
    _NVS = None
NVS_HALF = '#define NV_HALF 1' in _NVS_SRC
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
template <int NS>
__device__ __forceinline__ void dconsumer(Smem<NS>& sm, const __nv_bfloat16* __restrict__ qkv,
                                          const float* __restrict__ cs, Range rg, int split, int S, int qkv_w,
                                          float scale, int win, float* __restrict__ pacc, float* __restrict__ pml,
                                          uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                          const long long* __restrict__ loc, float ksv, float vsv, __nv_bfloat16* __restrict__ out) {
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
        const __nv_bfloat16* src = qkv + (long long)b * qkv_w + r * HD + half * 32;
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
        const int c0 = h * 4 + half * 2, z0 = (1 - h) * 4 + half * 2;  // 16-B chunks: live d, zero d
        *reinterpret_cast<uint4*>(qd + r * 128 + ((c0 ^ (r & 7)) << 4)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(qd + r * 128 + (((c0 + 1) ^ (r & 7)) << 4)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        *reinterpret_cast<uint4*>(qd + r * 128 + ((z0 ^ (r & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(qd + r * 128 + (((z0 + 1) ^ (r & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        if (last_split) {
            // The step's new token: lane = KV head (lane / 16), d 4 (lane % 16) + 0..3: RMS-norm + RoPE, E4M3 (as
            // _new), stored into the cache and into the new-token K row 0 / V^T key slot 0.
            const int hh = lane >> 4, d0 = (lane & 15) * 4, npos = rg.seq;
            const __nv_bfloat16* kp = qkv + (long long)b * qkv_w + (16 + hh) * HD + d0;
            const uint2 kr = *reinterpret_cast<const uint2*>(kp), vr = *reinterpret_cast<const uint2*>(kp + 2 * HD);
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

template <int NS>
#ifdef NV_MAXR
__global__ void __maxnreg__(NV_MAXR) nds(
#else
__global__ void __launch_bounds__(256, NV_MINB) nds(
#endif
const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cs,
                                              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                              const float* __restrict__ ks, const float* __restrict__ vsc,
                                              const long long* __restrict__ loc, const int* __restrict__ pt,
                                              const int* __restrict__ seq, float* __restrict__ pacc,
                                              float* __restrict__ pml, float qk_scale, int S, int ch, int maxp,
                                              int qkv_w, int win, __nv_bfloat16* __restrict__ out) {
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
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        dconsumer<NS>(sm, qkv, cs, rg, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc, loc, ks[0], vsc[0], out);
    }
}
}  // namespace nv

void nds(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vsc,
         torch::Tensor loc, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, torch::Tensor out, double qk_scale, int64_t S, int64_t ch,
         int64_t win) {
    auto f = nv::nds<NV_NS>;
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    static bool once = false;
    if (!once) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
    const int B = (int)seq.size(0);
    f<<<dim3((unsigned)(S * B)), 256, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)qkv.data_ptr(), (const float*)cs.data_ptr(), (uint8_t*)kc.data_ptr(),
        (uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const float*)vsc.data_ptr(),
        (const long long*)loc.data_ptr(), (const int*)pt.data_ptr(),
        (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale, (int)S, (int)ch,
        (int)pt.size(1), (int)qkv.size(-1), (int)win, (__nv_bfloat16*)out.data_ptr());
}
'''
try:
    _NDS = _load_inline(
        name="muse_nds_a2s10",
        cpp_sources="void nds(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                    "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                    "int64_t, int64_t, int64_t);",
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
        gate = gl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col).to(gl.float32)
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
                         D: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr, LAG: gl.constexpr):
        """_gw_producer_cv over the program's segments (the ring position carries across them)."""
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
                        vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                        vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                        fence_async_shared()
                        mbarrier.arrive(ready.index(sj))
            it += ntiles

    @gluon.jit
    def _gw_consumer_bal(qkv_ptr, q_s, k_ring, vh, ready, empty, seq_ptr, pid, h, scale, pacc_ptr, pml_ptr,
                         HEAD: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr,
                         B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, NID: gl.constexpr, M: gl.constexpr,
                         D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr, LZ: gl.constexpr = 0):
        """_gw_consumer_cv (PEEL, LSUM) over the program's segments: each loads its request's queries and stores
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
                for i in range(LZ_PK, ntiles):
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
    def _gd_full_body(seq_in, SEQ_IN: gl.constexpr, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                 cs_ptr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
                 MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
                 TPV: gl.constexpr = False, WINDOW: gl.constexpr = 0, ROPE: gl.constexpr = False,
                 SEP: gl.constexpr = False, Q16: gl.constexpr = True, KST: gl.constexpr = True,
                 BAL: gl.constexpr = False, B: gl.constexpr = 1, B_PAD: gl.constexpr = 32, NP: gl.constexpr = 1,
                 NID: gl.constexpr = 1, RED: gl.constexpr = False, red_ptr=None, rcnt_ptr=None, out_ptr=None,
                 NR: gl.constexpr = 4, epi=True, RCW: gl.constexpr = 0):
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
        x = gl.load(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + (c_g % D)[None, :], mask=q_ok,
                    other=0.0).to(gl.float32)
        x = gl.where(q_ok, x, 0.0)
        rs = gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]
        if ROPE:
            # Interleaved RoPE at position seq (the norm commutes with the rotation).
            xp = gl.load(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + ((c_g % D) ^ 1)[None, :],
                         mask=q_ok, other=0.0).to(gl.float32)
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
            t = t + BN
            pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)
            if SEQ_IN:
                pg = gl.maximum(pg, 0)  # speculative length: entries past the real one are -1
            cp.wait_group(NS - 1)
            gl.barrier()
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
            kx = gl.where(q_ok, gl.load(qkv_ptr + b * QKV_W + (HQ + h) * D + c_g)[None, :].to(gl.float32) + zero, 0.0)
            krs = gl.rsqrt(gl.sum(kx * kx, 1) / D + 1e-5)[:, None]
            if ROPE:
                kxp = gl.where(q_ok, gl.load(qkv_ptr + b * QKV_W + (HQ + h) * D + (c_g ^ 1))[None, :].to(gl.float32)
                               + zero, 0.0)
                kx = kx * cos + kxp * sin
            kq = (kx * krs / k_scale).to(gl.float8e4nv)
            vx = gl.load(qkv_ptr + b * QKV_W + (HQ + HKV + h) * D + c_g)[None, :].to(gl.float32) + zero
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
                 rcnt_ptr=None, out_ptr=None, NR: gl.constexpr = 4, RCW: gl.constexpr = 0):
        # SPEC > 0 (batch 1, page 1): seq = table width - 1 known from the shape (SGLang slices the page table to the
        # batch's longest length); start on it, redo with the real length if it differs (as in _attn_split).
        if SPEC > 0:
            real = gl.load(seq_ptr + gl.program_id(1))
            _gd_full_body(SPEC, True, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale, cs_ptr, HQ, HKV, G, D, PAGE, MAX_PAGES, S, BN, NS, TPV, WINDOW, ROPE, SEP, Q16, KST, BAL, B, B_PAD, NP, NID, RED, red_ptr, rcnt_ptr, out_ptr, NR, real == SPEC, RCW)
            if real != SPEC:
                _gd_full_redo(real, True, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale, cs_ptr, HQ, HKV, G, D, PAGE, MAX_PAGES, S, BN, NS, TPV, WINDOW, ROPE, SEP, Q16, KST, BAL, B, B_PAD, NP, NID, RED, red_ptr, rcnt_ptr, out_ptr, NR, True, RCW)
        else:
            _gd_full_body(0, False, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale, cs_ptr, HQ, HKV, G, D, PAGE, MAX_PAGES, S, BN, NS, TPV, WINDOW, ROPE, SEP, Q16, KST, BAL, B, B_PAD, NP, NID, RED, red_ptr, rcnt_ptr, out_ptr, NR, True, RCW)

    @gluon.jit
    def _gw_verify(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                   HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
                   PAGE: gl.constexpr, MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr,
                   NS: gl.constexpr, PW: gl.constexpr, PREGS: gl.constexpr, CREGS: gl.constexpr,
                   LAG: gl.constexpr = 0, PEEL: gl.constexpr = False, LSUM: gl.constexpr = False,
                   FUSE: gl.constexpr = False, cnt_ptr=None, out_ptr=None, RS: gl.constexpr = 1,
                   R_PAD: gl.constexpr = 1, BAL: gl.constexpr = False, B: gl.constexpr = 1,
                   B_PAD: gl.constexpr = 1, NID: gl.constexpr = 1, NP: gl.constexpr = 1, LZ: gl.constexpr = 0):
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
                                    pml_ptr, 0, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS, LZ)),
                (_gw_consumer_bal, (qkv_ptr, q1, k_ring, vh1, ready, empty, seq_ptr, split, h, scale, pacc_ptr,
                                    pml_ptr, 1, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS, LZ)),
                (_gw_producer_bal, (seq_ptr, pt_ptr, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, vh0, vh1, ready,
                                    empty, split, B, B_PAD, NP, MAX_PAGES, PAGE, ROW_W, BN, C, D, NS, PW,
                                    LAG)),
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
    gate = tl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col, mask=ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + tok[:, None] * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16), mask=ok)


@triton.jit
def _combine(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
             HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
             D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, PDL: tl.constexpr, ONE: tl.constexpr = True,
             REDO: tl.constexpr = False, seq_ptr=None, kc_ptr=None, vc_ptr=None, ks_ptr=None, pt_ptr=None, qk_scale=1.0,
             MAX_PAGES: tl.constexpr = 1, PAGE: tl.constexpr = 1, CHUNK: tl.constexpr = 1):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    b = tok // L
    h = head // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + head * D + d).to(tl.float32)
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
            x = tl.load(qkv_ptr + tok * QKV_W + head * D + d).to(tl.float32)
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
                PDL: tl.constexpr, ONE: tl.constexpr = True):
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
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col).to(tl.float32)
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
                 pt_ptr=None, qk_scale=1.0, MAX_PAGES: tl.constexpr = 1, PAGE: tl.constexpr = 1):
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
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col).to(tl.float32)
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
            x = tl.load(qkv_ptr + tok * QKV_W + col).to(tl.float32)
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
            return dict(splits=127, block_n=64, warps=4, stages=8, hpp=1, fuse=False, qk8=True, pdt="f32", tsep=True,
                        pf=True, pre=2, fix=1)
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
             SL: tl.constexpr = 0, RCW: tl.constexpr = 0, BAL: tl.constexpr = False):
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
    gate = tl.load(qkv_ptr + (HQ + 2 * HKV) * D + col).to(tl.float32)
    vs = tl.load(vs_ptr)
    gdc_wait()
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
    _NVS.nvs(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], data["k_scale"],
             data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml,
             cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, S, chunk, cfg["window"])
    _combine_hb[(batch * length, hq // 2)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=hq // hkv, L=length, M=rows, D=d, S=S, S_PAD=triton.next_power_of_2(S), SB=S, HB=2,
        ONE=MERGE1, PDL=USE_PDL, num_warps=1, launch_pdl=USE_PDL)
    return out


def _decode_nds(data, cfg):
    """Decode sliding (L 1, G 8, D 64, HKV 2, page 1) on the native nds kernel + _combine_hb."""
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv, device = data["qkv"], data["qkv"].device
    S = 8  # 256 CTAs at batch 32, 2 per SM
    chunk = triton.cdiv(triton.cdiv(cfg["window"], S), 64) * 64
    rows = hq // hkv
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device=device)
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    _NDS.nds(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], data["k_scale"],
             data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml, out,
             cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, S, chunk, cfg["window"])
    _combine_hb[(batch, hq // 2)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=rows, L=1, M=rows, D=d, S=S, S_PAD=S, SB=S, HB=2,
        ONE=MERGE1, PDL=USE_PDL, num_warps=1, launch_pdl=USE_PDL)
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
    if _CSL is not None and cfg["window"] == 2048 and length == 1 and batch <= CSL_MAX_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] in (1, 64) and cfg["rope_theta"]:
        out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
        _CSL.csl_decode(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], page_table,
                        data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out, cfg["page_size"],
                        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E)
        return out
    if _CSL is not None and cfg["window"] == 2048 and length == 8 and batch <= VTK_MAX_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] in (1, 64) and cfg["rope_theta"]:
        out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
        _CSL.vtk_verify(qkv, _cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], page_table,
                        data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out, cfg["page_size"],
                        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E)
        return out
    if _NVS is not None and cfg["window"] == 2048 and length == 8 and batch >= NVS_MIN_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] == 1 and cfg["rope_theta"]:
        return _verify_nvs(data, cfg)
    if _NDS is not None and cfg["window"] == 2048 and length == 1 and batch >= NDS_MIN_BATCH and hq == 16 \
            and hkv == 2 and d == 64 and cfg["page_size"] == 1 and cfg["rope_theta"]:
        return _decode_nds(data, cfg)
    plan = _plan(cfg, max_pages, device)
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
        LAG=plan.get("lag", 1), BAL=True, B=batch, B_PAD=b_pad, NID=nid, NP=progs, LZ=plan.get("lz", LZ), num_warps=4, maxnreg=plan["maxnreg"],
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
RF_NW = 4


def _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, sl, bal):
    red, rcnt, rcw = _red_bufs(batch, group, d, sl, qkv.device)
    _red_fin[(batch,)](qkv, red, rcnt, pacc, pml, data["v_scale"], out, HQ=hq, HKV=hkv, G=group, D=d, M=2 * group,
                       S=splits, NR=4 if batch == 1 else BNR, SL=0 if batch == 1 else sl, RCW=rcw, BAL=bal, num_warps=RF_NW1 if batch == 1 else RF_NW, launch_pdl=True)


def _decode_gluon(data, cfg, plan):
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    qkv = data["qkv"]
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
            **(_red_args(batch, group, d, nid, device, out) if red else {}), num_warps=1)
        if red:
            _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, nid, True)
            return out
        hb = plan.get("hb", 2)
        _combine_bal[(batch, hq // hb)](
            qkv, data["seq_lens"], pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, B=batch, B_PAD=b_pad, P=progs, NID=nid,
            BN=plan["block_n"], SB=plan.get("sb", min(128, triton.next_power_of_2(2 * progs // batch + 1))), HB=hb,
            ONE=MERGE1, PROP=True, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
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
        **(_red_args(batch, group, d, s, device, out) if red else {}), num_warps=1)
    if red:
        _red_launch(qkv, pacc, pml, data, out, batch, hq, hkv, group, d, splits, s, False)
        return out
    hb = plan.get("hb", 1)
    if hb > 1:
        _combine_hb[(batch, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, S=s, S_PAD=triton.next_power_of_2(s),
            SB=triton.next_power_of_2(s), HB=hb,
            ONE=MERGE1, PDL=USE_PDL, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
        return out
    _combine[(batch, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=plan.get("csb", min(128, triton.next_power_of_2(s))), ONE=MERGE1, PDL=USE_PDL,
        num_warps=plan.get("cw", 4 if s > 32 else 2), launch_pdl=USE_PDL)
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

#define KD 4096          // K of both GEMMs
#define NOUT 4096        // output rows of both GEMMs (gate/up each I, w2 H)

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
                                                int start, int count, float eps, __nv_bfloat16* __restrict__ out) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = ids[tok * 8 + s] - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      uint4 raw = *reinterpret_cast<const uint4*>(Y + ((size_t)(tok * 8 + s)) * NOUT + tid * 8);
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
    const float f = tw[tok * 8 + s] * rsqrtf(tot / NOUT + eps);
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
__global__ void __launch_bounds__(512) moe_norm2(const float* __restrict__ Y, const int* __restrict__ inv,
                                                 const int* __restrict__ ids, const float* __restrict__ tw,
                                                 const float* __restrict__ w1, const float* __restrict__ gs, int start,
                                                 int count, float eps, __nv_bfloat16* __restrict__ out) {
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float ssum[16][8];
  float wv[8];
  {
    const float4* wp = reinterpret_cast<const float4*>(w1 + tid * 8);
    float4 a = wp[0], b = wp[1];
    wv[0] = a.x; wv[1] = a.y; wv[2] = a.z; wv[3] = a.w; wv[4] = b.x; wv[5] = b.y; wv[6] = b.z; wv[7] = b.w;
  }
  float v[8][8], ss[8];
  bool loc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = ids[tok * 8 + s] - start;
    loc[s] = e >= 0 && e < count;
    ss[s] = 0.f;
    if (loc[s]) {
      const float gsc = gs[e] * 256.0f;
      const float4* p = reinterpret_cast<const float4*>(Y + (size_t)inv[tok * 8 + s] * NOUT + tid * 8);
      const float4 a = p[0], b = p[1];
      const float f[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int k = 0; k < 8; ++k) v[s][k] = f[k] * gsc * wv[k];
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
    const float f = tw[tok * 8 + s] * rsqrtf(tot / NOUT + eps);
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
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
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
template <int NRT, int NC, int NSTG, bool GATED>
__global__ void __launch_bounds__((NC + 1) * 32, 1) moe_tk2(
    const uint4* __restrict__ Wq, const uint32_t* __restrict__ S2, const uint32_t* __restrict__ lut,
    const float* __restrict__ gs, const __nv_bfloat16* __restrict__ X, const int* __restrict__ ids, int nroutes,
    int start, int count, int* __restrict__ meta, int* __restrict__ sroutes, int* __restrict__ inv,
    void* __restrict__ Out) {
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
  if (GATED) {
    // routing (M * 8 <= 256 routes): warp w takes routes [32 w, 32 w + 32)
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
      if (blockIdx.x == 0) { sroutes[pos] = i; inv[i] = pos; }
    }
    if (blockIdx.x == 0 && tid < 33) {
      meta[33 + tid] = s_off[tid];
      if (tid < 32) meta[1 + tid] = s_e[tid];
      if (tid == 0) meta[0] = s_nact;
    }
  } else {
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
  uint4 xf[NQ][2];
  int a = 0, seg_end = -1, seg_start = 0, off = 0, n = 0, st = 0, ph = 0, rb = 0;
  float gsc = 0.f;
  const uint4* wsm = smem + warp * 32 + lane;
  for (int R = r0; R < r1; ++R) {
    if (R >= seg_end) {
      while (s_rs[a + 1] <= R) ++a;
      const int gi = (R - s_rs[a]) / RPE;
      seg_start = s_rs[a] + gi * RPE;  // segment = (expert, token group)
      seg_end = seg_start + RPE;
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
    const int rp = (R - seg_start) & (RPE - 1);
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
  moe_tk2<NRT, NC, NSTG, GATED><<<grid, (NC + 1) * 32, SMEM, st>>>(
      (const uint4*)wq, (const uint32_t*)s2, (const uint32_t*)lut, (const float*)gs, (const __nv_bfloat16*)x,
      (const int*)ids, nroutes, start, count, (int*)meta, (int*)sroutes, (int*)inv, out);
  return (int)cudaGetLastError();
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
struct TMaps { CUtensorMap m[4]; };  // box rows 8, 16, 32, 64

template <int RT, bool GATED>
struct TWC {
  static constexpr int XB = 64 * 128;
  static constexpr int WB = 8 * RT * 512;
  static constexpr int SB = 8 * RT * 64;
  static constexpr int STG = XB + WB + SB;
  static constexpr int NRB = (GATED ? 2 * NOUT : NOUT) / 16 / (8 * RT);
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

template <int RT, int KG, int N, int NSTG, bool GATED>
__device__ __forceinline__ void tw_item(const char* smem, uint64_t* full, uint64_t* empty, int& st, int& ph,
                                        const uint32_t* slut, int warp, int lane, int nv, int off, int rb, float gsc,
                                        const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  using C = TWC<RT, GATED>;
  constexpr int NA = N / 2;
  float acc[RT][NA];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[r][i] = 0.f;
  uint32_t afA[KG][RT][4], afB[KG][RT][4];
  int pst = -1;
#pragma unroll 1
  for (int kc = 0; kc < 64; ++kc) {
    const char* sb = smem + st * C::STG;
    if (kc) mbar_wait(&full[st], ph);
    uint4 wv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) wv[r] = reinterpret_cast<const uint4*>(sb + C::XB)[(warp * RT + r) * 32 + lane];
    uint32_t sc[RT / 2];
    if (RT == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(sb + C::XB + C::WB + (warp * 32 + lane) * RT * 2);
      sc[0] = v.x; sc[RT / 2 - 1] = v.y;
    } else {
      sc[0] = *reinterpret_cast<const uint32_t*>(sb + C::XB + C::WB + (warp * 32 + lane) * RT * 2);
    }
    uint32_t s0[RT], s1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const uint32_t sw = sc[r >> 1] >> ((r & 1) * 16);
      s0[r] = slut[sw & 0xFF];
      s1[r] = slut[(sw >> 8) & 0xFF];
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
  constexpr int H = GATED ? RT / 2 : RT;
#pragma unroll
  for (int r = 0; r < H; ++r) {
    const int tile = rb * 8 * H + warp * H + r;
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

template <int RT, int KG, int NSTG, bool GATED>
__global__ void __launch_bounds__(384, 1) moe_tw(
    const char* __restrict__ Wb, const char* __restrict__ Sb, const float* __restrict__ gs,
    const uint32_t* __restrict__ lut, const __grid_constant__ TMaps maps,
    int* __restrict__ meta, const int* __restrict__ sroutes, __nv_bfloat16* __restrict__ Out) {
  using C = TWC<RT, GATED>;
  constexpr int NRB = C::NRB;
  extern __shared__ __align__(1024) uint4 smem_tw[];
  char* smem = reinterpret_cast<char*>(smem_tw) + ((1024 - (smem_u32(smem_tw) & 1023)) & 1023);
  __shared__ uint32_t slut[256];
  __shared__ int s_e[32], s_off[33], s_cum[33], s_item[NSTG];
  __shared__ __align__(8) uint64_t full[NSTG], empty[NSTG];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
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
    if (warp > 8 || lane != 0) return;
    int* counter = meta + (GATED ? 66 : 67);
    int st = 0, ph = 0, i = 0;
    for (;;) {
      const int item = atomicAdd(counter, 1);
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
      const int off = s_off[e] + p * 64, nv = min(64, s_off[e + 1] - off);
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
#define TWI(NN) tw_item<RT, KG, NN, NSTG, GATED>(smem, full, empty, st, ph, slut, warp, lane, nv, off, rb, gsc, sroutes, Out)
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
#undef TWI
  }
}

template <int RT, int KG, int NSTG, bool GATED>
static int launch_tw(const void* wb, const void* sb, const void* gs, const void* lut, const TMaps& maps,
                     const void* meta, const void* sroutes, void* out, cudaStream_t st) {
  static int grid = 0;
  constexpr int SMEM = NSTG * TWC<RT, GATED>::STG + 1024;
  if (!grid) {
    int dev, nsm;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_tw<RT, KG, NSTG, GATED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    grid = nsm;
  }
  moe_tw<RT, KG, NSTG, GATED><<<grid, 384, SMEM, st>>>(
      (const char*)wb, (const char*)sb, (const float*)gs, (const uint32_t*)lut, maps,
      (int*)meta, (const int*)sroutes, (__nv_bfloat16*)out);
  return (int)cudaGetLastError();
}

// xg[pos] = x[sroutes[pos] >> 3] with each 64-chunk K-permuted (logical word (j*2+h)*4+t <- physical word 8t+2j+h).
__global__ void __launch_bounds__(256) moe_gather(const uint32_t* __restrict__ X, const int* __restrict__ meta,
                                                  const int* __restrict__ sroutes, uint32_t* __restrict__ XG) {
  const int pos = blockIdx.x;
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
  moe_norm<<<M, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)y.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (int)start, (int)count, (float)eps, (__nv_bfloat16*)out.data_ptr());
  return (int64_t)cudaGetLastError();
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
  for (int i = 0; i < 4; ++i) {
    cuuint64_t dims[2] = {(cuuint64_t)KD, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)KD * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)(8 << i)};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g_encode(&maps.m[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  }
}
int64_t moe_gather_launch(torch::Tensor x, torch::Tensor meta, torch::Tensor sroutes, torch::Tensor xg) {
  moe_gather<<<(int)sroutes.numel(), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint32_t*)x.data_ptr(), (const int*)meta.data_ptr(), (const int*)sroutes.data_ptr(), (uint32_t*)xg.data_ptr());
  return (int64_t)cudaGetLastError();
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
  return -1;
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
                         torch::Tensor gs, int64_t start, int64_t count, double eps, torch::Tensor out) {
  const int M = (int)out.size(0);
  moe_norm2<<<M, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const float*)acc2.data_ptr(), (const int*)inv.data_ptr(), (const int*)ids.data_ptr(), (const float*)tw.data_ptr(),
      (const float*)w1.data_ptr(), (const float*)gs.data_ptr(), (int)start, (int)count, (float)eps,
      (__nv_bfloat16*)out.data_ptr());
  return (int64_t)cudaGetLastError();
}
"""


def _moe__gen_wgmma():
    out = ["template <int N> __device__ __forceinline__ void wgmma_bf16(float* d, const uint32_t* a, uint64_t desc);"]
    for n in range(8, 72, 8):
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
        "int64_t moe_norm_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, "
        "torch::Tensor);"
        "int64_t moe_sk_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tk2_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tb_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_gather_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_tw_launch(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
        "int64_t moe_route2_launch(torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t);"
        "int64_t moe_act_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, "
        "torch::Tensor, torch::Tensor);"
        "int64_t moe_norm2_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, int64_t, int64_t, double, torch::Tensor);")


def _moe__lib():
    global _moe__LIB
    if _moe__LIB is None:
        from torch.utils.cpp_extension import load_inline
        h = hashlib.sha256(_moe_CUDA_SRC.encode()).hexdigest()[:12]
        _moe__LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_moe__CPP, cuda_sources=_moe_CUDA_SRC,
                           functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch", "moe_sk_launch", "moe_tk2_launch", "moe_tb_launch",
                                      "moe_gather_launch", "moe_tw_launch",
                                      "moe_route2_launch", "moe_act_launch", "moe_norm2_launch"],
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
    if _moe_TW_CFG:
        rt = _moe_TW_CFG // 100
        if rt not in _moe_TB_PACK:
            st[f"w13b{rt}"], st[f"w13sb{rt}"] = _moe__tb_pack(st["w13q"], st["w13s"], True, rt)
            st[f"w2b{rt}"], st[f"w2sb{rt}"] = _moe__tb_pack(st["w2q"], st["w2s"], False, rt)
    for rt in _moe_TB_PACK:
        st[f"w13b{rt}"], st[f"w13sb{rt}"] = _moe__tb_pack(st["w13q"], st["w13s"], True, rt)
        st[f"w2b{rt}"], st[f"w2sb{rt}"] = _moe__tb_pack(st["w2q"], st["w2s"], False, rt)
    st["gs13"] = weights["w13_gs"][:_moe_LOCAL].float().contiguous()
    st["gs2"] = weights["w2_gs"][:_moe_LOCAL].float().contiguous()
    st["lut"] = _moe__scale_lut(dev)
    st["w1"] = (1.0 + weights["post_norm_w"].float()).contiguous()
    torch.cuda.synchronize()
    return st


# (R, G) per batch size for gemm13 / gemm2: R row tiles (x16 rows) per item, G token groups of 8 per pass.
def _moe__plan(M):
    if M == 1:
        return 10, 10
    if M <= 32:
        return 11, 11
    return 118, 118


_moe_PLAN_OVERRIDE = None
_moe_TB_RT = 0  # big-M moe_tb on (1) or off (0: old moe_big path)
_moe_TB_CFG = 48  # RT*10 + stages
_moe_TW_CFG = 218  # big-M wgmma kernel moe_tw: RT*100 + KG*10 + stages (0: off)
_moe_TB_PACK = ()  # RT values packed in prepare for moe_tb
_moe__CFGS = (10, 11, 22, 118)
_moe_SK_CFG = 3088  # decode kernel cfg: 2DM = moe_sk (D, MB); 3[NC][NSTG] = moe_tk
_moe_TK2_NSTG = 8  # decode path: moe_tk2 stages (0: the 5-launch SK_CFG path)
_moe_SK_ZERO = 64  # CTAs zeroing Acc13 in route2


def _moe_kernel_fn(data, st):
    cfg = data["config"]
    assert cfg["local_start"] == st["start"] and cfg["local_count"] == st["count"]
    lib = st["lib"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].float().contiguous()
    M = x.shape[0]
    dev = x.device
    meta = torch.empty(68, dtype=torch.int32, device=dev)
    sroutes = torch.empty(M * _moe_TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((M * _moe_TOP_K, _moe_I), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, _moe_H), dtype=torch.bfloat16, device=dev)
    if M <= 32 and _moe_TK2_NSTG:
        inv = torch.empty(M * _moe_TOP_K, dtype=torch.int32, device=dev)
        y32 = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.float32, device=dev)
        lib.moe_tk2_launch(1, _moe_TK2_NSTG, st["w13q"], st["w13s2"], st["lut"], st["gs13"], x, ids, st["start"], st["count"],
                           meta, sroutes, inv, inter)
        lib.moe_tk2_launch(0, _moe_TK2_NSTG, st["w2q"], st["w2s2"], st["lut"], st["gs2"], inter, ids, st["start"],
                           st["count"], meta, sroutes, inv, y32)
        lib.moe_norm2_launch(y32, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], _moe_POST_NORM_EPS, out)
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
        lib.moe_norm2_launch(acc2, inv, ids, tw, st["w1"], st["gs2"], st["start"], st["count"], _moe_POST_NORM_EPS, out)
        return out
    p13, p2 = _moe_PLAN_OVERRIDE(M) if _moe_PLAN_OVERRIDE else _moe__plan(M)
    assert p13 in _moe__CFGS and p2 in _moe__CFGS, (p13, p2)
    lib.moe_route_launch(ids, st["start"], st["count"], meta, sroutes)
    if _moe_TW_CFG:
        # inter here holds gemm2's B in the K-permuted order (see moe_tw)
        rt = _moe_TW_CFG // 100
        xg = torch.empty((M * _moe_TOP_K, _moe_H), dtype=torch.bfloat16, device=dev)
        lib.moe_gather_launch(x, meta, sroutes, xg)
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
GEMV_PDL_SECONDARY = {(1, True): ("unpool_slice", "o"), (1, False): ("unpool_slice", "o"), (8, True): ("o",)}
LOCAL_FALLBACK = False  # no SGLang: also build the dequantizing reference experts (tests of the Marlin-side glue)  # PDL after a cuBLAS producer: no gain (H100 o GEMM + boundary 12.64 vs 12.74 us)


# ---- boundaries: post-norm of the branch, residual gate, pre-norm of the new stream ----

@triton.jit
def _boundary(br_ptr, s_ptr, alpha_ptr, beta_ptr, bgain_ptr, gain_ptr, s_out_ptr, h_ptr, h32_ptr,
              NB: tl.constexpr, N: tl.constexpr, BGAIN: tl.constexpr, H32: tl.constexpr,
              EPS_POST: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr):
    """One row. b = branch (its first NB columns; the rest are zero), times (1 + w) with BGAIN;
    s' = alpha s + beta rmsnorm(b, EPS_POST); h = rmsnorm(s', EPS) (1 + w) in BF16 (and FP32 with H32)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    in_b = cols < NB
    # Weights and the incoming stream first: with PDL they load while the branch's producer finishes.
    alpha = tl.load(alpha_ptr + cols)
    beta = tl.load(beta_ptr + cols)
    gain = tl.load(gain_ptr + cols)
    if BGAIN:
        bgain = tl.load(bgain_ptr + cols, mask=in_b, other=0.0)
    s = tl.load(s_ptr + row * N + cols)
    if PDL:
        gdc_wait()
    b = tl.load(br_ptr + row * NB + cols, mask=in_b, other=0.0).to(tl.float32)
    if BGAIN:
        b = b * bgain
    rb = tl.rsqrt(tl.sum(b * b, 0) / N + EPS_POST)
    s = alpha * s + beta * (b * rb)
    tl.store(s_out_ptr + row * N + cols, s)
    rs = tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    h = s * rs * gain
    tl.store(h_ptr + row * N + cols, h.to(tl.bfloat16))
    if H32:
        tl.store(h32_ptr + row * N + cols, h)


@triton.jit
def _rmsnorm_gain(x_ptr, gain_ptr, out_ptr, N: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr):
    """One row: rmsnorm(x) (1 + w), BF16 in and out."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    gain = tl.load(gain_ptr + cols)
    if PDL:
        gdc_wait()
    x = tl.load(x_ptr + row * N + cols).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(out_ptr + row * N + cols, (x * r * gain).to(tl.bfloat16))


# ---- router: split-K logits, then routing ----

@triton.jit
def _router_split(h_ptr, w_ptr, wlo_ptr, part_ptr, T, K: tl.constexpr, E: tl.constexpr,
                  BT: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr, KS: tl.constexpr, HILO: tl.constexpr):
    """part[ks, t, e] = sum over this split's K of h[t, k] w[e, k]. HILO: h BF16, w = hi + lo (BF16 pair, both
    [E, K]), FP32 accumulation; else h and w FP32 in FP32 FMA (ieee)."""
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
             TOPK: tl.constexpr, LOCAL: tl.constexpr, MULT: tl.constexpr, REPS: tl.constexpr):
    """One token: scores = sigmoid(MULT logits); the top TOPK of scores + bias (lowest index on ties);
    weights = picked scores / (their sum + REPS); local ids = (id - local_start) mod E, -1 off this GPU."""
    t = tl.program_id(0)
    e = tl.arange(0, E)
    logit = tl.zeros((E,), dtype=tl.float32)
    for ks in tl.static_range(KS):
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


def _route(state, h, h32, local_start, tokens):
    bt, be, bk, ks, warps = _router_plan(tokens)
    dev = h.device
    part = torch.empty((ks, tokens, N_EXPERTS), dtype=torch.float32, device=dev)
    fp32 = tokens <= ROUTER_FP32_MAX_TOKENS
    if fp32:
        _router_fma[(N_EXPERTS // be, ks)](h32, state["router"], part, tokens, K=HIDDEN, E=N_EXPERTS, BT=bt, BE=be,
                                           BK=bk, KS=ks, num_warps=warps)
    else:
        _router_split[(N_EXPERTS // be, ks, triton.cdiv(tokens, bt))](
            h, state["router_hi"], state["router_lo"], part, tokens, K=HIDDEN, E=N_EXPERTS, BT=bt, BE=be, BK=bk,
            KS=ks, HILO=True, num_warps=warps)
    ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    wts = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    _routing[(tokens,)](part, state["router_bias"], local_start, ids, lids, wts, tokens, E=N_EXPERTS, KS=ks,
                        TOPK=TOP_K, LOCAL=LOCAL_EXPERTS, MULT=ROUTER_MULTIPLIER, REPS=ROUTE_EPS, num_warps=2)
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
BUCKET_SIDE_MAX_TOKENS = 16  # side-stream bucketing won at T<=8 on the judge (c17d1581); T=128/256 unproven
TK2_MAX_TOKENS = 32  # moe's TMA decode kernel (moe_tk2: route fused into gemm13, 3 launches) at T <= this
# T > 32: moe's kernel_fn = route, gather, wgmma moe_tw gemm13/gemm2, norm (r-moe c39a176)
MARLIN_TOKENS = ()  # moe 6a1cda0 at T=32 beat Marlin on the judge: decode 32 sliding 218.3 (fc3e60d0) vs 225.0 (440361a9)


def _use_moe(tokens):
    return not HAVE_SGL or tokens not in MARLIN_TOKENS


def _moe_bucket(state, local_ids):
    """moe_src's route kernel: this GPU's routes bucketed by expert (meta, sorted route ids)."""
    dev = local_ids.device
    meta = torch.empty(68, dtype=torch.int32, device=dev)
    sroutes = torch.empty(local_ids.numel(), dtype=torch.int32, device=dev)
    state["moe"]["lib"].moe_route_launch(local_ids, 0, LOCAL_EXPERTS, meta, sroutes)
    return meta, sroutes


def _experts_moe(x, topk_weights, local_ids, state, bucket=None):
    """moe_src's route -> gemm13 -> gemm2 -> norm, on local ids (-1 off this GPU: skipped as out of range)."""
    st = state["moe"]
    lib = st["lib"]
    m = x.shape[0]
    if (m <= TK2_MAX_TOKENS and _moe_TK2_NSTG) or (m > 32 and _moe_TW_CFG and bucket is None):
        return _moe_kernel_fn({"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights,
                               "config": {"local_start": 0, "local_count": LOCAL_EXPERTS}}, st)
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
        "router_bias": weights["router_bias"].float().contiguous(),
        "res1": _residual(weights["attn_residual_gate"]), "res2": _residual(weights["ffn_residual_gate"]),
        "pre_ffn": gain("pre_ffn_norm"), "post_ffn": gain("post_ffn_norm"), "next_pre_attn": gain("next_pre_attn_norm"),
        "pre_expert": gain("pre_expert_norm"),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight in BF16.
        "post_expert_w": (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16),
        "side_stream": torch.cuda.Stream(weights["qkv_gate"].device),
    }
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
    torch.cuda.synchronize()
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
# (k, u, nv, minb, pf, blocks) per weight, m = 1, chosen by proj on the H100 (dev/g10.py)
G10_CFG = {"qkv_gate": (8192, 4, 4, 1, 1, 132), "pooler": (8192, 4, 4, 1, 1, 132), "o": (1024, 2, 4, 2, 1, 264),
           "unpool_slice": (4096, 2, 4, 1, 1, 132)}
_G10 = {}


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
            cases = " ".join("G10_CASE(1,%d,%d,%d,%d,%d)" % c[:5] for c in sorted(set(G10_CFG.values())))
            cases12 = " ".join("G12_CASE(%d,%d,%d,%d)" % ((G12_TOKENS,) + c[:3]) for c in sorted(set(G12_CFG.values())))
            src = _G10_SRC.replace("G10_CONFIGS", cases) + "\n" + _G12_SRC.replace("G12_CONFIGS", cases12)
            if UNPOOL_PDL:  # gemv10 may launch as a PDL secondary: its first W loads issue before griddepcontrol.wait
                anchor = "    for (int i = threadIdx.x; i < M * K8; i += 256 * NV) xs[i] = __ldg(x + i);"
                launch = ("        g10::gemv<M_, U_, V_, B_, K_, P_><<<blocks, 256 * V_, sm, s>>>((const uint4*)w.data_ptr(), "
                          "(const uint4*)x.data_ptr(), \\\n        (__nv_bfloat16*)out.data_ptr(), n); return; }")
                sig = "int64_t blocks) {"
                for x_ in (anchor, launch, sig):
                    assert src.count(x_) == 1, x_
                src = src.replace(anchor, '    asm volatile("griddepcontrol.wait;" ::: "memory");\n' + anchor)
                src = src.replace(sig, "int64_t blocks, int64_t pdl) {")
                src = src.replace(launch, (
                    "        cudaLaunchConfig_t lc = {}; lc.gridDim = blocks; lc.blockDim = 256 * V_; lc.dynamicSmemBytes = sm; \\\n"
                    "        lc.stream = s; cudaLaunchAttribute la[1]; la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \\\n"
                    "        la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0; \\\n"
                    "        cudaLaunchKernelEx(&lc, g10::gemv<M_, U_, V_, B_, K_, P_>, (const uint4*)w.data_ptr(), \\\n"
                    "                           (const uint4*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)n); return; }"))
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
            _G10["lib"] = load_inline(
                name="layer_g10_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                cpp_sources="void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, "
                            "int64_t, int64_t" + (", int64_t" if UNPOOL_PDL else "") + ");\n"
                            "void g12_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t" + (", int64_t" if UNPOOL_PDL else "") + ");",
                cuda_sources=src, functions=["g10_gemv", "g12_gemm"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
        except Exception as e:  # noqa: BLE001
            print("layer: gemv10/12 build failed, cuBLAS:", e)
    return _G10["lib"]


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


def _linear(x, w, key, state=None, sliding=False):
    """x @ w.T in BF16: gemv10 at 1 token (FP32 accumulate, no split-K reduce), gemv12 at 8 (FP32 accumulate and
    split-K partials), else cuBLAS."""
    lib = _G10.get("lib")
    if x.shape[0] == 1 and lib is not None and key in G10_CFG and x.shape[1] == G10_CFG[key][0]:
        out = torch.empty((1, w.shape[0]), dtype=torch.bfloat16, device=x.device)
        pdl = (int(key in GEMV_PDL_SECONDARY.get((1, sliding), ())),) if UNPOOL_PDL else ()
        lib.g10_gemv(w, x.contiguous(), out, *G10_CFG[key][1:], *pdl)
        return out
    if x.shape[0] == G12_TOKENS and state is not None and "g12" in state and key in G12_CFG:
        out = torch.empty((G12_TOKENS, w.shape[0]), dtype=torch.bfloat16, device=x.device)
        pdl = (int(key in GEMV_PDL_SECONDARY.get((G12_TOKENS, sliding), ())),) if UNPOOL_PDL else ()
        lib.g12_gemm(state["g12"][key], x.contiguous(), out, state["g12_ws"], state["g12_cnt"], w.shape[0] // 8,
                     *G12_CFG[key], *pdl)
        return out
    return F.linear(x, w)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    dense = state["dense"]
    dev = data["stream"].device
    qkv = _linear(data["attn_input"], dense["qkv_gate"], "qkv_gate", state)
    o = attn_kernel_fn({
        "qkv": qkv.view(batch, length, -1), "positions": data["positions"], "cache_loc": data["cache_loc"],
        "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
        "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
        "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": data["page_table"],
        "seq_lens": data["seq_lens"], "config": cfg}).view(tokens, hq * d)
    a = _linear(o, dense["o"], "o", state, cfg["window"] > 0)  # all-reduce: identity

    fp32_router = tokens <= ROUTER_FP32_MAX_TOKENS
    pdl = GLUE_PDL or (GEMV_PDL and tokens in PDL_TOKENS and _G10.get("lib") is not None)
    stream1 = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    h32 = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev) if fp32_router else stream1
    alpha, beta = state["res1"]
    _boundary[(tokens,)](a, data["stream"], alpha, beta, alpha, state["pre_ffn"], stream1, h, h32,
                         NB=HIDDEN, N=HIDDEN, BGAIN=False, H32=fp32_router, EPS_POST=POST_NORM_EPS, EPS=RMS_EPS,
                         PDL=pdl, num_warps=16, launch_pdl=pdl)
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream):
        topk_ids, local_ids, topk_weights = _route(state, h, h32, data["local_start"], tokens)
        # The experts' bucketing too, off the main stream's path.
        bucket = _moe_bucket(state, local_ids) if _use_moe(tokens) and TK2_MAX_TOKENS < tokens <= BUCKET_SIDE_MAX_TOKENS else None
    pooled = _linear(h, dense["pooler"], "pooler", state)
    p = torch.empty_like(pooled)
    _rmsnorm_gain[(tokens,)](pooled, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=pdl, num_warps=8,
                             launch_pdl=pdl)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for t in (topk_ids, local_ids, topk_weights) + (bucket or ()):
            t.record_stream(main_stream)
        for t in (h, h32):
            t.record_stream(side_stream)
    if _use_moe(tokens):
        moe = _experts_moe(p, topk_weights, local_ids, state, bucket)  # all-reduce: identity
    else:
        moe = (_experts_marlin if HAVE_SGL else _experts_local)(p, topk_weights, local_ids, state)
    sliced = tokens <= SLICED_UNPOOL_MAX_TOKENS
    f = _linear(moe, state["unpool_slice"], "unpool_slice", state) if sliced else F.linear(moe, state["unpool"])  # all-gather: identity (zeros)
    stream2 = torch.empty_like(stream1)
    hidden = torch.empty_like(h)
    alpha, beta = state["res2"]
    _boundary[(tokens,)](f, stream1, alpha, beta, state["post_ffn"], state["next_pre_attn"], stream2, hidden, hidden,
                         NB=UNPOOL_SLICE if sliced else HIDDEN, N=HIDDEN, BGAIN=True, H32=False,
                         EPS_POST=POST_NORM_EPS, EPS=RMS_EPS, PDL=pdl, num_warps=16, launch_pdl=pdl)
    return hidden, stream2, topk_ids
