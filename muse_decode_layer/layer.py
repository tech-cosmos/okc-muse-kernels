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

# @@ATTENTION@@
# @@MOE@@


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
              EPS_POST: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr, pf_ptr=None, PF: tl.constexpr = 0,
              PF_NT: tl.constexpr = 512):
    """One row. b = branch (its first NB columns; the rest are zero), times (1 + w) with BGAIN;
    s' = alpha s + beta rmsnorm(b, EPS_POST); h = rmsnorm(s', EPS) (1 + w) in BF16 (and FP32 with H32).
    PF > 0: the last PF programs instead L2-prefetch PF x PF_NT x 1 KB from pf_ptr (the router weight) and exit."""
    if PF > 0:
        if tl.program_id(0) >= tl.num_programs(0) - PF:
            i = (tl.program_id(0) - (tl.num_programs(0) - PF)) * PF_NT + tl.arange(0, PF_NT)
            tl.inline_asm_elementwise("cp.async.bulk.prefetch.L2.global [$1], 1024; mov.u32 $0, 0;", "=r,l",
                                      [pf_ptr + i.to(tl.int64) * 256], dtype=tl.int32, is_pure=False, pack=1)
            return
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
def _rmsnorm_gain(x_ptr, gain_ptr, out_ptr, N: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr,
                  TRIG: tl.constexpr):
    """One row: rmsnorm(x) (1 + w), BF16 in and out. TRIG: let the dependent (moe_tk2 gemm13) launch now."""
    if TRIG:
        gdc_launch_dependents()
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


def _route(state, h, h32, local_start, tokens, router_pdl=False):
    bt, be, bk, ks, warps = _router_plan(tokens)
    dev = h.device
    part = torch.empty((ks, tokens, N_EXPERTS), dtype=torch.float32, device=dev)
    fp32 = tokens <= ROUTER_FP32_MAX_TOKENS
    if fp32:
        _router_fma[(N_EXPERTS // be, ks)](h32, state["router"], part, tokens, K=HIDDEN, E=N_EXPERTS, BT=bt, BE=be,
                                           BK=bk, KS=ks, num_warps=warps, launch_pdl=router_pdl)
    else:
        _router_split[(N_EXPERTS // be, ks, triton.cdiv(tokens, bt))](
            h, state["router_hi"], state["router_lo"], part, tokens, K=HIDDEN, E=N_EXPERTS, BT=bt, BE=be, BK=bk,
            KS=ks, HILO=True, num_warps=warps, launch_pdl=router_pdl)
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
TW_TOKENS = ()  # moe_tw at decode T: dead (s5 H100 d16 +14, d32 +29 us vs tk2)
MARLIN_TOKENS = ()  # moe 6a1cda0 at T=32 beat Marlin on the judge: decode 32 sliding 218.3 (fc3e60d0) vs 225.0 (440361a9)


def _use_moe(tokens):
    return not HAVE_SGL or tokens not in MARLIN_TOKENS


def _moe_bucket(state, local_ids):
    """moe_src's route kernel: this GPU's routes bucketed by expert (meta, sorted route ids)."""
    dev = local_ids.device
    meta = torch.empty(100, dtype=torch.int32, device=dev)
    sroutes = torch.empty(local_ids.numel(), dtype=torch.int32, device=dev)
    state["moe"]["lib"].moe_route_launch(local_ids, 0, LOCAL_EXPERTS, meta, sroutes)
    return meta, sroutes


def _experts_moe(x, topk_weights, local_ids, state, bucket=None):
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
        "hi_stream": torch.cuda.Stream(weights["qkv_gate"].device, priority=-1),
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
    _lt_prepare(state)
    torch.cuda.synchronize()
    return state


# ---- BF16 GEMV at 1 token: the proj agent's gemv10 (g10_src.cu = r-proj d894f4e, inlined by build.py) ----

_G10_SRC = None  # @@G10@@
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
_G12_SRC = None  # @@G12@@
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


# ---- cuBLASLt with explicit algo configs (lt_src.cu = r-proj dev/lt.cu @f7b1c8b) at T >= 16 ----
# (layer, tokens): cfg (algo, tile, stages, split k, reduction, swizzle, custom option, cluster), proj's prepare-time
# picks on the judge's H200 (r-proj profiles/submission.22109e72563f4cc3.log), us vs F.linear: pooler m16 22.0/24.3,
# m32 22.2/24.6, m128 26.0/29.7, m256 28.2/34.4; qkv m128 18.9/20.0, m256 21.0/22.9. Not pooler m256: beside the
# router (side stream) in the layer it is 60.7 vs F.linear 49.2 us (H100, verify 32 sliding).
# Judge check f8e06892 (pooler m16/m32 too): d16 +0.7%, d32 -2.6% vs v15 (noise ~2%): m16/m32 not used.
LT_CFG = {("pooler", 128): (66, 143, 35, 1, 0, 0, 0, 2),
          ("qkv_gate", 128): (66, 92, 35, 1, 0, 0, 0, 5), ("qkv_gate", 256): (66, 124, 35, 1, 0, 0, 0, 2)}
_LT_SRC = None  # @@LT@@
_LT = {}


def _lt_prepare(state):
    """Build the cuBLASLt wrapper; keep the configs that launch on this GPU and match F.linear."""
    state["lt_ok"] = {}
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
    lib = _LT["lib"]
    if lib is None:
        return
    w = state["dense"]
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
    cfg = state.get("lt_ok", {}).get((key, x.shape[0])) if state is not None else None
    if cfg is not None:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        _LT["lib"].lt_gemm(w, x.contiguous(), out, state["lt_ws"], list(cfg))
        return out
    return F.linear(x, w)


def _tail(moe, stream1, h, tokens, state, pdl):
    """unpool -> second boundary: (next attention input, stream)."""
    sliced = tokens <= SLICED_UNPOOL_MAX_TOKENS
    f = _linear(moe, state["unpool_slice"], "unpool_slice", state) if sliced else F.linear(moe, state["unpool"])  # all-gather: identity (zeros)
    stream2 = torch.empty_like(stream1)
    hidden = torch.empty_like(h)
    alpha, beta = state["res2"]
    _boundary[(tokens,)](f, stream1, alpha, beta, state["post_ffn"], state["next_pre_attn"], stream2, hidden, hidden,
                         NB=UNPOOL_SLICE if sliced else HIDDEN, N=HIDDEN, BGAIN=True, H32=False,
                         EPS_POST=POST_NORM_EPS, EPS=RMS_EPS, PDL=pdl, num_warps=16, launch_pdl=pdl)
    return hidden, stream2


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
    npf = (state["router"].numel() * 4) // (512 * 1024) if tokens in ROUTER_L2_PF_TOKENS and fp32_router else 0
    _boundary[(tokens + npf,)](a, data["stream"], alpha, beta, alpha, state["pre_ffn"], stream1, h, h32,
                               NB=HIDDEN, N=HIDDEN, BGAIN=False, H32=fp32_router, EPS_POST=POST_NORM_EPS, EPS=RMS_EPS,
                               PDL=pdl, pf_ptr=state["router"], PF=npf, num_warps=16, launch_pdl=pdl)
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
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream):
        topk_ids, local_ids, topk_weights = _route(state, h, h32, data["local_start"], tokens)
        # The experts' bucketing too, off the main stream's path.
        bucket = _moe_bucket(state, local_ids) if _use_moe(tokens) and TK2_MAX_TOKENS < tokens <= BUCKET_SIDE_MAX_TOKENS else None
    if tokens in POOLER_HI_PRIORITY:
        # the pooler on a high-priority stream: its CTAs go ahead of the router's remaining waves
        hi = state["hi_stream"]
        hi.wait_stream(main_stream)
        with torch.cuda.stream(hi):
            pooled = _linear(h, dense["pooler"], "pooler", state)
        main_stream.wait_stream(hi)
        if not torch.cuda.is_current_stream_capturing():
            h.record_stream(hi)
            pooled.record_stream(main_stream)
    else:
        pooled = _linear(h, dense["pooler"], "pooler", state)
    p = torch.empty_like(pooled)
    tk2_pdl = TK2_PDL13 and tokens <= TK2_MAX_TOKENS and _use_moe(tokens)
    if tk2_pdl:
        # moe_tk2 gemm13 reads the routing before its griddepcontrol.wait: join first, so that its only
        # (programmatic) predecessor is this rmsnorm, which has full edges from the pooler and the routing.
        main_stream.wait_stream(side_stream)
    _rmsnorm_gain[(tokens,)](pooled, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=pdl and not tk2_pdl,
                             TRIG=tk2_pdl, num_warps=8, launch_pdl=pdl and not tk2_pdl)
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
    return _tail(moe, stream1, h, tokens, state, pdl) + (topk_ids,)
