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
# tokens -> split-K slabs: the pooler as a strided batched GEMM with FP32 out, summed in the rmsnorm (no cuBLAS
# split-K reduce kernel). H100 standalone m16/m32: 25.8/26.1 vs F.linear 28.0/28.1 us.
POOLER_BMM = {16: 2, 32: 2}
GEMV_PDL_SECONDARY = {(1, True): ("unpool_slice", "o"), (1, False): ("unpool_slice", "o"), (8, True): ("o", "unpool_slice"),
                      (8, False): ("unpool_slice",)}
LOCAL_FALLBACK = False  # no SGLang: also build the dequantizing reference experts (tests of the Marlin-side glue)  # PDL after a cuBLAS producer: no gain (H100 o GEMM + boundary 12.64 vs 12.74 us)


# ---- boundaries: post-norm of the branch, residual gate, pre-norm of the new stream ----

@triton.jit
def _boundary(br_ptr, s_ptr, alpha_ptr, beta_ptr, bgain_ptr, gain_ptr, s_out_ptr, h_ptr, h32_ptr,
              NB: tl.constexpr, N: tl.constexpr, BGAIN: tl.constexpr, H32: tl.constexpr,
              EPS_POST: tl.constexpr, EPS: tl.constexpr, PDL: tl.constexpr, pf_ptr=None, PF: tl.constexpr = 0,
              PF_NT: tl.constexpr = 512, RB: tl.constexpr = False, TRIG: tl.constexpr = False):
    """One row. b = branch (its first NB columns; the rest are zero), times (1 + w) with BGAIN;
    s' = alpha s + beta rmsnorm(b, EPS_POST); h = rmsnorm(s', EPS) (1 + w) in BF16 (and FP32 with H32).
    RB: the branch is FP32 (gemv19 output), rounded to BF16 first.
    PF > 0: the last PF programs instead L2-prefetch PF x PF_NT x 1 KB from pf_ptr (the router weight) and exit.
    TRIG: let the PDL dependent (the gemv19 pooler, which streams W before its griddepcontrol.wait) launch now."""
    if TRIG:
        gdc_launch_dependents()
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
    b = tl.load(br_ptr + row * NB + cols, mask=in_b, other=0.0)
    if RB:  # FP32 branch (gemv19): round to the BF16 the GEMM would return
        b = b.to(tl.bfloat16)
    b = b.to(tl.float32)
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
                  TRIG: tl.constexpr, XS: tl.constexpr = 0, ROUND: tl.constexpr = False, NS: tl.constexpr = 1,
                  SS: tl.constexpr = 0):
    """One row: rmsnorm(x) (1 + w), BF16 in and out. TRIG: let the dependent (moe_tk2 gemm13) launch now.
    XS: x row stride (0: N). ROUND: x is FP32, rounded to BF16 first (the pooler's output dtype).
    NS > 1: x is the sum of NS FP32 split-K partials SS elements apart (then rounded with ROUND)."""
    if TRIG:
        gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    gain = tl.load(gain_ptr + cols)
    if PDL:
        gdc_wait()
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
             TOPK: tl.constexpr, LOCAL: tl.constexpr, MULT: tl.constexpr, REPS: tl.constexpr,
             SKS: tl.constexpr = 0, ST: tl.constexpr = 0):
    """One token: scores = sigmoid(MULT logits); the top TOPK of scores + bias (lowest index on ties);
    weights = picked scores / (their sum + REPS); local ids = (id - local_start) mod E, -1 off this GPU."""
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


def _route(state, h, h32, local_start, tokens, router_pdl=False):
    bt, be, bk, ks, warps = _router_plan(tokens)
    dev = h.device
    fp32 = tokens <= ROUTER_FP32_MAX_TOKENS
    lib = _G10.get("lib") if fp32 and tokens in ROUTER3_CFG else None
    pr3 = _G10.get("lib") if not fp32 and tokens in PR3_CFG else None
    if lib is not None or pr3 is not None:
        ks = 1  # no split K: part = the logits
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
        out = _moe_kernel_fn({"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights,
                              "config": {"local_start": 0, "local_count": LOCAL_EXPERTS}}, st)
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
        "pool_router": torch.cat([weights["pooler"], hi, lo]).contiguous() if FUSED_ROUTER_TOKENS else None,
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
    _g19_prepare(state)
    _lt_prepare(state)
    torch.cuda.synchronize()
    _tune(state)
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
# above 8 tokens: pooler + SGLang's hi/lo BF16 router in one cuBLAS GEMM (FP32 out), no side stream. s7 H100: tie
# (d32 GEMM 30.8 + splitK reduce 2.1 + routing 3.4 vs pooler || router 38.3 us; v32 -2..-10, v16 full +5): the fused
# GEMM is the pooler + 3 us (router bytes at HBM rate), same as the overlap. Needs a better GEMM config to win (lt FP32 out).
FUSED_ROUTER_TOKENS = ()
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
# proj's gemv13 (m = 1) / gemv14 (m = 8) over losslessly compressed 12-bit BF16 weights (g13_src.cu / g14_src.cu =
# r-proj 3f173eb: bit-exact escape headers, inlined by build.py; _compress/_pack13/_pack14 below = proj's). Only where the weight compresses
# (else gemv10/12 as before). H200 (proj judge): pooler m1 17.8 -> 15.5, qkv m1 11.5 -> 11.0, pooler m8 18.8 -> 16.3,
# qkv m8 13.9 -> 13.0 us.
_G13_SRC = None  # @@G13@@
_G14_SRC = None  # @@G14@@
_G19_SRC = None  # @@G19@@
_GPR1_SRC = None  # @@GPR1@@
_GPR8_SRC = None  # @@GPR8@@
# proj's gemv19 (g19_src.cu = r-proj dev/gemv19.cu @6996d0c): m = 16/32 GEMM over the 12-bit weights, FP32 out zeroed
# in-kernel (epoch grid barrier on ctr[1]) + red.add split-K partials, launched as a PDL dependent of the previous
# kernel: its producer streams the first npre weight stages before griddepcontrol.wait. (key, T) -> (m, kc, ap, ns,
# npre, xb), proj's H100 picks (r-proj BF16_G19). Consumers round the FP32 out to BF16 (= the GEMM's BF16 output).
G19_CFG = {("pooler", 16): (16, 1024, 8, 3, 3, 1), ("unpool", 16): (16, 1024, 4, 4, 2, 2),
           # s15 (proj s15 handoff, with its escape-fixup rework): T=32 pooler/unpool, T=16 qkv (FP32 out + cast)
           ("pooler", 32): (32, 1024, 8, 2, 2, 1), ("unpool", 32): (32, 1024, 8, 2, 2, 1),
           ("qkv_gate", 16): (16, 1024, 4, 4, 4, 2),
           # s19: H200-tuned in prepare (TUNE): qkv T=32 and o T=16/32 on g19 (configs already built for other picks)
           ("qkv_gate", 32): (32, 1024, 8, 2, 2, 1), ("o", 16): (16, 1024, 4, 4, 2, 2), ("o", 32): (32, 1024, 8, 2, 2, 1)}
# proj's H200 diag (after a 3 us PDL predecessor): m16 pooler AP8 15.7 vs 19.0, unpool tie, o +0.8, every m32 loses
# (+1.5..5): m32 stays on cuBLAS / the bmm pooler. Was (H100 picks, s10 v35): pooler m16 (16, 1024, 4, 4, 2, 2),
# m32 (32, 1024, 4, 4, 2, 1), unpool the same, o m16 (16, 512, 4, 3, 2, 2).
G19_BUILD = [(16, 1024, 4, 4, 2, 2)]  # also compiled (A/B)
BOUNDARY_TRIG = True  # the first boundary triggers its dependents at start when the gemv19 pooler follows
G19_ON = {"pooler", "unpool", "qkv_gate", "o"}  # o: H100 tie (cuBLAS after attention)
G19_DEFAULT_OFF = {("qkv_gate", 32), ("o", 16), ("o", 32)}  # prepared, used only if TUNE picks them (H100: tie/+3 us)
G19_GRID = {("pooler", 16): 108, ("pooler", 32): 108}  # (key, T) -> blocks (default: all SMs): the pooler leaves SMs to the side-stream router (H100 d16, AP8 + BOUNDARY_TRIG: 104 213.2, 112 214.9, 120 221.4, 132 220.2; bmm 219.3 us)
_G19_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count if torch.cuda.is_available() else 132
_GPR3_SRC = None  # @@GPR3@@
KV_PF = True  # s9: the qkv gemv13 L2-prefetches the window's KV rows for csl (d1 sliding; H100 csl 6.1-7.3 -> ~4 us)
G13_CFG = {"qkv_gate": (2, 4, 1, 1, 132), "pooler": (2, 4, 1, 1, 132)}  # (u, nv, minb, pf, blocks), k = 8192
G14_CFG = {"qkv_gate": (2, 2, 4, 32, 1), "pooler": (8, 8, 4, 128, 1)}  # (kcu, wg, pf, bk, xp), m = 8
# s11: proj pr1 (r-proj dev/pr1.cu@149d115): T=1 pooler (gemv13 body) + FP32 router + routing top-8 in ONE launch,
# PDL secondary of the boundary (which triggers at start). (u, nv, minb, r, rb, blocks, npf). H100 d1 sliding 95.1 -> 92.5.
PR1_TOKENS = (1, 8)  # 8: proj pr8 (r-proj dev/pr8.cu@3be546b): gemv14 pooler blocks also take a router item, an extra
# finisher block sums the K slices + routes. H100 v1 158.1/172.4 -> 157.0/171.0 us (proj). cfg (kcu, wg, pf, bk, xp, npf)
PR8_CFG = {8: (8, 8, 4, 128, 1, 0)}
PR1_CFG = {1: (2, 4, 1, 4, 26, 132, 4)}
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


def _pack13(w):
    """gemv13 layout: flat byte plane, code words over 8 consecutive weights."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    return sm.to(torch.uint8).flatten().contiguous(), _nib(code.flatten().view(-1, 8)).contiguous(), hdr, scale


def _pack14(w, kc):
    """gemv14 layout: gemv12's atom order [ks][rg][atom][lane][8] for bytes and codes."""
    c = _compress(w)
    if c is None:
        return None
    code, sm, hdr, scale = c
    n, k = w.shape
    perm = lambda t: t.view(n // 8, 8, k // kc, kc // 32, 4, 8).permute(2, 0, 3, 1, 4, 5).contiguous()
    return perm(sm).to(torch.uint8), _nib(perm(code)).contiguous(), hdr, scale
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
                # s15: pdl == 2: first bulk-prefetch this thread group's W chunks into L2 (ptxas sinks the register W
                # loads above below the wait: SASS showed only the first stage before ACQBULK); x by __ldcg (no .nc)
                src = src.replace(anchor, '    if (pfl2)\n'
                                  '        for (long j = t; c + j * G < nch; j += 256)\n'
                                  '            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(W + (c + j * G) * (CE / 8)), "r"(CE * 2) : "memory");\n'
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
            src13 = (_G13_SRC.replace("G13_CONFIGS", " ".join("G13_CASE(8192,%d,%d,%d,%d)" % c[:4] for c in sorted(set(G13_CFG.values()))))
                     + "\n" + _G14_SRC.replace("G14_CONFIGS", " ".join("G14_CASE(%d,%d,%d,%d,%d)" % (G12_TOKENS, c[0], c[1], c[2], c[4])
                                                                      for c in sorted(set(G14_CFG.values())))))
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
                 "                                                       const char* __restrict__ pfv, int pfn, int pfrb) {"),
                ("        if (c + p * G < nch) load(p, c + p * G);\n",
                 "        if (c + p * G < nch) load(p, c + p * G);\n"
                 "    if (pfn > 0) {\n"
                 "        const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                 "        if (i < pfn) {\n"
                 "            const int pg = __ldg(pft + i);\n"
                 "            if (pg >= 0) {\n"
                 "                asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pfk + (long)pg * pfrb));\n"
                 "                asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pfv + (long)pg * pfrb));\n"
                 "            }\n        }\n    }\n"),
                ("int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks) {",
                 "int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks,\n"
                 "              torch::Tensor pft, torch::Tensor pfk, torch::Tensor pfv, int64_t pfn, int64_t pfrb) {"),
                ("(__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }",
                 "(__nv_bfloat16*)out.data_ptr(), n, (float)scale, \\\n"
                 "        (const int*)pft.data_ptr(), (const char*)pfk.data_ptr(), (const char*)pfv.data_ptr(), (int)pfn, (int)pfrb); return; }"),
            ]
            for x_, y_ in reps:
                assert src13.count(x_) == 1, x_
                src13 = src13.replace(x_, y_)
            src = src + "\n" + src13 + "\n" + _R3_SRC.replace("R3_CONFIGS", " ".join("R3_CASE(%d,%d,%d)" % c for c in sorted(set(ROUTER3_CFG.values()))))
            src = src + "\n" + _GPR3_SRC.replace("PR3_CONFIGS", " ".join("PR3_CASE(%d,%d,%d,%d,%d,%d)" % c for c in sorted(set(PR3_CFG.values()) | set(PR3_BUILD))))
            if G19_CFG and _G19_SRC is not None:
                src = src + "\n#define G19_ONLY2\n" + _G19_SRC.replace("G19_CONFIGS", " ".join(
                    "G19_CASE(%d,%d,%d,%d,1,%d)" % (c[0], c[1], c[2], c[3], c[5]) for c in sorted(set(G19_CFG.values()) | set(G19_BUILD))))
            else:
                src = src + "\nvoid g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}\n"
            src = src + "\n" + _G17_SRC.replace("G17_CONFIGS", " ".join("G17_CASE(%d,%d,%d)" % c for c in sorted(set(G17_CFG.values())))) + "\n" + _N2U_SRC
            if _GPR1_SRC is not None:
                src = src + "\n" + _GPR1_SRC.replace("PR1_CONFIGS", " ".join("PR1_CASE(%d,%d,%d,%d)" % c[:4] for c in sorted(set(PR1_CFG.values()))))
                if _GPR8_SRC is not None:
                    src = src + "\n" + _GPR8_SRC.replace("PR8_CONFIGS", " ".join("PR8_CASE(%d,%d,%d,%d)" % (c[0], c[1], c[2], c[4]) for c in sorted(set(PR8_CFG.values()))))
            else:
                src = src + "\nvoid pr1_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::vector<torch::Tensor>, double, double) {}\n"
            _G10["lib"] = load_inline(
                name="layer_g10_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                cpp_sources="void g10_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, "
                            "int64_t, int64_t" + (", int64_t" if UNPOOL_PDL else "") + ");\n"
                            "void g12_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t" + (", int64_t" if UNPOOL_PDL else "") + ");\n"
                            "void g13_gemv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                            "int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);\n"
                            "void g14_gemm(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, "
                            "torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                            "void r3_router(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);\n"
                            "void g17_gemv(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);\n"
                            "void pj_router3(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                            "void n2u_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "int64_t, int64_t, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, std::vector<torch::Tensor>, double, double);\n"
                            "void g19_gemm(torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);\n"
                            "void pr1_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, "
                            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                            "torch::Tensor, torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::vector<torch::Tensor>, double, double);\n"
                            + ("void pr8_launch(torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor, torch::Tensor, "
                               "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                               "torch::Tensor, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);" if _GPR1_SRC is not None and _GPR8_SRC is not None else ""),
                cuda_sources=src, functions=["g10_gemv", "g12_gemm", "g13_gemv", "g14_gemm", "r3_router", "g17_gemv", "n2u_launch", "pj_router3", "g19_gemm", "pr1_launch"]
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
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w + (size_t)r0 * (K / 8)), "r"(R * K * 2) : "memory");
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
                                          "      asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(upf + (long)r * upf_rb), \"r\"(upf_seg) : \"memory\");\n")
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
               "      asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(upf + (long)r * upf_rb), \"r\"(upf_seg) : \"memory\");\n")
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


def _moe__lib():  # = moe_src's _lib() + moe_set_upf
    global _moe__LIB
    if _moe__LIB is None:
        from torch.utils.cpp_extension import load_inline
        h = hashlib.sha256(_moe_CUDA_SRC.encode()).hexdigest()[:12]
        _moe__LIB = load_inline(name=f"moe_w4a16_{h}", cpp_sources=_moe__CPP, cuda_sources=_moe_CUDA_SRC,
                                functions=["moe_gemm13", "moe_gemm2", "moe_route_launch", "moe_norm_launch", "moe_sk_launch", "moe_tk2_launch", "moe_tk2f_launch", "moe_tk2fn_launch", "moe_tb_launch",
                                           "moe_gather_launch", "moe_tw_launch", "moe_twf_launch", "moe_twm_launch",
                                           "moe_route2_launch", "moe_act_launch", "moe_norm2_launch", "moe_norm3_launch", "moe_set_pdl", "moe_set_blk", "moe_set_sa",
                                           "moe_set_upf"],
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
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w + (size_t)r0 * (K / 8)), "r"(R * K * 2) : "memory");
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
'''

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


def _pack19(c, kc, ap):
    """proj's pack19 (r-proj dev/g19_lib.py): _compress output -> stage-major uint8 [stages, 8 * ap * 768]."""
    code, sm, hdr, scale = c
    n, k = code.shape
    perm = lambda t: t.view(n // 16, 2, 8, k // kc, kc // 32, 2, 2, 4, 2).permute(3, 0, 4, 2, 7, 5, 6, 1, 8).contiguous()  # noqa: E731
    S = perm(sm).to(torch.uint8).reshape(k // kc, n // 16, kc // 32, 32, 16)
    sh = torch.tensor([4 * (i // 2) + 16 * (i % 2) for i in range(8)], device=code.device)
    N = (perm(code).reshape(k // kc, n // 16, kc // 32, 32, 2, 8).long() << sh).sum(-1)
    N = torch.where(N >= 2 ** 31, N - 2 ** 32, N).int().contiguous()
    ks, ntr, spk = k // kc, n // 128, kc // 32 // ap
    S = S.view(ks, ntr, 8, spk, ap * 512).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, 8 * ap * 512)
    N = N.view(torch.uint8).view(ks, ntr, 8, spk, ap * 256).permute(0, 1, 3, 2, 4).reshape(ks * ntr * spk, 8 * ap * 256)
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
    del comp


# (s18/s19) Choices where the H100 and the judge's H200 may disagree, tuned in prepare(): the whole layer is timed
# (CUDA graph replays, cold L2, interleaved, median) on synthetic sliding decode inputs (shapes only, random data) with
# each candidate option set; the fastest set per T is kept in state["tune"][T]. Options (defaults = no entry):
# g19:<key> (bool) gemv19 for that GEMM; pr1_npf / pr1_rb (int) pr1's L2-prefetched pooler chunks / router blocks;
# kv_pf (bool) KV_PF; pr1 (bool) pr1/pr8 at that T. A candidate that fails to run is skipped.
# s18: the T=16/32 pooler grid (108..132): the H200 picks 108 too (check ea9421fa), so it is fixed now.
TUNE = {1: ({}, {"pr1_npf": 0}, {"pr1_npf": 8}, {"kv_pf": False}, {"pr1_rb": 20}, {"pr1_rb": 32}),
        16: ({}, {"g19:qkv_gate": False}, {"g19:o": True}),
        32: ({}, {"g19:qkv_gate": True}, {"g19:o": True}, {"g19:qkv_gate": True, "g19:o": True}),
        8: ({}, {"pr1": False})}
TUNE_REPS = 40
TUNE_SHAPE = {1: (1, 1), 8: (1, 8), 16: (16, 1), 32: (32, 1), 128: (16, 8), 256: (32, 8)}  # T -> (batch, q_len)


def _opt(state, tokens, name, default):
    return state.get("tune", {}).get(tokens, {}).get(name, default)


def _synth_input(batch, dev, ctx=3000, q_len=1):
    """A sliding-window decode step / verify block (page 1) of random data with the scored inputs' shapes and dtypes."""
    g = torch.Generator(device=dev).manual_seed(1234)
    tokens = batch * q_len
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
    maxp = ctx + q_len
    pt = torch.randperm(batch * maxp, generator=g, device=dev).to(torch.int32).view(batch, maxp)
    pos = (seq.long()[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    stream = torch.randn((tokens, HIDDEN), generator=g, device=dev)
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
                       "num_kv_heads": 2, "head_dim": 64, "page_size": 1, "window": 2048, "rope_theta": 500000.0,
                       "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0,
                       "n_experts": N_EXPERTS, "top_k": TOP_K, "local_experts": LOCAL_EXPERTS}}


def _tune(state):
    state["tune"] = {}
    if not TUNE:
        return
    dev = state["unpool_slice"].device
    try:
        flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        for tokens, cands in TUNE.items():
            data = _synth_input(TUNE_SHAPE[tokens][0], dev, q_len=TUNE_SHAPE[tokens][1])
            graphs, ok = [], []
            for opts in cands:
                state["tune"][tokens] = opts
                fn = lambda: kernel_fn(data, state)  # noqa: E731
                try:
                    fn()
                except Exception as ex:  # noqa: BLE001  (host-side checks only: a CUDA fault would raise below)
                    print(f"layer tune T={tokens} {opts} skipped ({type(ex).__name__}: {ex})", flush=True)
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
            best = min(idx, key=lambda i: med[i])
            state["tune"][tokens] = cands[best]
            print(f"layer tune T={tokens}: " + " | ".join(f"{cands[i] or 'default'} {med[i] * 1e3:.1f}us" for i in idx)
                  + f" -> {cands[best] or 'default'}", flush=True)
            del graphs, data
        del flush
    except Exception as ex:  # keep the defaults
        print(f"layer tune failed ({type(ex).__name__}: {ex}); using defaults", flush=True)
        state["tune"] = {}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _g19(x, key, state):
    """x @ W.T (key's weight) in FP32 by gemv19 (a PDL dependent of the previous kernel), or None."""
    m = x.shape[0]
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
    _G10["lib"].g19_gemm(e[0], e[1], e[2], x.contiguous(), out, ctr, c[1], c[2], c[3], 1, grid, 0, 2, c[4], c[5])
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
    state["pr1_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)  # pr1's router ticket (self-resetting)
    state["n2u_cnt"] = torch.zeros(1, dtype=torch.int32, device=dev)  # n2u's last-block counter (self-resetting)
    state["g13"] = {key: pk for key in G13_CFG if (pk := _pack13(w[key])) is not None}
    state["g14"] = {key: pk for key, c in G14_CFG.items() if (pk := _pack14(w[key], 1024 * c[0])) is not None}


def _linear(x, w, key, state=None, sliding=False, kv_pf=None):
    """x @ w.T in BF16: gemv10 at 1 token (FP32 accumulate, no split-K reduce), gemv12 at 8 (FP32 accumulate and
    split-K partials), else cuBLAS."""
    lib = _G10.get("lib")
    if state is not None and x.shape[0] in (1, G12_TOKENS):
        pk = state.get("g13" if x.shape[0] == 1 else "g14", {}).get(key)
        if pk is not None and key not in GEMV_PDL_SECONDARY.get((x.shape[0], sliding), ()):
            out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
            if x.shape[0] == 1:
                pf = kv_pf if kv_pf is not None else (x, x, x, 0, 0)
                lib.g13_gemv(pk[0], pk[1], pk[2], x.contiguous(), out, pk[3], *G13_CFG[key], *pf)
            else:
                lib.g14_gemm(pk[0], pk[1], pk[2], pk[3], x.contiguous(), out, state["g12_ws"], state["g12_cnt"],
                             w.shape[0] // 8, *G14_CFG[key])
            return out
    if lib is not None and key in G17_KEYS and x.shape[0] in G17_CFG:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
        lib.g17_gemv(w, x.contiguous(), out, *G17_CFG[x.shape[0]], int(key in GEMV_PDL_SECONDARY.get((x.shape[0], sliding), ())))
        return out
    if x.shape[0] == 1 and lib is not None and key in G10_CFG and x.shape[1] == G10_CFG[key][0]:
        out = torch.empty((1, w.shape[0]), dtype=torch.bfloat16, device=x.device)
        pdl = ((key in GEMV_PDL_SECONDARY.get((1, sliding), ())) * (1 + (key in G10_PFL2.get(sliding, ()))),) if UNPOOL_PDL else ()
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


def _experts_n2u(x, topk_weights, local_ids, state, bnd=()):
    """T = 1: moe_tk2f (as moe_src's kernel_fn) -> n2u (moe_norm2 + unpool slice): the unpool slice output."""
    st = state["moe"]
    lib = st["lib"]
    dev = x.device
    meta = torch.empty(100, dtype=torch.int32, device=dev)
    sroutes = torch.empty(TOP_K, dtype=torch.int32, device=dev)
    inter = torch.empty((TOP_K, _moe_I), dtype=torch.bfloat16, device=dev)
    inv = torch.empty(TOP_K, dtype=torch.int32, device=dev)
    y32 = torch.empty((TOP_K, D_EXPERT), dtype=torch.float32, device=dev)
    ids = local_ids.contiguous()
    slot = N2U_SLOT and _moe_TK2F == 204 and N2U_CFG[0] == 8
    lib.moe_tk2f_launch(st["w13q"], st["w13s2"], st["w2q"], st["w2s2"], st["lut"], st["gs13"], st["gs2"],
                        x.contiguous(), ids, 0, LOCAL_EXPERTS, meta, sroutes, inv, inter, y32, st["cnt"],
                        (84204 if slot else _moe_TK2F) + (100000 if _moe_TK2SA and st["sa"] else 0))
    f = torch.empty((1, UNPOOL_SLICE), dtype=torch.bfloat16, device=dev)
    _G10["lib"].n2u_launch(y32, inv, ids, topk_weights.float().contiguous(), st["w1"], st["gs2"], 0, LOCAL_EXPERTS,
                           POST_NORM_EPS, state["unpool_slice"], f, st["cnt"], N2U_CFG[0] + (1000 + 2000 * N2U_PFW) * slot, N2U_CFG[1], list(bnd), POST_NORM_EPS, RMS_EPS)
    return f


_moe_TWM_ORDER_M = {}  # s15: moe's H200 proxy (s16): order 76 at T=256 (r-moe 499bde6 default) loses ~3.6 us at v32


def _tail(moe, stream1, h, tokens, state, pdl, f=None):
    """unpool -> second boundary: (next attention input, stream)."""
    sliced = tokens <= SLICED_UNPOOL_MAX_TOKENS
    if f is None:
        f = _linear(moe, state["unpool_slice"], "unpool_slice", state) if sliced else _g19(moe, "unpool", state)
        if f is None:
            f = F.linear(moe, state["unpool"])  # all-gather: identity (zeros)
    stream2 = torch.empty_like(stream1)
    hidden = torch.empty_like(h)
    alpha, beta = state["res2"]
    _boundary[(tokens,)](f, stream1, alpha, beta, state["post_ffn"], state["next_pre_attn"], stream2, hidden, hidden,
                         NB=UNPOOL_SLICE if sliced else HIDDEN, N=HIDDEN, BGAIN=True, H32=False,
                         EPS_POST=POST_NORM_EPS, EPS=RMS_EPS, PDL=pdl, RB=f.dtype == torch.float32, num_warps=16,
                         launch_pdl=pdl)
    return hidden, stream2


def kernel_fn(data, state):
    cfg = data["config"]
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
    qkv = _g19(data["attn_input"], "qkv_gate", state)  # s15: T=16 (FP32 out, rounded to the GEMM's BF16)
    qkv = _linear(data["attn_input"], dense["qkv_gate"], "qkv_gate", state, kv_pf=kv_pf) if qkv is None else qkv.to(torch.bfloat16)
    o = attn_kernel_fn({
        "qkv": qkv.view(batch, length, -1), "positions": data["positions"], "cache_loc": data["cache_loc"],
        "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
        "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
        "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": data["page_table"],
        "seq_lens": data["seq_lens"], "config": cfg}).view(tokens, hq * d)
    a = _g19(o, "o", state)
    if a is None:
        a = _linear(o, dense["o"], "o", state, cfg["window"] > 0)  # all-reduce: identity

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
    bnd = pr1 and PR1_BND and tokens == 1 and a.dtype == torch.bfloat16
    if not bnd:
      _boundary[(tokens + npf,)](a, data["stream"], alpha, beta, alpha, state["pre_ffn"], stream1, h, h32,
                               NB=HIDDEN, N=HIDDEN, BGAIN=False, H32=fp32_router, EPS_POST=POST_NORM_EPS, EPS=RMS_EPS,
                               PDL=pdl, pf_ptr=state["router"], PF=npf, RB=a.dtype == torch.float32,
                               TRIG=pr1 or (BOUNDARY_TRIG and ("pooler", tokens) in state.get("g19", {}) and "pooler" in G19_ON), num_warps=16,
                               launch_pdl=pdl)
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
    if tokens in FUSED_ROUTER_TOKENS and not fp32_router:
        # SGLang's router above 8 tokens = BF16 h against the BF16 hi/lo split of the FP32 weight (FP32 accumulation):
        # the same h as the pooler, so one GEMM [pooler; hi; lo] with FP32 output; the routing sums the hi/lo logits,
        # the rmsnorm rounds the pooler part to BF16 (what the BF16-out pooler GEMM returns). No side stream.
        pr = torch.mm(h, state["pool_router"].t(), out_dtype=torch.float32)
        ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        lids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
        wts = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
        _routing[(tokens,)](pr[:, D_EXPERT:], state["router_bias"], data["local_start"], ids, lids, wts, tokens,
                            E=N_EXPERTS, KS=2, TOPK=TOP_K, LOCAL=LOCAL_EXPERTS, MULT=ROUTER_MULTIPLIER, REPS=ROUTE_EPS,
                            SKS=N_EXPERTS, ST=D_EXPERT + 2 * N_EXPERTS, num_warps=2)
        p = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
        _rmsnorm_gain[(tokens,)](pr, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=False, TRIG=False,
                                 XS=D_EXPERT + 2 * N_EXPERTS, ROUND=True, num_warps=8)
        moe = _experts_moe(p, wts, lids, state) if _use_moe(tokens) else (
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
                                   ROUTER_MULTIPLIER, ROUTE_EPS, LOCAL_EXPERTS, *PR1_CFG[tokens][:4], _opt(state, tokens, "pr1_rb", PR1_CFG[tokens][4]),
                                   PR1_CFG[tokens][5], _opt(state, tokens, "pr1_npf", PR1_CFG[tokens][6]), 1,
                                   [a, data["stream"], alpha, beta, state["pre_ffn"], stream1, h] if bnd else [],
                                   POST_NORM_EPS, RMS_EPS)
        else:
            pk = state["g14"]["pooler"]
            part = torch.empty(8 * tokens * N_EXPERTS, dtype=torch.float32, device=dev)
            _G10["lib"].pr8_launch(pk[0], pk[1], pk[2], pk[3], h, pooled, state["router"], h32, part, logits, state["pr1_cnt"],
                                   state["router_bias"], data["local_start"], topk_ids, local_ids, topk_weights,
                                   ROUTER_MULTIPLIER, ROUTE_EPS, LOCAL_EXPERTS, *PR8_CFG[tokens], 1)
    else:
        side_stream.wait_stream(main_stream)
        with torch.cuda.stream(side_stream):
            topk_ids, local_ids, topk_weights = _route(state, h, h32, data["local_start"], tokens)
            # The experts' bucketing too, off the main stream's path.
            bucket = _moe_bucket(state, local_ids) if _use_moe(tokens) and TK2_MAX_TOKENS < tokens <= BUCKET_SIDE_MAX_TOKENS else None
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
    elif tokens in POOLER_BMM:
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
    _rmsnorm_gain[(tokens,)](pooled, state["pre_expert"], p, N=D_EXPERT, EPS=RMS_EPS, PDL=pdl and not tk2_pdl,
                             TRIG=tk2_pdl, ROUND=pooled.dtype == torch.float32, NS=nsp, SS=tokens * D_EXPERT, num_warps=8,
                             launch_pdl=pdl and not tk2_pdl)
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
    if _use_moe(tokens):
        moe = _experts_moe(p, topk_weights, local_ids, state, bucket)  # all-reduce: identity
    else:
        moe = (_experts_marlin if HAVE_SGL else _experts_local)(p, topk_weights, local_ids, state)
    return _tail(moe, stream1, h, tokens, state, pdl) + (topk_ids,)
