// pr8: the layer's T=8 pooler (gemv14 body, S = 1: whole-K rows, x[8, 8192] staged in smem) + FP32 router
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
    int* rel;  // layer n15 IDF8: != nullptr: the finisher st.release's *rel = 1 once ids/lids/wts are written
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
    if (a.rel) {  // layer n15 IDF8: every routing store, then thread 0's release (cumulative after the barrier)
        __threadfence();
        __syncthreads();
        if (t == 0) asm volatile("st.release.gpu.global.s32 [%0], 1;" :: "l"(a.rel) : "memory");
    }
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
                int64_t npf, int64_t pdl, int64_t rel) {
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
                wts.data_ptr<float>(), (float)mult, (float)reps, (int)local, (int)npf, (int*)rel};
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
