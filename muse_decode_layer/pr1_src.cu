// pr1: the layer's pooler + router + routing at 1 token in ONE launch (PDL secondary of the boundary).
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
