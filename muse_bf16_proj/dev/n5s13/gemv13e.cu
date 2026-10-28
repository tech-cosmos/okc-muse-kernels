// gemv13: gemv10 (grid-stride GEMV, m = 1) over losslessly compressed BF16 weights: 12 bits per weight.
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
#ifdef E11
#define PH 0x1p-21f
#else
#define PH 0x1p-15f
#endif
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

#ifdef E11
__device__ __forceinline__ unsigned l3(unsigned a, unsigned b, unsigned c) { return lop_aob(a, b, c); }
// 11-bit: pair q from exponent bits E (bits 7..9 / 23..25 + const 120) and sign|mantissa bytes
__device__ __forceinline__ void e11pair(unsigned sw, unsigned sel, unsigned E, float* f) {
    const unsigned q = lop_aob(prmt(sw, sel), 0x807f807fu, E);
    f[0] = __uint_as_float(q << 16); f[1] = __uint_as_float(q & 0xffff0000u);
}
__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* xp) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; i += 4) *reinterpret_cast<float4*>(x + i) = *reinterpret_cast<const float4*>(xp + i);
    float f[16];
    const unsigned w = nb.x, T = prmt(nb.y, 0x4140u);
    const unsigned M = 0x03800380u, K = 0x3C003C00u;
    e11pair(s.x, 0x9180u, l3(w << 7, M, K), f + 0);
    e11pair(s.x, 0xB3A2u, l3(w << 4, M, K), f + 2);
    e11pair(s.y, 0x9180u, l3(w << 1, M, K), f + 4);
    e11pair(s.y, 0xB3A2u, l3(w >> 2, M, K), f + 6);
    e11pair(s.z, 0x9180u, l3(w >> 5, M, K), f + 8);
    e11pair(s.z, 0xB3A2u, l3(w >> 8, 0x00800080u, l3(T << 2, 0x03000300u, K)), f + 10);
    e11pair(s.w, 0x9180u, l3(T << 7, M, K), f + 12);
    e11pair(s.w, 0xB3A2u, l3(T << 4, M, K), f + 14);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i += 2) { a0 = fmaf(f[i], x[i], a0); a1 = fmaf(f[i + 1], x[i + 1], a1); }
    return a0 + a1;
}
#else
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
#endif


#if defined(E11) && !defined(NOESC)
// soft escapes (code 7): exponent from the warp unit's nibble record (16 B = 32 nibbles, in flat order); the decoded
// placeholder (code 7) is corrected by + x * (true - placeholder), both in decoded scale.
__device__ __forceinline__ float fix16(const uint4& s, const uint2& nb, const uint4& sec, const float* xp, int lane, float acc) {
    const unsigned w = nb.x, T = prmt(nb.y, 0x4140u);
    const unsigned a = w & (w >> 1) & (w >> 2), b = T & (T >> 1) & (T >> 2);
    // 16-bit escape mask in weight order
    unsigned m = 0;
#pragma unroll
    for (int p = 0; p < 5; ++p) m |= ((a >> (3 * p)) & 1u) << (2 * p) | ((a >> (16 + 3 * p)) & 1u) << (2 * p + 1);
    m |= ((w >> 15) & (T >> 6) & (T >> 7) & 1u) << 10 | ((w >> 31) & (T >> 22) & (T >> 23) & 1u) << 11;
    m |= (b & 1u) << 12 | ((b >> 16) & 1u) << 13 | ((b >> 3) & 1u) << 14 | ((b >> 19) & 1u) << 15;
    const int cnt = __popc(m);
    int inc = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += v; }
    int q = inc - cnt;
#ifdef NOLOOP
    return acc + (q > 1000 || sec.x == 0x12345678u ? 1.f : 0.f);
#endif
    while (m) {
        const int j = __ffs(m) - 1;
        m &= m - 1;
        const unsigned sw = j < 4 ? s.x : j < 8 ? s.y : j < 12 ? s.z : s.w;
        const unsigned by = prmt(sw, (unsigned)(j & 3));
        const unsigned nw = q < 8 ? sec.x : q < 16 ? sec.y : q < 24 ? sec.z : sec.w;
        const int v = (nw >> ((q & 7) * 4)) & 15;
        ++q;
        const int d = v < 2 ? 7 + v : 1 - v;  // exponent - lo
        const unsigned sm = (by & 0x80u) << 24 | (by & 0x7fu) << 16;
        const float t = __uint_as_float(sm | (unsigned)(120 + d) << 23), ph = __uint_as_float(sm | 127u << 23);
        acc = fmaf(xp[j], t - ph, acc);
    }
    return acc;
}
#endif
// VT: threads per virtual block (256 or 512): a chunk is U * VT * 16 weights. 512 halves the rows per round of a
// 2304-row qkv (4.36 -> 5 rounds of whole rows at 528 vbs: a 13% tail; 8.73 -> 9 half-size rounds at 264 vbs: 3%).
template <int U, int NV, int MINB, int KK, int PF, int VT = 256>
__global__ void __launch_bounds__(VT * NV, MINB) gemv(const uint4* __restrict__ S, const uint2* __restrict__ N,
                                                       const float2* __restrict__ H, const uint4* __restrict__ SEC, const __nv_bfloat16* __restrict__ x,
                                                       __nv_bfloat16* __restrict__ out, int n, float scale) {
    constexpr int CE = U * VT * 16, RPC = CE / KK, WV = VT / 32;
    static_assert(CE % KK == 0 && KK % 512 == 0 && (VT == 256 || VT == 512), "chunk must hold whole rows");
    const int t = threadIdx.x % VT, vg = threadIdx.x / VT, lane = t & 31, wp = t >> 5;
    const int nch = n / RPC;
    __shared__ float red_[NV][2][U * WV];
    __shared__ __align__(16) float xs[KK / 16 * 20];  // 16-float groups padded to 20: no bank conflicts
    __shared__ float2 hs_[NV][MAXC * RPC * ESC];  // this vb's escape headers: chunk j = j-th of c, c + G, ...
    float (*red)[U * WV] = red_[vg];
    const int G = gridDim.x * NV;
    int c = blockIdx.x * NV + vg;
    uint4 s[PF][U];
    uint2 nb[PF][U];
#if defined(E11) && !defined(NOESC)
    uint4 sc[PF][U];
#endif
    // escapes: the row's reducing thread adds x[col] * (value - 2^-15) per header entry to the row's sum
    auto load = [&](int p, int ch) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long i = (long)ch * (CE / 16) + u * VT + t;
            s[p][u] = ld4(S + i);
#ifdef E11
            nb[p][u].x = __ldg(reinterpret_cast<const unsigned*>(N) + i);
            nb[p][u].y = __ldg(reinterpret_cast<const unsigned short*>(reinterpret_cast<const unsigned*>(N) + (long)n * KK / 16) + i);
#else
            nb[p][u] = ld2(N + i);
#endif
#if defined(E11) && !defined(NOESC)
            sc[p][u] = __ldg(SEC + (i >> 5));
#endif
        }
    };
    // x and the escape headers first: they must not queue behind the W prologue
    constexpr int XV = (KK / 8 + VT * NV - 1) / (VT * NV), HV = (MAXC * RPC * ESC + VT - 1) / VT;
    uint4 xv[XV];
    float2 hv[HV];
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * VT * NV;
        if (i < KK / 8) xv[q] = __ldg(reinterpret_cast<const uint4*>(x) + i);
    }
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * VT, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hv[q] = __ldg(H + (long)ch * RPC * ESC + i % (RPC * ESC));
    }
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
#pragma unroll
    for (int q = 0; q < HV; ++q) {
        const int i = t + q * VT, ch = c + (i / (RPC * ESC)) * G;
        if (i < MAXC * RPC * ESC && ch < nch) hs_[vg][i] = hv[q];
    }
#pragma unroll
    for (int q = 0; q < XV; ++q) {
        const int i = threadIdx.x + q * VT * NV;
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
                    acc[u] = dot16(s[p][u], nb[p][u], xs + XI((u * VT * 16 + t * 16) % KK));
#if defined(E11) && !defined(NOESC)
                    acc[u] = fix16(s[p][u], nb[p][u], sc[p][u], xs + XI((u * VT * 16 + t * 16) % KK), lane, acc[u]);
#endif
                }
                const int cn = cc + PF * G;
                if (cn < nch) load(p, cn);
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    float v = acc[u];
#pragma unroll
                    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                    if (lane == 0) red[par][u * WV + wp] = v;
                }
                asm volatile("bar.sync %0, %1;" :: "r"(vg + 1), "n"(VT));
                if (t < RPC) {
                    float sum = 0.f;
#pragma unroll
                    for (int u = 0; u < U; ++u)
#pragma unroll
                        for (int w = 0; w < WV; ++w)
                            if ((u * VT * 16 + w * 512) / KK == t) sum += red[par][u * WV + w];
#ifndef NOESC
                    // escapes on the row's full sum, two fmas each: exact when the row holds only that weight (probes)
                    const float2* he = hs_[vg] + ((j + p) * RPC + t) * ESC;
                    for (int e = 0; e < ESC; ++e) {
                        const float2 h = he[e];
                        if (h.y == PH) break;  // slots fill in order; empty = 2^-15 (real: |value| < 2^-15)
                        const float xe = xs[XI(__float_as_int(h.x))];
                        sum = fmaf(xe, h.y, fmaf(xe, -PH, sum));
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

#define G13_DSM(U_, V_, P_, VT_) 0
void g13_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor sec, torch::Tensor x, torch::Tensor out, double scale,
              int64_t u, int64_t nv, int64_t minb, int64_t pf, int64_t blocks, int64_t vt) {
    const int k = x.size(1), n = out.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(x.size(0) == 1, "g13: m == 1");
    TORCH_CHECK((long)n * k <= (long)g13::MAXC * blocks * nv * u * vt * 16, "g13: too many chunks per virtual block");
#define G13_CASE(K_, U_, V_, B_, P_, VT_) \
    if (k == K_ && u == U_ && nv == V_ && minb == B_ && pf == P_ && vt == VT_) { \
        const int dsm_ = G13_DSM(U_, V_, P_, VT_); \
        if (dsm_) cudaFuncSetAttribute(g13::gemv<U_, V_, B_, K_, P_, VT_>, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm_); \
        g13::gemv<U_, V_, B_, K_, P_, VT_><<<blocks, VT_ * V_, dsm_, st>>>((const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), \
        (const float2*)h.data_ptr(), (const uint4*)sec.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }
    G13_CONFIGS
    TORCH_CHECK(false, "g13: no config");
}
