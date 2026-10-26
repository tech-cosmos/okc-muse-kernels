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
// 4 weights (bytes of sw) -> 2 BF16 pairs; nibble word nw, pairs P0, P0 + 1
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = lop_aob(prmt(sw, 0x9180u), 0x807f807fu, expo<P0>(nw));
    const unsigned q1 = lop_aob(prmt(sw, 0xB3A2u), 0x807f807fu, expo<P0 + 1>(nw));
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

// VT: threads per virtual block (256 or 512): a chunk is U * VT * 16 weights. 512 halves the rows per round of a
// 2304-row qkv (4.36 -> 5 rounds of whole rows at 528 vbs: a 13% tail; 8.73 -> 9 half-size rounds at 264 vbs: 3%).
template <int U, int NV, int MINB, int KK, int PF, int VT = 256>
__global__ void __launch_bounds__(VT * NV, MINB) gemv(const uint4* __restrict__ S, const uint2* __restrict__ N,
                                                       const float2* __restrict__ H, const __nv_bfloat16* __restrict__ x,
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
#ifdef G13_CPA
    // cp.async ring: PF chunks in flight in smem (each thread reads back only its own 16 + 8 B per u: no sync)
    extern __shared__ __align__(16) unsigned char dsm[];
    uint4* sS = reinterpret_cast<uint4*>(dsm);                  // [PF][U][VT * NV]
    uint2* sN = reinterpret_cast<uint2*>(sS + PF * U * VT * NV);  // [PF][U][VT * NV]
    auto load = [&](int p, int ch) {
        if (ch < nch) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const long i = (long)ch * (CE / 16) + u * VT + t;
                const unsigned ds = (unsigned)__cvta_generic_to_shared(sS + (p * U + u) * VT * NV + threadIdx.x);
                const unsigned dn = (unsigned)__cvta_generic_to_shared(sN + (p * U + u) * VT * NV + threadIdx.x);
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" :: "r"(ds), "l"(S + i) : "memory");
                asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 8;" :: "r"(dn), "l"(N + i) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
#else
    uint4 s[PF][U];
    uint2 nb[PF][U];
    // escapes: the row's reducing thread adds x[col] * (value - 2^-15) per header entry to the row's sum
    auto load = [&](int p, int ch) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long i = (long)ch * (CE / 16) + u * VT + t;
            s[p][u] = ld4(S + i);
            nb[p][u] = ld2(N + i);
        }
    };
#endif
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
#ifdef G13_CPA
    for (int p = 0; p < PF - 1; ++p) load(p, c + p * G);
#else
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
#endif
#ifdef G13_L2PF
    // L2 bulk prefetch of this vb's later chunks (no registers): HBM streams at full rate from the start
    if (t == 0)
        for (int cc = c + PF * G, d = 0; cc < nch && d < G13_L2PF; cc += G, ++d) {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(S + (long)cc * (CE / 16)), "r"(CE) : "memory");
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(N + (long)cc * (CE / 16)), "r"(CE / 2) : "memory");
        }
#endif
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
#ifdef G13_CPA
    for (int i = 0; c < nch; c += G, ++i) {
        load((i + PF - 1) % PF, c + (PF - 1) * G);  // the slot chunk i - 1 used
        asm volatile("cp.async.wait_group %0;" :: "n"(PF - 1) : "memory");
        const int p = i % PF, cc = c;
        {
            {
                float acc[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const uint4 sv = sS[(p * U + u) * VT * NV + threadIdx.x];
                    const uint2 nv = sN[(p * U + u) * VT * NV + threadIdx.x];
                    acc[u] = dot16(sv, nv, xs + XI((u * VT * 16 + t * 16) % KK));
                }
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
                    const float2* he = hs_[vg] + (i * RPC + t) * ESC;
                    for (int e = 0; e < ESC; ++e) {
                        const float2 h = he[e];
                        if (h.y == 0x1p-15f) break;
                        const float xe = xs[XI(__float_as_int(h.x))];
                        sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
                    }
                    out[(long)cc * RPC + t] = __float2bfloat16(sum * scale);
                }
                par ^= 1;
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
    for (; c < nch; c += PF * G) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int cc = c + p * G;
            if (cc < nch) {
                float acc[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    acc[u] = dot16(s[p][u], nb[p][u], xs + XI((u * VT * 16 + t * 16) % KK));
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
#endif
}
}  // namespace g13

#ifdef G13_CPA
#define G13_DSM(U_, V_, P_, VT_) ((P_) * (U_) * (VT_) * (V_) * 24)
#else
#define G13_DSM(U_, V_, P_, VT_) 0
#endif
void g13_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
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
        (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }
    G13_CONFIGS
    TORCH_CHECK(false, "g13: no config");
}
