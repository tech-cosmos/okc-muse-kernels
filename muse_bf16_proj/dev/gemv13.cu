// gemv13: gemv10 (grid-stride GEMV, m = 1) over losslessly compressed BF16 weights: 12 bits per weight.
// prepare() splits W [n, KK] into a sign|mantissa byte plane (flat, row-major) and a 4-bit exponent code plane:
// code c = e - base (base = max exponent - 15), so the decoded exponent is 112 | c (exact power-of-2 rescale of the
// sum by 2^(base - 112)). Weights with e < base (zeros, tiny values: ~1 per 8K) are escapes: code 0, byte 0 (decodes
// to D = 2^(base - 127) in true scale) and a per-row header of ESC (col, (true - D) / scale) pairs fixes the sum.
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
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    return (s & 0x07800780u) | 0x38003800u;
}
// 4 weights (bytes of sw) -> 2 BF16 pairs; nibble word nw, pairs P0, P0 + 1
template <int P0>
__device__ __forceinline__ void dec4(unsigned sw, unsigned nw, float* f) {
    const unsigned q0 = (prmt(sw, 0x9180u) & 0x807f807fu) | expo<P0>(nw);
    const unsigned q1 = (prmt(sw, 0xB3A2u) & 0x807f807fu) | expo<P0 + 1>(nw);
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
    // escapes: the first warp of each row (u, wp with (u * 4096 + wp * 512) % KK == 0), lanes e < ESC, add
    // x[col] * (delta / scale) of the row's entry e to their partial sum before the warp reduction
    auto load = [&](int p, int ch) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const long i = (long)ch * (CE / 16) + u * 256 + t;
            s[p][u] = ld4(S + i);
            nb[p][u] = ld2(N + i);
        }
    };
#pragma unroll
    for (int p = 0; p < PF; ++p)
        if (c + p * G < nch) load(p, c + p * G);
    for (int i = t; i < MAXC * RPC * ESC; i += 256) {
        const int ch = c + (i / (RPC * ESC)) * G;
        if (ch < nch) hs_[vg][i] = __ldg(H + (long)ch * RPC * ESC + i % (RPC * ESC));
    }
    for (int i = threadIdx.x; i < KK / 8; i += 256 * NV) {
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(x) + i);
        const unsigned v[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) { const int kk = XI(8 * i + 2 * j); xs[kk] = __uint_as_float(v[j] << 16); xs[kk + 1] = __uint_as_float(v[j] & 0xffff0000u); }
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
#ifndef NOESC
                    if ((u * 4096 + wp * 512) % KK == 0 && lane < ESC) {
                        const float2 h = hs_[vg][((j + p) * RPC + (u * 4096 + wp * 512) / KK) * ESC + lane];
                        acc[u] = fmaf(xs[XI(__float_as_int(h.x))], h.y, acc[u]);
                    }
#endif
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
