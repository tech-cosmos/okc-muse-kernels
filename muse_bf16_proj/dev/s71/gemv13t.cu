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
namespace g13t {
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


// s71 g13t: m = 1 over the gemv13 (DEC3) packing with a per-virtual-block TMA ring instead of register prefetch.
// Block b owns rows [n b / G, n (b + 1) / G) (balanced, contiguous); its NV virtual blocks of 512 threads take every
// NV-th row; one slot = a whole row: 8 KB sign|mantissa + 4 KB codes + 64 B escape header (3 bulk copies), NS slots
// per vb (NV * NS * 12.06 KB in flight per SM). W copies go out before griddepcontrol.wait, x after it.
// Sum order = gemv13 VT 512 U 1 (thread dot16 -> warp xor-shuffle -> 16 warp partials in order -> escapes).
template <int NS, int NV>
__global__ void __launch_bounds__(512 * NV, 1) gemv(const uint4* __restrict__ S, const uint2* __restrict__ N,
                                                    const float2* __restrict__ H, const __nv_bfloat16* x,
                                                    __nv_bfloat16* __restrict__ out, int n, float scale) {
    constexpr int KK = 8192, SB = 8192, NB = 4096, RB = SB + NB + ESC * 8;
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ __align__(16) float xs[KK / 16 * 20];
    __shared__ float red[NV][2][16];
    __shared__ __align__(8) unsigned long long full[NV][NS];
    const int vg = threadIdx.x >> 9, t = threadIdx.x & 511, lane = t & 31, wp = t >> 5;
    const int r0 = (int)((long)n * blockIdx.x / gridDim.x), r1 = (int)((long)n * (blockIdx.x + 1) / gridDim.x);
    const int nrows = r1 - r0 > vg ? (r1 - r0 - vg + NV - 1) / NV : 0;
    unsigned char* ring = sm + vg * NS * RB;
    auto issue = [&](int i) {
        if (i >= nrows) return;
        const long r = r0 + vg + (long)i * NV;
        const unsigned bar = (unsigned)__cvta_generic_to_shared(&full[vg][i % NS]);
        const unsigned d = (unsigned)__cvta_generic_to_shared(ring + (i % NS) * RB);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(bar), "r"(RB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "l"(S + r * 512), "r"(SB), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d + SB), "l"(N + r * 512), "r"(NB), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d + SB + NB), "l"(H + r * ESC), "r"(ESC * 8), "r"(bar) : "memory");
    };
    if (t == 0) {
        for (int s = 0; s < NS; ++s) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&full[vg][s])));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int i = 0; i < NS; ++i) issue(i);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = threadIdx.x; i < KK / 8; i += 512 * NV) {
        const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(x) + i);
        const unsigned v[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) { const int kk = XI(8 * i + 2 * j); xs[kk] = __uint_as_float(v[j] << 16); xs[kk + 1] = __uint_as_float(v[j] & 0xffff0000u); }
    }
    __syncthreads();
    for (int i = 0; i < nrows; ++i) {
        const unsigned bar = (unsigned)__cvta_generic_to_shared(&full[vg][i % NS]);
        asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared.b64 P, [%0], %1; @!P bra W; }" :: "r"(bar), "r"((i / NS) & 1) : "memory");
        const unsigned char* b = ring + (i % NS) * RB;
        const uint4 s = *reinterpret_cast<const uint4*>(b + t * 16);
        const uint2 nb = *reinterpret_cast<const uint2*>(b + SB + t * 8);
#ifdef G13T_NOC  // timing probe: no decode / FMA
        float v = __uint_as_float(s.x ^ nb.x);
#else
        float v = dot16(s, nb, xs + XI(t * 16));
#endif
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[vg][i & 1][wp] = v;
        asm volatile("bar.sync %0, 512;" :: "r"(vg + 1));
        if (t == 0) {
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < 16; ++w) sum += red[vg][i & 1][w];
            const float2* he = reinterpret_cast<const float2*>(b + SB + NB);
            for (int e = 0; e < ESC; ++e) {
                const float2 h = he[e];
                if (h.y == 0x1p-15f) break;
                const float xe = xs[XI(__float_as_int(h.x))];
                sum = fmaf(xe, h.y, fmaf(xe, -0x1p-15f, sum));
            }
            out[r0 + vg + (long)i * NV] = __float2bfloat16(sum * scale);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(i + NS);
        }
    }
}
}  // namespace g13t

void g13t_gemv(torch::Tensor s, torch::Tensor nb, torch::Tensor h, torch::Tensor x, torch::Tensor out, double scale,
               int64_t ns, int64_t nv, int64_t blocks) {
    const int n = out.size(1);
    TORCH_CHECK(x.size(0) == 1 && x.size(1) == 8192, "g13t: m == 1, k == 8192");
    auto st = c10::cuda::getCurrentCUDAStream();
#define G13T_CASE(NS_, V_) \
    if (ns == NS_ && nv == V_) { const int dsm_ = V_ * NS_ * (12288 + 64); \
        cudaFuncSetAttribute(g13t::gemv<NS_, V_>, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm_); \
        g13t::gemv<NS_, V_><<<blocks, 512 * V_, dsm_, st>>>((const uint4*)s.data_ptr(), (const uint2*)nb.data_ptr(), \
        (const float2*)h.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, (float)scale); return; }
    G13T_CASE(2, 2) G13T_CASE(3, 2) G13T_CASE(4, 2) G13T_CASE(6, 2) G13T_CASE(4, 1) G13T_CASE(8, 1) G13T_CASE(3, 3) G13T_CASE(4, 3)
    TORCH_CHECK(false, "g13t: no config");
}
