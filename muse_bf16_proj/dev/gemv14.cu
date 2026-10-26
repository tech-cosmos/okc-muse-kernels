// gemv14: gemv12 (tensor-core streaming GEMM, m = 8..32) over losslessly compressed BF16 weights, 12 bits each
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
#ifdef G14_BPF  // s36 (as pr8 npf 2xx): bulk L2 prefetch of atoms [PF, PF + G14_BPF) of each row group (WS, WN)
    if (t < 2 * NG) {
        const int nb = min(G14_BPF, NA - PF);
        const long o0 = (((long)ks * nrg_all + rg0 + (t >> 1)) * (KC / 32) + PF * WG) * 32;
        const char* src = (t & 1) ? (const char*)(WN + o0) : (const char*)(WS + o0);
        if (nb > 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"((unsigned)nb * WG * 32 * ((t & 1) ? 4 : 8)) : "memory");
    }
#endif
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
