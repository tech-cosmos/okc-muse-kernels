// gemv16: gemv15 (compressed weights, cluster split-K over DSMEM) with W as the mma A operand (16 rows x k16) and x as
// B (k16 x 8 tokens, ldmatrix from smem): half the x smem traffic per weight of gemv15 at m >= 16, and an atom is
// 16 rows x 32 k (lane: 16 B of sign|mantissa bytes + 8 B of exponent codes = two A fragments).
// Block = NG 16-row groups x WG warps (atoms of a group interleaved over its warps), one k-chunk; cluster = the S
// k-chunk blocks of a row run; partials reduced over DSMEM.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace g16 {
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
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void ldsm2(unsigned addr, unsigned& d0, unsigned& d1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(d0), "=r"(d1) : "r"(addr));
}
// M tokens (multiple of 8), KC = 1024 * KCU k-chunk, WG warps per 16-row group, PF atoms in flight per warp, XP x pieces
template <int M, int KCU, int WG, int PF, int XP, int MC, int LB>
__global__ void __launch_bounds__(LB, 1) gemm(const uint4* __restrict__ WS, const uint2* __restrict__ WN,
                                                const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, int n, int nrg_all, int k) {
    constexpr int KC = 1024 * KCU, NT = M / 8, NA = KC / 32 / WG;  // n-tiles, atoms per warp
    static_assert(M % 8 == 0 && NA % PF == 0 && NA % XP == 0 && XP <= 4 && (MC == 1 || XP == 1), "config");
    constexpr int PA = NA / XP, PB = KC / XP * 2;  // atoms per x piece (per warp), bytes per x piece row
    constexpr int XS = KC * 2 + 16;                // x row stride (bytes): 16 mod 128 -> ldmatrix conflict-free
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x [M][KC]
    float4* accp = (float4*)(smem + M * XS);         // [warps][NT][32 lanes] C fragments
    float2* hs = (float2*)(smem + M * XS + blockDim.x / 32 * NT * 512);  // [NG * 16 rows][ESC]
    float* part = (float*)(hs + blockDim.x / (32 * WG) * 16 * ESC);     // [NG * M * 16] this block's partials
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int S = k / KC, ks = blockIdx.x % S, rg0 = (blockIdx.x / S) * NG, rg = rg0 + gi;
    auto stage = [&](int p) {
        for (int i = t; i < M * (PB / 16); i += blockDim.x) {
            const int r = i / (PB / 16), q = i % (PB / 16);
            const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + p * PB + q * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa),
                         "l"((const uint4*)(x + (long)r * k + (long)ks * KC + p * (KC / XP)) + q));
        }
    };
    for (int i = t; i < NG * 16 * ESC / 2; i += blockDim.x) {
        const unsigned sa = (unsigned)__cvta_generic_to_shared(hs + 2 * i);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(H + (long)rg0 * 16 * ESC + 2 * i));
    }
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int mr = (blockIdx.x % (S * MC)) / S;  // rank among the MC blocks of the cluster sharing this k-chunk
    const unsigned bar = ((unsigned)__cvta_generic_to_shared(part + NG * M * 16) + 7u) & ~7u;
    if (MC == 1 && S > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;");  // peers started: waited before DSMEM stores
    if constexpr (MC > 1) {
        // x[:, chunk] by TMA bulk copies multicast to the MC blocks with the same k-chunk: block mr copies rows
        // r % MC == mr; one mbarrier per block collects all M rows.
        asm volatile("cp.async.commit_group;");  // escape headers
        if (t == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        cl.sync();
        if (t == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(M * KC * 2) : "memory");
        if (t < M && t % MC == mr) {
            unsigned short mask = 0;
#pragma unroll
            for (int j = 0; j < MC; ++j) mask |= (unsigned short)(1u << (j * S + ks));
            const unsigned dst = (unsigned)__cvta_generic_to_shared(xs + t * XS);
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                         :: "r"(dst), "l"(x + (long)t * k + (long)ks * KC), "r"(KC * 2), "r"(bar), "h"(mask) : "memory");
        }
    } else {
#ifndef G16_NOX
#pragma unroll
        for (int p = 0; p < XP; ++p) { stage(p); asm volatile("cp.async.commit_group;"); }
#endif
    }
    const long wo = (((long)ks * nrg_all + rg) * (KC / 32) + wi) * 32 + lane;  // atom j: wo + j*WG*32
    const uint4* wbs = WS + wo;
    const uint2* wbn = WN + wo;
    uint4 ws_[PF];
    uint2 wn_[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) { ws_[p] = ldw(wbs + p * WG * 32); wn_[p] = ld2(wbn + p * WG * 32); }
    float acc[NT][4], acc2[NT][4];  // k-steps s = 0 / 1 accumulate separately: two independent mma chains
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = acc2[j][0] = acc2[j][1] = acc2[j][2] = acc2[j][3] = 0.f;
    // ldmatrix row address: matrix q = lane / 8 (x4: n-tile pair member q / 2, k half q % 2), row = lane % 8
    const unsigned xbase = (unsigned)__cvta_generic_to_shared(xs) + ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    for (int a = 0; a < NA; a += PF) {
        if (MC > 1 && a == 0) {
            asm volatile("{\n.reg .pred p;\nW16_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra W16_%=;\n}" :: "r"(bar) : "memory");
        } else if (MC == 1 && a % PA == 0) {
            const int left = XP - 1 - a / PA;
            if (left >= 3) asm volatile("cp.async.wait_group 3;");
            else if (left == 2) asm volatile("cp.async.wait_group 2;");
            else if (left == 1) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const uint4 sv = ws_[p];
            const uint2 nv = wn_[p];
            if (a + p + PF < NA) { ws_[p] = ldw(wbs + (a + p + PF) * WG * 32); wn_[p] = ld2(wbn + (a + p + PF) * WG * 32); }
            const unsigned kb = (((a + p) * WG + wi) * 32) * 2;  // byte offset of the atom's k within the chunk
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const uint4 af = dec8(s ? sv.z : sv.x, s ? sv.w : sv.y, s ? nv.y : nv.x);
#ifdef G16_NOMMA
                acc[0][s] += __uint_as_float(af.x ^ af.y ^ af.z ^ af.w); continue;
#endif
#pragma unroll
                for (int j = 0; j + 1 < NT; j += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm4(xbase + j * 8 * XS + kb + s * 32, b0, b1, b2, b3);
                    mma(s ? acc2[j] : acc[j], af, b0, b1);
                    mma(s ? acc2[j + 1] : acc[j + 1], af, b2, b3);
                }
                if (NT & 1) {
                    unsigned b0, b1;
                    ldsm2(xbase + (NT - 1) * 8 * XS + kb + s * 32, b0, b1);  // lanes 0-15 give addresses (q/2 = 0)
                    mma(s ? acc2[NT - 1] : acc[NT - 1], af, b0, b1);
                }
            }
        }
    }
    float* accf = (float*)accp;  // [warp][NT][4 elements][32 lanes]: conflict-free
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) accf[((wp * NT + j) * 4 + e) * 32 + lane] = acc[j][e] + acc2[j][e];
#ifdef G16_NORED
    return;
#endif
    if (MC > 1) asm volatile("cp.async.wait_group 0;");
    if (MC == 1 && S > 1) asm volatile("barrier.cluster.wait.aligned;");
    __syncthreads();
#ifdef G16_SYNCONLY
    return;
#endif
    // partials i = ((rgl * NT + nt) * 4 + e) * 32 + l: row16 = l / 4 + 8 * (e / 2), tok = nt * 8 + (l % 4) * 2 + e % 2
    const int ne = NG * M * 16;
    for (int i = t; i < ne; i += blockDim.x) {
        const int l = i & 31, e = (i >> 5) & 3, nt = (i >> 7) % NT, rgl = (i >> 7) / NT;
        const int row16 = (l >> 2) + 8 * (e >> 1), tok = nt * 8 + (l & 3) * 2 + (e & 1);
        const float* base = accf + (rgl * WG * NT + nt) * 128 + e * 32 + l;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[q * NT * 128];
        const float2* he = hs + (rgl * 16 + row16) * ESC;
#ifndef G16_NOESC
        for (int e2 = 0; e2 < ESC; ++e2) {  // slots fill in order; an empty slot has value 2^-15 (real escapes: |value| < 2^-15)
            const float2 h = he[e2];
            if (h.y == 0x1p-15f) break;
            const int col = __float_as_int(h.x) - ks * KC;
            if (col >= 0 && col < KC) {  // two fmas: exact when the dot holds only this weight (probe rows)
                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xs + tok * XS + col * 2));
                s = fmaf(xv, h.y, fmaf(xv, -0x1p-15f, s));
            }
        }
#endif
        if (S == 1) out[(long)tok * n + (rg0 + rgl) * 16 + row16] = __float2bfloat16(s * scale);
        else {  // slice q = i / (ne / S) is summed by cluster rank q (of this row run): push into its part, slot ks
            const int q = i / (ne / S);
            float* dst = q == ks ? part : cl.map_shared_rank(part, mr * S + q);
            dst[ks * (ne / S) + i - q * (ne / S)] = s;
        }
    }
    if (S == 1) {
        if (MC > 1) cl.sync();  // peers' multicast copies into this block are done (own barrier), keep peers alive anyway
        return;
    }
#ifdef G16_NOP2
    return;
#endif
    cl.sync();  // every block's slices are in its owner's part
    const int sl = ne / S;
    for (int i = t; i < sl; i += blockDim.x) {
        float v = 0.f;
        for (int q = 0; q < S; ++q) v += part[q * sl + i];
        const int ig = ks * sl + i;
        const int l = ig & 31, e = (ig >> 5) & 3, nt = (ig >> 7) % NT, rgl = (ig >> 7) / NT;
        const int row16 = (l >> 2) + 8 * (e >> 1), tok = nt * 8 + (l & 3) * 2 + (e & 1);
        out[(long)tok * n + (rg0 + rgl) * 16 + row16] = __float2bfloat16(v * scale);
    }
}
}  // namespace g16
// nrg_all: 16-row groups of the packed weight; bk: row runs (clusters), each n/16/bk groups x WG warps, S = k/KC blocks.
void g16_gemm(torch::Tensor ws_, torch::Tensor wn, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              int64_t nrg_all, int64_t kcu, int64_t wg, int64_t pf, int64_t bk, int64_t xp, int64_t mc) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int S = k / (1024 * kcu), ng = n / 16 / bk, threads = ng * wg * 32;
    const int sm = m * (1024 * kcu * 2 + 16) + threads / 32 * (m / 8) * 512 + ng * 16 * g16::ESC * 8 + ng * m * 16 * 4 + 16;
    TORCH_CHECK(n / 16 % bk == 0 && bk % mc == 0 && threads <= 1024 && sm <= 227 * 1024 && S * mc <= 16, "g16: shape");
#define G16_CASE(M_, U_, G_, P_, X_, C_) \
    if (m == M_ && kcu == U_ && wg == G_ && pf == P_ && xp == X_ && mc == C_) { \
        auto kern = threads <= 512 ? g16::gemm<M_, U_, G_, P_, X_, C_, 512> : g16::gemm<M_, U_, G_, P_, X_, C_, 1024>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        if (S * mc > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(S * bk); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = sm; \
        cfg.stream = c10::cuda::getCurrentCUDAStream(); \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeClusterDimension; \
        attr[0].val.clusterDim.x = S * mc; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const uint4*)ws_.data_ptr(), (const uint2*)wn.data_ptr(), \
            (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
            (__nv_bfloat16*)out.data_ptr(), n, nrg_all, k); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G16_CONFIGS
    TORCH_CHECK(false, "g16: no config");
}
