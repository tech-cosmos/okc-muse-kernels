#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g27 {
#ifdef G27_TRACE
__device__ unsigned long long trace[1024 * 6];
#define TR(i) if (t == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v_)); trace[blockIdx.x * 6 + (i)] = v_; }
#define TRW(i) if (lane == 0 && w == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v_)); trace[blockIdx.x * 6 + (i)] = v_; }
#else
#define TR(i)
#define TRW(i)
#endif
#ifdef G24_TRACE
__device__ unsigned long long trace[4096 * 3];
#endif
constexpr int WB = 768, ESC = 8;  // bytes per 16 x 32 atom
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x4140u);
    r.y = decw<1>(s0, nw, 0x4342u);
    r.z = decw<2>(s1, nw, 0x4140u);
    r.w = decw<3>(s1, nw, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count)); }
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW27_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W27_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}

#ifndef G27_MC
#define G27_MC 1  // s58: 1 = x by bulk copies, one per (row, k slice), spread over threads (layer d16 -1.8 sl); 0 = cp.async xload; >1 cluster multicast
#endif
// s58 MC: x rows multicast into all G27_MC CTAs of the cluster (same smem offsets / barrier offset in each CTA)
__device__ __forceinline__ void bulk_mc(unsigned dst, const void* src, int bytes, unsigned b) {
#if G27_MC > 1
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b), "h"((unsigned short)((1u << G27_MC) - 1)) : "memory");
#else
    bulk(dst, src, bytes, b);
#endif
}
__device__ __forceinline__ unsigned cl_rank() {
#if G27_MC > 1
    unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
#else
    return 0;
#endif
}
__device__ __forceinline__ void cl_sync() {
#if G27_MC > 1
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}
// x [m, k] -> smem rows of XS bytes by all threads: 16 B cp.async (L2, .cg), then wait + block barrier
__device__ __forceinline__ void xload(unsigned xs, const __nv_bfloat16* x, int m, int k, int XS, int t, int nt) {
    const int r16 = k / 8;
#ifdef G27_XHALF
    m /= 2;  // timing probe only
#endif
    for (int i = t; i < m * r16; i += nt) {
        const int row = i / r16, col = i % r16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(xs + row * XS + col * 16), "l"(x + (long)row * k + col * 8) : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}
// gemv27 (s57): no cross-block split-K. Block b owns RGB whole row groups (16 rows each, all of k); its WPB = RGB * KQ
// warps each stream one (row group, k / KQ slice) through a per-warp NS-slot TMA ring (pack24 with kc = k / KQ, r 1),
// partials go through smem (aliasing the warp's ring), the block sums its KQ partials in a fixed order and writes
// out (FP32 or BF16) with plain stores: no out zeroing, no fence, no grid epoch barrier, no atomics; deterministic.
// x [M, k] whole per block (after griddepcontrol.wait); W stages issued before the wait (npre, PDL secondary).
template <int M, int KQ, int AP, int NS, int RGB, bool OBF>
__global__ void __launch_bounds__(RGB * KQ * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                         void* __restrict__ outp, int n, int k, int npre, float* __restrict__ zo, int zn4) {
    constexpr int WPB = RGB * KQ, NT = M / 8, SB = AP * WB, STR = 16 + 4;
    static_assert(NT % 2 == 0 && M <= 32 && NS * SB >= M * STR * 4, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    TR(0)
    const int rgl = w % RGB, kq = w / RGB, rg = blockIdx.x * RGB + rgl, nrt = n / 16;
    const int KW = k / KQ, SPK = KW / 32 / AP, XS = k * 2 + 16;
    const unsigned s0 = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned ring = s0 + w * NS * SB, xs = s0 + WPB * NS * SB;
    const unsigned fb = xs + M * XS + w * NS * 8, xfb = xs + M * XS + WPB * NS * 8;
    const unsigned char* wsrc = WP + ((long)kq * nrt + rg) * SPK * SB;
    auto issue = [&](int j) {
        bar_expect(fb + 8 * (j % NS), SB);
        bulk(ring + (j % NS) * SB, wsrc + (long)j * SB, SB, fb + 8 * (j % NS));
    };
    int jn = 0;
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) bar_init(xfb, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (; jn < SPK && jn < npre && jn < NS; ++jn) issue(jn);  // W only, before the predecessor is done
#if G27_MC
        if (w == 0) bar_expect(xfb, M * k * 2);
#endif
    }
    __syncthreads();
#if G27_MC
    cl_sync();  // peers' x barriers initialized + armed before any multicast lands
#endif
    asm volatile("griddepcontrol.wait;" ::: "memory");
    TR(1)
#if G27_MC
    {
        const int cs = G27_MC > 1 ? G27_MC : 1, rk = cl_rank();
        for (int qq = rk + cs * t; qq < M; qq += cs * WPB * 32) bulk_mc(xs + qq * XS, x + (long)qq * k, k * 2, xfb);
    }
#endif
#ifdef G27_XBULK
    if (t == 0) {
        bar_expect(xfb, M * k * 2);
        for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS, x + (long)qq * k, k * 2, xfb);
    }
#endif
    if (lane == 0) for (; jn < SPK && jn < NS; ++jn) issue(jn);
    jn = __shfl_sync(0xffffffffu, jn, 0);
#ifndef G27_NOFIX
    float4 hd[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) hd[h] = __ldg((const float4*)(H + (long)(rg * 16 + g + 8 * h) * ESC) + c);
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] = acc2[jj][e] = 0.f;
    const unsigned xbase = xs + xrow + kq * (KW / 32) * 64;
#if defined(G27_XBULK) || G27_MC
    bar_wait(xfb, 0);
#else
#ifdef G27_XREP
    xload(xs, x + (long)blockIdx.x * M * k, M, k, XS, t, WPB * 32);  // timing probe only: per-block x copy
#else
    xload(xs, x, M, k, XS, t, WPB * 32);
#endif
#endif
    TRW(2)
    for (int st = 0; st < SPK; ++st) {
        const int sl = st % NS;
        bar_wait(fb + 8 * sl, (st / NS) & 1);
        const unsigned sb = ring + sl * SB + lane * 16, nb = ring + sl * SB + AP * 512 + lane * 8;
        constexpr int G = AP < 2 ? AP : 2;
#pragma unroll
        for (int j0 = 0; j0 < AP; j0 += G) {
            uint4 sv[G];
            uint2 nv[G];
#pragma unroll
            for (int a = 0; a < G; ++a) {
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[a].x), "=r"(sv[a].y), "=r"(sv[a].z), "=r"(sv[a].w) : "r"(sb + (j0 + a) * 512));
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[a].x), "=r"(nv[a].y) : "r"(nb + (j0 + a) * 256));
            }
            unsigned bx[G][2][NT][2];
#pragma unroll
            for (int a = 0; a < G; ++a)
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int jj = 0; jj < NT; jj += 2)
                        ldsm4(xbase + jj * 8 * XS + (st * AP + j0 + a) * 64 + s * 32, bx[a][s][jj][0], bx[a][s][jj][1], bx[a][s][jj + 1][0], bx[a][s][jj + 1][1]);
#pragma unroll
            for (int a = 0; a < G; ++a)
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const uint4 af = dec8(s ? sv[a].z : sv[a].x, s ? sv[a].w : sv[a].y, s ? nv[a].y : nv[a].x);
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[jj] : acc[jj], af, bx[a][s][jj][0], bx[a][s][jj][1]);
                }
        }
        __syncwarp();
        if (jn < SPK) {
            if (lane == 0) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                issue(jn);
            }
            ++jn;
        }
    }
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] += acc2[jj][e];
#ifndef G27_NOFIX
    {  // escape fixups (g24's): escaped weights decoded as 2^-15 (scale units); add x * (true - 2^-15)
        const int klo = kq * KW, khi = klo + KW;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = hd[h];
            unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
            unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
            if (!(f0 | f1)) continue;
            f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
            f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {
                if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
                if (hy != 0x1p-15f && col >= klo && col < khi) {
                    const unsigned char* xc = smem + WPB * NS * SB + col * 2;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                            acc[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[jj][2 * h + b]));
                        }
                }
            }
        }
    }
#endif
    // partials -> [tok][row] tile in this warp's own ring (all its stages are consumed), then a fixed-order block sum
    TRW(3)
    float* stg = (float*)(smem + w * NS * SB);
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + g + 8 * (e >> 1)] = acc[jj][e] * scale;
    __syncthreads();
    TR(4)
    for (int i = t; i < M * RGB * 16; i += WPB * 32) {
        const int tok = i / (RGB * 16), rr = i % (RGB * 16);
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < KQ; ++q) s += ((const float*)(smem + (q * RGB + rr / 16) * NS * SB))[tok * STR + rr % 16];
        const long o = (long)tok * n + blockIdx.x * RGB * 16 + rr;
        if (OBF) ((__nv_bfloat16*)outp)[o] = __float2bfloat16_rn(s);
        else ((float*)outp)[o] = s;
    }
    // ZN (as g24's, layer proj s56): zero the next g24's FP32 out (zo, zn4 float4s); plain stores, no fence
    for (int i = blockIdx.x * WPB * 32 + t; i < zn4; i += gridDim.x * WPB * 32) ((float4*)zo)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef G27_TRACE
    __syncthreads();
#endif
    TR(5)
}
// s57 PD variant: NAW = k / KQ / 32 atoms per warp, all held in registers: every W stage is waited for and DECODED
// before griddepcontrol.wait (W does not depend on the predecessor); after the wait only x (ldmatrix) + mma + sum.
// Needs NS >= NAW / AP (whole slice in the ring, issued pre-wait). XQ: x arrives in KQ slices (one barrier each).
template <int M, int KQ, int AP, int NAW, int RGB, bool OBF>
__global__ void __launch_bounds__(RGB * KQ * 32, 1) gemm_pd(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                            void* __restrict__ outp, int n, int k, int npre, float* __restrict__ zo, int zn4) {
    constexpr int WPB = RGB * KQ, NT = M / 8, SB = AP * WB, STR = 16 + 4, NS = NAW / AP, KW = NAW * 32;
    static_assert(NT % 2 == 0 && M <= 32 && NS * SB >= M * STR * 4 && NAW % AP == 0, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    TR(0)
    const int rgl = w % RGB, kq = w / RGB, rg = blockIdx.x * RGB + rgl, nrt = n / 16;
    const int XS = k * 2 + 16;
    const unsigned s0 = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned ring = s0 + w * NS * SB, xs = s0 + WPB * NS * SB;
    const unsigned fb = xs + M * XS + w * NS * 8, xfb = xs + M * XS + WPB * NS * 8;
    const unsigned char* wsrc = WP + ((long)kq * nrt + rg) * NS * SB;
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) for (int i = 0; i < KQ; ++i) bar_init(xfb + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int j = 0; j < NS; ++j) {
            bar_expect(fb + 8 * j, SB);
            bulk(ring + j * SB, wsrc + (long)j * SB, SB, fb + 8 * j);
        }
#if G27_MC
        if (w == 0) for (int q = 0; q < KQ; ++q) bar_expect(xfb + 8 * q, M * KW * 2);
#endif
    }
#ifndef G27_NOFIX
    float4 hd[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) hd[h] = __ldg((const float4*)(H + (long)(rg * 16 + g + 8 * h) * ESC) + c);
#endif
#if G27_MC
    cl_sync();  // early: peers' x barriers armed; hidden under the W stream
#else
    __syncthreads();
#endif
    uint4 af[NAW][2];
#pragma unroll
    for (int st = 0; st < NS; ++st) {
        bar_wait(fb + 8 * st, 0);
        const unsigned sb = ring + st * SB + lane * 16, nb = ring + st * SB + AP * 512 + lane * 8;
#pragma unroll
        for (int a = 0; a < AP; ++a) {
            uint4 sv; uint2 nv;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + a * 512));
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + a * 256));
            af[st * AP + a][0] = dec8(sv.x, sv.y, nv.x);
            af[st * AP + a][1] = dec8(sv.z, sv.w, nv.y);
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    TR(1)
#if G27_MC
    {  // slice q of row qq by thread i = q * M + qq of rank qq % cs
        const int cs = G27_MC > 1 ? G27_MC : 1, rk = cl_rank();
        for (int i = t; i < KQ * M; i += WPB * 32) {
            const int q = i / M, qq = i % M;
            if (qq % cs == rk) bulk_mc(xs + qq * XS + q * KW * 2, x + (long)qq * k + q * KW, KW * 2, xfb + 8 * q);
        }
    }
#elif defined(G27_XBULK)
    if (t == 0) {
        for (int q = 0; q < KQ; ++q) {
            bar_expect(xfb + 8 * q, M * KW * 2);
            for (int qq = 0; qq < M; ++qq) bulk(xs + qq * XS + q * KW * 2, x + (long)qq * k + q * KW, KW * 2, xfb + 8 * q);
        }
    }
#else
#ifdef G27_XREP
    xload(xs, x + (long)blockIdx.x * M * k, M, k, XS, t, WPB * 32);  // timing probe only: per-block x copy
#else
    xload(xs, x, M, k, XS, t, WPB * 32);
#endif
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[NT][4], acc2[NT][4];
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] = acc2[jj][e] = 0.f;
    const unsigned xbase = xs + xrow + kq * NAW * 64;
#if defined(G27_XBULK) || G27_MC
    bar_wait(xfb + 8 * kq, 0);
#endif
    TRW(2)
#pragma unroll
    for (int a = 0; a < NAW; ++a) {
        unsigned bx[2][NT][2];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int jj = 0; jj < NT; jj += 2)
                ldsm4(xbase + jj * 8 * XS + a * 64 + s * 32, bx[s][jj][0], bx[s][jj][1], bx[s][jj + 1][0], bx[s][jj + 1][1]);
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[jj] : acc[jj], af[a][s], bx[s][jj][0], bx[s][jj][1]);
    }
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[jj][e] += acc2[jj][e];
#ifndef G27_NOFIX
    {
        const int klo = kq * KW, khi = klo + KW;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float4 hq = hd[h];
            unsigned f0 = __ballot_sync(0xffffffffu, hq.y != 0x1p-15f && __float_as_int(hq.x) >= klo && __float_as_int(hq.x) < khi);
            unsigned f1 = __ballot_sync(0xffffffffu, hq.w != 0x1p-15f && __float_as_int(hq.z) >= klo && __float_as_int(hq.z) < khi);
            if (!(f0 | f1)) continue;
            f0 |= f0 >> 16; f0 |= f0 >> 8; f0 |= f0 >> 4;
            f1 |= f1 >> 16; f1 |= f1 >> 8; f1 |= f1 >> 4;
#pragma unroll
            for (int e2 = 0; e2 < ESC; ++e2) {
                if (!((((e2 & 1) ? f1 : f0) >> (e2 >> 1)) & 1u)) continue;
                const int src = (lane & ~3) | (e2 >> 1);
                const float hx = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.z : hq.x, src);
                const float hy = __shfl_sync(0xffffffffu, (e2 & 1) ? hq.w : hq.y, src);
                const int col = __float_as_int(hx);
                if (hy != 0x1p-15f && col >= klo && col < khi) {
                    const unsigned char* xc = smem + WPB * NS * SB + col * 2;
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                            acc[jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[jj][2 * h + b]));
                        }
                }
            }
        }
    }
#endif
    TRW(3)
    float* stg = (float*)(smem + w * NS * SB);
#pragma unroll
    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + g + 8 * (e >> 1)] = acc[jj][e] * scale;
    __syncthreads();
    TR(4)
    for (int i = t; i < M * RGB * 16; i += WPB * 32) {
        const int tok = i / (RGB * 16), rr = i % (RGB * 16);
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < KQ; ++q) s += ((const float*)(smem + (q * RGB + rr / 16) * NS * SB))[tok * STR + rr % 16];
        const long o = (long)tok * n + blockIdx.x * RGB * 16 + rr;
        if (OBF) ((__nv_bfloat16*)outp)[o] = __float2bfloat16_rn(s);
        else ((float*)outp)[o] = s;
    }
    // ZN (as g24's, layer proj s56): zero the next g24's FP32 out (zo, zn4 float4s); plain stores, no fence
    for (int i = blockIdx.x * WPB * 32 + t; i < zn4; i += gridDim.x * WPB * 32) ((float4*)zo)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef G27_TRACE
    __syncthreads();
#endif
    TR(5)
}
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
}  // namespace g27
#ifdef G27_TRACE
torch::Tensor g27_trace() {
    auto t = torch::empty({1024 * 6}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(t.data_ptr(), g27::trace, sizeof(unsigned long long) * 1024 * 6, 0, cudaMemcpyDeviceToDevice);
    return t;
}
#else
torch::Tensor g27_trace() { return torch::empty({0}); }
#endif
void g27_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g27::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
template <int M_, int KQ_, int AP_, int NS_, int RGB_, bool OBF_>
static void launch27(torch::Tensor& wp, torch::Tensor& h, double scale, torch::Tensor& x, torch::Tensor& out, int sm, int n, int k, int64_t npre, torch::Tensor& zo, cudaStream_t stream) {
    auto kern = [] { if constexpr (NS_ >= 100) return g27::gemm_pd<M_, KQ_, AP_, NS_ - 100, RGB_, OBF_>; else return g27::gemm<M_, KQ_, AP_, NS_, RGB_, OBF_>; }();
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(n / (16 * RGB_)); cfg.blockDim = dim3(RGB_ * KQ_ * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
#if G27_MC > 1
    attr[1].id = cudaLaunchAttributeClusterDimension; attr[1].val.clusterDim.x = G27_MC; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1;
    cfg.numAttrs = 2;
    TORCH_CHECK((n / (16 * RGB_)) % G27_MC == 0, "g27 mc: grid");
#endif
    cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(),
                       out.data_ptr(), n, k, (int)npre, zo.numel() ? zo.data_ptr<float>() : nullptr, (int)(zo.numel() / 4));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void g27_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out,
              int64_t kq, int64_t ap, int64_t ns, int64_t rgb, int64_t npre, torch::Tensor zo) {
    TORCH_CHECK(zo.numel() % 4 == 0 && (!zo.numel() || (zo.is_contiguous() && zo.scalar_type() == torch::kFloat32)), "g27: zo");
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const bool obf = out.scalar_type() == torch::kBFloat16;
    const bool pd = ns >= 100;  // PD variant: ns = 100 + NAW (atoms per warp); ring = NAW / AP slots
    const int nsl = pd ? (ns - 100) / ap : ns;
    TORCH_CHECK(!pd || (ns - 100) * 32 * kq == k, "g27 pd: NAW");
    const int sm = rgb * kq * nsl * ap * g27::WB + m * (k * 2 + 16) + rgb * kq * nsl * 8 + kq * 8 + 16;
    TORCH_CHECK(n % (16 * rgb) == 0 && k % (kq * 32 * ap) == 0 && sm <= 227 * 1024, "g27: shape/smem ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G27_CASE(M_, KQ_, AP_, NS_, RGB_, OBF_) \
    if (m == M_ && kq == KQ_ && ap == AP_ && ns == NS_ && rgb == RGB_ && obf == OBF_) { \
        launch27<M_, KQ_, AP_, NS_, RGB_, OBF_>(wp, h, scale, x, out, sm, n, k, npre, zo, stream); return; }
    G27_CONFIGS
    TORCH_CHECK(false, "g27: no config");
}
