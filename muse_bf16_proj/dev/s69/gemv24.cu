// gemv24 (s53): m = 16/32 GEMM over lossless 12-bit weights (DEC3 layout) with PER-WARP TMA rings: no producer warp,
// no block-wide stage barriers. Every warp streams its own split-K tasks (16 rows x KC k = KC/32 atoms, contiguous:
// pack19 with rg 1) through its own NS-slot ring (lane 0 issues one bulk copy per slot, refills a slot right after
// consuming it), so a slow warp never gates the others (g19's per-stage producer/consumer handoff is the target).
// Tasks: t = ks * (n / 16) + row group; block b owns tasks [T b / G, T (b + 1) / G) (static), warp w every WPB-th;
// the block's x chunks (<= XB) are loaded once after griddepcontrol.wait. out FP32 [M, n] zeroed in the kernel
// (epoch-counter grid barrier before the first red), partials by red.global.add.v4 through a per-warp smem tile.
// SK (s54, template RR = R | 256): stream-K over stages instead (each warp an equal contiguous stage range, partial
// flushes at task boundaries): H200 proxy wins at m32 / qkv m16, loses at m16 pool/unp/o (NOTES-proj s54).
// Escape fixups at each task end (g19's register path); G24_NOFIX skips them (x @ W'^T, escaped weights = 2^-15 * scale).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
// s60 copy of r-layer g24_src.cu (85b5641) + WG (RR bit 9): warpgroup wgmma m64nMk16, A = decoded W in registers,
// B = x in smem (core-matrix layout, plain __ldcg loads after the wait); static tasks of 4 row groups x KC per warpgroup.
#ifndef G24_WG_A2
#define G24_WG_A2 0
#endif
namespace g24 {
#ifdef G24_CLK
__device__ long long clk[4096 * 8];  // s61: per warp cycles: stage loops, task ends, first-x wait, count
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
    asm volatile("{\n.reg .pred p;\nW24_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W24_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
template <int M> __device__ __forceinline__ void wgmma(float (&d)[M / 8][4], const uint4& a, unsigned long long b);
template <> __device__ __forceinline__ void wgmma<32>(float (&d)[4][4], const uint4& a, unsigned long long b) {
    asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(b) : "memory");
}
template <> __device__ __forceinline__ void wgmma<16>(float (&d)[2][4], const uint4& a, unsigned long long b) {
    asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(b) : "memory");
}
// G24_WG_SS (timing probe only, wrong results): A from smem (descriptor on the warpgroup's ring) instead of registers
__device__ __forceinline__ void wgmma_ss32(float (&d)[4][4], unsigned long long a, unsigned long long b) {
    asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;\n}"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                   "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3])
                 : "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void wgmma_ss16(float (&d)[2][4], unsigned long long a, unsigned long long b) {
    asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3])
                 : "l"(a), "l"(b) : "memory");
}
// wgmma accumulators: tie the registers to the asm stream (the compiler may move plain register math across wait_group)
template <int NT> __device__ __forceinline__ void fence_acc(float (&d)[NT][4]) {
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) asm volatile("" : "+f"(d[i][e]) :: "memory");
}
#define WG_WAIT(N_) do { asm volatile("wgmma.wait_group.sync.aligned " #N_ ";" ::: "memory"); fence_acc(aw[0]); fence_acc(aw[1]); } while (0)
// smem: ring [WPB][NS][SB] | x [XB][M][XS] | stage [WPB][M][STR] f32 | full [WPB][NS] | xfull [XB] | target
// R: row groups per task (one x fragment load feeds R atoms); stage (ks, tile of R row groups, st) =
// [R][AP][32 lanes x 16 B sign|mantissa] [R][AP][32 x 8 B codes] (pack19 with rg = R).
template <int M, int KC, int AP, int NS, int WPB, int XB, int RR>
__global__ void __launch_bounds__(WPB * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int* __restrict__ ctr, int n, int k, int npre, float* __restrict__ zo, int zn4) {
    constexpr int R = RR & 0xff;
    constexpr bool SK = (RR >> 8) & 1, WG = (RR >> 9) & 1;
    static_assert(!WG || (R == 1 && !SK && WPB % 4 == 0), "WG config");
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = R * AP * WB, XS = KC * 2 + 16, STR = 16 * R + 4;
    static_assert(NA % AP == 0 && NT % 2 == 0 && M <= 32, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    // npre bit 16: trigger the PDL dependent at start (the layer's g19 convention, s57: dependents that only touch out
    // after their own griddepcontrol.wait)
    if (npre & 0x10000) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    // npre bit 17 (proj s56 EZ): out arrives zeroed by an earlier g24 of the same layer call (its zo), complete before
    // our griddepcontrol.wait returns (every kernel in between waits on its predecessor): no zero/fence/epoch barrier
    const bool ez = (npre >> 17) & 1, ezpf = ez && ((npre >> 18) & 1);
    // s61 (runtime config bits, shape-only): bit 19 XSP = x barriers armed before griddepcontrol.wait, x bulk copies spread
    // over threads after it (not thread 0 alone); bit 20 DF = defer a task's flush while the out-zero barrier is pending
    const bool xsp = !WG && ((npre >> 19) & 1), df = (npre >> 20) & 1;
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem) + w * NS * SB, xs = (unsigned)__cvta_generic_to_shared(smem) + WPB * NS * SB;
    float* stg = (float*)(smem + WPB * NS * SB + XB * M * XS) + w * M * STR;
    const unsigned fb = (unsigned)__cvta_generic_to_shared(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4) + w * NS * 8;
    const unsigned xfb = (unsigned)__cvta_generic_to_shared(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4) + WPB * NS * 8;
    int* tgt = (int*)(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4 + WPB * NS * 8 + XB * 8);
    const int nrt = n / (16 * R), S = k / KC;
    // WG (s60): task unit = 4 row groups (64 rows) x KC, one per warpgroup; warp w % 4 = its row group in the unit
    const int nru = WG ? nrt / 4 : nrt, T = S * nru, NU = WG ? WPB / 4 : WPB, wu = WG ? w / 4 : w;
    auto tmap = [&](int u) { return WG ? (u / nru) * nrt + (u % nru) * 4 + (w & 3) : u; };
    // SK (s54, RR bit 8) = stream-K over stages: block b owns global stages [SG b / G, SG (b + 1) / G) (stage g = task
    // g / SPK, st g % SPK), warp w a contiguous 1/WPB of them, flushing its partials at every task boundary it crosses
    const long SG = (long)T * SPK;
    const int sa = SK ? (int)(SG * blockIdx.x / gridDim.x) : 0, sz = SK ? (int)(SG * (blockIdx.x + 1) / gridDim.x) : 0;
    const int wa = sa + (int)((long)(sz - sa) * w / WPB);
    const int ta = SK ? sa / SPK : (int)((long)T * blockIdx.x / gridDim.x);
    const int tb = SK ? (sz + SPK - 1) / SPK : (int)((long)T * (blockIdx.x + 1) / gridDim.x);
    const int ks0 = ta / nru;
    const int nq = SK ? 0 : tb - ta > wu ? (tb - ta - wu + NU - 1) / NU : 0;
    const int total = SK ? sa + (int)((long)(sz - sa) * (w + 1) / WPB) - wa : nq * SPK;
    auto issue = [&](int j) {  // lane 0: this warp's sequence element j -> slot j % NS
        const int tq = SK ? (wa + j) / SPK : tmap(ta + wu + (j / SPK) * NU), st = SK ? (wa + j) % SPK : j % SPK, sl = j % NS;
#ifdef G24_NOW
        bar_arrive(fb + 8 * sl);
#else
        bar_expect(fb + 8 * sl, SB);
        bulk(ring + sl * SB, WP + ((long)tq * SPK + st) * SB, SB, fb + 8 * sl);
#endif
    };
    int jn = 0;  // next sequence element to issue
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) for (int b = 0; b < XB; ++b) bar_init(xfb + 8 * b, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (; jn < total && jn < npre; ++jn) issue(jn);  // W only, before the predecessor is done
        if (xsp && w == 0) {  // s61 XSP (s58 idea): x barriers armed before the wait (shape-only), copies spread over threads
            const int ksl = (tb - 1) / nru;
            for (int b = 0; b < XB && ks0 + b <= ksl; ++b) bar_expect(xfb + 8 * b, M * KC * 2);
        }
    }
    if (xsp) __syncthreads();  // x barriers initialized + armed before any thread's copy completes on them
    if (ezpf)  // proj s56 (npre bit 18): out (zeroed long before, maybe evicted) back into L2 for the red.adds; prefetch only
        for (int i = (blockIdx.x * WPB * 32 + t) * 32; i < M * n; i += gridDim.x * WPB * 32 * 32)  // (no data dependence)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(out + i));
    asm volatile("griddepcontrol.wait;" ::: "memory");
#ifdef G24_CLK
    const long long ck_w = clock64();
    long long ck_pass = 0;
#endif
    if (WG) {  // s60: x chunks in the wgmma B core-matrix layout: 16 B (8 k) of token q, k8 group kk at (kk * M + q) * 16
        const int nb = min(XB, (tb - 1) / nru - ks0 + 1), per = M * KC / 8;
#ifndef G24_WG_NOX
        for (int i = t; i < nb * per; i += WPB * 32) {
            const int b = i / per, r_ = i % per, q_ = r_ / (KC / 8), kk = r_ % (KC / 8);
            const uint4 v = __ldcg((const uint4*)(x + (long)q_ * k + (long)(ks0 + b) * KC + kk * 8));
            *(uint4*)(smem + WPB * NS * SB + b * M * XS + (kk * M + q_) * 16) = v;
        }
#endif
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    } else if (xsp) {
        const int nb = min(XB, (tb - 1) / nru - ks0 + 1);
        for (int i = t; i < nb * M; i += WPB * 32) {
            const int b = i / M, qq = i % M;
            bulk(xs + b * M * XS + qq * XS, x + (long)qq * k + (long)(ks0 + b) * KC, KC * 2, xfb + 8 * b);
        }
    } else if (t == 0) {  // the block's x chunks
        const int ksl = (tb - 1) / nru;
        for (int b = 0; b < XB && ks0 + b <= ksl; ++b) {
            bar_expect(xfb + 8 * b, M * KC * 2);
            for (int qq = 0; qq < M; ++qq) bulk(xs + b * M * XS + qq * XS, x + (long)qq * k + (long)(ks0 + b) * KC, KC * 2, xfb + 8 * b);
        }
    }
    if (lane == 0) for (; jn < total && jn < NS; ++jn) issue(jn);
    jn = __shfl_sync(0xffffffffu, jn, 0);
    if (!ez) {  // zero this block's slice of out, then arrive on the epoch counter ctr[1] (+gridDim.x per launch, never reset)
        float4* o4 = (float4*)out;
        for (int i = blockIdx.x * WPB * 32 + t; i < M * n / 4; i += gridDim.x * WPB * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        __threadfence();
        __syncthreads();
        if (t == 0) {
            const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
            *tgt = (int)(old - old % gridDim.x + gridDim.x);
        }
        __syncthreads();
    }
    const int target = ez ? 0 : *tgt;
    bool passed = ez;
#ifdef G24_CLK
    const long long ck_setup = clock64() - ck_w;
#endif
#ifdef G24_TRACE
    unsigned long long tr0;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr0));
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[R][NT][4], acc2[R][NT][4];
    // task partials (16 R rows x M tokens, times sc) -> stage tile [tok][row] -> red.add.v4 into out [tok][n]
    auto flush = [&](float (&v)[R][NT][4], int tr_, float sc) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + r * 16 + g + 8 * (e >> 1)] = v[r][jj][e] * sc;
        __syncwarp();
#pragma unroll
        for (int i = lane; i < M * 4 * R; i += 32) {
            const int tok = i / (4 * R), r4 = (i % (4 * R)) * 4;
            const float4 a = *(const float4*)(stg + tok * STR + r4);
            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(out + (long)tok * n + tr_ * 16 * R + r4), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w) : "memory");
        }
        __syncwarp();
    };
    float pend[R][NT][4];
    int pend_tr = -1;
    int j = 0;
#ifdef G24_CLK
    long long ck_loop = 0, ck_end = 0, ck_x = 0, ck_a, ck_fix = 0, ck_bar = 0, ck_t;
#endif
    for (int q = 0; SK ? j < total : q < nq; ++q) {
        const int tq = SK ? (wa + j) / SPK : tmap(ta + wu + q * NU);
        const int st0 = SK ? (wa + j) % SPK : 0, st1 = SK ? min(SPK, st0 + total - j) : SPK;
        const int ks = tq / nrt, tr = tq % nrt, xb = ks - ks0;
#ifndef G24_NOFIX
        // escape headers of rows g, g + 8 of each row group (lane c: slots 2c, 2c + 1), loaded early: used at the task end
        float4 hd[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) hd[r][h] = __ldg((const float4*)(H + (long)((tr * R + r) * 16 + g + 8 * h) * ESC) + c);
#endif
#ifdef G24_CLK
        ck_a = clock64();
#endif
        if (!WG) bar_wait(xfb + 8 * xb, 0);
#ifdef G24_CLK
        { const long long c_ = clock64(); ck_x += c_ - ck_a; ck_a = c_; }
#endif
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[r][jj][e] = acc2[r][jj][e] = 0.f;
        const unsigned xbase = xs + xb * M * XS + xrow;
        if constexpr (WG) {  // s60: decode of stage st + 1 overlaps the wgmmas of stage st (two A register sets)
            const unsigned xa = xs + xb * M * XS;
            float aw[2][NT][4];  // two accumulator chains (k16 halves s = 0 / 1 when G24_WG_A2)
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) aw[i][jj][e] = 0.f;
            auto dec = [&](int j_, uint4 (&af)[AP][2]) {
                const int sl = j_ % NS;
                bar_wait(fb + 8 * sl, (j_ / NS) & 1);
                const unsigned sb = ring + sl * SB + lane * 16, nb = ring + sl * SB + AP * 512 + lane * 8;
#pragma unroll
                for (int a = 0; a < AP; ++a) {
                    uint4 sv; uint2 nv;
                    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + a * 512));
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + a * 256));
                    af[a][0] = dec8(sv.x, sv.y, nv.x);
                    af[a][1] = dec8(sv.z, sv.w, nv.y);
                }
                __syncwarp();
                if (jn < total) {
                    if (lane == 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        issue(jn);
                    }
                    ++jn;
                }
            };
            auto mm = [&](int st_, const uint4 (&af)[AP][2]) {
#ifdef G24_WG_NOMMA
                if (af[0][0].x == 0x12345u) acc[0][0][0] += 1.f;
                return;
#endif
                fence_acc(aw[0]); fence_acc(aw[1]);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
                for (int a = 0; a < AP; ++a)
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const unsigned ad = xa + ((st_ * AP + a) * 4 + s * 2) * M * 16;
                        const unsigned long long dsc = (unsigned long long)((ad & 0x3FFFF) >> 4) | ((unsigned long long)(M * 16 >> 4) << 16) | (8ull << 32);
#ifdef G24_WG_SS
                        {
                            const unsigned aa = (unsigned)__cvta_generic_to_shared(smem) + (w & ~3) * NS * SB + ((a * 2 + s) & 3) * 2048;
                            const unsigned long long ad_ = (unsigned long long)((aa & 0x3FFFF) >> 4) | (8ull << 16) | (16ull << 32);
                            if constexpr (M == 32) wgmma_ss32(aw[G24_WG_A2 ? s : 0], ad_, dsc); else wgmma_ss16(aw[G24_WG_A2 ? s : 0], ad_, dsc);
                            if (af[a][s].x == 0x9876u) aw[0][0][0] += 1.f;
                        }
#else
                        wgmma<M>(aw[G24_WG_A2 ? s : 0], af[a][s], dsc);
#endif
                    }
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            };
#if G24_WG_W1
            // wait_group 1: stage st's wgmmas run while st + 1 is decoded and issued (3 A register sets)
            uint4 f0[AP][2], f1[AP][2], f2[AP][2];
            const int ns_ = st1 - st0;
            dec(j, f0);
            if (ns_ > 1) dec(j + 1, f1);
            for (int i = 0; i < ns_; i += 3) {
                mm(st0 + i, f0);
                if (i + 2 < ns_) dec(j + i + 2, f2);
                WG_WAIT(1);
                if (i + 1 >= ns_) break;
                mm(st0 + i + 1, f1);
                if (i + 3 < ns_) dec(j + i + 3, f0);
                WG_WAIT(1);
                if (i + 2 >= ns_) break;
                mm(st0 + i + 2, f2);
                if (i + 4 < ns_) dec(j + i + 4, f1);
                WG_WAIT(1);
            }
            WG_WAIT(0);
#else
            uint4 fa[AP][2], fz[AP][2];
            dec(j, fa);
            for (int st = st0; st < st1; st += 2) {
                mm(st, fa);
                if (st + 1 < st1) dec(j + st + 1 - st0, fz);
                WG_WAIT(0);
                if (st + 1 < st1) {
                    mm(st + 1, fz);
                    if (st + 2 < st1) dec(j + st + 2 - st0, fa);
                    WG_WAIT(0);
                }
            }
#endif
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[0][jj][e] = aw[0][jj][e] + aw[1][jj][e];
            j += st1 - st0;
        } else
        for (int st = st0; st < st1; ++st, ++j) {
            const int sl = j % NS;
            bar_wait(fb + 8 * sl, (j / NS) & 1);
            const unsigned sb = ring + sl * SB + lane * 16, nb = ring + sl * SB + R * AP * 512 + lane * 8;
            #ifdef G24_G
            constexpr int G = R > 1 ? 1 : AP < G24_G ? AP : G24_G;  // s61: atoms whose smem loads are issued together
#else
            constexpr int G = R > 1 ? 1 : AP < 2 ? AP : 2;
#endif
#pragma unroll
            for (int j0 = 0; j0 < AP; j0 += G) {
                uint4 sv[G][R];
                uint2 nv[G][R];
#pragma unroll
                for (int a = 0; a < G; ++a)
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv[a][r].x), "=r"(sv[a][r].y), "=r"(sv[a][r].z), "=r"(sv[a][r].w) : "r"(sb + (r * AP + j0 + a) * 512));
                        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv[a][r].x), "=r"(nv[a][r].y) : "r"(nb + (r * AP + j0 + a) * 256));
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
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const uint4 af = dec8(s ? sv[a][r].z : sv[a][r].x, s ? sv[a][r].w : sv[a][r].y, s ? nv[a][r].y : nv[a][r].x);
#pragma unroll
                            for (int jj = 0; jj < NT; ++jj) mma(s ? acc2[r][jj] : acc[r][jj], af, bx[a][s][jj][0], bx[a][s][jj][1]);
                        }
            }
            __syncwarp();
#ifndef G24_FB
#define G24_FB 1
#endif
            // s69 FB: refill freed slots in batches of G24_FB (one proxy fence per batch); G24_NOFENCE = timing probe only
            if (jn < total && (G24_FB == 1 || (j + 1) % G24_FB == 0 || jn == j + 1)) {
                if (lane == 0) {
#ifndef G24_NOFENCE
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
                    for (int jj_ = jn; jj_ < total && jj_ < j + 1 + NS; ++jj_) issue(jj_);
                }
                jn = min(total, j + 1 + NS);
            }
        }
#ifdef G24_CLK
        { const long long c_ = clock64(); ck_loop += c_ - ck_a; ck_a = c_; }
#endif
        // task end: partials (16 R rows x M tokens) -> stage tile [tok][row] -> red.add.v4 into out [tok][n]
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[r][jj][e] += acc2[r][jj][e];
#ifndef G24_NOFIX
        {  // escape fixups (g19's register path): escaped weights were decoded as 2^-15 (scale units); add x * (true - 2^-15)
            const int klo = ks * KC + st0 * AP * 32, khi = ks * KC + st1 * AP * 32;
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float4 hq = hd[r][h];
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
                        const unsigned char* xc = smem + WPB * NS * SB + xb * M * XS + (WG ? ((col - ks * KC) >> 3) * M * 16 + ((col - ks * KC) & 7) * 2 : (col - ks * KC) * 2);
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * (WG ? 16 : XS)));
                                acc[r][jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[r][jj][2 * h + b]));
                            }
                    }
                }
            }
        }
#endif
#ifdef G24_CLK
        ck_t = clock64(); ck_fix += ck_t - ck_a;
#endif
        // s61 DF: if the out-zero epoch barrier has not passed yet, keep this task's scaled partials in registers and go
        // on with the next task (its W stream keeps flowing); they are flushed with the next flush (or at the end)
        if (df && !passed) {
            unsigned v = 0;
            if (lane == 0) asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory");
            v = __shfl_sync(0xffffffffu, v, 0);
            if ((int)(v - (unsigned)target) >= 0) {
                passed = true;
#ifdef G24_CLK
                ck_pass = clock64() - ck_w;
#endif
            }
            else if (pend_tr < 0) {
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                        for (int e = 0; e < 4; ++e) pend[r][jj][e] = acc[r][jj][e] * scale;
                pend_tr = tr;
#ifdef G24_CLK
                ck_end += clock64() - ck_a;
#endif
                continue;
            }
        }
        if (!passed) {  // every block has zeroed its slice of out
            if (lane == 0) {
                unsigned v;
                do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - (unsigned)target) < 0);
            }
            passed = true;
#ifdef G24_CLK
            ck_pass = clock64() - ck_w;
#endif
        }
        __syncwarp();
#ifdef G24_CLK
        { const long long c_ = clock64(); ck_bar += c_ - ck_t; }
#endif
        if (pend_tr >= 0) {
            flush(pend, pend_tr, 1.f);
            pend_tr = -1;
        }
        flush(acc, tr, scale);
#ifdef G24_CLK
        ck_end += clock64() - ck_a;
#endif
    }
    if (pend_tr >= 0) {
        if (lane == 0) {
            unsigned v;
            do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - (unsigned)target) < 0);
        }
        __syncwarp();
        flush(pend, pend_tr, 1.f);
    }
#ifdef G24_CLK
    if (lane == 0) { const int id_ = blockIdx.x * WPB + w; clk[8 * id_] = ck_loop; clk[8 * id_ + 1] = ck_end; clk[8 * id_ + 2] = ck_x; clk[8 * id_ + 3] = total; clk[8 * id_ + 4] = ck_fix; clk[8 * id_ + 5] = ck_bar; clk[8 * id_ + 6] = ck_setup; clk[8 * id_ + 7] = ck_pass; }
#endif
    // proj s56 ZN: zero the next g24's out (zo, zn4 float4s) as each warp finishes; plain stores, no fence: the next g24
    // reads it only after its griddepcontrol.wait, i.e. after this grid has completed
    for (int i = (blockIdx.x * WPB + w) * 32 + lane; i < zn4; i += gridDim.x * WPB * 32) ((float4*)zo)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#ifdef G24_TRACE
    if (lane == 0) {
        unsigned long long tr1;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr1));
        const int id = blockIdx.x * WPB + w;
        unsigned smid_;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_));
        trace[3 * id] = tr0; trace[3 * id + 1] = tr1; trace[3 * id + 2] = (unsigned long long)total | ((unsigned long long)smid_ << 32);
    }
#endif
}
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
}  // namespace g24
torch::Tensor g24_clk() {
#ifdef G24_CLK
    auto t = torch::empty({4096 * 8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(t.data_ptr(), g24::clk, sizeof(long long) * 4096 * 8, 0, cudaMemcpyDeviceToDevice);
    return t;
#else
    return torch::empty({0});
#endif
}
#ifdef G24_TRACE
torch::Tensor g24_trace() {
    auto t = torch::empty({4096 * 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(t.data_ptr(), g24::trace, sizeof(unsigned long long) * 4096 * 3, 0, cudaMemcpyDeviceToDevice);
    return t;
}
#else
torch::Tensor g24_trace() { return torch::empty({0}); }
#endif
void g24_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g24::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g24_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t wpb, int64_t xb, int64_t grid, int64_t npre, int64_t rr, torch::Tensor zo) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sk = (rr >> 8) & 1, wg = (rr >> 9) & 1;
    rr &= 0xff;
    const int sm = wpb * ns * rr * ap * g24::WB + xb * m * (kc * 2 + 16) + wpb * m * (16 * rr + 4) * 4 + wpb * ns * 8 + xb * 8 + 16;
    const int nrg = n / (16 * rr) / (wg ? 4 : 1), T = (k / kc) * nrg;
    for (int b = 0; b < grid; ++b) {  // x chunks per block <= xb
        const long spk = kc / 32 / ap, SG = (long)T * spk;
        const int ta = sk ? (int)(SG * b / grid / spk) : (int)((long)T * b / grid);
        const int tb = sk ? (int)((SG * (b + 1) / grid + spk - 1) / spk) : (int)((long)T * (b + 1) / grid);
        TORCH_CHECK(tb == ta || (tb - 1) / nrg - ta / nrg < xb, "g24: block spans more than xb x chunks");
    }
    TORCH_CHECK(n % (16 * rr) == 0 && k % kc == 0 && sm <= 227 * 1024, "g24: shape/smem ", sm);
    TORCH_CHECK(zo.numel() % 4 == 0 && (!zo.numel() || (zo.is_contiguous() && zo.scalar_type() == torch::kFloat32)), "g24: zo");
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G24_CASE(M_, KC_, AP_, NS_, WPB_, XB_, R_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && wpb == WPB_ && xb == XB_ && (rr | sk << 8 | wg << 9) == R_) { \
        auto kern = g24::gemm<M_, KC_, AP_, NS_, WPB_, XB_, R_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(WPB_ * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
                           out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)npre, zo.numel() ? zo.data_ptr<float>() : nullptr, (int)(zo.numel() / 4)); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G24_CONFIGS
    TORCH_CHECK(false, "g24: no config");
}
