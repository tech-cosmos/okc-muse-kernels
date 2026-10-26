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
namespace g24 {
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
// smem: ring [WPB][NS][SB] | x [XB][M][XS] | stage [WPB][M][STR] f32 | full [WPB][NS] | xfull [XB] | target
// R: row groups per task (one x fragment load feeds R atoms); stage (ks, tile of R row groups, st) =
// [R][AP][32 lanes x 16 B sign|mantissa] [R][AP][32 x 8 B codes] (pack19 with rg = R).
template <int M, int KC, int AP, int NS, int WPB, int XB, int RR>
__global__ void __launch_bounds__(WPB * 32, 1) gemm(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int* __restrict__ ctr, int n, int k, int npre) {
    constexpr int R = RR & 0xff;
    constexpr bool SK = RR >> 8, PF = SK && ((RR >> 9) & 1);  // PF (s55): finished warps L2-prefetch block-mates' unissued stages
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = R * AP * WB, XS = KC * 2 + 16, STR = 16 * R + 4;
    static_assert(NA % AP == 0 && NT % 2 == 0 && M <= 32, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    // npre bit 16: trigger the PDL dependent at start (the layer's g19 convention, s57: dependents that only touch out
    // after their own griddepcontrol.wait)
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem) + w * NS * SB, xs = (unsigned)__cvta_generic_to_shared(smem) + WPB * NS * SB;
    float* stg = (float*)(smem + WPB * NS * SB + XB * M * XS) + w * M * STR;
    const unsigned fb = (unsigned)__cvta_generic_to_shared(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4) + w * NS * 8;
    const unsigned xfb = (unsigned)__cvta_generic_to_shared(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4) + WPB * NS * 8;
    int* tgt = (int*)(smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4 + WPB * NS * 8 + XB * 8);
    volatile int* prog = tgt + 4;  // PF: [WPB] next stage each warp will issue, [WPB] prefetched flags
    const int nrt = n / (16 * R), S = k / KC, T = S * nrt;
    // SK (s54, RR bit 8) = stream-K over stages: block b owns global stages [SG b / G, SG (b + 1) / G) (stage g = task
    // g / SPK, st g % SPK), warp w a contiguous 1/WPB of them, flushing its partials at every task boundary it crosses
    const long SG = (long)T * SPK;
    const int sa = SK ? (int)(SG * blockIdx.x / gridDim.x) : 0, sz = SK ? (int)(SG * (blockIdx.x + 1) / gridDim.x) : 0;
    const int wa = sa + (int)((long)(sz - sa) * w / WPB);
    const int ta = SK ? sa / SPK : (int)((long)T * blockIdx.x / gridDim.x);
    const int tb = SK ? (sz + SPK - 1) / SPK : (int)((long)T * (blockIdx.x + 1) / gridDim.x);
    const int ks0 = ta / nrt;
    const int nq = SK ? 0 : tb - ta > w ? (tb - ta - w + WPB - 1) / WPB : 0;
    const int total = SK ? sa + (int)((long)(sz - sa) * (w + 1) / WPB) - wa : nq * SPK;
    auto issue = [&](int j) {  // lane 0: this warp's sequence element j -> slot j % NS
        const int tq = SK ? (wa + j) / SPK : ta + w + (j / SPK) * WPB, st = SK ? (wa + j) % SPK : j % SPK, sl = j % NS;
#ifdef G24_NOW
        bar_arrive(fb + 8 * sl);
#else
        bar_expect(fb + 8 * sl, SB);
        bulk(ring + sl * SB, WP + ((long)tq * SPK + st) * SB, SB, fb + 8 * sl);
#endif
    };
    int jn = 0;  // next sequence element to issue
    if (PF && lane == 0) { prog[w] = wa; prog[WPB + w] = 0; }
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) for (int b = 0; b < XB; ++b) bar_init(xfb + 8 * b, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (; jn < total && jn < npre; ++jn) issue(jn);  // W only, before the predecessor is done
    }
#ifndef G24_XPRE
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
#ifdef G24_XPAR
    if (w == 0) {  // s55: the block's x chunks, row copies spread over warp 0's lanes (thread 0 alone issued XB * M serially)
        const int ksl = (tb - 1) / nrt;
        if (lane == 0) for (int b = 0; b < XB && ks0 + b <= ksl; ++b) bar_expect(xfb + 8 * b, M * KC * 2);
        __syncwarp();
        for (int b = 0; b < XB && ks0 + b <= ksl; ++b)
            for (int qq = lane; qq < M; qq += 32) bulk(xs + b * M * XS + qq * XS, x + (long)qq * k + (long)(ks0 + b) * KC, KC * 2, xfb + 8 * b);
    }
#else
    if (t == 0) {  // the block's x chunks (G24_XPRE: before the wait = timing probe only, NOT correct in a real chain)
        const int ksl = (tb - 1) / nrt;
        for (int b = 0; b < XB && ks0 + b <= ksl; ++b) {
            bar_expect(xfb + 8 * b, M * KC * 2);
            for (int qq = 0; qq < M; ++qq) bulk(xs + b * M * XS + qq * XS, x + (long)qq * k + (long)(ks0 + b) * KC, KC * 2, xfb + 8 * b);
        }
    }
#endif
#ifdef G24_XPRE
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (lane == 0) for (; jn < total && jn < NS; ++jn) issue(jn);
    jn = __shfl_sync(0xffffffffu, jn, 0);
    if (PF && lane == 0) prog[w] = wa + jn;
    {  // zero this block's slice of out, then arrive on the epoch counter ctr[1] (+gridDim.x per launch, never reset)
        float4* o4 = (float4*)out;
#ifndef G24_ZPROBE  // timing probe only: no zeroing / fence (output wrong)
        for (int i = blockIdx.x * WPB * 32 + t; i < M * n / 4; i += gridDim.x * WPB * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        __threadfence();
#endif
#ifdef G24_LZ
        if (t == 0) { tgt[0] = 0; tgt[2] = 0; }
        __syncthreads();  // mbarrier inits; the fence + epoch arrive move to after each warp's first stage (s55 LZ)
    }
    int target = 0;
    bool arrived = false;
    volatile int* zs = tgt;  // [0] warps arrived, [1] target, [2] flag (zeroed before the __syncthreads above)
    auto arrive = [&]() {  // LZ2: this warp's zero stores visible; the block's last warp arrives on ctr[1] (+gridDim.x / launch)
        __threadfence();
        __syncwarp();
        if (lane == 0 && atomicAdd((int*)zs, 1) == WPB - 1) {
            const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
            zs[1] = (int)(old - old % gridDim.x + gridDim.x);
            __threadfence_block();
            zs[2] = 1;
        }
        arrived = true;
    };
#else
        __syncthreads();
        if (t == 0) {
            const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
            *tgt = (int)(old - old % gridDim.x + gridDim.x);
        }
        __syncthreads();
    }
    const int target = *tgt;
#endif
    bool passed = false;
#ifdef G24_TRACE
    unsigned long long tr0;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr0));
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[R][NT][4], acc2[R][NT][4];
    int j = 0;
    for (int q = 0; SK ? j < total : q < nq; ++q) {
        const int tq = SK ? (wa + j) / SPK : ta + w + q * WPB;
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
        bar_wait(xfb + 8 * xb, 0);
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[r][jj][e] = acc2[r][jj][e] = 0.f;
        const unsigned xbase = xs + xb * M * XS + xrow;
        for (int st = st0; st < st1; ++st, ++j) {
            const int sl = j % NS;
            bar_wait(fb + 8 * sl, (j / NS) & 1);
            const unsigned sb = ring + sl * SB + lane * 16, nb = ring + sl * SB + R * AP * 512 + lane * 8;
            constexpr int G = R > 1 ? 1 : AP < 2 ? AP : 2;
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
            if (jn < total) {  // refill the slot just read (generic-proxy reads before the async-proxy write)
                if (lane == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    issue(jn);
                    if (PF) prog[w] = wa + jn + 1;
                }
                ++jn;
            }
#ifdef G24_LZ
            if (!arrived) arrive();
#endif
        }
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
                        const unsigned char* xc = smem + WPB * NS * SB + xb * M * XS + (col - ks * KC) * 2;
#pragma unroll
                        for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                const float xv = __bfloat162float(*(const __nv_bfloat16*)(xc + (jj * 8 + 2 * c + b) * XS));
                                acc[r][jj][2 * h + b] = fmaf(xv, hy, fmaf(xv, -0x1p-15f, acc[r][jj][2 * h + b]));
                            }
                    }
                }
            }
        }
#endif
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + r * 16 + g + 8 * (e >> 1)] = acc[r][jj][e] * scale;
        if (!passed) {  // every block has zeroed its slice of out
            if (lane == 0) {
#ifdef G24_LZ
                while (zs[2] == 0) {}
                __threadfence_block();
                target = zs[1];
#endif
                unsigned v;
                do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - (unsigned)target) < 0);
            }
            passed = true;
        }
        __syncwarp();
#pragma unroll
        for (int i = lane; i < M * 4 * R; i += 32) {
            const int tok = i / (4 * R), r4 = (i % (4 * R)) * 4;
            const float4 a = *(const float4*)(stg + tok * STR + r4);
            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(out + (long)tok * n + tr * 16 * R + r4), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w) : "memory");
        }
        __syncwarp();
    }
#ifdef G24_LZ
    if (!arrived) arrive();  // warps without stages
#endif
    if (PF && lane == 0) {  // done: prefetch the not yet issued remainder of every other warp's range into L2 (once each)
        for (int i = 1; i < WPB; ++i) {
            const int w2 = (w + i) % WPB;
            if (atomicExch((int*)prog + WPB + w2, 1)) continue;
            const int p = prog[w2], e2 = sa + (int)((long)(sz - sa) * (w2 + 1) / WPB);
            if (e2 > p) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(WP + (long)p * SB), "r"((e2 - p) * SB) : "memory");
        }
    }
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
// gemv25 (s55) = gemv24 SK (R = 1) + a DYNAMIC TAIL: per k chunk ks the last pt row-group tasks form a pool of
// chunks (C stages each) that warps claim by a global atomicAdd once their static range is done (only pools of
// the ks chunks whose x the block holds). Static part = SK over the remaining S * (nrt - pt) * SPK stages.
// The ring's sequence is data-driven: lane 0 writes each slot's global stage index (or -1 = end) into meta[slot]
// before the copy (arrive.expect_tx releases it); consumers flush at every task change / stage gap.
// Pool counters ctr[4 + ks] are reset by block 0 before its epoch arrive; claims only after the epoch barrier.
// smem: ring [WPB][NS][SB] | x [XB][M][XS] | stage [WPB][M][20] f32 | full [WPB][NS] | xfull [XB] | tgt 16 B | meta [WPB][NS]
template <int M, int KC, int AP, int NS, int WPB, int XB>
__global__ void __launch_bounds__(WPB * 32, 1) gemm_dt(const unsigned char* __restrict__ WP, const float2* __restrict__ H, float scale, const __nv_bfloat16* __restrict__ x,
                                                       float* __restrict__ out, int* __restrict__ ctr, int n, int k, int npre, int dyn) {
    constexpr int NT = M / 8, NA = KC / 32, SPK = NA / AP, SB = AP * WB, XS = KC * 2 + 16, STR = 20;
    static_assert(NA % AP == 0 && NT % 2 == 0 && M <= 32, "config");
    extern __shared__ __align__(128) unsigned char smem[];
    if (npre >> 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    npre &= 0xffff;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, g = lane >> 2, c = lane & 3;
    const unsigned s0 = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned ring = s0 + w * NS * SB, xs = s0 + WPB * NS * SB;
    float* stg = (float*)(smem + WPB * NS * SB + XB * M * XS) + w * M * STR;
    unsigned char* tl = smem + WPB * NS * SB + XB * M * XS + WPB * M * STR * 4;
    const unsigned fb = (unsigned)__cvta_generic_to_shared(tl) + w * NS * 8, xfb = (unsigned)__cvta_generic_to_shared(tl) + WPB * NS * 8;
    int* tgt = (int*)(tl + WPB * NS * 8 + XB * 8);
    volatile int* meta = (volatile int*)(tl + WPB * NS * 8 + XB * 8 + 16) + w * NS;
    const int nrt = n / 16, S = k / KC, pt = dyn & 0xffff, C = dyn >> 16;
    const int Qs = (nrt - pt) * SPK, nch = pt * SPK / C, QK = nrt * SPK;  // static stages / pool chunks / stages per ks
    const long SGs = (long)S * Qs;
    const int sa = (int)(SGs * blockIdx.x / gridDim.x), sz = (int)(SGs * (blockIdx.x + 1) / gridDim.x);
    const int wa = sa + (int)((long)(sz - sa) * w / WPB), wz = sa + (int)((long)(sz - sa) * (w + 1) / WPB), total = wz - wa;
    const int ks0 = sa / Qs, ksl = (sz > sa ? sz - 1 : sa) / Qs, nres = ksl - ks0 + 1;
    int* pool = ctr + 4;
    // lane 0's issue state
    int jn = 0, pk = total > 0 ? (wz - 1) / Qs : ks0, ntried = 0, cb = 0, ci = C, pv = 0, target = 0;
    bool pend = false, fin = false, ep = false;
    auto stat_gs = [&](int gl) { return (gl / Qs) * QK + gl % Qs; };
    auto issue = [&](int sl, int gs) {
        if (gs < 0) {
            meta[sl] = -1;
            bar_arrive(fb + 8 * sl);
            return;
        }
        meta[sl] = gs;
#ifdef G24_NOW
        bar_arrive(fb + 8 * sl);
#else
        bar_expect(fb + 8 * sl, SB);
        bulk(ring + sl * SB, WP + (long)gs * SB, SB, fb + 8 * sl);
#endif
    };
    auto claim = [&]() {
        if (!ep) {
            unsigned v;
            do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - (unsigned)target) < 0);
            ep = true;
        }
        pv = atomicAdd(pool + pk, 1);
        pend = true;
    };
    auto next = [&]() -> int {  // lane 0: the next sequence element's global stage index, -1 = done
        if (jn < total) {
            if (jn == total - NS && nch) claim();  // first claim NS stages ahead (latency hidden; committed)
            return stat_gs(wa + jn++);
        }
        if (ci == C) {  // only the pool of pk (the warp's last static ks): no blocking fallback claims
            if (!pend) {
                if (ntried || nch == 0) return -1;
                claim();
            }
            const int v = pv;
            pend = false;
            if (v >= nch) { ntried = 1; return -1; }
            cb = pk * QK + Qs + v * C;
            ci = 0;
            claim();  // pre-claim the next chunk (committed: it will be processed)
        }
        return cb + ci++;
    };
    if (lane == 0) {
        for (int i = 0; i < NS; ++i) bar_init(fb + 8 * i, 1);
        if (w == 0) for (int b = 0; b < XB; ++b) bar_init(xfb + 8 * b, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (; jn < total && jn < npre && jn < NS; ) issue(jn, stat_gs(wa + jn)), ++jn;  // W only, before the predecessor is done
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (t == 0) {  // the block's x chunks
        for (int b = 0; b < XB && ks0 + b <= ksl; ++b) {
            bar_expect(xfb + 8 * b, M * KC * 2);
            for (int qq = 0; qq < M; ++qq) bulk(xs + b * M * XS + qq * XS, x + (long)qq * k + (long)(ks0 + b) * KC, KC * 2, xfb + 8 * b);
        }
    }
    {  // zero this block's slice of out (+ block 0: the pool counters), then arrive on the epoch counter ctr[1]
        float4* o4 = (float4*)out;
        for (int i = blockIdx.x * WPB * 32 + t; i < M * n / 4; i += gridDim.x * WPB * 32) o4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (blockIdx.x == 0 && t < S) pool[t] = 0;
        __threadfence();
        __syncthreads();
        if (t == 0) {
            const unsigned old = atomicAdd((unsigned*)ctr + 1, 1u);
            *tgt = (int)(old - old % gridDim.x + gridDim.x);
        }
        __syncthreads();
    }
    target = *tgt;
    if (lane == 0) {
        for (int i = jn; i < NS && !fin; ++i) {
            const int gs = next();
            issue(i, gs);
            fin = gs < 0;
        }
    }
    bool passed = false;
#ifdef G24_TRACE
    unsigned long long tr0;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr0));
    int nst = 0;
#endif
    const unsigned xrow = ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 16;
    float acc[NT][4], acc2[NT][4];
    float4 hd[2];
    int cur = -1, sto = 0, stp = -2, ks = 0, tr = 0, xb = 0;
    for (int j = 0;; ++j) {
        const int sl = j % NS;
        bar_wait(fb + 8 * sl, (j / NS) & 1);
        const int gs = meta[sl];
        const int tq = gs / SPK, st = gs % SPK;
        if (cur >= 0 && (gs < 0 || tq != cur || st != stp + 1)) {  // flush task cur's segment [sto, stp]
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[jj][e] += acc2[jj][e];
#ifndef G24_NOFIX
            {  // escape fixups (g19's register path) restricted to the segment's k range
                const int klo = ks * KC + sto * AP * 32, khi = ks * KC + (stp + 1) * AP * 32;
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
                            const unsigned char* xc = smem + WPB * NS * SB + xb * M * XS + (col - ks * KC) * 2;
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
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) stg[(jj * 8 + 2 * c + (e & 1)) * STR + g + 8 * (e >> 1)] = acc[jj][e] * scale;
            if (!passed) {  // every block has zeroed its slice of out
                if (lane == 0) {
                    unsigned v;
                    do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(ctr + 1) : "memory"); } while ((int)(v - (unsigned)target) < 0);
                }
                passed = true;
            }
            __syncwarp();
#pragma unroll
            for (int i = lane; i < M * 4; i += 32) {
                const int tok = i / 4, r4 = (i % 4) * 4;
                const float4 a = *(const float4*)(stg + tok * STR + r4);
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(out + (long)tok * n + tr * 16 + r4), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w) : "memory");
            }
            __syncwarp();
            cur = -1;
        }
        if (gs < 0) break;
        if (cur < 0) {  // new segment
            cur = tq; sto = st; ks = tq / nrt; tr = tq % nrt; xb = ks - ks0;
#ifndef G24_NOFIX
#pragma unroll
            for (int h = 0; h < 2; ++h) hd[h] = __ldg((const float4*)(H + (long)(tr * 16 + g + 8 * h) * ESC) + c);
#endif
            bar_wait(xfb + 8 * xb, 0);
#pragma unroll
            for (int jj = 0; jj < NT; ++jj)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[jj][e] = acc2[jj][e] = 0.f;
        }
        stp = st;
#ifdef G24_TRACE
        ++nst;
#endif
        const unsigned xbase = xs + xb * M * XS + xrow;
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
        if (lane == 0 && !fin) {  // refill the slot just read (generic-proxy reads before the async-proxy write)
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            const int gs2 = next();
            issue(sl, gs2);
            fin = gs2 < 0;
        }
    }
#ifdef G24_TRACE
    if (lane == 0) {
        unsigned long long tr1;
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tr1));
        const int id = blockIdx.x * WPB + w;
        unsigned smid_;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_));
        trace[3 * id] = tr0; trace[3 * id + 1] = tr1; trace[3 * id + 2] = (unsigned long long)nst | ((unsigned long long)smid_ << 32);
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
              int64_t kc, int64_t ap, int64_t ns, int64_t wpb, int64_t xb, int64_t grid, int64_t npre, int64_t rr) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sk = rr >> 8;
    rr &= 0xff;
    const int sm = wpb * ns * rr * ap * g24::WB + xb * m * (kc * 2 + 16) + wpb * m * (16 * rr + 4) * 4 + wpb * ns * 8 + xb * 8 + 16 + wpb * 8;
    const int nrg = n / (16 * rr), T = (k / kc) * nrg;
    for (int b = 0; b < grid; ++b) {  // x chunks per block <= xb
        const long spk = kc / 32 / ap, SG = (long)T * spk;
        const int ta = sk ? (int)(SG * b / grid / spk) : (int)((long)T * b / grid);
        const int tb = sk ? (int)((SG * (b + 1) / grid + spk - 1) / spk) : (int)((long)T * (b + 1) / grid);
        TORCH_CHECK(tb == ta || (tb - 1) / nrg - ta / nrg < xb, "g24: block spans more than xb x chunks");
    }
    TORCH_CHECK(n % (16 * rr) == 0 && k % kc == 0 && sm <= 227 * 1024, "g24: shape/smem ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G24_CASE(M_, KC_, AP_, NS_, WPB_, XB_, R_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && wpb == WPB_ && xb == XB_ && (rr | sk << 8) == R_) { \
        auto kern = g24::gemm<M_, KC_, AP_, NS_, WPB_, XB_, R_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(WPB_ * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
                           out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G24_CONFIGS
    TORCH_CHECK(false, "g24: no config");
}

void g25_gemm(torch::Tensor wp, torch::Tensor h, double scale, torch::Tensor x, torch::Tensor out, torch::Tensor ctr,
              int64_t kc, int64_t ap, int64_t ns, int64_t wpb, int64_t xb, int64_t grid, int64_t npre, int64_t dyn) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = wpb * ns * ap * g24::WB + xb * m * (kc * 2 + 16) + wpb * m * 20 * 4 + wpb * ns * 8 + xb * 8 + 16 + wpb * ns * 4;
    const int nrt = n / 16, S = k / kc, spk = kc / 32 / ap, pt = dyn & 0xffff, C = dyn >> 16;
    TORCH_CHECK(n % 16 == 0 && k % kc == 0 && sm <= 227 * 1024 && pt < nrt && C >= 1 && (pt * spk) % C == 0 && ctr.numel() >= 4 + S, "g25: shape/smem/dyn ", sm);
    const long SGs = (long)S * (nrt - pt) * spk, Qs = (long)(nrt - pt) * spk;
    for (int b = 0; b < grid; ++b) {
        const long sa = SGs * b / grid, sz = SGs * (b + 1) / grid;
        TORCH_CHECK((sz > sa ? sz - 1 : sa) / Qs - sa / Qs < xb, "g25: block spans more than xb x chunks");
    }
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G25_CASE(M_, KC_, AP_, NS_, WPB_, XB_) \
    if (m == M_ && kc == KC_ && ap == AP_ && ns == NS_ && wpb == WPB_ && xb == XB_) { \
        auto kern = g24::gemm_dt<M_, KC_, AP_, NS_, WPB_, XB_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(WPB_ * 32); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (const float2*)h.data_ptr(), (float)scale, (const __nv_bfloat16*)x.data_ptr(), \
                           out.data_ptr<float>(), ctr.data_ptr<int>(), n, k, (int)npre, (int)dyn); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G25_CONFIGS
    TORCH_CHECK(false, "g25: no config");
}
