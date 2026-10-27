// gemv28 (s64): m <= 64 tokens GEMM over lossless 12-bit weights (DEC3), "decode to smem, wide-N wgmma".
// Every earlier wgmma try (g20, g22, s60 WG) put the decoded W in registers as the wgmma A operand with the tokens as N
// (n16/n32): narrow-N wgmma is no faster than mma.sync, and decode + mma add serially in one warp. Here the roles flip:
// A = x (TMA box [64 token rows (OOB rows zero-filled)][64 k], SW128), B = decoded W ([BN rows][64 k] BF16, SW128,
// K-major) in smem, wgmma m64nBNk16 SS (async). Decoder warpgroups turn raw 12-bit stages into B tiles while the MMA
// warpgroup's wgmmas run: decode and tensor work overlap across warpgroups.
// Warpgroups: WG0 = MMA (accumulator BN/2 regs/thread), WG1-2 = decoders, WG3 = producer (one thread issues TMA).
// Work: stage u = tile * SPT + ks (tile-major; tile = BN rows, ks = 64-k step); block b streams stages
// [U b / G, U (b + 1) / G) (stream-K), flushing acc * scale by red.global.add at each run end (tile change / range end).
// out FP32 [M, n] must be zero before the call (prototype). Escapes: none (G28: x @ W'^T, escaped = 2^-15 * scale).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#ifndef DECW
#define DECW 0
#endif
namespace g28 {
__device__ unsigned long long trace[3 * 256];
#ifdef G28_NOTR
#define TRON 0
#else
#define TRON 1
#endif
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel3) {
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(0x38383838u), "r"(sel3));
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
__device__ __forceinline__ void bar_init(unsigned b, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count)); }
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
#ifdef G28_NOARR
__device__ __forceinline__ void bar_arrive(unsigned b) {}
#else
__device__ __forceinline__ void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
#endif
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
#ifdef G28_TESTWAIT
    asm volatile("{\n.reg .pred p;\nW28_%=:\nmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W28_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nW28_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W28_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
#endif
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
#ifdef G28_NOWG
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
#else
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
#endif
// K-major SW128 descriptor: rows of 128 B, 8-row core groups 1024 B apart
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int BN> __device__ __forceinline__ void wgmma_ss(float* d, unsigned long long da, unsigned long long db);
WGMMA_DEFS
__device__ __forceinline__ void red2(float* p, float a, float b) {
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory");
}
// smem: raw ring [NS][WB + XBY] (W raw stage, then the x box) | dec ring [ND][BN * 128] | barriers
// W raw stage (BN rows x 64 k): [BN * 8 units x 8 B sign|mantissa (DEC3)][BN * 8 units x 4 B codes], unit = (row, 8 k)
template <int M, int BN, int NS, int ND, int KA>
__global__ void __launch_bounds__(512, 1) gemm(const unsigned char* __restrict__ WP, float scale, const __grid_constant__ CUtensorMap xmap,
                                               float* __restrict__ out, int n, int k, int npre) {
    constexpr int WB = KA * BN * 96, XBY = KA * 64 * 128, SL = WB + XBY, DB = KA * BN * 128, NACC = BN / 2;
    extern __shared__ __align__(1024) unsigned char smem_[];
    unsigned char* smem = (unsigned char*)(((unsigned long)smem_ + 1023) & ~1023ul);
    const unsigned sb = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned raw = sb, dec = sb + NS * SL, bars = dec + ND * DB;
    const unsigned rfull = bars, rempty = bars + 8 * NS, dfull = bars + 16 * NS, dempty = bars + 16 * NS + 8 * ND;
    const int t = threadIdx.x, wg = __shfl_sync(0xffffffffu, t >> 7, 0), lane = t & 31;
    const int SPT = k / (64 * KA), U = (n / BN) * SPT;
    const int ua = (int)((long)U * blockIdx.x / gridDim.x), ub = (int)((long)U * (blockIdx.x + 1) / gridDim.x), tot = ub - ua;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(rfull + 8 * i, 1); bar_init(rempty + 8 * i, 9); }
        for (int i = 0; i < ND; ++i) { bar_init(dfull + 8 * i, 8); bar_init(dempty + 8 * i, 1); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
#ifdef G28_EMPTY
    return;
#endif
#ifdef G28_SOLO
    if (wg >= 1) return;
#endif
    if (wg == 3) {  // producer
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory");
        if (t == 384) {
            const int np = min(min(npre, NS), tot);
            for (int j = 0; j < np; ++j) {
#if defined(G28_NOW) && defined(G28_NOX)
#elif defined(G28_NOW)
                bar_expect(rfull + 8 * j, XBY);
#else
                bar_expect(rfull + 8 * j, SL);
                bulk(raw + j * SL, WP + (long)(ua + j) * WB, WB, rfull + 8 * j);
#endif
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
#ifdef G28_NOX
            for (int j = 0; j < np; ++j) bar_arrive(rfull + 8 * j);
#else
            for (int j = 0; j < np; ++j)
                for (int a = 0; a < KA; ++a) tma2d(raw + j * SL + WB + a * 8192, &xmap, ((ua + j) % SPT) * 64 * KA + 64 * a, 0, rfull + 8 * j);
#endif
            for (int j = np; j < tot; ++j) {
                const int s = j % NS;
                if (j >= NS) bar_wait(rempty + 8 * s, ((j / NS) + 1) & 1);
#if defined(G28_NOW) && defined(G28_NOX)
#elif defined(G28_NOW)
                bar_expect(rfull + 8 * s, XBY);
#else
                bar_expect(rfull + 8 * s, SL);
                bulk(raw + s * SL, WP + (long)(ua + j) * WB, WB, rfull + 8 * s);
#endif
#ifdef G28_NOX
                bar_arrive(rfull + 8 * s);
#else
                for (int a = 0; a < KA; ++a) tma2d(raw + s * SL + WB + a * 8192, &xmap, ((ua + j) % SPT) * 64 * KA + 64 * a, 0, rfull + 8 * s);
#endif
            }
        }
        return;
    }
    if (wg >= 1) {  // decoders: 256 threads, unit pairs p = it * 256 + i (row p / 4, chunks 2 (p % 4) + 0/1)
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
        const int i = t - 128;
        for (int j = 0; j < tot; ++j) {
            const int s = j % NS, d = j % ND;
            bar_wait(rfull + 8 * s, (j / NS) & 1);
            if (j >= ND) bar_wait(dempty + 8 * d, ((j / ND) + 1) & 1);
            const unsigned char* rs = smem + s * SL;
            unsigned char* ds = smem + NS * SL + d * DB;
#ifndef G28_NODEC
#pragma unroll
            for (int it = 0; it < KA * BN / 64; ++it) {
                const int a = it / (BN / 64), p = (it % (BN / 64)) * 256 + i, r = p >> 2, c0 = (p & 3) * 2;
                const uint4 b = *(const uint4*)(rs + a * BN * 96 + p * 16);
                const uint2 cw = *(const uint2*)(rs + a * BN * 96 + BN * 64 + p * 8);
                const uint4 v0 = dec8(b.x, b.y, cw.x), v1 = dec8(b.z, b.w, cw.y);
                *(uint4*)(ds + a * BN * 128 + r * 128 + ((c0 ^ (r & 7)) << 4)) = v0;
                *(uint4*)(ds + a * BN * 128 + r * 128 + (((c0 + 1) ^ (r & 7)) << 4)) = v1;
            }
#endif
#ifndef G28_NOFENCE
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
            if (TRON && blockIdx.x == 0 && i == DECW && j < 256) trace[256 + j] = gt();
            __syncwarp();
            if (lane == 0) { bar_arrive(dfull + 8 * d); bar_arrive(rempty + 8 * s); }
        }
        return;
    }
    // MMA warpgroup
#ifndef G28_SOLO
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;" ::: "memory");
#endif
    float acc[NACC];
#pragma unroll
    for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
    const int w = t >> 5;
    auto flush = [&](int tile) {
        const int r0 = 16 * w + (lane >> 2), cb = tile * BN + 2 * (lane & 3);
#pragma unroll
        for (int jj = 0; jj < BN / 8; ++jj) {
#ifdef G28_NOFLUSH
            if (r0 < 0)
#else
            if (r0 < M)
#endif
                red2(out + (long)r0 * n + cb + 8 * jj, acc[4 * jj] * scale, acc[4 * jj + 1] * scale);
            if (r0 + 8 < M) red2(out + (long)(r0 + 8) * n + cb + 8 * jj, acc[4 * jj + 2] * scale, acc[4 * jj + 3] * scale);
            acc[4 * jj] = acc[4 * jj + 1] = acc[4 * jj + 2] = acc[4 * jj + 3] = 0.f;
        }
    };
    for (int j = 0; j < tot; ++j) {
        const int s = j % NS, d = j % ND;
#ifndef G28_SOLO
#ifndef G28_MMA_NORF
        bar_wait(rfull + 8 * s, (j / NS) & 1);
#endif
        bar_wait(dfull + 8 * d, (j / ND) & 1);
#endif
        if (TRON && blockIdx.x == 0 && t == 0 && j < 256) trace[j] = gt();
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 4 * KA; ++kk)
#ifndef G28_NOMMA
            wgmma_ss<BN>(acc, sw128_desc(raw + s * SL + WB + (kk >> 2) * 8192 + (kk & 3) * 32), sw128_desc(dec + d * DB + (kk >> 2) * BN * 128 + (kk & 3) * 32));
#endif
        wg_commit();
        if (TRON && blockIdx.x == 0 && t == 0 && j < 256) trace[512 + j] = gt();
        const bool last = j + 1 == tot || (ua + j + 1) % SPT == 0;
        if (last) {
            wg_wait<0>();
#pragma unroll
            for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q])::"memory");
            if (t == 0) {
                if (j > 0 && (ua + j) % SPT != 0) { bar_arrive(dempty + 8 * ((j - 1) % ND)); bar_arrive(rempty + 8 * ((j - 1) % NS)); }
                bar_arrive(dempty + 8 * d); bar_arrive(rempty + 8 * s);
            }
            flush((ua + j) / SPT);
        } else {
            wg_wait<1>();
            if (j > 0 && (ua + j) % SPT != 0 && t == 0) { bar_arrive(dempty + 8 * ((j - 1) % ND)); bar_arrive(rempty + 8 * ((j - 1) % NS)); }
        }
    }
}
__global__ void pred(const uint4* __restrict__ src, uint4* __restrict__ dst, int n8, int ns_) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += gridDim.x * blockDim.x) dst[i] = src[i];
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)ns_);
}
}  // namespace g28
static PFN_cuTensorMapEncodeTiled_v12000 g28_encode = nullptr;
static void g28_xmap(CUtensorMap& map, void* ptr, int m, int k) {
    if (!g28_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g28_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g28_encode, "g28: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, 64};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g28_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g28: cuTensorMapEncodeTiled failed ", (int)r);
}
torch::Tensor g28_trace() { auto o = torch::zeros({3 * 256}, torch::dtype(torch::kInt64).device(torch::kCUDA)); cudaMemcpyFromSymbol(o.data_ptr(), g28::trace, 8 * 3 * 256, 0, cudaMemcpyDeviceToDevice); return o; }
void g28_pred(torch::Tensor src, torch::Tensor dst, int64_t ns_) {
    g28::pred<<<32, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), src.numel() * src.element_size() / 16, ns_);
}
void g28_gemm(torch::Tensor wp, double scale, torch::Tensor x, torch::Tensor out, int64_t bn, int64_t ns, int64_t nd, int64_t ka, int64_t grid, int64_t npre) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = 1024 + ns * ka * (bn * 96 + 64 * 128) + nd * ka * bn * 128 + 16 * ns + 16 * nd;
    CUtensorMap xmap;
    g28_xmap(xmap, x.data_ptr(), m, k);
    TORCH_CHECK(ns >= 2 && nd >= 2 && n % bn == 0 && k % (64 * ka) == 0 && sm <= 227 * 1024 && m <= 64, "g28: shape/smem ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
#define G28_CASE(M_, BN_, NS_, ND_, KA_) \
    if (m == M_ && bn == BN_ && ns == NS_ && nd == ND_ && ka == KA_) { \
        auto kern = g28::gemm<M_, BN_, NS_, ND_, KA_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[1]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        cfg.attrs = attr; cfg.numAttrs = 1; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (float)scale, xmap, out.data_ptr<float>(), n, k, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G28_CONFIGS
    TORCH_CHECK(false, "g28: no config");
}
