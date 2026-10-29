// n7s15 gate: physical-order decode full with the inversion FUSED into the streaming kernel.
// One CTA per SM (132), CTA w owns physical window w (WS rows). A producer warp streams the window's K and V rows
// densely (cp.async.bulk, NST x CH-row stages) from the first instruction. Meanwhile 32 work warps:
//   MODE 1: (a) p1: counting-sort slice s (= request s / S, part s % S, ENT positions) of the page table by window,
//           write u16 offsets + per-window run starts; (b) grid barrier (monotonic counter); (c) map build: own[] =
//           request id per window row from the 128 runs; then consume.
//   MODE 0: no inversion (own[] = 0), consume at once: the pure dense-stream time.
// Consumer (stand-in for the attention math): per stage each warp takes CH/32 rows, reads owner + one K word.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

#define NWP 24
#define NT (NWP * 32 + 32)
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(unsigned long long* b, int c) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(sa(b)), "r"(c)); }
__device__ __forceinline__ void mb_expect(unsigned long long* b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_wait(unsigned long long* b, unsigned ph) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra W;\n}" ::"r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void bulk(void* dst, const void* src, unsigned bytes, unsigned long long* b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }

template <int PER, int CH, int NST, int MODE>
__global__ void __launch_bounds__(NT, 1) fz(const uint4* __restrict__ kc, const uint4* __restrict__ vc, int rows, int WS,
        unsigned MG, int SH, const int* __restrict__ pt, const int* __restrict__ seq, int MP, int S, int NSL,
        unsigned short* __restrict__ out, unsigned short* __restrict__ off, unsigned* cnt, int* sink, unsigned long long* ts) {
    auto TS = [&](int k) { if (threadIdx.x == 0 && ts) { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); ts[blockIdx.x * 8 + k] = g; } };
    TS(0);
    constexpr int ENT = PER * NWP * 32, NW = 132;
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* ring = smem;                                     // NST x (K CH*128 | V CH*128)
    unsigned short* srt = (unsigned short*)(smem + NST * CH * 256); // ENT
    unsigned char* own = (unsigned char*)(srt + ENT);               // WS (+pad)
    int* hist = (int*)(own + ((WS + 15) & ~15));                    // NW + 1
    __shared__ unsigned long long full[NST], empty[NST];
    const int w = blockIdx.x, tid = threadIdx.x, wp = tid >> 5, lane = tid & 31;
    const int r0 = w * WS, nr = max(0, min(WS, rows - r0)), nch = (nr + CH - 1) / CH;
    if (tid == 0) for (int s = 0; s < NST; ++s) { mb_init(&full[s], 1); mb_init(&empty[s], NWP); }
    for (int i = tid; i <= NW; i += NT) hist[i] = 0;
    __syncthreads();
    if (wp == NWP) {  // producer
        if (lane == 0)
            for (int c = 0; c < nch; ++c) {
                const int s = c % NST;
                if (c >= NST) mb_wait(&empty[s], ((c / NST) - 1) & 1);
                const int n = min(CH, nr - c * CH);
                mb_expect(&full[s], n * 256);
                bulk(ring + s * CH * 256, kc + (long)(r0 + c * CH) * 8, n * 128, &full[s]);
                bulk(ring + s * CH * 256 + CH * 128, vc + (long)(r0 + c * CH) * 8, n * 128, &full[s]);
            }
        return;
    }
    if (MODE >= 1) {
        const int t = tid;  // 0 .. NWP*32-1
        int p[PER];
        if (w < NSL) {
            const int b = w / S, t0 = (w % S) * ENT, n = max(0, min(ENT, seq[b] + 1 - t0));
            const int* row = pt + (long)b * MP + t0 + t;
#pragma unroll
            for (int k = 0; k < PER; ++k) p[k] = k * NWP * 32 + t < n ? __ldg(row + k * NWP * 32) : -1;
            unsigned rk[PER];
            if (MODE == 3) { int z = 0;
#pragma unroll
                for (int k = 0; k < PER; ++k) z += p[k];
                if (z == 0x7fffffff) sink[1] = z; }
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (MODE != 3 && p[k] >= 0) rk[k] = atomicAdd(&hist[__umulhi((unsigned)p[k], MG) >> SH], 1);
            asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
            if (t < 32) {
                constexpr int PL = (NW + 31) / 32;
                int c[PL], sum = 0;
#pragma unroll
                for (int k = 0; k < PL; ++k) { const int i = t * PL + k; c[k] = i < NW ? hist[i] : 0; sum += c[k]; }
                int inc = sum;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(~0u, inc, o); if (t >= o) inc += y; }
                int run = inc - sum;
                __syncwarp();
#pragma unroll
                for (int k = 0; k < PL; ++k) { const int i = t * PL + k; if (i < NW) hist[i] = run; run += c[k]; }
                if (t == 31) hist[NW] = run;
            }
            asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (MODE == 1 && p[k] >= 0) {
                    const unsigned ww = __umulhi((unsigned)p[k], MG) >> SH;
                    srt[hist[ww] + rk[k]] = (unsigned short)(p[k] - ww * WS);
                }
            asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
            uint4* o16 = reinterpret_cast<uint4*>(out + (long)w * ENT);
            const uint4* s16 = reinterpret_cast<const uint4*>(srt);
            for (int i = t; i < (n + 7) / 8; i += NWP * 32) o16[i] = s16[i];
            for (int i = t; i <= NW; i += NWP * 32) off[(long)w * (NW + 1) + i] = (unsigned short)hist[i];
        }
        TS(1);
        // own = 0xff (holes), grid barrier
        for (int i = t; i < (WS + 3) / 4; i += NWP * 32) reinterpret_cast<unsigned*>(own)[i] = 0xffffffffu;
        __threadfence();
        asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
        if (t == 0) {
            const unsigned old = atomicAdd(cnt, 1u), tgt = (old / gridDim.x + 1) * gridDim.x;
            unsigned v;
            do { asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(cnt)); } while ((int)(v - tgt) < 0);
        }
        asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
        TS(2);
        // map build: one warp per slice, 16-B vector loads of the run.
        for (int sl = wp; sl < NSL; sl += NWP) {
            const unsigned short* o = off + (long)sl * (NW + 1) + w;
            const int a = __ldcg(o), e = __ldcg(o + 1);
            const unsigned char bb = (unsigned char)(sl / S);
            const unsigned short* src = out + (long)sl * ENT;
            for (int i = (a & ~7) + lane * 8; i < e; i += 256) {
                const uint4 v = __ldcg(reinterpret_cast<const uint4*>(src + i));
                const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int k = i + j;
                    if (k >= a && k < e) own[(u[j >> 1] >> (16 * (j & 1))) & 0xffff] = bb;
                }
            }
        }
        asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
        TS(3);
    } else {
        for (int i = tid; i < (WS + 3) / 4; i += NWP * 32) reinterpret_cast<unsigned*>(own)[i] = 0;
        asm volatile("bar.sync 1, %0;" ::"n"(NWP * 32));
    }
    int z = 0;
    for (int c = 0; c < nch; ++c) {
        const int s = c % NST;
        mb_wait(&full[s], (c / NST) & 1);
        const int n = min(CH, nr - c * CH);
        for (int r = wp * (CH / NWP) + lane; r < (wp + 1) * (CH / NWP) && r < n; r += 32) {
            const unsigned ob = own[c * CH + r];
            const unsigned kw = *reinterpret_cast<const unsigned*>(ring + s * CH * 256 + r * 128);
            const unsigned vw = *reinterpret_cast<const unsigned*>(ring + s * CH * 256 + CH * 128 + r * 128);
            z += (ob != 0xff) ? (int)(kw ^ vw ^ ob) : 0;
        }
        __syncwarp();
        if (lane == 0) mb_arrive(&empty[s]);
    }
    TS(4);
    if (z == 0x7fffffff) sink[0] = z;
}

void fz_run(torch::Tensor kc, torch::Tensor vc, int64_t ws, int64_t mg, int64_t sh, torch::Tensor pt, torch::Tensor seq,
            int64_t S, torch::Tensor out, torch::Tensor off, torch::Tensor cnt, torch::Tensor sink, torch::Tensor ts, int64_t per,
            int64_t mode, int64_t ch, int64_t nst) {
    const int rows = kc.size(0), MP = pt.size(1), B = seq.size(0), NSL = B * S;
    auto st = c10::cuda::getCurrentCUDAStream();
#define FZ(P, C, N, M) { auto k = fz<P, C, N, M>; const int ENT = P * NWP * 32; \
    const int smem = N * C * 256 + ENT * 2 + (((int)ws + 15) & ~15) + 133 * 4; \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
    k<<<132, NT, smem, st>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), rows, (int)ws, (unsigned)mg, (int)sh, \
        pt.data_ptr<int>(), seq.data_ptr<int>(), MP, (int)S, NSL, (unsigned short*)out.data_ptr(), \
        (unsigned short*)off.data_ptr(), (unsigned*)cnt.data_ptr(), sink.data_ptr<int>(), ts.numel() ? (unsigned long long*)ts.data_ptr() : nullptr); }
#define FZM(P, C, N) { if (mode == 1) FZ(P, C, N, 1) else if (mode == 2) FZ(P, C, N, 2) else if (mode == 3) FZ(P, C, N, 3) else FZ(P, C, N, 0) }
    if (per == 14) { if (ch == 128) FZM(14, 128, 4) else FZM(14, 256, 2) }
    else { if (ch == 128) FZM(28, 128, 4) else FZM(28, 256, 2) }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
