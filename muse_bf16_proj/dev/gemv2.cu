// Split-K streaming skinny GEMM out[m, n] = x[m, k] @ W[n, k]^T for m <= 32 (BF16 in, FP32 accumulate).
// Swap-AB mma.sync m16n8k16: A = 16 weight rows (one "stripe"), B = 8 tokens. W is repacked in prepare() into
// [stripe][k-tile][lane][8] so a 16x16 tile's A fragments are 512 contiguous bytes (lane l: 16 B at 16 l).
// Block = (group of SP stripes, K range). Warp NW (producer) streams NS stages through smem with cp.async.bulk
// (TMA 1-D): per stage SP W segments of C tiles and m x rows of C*16 values. NW consumer warps split each stage's
// tiles, accumulate in registers over the whole K range, reduce through smem atomics; with split K, FP32
// red.global into a workspace and the last block of a group writes the output and re-zeroes workspace + counter.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace pj {
__device__ __forceinline__ void mma(float* d, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned* r, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ uint4 lds128(unsigned addr) {
    uint4 r;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr));
    return r;
}
__device__ __forceinline__ void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile(
        "{\n .reg .pred p;\n LAB_WAIT:\n"
        " mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        " @!p bra LAB_WAIT;\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma1d(unsigned dst, const void* src, unsigned bytes, unsigned bar, uint64_t pol) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma1d_nohint(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

template <int MT, int SP, int C, int NW, int NS>
struct Cfg {
    static constexpr int WB = SP * C * 512;          // W bytes per stage
    static constexpr int XLD = C * 16 + 8;           // x row stride (bf16) in smem
    static constexpr int XB = MT * 8 * XLD * 2;      // x bytes per stage
    static constexpr int SB = WB + XB;
    static constexpr int SMEM = NS * SB + MT * 8 * SP * 16 * 4 + 2 * NS * 8 + 16;
};

template <int MT, int SP, int C, int NW, int NS, bool F32OUT>
__global__ void __launch_bounds__((NW + 1) * 32) gemv(const __nv_bfloat16* __restrict__ w,
                                                      const __nv_bfloat16* __restrict__ x, void* __restrict__ out,
                                                      float* __restrict__ ws, int* __restrict__ cnt, int m, int n,
                                                      int k, int kx) {
    using G = Cfg<MT, SP, C, NW, NS>;
    constexpr int TW = C / NW;  // tiles per consumer warp per stage
    static_assert(C % NW == 0 && TW % 2 == 0, "tiles");
    extern __shared__ __align__(128) unsigned char smem[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    float* red = reinterpret_cast<float*>(smem + NS * G::SB);
    const unsigned full = sbase + NS * G::SB + MT * 8 * SP * 16 * 4, empty = full + NS * 8;
    __shared__ int last;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kt16 = k >> 4;
    const int it_n = k / (gridDim.y * C * 16);           // stages this block streams
    const int kt_begin = blockIdx.y * it_n * C;           // first k tile

    if (threadIdx.x == 0) {
        for (int s = 0; s < NS; ++s) { mbar_init(full + 8 * s, 1); mbar_init(empty + 8 * s, NW); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    // zero the padding x rows (never written by TMA) and the reduction buffer
    for (int i = threadIdx.x; i < NS * (MT * 8 - m) * G::XLD / 8; i += (NW + 1) * 32) {
        const int s = i / ((MT * 8 - m) * G::XLD / 8), r = i % ((MT * 8 - m) * G::XLD / 8);
        *reinterpret_cast<uint4*>(smem + s * G::SB + G::WB + m * G::XLD * 2 + r * 16) = make_uint4(0, 0, 0, 0);
    }
    for (int i = threadIdx.x; i < MT * 8 * SP * 16; i += (NW + 1) * 32) red[i] = 0.f;
    __syncthreads();

    if (warp == NW) {  // producer
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
        for (int it = 0; it < it_n; ++it) {
            const int s = it % NS;
            if (it >= NS) mbar_wait(empty + 8 * s, ((it / NS) - 1) & 1);
            const unsigned dst = sbase + s * G::SB;
            const int kt = kt_begin + it * C;
            if (lane == 0) mbar_expect(full + 8 * s, G::WB + m * C * 32);
            __syncwarp();
            if (lane < SP) {
                const __nv_bfloat16* src = w + ((size_t)(blockIdx.x * SP + lane) * kt16 + kt) * 256;
                tma1d(dst + lane * C * 512, src, C * 512, full + 8 * s, pol);
            }
            const int xc = (kt * 16) % kx;
            for (int r = lane; r < m; r += 32)
                tma1d_nohint(dst + G::WB + r * G::XLD * 2, x + (size_t)r * kx + xc, C * 32, full + 8 * s);
        }
    } else {  // consumers
        float acc[SP][MT][4];
#pragma unroll
        for (int s = 0; s < SP; ++s)
#pragma unroll
            for (int j = 0; j < MT; ++j) acc[s][j][0] = acc[s][j][1] = acc[s][j][2] = acc[s][j][3] = 0.f;
        for (int it = 0; it < it_n; ++it) {
            const int s = it % NS;
            mbar_wait(full + 8 * s, (it / NS) & 1);
            const unsigned wb = sbase + s * G::SB, xb = wb + G::WB;
#pragma unroll
            for (int t = 0; t < TW; t += 2) {
                const int tile = warp * TW + t;
                uint4 a0[SP], a1[SP];
#pragma unroll
                for (int p = 0; p < SP; ++p) {
                    a0[p] = lds128(wb + p * C * 512 + tile * 512 + lane * 16);
                    a1[p] = lds128(wb + p * C * 512 + (tile + 1) * 512 + lane * 16);
                }
#pragma unroll
                for (int j = 0; j < MT; ++j) {
                    unsigned b[4];
                    ldsm4(b, xb + ((j * 8 + (lane & 7)) * G::XLD + tile * 16 + (lane >> 3) * 8) * 2);
#pragma unroll
                    for (int p = 0; p < SP; ++p) {
                        mma(acc[p][j], a0[p], b[0], b[1]);
                        mma(acc[p][j], a1[p], b[2], b[3]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(empty + 8 * s);
        }
        // red[tok][row] (row within the group's SP*16 rows)
        const int g = lane >> 2, tq = lane & 3;
#pragma unroll
        for (int p = 0; p < SP; ++p)
#pragma unroll
            for (int j = 0; j < MT; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int row = p * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
                    atomicAdd(red + tok * SP * 16 + row, acc[p][j][r]);
                }
    }
    __syncthreads();
    constexpr int R = SP * 16;
    const int row0 = blockIdx.x * R;
    const bool split = gridDim.y > 1;
    if (!split) {
        for (int i = threadIdx.x; i < m * R; i += (NW + 1) * 32) {
            const int tok = i / R, row = row0 + i % R;
            if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = red[i];
            else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(red[i]);
        }
        return;
    }
    for (int i = threadIdx.x; i < m * R / 4; i += (NW + 1) * 32) {
        const int tok = (4 * i) / R, row = row0 + (4 * i) % R;
        const float4 v = *reinterpret_cast<const float4*>(red + 4 * i);
        float* p = ws + (size_t)tok * n + row;
        asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                     : "memory");
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = atomicAdd(cnt + blockIdx.x, 1);
        last = prev == (int)gridDim.y - 1;
        if (last) cnt[blockIdx.x] = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < m * R / 4; i += (NW + 1) * 32) {
        const int tok = (4 * i) / R, row = row0 + (4 * i) % R;
        float4* p = reinterpret_cast<float4*>(ws + (size_t)tok * n + row);
        const float4 v = __ldcg(p);
        __stcg(p, make_float4(0.f, 0.f, 0.f, 0.f));
        if (F32OUT) {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(out) + (size_t)tok * n + row) = v;
        } else {
            __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
            uint2 u = make_uint2(*reinterpret_cast<unsigned*>(&lo), *reinterpret_cast<unsigned*>(&hi));
            *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(out) + (size_t)tok * n + row) = u;
        }
    }
}

template <int MT, int SP, int C, int NW, int NS, bool F32OUT>
void launch(const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out, torch::Tensor& ws, torch::Tensor& cnt,
            int n, int k, int kx, int splits) {
    constexpr int SM = Cfg<MT, SP, C, NW, NS>::SMEM;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(gemv<MT, SP, C, NW, NS, F32OUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
        set = true;
    }
    const int m = (int)x.size(0);
    TORCH_CHECK(m <= MT * 8 && n % (SP * 16) == 0 && k % (splits * C * 16) == 0 && kx % (C * 16) == 0, "pj: bad tiling");
    dim3 grid(n / (SP * 16), splits);
    gemv<MT, SP, C, NW, NS, F32OUT><<<grid, (NW + 1) * 32, SM, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k, kx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace pj

#define PJ_CASE(MT, SP, C, NW, NS)                                                                \
    if (mt == MT && sp == SP && c == C && nw == NW && ns == NS) {                                 \
        if (f32) pj::launch<MT, SP, C, NW, NS, true>(w, x, out, ws, cnt, n, k, kx, splits);       \
        else pj::launch<MT, SP, C, NW, NS, false>(w, x, out, ws, cnt, n, k, kx, splits);          \
        return;                                                                                   \
    }

void pj_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n,
             int64_t k, int64_t kx, int64_t mt, int64_t sp, int64_t c, int64_t nw, int64_t ns, int64_t splits) {
    const bool f32 = out.scalar_type() == torch::kFloat32;
    PJ_CONFIGS
    TORCH_CHECK(false, "pj: no such config");
}
