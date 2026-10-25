// Split-K streaming GEMV/skinny GEMM for out[m, n] = x[m, k] @ W[n, k]^T, m <= 32.
// Swap-AB mma.sync m16n8k16 (A = 16 weight rows, B = up to 8 tokens). W is repacked in prepare() so each
// lane's A fragment of a 16x16 tile is 16 contiguous bytes: one coalesced 512-B load per warp per tile, no smem.
// Block (stripe group of SP x 16 rows, K chunk of NW*KT*16): every warp loads all its fragments up front,
// x[:, chunk] is staged in smem meanwhile, then mma, warp reduction in smem and (split K) FP32 atomics into a
// workspace; the last block of a stripe group writes the output and re-zeroes the workspace and its counter.
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
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ void ldsm4(unsigned* r, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

template <int MT, int SP, int KT, int NW, bool F32OUT>
__global__ void __launch_bounds__(NW * 32) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                void* __restrict__ out, float* __restrict__ ws, int* __restrict__ cnt,
                                                int m, int n, int k, int kx) {
    constexpr int KC = NW * KT * 16, LD = KC + 8, ACC = SP * MT * 4;
    extern __shared__ __align__(16) unsigned char smem[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    float* red = reinterpret_cast<float*>(smem);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kt16 = k >> 4;
    // 1. all of this warp's weight fragments in flight
    uint4 a[SP][KT];
    {
        const int kt0 = blockIdx.y * (NW * KT) + warp * KT;
#pragma unroll
        for (int s = 0; s < SP; ++s) {
            const uint4* p = w + ((size_t)(blockIdx.x * SP + s) * kt16 + kt0) * 32 + lane;
#pragma unroll
            for (int t = 0; t < KT; ++t) a[s][t] = ldw(p + t * 32);
        }
    }
    // 2. stage x[:, chunk] (rows >= m zero)
    {
        const int c0 = (blockIdx.y * KC) % kx;
        constexpr int V = MT * 8 * KC / 8;
        for (int i = threadIdx.x; i < V; i += NW * 32) {
            const int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < m) v = *reinterpret_cast<const uint4*>(x + (size_t)r * kx + c0 + c);
            *reinterpret_cast<uint4*>(xs + r * LD + c) = v;
        }
    }
    __syncthreads();
    // 3. mma
    float acc[SP][MT][4];
#pragma unroll
    for (int s = 0; s < SP; ++s)
#pragma unroll
        for (int j = 0; j < MT; ++j) acc[s][j][0] = acc[s][j][1] = acc[s][j][2] = acc[s][j][3] = 0.f;
    const unsigned xbase = (unsigned)__cvta_generic_to_shared(xs);
#pragma unroll
    for (int t = 0; t < KT; t += 2) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            unsigned b[4];
            const int row = j * 8 + (lane & 7), col = (warp * KT + t) * 16 + (lane >> 3) * 8;
            ldsm4(b, xbase + (row * LD + col) * 2);
#pragma unroll
            for (int s = 0; s < SP; ++s) {
                mma(acc[s][j], a[s][t], b[0], b[1]);
                if (t + 1 < KT) mma(acc[s][j], a[s][t + 1], b[2], b[3]);
            }
        }
    }
    // 4. reduce over warps
    if (NW > 1) {
        __syncthreads();
#pragma unroll
        for (int s = 0; s < SP; ++s)
#pragma unroll
            for (int j = 0; j < MT; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r) red[((warp * ACC) + (s * MT + j) * 4 + r) * 32 + lane] = acc[s][j][r];
        __syncthreads();
    }
    const bool split = gridDim.y > 1;
    for (int e = threadIdx.x; e < ACC * 32; e += NW * 32) {
        float v;
        if (NW > 1) {
            v = 0.f;
#pragma unroll
            for (int q = 0; q < NW; ++q) v += red[q * ACC * 32 + e];
        } else {
            v = 0.f;  // NW == 1: e == lane + 32 * idx, fetch from registers
            const int idx = e >> 5;
#pragma unroll
            for (int s = 0; s < SP; ++s)
#pragma unroll
                for (int j = 0; j < MT; ++j)
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        if ((s * MT + j) * 4 + r == idx) v = acc[s][j][r];
        }
        const int l = e & 31, idx = e >> 5, r = idx & 3, j = (idx >> 2) % MT, s = idx / (4 * MT);
        const int g = l >> 2, tq = l & 3;
        const int row = (blockIdx.x * SP + s) * 16 + g + ((r & 2) ? 8 : 0);
        const int tok = j * 8 + 2 * tq + (r & 1);
        if (tok < m) {
            if (!split) {
                if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = v;
                else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(v);
            } else {
                atomicAdd(ws + (size_t)tok * n + row, v);
            }
        }
    }
    if (!split) return;
    // 5. last block of this stripe group finalizes
    __shared__ int last;
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
    constexpr int R = SP * 16;
    for (int i = threadIdx.x; i < m * R; i += NW * 32) {
        const int tok = i / R, row = blockIdx.x * R + i % R;
        float* p = ws + (size_t)tok * n + row;
        const float v = __ldcg(p);
        __stcg(p, 0.f);
        if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = v;
        else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(v);
    }
}

template <int MT, int SP, int KT, int NW, bool F32OUT>
void launch(const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out, torch::Tensor& ws, torch::Tensor& cnt,
            int n, int k, int kx) {
    constexpr int KC = NW * KT * 16;
    const int xs = MT * 8 * (KC + 8) * 2, rs = NW * SP * MT * 4 * 32 * 4;
    const int sm = xs > rs ? xs : rs;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(gemv<MT, SP, KT, NW, F32OUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        set = true;
    }
    TORCH_CHECK(n % (SP * 16) == 0 && k % KC == 0 && kx % KC == 0, "pj: bad tiling");
    dim3 grid(n / (SP * 16), k / KC);
    gemv<MT, SP, KT, NW, F32OUT><<<grid, NW * 32, sm, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), (int)x.size(0), n, k, kx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace pj

#define PJ_CASE(MT, SP, KT, NW)                                                          \
    if (mt == MT && sp == SP && kt == KT && nw == NW) {                                  \
        if (f32) pj::launch<MT, SP, KT, NW, true>(w, x, out, ws, cnt, n, k, kx);         \
        else pj::launch<MT, SP, KT, NW, false>(w, x, out, ws, cnt, n, k, kx);            \
        return;                                                                          \
    }

void pj_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n,
             int64_t k, int64_t kx, int64_t mt, int64_t sp, int64_t kt, int64_t nw) {
    const bool f32 = out.scalar_type() == torch::kFloat32;
    PJ_CONFIGS
    TORCH_CHECK(false, "pj: no such config");
}
