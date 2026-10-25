// Split-K streaming skinny GEMM out[m, n] = x[m, k] @ W[n, k]^T, m <= 32, BF16 in, FP32 accumulate.
// Swap-AB mma.sync m16n8k16: A = a 16-row weight stripe, B = 8 tokens. W is repacked in prepare() into
// [stripe][k-tile][lane][8] (a 16x16 tile's A fragments = 512 contiguous bytes, 16 B per lane).
// Warp-level stream-K: the (stripe, k-tile) tiles are split evenly over all warps of the grid; each warp walks
// its contiguous run with U tiles in flight in registers (next batch loads while this one multiplies), x
// fragments straight from global (L1/L2). A stripe a warp covers whole is written directly; partial stripes are
// added into an FP32 workspace (red.global) and counted per stripe; the warp completing a stripe writes it and
// re-zeroes the workspace and counter, so the call is graph-replayable with no memset.
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
__device__ __forceinline__ unsigned ldx(const __nv_bfloat16* p) {
    return __ldg(reinterpret_cast<const unsigned*>(p));
}

template <int MT, int U, bool F32OUT>
__device__ __forceinline__ void flush(float (&acc)[MT][4], int stripe, bool whole, int ntiles, int kt16, int lane,
                                      int m, int n, void* out, float* ws, int* cnt) {
    const int g = lane >> 2, tq = lane & 3;
    if (whole) {
#pragma unroll
        for (int j = 0; j < MT; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = stripe * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
                if (tok < m) {
                    if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = acc[j][r];
                    else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(acc[j][r]);
                }
            }
        return;
    }
#ifdef PJ_NOFLUSH
    if (acc[0][0] != 12345.f) return;
#endif
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = stripe * 16 + g + ((r & 2) ? 8 : 0), tok = j * 8 + 2 * tq + (r & 1);
            if (tok < m) atomicAdd(ws + (size_t)tok * n + row, acc[j][r]);
        }
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) {
        const int prev = atomicAdd(cnt + stripe, ntiles);
        last = prev + ntiles == kt16;
        if (last) cnt[stripe] = 0;
    }
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) return;
    __threadfence();
    for (int i = lane; i < m * 16; i += 32) {
        const int tok = i >> 4, row = stripe * 16 + (i & 15);
        float* p = ws + (size_t)tok * n + row;
        const float v = __ldcg(p);
        __stcg(p, 0.f);
        if (F32OUT) reinterpret_cast<float*>(out)[(size_t)tok * n + row] = v;
        else reinterpret_cast<__nv_bfloat16*>(out)[(size_t)tok * n + row] = __float2bfloat16(v);
    }
}

template <int MT, int U, int WPB, bool F32OUT>
__global__ void __launch_bounds__(WPB * 32) gemv(const uint4* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                 void* __restrict__ out, float* __restrict__ ws,
                                                 int* __restrict__ cnt, int m, int n, int k, int kx) {
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * WPB + (threadIdx.x >> 5), nwarps = gridDim.x * WPB;
    const int kt16 = k >> 4;
    const int T = (n >> 4) * kt16;
    const int t0 = (int)((long long)gw * T / nwarps), t1 = (int)((long long)(gw + 1) * T / nwarps);
    if (t0 >= t1) return;
    const int g = lane >> 2, tq = lane & 3;
    // x fragment base pointers per n8 tile (rows >= m read row 0 and are zeroed)
    const __nv_bfloat16* xr[MT];
    bool xv[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) {
        xv[j] = j * 8 + g < m;
        xr[j] = x + (size_t)(xv[j] ? j * 8 + g : 0) * kx + 2 * tq;
    }
    uint4 a[U], na[U];
    unsigned b[U][MT][2], nb[U][MT][2];
    auto load = [&](uint4 (&aa)[U], unsigned (&bb)[U][MT][2], int tb) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = min(tb + u, t1 - 1);
            aa[u] = ldw(w + (size_t)t * 32 + lane);
            const int kc = ((t % kt16) * 16) % kx;
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                bb[u][j][0] = xv[j] ? ldx(xr[j] + kc) : 0u;
                bb[u][j][1] = xv[j] ? ldx(xr[j] + kc + 8) : 0u;
            }
        }
    };
    float acc[MT][4];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    int stripe = t0 / kt16, sbeg = t0;
    load(a, b, t0);
    for (int tb = t0; tb < t1; tb += U) {
        if (tb + U < t1) load(na, nb, tb + U);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = tb + u;
            if (t < t1) {
                if (t - stripe * kt16 >= kt16) {  // crossed into the next stripe
                    flush<MT, U, F32OUT>(acc, stripe, sbeg == stripe * kt16, t - sbeg, kt16, lane, m, n, out, ws, cnt);
#pragma unroll
                    for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
                    ++stripe;
                    sbeg = t;
                }
#pragma unroll
                for (int j = 0; j < MT; ++j) mma(acc[j], a[u], b[u][j][0], b[u][j][1]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            a[u] = na[u];
#pragma unroll
            for (int j = 0; j < MT; ++j) { b[u][j][0] = nb[u][j][0]; b[u][j][1] = nb[u][j][1]; }
        }
    }
    flush<MT, U, F32OUT>(acc, stripe, sbeg == stripe * kt16 && t1 == (stripe + 1) * kt16, t1 - sbeg, kt16, lane, m, n,
                         out, ws, cnt);
}

template <int MT, int U, int WPB, bool F32OUT>
void launch(const torch::Tensor& w, const torch::Tensor& x, torch::Tensor& out, torch::Tensor& ws, torch::Tensor& cnt,
            int n, int k, int kx, int blocks) {
    const int m = (int)x.size(0);
    TORCH_CHECK(m <= MT * 8 && n % 16 == 0 && k % 16 == 0 && kx % 16 == 0, "pj: bad shape");
    gemv<MT, U, WPB, F32OUT><<<blocks, WPB * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), m, n, k, kx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace pj

#define PJ_CASE(MT, U, WPB)                                                                  \
    if (mt == MT && u == U && wpb == WPB) {                                                  \
        if (f32) pj::launch<MT, U, WPB, true>(w, x, out, ws, cnt, n, k, kx, blocks);         \
        else pj::launch<MT, U, WPB, false>(w, x, out, ws, cnt, n, k, kx, blocks);            \
        return;                                                                              \
    }

void pj_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t n,
             int64_t k, int64_t kx, int64_t mt, int64_t u, int64_t wpb, int64_t blocks) {
    const bool f32 = out.scalar_type() == torch::kFloat32;
    PJ_CONFIGS
    TORCH_CHECK(false, "pj: no such config");
}
