// Router at m >= 9 as SGLang's BF16 hi + lo split with FP32 accumulation, in ONE GEMM kernel: split-K tensor-core
// GEMM (mma.sync m16n8k16, x = A, W rows = B), hi and lo MMAs into the same FP32 accumulators (sharing x fragments).
// Block = MT tokens x NT experts x KC of k; cp.async ring of 64-wide k stages (swizzled 128-B rows, ldmatrix).
// Split-K partials: staged in smem, then float4 FP32 atomics (red at L2) straight into out, which a tiny kernel
// zeroes first; the GEMM is its programmatic dependent (PDL): it loads and multiplies while the zeroing runs and
// waits (griddepcontrol.wait) only before its atomics. No workspace, no counters.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace pr3 {
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void ldm4(unsigned addr, unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// byte offset of 16-B chunk ch (0..7) of a 128-B row
__device__ __forceinline__ unsigned swz(int row, int ch) { return row * 128 + ((ch ^ (row & 7)) << 4); }

// MT tokens, NT experts per block, WM x WN warps, KC = k per block, NS stages of 64 k.
__device__ unsigned pr3_cnt[4096];
__device__ unsigned pr3_flag[4096];
template <int MT, int NT, int WM, int WN, int KC, int NS, int SLOT>
__global__ void __launch_bounds__(WM * WN * 32) router(const __nv_bfloat16* __restrict__ w,  // [2][256][K] hi, lo
                                                       const __nv_bfloat16* __restrict__ x,  // [m][K]
                                                       float* __restrict__ out, int m, int K) {
    constexpr int NTH = WM * WN * 32, TM = MT / WM, TN = NT / WN, FM = TM / 16, FN = TN / 8;
    constexpr int XB = MT * 128, WB = NT * 128, SB = XB + 2 * WB, NK = KC / 64;
    static_assert(TM % 16 == 0 && TN % 16 == 0, "warp tile");
    extern __shared__ __align__(128) unsigned char sm[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, wm = warp / WN, wn = warp % WN;
    const int s = blockIdx.x, n0 = blockIdx.y * NT, m0 = blockIdx.z * MT, k0 = s * KC;
    const unsigned sb = smem_u32(sm);

    auto load = [&](int it) {
        if (it < NK) {
            const unsigned st = sb + (it % NS) * SB;
            const int kk = k0 + it * 64;
            for (int i = t; i < MT * 8; i += NTH) {
                const int r = i >> 3, ch = i & 7, row = m0 + r;
                const bool ok = row < m;
                cp16(st + swz(r, ch), x + (size_t)(ok ? row : 0) * K + kk + ch * 8, ok ? 16 : 0);
            }
            for (int i = t; i < 2 * NT * 8; i += NTH) {
                const int h = i / (NT * 8), r = (i >> 3) % NT, ch = i & 7;
                cp16(st + XB + h * WB + swz(r, ch), w + ((size_t)h * 256 + n0 + r) * K + kk + ch * 8, 16);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };

    float acc[FM][FN][4];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
    for (int i = 0; i < NS - 1; ++i) load(i);
    for (int it = 0; it < NK; ++it) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NS - 2));
        __syncthreads();
        load(it + NS - 1);
        const unsigned st = sb + (it % NS) * SB;
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {  // k16 steps: chunks 2kq, 2kq+1
            unsigned a[FM][4];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                const int r = wm * TM + i * 16 + (lane & 15);
                ldm4(st + swz(r, 2 * kq + (lane >> 4)), a[i][0], a[i][1], a[i][2], a[i][3]);
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < FN; j += 2) {
                    const int r = wn * TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                    unsigned b0, b1, b2, b3;
                    ldm4(st + XB + h * WB + swz(r, 2 * kq + ((lane >> 3) & 1)), b0, b1, b2, b3);
#pragma unroll
                    for (int i = 0; i < FM; ++i) {
                        mma(acc[i][j], a[i], b0, b1);
                        mma(acc[i][j + 1], a[i], b2, b3);
                    }
                }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    // partial tile -> smem [MT][NT + 8] (conflict-free float2 stores), then float4 atomics into out
    constexpr int RS = NT + 8;
    static_assert(MT * RS * 4 <= NS * SB, "reduction tile");
    float* red = reinterpret_cast<float*>(sm);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            const int row = wm * TM + i * 16 + (lane >> 2), col = wn * TN + j * 8 + 2 * (lane & 3);
            *reinterpret_cast<float2*>(red + row * RS + col) = make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2*>(red + (row + 8) * RS + col) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
#ifdef R3_CL
    // cluster = all S K-splits of this tile: partials stay in smem, rank q sums float4 slice q over DSMEM, plain stores
    {
        namespace cg = cooperative_groups;
        cg::cluster_group cl = cg::this_cluster();
        __syncthreads();
        cl.sync();
        const int S = gridDim.x, rank = (int)cl.block_rank();
        constexpr int TOT1 = MT * (NT / 4);
        for (int i = rank * NTH + t; i < TOT1; i += S * NTH) {
            const int r = i / (NT / 4), c = (i % (NT / 4)) * 4;
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int q = 0; q < S; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(cl.map_shared_rank(red, q) + r * RS + c);
                a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
            }
            if (m0 + r < m) *reinterpret_cast<float4*>(out + (size_t)(m0 + r) * 256 + n0 + c) = a;
        }
        cl.sync();
        return;
    }
#endif
#ifdef R3_FA
    // first arriver per tile stores its partial; later arrivers wait for its flag, then add (no zero kernel)
    __shared__ unsigned fa_t;
    const unsigned S = gridDim.x, tile = SLOT + blockIdx.y + blockIdx.z * gridDim.y;
    __syncthreads();
    if (t == 0) fa_t = atomicAdd(&pr3_cnt[tile], 1u);
    __syncthreads();
    const unsigned tk = fa_t, call = tk / S;
    if (tk % S == 0) {
        constexpr int TOT0 = MT * (NT / 4);
        for (int i = t; i < TOT0; i += NTH) {
            const int r = i / (NT / 4), c = (i % (NT / 4)) * 4;
            if (m0 + r < m)
                *reinterpret_cast<float4*>(out + (size_t)(m0 + r) * 256 + n0 + c) = *reinterpret_cast<const float4*>(red + r * RS + c);
        }
        __threadfence();
        __syncthreads();
        if (t == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(&pr3_flag[tile]), "r"(call + 1) : "memory");
        return;
    }
    if (t == 0) {
        unsigned v;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(&pr3_flag[tile]) : "memory"); } while ((int)(v - (call + 1)) < 0);
    }
    __syncthreads();
#else
    asm volatile("griddepcontrol.wait;\n" ::: "memory");  // out zeroed
    __syncthreads();
#endif
    constexpr int TOT = MT * (NT / 4);
    const int rot = (int)(((long)s * TOT) / gridDim.x);  // splits of a tile start at different rows
    for (int i0 = t; i0 < TOT; i0 += NTH) {
        const int i = (i0 + rot) % TOT;
        const int r = i / (NT / 4), c = (i % (NT / 4)) * 4;
#ifndef R3_NORED
        if (m0 + r < m)
#else
        if (m0 + r < 0)
#endif
            atomicAdd(reinterpret_cast<float4*>(out + (size_t)(m0 + r) * 256 + n0 + c),
                      *reinterpret_cast<const float4*>(red + r * RS + c));
    }
}

__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;\n");
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace pr3

#if defined(R3_NOZERO) || defined(R3_FA) || defined(R3_CL)
#define NOZERO 1
#else
#define NOZERO 0
#endif
#ifdef R3_CL
#define CLX 1
#else
#define CLX 0
#endif
#define PR3_CASE(MT, NT, WM, WN, KC, NS)                                                                           \
    if (mt == MT && nt == NT && wmn == WM && wnn == WN && kc == KC && ns == NS) {                                  \
        auto f = pr3::router<MT, NT, WM, WN, KC, NS, ((MT * 7 + NT * 3 + KC / 64 * 11 + WM * 13 + WN * 17 + NS) % 31) * 128>;                                                              \
        constexpr int SMEM = NS * (MT * 128 + 2 * NT * 128);                                                       \
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);                                \
        if (!NOZERO) pr3::zero<<<(m * 64 + 255) / 256, 256, 0, stream>>>(reinterpret_cast<float4*>(out.data_ptr<float>()),      \
                                                            m * 64);                                               \
        cudaLaunchConfig_t lc = {};                                                                                \
        lc.gridDim = dim3(K / KC, 256 / NT, (m + MT - 1) / MT);                                                    \
        lc.blockDim = dim3(WM * WN * 32);                                                                          \
        lc.dynamicSmemBytes = SMEM;                                                                                \
        lc.stream = stream;                                                                                        \
        cudaLaunchAttribute at[2];                                                                                 \
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                             \
        at[0].val.programmaticStreamSerializationAllowed = !NOZERO;                                                      \
        at[1].id = cudaLaunchAttributeClusterDimension;                                                            \
        at[1].val.clusterDim.x = CLX ? K / KC : 1; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;         \
        if (CLX) cudaFuncSetAttribute(f, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);                       \
        lc.attrs = at; lc.numAttrs = 2;                                                                            \
        cudaLaunchKernelEx(&lc, f, reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),                           \
                           reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), m, K);     \
        return;                                                                                                    \
    }

// w bf16 [2, 256, K] (hi, lo), x bf16 [m, K] contiguous, out fp32 [m, 256] contiguous (overwritten)
void pj_router3(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t mt, int64_t nt, int64_t wmn, int64_t wnn,
                int64_t kc, int64_t ns) {
    const int m = x.size(0), K = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    PR3_CONFIGS
    TORCH_CHECK(false, "pj_router3: no config");
}
