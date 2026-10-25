// gemv11: tensor-core streaming GEMM for m = 8..32 (out[m, n] = x[m, k] @ W[n, k]^T, BF16 in, FP32 accumulate).
// mma.sync m16n8k16 with x as A (tokens) and W as B (8 rows of W = the 8 n columns). W is repacked in prepare()
// into 512-B "atoms" = 8 rows x 32 k, lane l holding row l/4, k (l%4)*8..+8 (one uint4 = the B fragments of two
// k16 mmas; the A fragments of x use the same k permutation). Layout [ks][rg][KC/32 atoms][lane], ks = k-chunk of
// KC = 1024*U, rg = 8-row group. Block b (1024 threads) owns k-chunk ks = b / BK and a contiguous run of rgs; it
// stages x[:, ks*KC..+KC] in smem once, then each iteration reads one unit (rg, ks) = 32 warps x U atoms, 16 KB
// contiguous per u-step (the grid reads contiguous windows), PF units in flight in registers. Warps' FP32 C
// fragments are summed in smem; with one k-chunk the result is written as BF16, else it is added to an FP32
// workspace (red.global.add.v4.f32) and the block completing a rg's S chunks (counter) writes BF16 and re-zeroes.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace g11 {
__device__ __forceinline__ uint4 ldw(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// M tokens (8, 16 or 32); KC = 1024*KCU; the 32 warps form NG groups of WG = 32/NG warps; group gi takes the block's
// rgs rg0+gi, rg0+gi+NG, ...; within a unit (rg, ks) warp wi of the group takes atoms u*WG + wi (u < U = KC/32/WG),
// so each u-step a group reads WG*512 contiguous bytes. PF units in flight. Each warp keeps its FP32 C fragments
// of a unit in registers and stores them to a private smem slot (no atomics); the block sums the slots at the end.
template <int M, int KCU, int NG, int PF>
__global__ void __launch_bounds__(1024, 1) gemm(const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out, float* __restrict__ ws,
                                                int* __restrict__ cnt, int n, int nrg_all, int k, int bk, int maxj) {
    constexpr int KC = 1024 * KCU, TT = (M + 15) / 16, XS = KC * 2 + 64;  // x row stride in smem (bytes), padded
    constexpr int NT = M < 16 ? M : 16, WG = 32 / NG, U = KC / 32 / WG;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;                        // x[M][KC] (row stride XS)
    float4* accp = (float4*)(smem + M * XS);         // [32 warps][maxj][TT][32 lanes] C fragments
    __shared__ int flag[128];
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG;
    const int S = k / KC, ks = blockIdx.x / bk, bi = blockIdx.x % bk, nrg = n / 8;
    const int rg0 = (int)((long)bi * nrg / bk), rg1 = (int)((long)(bi + 1) * nrg / bk);
    auto wptr = [&](int rg, int u) { return W + (((long)ks * nrg_all + rg) * (KC / 32) + u * WG + wi) * 32 + lane; };
    // x chunk first (cp.async, small), then the W prologue: x must not queue behind the W stream
    for (int i = t; i < M * (KC / 8); i += 1024) {
        const int r = i / (KC / 8), q = i % (KC / 8);
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + q * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"((const uint4*)(x + (long)r * k + (long)ks * KC) + q));
    }
    asm volatile("cp.async.commit_group;");
    uint4 w[PF][U];
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        const int r = rg0 + gi + p * NG;
        if (r < rg1) {
#pragma unroll
            for (int u = 0; u < U; ++u) w[p][u] = ldw(wptr(r, u));
        }
    }
    asm volatile("cp.async.wait_all;");
    __syncthreads();
    int j = 0;
    for (int rg = rg0 + gi; rg < rg1; rg += PF * NG) {
#pragma unroll
        for (int p = 0; p < PF; ++p, ++j) {
            const int r = rg + p * NG;
            if (r < rg1) {
                float acc[TT][4];
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) acc[tt][0] = acc[tt][1] = acc[tt][2] = acc[tt][3] = 0.f;
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int kb = (u * WG + wi) * 32 + c * 8;
#pragma unroll
                    for (int tt = 0; tt < TT; ++tt) {
                        uint4 xa = *(const uint4*)(xs + (tt * 16 + g) * XS + kb * 2);
                        uint4 xb = make_uint4(0, 0, 0, 0);
                        if (NT > 8) xb = *(const uint4*)(xs + (tt * 16 + g + 8) * XS + kb * 2);
                        mma(acc[tt], xa.x, xb.x, xa.y, xb.y, w[p][u].x, w[p][u].y);
                        mma(acc[tt], xa.z, xb.z, xa.w, xb.w, w[p][u].z, w[p][u].w);
                    }
                }
                if (r + PF * NG < rg1) {
#pragma unroll
                    for (int u = 0; u < U; ++u) w[p][u] = ldw(wptr(r + PF * NG, u));
                }
#pragma unroll
                for (int tt = 0; tt < TT; ++tt)
                    accp[((wp * maxj + j) * TT + tt) * 32 + lane] = make_float4(acc[tt][0], acc[tt][1], acc[tt][2], acc[tt][3]);
            }
        }
    }
    __syncthreads();
    // outputs i = ((rgl * M + tok) * 8 + row8): sum the group's WG warp slots
    const int ne = (rg1 - rg0) * M * 8;
    auto total = [&](int i) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        const int gg = rgl % NG, jj = rgl / NG, tt = tok >> 4, h = (tok >> 3) & 1;
        const int l = (tok & 7) * 4 + (row8 >> 1), e = h * 2 + (row8 & 1);
        const float* base = (const float*)accp + (((long)(gg * WG) * maxj + jj) * TT + tt) * 128 + l * 4 + e;
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) s += base[(long)q * maxj * TT * 128];
        return s;
    };
    if (S == 1) {
        for (int i = t; i < ne; i += 1024) {
            const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(total(i));
        }
        return;
    }
    for (int i = t; i < ne; i += 1024) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        __stcg(ws + ((long)ks * M + tok) * n + (rg0 + rgl) * 8 + row8, total(i));
    }
    __threadfence();
    __syncthreads();
    if (t < rg1 - rg0) {
        const int last = atomicAdd(cnt + rg0 + t, 1) == S - 1;
        flag[t] = last;
        if (last) cnt[rg0 + t] = 0;
    }
    __syncthreads();
    __threadfence();
    for (int i = t; i < ne; i += 1024) {
        const int row8 = i & 7, tok = (i >> 3) % M, rgl = (i >> 3) / M;
        if (flag[rgl]) {
            float s = 0.f;
            for (int q = 0; q < S; ++q) s += __ldcg(ws + ((long)q * M + tok) * n + (rg0 + rgl) * 8 + row8);
            out[(long)tok * n + (rg0 + rgl) * 8 + row8] = __float2bfloat16(s);
        }
    }
}
}  // namespace g11

static int g11_smem(int m, int kcu, int maxj) { const int tt = (m + 15) / 16; return m * (1024 * kcu * 2 + 64) + 32 * maxj * tt * 32 * 16; }
void g11_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t nrg_all,
              int64_t kcu, int64_t ng, int64_t pf, int64_t bk) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    auto s = c10::cuda::getCurrentCUDAStream();
    const int maxrg = (n / 8 + bk - 1) / bk, maxj = (maxrg + ng * pf - 1) / (ng * pf) * pf;
    const int sm = g11_smem(m, kcu, maxj), blocks = (k / (1024 * kcu)) * bk;
    TORCH_CHECK(maxrg <= 128 && sm <= 227 * 1024, "g11: smem");
#define G11_CASE(M_, U_, G_, P_) \
    if (m == M_ && kcu == U_ && ng == G_ && pf == P_) { \
        cudaFuncSetAttribute(g11::gemm<M_, U_, G_, P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        g11::gemm<M_, U_, G_, P_><<<blocks, 1024, sm, s>>>((const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
        (__nv_bfloat16*)out.data_ptr(), ws.data_ptr<float>(), cnt.data_ptr<int>(), n, nrg_all, k, bk, maxj); return; }
    G11_CONFIGS
    TORCH_CHECK(false, "g11: no config");
}
