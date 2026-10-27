// layer s86 OB8: the T=8 o GEMM (proj's gemv12 body, m = 8, one k-chunk) with the first boundary fused into its
// epilogue, so pr8 (its PDL secondary) starts right after it (H100 oracle proj s80: no boundary = o->pr8 end -4.2 us).
// Boundary (the Triton _boundary, RB off): b = bf16(o out); s' = alpha s + beta b rb, rb = rsqrt(sum b^2 / N + eps_post);
// h = s' rs gain, rs = rsqrt(sum s'^2 / N + eps). Each thread owns one (token, column) of its block's 64 columns.
// Per block and token, four FP32 partial sums over its columns (sum b^2, (alpha s)^2, (alpha s)(beta b), (beta b)^2;
// sum s'^2 = A + 2 rb C + rb^2 D) go to part[block][token][4]; a grid barrier (all blocks are resident: the PDL
// dependent launches only after every block triggered); then every block sums the partials in a fixed order
// (deterministic) and writes s', h (BF16) and h32 (FP32) for its own columns.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
namespace ob8 {
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
struct Bnd {
    const float* s; const float* alpha; const float* beta; const float* gain;
    float* s_out; __nv_bfloat16* h; float* h32; float* part; unsigned* bar; float eps_post, eps; int pdl;
};
constexpr int M = 8, WG = 2, NT = 8;
template <int PF>
__global__ void __launch_bounds__(512, 1) gemm(const uint4* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                               int n, int nrg_all, int k, const Bnd B) {
    if (!(B.pdl & 2)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // pdl & 2: after the W loop
    constexpr int KC = 1024, XS = KC * 2 + 64, NA = KC / 32 / WG;
    static_assert(NA % PF == 0, "PF must divide the atoms per warp");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* xs = smem;
    float4* accp = (float4*)(smem + M * XS);
    __shared__ float red[16][4][4];   // [warp][token in the warp][quantity]
    __shared__ float red2[16][32];
    __shared__ float fin[32];         // [token][quantity] grid totals
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5, g = lane >> 2, c = lane & 3;
    const int gi = wp / WG, wi = wp % WG, NG = blockDim.x / (32 * WG);
    const int rg0 = blockIdx.x * NG, rg = rg0 + gi;
    // this thread's boundary element (independent of the producer: before the wait)
    const int row8 = t & 7, tok = (t >> 3) % M, rgl = (t >> 3) / M, col = (rg0 + rgl) * 8 + row8;
    const long ei = (long)tok * n + col;
    const float sv = __ldg(B.s + ei), al = __ldg(B.alpha + col), be = __ldg(B.beta + col), gn = __ldg(B.gain + col);
    const uint4* wb = W + ((long)rg * (KC / 32) + wi) * 32 + lane;
    uint4 w[PF];
    if (B.pdl & 1) {
        const char* pb = (const char*)(W + (long)rg0 * KC);
        for (long o = (long)t * 128; o < (long)NG * KC * 16; o += (long)blockDim.x * 128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(pb + o));
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    for (int i = t; i < M * (KC / 8); i += blockDim.x) {
        const int r = i / (KC / 8), q = i % (KC / 8);
        const unsigned sa = (unsigned)__cvta_generic_to_shared(xs + r * XS + q * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"((const uint4*)(x + (long)r * k) + q));
    }
    asm volatile("cp.async.commit_group;");
#pragma unroll
    for (int p = 0; p < PF; ++p) w[p] = ldw(wb + p * WG * 32);
    asm volatile("cp.async.wait_all;");
    __syncthreads();
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int a = 0; a < NA; a += PF) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int kb = ((a + p) * WG + wi) * 32 + c * 8;
            uint4 xa = *(const uint4*)(xs + g * XS + kb * 2);
            mma(acc, xa.x, 0u, xa.y, 0u, w[p].x, w[p].y);
            mma(acc, xa.z, 0u, xa.w, 0u, w[p].z, w[p].w);
            if (a + p + PF < NA) w[p] = ldw(wb + (a + p + PF) * WG * 32);
        }
    }
    if (B.pdl & 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    accp[wp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    // gemv12's output i = (rgl * M + tok) * 8 + row8 = t: sum the WG warp slots of group rgl
    float bt;
    {
        const int l = tok * 4 + (row8 >> 1), e = row8 & 1;
        const float* base = (const float*)accp + (rgl * WG) * 128 + l * 4 + e;
        bt = 0.f;
#pragma unroll
        for (int q = 0; q < WG; ++q) bt += base[q * 128];
    }
    const float bf = __bfloat162float(__float2bfloat16(bt));  // the BF16 the GEMM returns
    const float xa = al * sv, yb = be * bf;
    float v[4] = {bf * bf, xa * xa, xa * yb, yb * yb};
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) v[q] += __shfl_xor_sync(0xffffffffu, v[q], o);
    if (row8 == 0)
#pragma unroll
        for (int q = 0; q < 4; ++q) red[wp][lane >> 3][q] = v[q];
    __syncthreads();
    // warp wp holds tokens (wp & 1) * 4 + 0..3 (row groups wp / 2): block total per (token, quantity), fixed order
    if (t < 32) {
        const int tk = t >> 2, q = t & 3;
        float s = 0.f;
        for (int w2 = tk >> 2; w2 < 16; w2 += 2) s += red[w2][tk & 3][q];
        __stcg(B.part + ((long)blockIdx.x * 32 + t), s);
    }
    __syncthreads();
    if (t == 0) {  // grid barrier (epoch counter: never reset; gridDim divides 2^32)
        __threadfence();
        const unsigned tk = atomicAdd(B.bar, 1u), target = (tk / gridDim.x + 1u) * gridDim.x;
        unsigned cur;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(B.bar) : "memory");
        } while ((int)(cur - target) < 0);
    }
    __syncthreads();
    {   // grid totals: 16 chunks of gridDim / 16 blocks per (token, quantity), then the 16 chunk sums in order
        const int p = t & 31, j = t >> 5, per = gridDim.x / 16;
        if (j < 16) {
            float s = 0.f;
            for (int b = j * per; b < (j + 1) * per; ++b) s += __ldcg(B.part + (long)b * 32 + p);
            red2[j][p] = s;
        }
    }
    __syncthreads();
    if (t < 32) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) s += red2[j][t];
        fin[t] = s;
    }
    __syncthreads();
    const float rb = rsqrtf(fin[tok * 4] / n + B.eps_post);
    const float s2 = fin[tok * 4 + 1] + 2.f * rb * fin[tok * 4 + 2] + rb * rb * fin[tok * 4 + 3];
    const float rs = rsqrtf(s2 / n + B.eps);
    const float sp = xa + be * (bf * rb);
    const float hv = sp * rs * gn;
    B.s_out[ei] = sp;
    B.h[ei] = __float2bfloat16(hv);
    if (B.h32) B.h32[ei] = hv;
}
}  // namespace ob8

// w: _pack12(o W, 1024) [1][n/8][32][32][8] bf16; x bf16 [8, 1024]; s fp32 [8, n] (the stream); alpha/beta/gain fp32 [n];
// s_out fp32 [8, n]; h bf16 [8, n]; h32 fp32 [8, n] or empty; part fp32 [blocks * 32]; bar int32 [1] (zeros once).
void ob8_gemm(torch::Tensor w, torch::Tensor x, torch::Tensor s, torch::Tensor alpha, torch::Tensor beta, torch::Tensor gain,
              torch::Tensor s_out, torch::Tensor h, torch::Tensor h32, torch::Tensor part, torch::Tensor bar,
              int64_t nrg_all, int64_t pf, int64_t bk, double eps_post, double eps, int64_t pdl) {
    const int m = x.size(0), k = x.size(1), n = h.size(1);
    constexpr int WG = ob8::WG;
    const int ng = n / 8 / bk, threads = ng * WG * 32;
    const int sm = m * (1024 * 2 + 64) + threads / 32 * 512;
    TORCH_CHECK(m == 8 && k == 1024 && threads == 512 && n / 8 % bk == 0 && bk % 16 == 0 && part.numel() >= bk * 32
                && s.size(1) == n, "ob8: shape");
    ob8::Bnd B{s.data_ptr<float>(), alpha.data_ptr<float>(), beta.data_ptr<float>(), gain.data_ptr<float>(),
               s_out.data_ptr<float>(), (__nv_bfloat16*)h.data_ptr(), h32.numel() ? h32.data_ptr<float>() : nullptr,
               part.data_ptr<float>(), (unsigned*)bar.data_ptr(), (float)eps_post, (float)eps, (int)pdl};
    cudaLaunchConfig_t lc = {}; lc.gridDim = bk; lc.blockDim = threads; lc.dynamicSmemBytes = sm;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = (pdl & 1) ? 1 : 0;
#define OB8_CASE(P_) \
    if (pf == P_) { cudaFuncSetAttribute(ob8::gemm<P_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ob8::gemm<P_>, (const uint4*)w.data_ptr(), (const __nv_bfloat16*)x.data_ptr(), \
                                          n, (int)nrg_all, k, B)); return; }
    OB8_CONFIGS
    TORCH_CHECK(false, "ob8: no config");
}
