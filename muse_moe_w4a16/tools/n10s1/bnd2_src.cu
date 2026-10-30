// bnd2 (moe n10s1): CUDA second boundary for T <= 32 (one row x SPL programs, 512 threads, thread t owns the 16
// columns h * 4096 + t * 8 + k). Same math as layer _boundary (BGAIN, H32 off): b = branch (MODE 0: BF16 [T, NB];
// 1: FP32 [T, NB] rounded to BF16; 2: SC: sum over the row's 8 local slots j of tw_j rsqrt(sc_j / 4096 + sc_eps) br[j]
// (FP32 [T * 8, NB]) rounded to BF16), b *= bgain, rb = rsqrt(sum b^2 / N + eps_post), s' = alpha s + beta b rb,
// h = bf16(s' rsqrt(sum s'^2 / N + eps) gain). Program p of a row stores columns [p N / SPL, (p + 1) N / SPL).
namespace bnd2 {
constexpr int N = 8192, NT = 512, NW = 16;
__device__ __forceinline__ float bsum(float x, float* red, int slot) {
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[slot * NW + warp] = x;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int q = 0; q < NW; ++q) t += red[slot * NW + q];
    return t;
}
// asm volatile loads: they stay above the asm volatile griddepcontrol.wait (ptxas sinks plain loads below it)
__device__ __forceinline__ void ld8(const float* p, float* v) {
    asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
    asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p + 4));
}
__device__ __forceinline__ int ldi(const int* p) { int v; asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ float ldf(const float* p) { float v; asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ void ld8cg(const float* p, float* v) {
    const float4 a = __ldcg(reinterpret_cast<const float4*>(p)), b = __ldcg(reinterpret_cast<const float4*>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
template <int NB, int MODE, int SPL, bool TRIG = false>
__global__ void __launch_bounds__(NT) bnd2(const void* __restrict__ br, const float* __restrict__ s1,
                                           const float* __restrict__ alpha, const float* __restrict__ beta,
                                           const float* __restrict__ bgain, const float* __restrict__ gain,
                                           float* __restrict__ s2, __nv_bfloat16* __restrict__ hid,
                                           const float* __restrict__ sc, const int* __restrict__ ids,
                                           const float* __restrict__ tw, int sc_local, float sc_eps, float eps_post, float eps) {
    __shared__ float red[2 * NW];
    if (TRIG) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int row = blockIdx.x / SPL, p = blockIdx.x % SPL, t = threadIdx.x;
    float s[16], a[16], be[16], bg[16], g[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) bg[i] = g[i] = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int c0 = h * 4096 + t * 8;
        ld8(s1 + (size_t)row * N + c0, s + h * 8);
        ld8(alpha + c0, a + h * 8);
        ld8(beta + c0, be + h * 8);
        if (c0 < NB) ld8(bgain + c0, bg + h * 8);
        if (c0 / (N / SPL) == p) ld8(gain + c0, g + h * 8);
    }
    float tws[8];
    bool loc[8];
    if (MODE == 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int e = ldi(ids + row * 8 + j);
            loc[j] = e >= 0 && e < sc_local;
            tws[j] = ldf(tw + row * 8 + j);
        }
    }
    {  // pin the loads above the wait (ptxas hoists ACQBULK over plain loads): a branch on all loaded words
        unsigned u = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) u ^= __float_as_uint(s[i]) ^ __float_as_uint(a[i]) ^ __float_as_uint(be[i]) ^ __float_as_uint(bg[i]) ^ __float_as_uint(g[i]);
        if (MODE == 2) {
#pragma unroll
            for (int j = 0; j < 8; ++j) u ^= __float_as_uint(tws[j]) ^ (unsigned)loc[j];
        }
        __shared__ unsigned pin;
        if (u == 0x7fbfffffu) *(volatile unsigned*)&pin = u;  // never read: pins the loads above the wait
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float b[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) b[i] = 0.f;
    if (MODE == 2) {
        if (t * 8 < NB) {
            float sf[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) sf[j] = loc[j] ? tws[j] * rsqrtf(__ldcg(sc + row * 8 + j) / 4096.f + sc_eps) : 0.f;
            const float* B = reinterpret_cast<const float*>(br);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (!loc[j]) continue;
                float v[8];
                ld8cg(B + (size_t)(row * 8 + j) * NB + t * 8, v);
#pragma unroll
                for (int k = 0; k < 8; ++k) b[k] += v[k] * sf[j];
            }
        }
    } else {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c0 = h * 4096 + t * 8;
            if (c0 >= NB) continue;
            if (MODE == 1) {
                ld8cg(reinterpret_cast<const float*>(br) + (size_t)row * NB + c0, b + h * 8);
            } else {
                const uint4 q = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(br) + (size_t)row * NB + c0));
                const __nv_bfloat162* qq = reinterpret_cast<const __nv_bfloat162*>(&q);
#pragma unroll
                for (int k = 0; k < 4; ++k) { b[h * 8 + 2 * k] = __low2float(qq[k]); b[h * 8 + 2 * k + 1] = __high2float(qq[k]); }
            }
        }
    }
    float q1 = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        if (h * 4096 + t * 8 >= NB) continue;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = h * 8 + k;
            b[i] = (MODE == 0 ? b[i] : rbf(b[i])) * bg[i];
            q1 += b[i] * b[i];
        }
    }
    const float rb = rsqrtf(bsum(q1, red, 0) / N + eps_post);
    float q2 = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        s[i] = a[i] * s[i] + be[i] * (b[i] * rb);
        q2 += s[i] * s[i];
    }
    const float rs = rsqrtf(bsum(q2, red, 1) / N + eps);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int c0 = h * 4096 + t * 8;
        if (c0 / (N / SPL) != p) continue;
        float* o = s2 + (size_t)row * N + c0;
        *reinterpret_cast<float4*>(o) = make_float4(s[h * 8], s[h * 8 + 1], s[h * 8 + 2], s[h * 8 + 3]);
        *reinterpret_cast<float4*>(o + 4) = make_float4(s[h * 8 + 4], s[h * 8 + 5], s[h * 8 + 6], s[h * 8 + 7]);
        uint4 o4;
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            op[k] = __floats2bfloat162_rn(s[h * 8 + 2 * k] * rs * g[h * 8 + 2 * k], s[h * 8 + 2 * k + 1] * rs * g[h * 8 + 2 * k + 1]);
        *reinterpret_cast<uint4*>(hid + (size_t)row * N + c0) = o4;
    }
}
}  // namespace bnd2
// cfg = NB_code * 100 + MODE * 10 + SPL; NB_code 1 = 1024, 8 = 8192
void bnd2_launch(torch::Tensor br, torch::Tensor s1, torch::Tensor alpha, torch::Tensor beta, torch::Tensor bgain,
                 torch::Tensor gain, torch::Tensor s2, torch::Tensor hid, std::vector<torch::Tensor> scv, int64_t sc_local,
                 double sc_eps, double eps_post, double eps, int64_t cfg, int64_t pdl) {
    const int T = (int)s1.size(0), spl = (int)(cfg % 10);
    cudaLaunchConfig_t lc = {}; lc.gridDim = T * spl; lc.blockDim = bnd2::NT; lc.dynamicSmemBytes = 0;
    lc.stream = c10::cuda::getCurrentCUDAStream(); cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1; lc.attrs = la; lc.numAttrs = pdl ? 1 : 0;
    const bool S = scv.size() == 3;
#define BND2_ARGS br.data_ptr(), (const float*)s1.data_ptr(), (const float*)alpha.data_ptr(), (const float*)beta.data_ptr(), \
        (const float*)bgain.data_ptr(), (const float*)gain.data_ptr(), (float*)s2.data_ptr(), (__nv_bfloat16*)hid.data_ptr(), \
        S ? (const float*)scv[0].data_ptr() : nullptr, S ? (const int*)scv[1].data_ptr() : nullptr, \
        S ? (const float*)scv[2].data_ptr() : nullptr, (int)sc_local, (float)sc_eps, (float)eps_post, (float)eps
#define BND2_C(c, nb, m, sp) else if (cfg == c) cudaLaunchKernelEx(&lc, bnd2::bnd2<nb, m, sp>, BND2_ARGS); \
    else if (cfg == c + 1000) cudaLaunchKernelEx(&lc, bnd2::bnd2<nb, m, sp, true>, BND2_ARGS);
    if (false) {}
    BND2_C(101, 1024, 0, 1) BND2_C(102, 1024, 0, 2) BND2_C(104, 1024, 0, 4)
    BND2_C(121, 1024, 2, 1) BND2_C(122, 1024, 2, 2) BND2_C(124, 1024, 2, 4) BND2_C(128, 1024, 2, 8)
    BND2_C(108, 1024, 0, 8) BND2_C(808, 8192, 0, 8) BND2_C(818, 8192, 1, 8)
    BND2_C(801, 8192, 0, 1) BND2_C(802, 8192, 0, 2) BND2_C(804, 8192, 0, 4)
    BND2_C(811, 8192, 1, 1) BND2_C(812, 8192, 1, 2) BND2_C(814, 8192, 1, 4)
    else TORCH_CHECK(false, "bnd2: cfg");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
