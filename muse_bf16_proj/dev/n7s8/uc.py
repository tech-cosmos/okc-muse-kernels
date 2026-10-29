"""n7s8 gate: N-compacted T=256 unpool GEMM (lead 07:40Z). out[tok, 8192] BF16 = y[tok, 4096] @ U[8192, 4096]^T over the
first `cnt` (compacted) tokens only. Swap-AB wgmma SS: 64 U rows per CTA = wgmma M (one consumer warpgroup), tokens = wgmma N
(compile-time NT, a multiple of 16); 128 CTAs in clusters of CL (x boxes multicast across the cluster, U per CTA); TMA SW128
rings of NS stages (64 k each: U 8 KB + x NT*128 B). Gate: ours at NT=208 vs torch.mm (nvjet) at M=256 and M=208.
Usage: python uc.py [NT,...]   (H100; cold L2 like the judge)."""
import os, sys, fcntl, torch
from torch.utils.cpp_extension import load_inline

NTS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "208,256").split(",")]
CFGS = [(nt, ns, cl) for nt in NTS for ns in (3, 4, 5) for cl in (1, 2) if ns * (8192 + nt * 128) + 1024 + 256 <= 227 * 1024]


def wgmma_fn(nt):
    nd = nt // 2
    regs = ",".join(f"%{i}" for i in range(nd))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nd))
    return (f'__device__ __forceinline__ void wg{nt}(float* d, unsigned long long da, unsigned long long db) {{\n'
            f'    asm volatile("wgmma.mma_async.sync.aligned.m64n{nt}k16.f32.bf16.bf16 {{{regs}}}, %{nd}, %{nd + 1}, 1, 1, 1, 0, 0;"\n'
            f'                 : {outs} : "l"(da), "l"(db));\n}}\n')


SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
namespace uc {
__device__ __forceinline__ void bar_init(unsigned b, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count)); }
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_arrive_peer(unsigned b, unsigned peer) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(b), "r"(peer));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nWU_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WU_%=;\n}" :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d_mc(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar, unsigned short mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// K-major SW128 descriptor: 8 rows x 128 B core groups, 1024 B apart
__device__ __forceinline__ unsigned long long desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
@WGMMA@
// smem: ring [NS][U 8 KB | x NT*128 B] | out stage [NT][64] bf16 (+pad) | barriers
template <int NT, int NS, int CL>
__global__ void __launch_bounds__(256, 1) gemm(const __grid_constant__ CUtensorMap umap, const __grid_constant__ CUtensorMap xmap,
                                               __nv_bfloat16* __restrict__ out, int n, int k) {
    constexpr int XS = NT * 128, SL = 8192 + XS, NB = NT / 16;  // x: NB boxes of 16 tokens
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), bars = ring + NS * SL;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, row0 = blockIdx.x * 64, S = k / 64;
    const unsigned cr = CL == 2 ? blockIdx.x & 1 : 0;
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), 4 * CL); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (CL == 1) __syncthreads(); else cluster_sync();
    if (w >= 4) {  // producer warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (w == 4 && lane == 0) {
            for (int j = 0; j < S; ++j) {
                const int slot = j % NS;
                if (j >= NS) bar_wait(bars + 8 * (NS + slot), ((j / NS) - 1) & 1);
                const unsigned sb = ring + slot * SL;
                bar_expect(bars + 8 * slot, SL);
                tma2d(sb, &umap, j * 64, row0, bars + 8 * slot);
#pragma unroll
                for (int b = 0; b < NB; ++b) {
                    if (CL == 1) tma2d(sb + 8192 + b * 2048, &xmap, j * 64, b * 16, bars + 8 * slot);
                    else if ((b & 1) == cr) tma2d_mc(sb + 8192 + b * 2048, &xmap, j * 64, b * 16, bars + 8 * slot, 3);
                }
            }
        }
        __syncwarp();
        if (CL == 2) cluster_sync();
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    float acc[NT / 2];
#pragma unroll
    for (int e = 0; e < NT / 2; ++e) acc[e] = 0.f;
    int pslot = -1;
#pragma unroll 1
    for (int j = 0; j < S; ++j) {
        const int slot = j % NS;
        bar_wait(bars + 8 * slot, (j / NS) & 1);
        const unsigned sb = ring + slot * SL;
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) wg@NT@<NT>(acc, desc(sb + kk * 32), desc(sb + 8192 + kk * 32));
        wg_commit();
        wg_wait<1>();
        __syncwarp();
        if (lane == 0 && pslot >= 0) { bar_arrive(bars + 8 * (NS + pslot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + pslot), cr ^ 1); }
        pslot = slot;
    }
    wg_wait<0>();
#pragma unroll
    for (int e = 0; e < NT / 2; ++e) fence_f(acc[e]);
    // stage [NT][64 rows] bf16 (row stride 136 B), then 16 B stores along rows
    unsigned char* st = smem + NS * SL + 1024;
    const int g = lane >> 2, c = lane & 3;
#pragma unroll
    for (int jj = 0; jj < NT / 8; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int tok = 8 * jj + 2 * c + (e & 1), r = 16 * w + g + 8 * (e >> 1);
            *(__nv_bfloat16*)(st + tok * 136 + r * 2) = __float2bfloat16(acc[jj * 4 + e]);
        }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    for (int i = t; i < NT * 8; i += 128) {
        const int tok = i >> 3, q = i & 7;
        const uint2 a = *(const uint2*)(st + tok * 136 + q * 16), b = *(const uint2*)(st + tok * 136 + q * 16 + 8);
        *(uint4*)(out + (long)tok * n + row0 + q * 8) = make_uint4(a.x, a.y, b.x, b.y);
    }
    if (CL == 2) cluster_sync();
}
}  // namespace uc
static PFN_cuTensorMapEncodeTiled_v12000 uc_encode = nullptr;
static void uc_map(CUtensorMap& map, void* ptr, int rows, int k, int brows) {
    if (!uc_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&uc_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(uc_encode, "uc: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)brows};
    cuuint32_t es[2] = {1, 1};
    CUresult r = uc_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "uc: encode failed ", (int)r);
}
void uc_gemm(torch::Tensor u, torch::Tensor x, torch::Tensor out, int64_t nt, int64_t ns, int64_t cl) {
    const int n = u.size(0), k = u.size(1);
    CUtensorMap umap, xmap;
    uc_map(umap, u.data_ptr(), n, k, 64);
    uc_map(xmap, x.data_ptr(), x.size(0), k, 16);
    const int sm = 1024 + ns * (8192 + nt * 128) + 1024 + nt * 136 + 64;
    auto stream = c10::cuda::getCurrentCUDAStream();
@CASES@
    TORCH_CHECK(false, "uc: no config");
}
"""
CASE = """    if (nt == {nt} && ns == {ns} && cl == {cl}) {{
        auto kern = uc::gemm<{nt}, {ns}, {cl}>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        cudaLaunchConfig_t cfg = {{}};
        cfg.gridDim = dim3(n / 64); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = sm; cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = {cl}; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kern, umap, xmap, (__nv_bfloat16*)out.data_ptr(), n, k);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }}
"""


def build():
    src = SRC.replace("@WGMMA@", "".join(wgmma_fn(nt).replace(f"void wg{nt}(", f"void wg{nt}_(") for nt in sorted(set(NTS))))
    # templated dispatch: wg@NT@<NT> -> a constexpr-if chain
    disp = "template <int NT> __device__ __forceinline__ void wgN(float* d, unsigned long long da, unsigned long long db) {\n"
    for nt in sorted(set(NTS)):
        disp += f"    if constexpr (NT == {nt}) wg{nt}_(d, da, db);\n"
    disp += "}\n"
    src = src.replace("// smem: ring", disp + "// smem: ring").replace("wg@NT@<NT>", "wgN<NT>")
    src = src.replace("@CASES@", "".join(CASE.format(nt=nt, ns=ns, cl=cl) for nt, ns, cl in CFGS))
    return load_inline("pj_uc_" + "_".join(map(str, NTS)), "void uc_gemm(torch::Tensor u, torch::Tensor x, torch::Tensor out, int64_t nt, int64_t ns, int64_t cl);", cuda_sources=src, functions=["uc_gemm"], with_cuda=True,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"],
                       extra_ldflags=["-lcuda"], verbose=False)


def time_graph(fn, reps=100):
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda").view(torch.int64); fs = torch.empty((), dtype=torch.int64, device="cuda")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(reps):
            torch.sum(flush, 0, out=fs)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
    del flush
    ts.sort()
    return ts[len(ts) // 2]


if __name__ == "__main__":
    mod = build()
    torch.manual_seed(0)
    N, K = 8192, 4096
    u = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    y = torch.randn(256, K, device="cuda").bfloat16()
    e = time_graph(lambda: None, 200)
    print(f"empty graph {e:.2f} us")
    for m in (256, 224, 208, 192):
        o = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
        print(f"torch.mm M={m}: {time_graph(lambda: torch.mm(y[:m], u.t(), out=o)) - e:.2f} us")
    for nt, ns, cl in CFGS:
        o = torch.empty(nt, N, dtype=torch.bfloat16, device="cuda")
        mod.uc_gemm(u, y[:nt], o, nt, ns, cl); torch.cuda.synchronize()
        ref = torch.mm(y[:nt], u.t())
        err = (o.float() - ref.float()).abs().max().item()
        tt = time_graph(lambda: mod.uc_gemm(u, y[:nt], o, nt, ns, cl)) - e
        print(f"uc NT={nt} NS={ns} CL={cl}: {tt:.2f} us  maxerr {err:.3g}")
