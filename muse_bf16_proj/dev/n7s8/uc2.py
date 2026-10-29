"""n7s8 gate, form 2 (K split over a 2-CTA cluster): CTA pair = one 128-row U tile, CTA cr does k in [cr*K/2, (cr+1)*K/2);
2 consumer warpgroups (64 rows each) x NT tokens, 1 producer warp; stage = U box 128 x 64 k (16 KB) + x NT x 64 k.
End: each CTA's 'giver' WG (cr0: rows 64-127, cr1: rows 0-63) parks its FP32 acc in its own smem, cluster barrier, the
peer's 'keeper' WG adds it via DSMEM and stores BF16 out rows. 64 tiles x 2 = 128 CTAs.
Usage: python uc2.py [NT,...] [NS,...]"""
import sys, torch
from torch.utils.cpp_extension import load_inline
import uc

NTS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "208,256").split(",")]
import os
XBOX = int(os.environ.get("XBOX", "16"))
DIAG = int(os.environ.get("DIAG", "0"))
UPK = int(os.environ.get("UPK", "1"))
NSS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "3,4,5").split(",")]


def smem(nt, ns):
    return 1024 + max(ns * (16384 + nt * 128), 64 * nt * 4 + 64 * nt * 2 + 1024) + 64


BULK = r"""
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
"""


def pack(u):
    """[n, k] -> [n/128][k/64][128 rows][64 k] with the SW128 16-B chunk swizzle (chunk c of row r at c ^ (r & 7))."""
    n, k = u.shape
    t = u.view(n // 128, 128, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [tile][ks][r][c][8]
    r = torch.arange(128, device=u.device).view(128, 1)
    src = torch.arange(8, device=u.device).view(1, 8) ^ (r & 7)  # stored chunk c' holds source chunk c' ^ (r & 7)
    return torch.take_along_dim(t, src.view(1, 1, 128, 8, 1).expand(t.shape), dim=3).contiguous()


CFGS = [(nt, ns) for nt in NTS for ns in NSS if smem(nt, ns) <= 227 * 1024]
PRE = uc.SRC.split("@WGMMA@")[0] + BULK
PRE = PRE.replace("namespace uc {", "#define XBOX " + str(XBOX) + "\n#define DIAG " + str(DIAG) + "\n#define UPK " + str(UPK) + "\nnamespace uc {")
KERN = r"""
template <int NT, int NS>
__global__ void __launch_bounds__(384, 1) gemmk(const __grid_constant__ CUtensorMap umap, const __grid_constant__ CUtensorMap xmap,
                                                __nv_bfloat16* __restrict__ out, int n, int k, const unsigned char* __restrict__ up) {
    constexpr int XS = NT * 128, SL = 16384 + XS, NB = NT / 16, XB = XBOX / 16;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    constexpr int RING = (NS * SL > 64 * NT * 6 + 1024) ? NS * SL : 64 * NT * 6 + 1024;
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem), bars = ring + RING;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31;
    const unsigned cr = blockIdx.x & 1;
    const int row0 = (blockIdx.x >> 1) * 128, S = k / 128, k0 = cr * (k / 2);
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (w >= 8) {  // producer warpgroup
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (w == 8 && lane == 0) {
            for (int j = 0; j < S; ++j) {
                const int slot = j % NS;
                if (j >= NS) bar_wait(bars + 8 * (NS + slot), ((j / NS) - 1) & 1);
                const unsigned sb = ring + slot * SL;
                if (DIAG == 2 || DIAG == 5 || DIAG == 7 || DIAG == 8) { bar_arrive(bars + 8 * slot); continue; }
                bar_expect(bars + 8 * slot, DIAG == 3 ? XS : DIAG == 4 ? 16384 : SL);
                if (DIAG != 3) {
                    if (UPK) bulk(sb, up + ((long)(blockIdx.x >> 1) * (k / 64) + k0 / 64 + j) * 16384, 16384, bars + 8 * slot);
                    else tma2d(sb, &umap, k0 + j * 64, row0, bars + 8 * slot);
                }
#pragma unroll
                for (int b = 0; b < (DIAG == 4 ? 0 : NB / XB); ++b) tma2d(sb + 16384 + b * 2048 * XB, &xmap, k0 + j * 64, b * 16 * XB, bars + 8 * slot);
            }
        }
        __syncwarp();
        cluster_sync();
        cluster_sync();
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    const int wg = w >> 2, wl = w & 3, tw = t & 127;
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
        if (DIAG < 1 || DIAG == 2) for (int kk = 0; kk < 4; ++kk) wgN<NT>(acc, desc(sb + wg * 8192 + kk * 32), desc(sb + 16384 + kk * 32));
        wg_commit();
        wg_wait<1>();
        __syncwarp();
        if (lane == 0 && pslot >= 0) bar_arrive(bars + 8 * (NS + pslot));
        pslot = slot;
    }
    wg_wait<0>();
#pragma unroll
    for (int e = 0; e < NT / 2; ++e) fence_f(acc[e]);
    asm volatile("bar.sync 1, 256;" ::: "memory");  // both WGs done reading the ring
    if (DIAG == 7) { cluster_sync(); cluster_sync(); return; }
    const bool giver = wg != (int)cr;  // cr0 keeps rows 0-63 (wg 0), cr1 keeps rows 64-127 (wg 1)
    float4* park = (float4*)smem;  // [NT/8][128 threads] float4
    if (giver && DIAG != 8) {
#pragma unroll
        for (int e = 0; e < NT / 8; ++e) park[e * 128 + tw] = make_float4(acc[4 * e], acc[4 * e + 1], acc[4 * e + 2], acc[4 * e + 3]);
    }
    if (DIAG != 8) cluster_sync();
    if (!giver) {
        unsigned rp;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rp) : "r"(ring), "r"(cr ^ 1));
#pragma unroll
        for (int e = 0; e < (DIAG == 8 ? 0 : NT / 8); ++e) {
            float4 v;
            asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rp + (e * 128 + tw) * 16));
            acc[4 * e] += v.x; acc[4 * e + 1] += v.y; acc[4 * e + 2] += v.z; acc[4 * e + 3] += v.w;
        }
        unsigned char* st = smem + 64 * NT * 4 + 1024;  // [NT][64] bf16, row stride 136 B
        const int g = lane >> 2, c = lane & 3;
#pragma unroll
        for (int jj = 0; jj < NT / 8; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int tok = 8 * jj + 2 * c + (e & 1), r = 16 * wl + g + 8 * (e >> 1);
                *(__nv_bfloat16*)(st + tok * 136 + r * 2) = __float2bfloat16(acc[jj * 4 + e]);
            }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        const int rb = row0 + 64 * cr;
        for (int i = tw; i < NT * 8; i += 128) {
            const int tok = i >> 3, q = i & 7;
            const uint2 a = *(const uint2*)(st + tok * 136 + q * 16), b = *(const uint2*)(st + tok * 136 + q * 16 + 8);
            *(uint4*)(out + (long)tok * n + rb + q * 8) = make_uint4(a.x, a.y, b.x, b.y);
        }
    }
    if (DIAG == 8) cluster_sync();
    cluster_sync();  // the peer has read our park
}
}  // namespace uc
"""
POST = uc.SRC.split("}  // namespace uc", 1)[1].split("void uc_gemm(")[0]
ENTRY = r"""
void uc_gemmk(torch::Tensor u, torch::Tensor x, torch::Tensor out, int64_t nt, int64_t ns, torch::Tensor up) {
    const int n = u.size(0), k = u.size(1);
    CUtensorMap umap, xmap;
    uc_map(umap, u.data_ptr(), n, k, 128);
    uc_map(xmap, x.data_ptr(), x.size(0), k, XBOX);
    const int sm = @SM@;
    auto stream = c10::cuda::getCurrentCUDAStream();
@CASES@
    TORCH_CHECK(false, "uc: no config");
}
"""
CASE = """    if (nt == {nt} && ns == {ns}) {{
        auto kern = uc::gemmk<{nt}, {ns}>;
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        cudaLaunchConfig_t cfg = {{}};
        cfg.gridDim = dim3(n / 64); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = sm; cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = 2; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kern, umap, xmap, (__nv_bfloat16*)out.data_ptr(), n, k, (const unsigned char*)up.data_ptr());
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }}
"""


def build():
    wg = "".join(uc.wgmma_fn(nt).replace(f"void wg{nt}(", f"void wg{nt}_(") for nt in sorted(set(NTS)))
    disp = "template <int NT> __device__ __forceinline__ void wgN(float* d, unsigned long long da, unsigned long long db) {\n"
    disp += "".join(f"    if constexpr (NT == {nt}) wg{nt}_(d, da, db);\n" for nt in sorted(set(NTS))) + "}\n"
    sm = "1024 + std::max<int>(ns * (16384 + nt * 128), 64 * nt * 6 + 1024) + 64"
    src = PRE + wg + disp + KERN + POST + ENTRY.replace("@SM@", sm).replace("@CASES@", "".join(CASE.format(nt=a, ns=b) for a, b in CFGS))
    return load_inline(f"pj_uck{XBOX}d{DIAG}u{UPK}_" + "_".join(map(str, NTS)) + "_" + "_".join(map(str, NSS)),
                       "void uc_gemmk(torch::Tensor u, torch::Tensor x, torch::Tensor out, int64_t nt, int64_t ns, torch::Tensor up);",
                       cuda_sources=src, functions=["uc_gemmk"], with_cuda=True,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"], extra_ldflags=["-lcuda"])


if __name__ == "__main__":
    mod = build()
    torch.manual_seed(0)
    N, K = 8192, 4096
    u = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    y = torch.randn(256, K, device="cuda").bfloat16()
    up = pack(u)
    e = uc.time_graph(lambda: None, 200)
    for m in (256, 208):
        o = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
        print(f"torch.mm M={m}: {uc.time_graph(lambda: torch.mm(y[:m], u.t(), out=o)) - e:.2f} us")
    KK = int(os.environ.get('KK', '0'))
    if KK: u = u[:, :KK].contiguous(); up = pack(u); y = y[:, :KK].contiguous()
    for nt, ns in CFGS:
        o = torch.empty(nt, N, dtype=torch.bfloat16, device="cuda")
        mod.uc_gemmk(u, y[:nt], o, nt, ns, up); torch.cuda.synchronize()
        ref = torch.mm(y[:nt], u.t())
        err = (o.float() - ref.float()).abs().max().item()
        tt = uc.time_graph(lambda: mod.uc_gemmk(u, y[:nt], o, nt, ns, up)) - e
        print(f"ucK NT={nt} NS={ns}: {tt:.2f} us  maxerr {err:.3g}  smem {smem(nt, ns)}")
