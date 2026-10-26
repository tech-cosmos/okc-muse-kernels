// gemv23: BF16 GEMM for m = 128 / 256 tokens (the layer's T=128/256: verify 16 / 32), out FP32 [m, n] (red.add) or BF16.
// Swap-AB SS wgmma: W rows = wgmma M (A = W box [BW rows][64 k] SW128 by tensor map), tokens = N (B = x box [M][64 k]).
// Tile-aligned split-K: tile = BW W rows; each block streams one contiguous k range of one tile (64 k per ring stage).
// 2 consumer warpgroups: BW 128 -> each owns 64 rows of the tile; BW 64 -> both own the same 64 rows, each does half the
// k16 steps of every stage, summed through smem at the end. Producer warpgroup: one lane issues the TMA loads.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
namespace g23 {
__device__ __forceinline__ void bar_init(unsigned b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW23_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W23_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
// K-major SW128 descriptor (8 rows x 128 B core groups, 1024 B apart)
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int N> __device__ __forceinline__ void wgss(float* d, unsigned long long da, unsigned long long db);
template <> __device__ __forceinline__ void wgss<16>(float* d, unsigned long long da, unsigned long long db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da), "l"(db));
}
template <> __device__ __forceinline__ void wgss<32>(float* d, unsigned long long da, unsigned long long db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(da), "l"(db));
}
template <> __device__ __forceinline__ void wgss<64>(float* d, unsigned long long da, unsigned long long db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da), "l"(db));
}
template <> __device__ __forceinline__ void wgss<128>(float* d, unsigned long long da, unsigned long long db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "l"(da), "l"(db));
}
template <> __device__ __forceinline__ void wgss<256>(float* d, unsigned long long da, unsigned long long db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, 1, 1, 1, 0, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "l"(da), "l"(db));
}


// out FP32 [M, n]: zeroed before (PDL primary) unless every tile has one block (G == n / BW): then plain stores.
// OB: out is BF16 (only with one block per tile).
template <int M, int BW, int NS, int OB, int CL>
__global__ void __launch_bounds__(384, 1) gemm(const unsigned char* __restrict__ WP, const __grid_constant__ CUtensorMap xmap,
                                               void* __restrict__ outp, int n, int k, int npre) {
    constexpr int WB = BW * 128, XB = M * 128, SL = WB + XB, NT = M / 8, NKS = BW == 128 ? 4 : 2;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned bars = ring + NS * SL;
    const int t = threadIdx.x, lane = t & 31, w = __shfl_sync(0xffffffffu, t >> 5, 0), g = lane >> 2, c = lane & 3;
    const int S = k / 64, ntr = n / BW;
    int tile, ra0, rb0, nb, rank = 0;
    if (CL > 1) {  // cluster of CL blocks per tile, each one k range; partials summed into rank 0 through DSMEM
        tile = blockIdx.x / CL; rank = blockIdx.x % CL; nb = 1;
        ra0 = S * rank / CL; rb0 = S * (rank + 1) / CL;
    } else {
        const int G = gridDim.x, b = blockIdx.x, base = G / ntr, ex = G % ntr;
        int j;
        if (b < ex * (base + 1)) { tile = b / (base + 1); j = b % (base + 1); nb = base + 1; }
        else { const int b2 = b - ex * (base + 1); tile = ex + b2 / base; j = b2 % base; nb = base; }
        ra0 = S * j / nb; rb0 = S * (j + 1) / nb;
    }
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (w >= 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (w == 8 && lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
            int i = 0;
            for (; i < npre && i < NS && ra0 + i < rb0; ++i) {
                bar_expect(bars + 8 * i, SL);
                bulk(ring + i * SL, WP + ((long)tile * S + ra0 + i) * WB, WB, bars + 8 * i);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS;
                if (j >= i) {
                    if (j >= NS) bar_wait(bars + 8 * (NS + slot), ((j / NS) - 1) & 1);
#if defined(G23_NOX)
                    bar_expect(bars + 8 * slot, WB);
#elif defined(G23_NOW)
                    bar_expect(bars + 8 * slot, XB);
#else
                    bar_expect(bars + 8 * slot, SL);
#endif
#ifndef G23_NOW
                    bulk(ring + slot * SL, WP + ((long)tile * S + u) * WB, WB, bars + 8 * slot);
#endif
                }
#ifndef G23_NOX
                tma2d(ring + slot * SL + WB, &xmap, u * 64, 0, bars + 8 * slot);
#endif
            }
        }
        __syncwarp();
        if (CL > 1) { cluster_sync(); cluster_sync(); }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    const int wg = w >> 2;
    float acc[NT * 4];
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
    const unsigned aoff = BW == 128 ? wg * 64 * 128 : 0, koff = BW == 128 ? 0 : wg * 64;  // bytes
    int pslot = -1;
#pragma unroll 1
    for (int u = ra0, i = 0; u < rb0; ++u, ++i) {
        const int slot = i % NS;
        bar_wait(bars + 8 * slot, (i / NS) & 1);
        const unsigned wb = ring + slot * SL + aoff + koff, xb = ring + slot * SL + WB + koff;
        wg_fence();
#ifndef G23_NOMMA
#pragma unroll
        for (int s = 0; s < NKS; ++s) wgss<M>(acc, sw128_desc(wb + s * 32), sw128_desc(xb + s * 32));
#endif
        wg_commit();
        wg_wait<1>();
        if (lane == 0 && pslot >= 0) bar_arrive(bars + 8 * (NS + pslot));
        pslot = slot;
    }
    wg_wait<0>();
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) fence_f(acc[e]);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (BW == 64) {  // WG 1 hands its partial to WG 0 through the (drained) ring
        float* sc = (float*)smem;
        const int tw = t & 127;
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (wg == 1) {
#pragma unroll
            for (int e = 0; e < NT * 4; ++e) sc[e * 128 + tw] = acc[e];
        }
        asm volatile("bar.sync 1, 256;" ::: "memory");
        if (wg == 1) { if (CL > 1) { cluster_sync(); cluster_sync(); } return; }
#pragma unroll
        for (int e = 0; e < NT * 4; ++e) acc[e] += sc[e * 128 + tw];
        asm volatile("bar.sync 2, 128;" ::: "memory");  // all of WG 0 read sc before the flush scratch (same smem) is reused
    }
    if (BW == 128) asm volatile("bar.sync 1, 256;" ::: "memory");  // both WGs done reading the ring (scratch reuses it)
    if (CL > 1) {
        constexpr int NC = BW == 128 ? 256 : 128;
        float4* pp = (float4*)smem;
        const int ct = t & (NC - 1);
        if (rank != 0) {
#pragma unroll
            for (int e = 0; e < NT; ++e) pp[e * NC + ct] = make_float4(acc[4 * e], acc[4 * e + 1], acc[4 * e + 2], acc[4 * e + 3]);
        }
        cluster_sync();
        if (rank == 0) {
#pragma unroll 1
            for (int r = 1; r < CL; ++r) {
                unsigned ra;
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(ring + ct * 16), "r"(r));
#pragma unroll
                for (int e = 0; e < NT; ++e) {
                    float4 v;
                    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ra + e * NC * 16));
                    acc[4 * e] += v.x; acc[4 * e + 1] += v.y; acc[4 * e + 2] += v.z; acc[4 * e + 3] += v.w;
                }
            }
        }
        cluster_sync();
        if (rank != 0) return;
    }
    const bool direct = nb == 1;
    constexpr int G4 = NT < 4 ? NT : 4;
    float* sc = (float*)smem + w * 32 * 20;
    const long rbase = (long)tile * BW + 16 * w;
#pragma unroll
    for (int t0 = 0; t0 < NT; t0 += G4) {
#pragma unroll
        for (int jj = t0; jj < t0 + G4; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sc[((jj - t0) * 8 + 2 * c + (e & 1)) * 20 + g + 8 * (e >> 1)] = acc[jj * 4 + e];
        __syncwarp();
#pragma unroll
        for (int i2 = 0; i2 < G4; ++i2) {
            const int f = lane + 32 * i2, tl = f >> 2, q4 = f & 3;
            const float4 v = *(const float4*)(sc + tl * 20 + 4 * q4);
            const long o = (long)(t0 * 8 + tl) * n + rbase + 4 * q4;
            if (OB) {
                __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b2 = __floats2bfloat162_rn(v.z, v.w);
                uint2 pk; pk.x = *(unsigned*)&a; pk.y = *(unsigned*)&b2;
                *(uint2*)((__nv_bfloat16*)outp + o) = pk;
            } else if (direct) {
                *(float4*)((float*)outp + o) = v;
            } else {
                asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"((float*)outp + o), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
            }
        }
        __syncwarp();
    }
}
// PDL primary: lets the gemm launch at once and zeroes its output (the gemm waits before touching it)
__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace g23
static PFN_cuTensorMapEncodeTiled_v12000 g23_encode = nullptr;
static void g23_map(CUtensorMap& map, void* ptr, int rows_total, int k, int rows) {
    if (!g23_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g23_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g23_encode, "g23: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows_total};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)rows};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g23_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g23: cuTensorMapEncodeTiled failed ", (int)r);
}
// mode 0: (zero kernel if split) then the gemm as its PDL dependent. mode 1: out already zero (or direct), the gemm a
// PDL dependent of the previous kernel (npre W stages stream before griddepcontrol.wait).
void g23_gemm(torch::Tensor w, int64_t n, torch::Tensor x, torch::Tensor out, int64_t bw, int64_t ns, int64_t cl, int64_t grid, int64_t mode, int64_t npre) {
    const int m = x.size(0), k = x.size(1);
    const int ob = out.scalar_type() == at::kBFloat16;
    const int sm = 1024 + ns * (bw * 128 + m * 128) + 16 * ns;
    TORCH_CHECK(n % bw == 0 && k % 64 == 0 && sm <= 227 * 1024 && n / bw <= grid && grid <= (n / bw) * (k / 64)
                && (!ob || grid == n / bw * cl) && (cl == 1 || grid == n / bw * cl) && out.size(0) == m && out.size(1) == n, "g23: shape/smem/grid ", sm);
    TORCH_CHECK(w.numel() * w.element_size() == (long)n * k * 2, "g23: packed W size");
    CUtensorMap xmap;
    g23_map(xmap, x.data_ptr(), m, k, m);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode && grid != n / bw * cl) g23::zero<<<132, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4);
#define G23_CASE(M_, BW_, NS_, OB_, CL_) \
    if (m == M_ && bw == BW_ && ns == NS_ && ob == OB_ && cl == CL_) { \
        auto kern = g23::gemm<M_, BW_, NS_, OB_, CL_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[2]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        attr[1].id = cudaLaunchAttributeClusterDimension; attr[1].val.clusterDim.x = CL_; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1; \
        cfg.attrs = attr; cfg.numAttrs = 2; \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)w.data_ptr(), xmap, out.data_ptr(), n, k, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G23_CONFIGS
    TORCH_CHECK(false, "g23: no config");
}
