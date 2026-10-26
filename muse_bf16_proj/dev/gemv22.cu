// gemv22: GEMM for m = 128 / 256 tokens (the layer's T=128/256: verify 16 / 32) over lossless 12-bit BF16 weights
// (gemv19's packed atoms, 4-bit exponent codes + per-row escape headers), FP32 output [m, n].
// Swap-AB wgmma: W rows = the wgmma M (64 per warpgroup, A decoded into registers), tokens = N (128 or 256, B = x by a
// SW128 tensor map). Tile-major stream-K: stage u = (128-row tile, 64-k step), each block streams a contiguous range of
// stages; W (12 KB) and the stage's x box (M x 64 k) share one ring slot. Consumers (8 warps = 2 warpgroups) flush their
// partials at run ends (tile change or range end) by red.global.add into out, which a zero kernel cleared first.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#ifdef G22_NOPEER
#define G22_NOPEERV 1
#else
#define G22_NOPEERV 0
#endif
#if defined(G22_NOFIX) || defined(G22_NOW)
#define G22_NOFIXV 1
#else
#define G22_NOFIXV 0
#endif
namespace g22 {
constexpr int ESC = 8;
#ifdef G22_TRACE
__device__ unsigned long long trace[132 * 64];
__device__ __forceinline__ unsigned long long gt() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#define TR(slot) trace[blockIdx.x * 64 + (slot)] = gt()
#else
#define TR(slot)
#endif
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned lop_aob(unsigned a, unsigned b, unsigned c) {  // s34: (a & b) | c
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
template <int P>
__device__ __forceinline__ unsigned expo(unsigned nw) {
    const unsigned s = P == 0 ? nw << 7 : P == 1 ? nw << 3 : P == 2 ? nw >> 1 : nw >> 5;
    const unsigned z = gridDim.y >> 16, m2 = 0x07800780u >> z, cc = 0x38003800u >> z;  // s34 DEC4: opaque (== the immediates)
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(s), "r"(m2), "r"(cc));  // (s & m2) | cc
    return r;
}
template <int P>  // s35 DEC3: one prmt (exponent constant 0x38 from a register) + one lop3 per BF16 pair
__device__ __forceinline__ unsigned decw(unsigned sw, unsigned nw, unsigned sel4, unsigned sel3) {
#ifdef PJ_DEC4  // s34 layout (sign in the byte, code nibbles): for A/B builds only
    return lop_aob(prmt(sw, sel4), 0x807f807fu, expo<P>(nw));
#else
    const unsigned sh = P == 0 ? nw : P == 1 ? nw << 4 : P == 2 ? nw << 8 : __funnelshift_r(nw, nw, 4);
    const unsigned k38 = 0x38383838u >> (gridDim.y >> 16);  // opaque register (== the constant)
    unsigned p;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(p) : "r"(sw), "r"(k38), "r"(sel3));
    return lop_aob(sh, 0x87008700u, p);
#endif
}
__device__ __forceinline__ uint4 dec8(unsigned s0, unsigned s1, unsigned nw) {
    uint4 r;
    r.x = decw<0>(s0, nw, 0x9180u, 0x4140u);
    r.y = decw<1>(s0, nw, 0xB3A2u, 0x4342u);
    r.z = decw<2>(s1, nw, 0x9180u, 0x4140u);
    r.w = decw<3>(s1, nw, 0xB3A2u, 0x4342u);
    return r;
}
__device__ __forceinline__ void mma(float* c, const uint4& a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
}
__device__ __forceinline__ void bar_init(unsigned b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count));
}
__device__ __forceinline__ void bar_expect(unsigned b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
// arrive on the mbarrier at the same smem offset in CTA `peer` of the cluster
__device__ __forceinline__ void bar_arrive_peer(unsigned b, unsigned peer) {
#ifdef G22_NOPEER
    return;
#endif
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(b), "r"(peer));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(r) : "memory");
}
__device__ __forceinline__ void tma2d_mc(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar, unsigned short mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW22_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W22_%=;\n}"
                 :: "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, int bytes, unsigned b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory");
}
#ifdef G22_NOWG
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
template <int K> __device__ __forceinline__ void wg_wait() {}
#else
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int K>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K) : "memory"); }
#endif
__device__ __forceinline__ void fence_r(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void fence_f(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void tma2d(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
// K-major SW128 B descriptor (8 token rows x 128 B core groups, 1024 B apart)
__device__ __forceinline__ unsigned long long sw128_desc(unsigned a) {
    return (unsigned long long)((a & 0x3FFFF) >> 4) | (1ull << 16) | ((unsigned long long)(1024 >> 4) << 32) | (1ull << 62);
}
template <int M> __device__ __forceinline__ void wgmma(float* d, const unsigned* a, unsigned long long desc);
template <> __device__ __forceinline__ void wgmma<16>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<32>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                   "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<128>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
template <> __device__ __forceinline__ void wgmma<256>(float* d, const unsigned* a, unsigned long long desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, 1, 1, 1, 0;"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

// one ring stage of a consumer warp: AP atoms (32 k) of its 16-row group decoded into af (one A fragment per k16),
// then 2 * AP wgmma m64nMk16 of its warpgroup over the slot's x box(es) ([M tokens][64 k] SW128 per 64 k)
template <int M, int AP>
__device__ __forceinline__ void stage(float* acc, unsigned (&af)[2 * AP][4], unsigned sb, unsigned nb, unsigned xbuf) {
#pragma unroll
    for (int j = 0; j < AP; ++j) {
        uint4 sv;
        uint2 nv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sv.x), "=r"(sv.y), "=r"(sv.z), "=r"(sv.w) : "r"(sb + j * 512));
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(nv.x), "=r"(nv.y) : "r"(nb + j * 256));
        const uint4 a0 = dec8(sv.x, sv.y, nv.x), a1 = dec8(sv.z, sv.w, nv.y);
        af[2 * j][0] = a0.x; af[2 * j][1] = a0.y; af[2 * j][2] = a0.z; af[2 * j][3] = a0.w;
        af[2 * j + 1][0] = a1.x; af[2 * j + 1][1] = a1.y; af[2 * j + 1][2] = a1.z; af[2 * j + 1][3] = a1.w;
    }
    wg_fence();
#pragma unroll
    for (int j = 0; j < 2 * AP; ++j) {
        const int kk = j * 16;
#ifdef G22_NOMMA
        acc[j & 7] += __uint_as_float(af[j][0] ^ af[j][1] ^ af[j][2] ^ af[j][3] ^ (unsigned)kk ^ xbuf);
#else
        wgmma<M>(acc, af[j], sw128_desc(xbuf + (kk >> 6) * M * 128 + (kk & 63) * 2));
#endif
    }
    wg_commit();
}
// M tokens (128/256), AP atoms (32 k) per warp per stage (AP * 32 = k per stage, a multiple of 64), NS ring stages.
// WP: packed stages, tile-major (u = tile * S + ks, S = k / (32 AP)). out FP32 [M, n], zeroed before (PDL primary).
// packed stage (PB bytes): W atoms (SB) | E escapes (row in tile | k in stage << 8, true / scale; row -1: none)
// smem (1024-aligned): ring [NS][stage PB | pad to XO | XS x] | barriers | flush scratch
template <int M, int AP, int NS, int E, int CL>
__global__ void __launch_bounds__(384, 1) gemm(const unsigned char* __restrict__ WP,                                                float scale, const __grid_constant__ CUtensorMap xmap,
                                               const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int n, int k, int npre) {
    constexpr int NCW = 8, NT = M / 8, SB = NCW * AP * 768, XS = AP / 2 * M * 128, PB = SB + 8 * E, XO = SB + 1024, SL = XO + XS;
    static_assert(AP % 2 == 0 && SB % 1024 == 0 && E % 32 == 0 && E <= 128, "config");
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem = smem_raw + ((1024 - ((unsigned)__cvta_generic_to_shared(smem_raw) & 1023)) & 1023);
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned bars = ring + NS * SL;
    const int t = threadIdx.x, lane = t & 31, w = __shfl_sync(0xffffffffu, t >> 5, 0), g = lane >> 2, c = lane & 3;
    const int S = k / (32 * AP), ntr = n / (16 * NCW);
    // tile-aligned split: tile t gets nb = G / ntr (+1 for the first G % ntr tiles) blocks: one run per block
    // CL 2: clusters of 2 blocks on tiles (2p, 2p + 1) with the same k range; each loads half the x box, multicast to both
    const int cr = CL == 2 ? blockIdx.x & 1 : 0;
    int ra0, rb0;
    {
        const int G = gridDim.x / CL, b = blockIdx.x / CL, base = G / (ntr / CL), ex = G % (ntr / CL);
        int tt, j, nb;
        if (b < ex * (base + 1)) { tt = b / (base + 1); j = b % (base + 1); nb = base + 1; }
        else { const int b2 = b - ex * (base + 1); tt = ex + b2 / base; j = b2 % base; nb = base; }
        tt = tt * CL + cr;
        ra0 = tt * S + S * j / nb; rb0 = tt * S + S * (j + 1) / nb;
    }
    if (t == 0) {
        for (int i = 0; i < NS; ++i) { bar_init(bars + 8 * i, 1); bar_init(bars + 8 * (NS + i), (NCW + (G22_NOFIXV ? 0 : 1)) * (G22_NOPEERV ? 1 : CL)); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (CL == 1) __syncthreads();
    else cluster_sync();  // the peer's barriers are initialized before any multicast / remote arrive
    if (t == 0) TR(0);
    if (w >= NCW) {  // producer warpgroup: one lane streams the block's stages
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (w == NCW && lane == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
            int i = 0;
            // W of the first stages may stream before the previous kernel ends (x must wait)
            for (; i < npre && i < NS && ra0 + i < rb0; ++i) {
                bar_expect(bars + 8 * i, PB + XS);
                bulk(ring + i * SL, WP + (long)(ra0 + i) * PB, PB, bars + 8 * i);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            TR(1);
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS, ks = u % S;
                if (j >= i) {
                    if (j >= NS) bar_wait(bars + 8 * (NS + slot), ((j / NS) - 1) & 1);
#ifdef G22_NOX
                    bar_expect(bars + 8 * slot, u == ra0 ? PB + XS : PB);
#elif defined(G22_NOW)
                    bar_expect(bars + 8 * slot, XS);  // diag: no W loads (consumers decode stale smem)
#else
                    bar_expect(bars + 8 * slot, PB + XS);
#endif
#ifndef G22_NOW
                    bulk(ring + slot * SL, WP + (long)u * PB, PB, bars + 8 * slot);
#endif
                }
#ifdef G22_NOX
                if (u == ra0) {
#else
                {
#endif
#pragma unroll
                for (int b = 0; b < AP / 2; ++b) {
#ifdef G22_NOMC
                    if (true) { tma2d(ring + slot * SL + XO + b * M * 128, &xmap, ks * AP * 32 + b * 64, 0, bars + 8 * slot);
                        tma2d(ring + slot * SL + XO + b * M * 128 + (M / 2) * 128, &xmap, ks * AP * 32 + b * 64, M / 2, bars + 8 * slot); }
#else
                    if (CL == 1) tma2d(ring + slot * SL + XO + b * M * 128, &xmap, ks * AP * 32 + b * 64, 0, bars + 8 * slot);
#endif
                    else tma2d_mc(ring + slot * SL + XO + b * M * 128 + cr * (M / 2) * 128, &xmap, ks * AP * 32 + b * 64, cr * (M / 2), bars + 8 * slot, 3);
                }
                }
            }
        } else if (w == NCW + 1 && !G22_NOFIXV) {  // fixup warp: escapes, x from the ring slot, red.add into out
            // escapes of stage u: EV[EST[u] .. EST[u + 1]) = (row | (k % 32AP) << 16, true / scale); the block's are contiguous
            asm volatile("griddepcontrol.wait;" ::: "memory");
            for (int u = ra0, j = 0; u < rb0; ++u, ++j) {
                const int slot = j % NS;
                bar_wait(bars + 8 * slot, (j / NS) & 1);
                const unsigned xb = ring + slot * SL + XO;
                const int row0 = (u / S) * 128;
#pragma unroll 1
                for (int e0 = 0; e0 < E; e0 += 32) {
                    int2 ev;
                    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(ev.x), "=r"(ev.y) : "r"(ring + slot * SL + SB + (e0 + lane) * 8));
                    const unsigned full = __ballot_sync(0xffffffffu, ev.x >= 0);
                    unsigned bal = full;
                    while (bal) {
                        const int src = __ffs(bal) - 1;
                        bal &= bal - 1;
                        const int ex = __shfl_sync(0xffffffffu, ev.x, src), row = row0 + (ex & 0xFF), kk = ex >> 8;
                        const float hy = __int_as_float(__shfl_sync(0xffffffffu, ev.y, src));
#pragma unroll 4
                        for (int tok = lane; tok < M; tok += 32) {
                            const unsigned ad = xb + (kk >> 6) * M * 128 + tok * 128 + ((((kk & 63) >> 3) ^ (tok & 7)) << 4) + (kk & 7) * 2;
                            unsigned short xr;
                            asm volatile("ld.shared.u16 %0, [%1];" : "=h"(xr) : "r"(ad));
                            const float xv = __uint_as_float((unsigned)xr << 16);
                            asm volatile("red.global.add.f32 [%0], %1;" :: "l"(out + (long)tok * n + row), "f"(fmaf(xv, hy, -xv * 0x1p-15f) * scale));
                        }
                    }
                    if (full != 0xffffffffu) break;
                }
                __syncwarp();
                if (lane == 0) { bar_arrive(bars + 8 * (NS + slot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + slot), cr ^ 1); }
            }
            if (lane == 0) TR(7);
        }
        __syncwarp();
        if (CL == 2) cluster_sync();  // no block exits while its peer may still multicast into it / arrive on it
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[NT * 4];  // wgmma m64nM accumulator: per n8 tile jj, [4] = mma.sync C layout (rows g, g + 8; tokens 2c, 2c + 1)
#pragma unroll
    for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
    unsigned afA[2 * AP][4], afB[2 * AP][4];
    int i = 0, pslot = -1, nrun = 0;
    // one stage with A set af (ao: the previous stage's set, maybe in flight)
#define G22_STEP(af, ao) { \
        const int slot = i % NS; \
        bar_wait(bars + 8 * slot, (i / NS) & 1); \
        stage<M, AP>(acc, af, ring + slot * SL + w * AP * 512 + lane * 16, ring + slot * SL + NCW * AP * 512 + w * AP * 256 + lane * 8, ring + slot * SL + XO); \
        wg_wait<1>(); \
        _Pragma("unroll") for (int j = 0; j < 2 * AP; ++j) _Pragma("unroll") for (int e = 0; e < 4; ++e) fence_r(ao[j][e]); \
        __syncwarp(); \
        if (lane == 0 && pslot >= 0) { bar_arrive(bars + 8 * (NS + pslot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + pslot), cr ^ 1); } \
        pslot = slot; ++i; }
#pragma unroll 1
    for (int ra = ra0; ra < rb0;) {  // runs: the block's range split at tile boundaries
        const int tr = ra / S, rb = min(rb0, (tr + 1) * S), nst = rb - ra;
        int s = 0;
        if (t == 0 && nrun == 0) { bar_wait(bars, 0); TR(2); }
#pragma unroll 1
        for (; s + 1 < nst; s += 2) {
            G22_STEP(afA, afB)
            G22_STEP(afB, afA)
        }
        if (s < nst) G22_STEP(afA, afB)
        wg_wait<0>();
        if (t == 0) TR(3 + 2 * nrun);
#pragma unroll
        for (int j = 0; j < 2 * AP; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { fence_r(afA[j][e]); fence_r(afB[j][e]); }
#pragma unroll
        for (int e = 0; e < NT * 4; ++e) fence_f(acc[e]);
        __syncwarp();
        if (lane == 0) { bar_arrive(bars + 8 * (NS + pslot)); if (CL == 2) bar_arrive_peer(bars + 8 * (NS + pslot), cr ^ 1); }
        pslot = -1;
        constexpr int G4 = NT < 4 ? NT : 4;
        // flush: 32-token groups through smem ([tok][16 rows], stride 20 floats), red.add.v4 along rows
        {
            float* sc = (float*)(smem + NS * SL + 16 * NS + 64) + w * 32 * 20;
            float* ob = out + (tr * NCW + w) * 16;
#pragma unroll
            for (int t0 = 0; t0 < NT; t0 += G4) {
#pragma unroll
                for (int jj = t0; jj < t0 + G4; ++jj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) sc[((jj - t0) * 8 + 2 * c + (e & 1)) * 20 + g + 8 * (e >> 1)] = acc[jj * 4 + e] * scale;
                __syncwarp();
#pragma unroll
                for (int i2 = 0; i2 < G4; ++i2) {
                    const int f = lane + 32 * i2, tl = f >> 2, q4 = f & 3;
                    const float4 v = *(const float4*)(sc + tl * 20 + 4 * q4);
#ifdef G22_NORED
                    if (v.x == 1.2345f) ob[0] = v.y;
#else
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ob + (long)(t0 * 8 + tl) * n + 4 * q4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#endif
                }
                __syncwarp();
            }
#pragma unroll
            for (int e = 0; e < NT * 4; ++e) acc[e] = 0.f;
        }
        if (t == 0) TR(4 + 2 * nrun);
        ++nrun;
        ra = rb;
    }
    if (t == 0) TR(9);
    if (CL == 2) cluster_sync();
#undef G22_STEP
}
// PDL primary: lets the gemm launch at once and zeroes its output (the gemm waits before touching it)
__global__ void zero(float4* __restrict__ p, int n4) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
}  // namespace g22
#ifdef G22_TRACE
torch::Tensor g22_trace() {
    auto o = torch::zeros({132, 64}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    cudaMemcpyFromSymbol(o.data_ptr(), g22::trace, sizeof(unsigned long long) * 132 * 64);
    return o;
}
#endif
static PFN_cuTensorMapEncodeTiled_v12000 g22_encode = nullptr;
static void g22_xmap(CUtensorMap& map, void* ptr, int m, int k, int rows) {
    if (!g22_encode) {
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g22_encode, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(g22_encode, "g22: no cuTensorMapEncodeTiled");
    }
    cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)m};
    cuuint64_t strides[1] = {(cuuint64_t)k * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)rows};
    cuuint32_t es[2] = {1, 1};
    CUresult r = g22_encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "g22: cuTensorMapEncodeTiled failed ", (int)r);
}
// mode 0: zero kernel then the gemm (PDL dependent). mode 1: out already zero, the gemm a PDL dependent of the
// previous kernel (npre W stages stream before griddepcontrol.wait).
void g22_gemm(torch::Tensor wp, double scale, torch::Tensor x, torch::Tensor out,
              int64_t ap, int64_t ns, int64_t e, int64_t cl, int64_t grid, int64_t mode, int64_t npre) {
    const int m = x.size(0), k = x.size(1), n = out.size(1);
    const int sm = 1024 + ns * (8 * ap * 768 + 1024 + ap / 2 * m * 128) + 16 * ns + 64 + 8 * 32 * 20 * 4;
    CUtensorMap xmap;
    g22_xmap(xmap, x.data_ptr(), m, k, m / cl);
    TORCH_CHECK(n % 128 == 0 && k % (32 * ap) == 0 && sm <= 227 * 1024 && n / 128 <= grid && grid <= (n / 128) * (k / (32 * ap))
                && grid % cl == 0 && (n / 128) % cl == 0, "g22: shape/smem/grid ", sm);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (!mode) g22::zero<<<132, 256, 0, stream>>>((float4*)out.data_ptr(), m * n / 4);
#define G22_CASE(M_, AP_, NS_, E_, CL_) \
    if (m == M_ && ap == AP_ && ns == NS_ && e == E_ && cl == CL_) { \
        auto kern = g22::gemm<M_, AP_, NS_, E_, CL_>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        cudaLaunchConfig_t cfg = {}; \
        cfg.gridDim = dim3(grid); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = sm; cfg.stream = stream; \
        cudaLaunchAttribute attr[2]; \
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1; \
        attr[1].id = cudaLaunchAttributeClusterDimension; attr[1].val.clusterDim.x = CL_; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1; \
        cfg.attrs = attr; cfg.numAttrs = 2; \
        if (CL_ > 1) { int nc_ = 0; cudaOccupancyMaxActiveClusters(&nc_, kern, &cfg); \
            TORCH_CHECK(nc_ * CL_ >= grid, "g22: grid not co-resident as clusters: ", nc_, " clusters"); } \
        cudaLaunchKernelEx(&cfg, kern, (const unsigned char*)wp.data_ptr(), (float)scale, xmap, \
            (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), n, k, (int)npre); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    G22_CONFIGS
    TORCH_CHECK(false, "g22: no config");
}
