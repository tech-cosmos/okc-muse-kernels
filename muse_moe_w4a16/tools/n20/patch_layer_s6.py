# n20: port S6 (6-bit block-scale codes on the SQ stages, lossless, bitwise = SQ) into the layer's moe_src.py.
# python tools/n20/patch_layer_s6.py <path/to/moe_src.py>   (in place; asserts on every anchor)
# Effect: prepare builds w13s6/w2s6 instead of w13sq/w2sq when every scale byte is in [80, 0x7E] (else SQ as before);
# _tk2f_w returns cfg + 20000000 on them; moe_tk2f_launch runs PF | 1024 (SQ 512 + S6 1024) kernels for 184204 (+BAL)
# and the tk2h 14w004 cfgs. T=1 (layer _experts_n2u / non-SQ PF 4) is unchanged.
import sys
p = sys.argv[1]; s = open(p).read()


def region(sig):
    a = s.index(sig)
    b = s.index("\ntemplate <", a + 10)
    return a, b


def rep_in(sig, old, new, cnt=1):
    global s
    a, b = region(sig)
    body = s[a:b]
    assert body.count(old) == cnt, (sig[:40], old[:60], body.count(old))
    s = s[:a] + body.replace(old, new) + s[b:]


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


S6DEC = '''// moe n20 S6: 6-bit block-scale codes (prepare _s6_pack: c = e4m3 byte - 80, all scales in [80, 0x7E]). Per lane and
// SQ stage (KPS 2) 8 codes k = 4 h + r: n = the low nibbles (bits 4k), hw = the high 2 bits (bits 2k). One PRMT puts
// the nibble in bits 4-7 and the high bits in 8-9 of each half; | 0x7C00 = (c + 80) * 16 + 0x7700 (bits 4-9 of 0x7C00
// are 0): the same bf16x2 as sca<r>() of the original byte, 2 instructions per scale.
__device__ __forceinline__ void s6dec(uint32_t n, uint32_t hw, int h, uint32_t* sc) {
  const uint32_t n4 = n << 4;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const uint32_t b = 2 * h + (r >> 1), c = 4 + h;
    const uint32_t x = __byte_perm((r & 1) ? n : n4, hw >> (2 * r), b | (c << 4) | (b << 8) | (c << 12));
    sc[r] = (x & 0x03F003F0u) | 0x7C007C00u;
  }
}

'''
rep("__device__ __forceinline__ uint32_t u4get(const uint4& v, int i) {", S6DEC + "__device__ __forceinline__ uint32_t u4get(const uint4& v, int i) {")
# producer
P = "template <int NRT, int NC, int NSTG, int KPS, bool SQ = false>\n__device__ __forceinline__ void tk2f_produce("
rep_in(P, "SB = NC * KPS * 1152, KB = NC * KPS;", "SB = NC * KPS * (S6 ? 1120 : 1152), KB = NC * KPS;\n  static_assert(!S6 || (SQ && KPS == 2), \"S6: SQ stages, KPS 2\");")
rep(P, "template <int NRT, int NC, int NSTG, int KPS, bool SQ = false, bool S6 = false>\n__device__ __forceinline__ void tk2f_produce(")
LOAD_OLD = "sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];"
LOAD_NEW = "if (!S6) sw[h] = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane)[(h * NC + warp) * 32 + lane];"
S6LD = '''if (S6) {  // moe n20
{I}  const uint32_t* sb = reinterpret_cast<const uint32_t*>(sw4 + 2 * KPS * SW - warp * 32 - lane);
{I}  s6n = sb[warp * 32 + lane];
{I}  s6h = reinterpret_cast<const unsigned short*>(sb + NC * 32)[warp * 32 + lane];
{I}}
{I}'''
for sig, ind in (("template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA = false, int PF = 0>\n__device__ __forceinline__ void tk2f_consume(", "      "),
                 ("template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false, bool SLOT = false>\n__device__ __forceinline__ void tk2g_row(", "    ")):
    if "tk2g_row" in sig:
        rep_in(sig, "SB = NC * KPS * 1152", "SB = NC * KPS * (S6 ? 1120 : 1152)")
    else:  # keep tk2f_consume's first line byte-identical (layer.py's NU8 patch anchors on it): stage stride SB6
        rep_in(sig, "st * (SB / 16)", "st * (SB6 / 16)")
    rep_in(sig, "uint32_t sw[KPS];", "uint32_t sw[KPS], s6n = 0, s6h = 0;")
    rep_in(sig, LOAD_OLD, LOAD_NEW)
    rep_in(sig, "const int cs = st;", S6LD.replace("{I}", ind) + "const int cs = st;")
    i2 = ind + "  "
    rep_in(sig, "if (SA) {\n" + i2 + "  sc[0] = sca<0>(sw[h]);", "if (S6) {\n" + i2 + "  s6dec(s6n, s6h, h, sc);\n" + i2 + "} else if (SA) {\n" + i2 + "  sc[0] = sca<0>(sw[h]);")
rep_in("template <int NRT, int NC, int NSTG, bool GATED, int KPS, bool SA = false, int PF = 0>\n__device__ __forceinline__ void tk2f_consume(",
       "SW = NC * 32, SB = NC * KPS * 1152;\n",
       "SW = NC * 32, SB = NC * KPS * 1152;\n  constexpr bool S6 = (PF & 1024) != 0;  // moe n20: 6-bit scale codes\n  constexpr int SB6 = S6 ? NC * KPS * 1120 : SB;  // moe n20: S6 stage bytes\n")
rep("template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false, bool SLOT = false>\n__device__ __forceinline__ void tk2g_row(",
    "template <int NRT, int NC, int NSTG, bool GATED_, int KPS, bool SA, bool TWO, bool RTG = false, bool SLOT = false,\n          bool S6 = false>\n__device__ __forceinline__ void tk2g_row(")
rep("template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false>\n__device__ __forceinline__ void tk2h_consume(",
    "template <int NC, int NSTG, int KPS, bool SA, int GS, bool PSC = false, bool SLOT = false, bool S6 = false>\n__device__ __forceinline__ void tk2h_consume(")
rep("tk2g_row<0, NC, NSTG, false, KPS, SA, true, true, SLOT>(", "tk2g_row<0, NC, NSTG, false, KPS, SA, true, true, SLOT, S6>(")
rep("tk2g_row<0, NC, NSTG, false, KPS, SA, false, true, SLOT>(", "tk2g_row<0, NC, NSTG, false, KPS, SA, false, true, SLOT, S6>(")
rep("tk2f_produce<512, NC, NSTG, KPS, SQ>(", "tk2f_produce<512, NC, NSTG, KPS, SQ, (PF & 1024) != 0>(")
rep("tk2f_produce<256, NC, NSTG, KPS, SQ>(", "tk2f_produce<256, NC, NSTG, KPS, SQ, (PF & 1024) != 0>(")
rep("tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0>(", "tk2h_consume<NC, NSTG, KPS, SA, GS, MRG == 2, (PF & 4) != 0, (PF & 1024) != 0>(")
# launcher
rep("  const bool sq = cfg >= 10000000;", "  const bool s6 = cfg >= 20000000;  // moe n20: + 20000000 = S6 stages (w13_ / w2_ = _s6_pack buffers)\n  const bool sq = cfg >= 10000000;")
rep('''  if (sq) {  // SA 84204 only (+ BAL): d1 slot, T = 16 / 32
''', '''  if (s6) {  // moe n20: as SQ below, PF | 1024
    if (cfg >= 140000 && cfg < 150000 && cfg % 1000 == 4)
      return tk2f_go<8, 4, 2, false, false, true, 16, 1, 4 | 1536>(8 + (int)(cfg / 1000) % 10, TK2F_ARGS0);
    TORCH_CHECK(cfg == 184204, "moe_tk2f S6: cfg ", cfg);
    if (bal) return tk2f_go<8, 4, 2, false, false, true, 8, 0, 36 | 1536>(0, TK2F_ARGS0);
    return tk2f_go<8, 4, 2, false, false, true, 8, 0, 4 | 1536>(0, TK2F_ARGS0);
  }
  if (sq) {  // SA 84204 only (+ BAL): d1 slot, T = 16 / 32
''')
# python
rep('''def _pair_scales(ps):''', '''def _s6_pack(wq, s2, nq=4):
    """moe n20 S6: as _sq_pack, but the stage's scales as 6-bit codes c = e4m3 byte - 80 (caller checked 80 <= byte <= 0x7E):
    per (lr, q) {tile lr, tile lr + NRT/2, nibbles int32 [8 warps][32 lanes], high bits int16 [8][32]}. 17920 B/stage."""
    E, nrt = wq.shape[0], wq.shape[1]
    w = wq.reshape(E, 2, nrt // 2, nq, -1)
    c = s2.contiguous().view(torch.uint8).reshape(E, nrt // 2, nq, 2, 8, 32, 4).long() - 80  # E lr q h warp lane r
    c = c.permute(0, 1, 2, 4, 5, 3, 6).reshape(E, nrt // 2, nq, 8, 32, 8)                     # ... warp lane k
    k = torch.arange(8, device=c.device)
    nib = ((c & 15) << (4 * k)).sum(-1)
    hi = ((c >> 4) << (2 * k)).sum(-1)
    nib = (nib - ((nib >> 31) << 32)).to(torch.int32).reshape(E, nrt // 2, nq, 256)
    hi = (hi - ((hi >> 15) << 16)).to(torch.int16).reshape(E, nrt // 2, nq, 256).view(torch.int32)
    return torch.cat((w[:, 0], w[:, 1], nib, hi), dim=-1).contiguous()


def _pair_scales(ps):''')
rep('''    if TK2F_SQ:  # layer s68 (moe n14): tk2f single-copy stage images (lossless repack of w13q/w2q + w13s2/w2s2)
        st["w13sq"] = _sq_pack(st["w13q"], st["w13s2"])
        st["w2sq"] = _sq_pack(st["w2q"], st["w2s2"])''', '''    s6 = TK2F_S6 and all(bool(((b >= 80) & (b <= 0x7E)).all()) for b in (  # moe n20: S6 window check
        st["w13s2"].view(torch.uint8), st["w2s2"].view(torch.uint8)))
    if TK2F_SQ and s6:  # moe n20: 6-bit scale codes (lossless; 17920 instead of 18432 B per stage)
        st["w13s6"] = _s6_pack(st["w13q"], st["w13s2"])
        st["w2s6"] = _s6_pack(st["w2q"], st["w2s2"])
    elif TK2F_SQ:  # layer s68 (moe n14): tk2f single-copy stage images (lossless repack of w13q/w2q + w13s2/w2s2)
        st["w13sq"] = _sq_pack(st["w13q"], st["w13s2"])
        st["w2sq"] = _sq_pack(st["w2q"], st["w2s2"])''')
rep('''def _tk2f_w(c, st):
    """(cfg, w13, w2) for moe_tk2f_launch: SA 84204 (+ BAL) -> + 10000000 on the single-copy buffers (SQ)."""
''', '''TK2F_S6 = 1  # moe n20: S6 stage images when every scale byte is in [80, 0x7E] (bitwise = SQ)


def _tk2f_w(c, st):
    """(cfg, w13, w2) for moe_tk2f_launch: SA 84204 (+ BAL) -> + 10000000 on the single-copy buffers (SQ)."""
    if TK2F_SQ and "w13s6" in st and (c % 1000000 == 184204 or (140000 <= c % 1000000 < 150000 and c % 1000 == 4)):
        return c + 20000000, st["w13s6"], st["w2s6"]  # moe n20 S6
''')
open(p, "w").write(s)
print("patched", p)
