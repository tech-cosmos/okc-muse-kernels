# s27: port SCB (two block scales per decode, broadcast folded into HMUL2 .H0_H0/.H1_H1) into the layer's moe_src.py.
# python tools/s27m/patch_layer_scb.py <path/to/moe_src.py>   (in place; bitwise-identical outputs)
import sys
p = sys.argv[1]; s = open(p).read()
assert "scp<" not in s, "already patched"
k = s.index("template <int K>\n__device__ __forceinline__ uint32_t sca(uint32_t w) {")
s = s[:k] + '''// s27 SCB (r-moe): two scales per decode: scp<P>(w) = bf16 halves (byte 2P, byte 2P + 1) in the format of sca<>; the
// broadcast of one half (blo / bhi) folds into HMUL2's .H0_H0 / .H1_H1 operand selector (free). Bitwise = sca<>.
template <int P>
__device__ __forceinline__ uint32_t scp(uint32_t w) {
  return __byte_perm(w, 0u, P ? 0x4342u : 0x4140u) * 16u + 0x77007700u;
}
__device__ __forceinline__ uint32_t blo(uint32_t p) { uint32_t r; asm("{.reg .b16 l,h; mov.b32 {l,h}, %1; mov.b32 %0, {l,l};}" : "=r"(r) : "r"(p)); return r; }
__device__ __forceinline__ uint32_t bhi(uint32_t p) { uint32_t r; asm("{.reg .b16 l,h; mov.b32 {l,h}, %1; mov.b32 %0, {h,h};}" : "=r"(r) : "r"(p)); return r; }

''' + s[k:]
reps = [("""        s0[r] = (r & 1) ? sca<2>(sc[r >> 1]) : sca<0>(sc[r >> 1]);
        s1[r] = (r & 1) ? sca<3>(sc[r >> 1]) : sca<1>(sc[r >> 1]);""",
         """        const uint32_t pr = (r & 1) ? scp<1>(sc[r >> 1]) : scp<0>(sc[r >> 1]);
        s0[r] = blo(pr);
        s1[r] = bhi(pr);"""),
        ("s0 = sca<0>(word); s1 = sca<1>(word);", "const uint32_t pr = scp<0>(word); s0 = blo(pr); s1 = bhi(pr);"),
        ("s0[0] = sca<0>(word); s1[0] = sca<1>(word); s0[1] = sca<2>(word); s1[1] = sca<3>(word);",
         "const uint32_t p0 = scp<0>(word), p1 = scp<1>(word);\n      s0[0] = blo(p0); s1[0] = bhi(p0); s0[1] = blo(p1); s1[1] = bhi(p1);"),
        ("sc[0] = sca<0>(sw[h]); sc[1] = sca<1>(sw[h]); sc[2] = sca<2>(sw[h]); sc[3] = sca<3>(sw[h]);",
         "{ const uint32_t p0 = scp<0>(sw[h]), p1 = scp<1>(sw[h]); sc[0] = blo(p0); sc[1] = bhi(p0); sc[2] = blo(p1); sc[3] = bhi(p1); }")]
n = 0
for a, b in reps:
    n += s.count(a); s = s.replace(a, b)
left = [l for l in s.split("\n") if "sca<" in l and "uint32_t sca(" not in l and not l.lstrip().startswith("//")]
print(f"replaced {n} sites; remaining sca<> uses: {len(left)}"); [print("  ", l.strip()) for l in left]
open(p, "w").write(s)
