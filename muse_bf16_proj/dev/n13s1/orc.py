"""proj n13s1: d1 pooler+router oracles. Writes variants of the layer submission.py with pr1 changed:
  nomath: pooler dot16 and router FMAs replaced by an xor of the loaded words (same loads, same structure, routing kept)
  flat:   every pr1 block grid-stride reads the same bytes (12-bit S + N planes + FP32 router) with 4 x 16 B in flight per thread,
          griddepcontrol.wait after the first batch (as the real pr1 waits after its first stage); no outputs written
  nopool: pooler role exits at once (router + routing only)   norout: router blocks only ticket+route (no W)
python orc.py <layer submission.py> <outdir>"""
import sys, pathlib
full = pathlib.Path(sys.argv[1]).read_text(); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
i0 = full.index("_GPR1_SRC = r'''"); i1 = full.index("'''", i0 + 20); src = full[i0:i1]
a = "__device__ __forceinline__ float dot16(const uint4& s, const uint2& nb, const float* xp) {\n"
assert src.count(a) == 1
nm_dot = a + "#ifdef PR1_NOMATH\n    return __int_as_float((s.x ^ s.y ^ s.z ^ s.w ^ nb.x ^ nb.y) & 0x3c00ffffu);\n#endif\n"
b = "                s0 = fmaf(w[r][j].x, xr[j].x, s0); s1 = fmaf(w[r][j].y, xr[j].y, s1);\n                s0 = fmaf(w[r][j].z, xr[j].z, s0); s1 = fmaf(w[r][j].w, xr[j].w, s1);\n"
assert src.count(b) == 1
nm_rt = "#ifdef PR1_NOMATH\n                s0 = __int_as_float(__float_as_int(s0) ^ ((__float_as_int(w[r][j].x) ^ __float_as_int(w[r][j].y) ^ __float_as_int(w[r][j].z) ^ __float_as_int(w[r][j].w)) & 0x3c00ffff));\n#else\n" + b + "#endif\n"
c = "    if ((int)blockIdx.x < RB) router<R, 256 * NV>(a, blockIdx.x, RB);\n    else pooler<U, NV, 8192, PF>(a, blockIdx.x - RB, gridDim.x - RB);\n"
assert src.count(c) == 1
flat = """#ifdef PR1_FLAT
    {
        const long ns = (long)a.n * 512, nn = (long)a.n * 512 / 2, nr = 256L * 8192 / 4;  // uint4 counts: S, N, router
        const long tot = ns + nn + nr, stride = (long)gridDim.x * blockDim.x;
        unsigned acc = 0;
        long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
        auto at = [&](long q) -> const uint4* { return q < ns ? reinterpret_cast<const uint4*>(a.S) + q : q < ns + nn ? reinterpret_cast<const uint4*>(a.N) + (q - ns) : reinterpret_cast<const uint4*>(a.rw) + (q - ns - nn); };
        bool waited = false;
        for (; i < tot; i += 4 * stride) {
            uint4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) { const long q = i + u * stride; v[u] = q < tot ? ld4(at(q)) : make_uint4(0, 0, 0, 0); }
            if (!waited) { asm volatile("griddepcontrol.wait;" ::: "memory"); waited = true; }
#pragma unroll
            for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
        }
        if (!waited) asm volatile("griddepcontrol.wait;" ::: "memory");
        if (acc == 0x9e3779b9u) a.logits[0] = 1.f;
        return;
    }
#endif
#ifdef PR1_NOPOOL
    if ((int)blockIdx.x >= RB) { asm volatile("griddepcontrol.wait;" ::: "memory"); return; }
#endif
"""
nr = "    if (g < NG) load(g);\n    pf(g + RB);\n"
assert src.count(nr) == 1
nr2 = "#ifdef PR1_NOROUT\n    g = NG;\n#endif\n" + nr
base = src.replace(a, nm_dot).replace(b, nm_rt).replace(c, flat + c).replace(nr, nr2)
mark = "namespace pr1 {\n"
assert base.count(mark) == 1
# rt: one build, FLAT chosen at launch time by env PR1_FLAT (npf bit 8), for a same-process A/B (H200 Modal)
rt = base.replace("#ifdef PR1_FLAT\n    {\n", "    if (a.npf & 256) {\n", 1).replace("        return;\n    }\n#endif\n#ifdef PR1_NOPOOL", "        return;\n    }\n#ifdef PR1_NOPOOL", 1)
rt = rt.replace("for (int p = PF; p < PF + a.npf; ++p)", "for (int p = PF; p < PF + (a.npf & 255); ++p)", 1)
rt = rt.replace("    a.rel = rel;\n", "    a.rel = rel;\n    if (getenv(\"PR1_FLAT\")) a.npf |= 256;\n", 1)
assert rt.count("a.npf & 256") == 1 and rt.count("a.npf & 255") == 1 and rt.count("getenv") == 1
(out / "sub_rt.py").write_text(full[:i0] + rt + full[i1:])
for v, d in [("base", ""), ("nomath", "PR1_NOMATH"), ("flat", "PR1_FLAT"), ("nopool", "PR1_NOPOOL"), ("norout", "PR1_NOROUT"),
             ("nopnr", "PR1_NOPOOL\n#define PR1_NOROUT")]:
    (out / f"sub_{v}.py").write_text(full[:i0] + base.replace(mark, (f"#define {d}\n" if d else "") + mark) + full[i1:])
print("ok")
