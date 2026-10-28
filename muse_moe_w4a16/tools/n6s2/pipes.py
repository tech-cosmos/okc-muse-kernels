# n6s2: per-pipe issue throughput on GH100 (warp-instr / clk / SMSP), 16 warps/SM, 8 independent chains per thread.
import torch, os, sys, fcntl, subprocess, glob, re
from torch.utils.cpp_extension import load_inline
OPS = {
  "LOP3": 'asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a[i]) : "r"(b), "r"(a[(i + 1) & 7]));',
  "SHF": 'asm volatile("shf.l.wrap.b32 %0, %0, %0, 10;" : "+r"(a[i]));',
  "IMAD": 'asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b));',
  "IADD": 'asm volatile("add.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a[(i + 1) & 7]));',
  "HMUL2": 'asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(a[i]) : "r"(b));',
  "PRMT": 'asm volatile("prmt.b32 %0, %0, %1, 0x5140;" : "+r"(a[i]) : "r"(b));',
  "LOP3+SHF": 'if (i & 1) { asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a[i]) : "r"(b), "r"(a[(i + 1) & 7])); } else { asm volatile("shf.l.wrap.b32 %0, %0, %0, 10;" : "+r"(a[i])); }',
  "LOP3+IMAD": 'if (i & 1) { asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a[i]) : "r"(b), "r"(a[(i + 1) & 7])); } else { asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b)); }',
  "LOP3+HMUL2": 'if (i & 1) { asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a[i]) : "r"(b), "r"(a[(i + 1) & 7])); } else { asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(a[i]) : "r"(b)); }',
  "SHF+IMAD": 'if (i & 1) { asm volatile("shf.l.wrap.b32 %0, %0, %0, 10;" : "+r"(a[i])); } else { asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b)); }',
  "SHF+HMUL2": 'if (i & 1) { asm volatile("shf.l.wrap.b32 %0, %0, %0, 10;" : "+r"(a[i])); } else { asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(a[i]) : "r"(b)); }',
  "IMAD+HMUL2": 'if (i & 1) { asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b)); } else { asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(a[i]) : "r"(b)); }',
  "PRMT+IMAD": 'if (i & 1) { asm volatile("prmt.b32 %0, %0, %1, 0x5140;" : "+r"(a[i]) : "r"(b)); } else { asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b)); }',
  "IADD+IMAD": 'if (i & 1) { asm volatile("add.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a[(i + 1) & 7])); } else { asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b)); }',
  "DQ7:1:4": 'if ((i & 7) < 4) { asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a[i]) : "r"(b), "r"(a[(i + 1) & 7])); } else if ((i & 7) < 6) { asm volatile("shf.l.wrap.b32 %0, %0, %0, 10;" : "+r"(a[i])); } else if ((i & 7) == 6) { asm volatile("mad.lo.u32 %0, %0, 16648, %1;" : "+r"(a[i]) : "r"(b)); } else { asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(a[i]) : "r"(b)); }',
}
src = "#include <cstdint>\n"
for k, (nm, op) in enumerate(OPS.items()):
    src += f"""__global__ void __launch_bounds__(512) k{k}(uint32_t* o, int iters, uint32_t b) {{
  uint32_t a[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = threadIdx.x * 977u + i * 131u;
  for (int it = 0; it < iters; ++it) {{
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) {op}
  }}
  uint32_t s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) s ^= a[i];
  o[blockIdx.x * 512 + threadIdx.x] = s;
}}
"""
src += "void run(int64_t k, torch::Tensor o, int64_t iters, int64_t b) {\n"
for k in range(len(OPS)):
    src += f"  if (k == {k}) k{k}<<<132, 512>>>((uint32_t*)o.data_ptr(), (int)iters, (uint32_t)b);\n"
src += "}\n"
NAME = "n6s2_pipes2"
lib = load_inline(NAME, cpp_sources="void run(int64_t, torch::Tensor, int64_t, int64_t);", cuda_sources="#include <torch/extension.h>\n" + src,
                  functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
so = glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/*/{NAME}/{NAME}.so"))[0]
sass = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", so], capture_output=True, text=True).stdout
mix = {}
for blk in sass.split("Function : ")[1:]:
    nm = blk.split()[0]; ops = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z0-9_]+)", blk)
    c = {}
    for x in ops: c[x] = c.get(x, 0) + 1
    mix[nm] = sorted(c.items(), key=lambda t: -t[1])[:4]
o = torch.zeros(132 * 512, dtype=torch.int32, device="cuda")
IT = 4000
for k, nm in enumerate(OPS):
    lib.run(k, o, 10, 3); torch.cuda.synchronize()
    with open('/tmp/okc-gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); lib.run(k, o, IT, 3); e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)
    clk = 1.98e9 if len(sys.argv) < 2 else float(sys.argv[1])  # H100 SXM boost
    winstr_per_smsp = IT * 64 * 4  # 16 warps/SM -> 4 per SMSP, 64 ops per it
    print(f"{nm:11s} {ms * 1e-3 * clk / winstr_per_smsp:5.2f} clk per warp-instr per SMSP   sass top: {mix[f'_Z{len(str(k)) + 1}k{k}Pjij']}", flush=True)
