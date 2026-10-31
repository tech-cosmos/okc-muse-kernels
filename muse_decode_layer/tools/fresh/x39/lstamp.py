"""fresh x39: WHOLE-LAYER per-warp %globaltimer stamps (timing copy only; lead 22:55Z 'hop stamps at d16 sl / d32 sl').
Every CUDA kernel compiled through torch load_inline gets (hooked at compile time, the submission text is not changed):
  entry  - lane 0 of each warp, first statement of the __global__ body (RAII guard; its destructor = exit stamp)
  wait   - right after each griddepcontrol.wait (record found via a (smid, %warpid) slot map; first wait only)
  exit   - guard destructor (every return path)
Triton @jit kernels launched in the file get the same per program (entry / after gdc_wait() / end).
Records: u64[4] = meta (tag << 48 | cta << 16 | warp), entry, wait, exit. Instances of one kernel are split by entry gaps.
python tools/fresh/x39/lstamp.py <submission.py> '<load layer>' [reps]"""
import sys, os, re, fcntl, statistics, torch
sys.path.insert(0, ".")
import bench

src, flt = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
NREC = 1 << 18
buf = torch.zeros(NREC * 8 + 64 + 8, dtype=torch.int64, device="cuda")
slot = torch.zeros(1024 * 128, dtype=torch.int32, device="cuda")
BUF, CNT, SLOT = buf.data_ptr() + 64 * 8, buf.data_ptr(), slot.data_ptr()
TAGS = []

HDR = r'''
#ifndef STP_HDR
#define STP_HDR
__device__ __forceinline__ unsigned long long stp_gt() { unsigned long long t; asm volatile("mov.u64 %%0, %%%%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned stp_slot() { unsigned s, w; asm volatile("mov.u32 %%0, %%%%smid;" : "=r"(s)); asm volatile("mov.u32 %%0, %%%%warpid;" : "=r"(w)); return (s & 1023) * 128 + (w & 127); }
struct StpG {
  unsigned long long* r;
  __device__ __forceinline__ StpG(unsigned tag) {
    r = nullptr;
    const unsigned tl = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    if ((tl & 31) == 0) {
      unsigned long long t = stp_gt();
      unsigned i = atomicAdd(reinterpret_cast<unsigned*>(%(CNT)dULL), 1u);
      if (i < %(NREC)d) {
        r = reinterpret_cast<unsigned long long*>(%(BUF)dULL) + 8ull * i;
        const unsigned long long cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        r[0] = ((unsigned long long)tag << 48) | ((cta & 0xffffffffull) << 16) | (tl >> 5);
        r[1] = t;
        reinterpret_cast<unsigned*>(%(SLOT)dULL)[stp_slot()] = i + 1;
      }
    }
  }
#ifndef STP_NODTOR
  __device__ __forceinline__ ~StpG() { if (r) r[3] = stp_gt(); }
#endif
};
__device__ __forceinline__ void stp_pre() {
  const unsigned tl = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  if ((tl & 31) == 0) {
    unsigned i = reinterpret_cast<volatile unsigned*>(%(SLOT)dULL)[stp_slot()];
    if (i) { unsigned long long* r = reinterpret_cast<unsigned long long*>(%(BUF)dULL) + 8ull * (i - 1); if (r[4] == 0) r[4] = stp_gt(); }
  }
}
__device__ __forceinline__ void stp_wait() {
  const unsigned tl = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  if ((tl & 31) == 0) {
    unsigned i = reinterpret_cast<volatile unsigned*>(%(SLOT)dULL)[stp_slot()];
    if (i) { unsigned long long* r = reinterpret_cast<unsigned long long*>(%(BUF)dULL) + 8ull * (i - 1); if (r[2] == 0) r[2] = stp_gt(); }
  }
}
#endif
''' % dict(CNT=CNT, NREC=NREC, BUF=BUF, SLOT=SLOT)
HDR = HDR.replace("%%", "%")
WAIT_RE = re.compile(r'asm\s+volatile\s*\(\s*"griddepcontrol\.wait;(?:\\n)?"\s*:::\s*"memory"\s*\)\s*;')


def patch_cuda(code, modname):
    if "__global__" not in code:
        return code
    out, pos = [], 0
    for m in re.finditer(r"__global__", code):
        a = m.start()
        if a < pos:
            continue
        b = code.find("{", a)
        semi = code.find(";", a)
        hdr = code[a:b]
        if b < 0 or (0 <= semi < b) or "(" not in hdr:
            continue
        line_start = code.rfind("\n", 0, a)
        if "//" in code[line_start:a]:
            continue
        nm = re.findall(r"(\w+)\s*\(", hdr)
        nm = [n for n in nm if n not in ("__launch_bounds__", "__cluster_dims__", "__maxnreg__")]
        tag = len(TAGS)
        TAGS.append(f"{modname}:{nm[-1] if nm else '?'}")
        out.append(code[pos:b + 1] + ("" if os.environ.get("STP_NOGUARD") else f" StpG _stp_g({tag}u);"))
        pos = b + 1
    out.append(code[pos:])
    code = "".join(out)
    if not os.environ.get("STP_NOWAIT"): code = WAIT_RE.sub(lambda m: "{ stp_pre(); " + m.group(0) + " stp_wait(); }", code)
    return ("#define STP_NODTOR\n" if os.environ.get("STP_NODTOR") else "") + HDR + code


import torch.utils.cpp_extension as cx
_orig = cx.load_inline


def hooked(*a, **kw):
    cs = kw.get("cuda_sources")
    nm = kw.get("name", "?")
    if cs is not None and not any(x and x in nm for x in os.environ.get("STP_SKIP", "").split(",")) and os.environ.get("STP_ONLY", "") in nm:
        kw["cuda_sources"] = [patch_cuda(c, nm) for c in cs] if isinstance(cs, (list, tuple)) else patch_cuda(cs, nm)
        kw["name"] = kw["name"] + "_stp39_%x" % (CNT & 0xffffffffff)
    return _orig(*a, **kw)


cx.load_inline = hooked

# --- Triton: patch launched @triton.jit kernels in the text ---
s = open(src).read()
launched = set() if os.environ.get("STP_ONLY") else set(re.findall(r"\b(_\w+)\[", s))
TPRE = ("    _stp_t0 = tl.inline_asm_elementwise(\"mov.u64 $0, %globaltimer;\", \"=l\", [], dtype=tl.int64, is_pure=False, pack=1)\n"
        "    _stp_i = tl.atomic_add(tl.full([], {CNT}, tl.int64).to(tl.pointer_type(tl.int32)), 1)\n"
        "    _stp_r = tl.full([], {BUF}, tl.int64).to(tl.pointer_type(tl.int64)) + 8 * tl.minimum(_stp_i, {NREC}).to(tl.int64)\n"
        "    _stp_cta = tl.program_id(0) + tl.num_programs(0) * (tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))\n"
        "    tl.store(_stp_r, ({TAG} << 48) | (_stp_cta.to(tl.int64) << 16))\n"
        "    tl.store(_stp_r + 1, _stp_t0)\n")
TWAIT = "{ind}tl.store(_stp_r + 2, tl.inline_asm_elementwise(\"mov.u64 $0, %globaltimer;\", \"=l\", [], dtype=tl.int64, is_pure=False, pack=1))\n"
TEND = "    tl.store(_stp_r + 3, tl.inline_asm_elementwise(\"mov.u64 $0, %globaltimer;\", \"=l\", [], dtype=tl.int64, is_pure=False, pack=1))\n"
lines = s.split("\n")
res, i, ntr = [], 0, 0
while i < len(lines):
    L = lines[i]
    res.append(L)
    if L.startswith("@triton.jit") and i + 1 < len(lines):
        m = re.match(r"def (\w+)\(", lines[i + 1])
        if m and m.group(1) in launched:
            name = m.group(1)
            j = i + 1
            while not lines[j].rstrip().endswith(":"):
                res.append(lines[j]); j += 1
            res.append(lines[j]); j += 1
            if lines[j].strip().startswith('"""'):
                if lines[j].strip().count('"""') >= 2 and len(lines[j].strip()) > 3:
                    res.append(lines[j]); j += 1
                else:
                    res.append(lines[j]); j += 1
                    while '"""' not in lines[j]:
                        res.append(lines[j]); j += 1
                    res.append(lines[j]); j += 1
            body = []
            while j < len(lines) and (lines[j].startswith(" ") or lines[j].strip() == "" ):
                body.append(lines[j]); j += 1
            while body and body[-1].strip() == "":
                j -= 1; body.pop()
            if True:
                tag = len(TAGS); TAGS.append("triton:" + name); ntr += 1
                res.append(TPRE.format(CNT=CNT, BUF=BUF, TAG=tag, NREC=NREC).rstrip("\n"))
                for b in body:
                    if re.match(r"\s+return\s*$", b):
                        res.append(b[:len(b) - len(b.lstrip())] + TEND.strip())
                    res.append(b)
                    if b.strip() == "gdc_wait()":
                        res.insert(len(res) - 1, TWAIT.replace("_stp_r + 2", "_stp_r + 4").format(ind=b[:len(b) - len(b.lstrip())]).rstrip("\n"))
                        res.append(TWAIT.format(ind=b[:len(b) - len(b.lstrip())]).rstrip("\n"))
                res.append(TEND.rstrip("\n"))
            i = j
            continue
    i += 1
s = "\n".join(res)
out = f"/tmp/x39_stamped_{os.getpid()}.py"
open(out, "w").write(s)
print(f"triton kernels stamped: {ntr}", flush=True)
sub = bench.load_sub(out)
w, st = bench.weights_state(sub)
print("CNT", CNT, "count after prepare", int(buf[0].item() & 0xffffffff), flush=True)
(name, c), = [(f"{l} {c['layer']}", c) for l, c in bench.SCORED if f"{l} {c['layer']}" == flt]
d = bench.make_input(c, w, 99)
g = bench.capture(lambda: sub.kernel_fn(d, st))
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
rows, tot = [], []
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(reps):
        flush.zero_(); buf.zero_(); slot.zero_(); torch.cuda.synchronize()
        ev0.record(); g.replay(); ev1.record(); torch.cuda.synchronize()
        tot.append(ev0.elapsed_time(ev1) * 1000)
        n = int(buf[0].item() & 0xffffffff)
        rows.append(buf[64:64 + 8 * min(n, NREC)].view(-1, 8).cpu().clone())
print(name, f"layer replay median {statistics.median(tot):.1f} us (stamped copy), records {rows[-1].shape[0]}")
torch.save(dict(tags=TAGS, rows=rows, tot=tot), f"/tmp/x39_{flt.replace(' ', '_')}.pt")
os.remove(out)
