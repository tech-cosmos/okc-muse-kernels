"""n8s1: verify-sliding attention oracles (variants/orc_*.py from submission.py). A = consumers skip QK/softmax/PV
(wait ready, release empty: KV loads only), C = producer gathers the same 64 rows for every tile (L2-resident, real math),
B = no _combine_hb at v16 (out uninitialized), Z = no attention at all (out = qkv slice copy-free empty)."""
import sys, re
src = open("submission.py").read()
A_OLD = "    if (n > 0) {\n        mb_wait(&sm.ready[0], 0);\n"
A_NEW = ("    if (ORC_A && n > 0) { for (int i = 0; i < n; ++i) { mb_wait(&sm.ready[i % NS], (i / NS) & 1); "
         "if (lane == 0) mb_arrive(&sm.empty[i % NS]); } l[0] = l[1] = 1.f; m[0] = m[1] = 0.f; }\n"
         "    else if (n > 0) {\n        mb_wait(&sm.ready[0], 0);\n")
C_OLD = "const long long off = (long long)cr[j] * 128;"
C_NEW = "const long long off = ORC_C ? (long long)(r + 1000) * 128 : (long long)cr[j] * 128;"
B_OLD = "    _combine_hb[(batch * length, hq // 2)](\n        qkv, pacc, pml, data[\"v_scale\"], out,"
B_NEW = "    if not ORC_B: _combine_hb[(batch * length, hq // 2)](\n        qkv, pacc, pml, data[\"v_scale\"], out,"
Z_OLD = "def _verify_nvs(data, cfg):\n"
Z_NEW = ("def _verify_nvs(data, cfg):\n    if ORC_Z:\n        return torch.empty((cfg['batch'], cfg['q_len'], cfg['num_q_heads'] * cfg['head_dim']), "
         "dtype=torch.bfloat16, device=data['qkv'].device)\n")
def mk(a, b, c, z):
    s = src
    i = s.index(A_OLD); j = s.index(A_OLD, i + 1)  # nvs + nvscl consumers (the 3rd is nds: decode)
    s = s[:j] + A_NEW + s[j + len(A_OLD):]; s = s[:i] + A_NEW + s[i + len(A_OLD):]
    assert s.count(C_OLD) == 2; s = s.replace(C_OLD, C_NEW)
    assert s.count(B_OLD) == 1; s = s.replace(B_OLD, B_NEW)
    assert s.count(Z_OLD) == 1; s = s.replace(Z_OLD, Z_NEW)
    for k in ("_NVS_SRC = r'''", "_NVSCL_SRC = r'''"):
        assert s.count(k) == 1; s = s.replace(k, k + f"#define ORC_A {a}\n#define ORC_C {c}\n")
    s = s.replace("\ndef _verify_nvs", f"\nORC_B = {b}\nORC_Z = {z}\ndef _verify_nvs", 1)
    return s
for name, f in {"base": (0, 0, 0, 0), "A": (1, 0, 0, 0), "B": (0, 1, 0, 0), "AB": (1, 1, 0, 0), "C": (0, 0, 1, 0),
                "Z": (0, 0, 0, 1)}.items():
    open(f"variants/orc_{name}.py", "w").write(mk(*f))
print("ok")
