# s54 oracle: d1 sliding qkv split: q/k/v (1280 rows) as the first op, gate (1024 rows) as a g13 PDL secondary after
# csl (W prologue before its wait), o (g13o) its PDL secondary. Gate output goes to a dummy: OUTPUT IS WRONG (timing only).
import sys
s = open("submission.py").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep("TUNE_REPS = 40", "TUNE = {}\nTUNE_REPS = 40")
if sys.argv[1] == "gs":
    rep('    state["g13"] = {key: pk',
        '    state["g13q"] = _pack13(w["qkv_gate"][:1280].contiguous()); state["g13g"] = _pack13(w["qkv_gate"][1280:].contiguous())\n    state["g13"] = {key: pk')
    rep("    if qkv is None:\n        qkv = _linear(data[\"attn_input\"], dense[\"qkv_gate\"], \"qkv_gate\", state, kv_pf=kv_pf)",
        "    gs = tokens == 1 and kv_pf is not None\n"
        "    if qkv is None and gs:\n"
        "        qkv = torch.empty((1, 2304), dtype=torch.bfloat16, device=dev)\n"
        "        pk = state['g13q']\n"
        "        _G10['lib'].g13_gemv(pk[0], pk[1], pk[2], data['attn_input'].contiguous(), qkv[:, :1280], pk[3], *_cfg(state, 1, 'g13', 'qkv_gate', G13_CFG), *kv_pf, 0)\n"
        "    if qkv is None:\n        qkv = _linear(data[\"attn_input\"], dense[\"qkv_gate\"], \"qkv_gate\", state, kv_pf=kv_pf)")
    rep('    a = _g19(o, "o", state)',
        "    if gs:\n"
        "        gd = torch.empty((1, 1024), dtype=torch.bfloat16, device=dev); pk = state['g13g']\n"
        "        _G10['lib'].g13_gemv(pk[0], pk[1], pk[2], data['attn_input'].contiguous(), gd, pk[3], *GS_CFG, data['attn_input'], data['attn_input'], data['attn_input'], 0, 0, GS_PDL)\n"
        '    a = _g19(o, "o", state)')
    rep("TUNE = {}\n", "TUNE = {}\nimport os\nGS_CFG = tuple(int(v) for v in os.environ.get('GS_CFG', '2,4,1,1,132').split(','))\n")
    rep('        j += PF;\n    }\n}\n}  // namespace g13', '        j += PF;\n    }\n    if (pdl == 2) asm volatile("griddepcontrol.wait;" ::: "memory");\n}\n}  // namespace g13')
    rep('"        if (i < KK / 8 && !pdl) xv[q]', '"        if (i < KK / 8 && pdl != 1) xv[q]')
    rep('"    if (pdl) {  // s33', '"    if (pdl == 1) {  // s33')
    rep("GS_CFG = ", "GS_PDL = int(os.environ.get('GS_PDL', '2'))\nGS_CFG = ")
    rep('G13_ALT = {"qkv_gate": [(2, 4, 1, 2, 132)]}', 'G13_ALT = {"qkv_gate": [(2, 4, 1, 2, 132), (2, 2, 2, 1, 264), (2, 1, 4, 1, 528), (2, 2, 1, 1, 264)]}')
open(f"variants/{sys.argv[1]}.py", "w").write(s)
