import pathlib
p=pathlib.Path('pr8_src.cu'); s=p.read_text()
a='''    if (t == 0) ls_ = __ldg(a.ls);
    if (t == 0) {
        int v;'''
assert s.count(a)==1
s=s.replace(a,'''    if (t == 0) ls_ = __ldg(a.ls);
#ifdef PR8_DP  // n15 DP: no counter; poll the partials themselves (sentinel 0xFFFFFFFF = a NaN no FMA produces), re-arm after
    __syncthreads();
#ifdef PR8_TRACE
    if (t == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v_)); ((unsigned long long*)(a.part + 8 * MT * E))[a.bk * 8 + 2] = v_; }
#endif
    for (int i = t; i < MT * E; i += blockDim.x) {
        unsigned v[RSL];
        bool ok;
        do {
#pragma unroll
            for (int q = 0; q < RSL; ++q) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v[q]) : "l"(a.part + (long)q * MT * E + i) : "memory");
            ok = true;
#pragma unroll
            for (int q = 0; q < RSL; ++q) ok = ok && v[q] != 0xFFFFFFFFu;
            if (!ok) __nanosleep(128);
        } while (!ok);
        float s = 0.f;
#pragma unroll
        for (int q = 0; q < RSL; ++q) s += __uint_as_float(v[q]);
        lgs[i] = s;
        a.logits[i] = s;
#pragma unroll
        for (int q = 0; q < RSL; ++q) asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(a.part + (long)q * MT * E + i), "r"(0xFFFFFFFFu) : "memory");
    }
    __syncthreads();
#else
    if (t == 0) {
        int v;''')
a='''        a.logits[i] = s;
    }
    __syncthreads();
#ifdef PR8_TRACE
    if (t == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v_)); ((unsigned long long*)(a.part + 8 * MT * E))[a.bk * 8 + 3]'''
assert s.count(a)==1
s=s.replace(a,'''        a.logits[i] = s;
    }
    __syncthreads();
#endif
#ifdef PR8_TRACE
    if (t == 0) { unsigned long long v_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v_)); ((unsigned long long*)(a.part + 8 * MT * E))[a.bk * 8 + 3]''')
a='''        __stcg(a.part + ((long)sl * MT + tok) * E + ri * RROWS + r, rred[r][0][tok] + rred[r][1][tok]);
    }
    TR(2)
    if (has_r && a.npf != -2) {'''
assert s.count(a)==1
s=s.replace(a,'''#ifdef PR8_DP
        asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" :: "l"(a.part + ((long)sl * MT + tok) * E + ri * RROWS + r), "f"(rred[r][0][tok] + rred[r][1][tok]) : "memory");
#else
        __stcg(a.part + ((long)sl * MT + tok) * E + ri * RROWS + r, rred[r][0][tok] + rred[r][1][tok]);
#endif
    }
    TR(2)
#ifdef PR8_DP
    if (false) {
#else
    if (has_r && a.npf != -2) {
#endif''')
p.write_text(s)
p=pathlib.Path('layer.py'); s=p.read_text()
a='''            part = state.setdefault("pr8_part", torch.zeros(8 * tokens * N_EXPERTS + 129 * 16, dtype=torch.float32, device=dev))  # n15 trace\n'''
assert s.count(a)==1
s=s.replace(a,'''            part = state["pr8_part"]  # n15 DP: persistent, armed with the 0xFFFFFFFF sentinel (+ trace room)\n''')
a='''    state["pr1_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)  # pr1's router ticket (self-resetting)\n'''
assert s.count(a)==1
s=s.replace(a,a+'''    state["pr8_part"] = torch.full((8 * 8 * N_EXPERTS + 129 * 16,), -1, dtype=torch.int32, device=dev).view(torch.float32)\n''')
p.write_text(s)
print("patched")
