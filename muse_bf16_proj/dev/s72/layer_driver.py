# Guest driver (exec'd by proj's _diag_guest): the layer's submission on its judge weights (reference.generate_weights
# (WEIGHTS_SEED)); its prepare() runs the H200 tuner ('layer tune T=...' lines) and any DIAG switched on in the file.
import time
ref = mods["reference"]
w = ref.generate_weights(ref.WEIGHTS_SEED)
torch.cuda.synchronize()
t0 = time.time()
st = sub["prepare"](w)
torch.cuda.synchronize()
print(f"guest layer prepare {time.time() - t0:.1f} s", flush=True)
del st, w
