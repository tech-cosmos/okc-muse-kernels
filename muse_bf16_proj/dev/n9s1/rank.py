# n9s1: per-GEMM critical-path time in the layer (H200 timeline r-layer dev/n7s1_tl_h200.txt, v164; end - predecessor end)
# vs ideal = max(bytes/4.3 TB/s, FLOP/750 TF/s); weighted by the lead's %/us per load (sl, full).
MB = dict(qkv=28.3, o=12.6, pool=50.3, unp=50.3, unp8=6.3, router=8.4)        # 12-bit dense + FP32 router
MB16 = dict(qkv=37.7, o=16.8, FR=67.1 + 16.8, unp=67.1)                        # BF16 (lt at T>=128), FR = pooler + router hi/lo
NK = dict(qkv=2304 * 8192, o=8192 * 1024, pool=4096 * 8192, unp=8192 * 4096, router=256 * 8192)
W = {'d1': (.161, .056), 'd16': (.072, .025), 'd32': (.056, .020), 'v1': (.096, .033), 'v16': (.041, .014), 'v32': (.031, .011)}
T = dict(d1=1, d16=16, d32=32, v1=8, v16=128, v32=256)
# (load, gemm, kernel, c_sl, c_full)
rows = [
 ('d1', 'qkv', 'g13', 11.8, 10.3), ('d1', 'o', 'g13o', 3.5, 3.6), ('d1', 'pool+rt', 'pr1', 16.1, 14.8), ('d1', 'unp8+norm2', 'n2u', 4.3, 4.7),
 ('d16', 'qkv', 'g24', 11.8, 11.5), ('d16', 'o', 'g27|g30', 4.6, 6.9), ('d16', 'pool+rt', 'g24|rsplit', 15.6, 18.3), ('d16', 'unp', 'g30|g24', 14.2, 14.0),
 ('d32', 'qkv', 'g30', 13.5, 14.0), ('d32', 'o', 'g30', 7.1, 6.7), ('d32', 'pool+rt', 'g30k|rsplit', 19.7, 19.2), ('d32', 'unp', 'g30', 14.7, 14.9),
 ('v1', 'qkv', 'g14', 11.4, 11.0), ('v1', 'o', 'g30', 5.9, 7.6), ('v1', 'pool+rt', 'pr8', 18.4, 17.5), ('v1', 'unp8+norm', 'nu8', 5.3, 5.1),
 ('v16', 'qkv', 'nvjet', 15.7, 16.4), ('v16', 'o', 'nvjet', 7.1, 8.3), ('v16', 'FR', 'nvjet', 28.1, 26.7), ('v16', 'unp', 'g30', 20.3, 19.5),
 ('v32', 'qkv', 'nvjet', 19.0, 18.7), ('v32', 'o', 'nvjet', 9.9, 9.7), ('v32', 'FR', 'nvjet', 33.9, 33.6), ('v32', 'unp', 'nvjet', 27.3, 28.9),
]
def ideal(load, g):
    t = T[load]; big = t >= 128
    if g.startswith('pool') or g == 'FR':
        by = (MB16['FR'] if big else MB['pool'] + MB['router']); fl = 2 * t * (NK['pool'] + 2 * NK['router'])
    elif g.startswith('unp8'):
        by = MB['unp8']; fl = 0
    else:
        by = (MB16 if big else MB)[g]; fl = 2 * t * NK[g]
    return max(by / 4.3, fl / 750e6)
out = []
for load, g, k, cs, cf in rows:
    i = ideal(load, g); ws, wf = W[load]
    pct = ws * (cs - i) + wf * (cf - i)
    out.append((pct, load, g, k, cs, cf, i))
out.sort(reverse=True)
print('load gemm kernel        c_sl  c_full ideal  exc_sl  %score')
for pct, load, g, k, cs, cf, i in out:
    print(f'{load:4s} {g:11s} {k:11s} {cs:5.1f} {cf:6.1f} {i:5.1f} {cs - i:6.1f} {pct:6.2f}')
print('total %', round(sum(o[0] for o in out), 2))
