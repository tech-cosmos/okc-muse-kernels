"""python dev/s72/readguest.py <judge log> [grep]: the guest's stdout from a carrier check (guestz chunks or raw)."""
import base64, re, sys, zlib
lines = open(sys.argv[1]).read().splitlines()
ch = {int(m.group(1)): m.group(2) for m in (re.match(r"guestz (\d+) (\S+)$", l) for l in lines) if m}
out = zlib.decompress(base64.b64decode("".join(ch[i] for i in sorted(ch)))).decode() if ch else "\n".join(lines)
if ch and sorted(ch) != list(range(len(ch))):
    print("WARNING: guestz chunks missing (log tail cut)", file=sys.stderr)
for l in out.splitlines():
    if len(sys.argv) < 3 or sys.argv[2] in l:
        print(l)
