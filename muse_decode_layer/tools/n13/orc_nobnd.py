"""n13s3 oracle: write a copy of submission.py whose first boundary is skipped for tokens in ORC_NOBND (timing only:
pr1/pr8 read a stale h, routing changes, so only the o-end -> pr1/pr8-end interval is valid).
usage: python tools/n13/orc_nobnd.py submission.py /tmp/sub_nobnd.py; STMT="ORC_NOBND=(1,8)" python tools/n13/tlmed.py ..."""
import sys
s = open(sys.argv[1]).read()
assert s.count("    if not bnd:\n") == 1
s = s.replace("    if not bnd:\n", "    if not bnd and not (tokens in ORC_NOBND):\n")
s = s.replace("\nPR1_BND = ", "\nORC_NOBND = ()\nPR1_BND = ", 1)
open(sys.argv[2], "w").write(s)
