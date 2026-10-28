# n5s29: variants/sub_pr.py = build.py + TK2PR on + TUNE {} + _ab (as variants/n5s20_notune.py)
set -e
python build.py variants/sub_pr.py
sed -i 's/^TK2PR = False$/TK2PR = True/' variants/sub_pr.py
sed -n '/^TUNE = {}  # n5s20/,$p' variants/n5s20_notune.py | sed 's/n5s20/n5s29/' >> variants/sub_pr.py
grep -c '^TK2PR = True' variants/sub_pr.py
