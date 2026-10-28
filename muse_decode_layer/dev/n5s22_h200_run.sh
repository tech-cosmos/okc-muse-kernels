export SEEDS=99,7,13
python tools/ab_flag.py submission.py "verify 16" 2 "_ab(128)" "_ab(128,drop=('g30:unpool',))" "_ab(128,fr=False)" "_ab(128,fr=False,drop=('g30:unpool',))" "_ab(128)"
