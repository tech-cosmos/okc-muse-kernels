#!/bin/sh
# fresh x42: OFF / ON files from this branch's submission.py (NOCAP build), tuner pinned off in both.
# ON = SLACK True, SLACK_TOKENS (128, 256) (moe/slack kernel, v16/v32), SLACK_T1 True (layer/slack1 PR1_SLK, d1); margins unchanged.
set -e
D=${D:-/tmp/x42}; mkdir -p $D
sed 's/^TUNE = {1: /TUNE = {} and {1: /' submission.py > $D/off.py
sed -e 's/^SLACK = False.*/SLACK = True/' -e 's/^SLACK_TOKENS = ()$/SLACK_TOKENS = (128, 256)/' $D/off.py > $D/on.py
grep -c '^TUNE = {} and' $D/off.py $D/on.py
grep -n '^SLACK = \|^SLACK_TOKENS = \|^SLACK_T1 = \|^SLACK_MR\|^SLACK1_MR' $D/off.py $D/on.py
