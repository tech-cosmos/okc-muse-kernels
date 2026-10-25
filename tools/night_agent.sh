#!/bin/bash
# Runs headless Claude Code sessions for one agent ($1: layer, moe or proj) in ~/r-$1 until
# ~/STOP or ~/STOP-$1 exists. Each session follows CLAUDE.md and AGENT-$1.md, does one cycle.
name=$1
cd ~/r-$name || exit 1
. ~/okc/bin/activate
export OKC_TOKEN="$(cat ~/.okc_token)"
export PATH="$HOME/.local/bin:$HOME/.okc/bin:$PATH"
mkdir -p ~/logs
i=0
while [ ! -f ~/STOP ] && [ ! -f ~/STOP-$name ]; do
  i=$((i + 1))
  start=$(date +%s)
  echo "$(date -u +%FT%TZ) $name session $i starting" >> ~/logs/night-$name.log
  git pull -q --rebase 2>> ~/logs/night-$name.log
  claude -p --dangerously-skip-permissions --output-format stream-json --verbose \
    "You are agent \"$name\". Read CLAUDE.md, then AGENT-$name.md, then ~/COORD.md's Human decisions, then STATE-$name.md, and do the next cycle of work as they describe. It is $(date -u +%FT%TZ) (UTC). This is unattended session $i." \
    > ~/logs/$name-$(date -u +%m%dT%H%M)-$i.jsonl 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) $name session $i exited $? after $(( $(date +%s) - start ))s" >> ~/logs/night-$name.log
  git pull -q --rebase 2>> ~/logs/night-$name.log; git push -q origin HEAD 2>> ~/logs/night-$name.log
  # A session that ends at once is usually auth, usage limits or an outage: back off.
  if [ $(( $(date +%s) - start )) -lt 300 ]; then sleep 900; else sleep 30; fi
done
echo "$(date -u +%FT%TZ) $name stopped" >> ~/logs/night-$name.log
