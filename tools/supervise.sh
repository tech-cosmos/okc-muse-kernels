#!/bin/bash
# Keeps each agent's loop running until ~/STOP exists; restarts a loop that died.
mkdir -p ~/logs
while [ ! -f ~/STOP ]; do
  for name in layer moe proj; do
    if [ ! -f ~/STOP-$name ] && ! pgrep -f "night_agent.sh $name" > /dev/null; then
      echo "$(date -u +%FT%TZ) starting $name loop" >> ~/logs/supervise.log
      setsid nohup ~/r-$name/tools/night_agent.sh $name >> ~/logs/night-$name.out 2>&1 < /dev/null &
      sleep 20
    fi
  done
  sleep 120
done
echo "$(date -u +%FT%TZ) supervisor stopped" >> ~/logs/supervise.log
