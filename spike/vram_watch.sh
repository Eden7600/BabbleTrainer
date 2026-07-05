#!/bin/bash
# Samples GPU memory every 2s while a command runs; reports peak. Usage: ./vram_watch.sh <cmd...>
LOG=$(mktemp)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 2; done > "$LOG" ) &
MON=$!
"$@"
RC=$?
kill $MON 2>/dev/null
PEAK=$(sort -n "$LOG" | tail -1)
echo "VRAM peak during run: ${PEAK} MiB (of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) MiB)"
rm -f "$LOG"
exit $RC
