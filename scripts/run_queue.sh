#!/usr/bin/env bash
# Run training commands sequentially; each line of the queue file is one command.
# Finished runs are skipped by the trainers themselves (DONE marker), so the queue is restartable.
# Usage: scripts/run_queue.sh <queue_file> <log_dir>
set -uo pipefail
queue=$1; logdir=$2; mkdir -p "$logdir"
while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  name=$(echo "$cmd" | grep -oP 'name=\K\S+' || echo job)
  echo "[$(date +%F' '%T)] START $name" >> "$logdir/queue.log"
  if bash -c "$cmd" > "$logdir/$name.log" 2>&1; then s=OK; else s=FAIL; fi
  echo "[$(date +%F' '%T)] $s $name" >> "$logdir/queue.log"
done < "$queue"
