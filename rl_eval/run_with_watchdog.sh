#!/bin/bash
# Usage: run_with_watchdog.sh <logfile> <stall_secs> <cmd...>
# Runs cmd in background; if logfile stops growing for stall_secs, kills the process tree.
LOG="$1"; STALL="$2"; shift 2
"$@" > "$LOG" 2>&1 &
PID=$!
echo "[watchdog] pid=$PID log=$LOG stall_limit=${STALL}s"
last_size=-1; last_change=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
  sz=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$sz" != "$last_size" ]; then last_size=$sz; last_change=$now; fi
  if [ $((now - last_change)) -ge "$STALL" ]; then
    echo "[watchdog] LOG STALE ${STALL}s — killing $PID (hang detected)" | tee -a "$LOG"
    pkill -9 -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null
    # also sweep any orphaned venv gpu procs from this run
    sleep 3
    for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      cmd=$(ps -o cmd= -p "$gp" 2>/dev/null); case "$cmd" in *venv_titanrl*) kill -9 "$gp" 2>/dev/null;; esac
    done
    echo "[watchdog] cleanup done" | tee -a "$LOG"; exit 42
  fi
done
wait "$PID"; rc=$?
echo "[watchdog] pid=$PID exited rc=$rc"; exit $rc
