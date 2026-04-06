#!/bin/bash
# release_jobs.sh
# Releases held SLURM jobs so that at most MAX_RUNNING jobs run at a time.
# Usage: bash release_jobs.sh [max_running]  (default: 2)

MAX_RUNNING=${1:-2}
POLL_INTERVAL=30   # seconds between checks

echo "Releasing held jobs with max $MAX_RUNNING running at a time..."

while true; do
    # Count currently running jobs
    RUNNING=$(squeue -u "$USER" -o "%T %r" | awk '$1=="RUNNING"' | wc -l)

    # Count held jobs (PENDING with JobHeldUser reason)
    HELD_JOBS=$(squeue -u "$USER" -o "%i %T %r" | awk '$2=="PENDING" && $3=="JobHeldUser" {print $1}')
    N_HELD=$(echo "$HELD_JOBS" | grep -c '[0-9]' || true)

    if [[ $N_HELD -eq 0 ]]; then
        echo "No more held jobs. Done."
        break
    fi

    SLOTS=$(( MAX_RUNNING - RUNNING ))
    echo "[$(date '+%H:%M:%S')] Running: $RUNNING / $MAX_RUNNING  |  Held: $N_HELD  |  Slots free: $SLOTS"

    if [[ $SLOTS -gt 0 ]]; then
        echo "$HELD_JOBS" | head -n "$SLOTS" | while read -r JID; do
            echo "  Releasing job $JID"
            scontrol release "$JID"
        done
    fi

    sleep "$POLL_INTERVAL"
done
