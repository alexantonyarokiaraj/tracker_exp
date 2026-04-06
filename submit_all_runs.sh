#!/bin/bash
# submit_all_runs.sh
# Submits SLURM jobs for runs 53..131.
# Events are batched into ~50K-event jobs (4 h each), each using 96 cores
# with GNU parallel.  Runs with >50K events get multiple jobs.
# Usage: bash submit_all_runs.sh [first_run] [last_run]
#   Defaults: first_run=53, last_run=131

set -e

FIRST_RUN=${1:-53}
LAST_RUN=${2:-131}
CORES=96                 # cpus-per-task
EVENTS_PER_JOB=50000     # ~5 h per job
TIME="05:00:00"

TRACKER_DIR="/home2/user/u0100486/linux/doctorate/github/tracker_exp"
ENV_SCRIPT="$TRACKER_DIR/setup_tracker_env.sh"
LOG_DIR="$TRACKER_DIR/jobs/logs"
mkdir -p "$LOG_DIR"

echo "Submitting jobs for runs $FIRST_RUN..$LAST_RUN ..."

for RUN in $(seq "$FIRST_RUN" "$LAST_RUN"); do
    # Count entries — suppress ROOT dictionary warnings
    N=$(cd "$TRACKER_DIR" && python3 count_entries.py "$RUN" 2>/dev/null | awk '{print $(NF-1)}')
    if [[ -z "$N" || ! "$N" =~ ^[0-9]+$ ]]; then
        echo "  Run $RUN: file not found or unreadable, skipping"
        continue
    fi

    N_JOBS=$(( (N + EVENTS_PER_JOB - 1) / EVENTS_PER_JOB ))
    echo "  Run $RUN: $N entries → $N_JOBS job(s)"

    for JOB_IDX in $(seq 0 $((N_JOBS - 1))); do
        JOB_START=$(( JOB_IDX * EVENTS_PER_JOB ))
        JOB_END=$(( JOB_START + EVENTS_PER_JOB - 1 ))
        [ $JOB_END -ge $N ] && JOB_END=$(( N - 1 ))
        JOB_N=$(( JOB_END - JOB_START + 1 ))

        # chunk size = ceil(JOB_N / CORES) so we fill exactly CORES parallel slots
        CHUNK_SIZE=$(( (JOB_N + CORES - 1) / CORES ))

        JOB_SCRIPT=$(mktemp /tmp/job_run${RUN}_${JOB_IDX}_XXXXXX.sh)
        cat > "$JOB_SCRIPT" <<JOBEOF
#!/bin/bash
#SBATCH --job-name=tracker_${RUN}_${JOB_IDX}
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=64G
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/tracker_${RUN}_${JOB_IDX}_%j.out
#SBATCH --error=${LOG_DIR}/tracker_${RUN}_${JOB_IDX}_%j.err

cd "$TRACKER_DIR"
source "$ENV_SCRIPT" 2>/dev/null

echo "[\$(date)] Starting run ${RUN} job ${JOB_IDX}: events ${JOB_START}..${JOB_END} (${JOB_N} events, chunk ${CHUNK_SIZE}, ${CORES} cores)"

seq ${JOB_START} ${CHUNK_SIZE} ${JOB_END} | while read S; do
    E=\$(( S + ${CHUNK_SIZE} - 1 ))
    [ \$E -gt ${JOB_END} ] && E=${JOB_END}
    echo "\${S}_\${E}"
done | parallel --line-buffer -j \$SLURM_CPUS_PER_TASK python3 -u TrackReader.py ${RUN}@{}

echo "[\$(date)] Run ${RUN} job ${JOB_IDX} complete"
JOBEOF

        JID=$(sbatch --parsable "$JOB_SCRIPT")
        echo "    Submitted job $JID (run $RUN, events $JOB_START..$JOB_END)"
        rm -f "$JOB_SCRIPT"
    done
done

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
echo "Logs in: $LOG_DIR"
