#!/bin/bash
# resubmit_failed.sh
# Splits all files in seed_fixed_failed/ into batches of 96 and submits
# one SLURM job per batch (96 cores, one file per core, one round of execution).

set -e

CORES=96
BATCH_SIZE=96
TIME="07:00:00"

TRACKER_DIR="/home2/user/u0100486/linux/doctorate/github/tracker_exp"
FAILED_DIR="$TRACKER_DIR/seed_fixed_failed"
ENV_SCRIPT="$TRACKER_DIR/setup_tracker_env.sh"
LOG_DIR="$TRACKER_DIR/jobs/logs"
TASKLIST_DIR="$TRACKER_DIR/jobs/tasklists"
mkdir -p "$LOG_DIR" "$TASKLIST_DIR"

shopt -s nullglob
FILES=("$FAILED_DIR"/output_run_*.root)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files found in $FAILED_DIR — nothing to resubmit."
    exit 0
fi

# Build full task list
TASKS=()
for FPATH in "${FILES[@]}"; do
    FNAME=$(basename "$FPATH")
    if [[ ! "$FNAME" =~ ^output_run_([0-9]+)_([0-9]+)_([0-9]+)\.root$ ]]; then
        echo "  Skipping unrecognised filename: $FNAME"
        continue
    fi
    TASKS+=("${BASH_REMATCH[1]}@${BASH_REMATCH[2]}_${BASH_REMATCH[3]}")
done

NTASKS=${#TASKS[@]}
N_JOBS=$(( (NTASKS + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Found $NTASKS tasks — submitting $N_JOBS job(s) of up to $BATCH_SIZE tasks each (TIME=$TIME) ..."

for (( JOB=0; JOB<N_JOBS; JOB++ )); do
    START=$(( JOB * BATCH_SIZE ))
    BATCH=("${TASKS[@]:$START:$BATCH_SIZE}")
    BSIZE=${#BATCH[@]}

    TASKLIST_FILE="$TASKLIST_DIR/tasklist_batch_${JOB}.txt"
    printf '%s\n' "${BATCH[@]}" > "$TASKLIST_FILE"

    JOB_SCRIPT=$(mktemp /tmp/job_resubmit_${JOB}_XXXXXX.sh)
    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
#SBATCH --job-name=retry_failed_${JOB}
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=128G
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/retry_failed_${JOB}_%j.out
#SBATCH --error=${LOG_DIR}/retry_failed_${JOB}_%j.err

cd "$TRACKER_DIR"
source "$ENV_SCRIPT" 2>/dev/null

echo "[\$(date)] Job ${JOB}: $BSIZE tasks on \$SLURM_CPUS_PER_TASK cores"
parallel --line-buffer -j \$SLURM_CPUS_PER_TASK python3 -u TrackReader.py {} < "$TASKLIST_FILE"
echo "[\$(date)] Job ${JOB} complete"
JOBEOF

    JID=$(sbatch --parsable "$JOB_SCRIPT")
    rm -f "$JOB_SCRIPT"
    echo "  Submitted job $JID  (batch $JOB: $BSIZE tasks)"
done

echo ""
echo "All $N_JOBS job(s) submitted."
echo "Monitor:  squeue -u \$USER"
echo "Logs:     $LOG_DIR/"
