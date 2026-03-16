#!/bin/bash
#SBATCH --job-name=tr_87_10K
#SBATCH --cpus-per-task=96        # max CPUs per node on this cluster
#SBATCH --mem=64G
#SBATCH --time=00:30:00            # 100 events/core x ~10s/event = ~17min, 30min for safety
#SBATCH --output=/home2/user/u0100486/linux/doctorate/github/tracker_exp/jobs/job_%j.log
#SBATCH --error=/home2/user/u0100486/linux/doctorate/github/tracker_exp/jobs/job_%j.err

# Go to the working directory
cd ~/doctorate/github/tracker_exp

# Load your environment
source setup_tracker_env.sh

# Event parameters
RUN=88
START=1
END=10000
PER_JOB=105      # ceil(10000/96) = 105 events per core → exactly 96 chunks, one wave

# Generate start_end ranges and run in parallel
seq $START $PER_JOB $END | while read s; do
    e=$((s+PER_JOB-1))
    echo "${s}_${e}"
done | parallel --line-buffer -j $SLURM_CPUS_PER_TASK python3 -u TrackReader.py ${RUN}@{}