#!/bin/bash
#SBATCH --job-name=test_trackreader
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err

cd ~/doctorate/github/tracker_exp
source setup_tracker_env.sh

START=1
END=10
PER_JOB=2

seq $START $PER_JOB $END | while read s; do
    e=$((s+PER_JOB-1))
    echo "${s}_${e}"
done | parallel -j $SLURM_CPUS_PER_TASK python3 TrackReader.py 53@{}