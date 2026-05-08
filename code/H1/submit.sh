#!/bin/bash
# H1: FW calibration on rescaled common-sample returns.
# 3 markets x 10 runs (non-overlapping seed blocks 1..100, ..., 901..1000) = 30 jobs.
# Each job: 8 cores, 2GB RAM, 4h walltime, DE maxiter=100, popsize=30, NM maxiter=500.

BASE="/storage/brno2/home/$USER/diplomka"
DATADIR="$BASE/h1_fw"
JOB_SCRIPT="$DATADIR/code/H1/job.sh"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: $JOB_SCRIPT not found"
    exit 1
fi

count=0
for MARKET in sp500 btc eth; do
    for RUN in $(seq 1 10); do
        JOBNAME="h1fw_${MARKET}_r$(printf "%02d" $RUN)"
        qsub -N "$JOBNAME" \
             -v "DATADIR=$DATADIR,MARKET=$MARKET,RUN_ID=$RUN" \
             "$JOB_SCRIPT"
        count=$((count+1))
        echo "[$count/30] Submitted $JOBNAME"
    done
done

echo ""
echo "Total: $count jobs submitted (FW H1, 3 markets x 10 runs)"
