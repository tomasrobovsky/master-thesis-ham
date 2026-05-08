#!/bin/bash
# H2 cross-market Wald test: synchronous block-bootstrap of FW estimates
# on BTC and S&P 500 rescaled returns.
# Same boot_id uses identical block indices on both markets to preserve
# cross-market co-movement (synchronous bootstrap, Taylor & McGuire 2005).
#
# 500 bootstrap reps x 2 markets = 1000 jobs (interleaved by boot_id so partial
# results contain both markets early).
# Per job: 6 cores, 2GB RAM, 4h walltime, 100 seeds, DE maxiter=100, popsize=30,
# NM maxiter=500, block size 60 trading days, FWCHL17.

BASE="/storage/brno2/home/$USER/diplomka"
DATADIR="$BASE/h2_fw"
JOB_SCRIPT="$DATADIR/code/H2/job.sh"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: $JOB_SCRIPT not found"
    exit 1
fi

count=0
for BOOT_ID in $(seq 1 500); do
    for MARKET in btc sp500; do
        qsub -N "h2fw_${MARKET}_b$(printf "%03d" $BOOT_ID)" \
             -v "DATADIR=$DATADIR,MARKET=$MARKET,BOOT_ID=$BOOT_ID" \
             "$JOB_SCRIPT"
        count=$((count+1))
    done
    if [ $((BOOT_ID % 50)) -eq 0 ]; then
        echo "[${count}/1000] Submitted up to boot_id=${BOOT_ID}"
    fi
done

echo ""
echo "Total: ${count} jobs submitted (FW H2, 500 reps x 2 markets, interleaved)"
echo "Synchronous bootstrap: same boot_id uses same block indices across markets"
