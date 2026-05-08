#!/bin/bash
# H3 structural break test: FW model, BTC pre-ETF (2022-01-19 to 2024-01-09)
# vs post-ETF (2024-01-10 to 2025-12-31).
#
# Total 1020 jobs:
#   - 10 original reruns per period x 2 periods = 20 jobs
#       (different seed blocks: 1-100, 101-200, ..., 901-1000)
#   - 500 bootstrap reps per period x 2 periods = 1000 jobs
#       (independent block bootstrap, always seeds 1-100)
#
# Per job: 6 cores, 2GB RAM, 4h walltime, 100 seeds, DE maxiter=100, popsize=30,
# NM maxiter=500, FWCHL16, block size 30 trading days.
#
# Submission is interleaved (pre, post, pre, post, ...) so partial results
# contain both periods early.

BASE="/storage/brno2/home/$USER/diplomka"
DATADIR="${BASE}/h3_fw"
JOB_SCRIPT="${DATADIR}/code/H3/job.sh"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "ERROR: $JOB_SCRIPT not found"
    exit 1
fi

TOTAL=0

# --- 20 original reruns: 10 seed blocks per period ---
for ORIG_RUN_ID in $(seq 1 10); do
    for PERIOD in pre post; do
        qsub -N "h3fw_btc_${PERIOD}_orig_r$(printf "%02d" $ORIG_RUN_ID)" \
             -v "DATADIR=${DATADIR},PERIOD=${PERIOD},BOOT_ID=0,ORIG_RUN_ID=${ORIG_RUN_ID}" \
             "$JOB_SCRIPT"
        TOTAL=$((TOTAL + 1))
    done
done
echo "[${TOTAL}/1020] Submitted 20 original reruns (10 per period)"

# --- 1000 bootstrap jobs: interleaved pre/post per boot_id ---
for BOOT_ID in $(seq 1 500); do
    for PERIOD in pre post; do
        qsub -N "h3fw_btc_${PERIOD}_b$(printf "%03d" $BOOT_ID)" \
             -v "DATADIR=${DATADIR},PERIOD=${PERIOD},BOOT_ID=${BOOT_ID}" \
             "$JOB_SCRIPT"
        TOTAL=$((TOTAL + 1))
    done
    if [ $((BOOT_ID % 50)) -eq 0 ]; then
        echo "[${TOTAL}/1020] Submitted up to boot_id=${BOOT_ID}"
    fi
done

echo ""
echo "Total: ${TOTAL} jobs (20 original + 1000 bootstrap, interleaved pre/post)"
echo "Settings: 100 seeds, DE maxiter=100 popsize=30, FWCHL16, block_size=30, 6 cores, 4h"
