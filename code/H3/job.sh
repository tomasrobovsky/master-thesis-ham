#!/bin/bash
#PBS -l select=1:ncpus=6:mem=2gb:scratch_local=2gb
#PBS -l walltime=4:00:00
#PBS -m ae
#PBS -j oe

# H3 structural break: re-estimate FW on one BTC sub-period (pre/post-ETF).
# Two job modes:
#   BOOT_ID=0  (original headline rerun) — uses non-overlapping seed block
#                selected by ORIG_RUN_ID (1..10 → seeds 1..100, ..., 901..1000).
#   BOOT_ID>0  (bootstrap replication)   — independent block bootstrap on the
#                respective sub-period; always uses seeds 1..100.
# The two sub-samples are non-overlapping in time, so the cross-covariance
# vanishes and each period is bootstrapped independently.
#
# Required env vars passed via qsub -v:
#   DATADIR      — project root on storage
#   PERIOD       — pre | post
#   BOOT_ID      — 0 (original) | 1..500 (bootstrap)
#   ORIG_RUN_ID  — 1..10  (only used when BOOT_ID=0; default 1)

DATADIR="${DATADIR:?DATADIR not set}"
PERIOD="${PERIOD:?PERIOD not set}"
BOOT_ID="${BOOT_ID:?BOOT_ID not set}"
ORIG_RUN_ID="${ORIG_RUN_ID:-1}"

RESULTS_DIR="$DATADIR/results"

trap 'clean_scratch' TERM EXIT
mkdir -p "$RESULTS_DIR" || exit 1
cp -r "$DATADIR/code" "$SCRATCHDIR"/ || exit 1
cp -r "$DATADIR/data" "$SCRATCHDIR"/ || exit 1
mkdir -p "$SCRATCHDIR/results"
cd "$SCRATCHDIR" || exit 1

module add python/3.10 || exit 1
source /storage/brno2/home/$USER/venvs/diplomka/bin/activate || exit 1
echo "Python: $(which python), Cores: $(nproc)"

echo "=== H3 FW BTC period=${PERIOD} boot_id=${BOOT_ID} orig_run_id=${ORIG_RUN_ID} started at $(date) on $(hostname) ==="

python code/H3/calibrate_fw.py \
    --period "$PERIOD" \
    --boot-id "$BOOT_ID" \
    --orig-run-id "$ORIG_RUN_ID" \
    --block-size 30 \
    --seeds 100 \
    --de-maxiter 100 \
    --de-popsize 30 \
    --nm-maxiter 500 \
    --parallel-seeds 0 \
    --moment-set FWCHL16 \
    --data-dir "$SCRATCHDIR/data" \
    --out-dir "$SCRATCHDIR/results"

cp "$SCRATCHDIR/results"/*.json "$RESULTS_DIR"/ || exit 2
echo "=== Finished at $(date) ==="
