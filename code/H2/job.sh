#!/bin/bash
#PBS -l select=1:ncpus=6:mem=2gb:scratch_local=2gb
#PBS -l walltime=4:00:00
#PBS -m ae
#PBS -j oe

# H2 synchronous bootstrap: re-estimate FW on one bootstrap replication of one
# market.  Same BOOT_ID across markets uses the same block indices to preserve
# cross-market co-movement (synchronous bootstrap).
# Required env vars passed via qsub -v:
#   DATADIR — project root on storage
#   MARKET  — sp500 | btc
#   BOOT_ID — 1..500

DATADIR="${DATADIR:?DATADIR not set}"
MARKET="${MARKET:?MARKET not set}"
BOOT_ID="${BOOT_ID:?BOOT_ID not set}"

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

echo "=== H2 FW ${MARKET} boot_id=${BOOT_ID} started at $(date) on $(hostname) ==="

python code/H2/calibrate_fw.py \
    --model fw \
    --market "$MARKET" \
    --boot-id "$BOOT_ID" \
    --block-size 60 \
    --seeds 100 \
    --de-maxiter 100 \
    --de-popsize 30 \
    --nm-maxiter 500 \
    --parallel-seeds 0 \
    --data-dir "$SCRATCHDIR/data" \
    --out-dir "$SCRATCHDIR/results"

cp "$SCRATCHDIR/results"/*.json "$RESULTS_DIR"/ || exit 2
echo "=== Finished at $(date) ==="
