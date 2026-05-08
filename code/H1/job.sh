#!/bin/bash
#PBS -l select=1:ncpus=8:mem=2gb:scratch_local=2gb
#PBS -l walltime=4:00:00
#PBS -m ae
#PBS -j oe

# Calibrate the FW (DCA-HPM) model on one market for one seed block (run_id).
# Required env vars passed via qsub -v:
#   DATADIR  — path to the project root on storage
#              (must contain: code/, data/, results/ will be created)
#   MARKET   — sp500 | btc | eth
#   RUN_ID   — 1..10  (selects the seed block: 1->seeds 1..100, 2->101..200, ...)

DATADIR="${DATADIR:?DATADIR not set}"
MARKET="${MARKET:?MARKET not set}"
RUN_ID="${RUN_ID:?RUN_ID not set}"

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

echo "=== H1 FW $MARKET run $RUN_ID started at $(date) on $(hostname) ==="

python code/H1/calibrate_fw.py \
    --market "$MARKET" \
    --run-id "$RUN_ID" \
    --data-dir "$SCRATCHDIR/data" \
    --out-dir "$SCRATCHDIR/results" \
    --parallel-seeds 0

cp "$SCRATCHDIR/results"/*.json "$RESULTS_DIR"/ || exit 2
echo "=== Finished at $(date) ==="
