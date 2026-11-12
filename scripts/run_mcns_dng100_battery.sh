#!/bin/zsh
# DNg100 battery on the authors' exact 4,310-cell mCNS export.
# Each arm writes its own timestamped outputs/mcns_dng100_* directory.
# Parameters: seed 1, 1024-replicate parameter block, indices as listed.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M=experiment.pugliese_rate_mcns_dng100
W=${WORKERS:-9}
BATTERY=${BATTERY:-outputs/mcns_dng100_battery_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$BATTERY"
run() { name=$1; shift; $PY -m $M --workers $W --out "$BATTERY/$name" "$@"; }

run intact_10056            --stim-bodies 10056 --replicates 0:1024
for R in E1 E2 E3 I1 I2; do
  run silence_${R}_10056    --stim-bodies 10056 --replicates 0:64 --silence $R
done
run silence_I1_I2_10056     --stim-bodies 10056 --replicates 0:64 --silence I1,I2
run shuffle_10056           --stim-bodies 10056 --replicates 0:64 --shuffle
run intact_10045            --stim-bodies 10045 --replicates 0:64
run intact_both             --stim-bodies 10045,10056 --replicates 0:64
for I in 150 200 250 300 350 450 500 600; do
  run sweep_I${I}_10056     --stim-bodies 10056 --replicates 0:32 --stim-amplitude $I
done
echo "battery done: $BATTERY"
