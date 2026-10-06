#!/bin/bash
# Map the floods of one DEM tile, the one argument, within an HPC environment.
#

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

module purge
module load miniforge3 http_proxy
conda activate flood

DEM=$1

python "$(dirname "${BASH_SOURCE[0]}")/compute.py" \
    --dem "$DEM" \
    --dem-dir "$DEM_DIR" \
    --landcover-dir "$LANDCOVER_DIR" \
    --streamlines-dir "$STREAMLINES_DIR" \
    --lakes "$LAKES" \
    --n-file "$N_FILE" \
    --return-period-file "$RETURN_PERIOD_FILE" \
    --output-dir "$OUTPUT_DIR" \
    ${COPY_DIR:+--copy-dir "$COPY_DIR"}