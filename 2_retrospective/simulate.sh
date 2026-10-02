#!/usr/bin/env bash
# The retrospective simulation, every step in order:
#
#   1_route_regions.py   route every region over the ERA5 record, one store of hourly discharge per region per year
#   2_concatenate.py     those stores -> hourly, daily, monthly, yearly and maximums.zarr over the whole record
#   3_return_periods.py  maximums.zarr -> return-periods.zarr
#
# Its settings are in config.sh at the repository root, and any of them can be set in the environment instead:
#
#   2_retrospective/simulate.sh
#   FIRST_YEAR=1940 WARM_UP_YEARS=0 2_retrospective/simulate.sh
#   S3_URL=s3://<bucket>/<prefix> 2_retrospective/simulate.sh
#
# Each step keeps what an earlier run finished and resumes what it left unfinished, so rerunning this script continues
# an interrupted run, and passes over a finished one. Nothing is ever deleted: to rebuild an output, delete it first.
set -euo pipefail
cd "$(dirname "$0")/.."  # the repository root, where the scripts, rfs_spec.py and the virtual environment are
source config.sh

step() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

step "1/3 routing ${FIRST_YEAR}-${LAST_YEAR} after a ${WARM_UP_YEARS} year warm-up into ${DISCHARGE_ROOT}"
"$PYTHON" 2_retrospective/1_route_regions.py \
    --routing "$ROUTING" \
    --runoff-root "$RUNOFF_ROOT" \
    --discharge-root "$DISCHARGE_ROOT" \
    --first-year "$FIRST_YEAR" \
    --last-year "$LAST_YEAR" \
    --warm-up "$WARM_UP_YEARS" \
    --processes "$ROUTE_PROCESSES" \
    --threads "$ROUTE_THREADS"

step "2/3 concatenating the record into ${RETROSPECTIVE}"
"$PYTHON" 2_retrospective/2_concatenate.py \
    --hydrography "$HYDROGRAPHY" \
    --routing "$ROUTING" \
    --discharge-root "$DISCHARGE_ROOT" \
    --out-dir "$RETROSPECTIVE" \
    --work-dir "$CONCATENATE_WORK_DIR" \
    --processes "$CONCATENATE_PROCESSES" \
    --threads "$CONCATENATE_THREADS" \
    ${S3_URL:+--s3-url "$S3_URL"}

step "3/3 fitting return periods into ${RETROSPECTIVE}/return-periods.zarr"
"$PYTHON" 2_retrospective/3_return_periods.py \
    --out-dir "$RETROSPECTIVE" \
    --processes "$RETURN_PERIOD_PROCESSES"

step "done"
