#!/usr/bin/env bash
# One 15 day forecast, every step in order:
#
#   1_route_forecast.py  route the 51 IFS members over the whole world -> one netCDF per member in FORECAST_WORK_DIR
#   2_discharge_zarr.py  the members -> forecasts15/year=YYYY/month=MM/day=DD/discharge.zarr
#   3_summary_files.py   the store -> maps/<styleset>/styles.{json,bin} and alerts.csv beside it
#
# Its settings are in config.sh at the repository root, and any of them can be set in the environment instead:
#
#   4_forecasts/simulate.sh                                   the newest initialization in IFS_DIR
#   INITIALIZATION=2026-09-27T00 INIT_STATE=<state.parquet> 4_forecasts/simulate.sh
#
# Step 3 compares the forecast with RETROSPECTIVE/return-periods.zarr and RETROSPECTIVE/fdc.zarr. No script builds
# fdc.zarr yet, so step 3 stops with "fdc.zarr does not exist" until one does.
#
# Each step keeps what an earlier run finished and resumes what it left unfinished, so rerunning this script continues
# an interrupted run. Nothing is ever deleted: the member files stay in FORECAST_WORK_DIR, about 120 GB a forecast,
# until you remove them, and to rebuild an output, delete it first.
set -euo pipefail
cd "$(dirname "$0")/.."  # the repository root, where the scripts, rfs_spec.py and the virtual environment are
source config.sh

# the newest control forecast in IFS_DIR, when no INITIALIZATION is set
if [[ -z $INITIALIZATION ]]; then
    newest=$(find "$IFS_DIR" -maxdepth 1 -name 'cf_00_*.grib' -exec basename {} \; | sort | tail -n 1)
    [[ -n $newest ]] || { echo "no cf_00_*.grib in $IFS_DIR" >&2; exit 1; }
    stamp=${newest#cf_00_}  # YYYYMMDD.grib, always the 00 UTC run
    INITIALIZATION="${stamp:0:4}-${stamp:4:2}-${stamp:6:2}T00"
fi

step() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

step "1/3 routing the members of ${INITIALIZATION} into ${FORECAST_WORK_DIR}${INIT_STATE:+ from ${INIT_STATE}}"
"$PYTHON" 4_forecasts/1_route_forecast.py \
    --initialization "$INITIALIZATION" \
    --ifs "$IFS_DIR" \
    --routing "$ROUTING" \
    --work-dir "$FORECAST_WORK_DIR" \
    --threads "$FORECAST_THREADS" \
    ${INIT_STATE:+--init-state "$INIT_STATE"}

step "2/3 writing the discharge store into ${FORECASTS}"
"$PYTHON" 4_forecasts/2_discharge_zarr.py \
    --initialization "$INITIALIZATION" \
    --work-dir "$FORECAST_WORK_DIR" \
    --hydrography "$HYDROGRAPHY" \
    --out-root "$FORECASTS" \
    --jobs "$FORECAST_JOBS"

step "3/3 writing the map stylesets and alerts"
"$PYTHON" 4_forecasts/3_summary_files.py \
    --initialization "$INITIALIZATION" \
    --out-root "$FORECASTS" \
    --retrospective "$RETROSPECTIVE" \
    --hydrography "$HYDROGRAPHY" \
    --jobs "$FORECAST_JOBS"

step "done"
