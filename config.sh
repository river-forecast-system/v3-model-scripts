# Settings of every pipeline step, sourced by 2_retrospective/simulate.sh and 4_forecasts/simulate.sh. Each is exported,
# and each keeps a value already set in the environment, so a run is configured without editing this file:
#
#   FIRST_YEAR=1940 WARM_UP_YEARS=0 2_retrospective/simulate.sh
#   INITIALIZATION=2026-09-27T00 INIT_STATE=<state.parquet> 4_forecasts/simulate.sh
#
# Paths are absolute or relative to the repository root, where the simulate scripts run.

# ----------------------------------------------------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------------------------------------------------
export DATA_ROOT=${DATA_ROOT:-$HOME/data/rfsv3}
export HYDROGRAPHY=${HYDROGRAPHY:-$DATA_ROOT/hydrography}
export ROUTING=${ROUTING:-$DATA_ROOT/routing}
export RETROSPECTIVE=${RETROSPECTIVE:-$DATA_ROOT/retrospective}  # the published retrospective stores
export PYTHON=${PYTHON:-.venv/bin/python}
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
export CORES

# ----------------------------------------------------------------------------------------------------------------------
# Retrospective, 2_retrospective/simulate.sh
# ----------------------------------------------------------------------------------------------------------------------
export RUNOFF_ROOT=${RUNOFF_ROOT:-$DATA_ROOT/forcings/era5}  # hourly ERA5 runoff, <yyyy>.zarr or year=<yyyy>/*.zarr
# routed discharge and channel states, the input of concatenation
export DISCHARGE_ROOT=${DISCHARGE_ROOT:-$DATA_ROOT/discharge}
# progress and kept values, 25 GB for 85 years
export CONCATENATE_WORK_DIR=${CONCATENATE_WORK_DIR:-$RETROSPECTIVE/.work}
# s3://bucket/prefix to upload the concatenated stores to as they are written; empty uploads nothing.
# return-periods.zarr is not uploaded by any step
export S3_URL=${S3_URL:-}

# the record. Each region first routes its first WARM_UP_YEARS from empty channels, discards their discharge, and
# starts the record at FIRST_YEAR from the state they end with; 0 starts it from empty channels
export FIRST_YEAR=${FIRST_YEAR:-1995}
export LAST_YEAR=${LAST_YEAR:-2024}
export WARM_UP_YEARS=${WARM_UP_YEARS:-2}

# each step's processes x threads may not exceed CORES
export ROUTE_PROCESSES=${ROUTE_PROCESSES:-14}  # regions routed at once, biggest first
export ROUTE_THREADS=${ROUTE_THREADS:-2}
export CONCATENATE_PROCESSES=${CONCATENATE_PROCESSES:-7}  # segments of rivers built at once, 0.8 GB each for 30 years
export CONCATENATE_THREADS=${CONCATENATE_THREADS:-4}
export RETURN_PERIOD_PROCESSES=${RETURN_PERIOD_PROCESSES:-20}  # 250,000 river shards fit at once; the world has 20

# ----------------------------------------------------------------------------------------------------------------------
# Forecasts, 4_forecasts/simulate.sh
# ----------------------------------------------------------------------------------------------------------------------
export IFS_DIR=${IFS_DIR:-$DATA_ROOT/forcings/ifs}  # the IFS runoff GRIB files, <cf_00|pf_NN>_<YYYYMMDD>.grib, 00 UTC runs
export INITIALIZATION=${INITIALIZATION:-}  # YYYY-MM-DDTHH; empty is the newest control forecast in IFS_DIR
# a channel state parquet every member starts from, one Q per river or per sub-reach of the stabilized global network
# in routing/global/routing.parquet order. Empty starts from empty channels, which is right only for testing
export INIT_STATE=${INIT_STATE:-}
# each member's discharge, about 120 GB a forecast, kept until you remove it
export FORECAST_WORK_DIR=${FORECAST_WORK_DIR:-$DATA_ROOT/forecasts-work}
# the forecasts15/ tree the stores and summary files are written into
export FORECASTS=${FORECASTS:-$DATA_ROOT/forecasts15}
export FORECAST_THREADS=${FORECAST_THREADS:-$CORES}  # every member routes in one Router on these threads
export FORECAST_JOBS=${FORECAST_JOBS:-$CORES}  # processes writing and summarizing blocks of rivers at once
