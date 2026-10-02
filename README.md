# rfs-v3-scripts

Scripts that build the RFS v3 products, one folder each:

1. `1_prepare_inputs/`: the routing files every product routes with, from the v3 hydrography
2. `2_retrospective/`: the retrospective simulation stores
3. `3_reforecasts/`: the reforecasts
4. `4_forecasts/`: the 15 day forecasts
5. `5_extended_forecasts/`: the extended range forecasts

Needs Python 3.14 and a local checkout of [river-route](../river-route) beside this one. `uv sync` installs both,
and this project, so every folder imports `rfs_spec`.

`rfs_spec.py` is the layout of every published store, from the RFS v3 specification
([rfs-specification-documents](../rfs-specification-documents), `docs/specs/rfs-v3.md`, "Zarr structuring"): the
dtypes, codec, keepbits, chunks and shards, the time axes and coordinates, and each store's metadata. The specification
names it the source of truth for encodings, and where the two disagree the specification wins and the module is fixed.

`config.sh` holds the settings of every step: the data paths, the record's years, and the processes and threads each
step runs on. Each folder's `simulate.sh` sources it and runs that folder's scripts in order. Every setting is exported
and keeps a value already set in the environment, so a run is configured without editing the file:
`FIRST_YEAR=1940 WARM_UP_YEARS=0 2_retrospective/simulate.sh`.

## 1_prepare_inputs

Writes the routing files into `--routing`, `~/data/rfsv3/routing` by default: one `region=<id>` folder per region of
the hydrography, and a `global` folder of the same files for the whole world. It is published beside the hydrography as
`routing/`. `regions.py` holds what the scripts share.

```bash
python 1_prepare_inputs/1_routing_files.py                           # routing.parquet
python 1_prepare_inputs/2_gridweights_era5.py                        # gridweights_ERA5_<id>.nc and .parquet
python 1_prepare_inputs/3_gridweights_ifs.py --grib <any IFS .grib>  # gridweights_O1280_<id>.nc
```

The router reads `routing.parquet` (`river_id`, `next_river_id`, `k`, `x`) and one weight table for its forcing, the
share of each catchment in each grid cell. Every file lists rivers in `riverIndex` order, which is topological. The
router pairs weights with rivers by position, not by id, so the weights read the river order from `routing.parquet`,
which must be written first. Each script skips files that already exist; a file is only rebuilt once you delete it.

Each script also writes the global version of its files into `global/`, as `routing.parquet` and
`gridweights_<grid>_global.nc`, once every region has its own, and keeps a global file that exists, so after
replacing a region's file, delete the global one too. Global files let the whole world route in one process on many threads. That is faster when reading the forcing costs more
than routing it. An example is a GRIB forecast on the networked storage of an HPC: routing the world at once reads it
one time, where routing by region reads it once per region.

A global file is its regions' files concatenated, with regions in `riverIndex` order; nothing is recomputed. Each
region is a closed network and one contiguous run of `riverIndex`. So the global table is still topological, and it is
checked against `hydrography/global/metadata.parquet`. A catchment's weights do not depend on its region, so the
concatenated weights are the world's. The 4.9 million rivers and 7.7 million ERA5 weights take under a second each.

### 1_routing_files.py

The region's metadata with its columns renamed, checked to be a closed, topologically sorted network. Nothing is
recomputed. The hydrography is only read, so the pipeline that builds it holds no routing configuration, except the
Muskingum `musk_k` and `musk_x` it computes so they are attributes of the GIS files.

### 2_gridweights_era5.py

The weights on the regular ERA5 grid, for the retrospective. `gridweights_ERA5_<id>.parquet` is an extra copy for jsrr,
the browser router, and does not replace the netCDF.

The weights match what `river_route.runoff.grid_weights` computes, without its Voronoi diagram. On a regular grid
each cell is the box between the midpoints to its neighbors. geopandas' spatial index pairs each catchment with the
boxes it overlaps in web mercator, where the catchments are published. Catchments inside one box are kept whole; the
rest are cut with `shapely.clip_by_rect`, which is about 17x faster than the general intersection `geopandas.overlay`
uses and gives the same areas. The pieces are measured in cylindrical equal-area with `GeoSeries.to_crs`. The boxes
stop at the antimeridian, which no v3 catchment comes within 1.5 degrees of; one that crosses it stops the run. On
region 5020000010 it gives the same (river, cell) rows as river-route, with proportions within 1e-6.

All 47 regions take 30 s on 32 cores, and the largest peaks at 6.5 GB. `--grid` is any file on the ERA5 grid; only its
coordinates are read.

### 3_gridweights_ifs.py

The weights on the reduced gaussian grid of the ECMWF IFS forecasts, for river-route's `ecmwf_grib` forcing, which
reads the forecast GRIB files as they are. They are `river_route.runoff.reduced_grid_weights`'s, and locate each cell by
`cell_index`, its position in a GRIB message's values, in place of `x_index` and `y_index`. The file is named for the
grid of `--grib`: `O1280` for the 9 km octahedral grid. The weights are checked to follow `routing.parquet` before the
file is renamed into place.

`reduced_grid_weights` reads a region's catchments whole: 11.7 GB for region 1020000010, and 31 GB for the largest,
3020000010, which takes 27 s. `--jobs` defaults to 4 regions at once.

## 2_retrospective

Builds the published retrospective stores, `hourly`, `daily`, `monthly`, `yearly`, `maximums` and `return-periods`,
from hourly ERA5 runoff and the v3 hydrography, laid out as `rfs_spec.py` says. It routes with the `routing.parquet`
and `gridweights_ERA5_<id>.nc` of `1_prepare_inputs/`.

```bash
2_retrospective/simulate.sh                               # every step below in order, settings from config.sh
python 2_retrospective/1_route_regions.py --first-year 1995 --last-year 2024 --warm-up 2 --processes 14 --threads 2
python 2_retrospective/2_concatenate.py --processes 7 --threads 4    # add --s3-url s3://<bucket>/<prefix> to upload
python 2_retrospective/3_return_periods.py --processes 20             # maximums.zarr -> return-periods.zarr
```

On the 32-core M3 Ultra, the 30 years 1995-2024 for all 4.9 million rivers routed in 25 minutes, concatenated in 30
and fit return periods in 7. The routed discharge is 1.5 TB, and the stores 1.75 TB, almost all of it `hourly`.

### 1_route_regions.py

The example in `river-route/examples/example_usage.py`, run for every region. Each region routes the years
`--first-year` to `--last-year` in order, one Router per year, each year starting from the channel state the last one
ended with. A year's runoff is every zarr in `--runoff-root/year=<yyyy>/`, by default the four 3-month zarrs of
`~/data/era5_zarr_16x16_3month`, routed in order by the year's Router and written as one store for the whole year:
`--discharge-root/region=<id>/discharge_era5_<yyyy>01_<yyyy>12.zarr`, through river-route's zarr writer compressing
with `rfs_spec.COMPRESSOR` and rounding to `rfs_spec.KEEP_BITS`. The state the year ended with is saved beside it as
`channel_state_era5_<yyyy>01_<yyyy>12.parquet`. The network and weight table are read once per region and reused by
every year.

Each region is warmed up first: its first `--warm-up` years, 2 by default, are routed from empty channels, their
discharge discarded, and the state they end with saved as `channel_state_warm_up.parquet`. The record then starts at
`--first-year` from that state, so it needs no runoff before the record and can start with ERA5 in 1940. `--processes` regions route at once, biggest first, each on `--threads`
threads; their product may not exceed the machine's cores. A year's state is written only after its discharge, so a
rerun resumes each region after its last saved year and skips a region that has `--last-year`'s.

### 2_concatenate.py

Reduces the per-region yearly zarrs from `1_route_regions.py` in one pass. It writes all five published stores over
the whole record, `hourly.zarr` as well, in each array's chunks and shards from `rfs_spec.LAYOUT`, and uploads them to
S3 as they are written. Its `daily`, `monthly`, `yearly` and `maximums` are reduced from the hourly values in the same
pass that writes them.

The 15 TB of hourly record is read once and written once. Each process streams a segment of rivers in `riverIndex`
order: it decodes the next 500-river input chunk of all 85 years into a 2.2 GB buffer, writes every shard that
buffer completes, and carries the leftover rivers to the next chunk. hourly and daily shards are written as the
segment streams; segments are cut on daily's 1,000-river shards. No shard is written twice or read back, and the only
input read twice is the chunk under each cut between segments, under 1% with the defaults. monthly, yearly and
maximums, whose shards are thousands of rivers, and `Q_timesteps` are written at the end from 25 GB of their values
kept in `--work-dir`, not by reading the stores back.

Whole shards are encoded with blosc and written directly, without zarr. zarr spends about 350 µs on every chunk,
which is 40 times the encoding of a yearly chunk. Each process first checks on a test shard that its bytes match
zarr's, and writes through zarr if they don't.

With `--s3-url`, the main process uploads each shard as soon as its segment reports it written, while the files are
still in the page cache, so the build never waits on the network and nothing is read back from disk. The metadata goes
first, so a bad bucket or credentials fail at once. Uploaded files are logged in `--work-dir`, and a rerun uploads
only what the log lacks, including a build finished earlier without `--s3-url`. `--storage-class` sets the class of
every object; a lifecycle rule on the bucket is the other way to move them to archive tiers.

On the 32-core M3 Ultra, on synthetic 85-year data, the build runs at about 1,500 rivers/s with the default 8
processes × 4 threads. That is about 55 minutes for all 4.9 million rivers, and about 19 ms of CPU per river, 70% of
it zstd. The hourly store comes out about the size of the input: 2.5× compressed on that data, 5.7 TB at full size.
Real discharge is smoother and compresses better. At that size the upload decides the total time: about 13 hours at
1 Gbit/s, and about 1.3 hours at 10 Gbit/s.

A rerun resumes. Each segment records the rivers it has written, after their shards, and continues from there.
Routing must be finished: every region needs every year. To build the stores again, delete them and `--work-dir`.

### 3_return_periods.py

Fits Gumbel, log-Pearson III, lognormal and Weibull, each by moments, to both annual maximum series of
`maximums.zarr`, the hourly maxima and the maxima of the daily means, and writes `return-periods.zarr` beside it: each
fit's flow at the recurrence intervals 1.5, 2, 5, 10, 25, 50 and 100 years, `max_simulated_hourly` and
`max_simulated_daily`, and the `recurrence_interval` and `annual_exceedance_probability` coordinates. The log fits
read maxima below 1e-4 m3/s as 1e-4, and a river whose maxima are all equal gets that value at every interval. Each
process fits and writes one 250,000-river shard, 20 for the world.

## 4_forecasts

Builds the 15 day forecast's `discharge.zarr` from the 51 IFS ensemble GRIB files of one initialization, routed over
the whole world at once with `global/` of `1_prepare_inputs`, on the `O1280` weights.

```bash
4_forecasts/simulate.sh                                                # every step below in order, settings from config.sh
python 4_forecasts/1_route_forecast.py --init-state <state.parquet>   # every member -> forecasts-work/<YYYYMMDDHH>/
python 4_forecasts/2_discharge_zarr.py                                # -> forecasts15/year=YYYY/month=MM/day=DD/discharge.zarr
python 4_forecasts/3_summary_files.py --initialization 2026-09-27T00  # -> maps/*/styles.{json,bin} and alerts.csv
```

The first two default to the newest initialization they find: in `~/data/rfsv3/forcings/ifs`, named
`<cf_00|pf_01..pf_50>_<YYYYMMDD>.grib` (00 UTC runs), and in the work directory.

### 1_route_forecast.py

Routes every member in one river-route `Router`, in ensemble mode, so each member starts from `--init-state` and the
network and weight table are read once. Each GRIB file is read once for the world. The IFS accumulations, at 1, 3 and
6 hour steps, are interpolated onto every hour and routed hourly, on the retrospective's stabilized network at its
`dt_routing`, so a retrospective channel state initializes them. The hourly discharge is averaged over 3 hours,
relabeled by the start of each interval, rounded to 13 keepbits and saved as `member_<NN>.nc`, a netCDF file of
`Q` `(member, riverId, time)` labeled with its member, river ids and interval start times.
Without `--init-state` the channels start empty, which is only right for testing.

river-route can resample irregular steps itself, but it does so one river at a time in pandas after aggregating, which
took 130 of the 150 s a member took. Every river shares one time axis and the aggregation is linear, so the script
interpolates the grid cells' accumulations instead. All 51 members route in 6.2 min on 32 cores, about 7.6 s each. Its
discharge matched river-route's own resampling bit for bit in 98.6% of values, and within 2e-4 relative in the rest.

### 2_discharge_zarr.py

Opens the members with `xarray.open_mfdataset`, concatenated along `member`, and checks their labels against the
published axes before writing anything. Writes `Q` `(riverId, member, time)`, `Qpercentiles` `(riverId, percentiles,
time)` and `Qmean` `(riverId, time)`, with the coordinates `riverId`, `member` 0..50, `time` in hours since the
initialization, `lead_time` on the time dimension, and `percentiles` 0..100 by 10. A chunk is one river's whole ensemble
over the whole horizon. So the members are read back a block of whole shards at a time, stacked, reduced and written by
`--jobs` processes. With 51 members every decile is a member's value, so the percentiles need no rounding, and the mean
is rounded to 13 keepbits. The 27 September 2026 forecast, 4.9 million rivers, took 4.5 min with 32 processes and is 61
GB. The member files are kept, about 120 GB a forecast, for you to remove once the store is published.

Members are numbered as ECMWF numbers them: the control forecast is 0 and the perturbed forecasts are 1..50. The
`member` coordinate says so in its `description`.

### 3_summary_files.py

Compares the forecast with the retrospective's `return-periods.zarr` and `fdc.zarr` and writes the four map stylesets
the web app reads, `timeseries`, `max-flow`, `time-to-peak` and `below-q95`, and `alerts.csv`. The stylesets read the
ensemble median, as the web app's own classification does, against `gumbel_hourly`. An alert is a river where at least
30% of members exceed a return period flow, as the web app's exceedance tables count, with the fields of a CAP alert.
The formats are documented in the specification's "Forecast map stylesets" and "Forecast alerts", and implemented in
`rfs_spec.write_styles` and the constants beside it.

Not built yet: `maps/esri_animation_tables/`, whose format the specification does not give, `fim.geo.parquet`, and the
warm states, whose partition the specification has not settled.
