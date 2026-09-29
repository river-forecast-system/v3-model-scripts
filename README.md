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
which must be written first. Each script skips files that already exist unless you pass `--overwrite`.

Each script also writes the global version of its files into `global/`, as `routing.parquet` and
`gridweights_<grid>_global.nc`, once every region has its own. It rebuilds them whenever it rewrote a region. Global
files let the whole world route in one process on many threads. That is faster when reading the forcing costs more
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

Builds `hourly`, `daily`, `monthly`, `yearly` and `maximums` from ERA5 runoff and the v3 hydrography, laid out as
`rfs_spec.py` says. It routes with the `routing.parquet`
and `gridweights_ERA5_<id>.nc` of `1_prepare_inputs/`.

```bash
python 2_retrospective/2_prepare_runoff.py                      # monthly ERA5 netCDFs -> one zarr per year
python ../river-route/benchmarks/concat_era5_yearly_zarr.py \
    ~/data/era5_zarr_16x16_12month ~/data/era5_1940_2024_16x16_18month.zarr   # -> one zarr for the whole record
python 2_retrospective/3_route_by_region.py --dry-run           # what each region costs, and its k/x stability
python 2_retrospective/3_route_by_region.py                     # by region, writing one store per calendar year
python 2_retrospective/3_route_by_year.py                       # by year: every region for a year, then its columns
python 2_retrospective/3_route_regions.py --processes 4 --threads 8   # stock river-route: a zarr per region per year
python 2_retrospective/3_reduce_hourly.py                        # those zarrs -> daily, monthly, yearly, maximums
python 2_retrospective/3_concatenate.py --s3-url s3://<bucket>/<prefix>  # or -> every whole-record store, uploaded
```

Step 3 has three versions. `3_route_by_year.py` writes the whole-record stores directly; `3_route_by_region.py` writes
one store per calendar year, to be merged along time afterwards; `2_route_regions.py` writes river-route's own output,
one zarr per region per year, and none of the published stores.

### 3_route_by_year.py

For each year, every region routes concurrently. Each region starts from its channel state at the end of the previous
year and writes its rivers into its rows of a single global (river, hour) buffer in shared memory. That buffer is the
concatenation of all regions. The script then rounds the year, reduces it to daily, monthly, yearly and maximum
values, and writes that year's columns of every store. Nothing is held on disk except the channel states.

The global buffer is 173 GB for a leap year, plus whatever the routers hold while they run. Each year's write updates
part of every shard, because a shard spans the whole time axis, so zarr reads and rewrites each shard once per year.
A rerun resumes after the last year recorded in `--out-dir/last_year`.

### 3_route_regions.py

The example in `river-route/examples/example_usage.py`, run for every region. Each region routes the 85 yearly runoff
zarrs from step 2 in order, one Router per year, each year starting from the channel state the last one ended with.
River-route's zarr writer, compressing with `rfs_spec.COMPRESSOR` in place of its lz4, saves each year as
`--discharge-root/region=<id>/discharge_era5_<yyyy>01_<yyyy>12.zarr`, and the state it ended with is saved beside it
as `channel_state_era5_<yyyy>01_<yyyy>12.parquet`. The network and weight table are read once per region and reused
by every year. The run ends by printing its total time.

`--processes` regions route at once, biggest first, each on `--threads` threads; their product may not exceed the
machine's cores. A year's state is written only after its discharge, so a rerun resumes each region after its last
saved year and skips a region that has 2024's.

### 3_reduce_hourly.py

Reduces the per-region yearly zarrs from `2_route_regions.py` to the published `daily`, `monthly`, `yearly` and
`maximums` stores, one per calendar year at `rfs_spec.store_path`: `daily/daily_1980.zarr` and so on. The values are
`rfs_spec.reduce_year`'s: daily, monthly and yearly means of the hourly values, and the annual maxima of the hourly values
and of the daily means. ERA5 starts at 1940-01-01 07:00, so the first day, month and year of 1940 are means of the
hours present. It does not write `hourly`.

The work is one region's year at a time, `--processes` at once. A year is reduced only once its channel state is
saved, so this can run behind routing. Each finished region year is marked in `--out-dir/.progress` and skipped by a
rerun, and when a year has every region, `Q_timesteps` is filled for monthly and yearly.

### 3_concatenate.py

The fast version of `3_reduce_hourly.py`. It writes all five published stores over the whole record, `hourly.zarr` as
well, one river's record per chunk and 250 rivers per shard, and uploads them to S3 as they are written. Its
`daily`, `monthly`, `yearly` and `maximums` are bit identical to `3_reduce_hourly.py`'s years concatenated along time.

The 15 TB of hourly record is read once and written once. Each process streams a segment of rivers in `riverIndex`
order: it decodes the next 500-river input chunk of all 85 years into a 2.2 GB buffer, writes every shard that
buffer completes, and carries the leftover rivers to the next chunk. No shard is written twice or read back, and the
only input read twice is the chunk under each cut between segments, under 1% with the defaults. `Q_timesteps` is written at the end from
22 GB of monthly and yearly values kept in `--work-dir`, not by reading the stores back.

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
Routing must be finished: every region needs every year. `--overwrite` deletes the stores and `--work-dir`.

### 3_route_by_region.py

Routes from the one concatenated ERA5 zarr, in windows of whole calendar years, and writes each year into its own
store: `hourly/hourly_1980.zarr`, `daily/daily_1980.zarr` and the same for monthly, yearly and maximums. Each store
holds every river in `riverIndex` order, so a year is whole in the river dimension as soon as its regions are
written. Merging a store's years along time is a concatenation of whole chunks and can happen at upload.

Nothing is held between years: no hourly record, no reductions, no working disk beyond the channel states. A region
costs one window, `n_rivers * window_hours * 4` bytes, which is 10.6 GB per year for the largest. Regions therefore
do not take turns — they start biggest first and as many run at once as `--memory-gb` and `--jobs` allow, each
admitted when its window fits in what is left, so the small regions fill in around the large ones.

A window is read out of the concatenated store by the chunk tiles the region's weight table touches: exactly the
chunks zarr has to decompress, at most 60 for any region, rather than a bounding box that is nearly the whole globe
for the four regions that straddle the prime meridian, where ERA5's 0..360 longitudes wrap.

Each region's rivers are one contiguous run of `riverIndex`, so each region writes its own rows directly. Only the
first and last 250-river shard of a region can hold a neighbor's rivers, so only writes to those two shards take a
file lock.

Writing costs about 80 µs per chunk whatever the chunk holds, and a chunk is one river's year, so the four reduced
stores together cost as much to write as `hourly` while holding 4% of the data. Writing all 42 years of all five
stores is roughly 30 core-hours.

`--dry-run` prints each region's memory, the chunks it reads, and how its `k` and `x` fare at `--dt-routing`, and
routes nothing. `--dt-routing`, `--network-conditioning`, `--k-scale` and `--x` change the routing parameters in
memory without rewriting `routing.parquet`.

A rerun resumes. A year's rows are written to every store before its window's channel state is saved, so a state
file means every year up to it is complete; a region continues after its last saved year and a finished region is
skipped. `--overwrite` starts over. Once every region is written, the last run fills `Q_timesteps` and consolidates
every year of every store.

## 4_forecasts

Builds the 15 day forecast's `discharge.zarr` from the 51 IFS ensemble GRIB files of one initialization, routed over
the whole world at once with `global/` of `1_prepare_inputs`, on the `O1280` weights.

```bash
python 4_forecasts/1_route_forecast.py --init-state <state.parquet>   # every member -> forecasts-work/<YYYYMMDDHH>/
python 4_forecasts/2_discharge_zarr.py                                # -> forecasts15/year=YYYY/month=MM/day=DD/discharge.zarr
python 4_forecasts/3_summary_files.py --initialization 2026-09-27T00  # -> maps/*/styles.{json,bin} and alerts.csv
```

The first two default to the newest initialization they find: in `~/data/rfsv3/forcings/ifs`, named
`ro_<YYYYMMDD>_<HH>z_<cf|pfN>.grib`, and in the work directory.

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
GB. The member files are always deleted once the store is in place.

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
