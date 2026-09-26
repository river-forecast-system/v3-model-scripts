# rfs-v3-scripts

Scripts that build the RFS v3 products, one folder each:

- `retrospective/`: the retrospective simulation stores
- `forecasts15/`: the 15 day forecasts
- `floodmaps/`: the flood maps

Needs Python 3.14 and a local checkout of [river-route](../river-route) beside this one. `uv sync` installs both.

## retrospective

Builds `hourly`, `daily`, `monthly`, `yearly` and `maximums` from ERA5 runoff and the v3 hydrography. `spec.py` holds
the layout from the RFS v3 specification, "Dataset Structure and Schematics".

```bash
python retrospective/1_prepare_hydrography.py                 # each region's routing files, into ~/data/rfsv3/routing
python retrospective/2_prepare_runoff.py                      # monthly ERA5 netCDFs -> one zarr per year
python ../river-route/benchmarks/concat_era5_yearly_zarr.py \
    ~/data/era5_zarr_16x16_12month ~/data/era5_1940_2024_16x16_18month.zarr   # -> one zarr for the whole record
python retrospective/3_route_by_region.py --dry-run           # what each region costs, and its k/x stability
python retrospective/3_route_by_region.py                     # by region, writing one store per calendar year
python retrospective/3_route_by_year.py                       # by year: every region for a year, then its columns
python retrospective/3_route_regions.py --processes 4 --threads 8   # stock river-route: a zarr per region per year
python retrospective/3_reduce_hourly.py                        # those zarrs -> daily, monthly, yearly, maximums
python retrospective/3_concatenate.py --s3-url s3://<bucket>/<prefix>  # or -> every whole-record store, uploaded
```

Step 3 has three versions. `3_route_by_year.py` writes the whole-record stores directly; `3_route_by_region.py` writes
one store per calendar year, to be merged along time afterwards; `2_route_regions.py` writes river-route's own output,
one zarr per region per year, and none of the published stores.

### 1_prepare_hydrography.py

Writes each region's routing files into `--routing`, one `region=<id>` folder per region of the hydrography, which
is published beside it as `routing/`. The hydrography is only read, so the pipeline that builds it holds no routing
configuration, except the Muskingum `musk_k` and `musk_x` it computes so they are attributes of the GIS files. The
router reads two of the files: `routing.parquet` (`river_id`, `next_river_id`, `k`, `x`) and
`gridweights_ERA5_<id>.nc`, the share of each catchment in each ERA5 cell. The third, `gridweights_ERA5_<id>.parquet`,
is an extra copy of the weights for jsrr, the browser router, and does not replace the netCDF. All list rivers in
`riverIndex` order, which is topological. The router pairs weights with rivers by position, not by id, so the files
must agree.

The routing table is the region's metadata with its columns renamed. Nothing is recomputed.

The weights match what `river_route.runoff.grid_weights` computes, without its Voronoi diagram. On a regular grid
each cell is the box between the midpoints to its neighbors. geopandas' spatial index pairs each catchment with the
boxes it overlaps in web mercator, where the catchments are published. Catchments inside one box are kept whole; the
rest are cut with `shapely.clip_by_rect`, which is about 17x faster than the general intersection `geopandas.overlay`
uses and gives the same areas. The pieces are measured in cylindrical equal-area with `GeoSeries.to_crs`. The boxes
stop at the antimeridian, which no v3 catchment comes within 1.5 degrees of; one that crosses it stops the run. On
region 5020000010 it gives the same (river, cell) rows as river-route, with proportions within 1e-6.

All 47 regions take 30 s on 32 cores, and the largest peaks at 6.5 GB. Regions with all three files are skipped unless
you pass `--overwrite`. `--grid` is any file on the ERA5 grid; only its coordinates are read.

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
River-route's zarr writer, compressing with `spec.COMPRESSOR` in place of its lz4, saves each year as
`--discharge-root/region=<id>/discharge_era5_<yyyy>01_<yyyy>12.zarr`, and the state it ended with is saved beside it
as `channel_state_era5_<yyyy>01_<yyyy>12.parquet`. The network and weight table are read once per region and reused
by every year. The run ends by printing its total time.

`--processes` regions route at once, biggest first, each on `--threads` threads; their product may not exceed the
machine's cores. A year's state is written only after its discharge, so a rerun resumes each region after its last
saved year and skips a region that has 2024's.

### 3_reduce_hourly.py

Reduces the per-region yearly zarrs from `2_route_regions.py` to the published `daily`, `monthly`, `yearly` and
`maximums` stores, one per calendar year at `spec.store_path`: `daily/daily_1980.zarr` and so on. The values are
`spec.reduce_year`'s: daily, monthly and yearly means of the hourly values, and the annual maxima of the hourly values
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
