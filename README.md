# rfs-v3-scripts

Scripts that build the RFS v3 products, one folder each:

- `retrospective/`: the retrospective simulation stores
- `forecasts15/`: the 15 day forecasts
- `floodmaps/`: the flood maps

Needs Python 3.14 and a local checkout of [river-route](../river-route) beside this one. `uv sync` installs both.

## retrospective

Builds `hourly.zarr`, `daily.zarr`, `monthly.zarr`, `yearly.zarr` and `maximums.zarr` from ERA5 runoff and the v3
hydrography. `spec.py` holds the layout from the RFS v3 specification, "Dataset Structure and Schematics".

```bash
python retrospective/1_prepare_runoff.py      # monthly ERA5 netCDFs -> one zarr per year
python retrospective/2_route.py               # by region: route every year of a region, then write its rows
python retrospective/2_route_by_year.py       # by year: route every region for a year, then write its columns
```

Step 2 has two versions that write identical stores. Pick one.

### 2_route_by_year.py

For each year, every region routes concurrently. Each region starts from its channel state at the end of the previous
year and writes its rivers into its rows of a single global (river, hour) buffer in shared memory. That buffer is the
concatenation of all regions. The script then rounds the year, reduces it to daily, monthly, yearly and maximum
values, and writes that year's columns of every store. Nothing is held on disk except the channel states.

The global buffer is 173 GB for a leap year, plus whatever the routers hold while they run. Each year's write updates
part of every shard, because a shard spans the whole time axis, so zarr reads and rewrites each shard once per year.
A rerun resumes after the last year recorded in `--out-dir/last_year`.

### 2_route.py

Every published array is chunked as one river over the whole time axis, so a region can't write a river until its
whole record is routed. Each region routes its years in order and holds its hourly record and its daily, monthly,
yearly and maximum reductions in memory mapped `.npy` files under `--out-dir`. The operating system pages them to
disk, so memory holds about one routed year per region: 10.6 GB for the largest. After its last year, a region writes
its rows of every store once and deletes its files.

Working disk is the constraint. The largest region's record is 446 GB for 1979–2020. Put `--out-dir` on fast local
disk and set `--jobs` so the largest regions running together fit. The script prints what the largest `--jobs`
regions need before it starts. The published stores need about 2.3 TB more.

Each region's rivers are one contiguous run of `riverIndex`, so each region writes its own rows directly into the
stores. Only the first and last 250-river shard of a region can hold a neighbor's rivers, so only writes to those two
shards take a file lock.

A rerun resumes. A region continues after its last completed year, and a finished region is skipped. `--overwrite`
starts over. Once every region is written, the last run fills `Q_timesteps` and consolidates the stores.
