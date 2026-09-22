# rfs-v3-scripts

The scripts that build the **RFS v3 retrospective simulation products** — `hourly.zarr`, `daily.zarr`,
`monthly.zarr`, `yearly.zarr` and `maximums.zarr` — from ERA5 runoff and the v3 hydrography.

Everything these scripts write follows
[the RFS v3 specification](../rfs-specification-documents/docs/specs/rfs-v3.md), "Dataset Structure and
Schematics". `rfs_spec.py` is the source of truth the spec points at for encodings: no other file here picks a
dtype, a codec, a chunk shape or a keepbits value.

## The sequence

Run in order. Each step's output is the next step's input.

| # | script | reads | writes |
|---|--------|-------|--------|
| 1 | `1_prepare_runoff_grids.py` | monthly ERA5 runoff netCDFs | `era5_zarr_16x16_12month/year=YYYY/*.zarr`, chunked for reading one region's grid cells |
| 2 | `2_rfs_v3_retro_router.py` | those grids + `hydrography/region=XXX/` | per region: `hourly_YYYY.zarr` for each year, and one complete `daily`/`monthly`/`yearly`/`maximums` store |
| 3 | `3_build_hourly_daily_maximums.py` | those per region stores | the five global stores under `retrospective/` |

Two shared modules, not steps:

- **`rfs_spec.py`** — the spec's encodings, chunking, period cascade and store creation.
- **`rfs_aggregate.py`** — the numba kernel that reduces a routed year into every product below hourly in one
  pass, the accumulators that hold them for a region's whole record, and the write queue step 2 overlaps its
  writes with. `test_rfs_aggregate.py` and `test_step2_products.py` cover both.
- **`fp16_regions.py`** — which regions can be routed narrowed, and the derivation behind the table.

### Why a year is the unit

Step 2 routes one calendar year at a time, and reduces that year into daily means, monthly means, a yearly mean
and the two annual maximum series while the hourly buffer is still in memory — one pass, about 0.1 s for the
largest region against minutes of routing. The reductions accumulate across the region's whole record, so each of
the four derived stores is written **once, complete**, with every chunk written exactly once.

That is the point. The spec chunks a published store as one river over the whole time axis, so writing a year into
one is not an append but a read-modify-write of the entire store: appending 42 years to `hourly.zarr` would move
about 28 TB to produce 2.2 TB, and `daily.zarr` 1.1 TB to produce 86 GB. Holding the derived products in memory
instead costs 19 GB for the largest region. Hourly is the only one too large to hold that way — a region's whole
hourly record is 446 GB — so it alone keeps a per year store on disk for step 3 to assemble.

Step 2 carries no writer of its own. river-route's `zarr_writer` does the bitrounding and compression, so the
script only patches the spec's keepbits and codec into the module constants that writer reads. Its hourly stores
are intermediates, never a published product, so they keep river-route's array names. Everything it writes is
`(river_id, time)` — the layout river-route's kernels produce, written without a transpose.

### Narrowed regions

Routing a region at `discharge_dtype='float16'` halves both its hourly stores on disk and the buffer held in
memory, and costs nothing downstream: the reductions are taken and written float32 either way, a float16 value
already sits on the keepbits grid, and step 3 promotes the values on the way into the global store. The one thing
it cannot survive is discharge above float16's 65,504 m³/s ceiling.

`fp16_regions.py` settles that statically, before anything routes: **35 of the 47 regions, 59.4% of the rivers**.
The bound is the v2 retrospective's own annual maximum series — 85 years of routed hourly maxima over 6.8M rivers
— reduced to each river's all-time peak, grouped to the HydroBASINS level 2 basin each region is, and cut at 95%
of the ceiling. v2 and v3 river ids share the nine-digit encoding whose leading two identify that basin, so the
join is exact. Checked against what v3 actually routes over 1940–1944, every region came in *below* its v2 peak
(median 0.45×, worst 0.83×). That module documents the derivation and regenerates the table.

A region that overflows anyway is not lost: the fold raises before writing that year, so it falls back to float32
from there and loses one year. The years of a region may therefore differ in dtype, which costs nothing.

### Resuming

Step 2 resumes by default: it skips a region whose products are all on disk, and restarts a half finished region
at its first missing year by reading back the years already written and re-folding them, which is exact because
the rounding is idempotent. A region interrupted part way through a year loses that year, nothing more.

`return-periods.zarr` and `fdc.zarr` have no script here yet. `maximums.zarr` carries both the `hourly` and `daily`
annual maximum series the return period fits need.

> **Step 3 has not been reworked for this layout yet.** It still expects the per region, per *quarter* stores the
> previous step 2 wrote, and builds `daily.zarr` and `maximums.zarr` itself rather than assembling the ones step 2
> now produces. Rework it to concatenate `hourly_YYYY.zarr` along time and copy the four derived stores along the
> river axis before running a build. `4_build_monthly_yearly.py` is superseded — step 2 produces monthly and
> yearly directly — and should be deleted once step 3 lands.

## Running a full build

```bash
# 1. runoff grids, one store per calendar year, only for years not already converted
python 1_prepare_runoff_grids.py --months 12

# 2. route every region a year at a time, biggest first, filling every core
python 2_rfs_v3_retro_router.py --plan          # the schedule, the dtypes and the peak memory, without routing
nohup python 2_rfs_v3_retro_router.py > ~/route.log 2>&1 &

# 3. assemble the global stores  (pending the rework above)
python 3_build_hourly_daily_maximums.py --workers 16 --zarr-threads 2 --read-threads 4 --slab 8
```

Every script takes `--help`. Defaults point at this machine's paths; `2_rfs_v3_retro_router.py --regions <id>
--start-year Y --end-year Y` routes one region over one year, which is the cheap way to get a rate before
committing to a full run. `--plan` prints the per region dtype, year buffer and accumulator sizes and the peak
resident memory of the widest phase.

## Sizing a global 1979–2020 build

47 regions, 4,917,183 rivers, 368,184 hourly steps, 32 cores.

```
uncompressed
  hourly_YYYY.zarr (step 2)      ~2.0 TB    consumed by step 3; 35 regions narrowed, 0.70x of all-float32
  derived stores   (step 2)       ~94 GB    consumed by step 3
  hourly.zarr                    ~2.2 TB    19,669 shards of ~112 MB
  daily.zarr                      ~86 GB
  monthly + yearly + maximums      ~8 GB
  peak on disk                   ~2.1 TB

peak resident, 4 big regions at once    ~124 GB of 512 GB
```

## Install

Needs Python 3.14 and a local checkout of [river-route](../river-route) beside this one.

```bash
uv sync
uv run pytest        # the kernel and step 2 product checks
```
