"""
Route every region of the RFS v3 hydrography a calendar year at a time, and reduce each year into every product
below hourly while it is still in memory.

Regions are routed biggest first, the big ones on wide thread pools and the rest on narrow ones, filling every core.

What a region process does
--------------------------
For each year, in order, so the channel state carries forward:

    route the year            ->  hourly[river, 8760] in memory
    rfs_aggregate.fold_year   ->  folded into this region's daily, monthly, yearly and annual maximum accumulators
    write queue               ->  region=XXX/hourly_YYYY.zarr, written while the next year routes

and once the last year is routed the accumulators are complete, so each of the four derived stores is written for
this region in one call -- every chunk of them written exactly once.

That is the whole reason a year is the unit rather than a quarter. The spec chunks a published store as one river
over the whole time axis, so writing a period into it a year at a time is not an append but a read-modify-write of
the entire store: a 42 year build would rewrite daily.zarr 42 times and hourly.zarr 42 times, tens of terabytes of
disk bandwidth to produce hundreds of gigabytes. Holding a region's derived products in memory instead costs 19 GB
for the largest region of the v3 hydrography and writes each of them once. Only hourly is too large to hold that
way -- a region's whole hourly record is 446 GB -- so it alone keeps a per year store on disk, and
``3_build_hourly_daily_maximums.py`` assembles the global stores from what this leaves behind.

The stores this writes are all (river_id, time), the layout river-route's kernels produce, so nothing is transposed
on the way to disk. The hourly ones are intermediates that only the assembling step reads, so they keep
river-route's array names; the four derived ones are already in the spec's layout and take their encoding, chunking
and metadata from ``rfs_spec.py`` like every published store.

Narrowed regions
----------------
Routing a region at ``discharge_dtype='float16'`` halves both its hourly stores on disk and the buffer held in
memory, and costs nothing downstream: the reductions are taken and written float32 either way, a float16 value
already sits on the spec's keepbits grid, and step 3 promotes the values on the way into hourly.zarr. What it
cannot survive is a discharge above float16's 65,504 m3 s-1 ceiling.

Which regions stay under it is settled before anything routes, by ``fp16_regions.FP16_REGIONS`` -- 35 of the 47,
covering 59.4% of the rivers -- derived from the 85 years of annual maxima the v2 retrospective already published.
See that module for the derivation and how to regenerate it. Overflowing anyway is recoverable rather than fatal:
the fold raises before writing anything for that year, so the region falls back to float32 from there and loses one
year. The years of a region may therefore differ in dtype, which costs nothing, since every one is promoted to
float32 on the way into hourly.zarr.

Resuming
--------
``--resume`` (the default) skips any region whose products are all on disk, and restarts a half finished region at
its first missing year: the years already written are read back and re-folded, which is exact because the rounding
the fold does is idempotent, and routing picks up from the channel state checkpointed beside them. A region is
therefore never routed twice, but a region interrupted mid-year loses that year.
"""

import argparse
import contextlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import river_route as rr
import zarr
from river_route.router import writers

import fp16_regions
import rfs_aggregate
import rfs_spec

# every store the writer consolidates warns that consolidated metadata is not part of the zarr 3 spec
warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

# The encoding of the hourly intermediates, patched into the module constants river-route's zarr_writer reads. They
# are set at import rather than in main(), because a spawned pool worker re-imports this module and that import is
# what carries them into the process that routes.
writers.ZARR_KEEPBITS = rfs_spec.KEEP_BITS
writers.ZARR_COMPRESSOR = rfs_spec.COMPRESSOR
# Reading one river's year costs one chunk, so chunks are narrow: 3.5 MB at 100 rivers a chunk, and compression does
# not suffer for the smaller blocks -- zstd measured 1.94x at 100 against 1.87x at 1,000, because blosc already
# splits a chunk into blocks internally. Without sharding, 100 wide chunks would put ten times as many files on disk.
writers.ZARR_RIVERS_PER_CHUNK = 100
writers.ZARR_CHUNKS_PER_SHARD = 10

# Every region in --regions-root is routed. Runoff is restricted to --start-year..--end-year. A year with no file
# in --era5-root is reported and skipped rather than routed as a gap, so asking for more than has been downloaded
# shortens the run instead of silently changing what it covers.
START_YEAR = 1979
END_YEAR = 2020

# The schedule. Regions are routed biggest first, the big ones on wide pools and the rest on narrow ones, because a
# big network is mostly one main stem whose tail is single threaded no matter how many threads it is given -- past
# that point the threads are better spent on another region. Each phase asks the machine for JOBS x THREADS
# threads, and both default to filling every core: the job count is the core count divided by the per region thread
# count, so a phase saturates the machine without oversubscribing it.
BIG_RIVERS = 150_000  # regions with at least this many rivers are routed in the big phase
BIG_THREADS = 8  # threads for each big region
SMALL_THREADS = 3  # threads for each small region -- a small network's tail does not use more

# One year of the largest region is 10.6 GB float32, so the queue holds one write in flight and the process holds at
# most two year buffers. One worker is enough: zarr_writer fans a store's chunks out over the region's own threads.
# A depth of 0 writes inline instead, which costs the overlap and saves a year buffer -- the trade region=global
# has to make, where one year of every river in the world is 172.8 GB.
WRITE_QUEUE_WORKERS = 1
WRITE_QUEUE_DEPTH = 1

DERIVED_STORES = ('daily', 'monthly', 'yearly', 'maximums')


def region_of(region_dir: Path) -> str:
    return region_dir.name.split('=')[1]


def hourly_path(discharge_dir: Path, year: int) -> Path:
    return discharge_dir / f'hourly_{year}.zarr'


def state_path(discharge_dir: Path, year: int) -> Path:
    """The channel state after ``year``, checkpointed so a resume restarts from the year after it."""
    return discharge_dir / f'state_{year}.parquet'


def is_complete(store: Path) -> bool:
    """
    A store is finished only once its metadata is consolidated, which the writer does after the last array is
    written. A store left half built by an interrupted run is discharge with no time axis, so it reads as missing
    rather than as a year that was routed.
    """
    try:
        return json.loads((store / 'zarr.json').read_text()).get('consolidated_metadata') is not None
    except (OSError, ValueError):
        return False


def dtype_for(region: str, fp16_mode: str) -> str:
    """The discharge dtype a region routes at: the static table, unless the run forces one."""
    if fp16_mode == 'never':
        return 'float32'
    if fp16_mode == 'always':
        return 'float16'
    return 'float16' if region in fp16_regions.FP16_REGIONS else 'float32'


def resumable_years(discharge_dir: Path, years: list[int]) -> list[int]:
    """
    The run of years from the start of the record that a resume can pick up after.

    Contiguous from the first year and no further: a gap means the channel state after it was never valid, so
    everything from the gap on has to be routed again. A year counts only with both a consolidated store and the
    state checkpointed beside it, which the year writer writes in that order for exactly this reason.
    """
    resumable = []
    for year in years:
        if not is_complete(hourly_path(discharge_dir, year)) or not state_path(discharge_dir, year).exists():
            break
        resumable.append(year)
    return resumable


def build_configs(region_dir: Path, discharge_dir: Path, runoff_files: list[Path], years: list[int],
                  dt_routing: int, discharge_dtype: str, state_file: Path | None, progress: bool) -> rr.Configs:
    region = region_of(region_dir)
    discharge_dir.mkdir(parents=True, exist_ok=True)
    return rr.Configs(
        params_file=region_dir / 'routing.parquet',
        grid_runoff_files=runoff_files,
        grid_weights_file=region_dir / f'gridweights_ERA5_{region}.nc',
        discharge_files=[hourly_path(discharge_dir, year) for year in years],
        channel_state_init_file=state_file,
        forcing='vlateral',
        coeff='static',
        dt_routing=dt_routing,
        discharge_dtype=discharge_dtype,
        runoff_processing_mode='sequential',
        var_grid_runoff='ro',
        var_x='longitude',
        var_y='latitude',
        var_t='valid_time',
        progress_bar=progress,
        log_level='ERROR',
        unstable_coefficients='ignore',
    )


def make_year_writer(accumulators: rfs_aggregate.RegionAccumulators,
                     write_queue: rfs_aggregate.WriteQueue | None, discharge_dir: Path):
    """
    The writer river-route calls once per routed year.

    The order is what makes a resume safe. The fold runs first, on the routing thread, because it rounds the buffer
    onto the keepbits grid in place and the writer must store exactly the values the reductions were taken from --
    which is what lets a maximum in maximums.zarr be found unchanged in hourly.zarr. The channel state is
    checkpointed next, before the year's store exists, so a state file is never the only thing on disk for a year
    whose discharge is missing: a resume trusts a state only once the year beside it is complete. The store is then
    handed to the queue and written while the next year routes.

    Folding costs about 0.1 s for the largest region against minutes of routing, so doing it inline rather than on a
    worker costs nothing and keeps the buffer's ownership obvious: it belongs to the queue from the submit onward.

    Without a queue the write runs here instead, so the router waits for it. That is what a caller with no room for
    a second year buffer asks for.
    """
    submit = write_queue.submit if write_queue is not None else rfs_aggregate.WriteQueue.inline

    def write_year(router, dates, discharge_array, discharge_file, runoff_file='') -> None:
        year = pd.Timestamp(dates[0]).year
        accumulators.fold(pd.DatetimeIndex(dates), discharge_array)
        pd.DataFrame({'Q': router.channel_state}).to_parquet(state_path(discharge_dir, year))
        submit(writers.zarr_writer, router, dates, discharge_array, discharge_file, runoff_file)

    return write_year


def refold_year(accumulators: rfs_aggregate.RegionAccumulators, store: Path) -> None:
    """
    Fold a year that a previous run already routed, read back from its store.

    Exact rather than approximate: the store holds values the fold already rounded onto the keepbits grid, and
    rounding one of those again cannot move it, so the accumulators end up holding what they would have held had
    the year never been interrupted. A narrowed store comes back float16 and is handed on as the uint16 bit
    patterns the kernel reads, the same layout river-route routes into.
    """
    group = zarr.open_group(str(store), mode='r')
    dates = rfs_spec.store_dates(group['time'])
    hourly = group['Q'][:]
    if hourly.dtype == np.float16:
        hourly = np.ascontiguousarray(hourly).view(np.uint16)
    accumulators.fold(dates, hourly)


def write_region_products(discharge_dir: Path, accumulators: rfs_aggregate.RegionAccumulators,
                          river_ids: np.ndarray, threads: int) -> None:
    """
    Write this region's daily, monthly, yearly and maximums stores, each complete and each chunk written once.

    They are per region rather than slices of the global stores because a region boundary is not a multiple of the
    spec's 250 river shard, so two regions routing at once would otherwise write into the same shard file from two
    processes. ``3_build_hourly_daily_maximums.py`` assembles them, which is a copy along the river axis.
    """
    products = {
        'daily': ({'Q': accumulators.daily}, accumulators.daily_times),
        'monthly': ({'Q': accumulators.monthly}, accumulators.monthly_times),
        'yearly': ({'Q': accumulators.yearly}, accumulators.yearly_times),
        'maximums': ({'hourly': accumulators.max_hourly, 'daily': accumulators.max_daily}, accumulators.yearly_times),
    }
    with zarr.config.set({'async.concurrency': threads, 'threading.max_workers': threads}):
        for name, (variables, times) in products.items():
            path = discharge_dir / f'{name}.zarr'
            rfs_spec.create_store(path, name, {'times': times}, river_ids, timesteps=False)
            group = zarr.open_group(str(path), mode='r+')
            for variable, values in variables.items():
                group[variable][:] = values
            rfs_spec.consolidate(path)


def route_years(region_dir: Path, discharge_dir: Path, runoff_by_year: dict, years: list[int], dtype: str,
                state_file: Path | None, dt_routing: int, threads: int, progress: bool,
                accumulators: rfs_aggregate.RegionAccumulators, queue_depth: int) -> np.ndarray:
    """
    Route a run of years into the accumulators and the per year stores, and return the region's river ids.

    The Router never creates a pool, so ``threads`` only routes concurrently on the one passed to ``route()``.
    Each call builds its own pool: sharing one across concurrently routed regions would let their region schedules
    queue behind each other, and the zarr writer's ``zarr.config.set`` is process-global, so two regions writing at
    once in one process would overwrite each other's concurrency setting.
    """
    configs = build_configs(region_dir, discharge_dir, [runoff_by_year[y] for y in years], years, dt_routing, dtype,
                            state_file, progress)
    router = rr.Router(configs)
    pool_context = ThreadPoolExecutor(threads) if threads > 1 else contextlib.nullcontext()
    queue_context = (
        rfs_aggregate.WriteQueue(WRITE_QUEUE_WORKERS, queue_depth) if queue_depth else contextlib.nullcontext()
    )
    with pool_context as pool, queue_context as write_queue:
        router.set_discharge_writer(make_year_writer(accumulators, write_queue, discharge_dir))
        router.route(thread_pool=pool, threads=threads)
    return router.network.river_ids


def route_region(job: tuple) -> str:
    """
    Route one region over every year at the dtype ``fp16_regions`` fixed for it before the run started.

    Overflowing float16 anyway is recoverable rather than fatal. The fold raises before it writes anything for the
    year, so the region's stores stop at the last good year and the run picks up there at float32. The years of a
    region may therefore differ in dtype, which costs nothing: every one is promoted to float32 on the way into
    hourly.zarr.
    """
    region_dir, discharge_dir, runoff_by_year, dt_routing, threads, fp16_mode, resume, progress, queue_depth = job
    region = region_of(region_dir)
    start = time.time()
    years = sorted(runoff_by_year)
    discharge_dir.mkdir(parents=True, exist_ok=True)

    products_done = all(is_complete(discharge_dir / f'{name}.zarr') for name in DERIVED_STORES)
    if resume and products_done and len(resumable_years(discharge_dir, years)) == len(years):
        return f'{region} already complete, skipped'

    n_rivers = pq.ParquetFile(region_dir / 'routing.parquet').metadata.num_rows
    routed = resumable_years(discharge_dir, years) if resume else []
    accumulators = rfs_aggregate.RegionAccumulators(n_rivers, years)
    for year in routed:
        refold_year(accumulators, hourly_path(discharge_dir, year))

    dtype = dtype_for(region, fp16_mode)
    river_ids = None
    narrowed = len(routed) if dtype == 'float16' else 0
    remaining = [year for year in years if year not in routed]

    while remaining:
        try:
            river_ids = route_years(region_dir, discharge_dir, runoff_by_year, remaining, dtype,
                                    state_path(discharge_dir, routed[-1]) if routed else None,
                                    dt_routing, threads, progress, accumulators, queue_depth)
            narrowed += len(remaining) if dtype == 'float16' else 0
            remaining = []
        except OverflowError:
            if dtype == 'float32':
                raise
            # The year that overflowed wrote nothing, so the region's stores stop at the last good one: rebuild the
            # accumulators from those and carry on wide from there. Reaching this means the v2 bound behind
            # fp16_regions.FP16_REGIONS did not hold for this region on v3, which is worth knowing.
            routed = resumable_years(discharge_dir, years)
            accumulators = rfs_aggregate.RegionAccumulators(n_rivers, years)
            for year in routed:
                refold_year(accumulators, hourly_path(discharge_dir, year))
            narrowed = len(routed)
            remaining = [year for year in years if year not in routed]
            dtype = 'float32'
            print(f'{region} exceeded float16 after {len(routed)} years; routing the rest at float32 and it should '
                  f'be dropped from fp16_regions.FP16_REGIONS', flush=True)

    if river_ids is None:
        river_ids = zarr.open_group(str(hourly_path(discharge_dir, years[0])), mode='r')['river_id'][:]
    write_region_products(discharge_dir, accumulators, np.asarray(river_ids), threads)

    return (f'{region} finished on {threads} threads, {narrowed} of {len(years)} years narrowed '
            f'in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Route every region a year at a time, reducing each year into every product below hourly.'
    )
    parser.add_argument('--regions-root', default='/Users/rchales/data/rfsv3/hydrography')
    parser.add_argument('--era5-root', default='/Users/rchales/data/era5_zarr_16x16_12month')
    parser.add_argument('--out-dir', default='/Users/rchales/data/discharge_zarr')
    parser.add_argument('--dt-routing', type=int, default=3600)
    parser.add_argument('--start-year', type=int, default=START_YEAR)
    parser.add_argument('--end-year', type=int, default=END_YEAR)
    parser.add_argument('--cores', type=int, default=os.cpu_count() or 8,
                        help='cores to fill; each phase runs (cores // its thread count) regions at once')
    parser.add_argument('--regions', nargs='+', default=None,
                        help='route only these regions, by number; default is every region in --regions-root. Use '
                             'with a short --start-year/--end-year to time one region before committing to the run')
    parser.add_argument('--big-threads', type=int, default=BIG_THREADS)
    parser.add_argument('--small-threads', type=int, default=SMALL_THREADS)
    parser.add_argument('--fp16', choices=['table', 'never', 'always'], default='table',
                        help='table (the default) narrows the regions fp16_regions.FP16_REGIONS lists, derived '
                             'from the v2 retrospective\'s 85 years of annual maxima')
    parser.add_argument('--write-queue-depth', type=int, default=WRITE_QUEUE_DEPTH,
                        help='writes allowed in flight while the next year routes. Each one costs a year buffer; '
                             '0 writes inline, which is what a network too wide to hold two of them needs')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='route every year again even where a finished store is already on disk')
    parser.add_argument('--plan', action='store_true', help='print the schedule and exit without routing anything')
    args = parser.parse_args()

    big_jobs = max(1, args.cores // args.big_threads)
    small_jobs = max(1, args.cores // args.small_threads)

    # One runoff file per calendar year: a year is the unit this routes, folds and checkpoints in.
    runoff_by_year = {}
    for year in range(args.start_year, args.end_year + 1):
        files = sorted(Path(args.era5_root).glob(f'year={year}/*.zarr'))
        if len(files) == 1:
            runoff_by_year[year] = files[0]
        elif files:
            raise SystemExit(f'{args.era5_root}/year={year} holds {len(files)} stores; this routes whole years, so '
                             f'it needs exactly one. Rebuild the grids with 1_prepare_runoff_grids.py --months 12')
    if not runoff_by_year:
        raise SystemExit(f'no runoff for {args.start_year}..{args.end_year} in {args.era5_root}')
    if missing := [str(y) for y in range(args.start_year, args.end_year + 1) if y not in runoff_by_year]:
        print(f'no runoff in {args.era5_root} for {", ".join(missing)}')
    years = sorted(runoff_by_year)
    if years != list(range(years[0], years[-1] + 1)):
        raise SystemExit(f'the years available are not contiguous: {years}')

    regions_root = Path(args.regions_root)
    regions = sorted(region_of(d) for d in regions_root.glob('region=*'))
    if args.regions:
        if unknown := [r for r in args.regions if r not in regions]:
            raise SystemExit(f'{", ".join(unknown)} is not a region in {regions_root}')
        regions = [r for r in regions if r in args.regions]
    elif 'global' in regions:
        # region=global is every river in one network, an alternative to routing the regions rather than one more of
        # them, so a default run would otherwise route the whole world twice. Name it in --regions to route it.
        print(f'skipping region=global, which duplicates the other {len(regions) - 1} regions; '
              f'route it on its own with --regions global')
        regions = [r for r in regions if r != 'global']

    # River count comes from the parquet footer, so sizing the schedule reads no routing table.
    sizes = {r: pq.ParquetFile(regions_root / f'region={r}' / 'routing.parquet').metadata.num_rows for r in regions}
    ordered = sorted(regions, key=lambda r: -sizes[r])
    big = [r for r in ordered if sizes[r] >= BIG_RIVERS]
    small = [r for r in ordered if sizes[r] < BIG_RIVERS]

    n_days = len(pd.date_range(f'{years[0]}-01-01', f'{years[-1] + 1}-01-01', freq='D', inclusive='left'))
    print(f'{len(regions)} regions, {sum(sizes.values()):,} rivers, {len(years)} years ({years[0]}..{years[-1]})')
    print(f'  {len(big)} big (>= {BIG_RIVERS:,} rivers): {big_jobs} at a time on {args.big_threads} threads each '
          f'= {big_jobs * args.big_threads} of {args.cores} cores')
    print(f'  {len(small)} small: {small_jobs} at a time on {args.small_threads} threads each '
          f'= {small_jobs * args.small_threads} of {args.cores} cores')
    dtypes = {r: dtype_for(r, args.fp16) for r in regions}
    narrow = [r for r in regions if dtypes[r] == 'float16']
    print(f'  {len(narrow)} of {len(regions)} regions narrowed to float16 '
          f'({sum(sizes[r] for r in narrow) / sum(sizes.values()):.1%} of rivers), by --fp16 {args.fp16}')
    print(f'{"region":>12} {"rivers":>9} {"dtype":>9} {"year buf":>9} {"daily acc":>10}')
    for region in ordered:
        year_gb = sizes[region] * 8784 * (2 if dtypes[region] == 'float16' else 4) / 1e9
        print(f'{region:>12} {sizes[region]:>9,} {dtypes[region]:>9} {year_gb:>8.1f}G '
              f'{sizes[region] * n_days * 4 / 1e9:>9.1f}G')
    # A region process holds its daily accumulator plus two year buffers; the widest phase is the peak.
    buffers = args.write_queue_depth + 1
    peak = max(
        sum(sizes[r] * (n_days * 4 + 8784 * (2 if dtypes[r] == 'float16' else 4) * buffers) / 1e9
            for r in phase[:jobs])
        for phase, jobs in ((big, big_jobs), (small, small_jobs)) if phase
    )
    print(f'peak resident across the widest phase: about {peak:,.0f} GB '
          f'({buffers} year buffer{"s" if buffers > 1 else ""} per region at --write-queue-depth '
          f'{args.write_queue_depth})')

    if args.plan:
        raise SystemExit(0)

    def job_for(region: str, threads: int, progress: bool) -> tuple:
        return (
            regions_root / f'region={region}',
            Path(args.out_dir) / f'region={region}',
            runoff_by_year,
            args.dt_routing,
            threads,
            args.fp16,
            args.resume,
            progress,
            args.write_queue_depth,
        )

    def run_phase(phase: str, phase_regions: list[str], concurrency: int, threads: int) -> None:
        if not phase_regions:
            return
        # A progress bar is only readable when one region is routing; concurrent ones report when they finish.
        progress = concurrency == 1
        print(f'{phase}: {len(phase_regions)} regions, {concurrency} at a time on {threads} threads each', flush=True)
        jobs = [job_for(region, threads, progress) for region in phase_regions]
        if concurrency > 1:
            with ProcessPoolExecutor(max_workers=concurrency) as pool:
                for message in pool.map(route_region, jobs):
                    print(message, flush=True)
        else:
            for job in jobs:
                print(route_region(job), flush=True)

    started = time.time()
    # Big regions first: they are the long pole, and finishing them before the small ones start keeps the tail of the
    # run from being one huge region alone on the machine.
    run_phase('big regions', big, big_jobs, args.big_threads)
    run_phase('small regions', small, small_jobs, args.small_threads)
    print(f'all regions finished in {(time.time() - started) / 60:.1f} min')
