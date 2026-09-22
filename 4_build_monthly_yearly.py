"""
Resample a retrospective discharge store into the period average stores of the RFS v3 specification
(river-forecast-system/rfs-specification-documents, docs/specs/rfs-v3.md, "Dataset Structure and Schematics").

``--source`` is the store to cascade from and decides what there is left to build:

    retrospective/daily.zarr   ->  retrospective/monthly.zarr   Q, Q_timesteps      (the usual run)
                                   retrospective/yearly.zarr    Q, Q_timesteps
    retrospective/hourly.zarr  ->  retrospective/daily.zarr     Q                   (everything, from scratch)
                                   monthly.zarr, yearly.zarr

``3_build_hourly_daily_maximums.py`` writes daily.zarr out of the blocks it already holds to build hourly.zarr, so the
daily means cost it nothing but a pass over memory. Starting here from daily.zarr is what keeps that saving: the
monthly and yearly means come from the daily values, a twenty-fourth of the hourly ones, and no run ever reads the
hourly store back. Point ``--source`` at hourly.zarr to rebuild every level from scratch, which is what a store
written by an older concat run needs.

What the spec fixes, and so what this writes regardless of how the source store was laid out:

    riverId      int32 (riverId,), one chunk, in the source store's river order (the hydrography's topological order)
    time         int32 (time,), one chunk, hours since the store's first period, e.g.
                 "hours since 1975-01-01T00:00:00+00:00", calendar proleptic_gregorian -- hours on every axis, not days
    Q            float32 (riverId, time), one river a chunk over half the time axis, in shards of 250 rivers, fill
                 NaN, bitrounded to KEEP_BITS, attributes long_name, standard_name, units "m3 s-1",
                 aggregation_method "mean", keepbits
    Q_timesteps  monthly and yearly only: the same values as Q chunked by timestep across rivers, unsharded, not
                 bitrounded again
    zarr.json    title and license, metadata consolidated
    every array  blosc(cname="zstd", clevel=5, shuffle="shuffle") and nothing else, typesize unset

None of that is spelled out here: it all comes from rfs_spec.py, which is the source of truth the spec points at
for encodings.

Averages are left aligned, like every v3 time axis: the value at a time is the mean of the source values from that
time up to the start of the next period, i.e. hours 00..23 of a day, every day of a calendar month or calendar year.
Only complete periods are written. A partial period at either end of the record is dropped rather than averaged over
the values it has, since the spec promises every value on the axis is a full mean; a partial or missing period
anywhere in the middle is a gap in the source and stops the run.

The source is read once. Worker processes each take a block of it -- one shard, 250 rivers, or several with
``--shards-per-block`` -- so a read decompresses exactly the rivers it needs, and cascade it while it is in memory:

    daily block -> monthly sums                          -> write monthly.zarr
                -> yearly sums, from the monthly sums    -> write yearly.zarr

Sums are float64 and carried down the chain with the count of source steps behind each one, so a year is the mean of
its days rather than a mean of twelve monthly means that would weight February like March. Every day holds the same
24 hours, so a month built out of daily means is the mean of its hours, up to the rounding the daily store already
did: measured, the two agree to about 1e-6 relative and never worse than 5e-6, well under the step between
neighbouring values on the KEEP_BITS grid. Rounding the result onto that grid is what makes the difference visible,
so a monthly value built from daily can land one step from one built straight from hourly, and never further.
Q_timesteps, whose chunks span every river at one timestep, is written last from the finished monthly and yearly Q.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# the spec's layout, codec, cascade and store creation, shared with 3_build_hourly_daily_maximums.py, which writes
# the daily store out of the blocks it is already holding
from rfs_spec import (
    KEEP_BITS,
    LAYOUTS,
    LEVELS,
    RIVERS_PER_SHARD,
    bitrounder,
    cascade,
    consolidate,
    create_store,
    levels_after,
    source_level,
    store_dates,
    store_path,
    write_timesteps,
)

SOURCE_Q = 'Q'
SOURCE_TIME = 'time'
SOURCE_IDS = ('riverId', 'river_id')  # the spec name, then the name the river-route concat script writes

_WORKER = {}


def init_worker(zarr_threads: int, context: dict) -> None:
    """Per process: zarr's thread settings (a spawned pool never sees the parent's) and one handle per array."""
    zarr.config.set({'async.concurrency': zarr_threads, 'threading.max_workers': zarr_threads})
    _WORKER.update(context)
    _WORKER['source'] = zarr.open_group(context['source_path'], mode='r')[SOURCE_Q]
    _WORKER['outs'] = {name: zarr.open_array(str(store_path(Path(context['out_dir']), name) / 'Q'), mode='r+')
                       for name in context['stores']}
    _WORKER['bitround'] = bitrounder()


def write_block(job: tuple) -> tuple[int, int, int]:
    """
    One block of rivers: read their whole series out of the source once, then cascade it in memory -- each level
    summed from the level above it and written before the next one is taken.

    A block is a whole number of source shards and lines up with whole chunks (or shards) of every output, so no
    two workers ever write the same object. Levels between the source and the stores being written are summed but
    not written, which is what lets a run write yearly alone and still weight every month by its own length.
    """
    g0, g1 = job
    block = _WORKER['source'][g0:g1, :]
    n_nan = int(np.isnan(block).sum())
    sums = None
    for name in _WORKER['levels']:
        p = _WORKER['periods'][name]
        if sums is None:
            # float64: accumulating up to 8,784 float32 values in float32 would add more relative error than the
            # keepbits the result is then rounded to. The source block is released as soon as it has been summed.
            sums = np.add.reduceat(block, p['starts'], axis=1, dtype=np.float64)
            del block
        else:
            sums = np.add.reduceat(sums, p['starts'], axis=1)
        out = _WORKER['outs'].get(name)
        if out is None:
            continue
        means = (sums / p['counts'][None, :])[:, p['keep']].astype(np.float32)
        out[g0:g1, :] = _WORKER['bitround'](means)
    with open(_WORKER['progress_file'], 'a') as log:
        log.write(f'block {g0}\n')
    return g0, g1 - g0, n_nan


def verify(source_path: str, out_dir: Path, stores: list[str], n_rivers: int, sample: int) -> bool:
    """
    Recompute sampled rivers with pandas, straight from the source store, and compare with what was written: the
    same means to within the KEEP_BITS rounding, the same time axis, the same riverId, and Q_timesteps
    identical to Q.

    This checks the stores against the source they were built from, which for monthly and yearly out of daily.zarr
    means against the daily means, not the hourly values. That is the invariant worth holding: every store is the
    mean of the store above it, so a reader who averages the published daily values gets the published monthly ones.
    """
    src = zarr.open_group(source_path, mode='r')
    dates = store_dates(src[SOURCE_TIME])
    ids = src[next(k for k in SOURCE_IDS if k in src)]
    rng = np.random.default_rng(0)
    rivers = sorted(set([0, n_rivers - 1] + rng.choice(n_rivers, min(sample, n_rivers), replace=False).tolist()))
    ok = True
    # keeping N of float32's mantissa bits bounds the relative error at 2^-(N+1); 2^-N leaves room for the
    # float32 mean on top of it
    tolerance = 2.0 ** -KEEP_BITS
    for name in stores:
        group = zarr.open_group(str(store_path(out_dir, name)), mode='r')
        out_times = store_dates(group['time'])
        freq = {'D': 'D', 'M': 'MS', 'Y': 'YS'}[LAYOUTS[name]['freq']]
        for i in rivers:
            series = pd.Series(src[SOURCE_Q][i, :].astype(np.float64), index=dates)
            # the store holds only complete periods, so reindexing onto its axis drops pandas' partial ends
            expected = series.resample(freq).mean().reindex(out_times).to_numpy()
            got = group['Q'][i, :].astype(np.float64)
            err = float(np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1e-30))) if len(got) else 0.0
            same_id = int(group['riverId'][i]) == int(ids[i])
            if not (np.isfinite(err) and err <= tolerance and same_id):
                ok = False
                print(f'  MISMATCH {name} river {i}: max relative error {err:.2e}, riverId match {same_id}')
        if 'Q_timesteps' in group:
            same = np.array_equal(group['Q'][rivers, :], group['Q_timesteps'][rivers, :], equal_nan=True)
            ok &= same
            if not same:
                print(f'  MISMATCH {name}: Q_timesteps differs from Q')
        print(f'  {name}: {len(out_times):,} steps {out_times[0]} .. {out_times[-1]}, {len(rivers)} rivers checked')
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', '--hourly', dest='source',
                        default='/Users/rchales/data/rfsv3/retrospective/daily.zarr',
                        help='the store to cascade from, hourly.zarr or daily.zarr')
    parser.add_argument('--out-dir', default=None, help='where the stores go (default: beside the source store)')
    parser.add_argument('--stores', nargs='+', choices=[name for name in LEVELS if name != 'hourly'], default=None,
                        help='which stores to write (default: every level below the source)')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 8,
                        help='blocks at once; each holds one block of the source (~440 MB for 50 years x 250 '
                             'rivers of hourly, a twenty-fourth of that from daily) plus ~2x that in float64 sums')
    parser.add_argument('--shards-per-block', type=int, default=1,
                        help='source shards read per block. A daily shard is 18 MB against 440 MB of hourly, so '
                             'reading one at a time makes the dispatch, not the work, the cost of the run')
    parser.add_argument('--zarr-threads', type=int, default=4,
                        help='threads each worker decompresses its source block and compresses its outputs with')
    parser.add_argument('--verify', type=int, default=16, metavar='N',
                        help='afterwards, recompute N random rivers (plus the first and last) with pandas; 0 skips')
    parser.add_argument('--overwrite', action='store_true',
                        help='start over, discarding the output stores and the progress log')
    parser.add_argument('--plan', action='store_true', help='print the layout and exit without writing anything')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.source).parent
    # beside the output directory rather than inside it, so nothing in the served tree is a pipeline file
    progress_file = out_dir.parent / f'{out_dir.name}.resample.progress'

    source_group = zarr.open_group(args.source, mode='r')
    source = source_group[SOURCE_Q]
    id_name = next((k for k in SOURCE_IDS if k in source_group), None)
    if id_name is None:
        raise SystemExit(f'{args.source} has no {" or ".join(SOURCE_IDS)} array')
    river_ids = source_group[id_name][:]
    if river_ids.min() < 0 or river_ids.max() >= 2 ** 31:
        raise SystemExit('river ids do not fit the spec\'s int32 riverId')
    dates = store_dates(source_group[SOURCE_TIME])
    n_rivers, n_time = source.shape
    if len(dates) != n_time or len(river_ids) != n_rivers:
        raise SystemExit(f'{args.source}: Q is {source.shape} but time has {len(dates)} and {id_name} {len(river_ids)}')

    # What the source holds decides what can be built from it, and a store is never rebuilt from itself: given
    # daily.zarr, which 3_build_hourly_daily_maximums.py now writes alongside the hourly store, this writes monthly and
    # yearly and never reads an hourly value.
    level = source_level(dates)
    stores = levels_after(level, args.stores)
    if not stores:
        raise SystemExit(f'a {level} source has nothing left to build')

    # Blocks follow the source store's shards (or chunks, unsharded), so one read never decodes a neighbour's rivers.
    block = (source.shards or source.chunks)[0] * args.shards_per_block
    for name in stores:
        layout = LAYOUTS[name]
        out_width = RIVERS_PER_SHARD
        if block % out_width:
            raise SystemExit(f'{name} writes objects {out_width} rivers wide, which does not divide the source block '
                             f'of {block}: two workers would write the same object')
    levels = cascade(dates, stores)
    period_info = {name: levels[name] for name in stores}

    print(f'{args.source}: {n_rivers:,} rivers x {n_time:,} {level} steps, {dates[0]} .. {dates[-1]}, read '
          f'{block} rivers a block')
    for name in stores:
        p, layout = period_info[name], LAYOUTS[name]
        n = len(p['times'])
        dropped = f', {p["dropped"]} partial dropped' if p['dropped'] else ''
        print(f'  {name:<8} {n:>7,} steps {p["times"][0]:%Y-%m-%d} .. {p["times"][-1]:%Y-%m-%d}{dropped} -> '
              f'{store_path(out_dir, name)}  {n * n_rivers * 4 / 1e9:,.1f} GB uncompressed')
    if level != 'hourly' and args.shards_per_block == 1 and -(-n_rivers // block) > 1_000:
        print(f'  hint: a block is {n_time * block * 4 / 1e6:.0f} MB and there are {-(-n_rivers // block):,} of '
              f'them, so most of this run is dispatch -- try --shards-per-block 20')
    if args.plan:
        raise SystemExit(0)

    finished_blocks, finished_timesteps = set(), set()
    resume = progress_file.exists() and not args.overwrite
    # One log serves every run that writes into this directory, so it records what it was written by: resuming a
    # monthly-and-yearly-from-daily run against a log left by a from-hourly run would skip blocks nothing wrote.
    run_id = f'{Path(args.source).name} -> {",".join(stores)} in blocks of {block}'
    if resume:
        recorded = None
        for line in progress_file.read_text().splitlines():
            kind, _, value = line.partition(' ')
            if kind == 'block' and value.isdigit():
                finished_blocks.add(int(value))
            elif kind == 'timesteps':
                finished_timesteps.add(value)
            elif kind == 'run':
                recorded = value
        if recorded is not None and recorded != run_id:
            raise SystemExit(f'{progress_file.name} records "{recorded}" but this run is "{run_id}" -- match it, or '
                             f'pass --overwrite to start over')
        for name in stores:
            path = store_path(out_dir, name)
            if not path.exists():
                raise SystemExit(f'{progress_file.name} records progress but {path} is missing -- pass --overwrite')
            q = zarr.open_array(str(path / 'Q'), mode='r')
            if q.shape != (n_rivers, len(period_info[name]['times'])):
                raise SystemExit(f'{path} is {q.shape}, this run wants '
                                 f'{(n_rivers, len(period_info[name]["times"]))} -- it cannot be resumed')
        print(f'resuming: {len(finished_blocks):,} of {-(-n_rivers // block):,} blocks already written')
    else:
        progress_file.unlink(missing_ok=True)
        progress_file.write_text(f'run {run_id}\n')
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in stores:
            create_store(store_path(out_dir, name), name, period_info[name], river_ids)

    jobs = [(g0, min(g0 + block, n_rivers)) for g0 in range(0, n_rivers, block) if g0 not in finished_blocks]
    context = {'source_path': args.source, 'out_dir': str(out_dir), 'stores': stores,
               'progress_file': str(progress_file),
               # every level of the chain, in order: the ones between the source and the stores being written are
               # summed on the way past even though nothing writes them
               'levels': list(levels),
               'periods': {name: {k: v for k, v in p.items() if k in ('starts', 'counts', 'keep')}
                           for name, p in levels.items()}}
    print(f'{args.workers} workers x {args.zarr_threads} zarr threads, {len(jobs):,} blocks to write, '
          f'~{args.workers * n_time * block * 12 / 1e9:,.0f} GB of worker memory at peak')

    started = time.time()
    done = nans = 0
    todo = sum(g1 - g0 for g0, g1 in jobs)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(args.zarr_threads, context)) as pool:
            for future in as_completed([pool.submit(write_block, job) for job in jobs]):
                _g0, width, n_nan = future.result()
                done += width
                nans += n_nan
                elapsed = (time.time() - started) / 60
                # one line a percent, not one a block: a global run is ~20,000 blocks
                if done == todo or done * 100 // todo != (done - width) * 100 // todo:
                    print(f'{done:,}/{todo:,} rivers ({done * 100 // todo}%) in {elapsed:.1f} min, '
                          f'{elapsed / done * (todo - done):.1f} min left', flush=True)
    if nans:
        # the spec treats NaN as a failure to investigate, never an expected absence
        print(f'WARNING: the source holds {nans:,} NaN values; the periods containing them are NaN too')

    for name in stores:
        if LAYOUTS[name]['timesteps'] and name not in finished_timesteps:
            t0 = time.time()
            write_timesteps(store_path(out_dir, name), args.workers)
            with open(progress_file, 'a') as log:
                log.write(f'timesteps {name}\n')
            print(f'{name} Q_timesteps written in {(time.time() - t0) / 60:.1f} min', flush=True)

    for name in stores:
        consolidate(store_path(out_dir, name))
    print(f'wrote {", ".join(stores)} in {(time.time() - started) / 60:.1f} min')
    if args.verify:
        print(f'verifying against {Path(args.source).name}:')
        if not verify(args.source, out_dir, stores, n_rivers, args.verify):
            raise SystemExit('verification FAILED')
        print('verify: OK')
