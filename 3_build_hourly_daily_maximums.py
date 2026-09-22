"""
Concatenate the per-region, per-quarter discharge stores written by ``2_rfs_v3_retro_router.py`` into one zarr store,
and write the daily means out of the same blocks.

Regions are concatenated along ``river_id`` in ascending region number and the quarterly stores of each region are
concatenated along ``time``. The output is chunked over every time step (time chunk of -1) and one river, with 250
chunks packed into each shard, so one file on disk holds 250 rivers.

A pass already holds one shard of rivers for every time step in memory -- the exact working set a resampling pass
would rebuild -- so it also reduces those hours into two more stores beside the hourly one: ``daily.zarr``, the
daily means, and ``maximums.zarr``, the annual maximum series. That is
the whole reason to do it here: the daily means cost one more pass over a buffer that is already in RAM, where
``4_build_monthly_yearly.py`` had to read the whole finished hourly store back to get them -- 6.0 TB
uncompressed for a 1990-2024 global run, 8.6 TB for a 50 year one. Daily is the only level worth doing this way;
monthly and yearly cascade from daily.zarr afterwards, and reading daily back costs a twenty-fourth of reading
hourly:

    quarter stores  ->  hourly.zarr + daily.zarr + maximums.zarr     (this script)
    daily.zarr      ->  monthly.zarr + yearly.zarr                   (4_build_monthly_yearly.py)

The annual maxima have to be taken here whatever the daily means cost, because a maximum does not cascade: the
highest hourly discharge of a year is not recoverable from daily means, which have already flattened the peak. Each
value is the largest hourly value in a complete calendar year, the same number the hourly store holds -- rounding to
15 keepbits leaves a value the router already rounded to 13 exactly as it was -- so a peak in maximums.zarr can be
found in hourly.zarr unchanged.

The daily values are summed from the buffer that is written to hourly.zarr, and the router already bitrounded the
quarters to 13 keepbits, so daily.zarr holds exactly what resampling the published hourly store would give -- it is
bit for bit the same store, which is what makes doing it here a move rather than a change.

It is not free: this run now also bitrounds and compresses the daily means at zstd level 5 -- 250 GB uncompressed
for 1990-2024 -- which the resampling pass used to do. What it does not do is read the hourly store back to get
them.

What this run concatenates -- the quarters every region shares, the time axis they make, and where each region
lands on the river axis -- is worked out once by opening every source store, and written beside the output as
``<out>.plan.json`` with its two large arrays in ``<out>.plan.npz``. A resume reads that back instead of rescanning,
which is what lets the run delete as it goes: each region's quarter stores are removed as soon as every shard
covering its rivers has been written, rather than all of them surviving until the run ends. Since hourly.zarr
chunks span the whole time axis, every pass reads all of a region's quarters at once and no single quarter is ever
finished on its own -- a region is the unit that becomes free, and it becomes free all at once. That roughly halves
the peak disk of a build, and costs the ability to rebuild without routing again.

``--skip-daily`` and ``--skip-maximums`` each leave one of them out. Both follow the RFS v3 spec, so their layout,
codec and metadata come from ``rfs_spec.py``; only complete periods are written, and a partial day or year at either
end of the record is dropped rather than reduced over the part of it the record has.
"""

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

import rfs_spec

# What the quarter stores written by 2_rfs_v3_retro_router.py call their arrays. Those are intermediates, never
# published, so they keep river-route's names; every store written here is published and takes its names, dtypes,
# codec, chunking and keepbits from rfs_spec, which is the spec's source of truth for encodings.
#
# A source array is (river, time) -- the layout river-route's kernels produce and its zarr_writer stores without a
# transpose -- while every store written here is (time, riverId), which is what the spec fixes. The transpose is
# therefore done on the way in, on the block each read returns rather than on a whole quarter.
SRC_Q = 'Q'
SRC_RIVER_IDS = ('riverId', 'river_id')

# One task assembles a whole shard of rivers for every time step, so a worker holds n_time x 250 float32 in memory:
# about 440 MB for a 50 year hourly run, plus what zarr buffers on top. That is the only real limit on --workers.
BLOCK_RIVERS = rfs_spec.RIVERS_PER_SHARD

_ARRAY_CACHE = {}
_WORKER = {}


def init_worker(zarr_threads: int, context: dict) -> None:
    """
    Give each worker its zarr thread settings and everything about the run that does not change block to block.

    A pool started by spawn -- the default on macOS -- shares no state with the parent, so the parent's
    ``zarr.config.set`` never reaches the processes that do the writing. Setting it here is what makes
    ``--zarr-threads`` mean anything.

    Holding the run's context here rather than in each job is what keeps dispatch cheap. Sent per block, the list
    of 200 source paths for every region a block touches is tens of KB pickled, 19,669 times over; sent once per
    worker it is nothing, and a job becomes an offset and a couple of integer ranges.
    """
    zarr.config.set({'async.concurrency': zarr_threads, 'threading.max_workers': zarr_threads})
    _WORKER.update(context)
    _WORKER['regions_root'] = Path(context['regions_root'])
    # Opened once per worker: reopening it per block re-reads the store metadata 19,669 times for nothing.
    _WORKER['out'] = zarr.open_array(str(Path(context['out_path']) / 'Q'), mode='r+')
    daily_path, max_path = context.get('daily_path'), context.get('maximums_path')
    _WORKER['daily'] = zarr.open_array(str(Path(daily_path) / 'Q'), mode='r+') if daily_path else None
    # maximums.zarr holds one array per source series rather than a 'Q': the annual maximum of the hourly values and
    # the annual maximum of the daily means, which return-periods.zarr fits a distribution against separately.
    _WORKER['max_hourly'] = zarr.open_array(str(Path(max_path) / 'hourly'), mode='r+') if max_path else None
    _WORKER['max_daily'] = zarr.open_array(str(Path(max_path) / 'daily'), mode='r+') if max_path else None
    # the daily means are what the annual daily maxima are reduced from, so they are computed for either store
    _WORKER['need_means'] = bool(daily_path or max_path)
    _WORKER['bitround'] = rfs_spec.bitrounder()


def open_source(store: Path, name: str):
    """Open one source array, reusing the handle for the rest of this process's blocks."""
    key = (str(store), name)
    if key not in _ARRAY_CACHE:
        _ARRAY_CACHE[key] = zarr.open_group(str(store), mode='r')[name]
    return _ARRAY_CACHE[key]


def is_complete(store: Path) -> bool:
    """
    A store is finished only once its metadata is consolidated, which the writer does after the last array is
    written. Stores still being written by a running simulation are half built -- discharge but no time axis -- so
    they are skipped rather than read.
    """
    try:
        return json.loads((store / 'zarr.json').read_text()).get('consolidated_metadata') is not None
    except (OSError, ValueError):
        return False


def store_dates(store: Path) -> pd.DatetimeIndex:
    """One store's time axis, as naive UTC: an aware index would carry its zone into the period cascade."""
    return rfs_spec.store_dates(zarr.open_group(str(store), mode='r')['time'])


def write_block(job: tuple) -> tuple[int, int]:
    """
    Fill one block of rivers -- one shard wide -- for every time step and write it in a single pass.

    Each pass is written by exactly one task and covers whole shards, so no two tasks ever touch the same shard
    file. A pass may span more than one region when a region boundary falls inside it, which is why the pieces are
    passed in rather than derived from a single region.

    With ``--slab`` a pass covers several neighbouring shards and reads them from each quarter in one call. That is
    worth doing because a 250 river read at an arbitrary offset drags in 1.39 chunks for every one it needs -- the
    source is chunked 100 rivers wide and region boundaries are not multiples of either -- and because 200 small
    reads pay the per call cost 200 times. Measured on a real quarter, going from 250 to 1000 rivers a pass cut the
    read time per river by 2.17x. The shards written are the same either way.

    Source and destination are both (river, time), so a read drops straight into the buffer: no transpose anywhere
    between the router and a published store.
    """
    start, end, blocks, pieces = job
    n_time, read_threads = _WORKER['n_time'], _WORKER['read_threads']
    width = end - start
    buf = np.empty((width, n_time), dtype='float32')

    # Every source array is opened first, on this thread: the handle cache is a plain dict and the reads below run
    # concurrently. After the first pass the opens are all cache hits anyway.
    reads, row = [], 0
    for region, l0, l1 in pieces:
        col = 0
        for stem in _WORKER['stems']:
            flow = open_source(_WORKER['regions_root'] / region / f'{stem}.zarr', SRC_Q)
            reads.append((row, flow, l0, l1, col))
            col += flow.shape[1]  # (river, time): a quarter's steps are the second axis
        row += l1 - l0

    def fill(task) -> None:
        row, flow, l0, l1, col = task
        buf[row:row + (l1 - l0), col:col + flow.shape[1]] = flow[l0:l1, :]

    # A pass reads 200 quarters one after another otherwise, and each one is a small decompress that leaves the
    # worker waiting on the disk. Threads overlap them; blosc releases the GIL while it works.
    if read_threads > 1:
        with ThreadPoolExecutor(read_threads) as pool:
            list(pool.map(fill, reads))
    else:
        for task in reads:
            fill(task)

    # Every published discharge array sits on the one keepbits grid rfs_spec fixes. The router already rounded
    # the quarters to it, so this is a no-op on a normal run and a guarantee on any other -- and it means the daily
    # means and annual maxima below are reduced from exactly the values hourly.zarr now holds.
    buf = _WORKER['bitround'](buf)
    _WORKER['out'][start:end, :] = buf

    # The days are summed out of the buffer that was just written, while it is still in memory. Those are the same
    # values hourly.zarr now holds, so daily.zarr is exactly what resampling the published hourly store gives,
    # without the 8.6 TB read that resampling it would cost. The sums are float64 because adding 24 float32 values
    # in float32 would introduce more relative error than the keepbits the mean is then rounded to.
    means = None
    if _WORKER['need_means']:
        period = _WORKER['daily_period']
        sums = np.add.reduceat(buf, period['starts'], axis=1, dtype=np.float64)
        means = _WORKER['bitround']((sums / period['counts'][None, :])[:, period['keep']].astype(np.float32))
        if _WORKER['daily'] is not None:
            _WORKER['daily'][start:end, :] = means

    # The annual maxima come off the same buffers, and the hourly one only ever off an hourly source: the highest
    # hourly discharge of a year cannot be recovered from any store that has already averaged those hours together.
    # Both are picked out of values that are already on the keepbits grid, so each maximum is bit for bit the value
    # it came from and can be found unchanged in hourly.zarr or daily.zarr.
    if _WORKER['max_hourly'] is not None:
        period = _WORKER['maximums_period']
        _WORKER['max_hourly'][start:end, :] = np.maximum.reduceat(buf, period['starts'], axis=1)[:, period['keep']]
        period = _WORKER['maximums_daily_period']
        _WORKER['max_daily'][start:end, :] = np.maximum.reduceat(means, period['starts'], axis=1)[:, period['keep']]

    # Logged one line per shard, never per pass, so the log means the same thing at any --slab: a run started at one
    # slab resumes at another. Recorded only once the shards are on disk, so a resume never counts work that was
    # still in flight. Appending a short line is atomic on POSIX, which is what lets every worker share the one file
    # without coordinating.
    with open(_WORKER['progress_file'], 'a') as log:
        log.write(''.join(f'{g0}\n' for g0 in blocks))
    return start, width


PLAN_VERSION = 1


def plan_paths(out) -> tuple[Path, Path]:
    """The plan lives beside the store, never inside it, so nothing in it can confuse a zarr reader."""
    return Path(f'{out}.plan.json'), Path(f'{out}.plan.npz')


def scan_sources(in_dir: Path, common_times: bool) -> dict:
    """
    Open every quarter of every region and work out what this run concatenates: which quarters every region shares,
    the time axis they make, and where each region lands on the river axis.

    This is the only code that reads the sources' metadata, and it is what a resume must not have to repeat once
    ``--delete-consumed`` has started removing regions. Its result is the plan, written to disk by ``write_plan``.
    """
    region_dirs = sorted(in_dir.glob('region=*'), key=lambda d: int(d.name.split('=')[1]))
    if not region_dirs:
        raise SystemExit(f'no region=* directories in {in_dir}')

    # Quarterly stems sort chronologically, so the lexical sort is the time order.
    stems = {d.name: sorted(s.name[:-len('.zarr')] for s in d.glob('*.zarr') if is_complete(s)) for d in region_dirs}
    unfinished = sum(len(list(d.glob('*.zarr'))) - len(stems[d.name]) for d in region_dirs)
    if unfinished:
        print(f'skipping {unfinished} stores that are still being written')
    if not all(stems.values()):
        raise SystemExit(f'no finished stores in {[r for r, v in stems.items() if not v]}')
    shared = sorted(set.intersection(*(set(v) for v in stems.values())))
    ragged = {region: sorted(set(v) - set(shared)) for region, v in stems.items() if set(v) != set(shared)}
    if ragged:
        for region, extra in ragged.items():
            print(f'{region} has {len(extra)} quarters the other regions do not: {extra[0]} .. {extra[-1]}')
        if not common_times:
            raise SystemExit('regions cover different quarters -- pass --common-times to use the shared ones only')
        print(f'using the {len(shared)} quarters every region has: {shared[0]} .. {shared[-1]}')

    # Time comes from the first region; every other region's stores are checked against it below so a misaligned
    # run is caught here rather than silently concatenated into a store whose time axis does not match its discharge.
    dates = pd.DatetimeIndex(np.concatenate([store_dates(region_dirs[0] / f'{s}.zarr') for s in shared]))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise SystemExit('the concatenated time axis is not strictly increasing -- check for overlapping quarters')
    # (river, time), so a quarter's step count is the second axis of its discharge array
    steps_per_stem = [zarr.open_group(str(region_dirs[0] / f'{s}.zarr'), mode='r')[SRC_Q].shape[1] for s in shared]
    epochs = [str(zarr.open_group(str(region_dirs[0] / f'{s}.zarr'), mode='r')['time'].attrs['units']) for s in shared]

    river_ids, regions, offset = [], [], 0
    for region_dir in region_dirs:
        source = zarr.open_group(str(region_dir / f'{shared[0]}.zarr'), mode='r')
        id_name = next((k for k in SRC_RIVER_IDS if k in source), None)
        if id_name is None:
            raise SystemExit(f'{region_dir.name} has no {" or ".join(SRC_RIVER_IDS)} array')
        ids = source[id_name][:]
        for stem, steps, epoch in zip(shared, steps_per_stem, epochs, strict=True):
            group = zarr.open_group(str(region_dir / f'{stem}.zarr'), mode='r')
            if group[SRC_Q].shape != (ids.size, steps):
                raise SystemExit(f'{region_dir.name}/{stem}.zarr is {group[SRC_Q].shape}, expected {(ids.size, steps)}')
            if str(group['time'].attrs['units']) != epoch:
                raise SystemExit(f'{region_dir.name}/{stem}.zarr starts at a different time than {region_dirs[0].name}')
        river_ids.append(ids)
        regions.append({'name': region_dir.name, 'start': offset, 'end': offset + int(ids.size)})
        offset += ids.size

    return {'version': PLAN_VERSION, 'in_dir': str(in_dir), 'stems': shared, 'steps_per_stem': steps_per_stem,
            'epochs': epochs, 'n_time': int(sum(steps_per_stem)), 'regions': regions,
            'block_rivers': BLOCK_RIVERS, 'keepbits': rfs_spec.KEEP_BITS,
            'dates': dates, 'river_ids': np.concatenate(river_ids)}


def write_plan(plan: dict, json_path: Path, npz_path: Path) -> None:
    """The plan as JSON, with the two arrays too big for it beside it: the time axis and the river ids."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    document = {k: v for k, v in plan.items() if k not in ('dates', 'river_ids')}
    document['written'] = datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    json_path.write_text(json.dumps(document, indent=2) + '\n')
    np.savez(npz_path, dates=plan['dates'].to_numpy().astype('datetime64[ns]').view(np.int64),
             river_ids=plan['river_ids'])


def read_plan(json_path: Path, npz_path: Path) -> dict:
    """Read a plan back without touching a single source store."""
    plan = json.loads(json_path.read_text())
    if plan.get('version') != PLAN_VERSION:
        raise SystemExit(f'{json_path.name} is plan version {plan.get("version")}, this is version {PLAN_VERSION} '
                         f'-- pass --overwrite to start over')
    if plan['block_rivers'] != BLOCK_RIVERS:
        raise SystemExit(f'{json_path.name} was written in blocks of {plan["block_rivers"]}, this run uses '
                         f'{BLOCK_RIVERS} -- the finished shards are not the blocks this would skip')
    with np.load(npz_path) as arrays:
        plan['dates'] = pd.DatetimeIndex(arrays['dates'].astype('datetime64[ns]'))
        plan['river_ids'] = arrays['river_ids']
    if len(plan['dates']) != plan['n_time'] or plan['river_ids'].size != plan['regions'][-1]['end']:
        raise SystemExit(f'{json_path.name} and {npz_path.name} disagree -- pass --overwrite to start over')
    return plan


def region_blocks(start: int, end: int, block: int = BLOCK_RIVERS) -> range:
    """Every block that holds any of the rivers in [start, end), including one shared with each neighbour."""
    return range((start // block) * block, end, block)


def drop_region(path: Path) -> tuple[str, int]:
    """Measure and delete one region's quarter stores. Returns what it freed."""
    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    shutil.rmtree(path, ignore_errors=True)
    return path.name, size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-dir', default='/Users/rchales/data/discharge_zarr')
    parser.add_argument('--out-dir', default='/Users/rchales/data/rfsv3/retrospective',
                        help='directory that hourly.zarr, daily.zarr and maximums.zarr are written into')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 8) // 2),
                        help='blocks assembled at once, each holding ~1 shard')
    parser.add_argument('--zarr-threads', type=int, default=2, help='threads each worker compresses its shard with')
    parser.add_argument('--read-threads', type=int, default=4, help='quarters each worker reads at once')
    parser.add_argument('--slab', type=int, default=4,
                        help='shards read in one pass; higher trades worker memory for less overread')
    parser.add_argument(
        '--common-times',
        action='store_true',
        help='use only the quarters present in every region instead of requiring every region to have the same ones',
    )
    parser.add_argument('--skip-daily', action='store_true',
                        help='write only the hourly store, leaving the daily means to a resampling pass')
    parser.add_argument('--skip-maximums', action='store_true',
                        help='do not write the annual maximum series; nothing downstream can rebuild it from the '
                             'averaged stores, only from the hourly one')
    parser.add_argument('--overwrite', action='store_true',
                        help='start over, discarding whatever hourly store is already in --out-dir')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out = out_dir / 'hourly.zarr'
    in_dir = Path(args.in_dir)
    progress_file = Path(f'{out}.progress')
    plan_json, plan_npz = plan_paths(out)
    resume = out.exists() and not args.overwrite

    if resume and plan_json.exists():
        plan = read_plan(plan_json, plan_npz)
        print(f'plan: {plan_json.name} written {plan["written"]}, {len(plan["regions"])} regions, '
              f'{len(plan["stems"])} quarters -- the sources were not rescanned')
    else:
        plan = scan_sources(in_dir, args.common_times)

    regions = plan['regions']  # in river order, each {'name', 'start', 'end'}
    offsets = {r['name']: (r['start'], r['end']) for r in regions}
    shared, dates, n_time = plan['stems'], plan['dates'], plan['n_time']
    river_ids = plan['river_ids']
    n_rivers = river_ids.size

    # Both are reduced straight off the hourly axis rather than through one another: a day is summed from its
    # hours, and a year's maximum is taken over its hours, never over daily means that have flattened the peak.
    extras = {name: out_dir / f'{name}.zarr'
              for name, skip in (('daily', args.skip_daily), ('maximums', args.skip_maximums)) if not skip}
    if river_ids.min() < 0 or river_ids.max() >= 2 ** 31:
        raise SystemExit("river ids do not fit the spec's int32 riverId")
    periods = {}
    if extras:
        # The daily means are needed for either store: daily.zarr is written from them, and maximums.zarr takes the
        # annual maximum of the daily series from them too.
        periods['daily'] = rfs_spec.level_periods(dates, 'daily')
        periods['maximums'] = rfs_spec.level_periods(dates, 'maximums')
        # the same year boundaries again, indexed into the daily axis rather than the hourly one
        periods['maximums_daily'] = rfs_spec.level_periods(periods['daily']['times'], 'maximums')
        if not periods['maximums']['times'].equals(periods['maximums_daily']['times']):
            raise SystemExit('the years complete in hours are not the years complete in days -- the source has a gap')
        # Every store is sharded RIVERS_PER_SHARD wide and a pass covers whole shards of the hourly store, so no two
        # passes ever write the same shard file. That is what lets them all be written in parallel.
        if BLOCK_RIVERS % rfs_spec.RIVERS_PER_SHARD:
            raise SystemExit(f'a {BLOCK_RIVERS} river block is not a whole number of '
                             f'{rfs_spec.RIVERS_PER_SHARD} river shards')

    print(f'{len(regions)} regions, {n_rivers:,} rivers, {len(shared)} quarters, {n_time:,} time steps')
    print(f'  {dates[0]} .. {dates[-1]}')
    t_chunk = rfs_spec.time_chunk(dates)
    print(f'  chunks ({t_chunk:,}, {rfs_spec.RIVERS_PER_CHUNK}), shards ({t_chunk:,}, {BLOCK_RIVERS}), '
          f'{-(-n_time // t_chunk)} chunk(s) on the time axis')
    print(f'  {n_time * n_rivers * 4 / 1e12:.2f} TB uncompressed, {-(-n_rivers // BLOCK_RIVERS):,} shard files')
    for name, path in extras.items():
        p = periods[name]
        dropped = f', {p["dropped"]} partial dropped' if p['dropped'] else ''
        print(f'  {name:<8} {len(p["times"]):>7,} steps {p["times"][0]:%Y-%m-%d} .. {p["times"][-1]:%Y-%m-%d}'
              f'{dropped} -> {path}  {len(p["times"]) * n_rivers * 4 / 1e9:,.1f} GB uncompressed')
    for name, skipped in (('daily', args.skip_daily), ('maximums', args.skip_maximums)):
        if skipped:
            print(f'  no {name} store: --skip-{name}')
    for region in regions:
        print(f'    {region["name"]} {region["end"] - region["start"]:>9,} rivers  '
              f'river_id[{region["start"]:,}:{region["end"]:,}]')
    print(f'  each region under {in_dir} is deleted once every shard covering its rivers is written, so this '
          f'build\n    cannot be started over once it is under way -- that would mean routing again.')

    # The progress log lives beside the store rather than inside it, so nothing in it can confuse a zarr reader.
    # Picking up where a previous run stopped is the default: a run this long should never be restarted by accident,
    # and starting over destroys hours of finished shards, so that is what takes a flag.
    outputs_line = ' '.join(f'{name}={extras.get(name, "none")}' for name in ('daily', 'maximums'))
    finished = set()
    if resume:
        if not progress_file.exists():
            raise SystemExit(f'{out} already exists but {progress_file.name} does not, so there is no record '
                             f'of which shards are finished -- pass --overwrite to start over')
        log_text = progress_file.read_text()
        finished = {int(line) for line in log_text.split() if line.isdigit()}
        finished_timesteps = {line.split(' ', 1)[1] for line in log_text.splitlines()
                              if line.startswith('timesteps ')}
        # The finished shards were written by a run that wrote some set of these stores. Resuming with a different
        # set would leave the ones it now writes holding fill value for every shard that is being skipped, with
        # nothing to say which, so the log records what it was built with and a mismatch stops the run.
        recorded = next((line.split(' ', 1)[1] for line in log_text.splitlines() if line.startswith('outputs ')), '')
        if recorded != outputs_line:
            raise SystemExit(f'{progress_file.name} records {len(finished):,} shards written with "{recorded}", but '
                             f'this run wants "{outputs_line}" -- match it, or pass --overwrite to start over')
        for name, path in extras.items():
            if not path.exists():
                raise SystemExit(f'{progress_file.name} records {name} shards but {path} is missing -- pass '
                                 f'--overwrite')
            if rfs_spec.store_shape(path, name) != (n_rivers, len(periods[name]['times'])):
                raise SystemExit(f'{path} is {rfs_spec.store_shape(path, name)}, this run wants '
                                 f'{(n_rivers, len(periods[name]["times"]))} -- it cannot be resumed')
        print(f'resuming: {len(finished):,} blocks already finished, {len(finished) * BLOCK_RIVERS:,} rivers')
    else:
        # A log left behind by a store that is gone would skip blocks that were never written into the new one.
        finished_timesteps = set()
        progress_file.unlink(missing_ok=True)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(f'outputs {outputs_line}\n')
        for name, path in extras.items():
            rfs_spec.create_store(path, name, periods[name], river_ids)
    # Written once the stores it describes exist, so a plan on disk always refers to a real build. A resume that had
    # to rescan writes one too, which upgrades a run started before plans existed.
    if not resume or not plan_json.exists():
        write_plan(plan, plan_json, plan_npz)

    # Blocks run the full river axis a shard at a time, so each one lands on whole shards no matter where the region
    # boundaries fall. Passes group neighbouring unfinished blocks so one read covers several shards; only adjacent
    # blocks are grouped, so the gaps a resume leaves behind never widen a read over shards that are already done.
    todo_blocks = [g0 for g0 in range(0, n_rivers, BLOCK_RIVERS) if g0 not in finished]
    runs, jobs = [], []
    for g0 in todo_blocks:
        if runs and runs[-1][-1] + BLOCK_RIVERS == g0 and len(runs[-1]) < args.slab:
            runs[-1].append(g0)
        else:
            runs.append([g0])
    for blocks in runs:
        start, end = blocks[0], min(blocks[-1] + BLOCK_RIVERS, n_rivers)
        pieces = []
        for region in regions:
            first, last = region['start'], region['end']
            if last <= start or first >= end:
                continue
            pieces.append((region['name'], max(start, first) - first, min(end, last) - first))
        jobs.append((start, end, blocks, pieces))

    remaining = jobs
    todo = sum(end - start for start, end, _, _ in remaining)
    todo_shards = sum(len(blocks) for _, _, blocks, _ in remaining)
    # the float64 daily sums are a twelfth of the buffer they are summed from, and the annual maxima a rounding
    # error next to it; both live only while a pass runs
    pass_bytes = n_time * 4 + sum(len(periods[name]['times']) * (8 if name == 'daily' else 4) for name in extras)
    print(f'{args.workers} workers x ({args.zarr_threads} write + {args.read_threads} read) threads on '
          f'{os.cpu_count()} cores, {len(remaining):,} passes of up to {args.slab} shards, '
          f'{args.workers * args.slab * BLOCK_RIVERS * pass_bytes / 1e9:.1f} GB of pass buffers')

    # Q_timesteps is written once every shard is on disk, so a run that died between the last shard and that write
    # has blocks left to do even though no block is left to write.
    pending_timesteps = [name for name in extras
                         if rfs_spec.LAYOUTS[name]['timesteps'] and name not in finished_timesteps]
    if resume and not remaining and not pending_timesteps:
        raise SystemExit(f'{out} is already complete -- pass --overwrite to build it again')

    # A region's quarters stop being needed the moment every block covering its rivers is written. Not one quarter
    # at a time: hourly.zarr chunks span the whole time axis, so every pass reads all of a region's quarters at
    # once and no single quarter is ever finished on its own. A region boundary falls inside a block 46 times out
    # of 47, so a block counts against every region it touches and the shared one has to land before either
    # neighbour is released.
    block_regions = defaultdict(list)
    outstanding = {}
    for region in regions:
        covering = list(region_blocks(region['start'], region['end']))
        outstanding[region['name']] = sum(1 for g0 in covering if g0 not in finished)
        for g0 in covering:
            block_regions[g0].append(region['name'])

    # Deleting a big region is tens of thousands of files, so it runs on its own threads rather than stalling the
    # dispatch loop. Workers hold open handles to the stores in _ARRAY_CACHE; on POSIX unlinking underneath them is
    # harmless, and by construction no job asks for a released region again.
    delete_pool = ThreadPoolExecutor(2, thread_name_prefix='delete')
    delete_futures = []

    def release(region_name: str) -> None:
        path = in_dir / region_name
        if path.exists():
            delete_futures.append(delete_pool.submit(drop_region, path))

    def collect_released() -> int:
        """Report the regions whose deletion has finished, and return the bytes they freed."""
        freed = 0
        for future in [f for f in delete_futures if f.done()]:
            delete_futures.remove(future)
            name, size = future.result()
            freed += size
            print(f'released {name}: {size / 1e9:,.1f} GB of quarters deleted', flush=True)
        return freed

    # Regions a previous run finished without deleting, so a resume with the flag on catches up rather than
    # starting from wherever it left off.
    for name, left in outstanding.items():
        if left == 0:
            release(name)

    with zarr.config.set({'async.concurrency': args.zarr_threads, 'threading.max_workers': args.zarr_threads}):
        if resume:
            # The layout has to be the one the finished shards were written into, or the blocks being skipped are
            # not the blocks already on disk. A new quarter finishing between runs changes n_time and is caught here.
            group = zarr.open_group(str(out), mode='r+')
            existing = group['Q']
            expected_shards = (BLOCK_RIVERS, rfs_spec.time_chunk(dates))
            if existing.shape != (n_rivers, n_time) or existing.shards != expected_shards:
                raise SystemExit(f'{out} is {existing.shape} in shards of {existing.shards}, but this run wants '
                                 f'{(n_rivers, n_time)} in shards of {expected_shards} -- it cannot be resumed')
            print(f'appending to the existing store, {todo_shards:,} shards left')
        else:
            # hourly.zarr is a published product, so it is laid out by the same code as every other one: riverId and
            # time as int32, hours since the first step, blosc, NaN fill, one river a chunk in shards of 250.
            rfs_spec.create_store(out, 'hourly', {'times': dates}, river_ids)

        context = {'out_path': str(out), 'regions_root': str(in_dir), 'stems': shared,
                   'n_time': n_time, 'read_threads': args.read_threads, 'progress_file': str(progress_file)}
        for name, path in extras.items():
            context[f'{name}_path'] = str(path)
        # the times are not sent: a worker writes into the store the parent already laid out
        for name in ('daily', 'maximums', 'maximums_daily') if extras else ():
            context[f'{name}_period'] = {k: periods[name][k] for k in ('starts', 'counts', 'keep')}

        started = time.time()
        done = freed = 0
        if remaining:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                     initargs=(args.zarr_threads, context)) as pool:
                # as_completed rather than map: map hands results back in the order they were submitted, so a block
                # that finishes early waits for every block ahead of it and progress arrives in batches the width of
                # the pool instead of one block at a time. The work is dispatched the same either way; only the
                # reporting changes.
                jobs_by_future = {pool.submit(write_block, job): job for job in remaining}
                for future in as_completed(jobs_by_future):
                    _g0, width = future.result()
                    done += width
                    # the shards are on disk and logged by the time the future returns, so a region released here
                    # is one nothing will read again even if the run dies on the next block
                    for g0 in jobs_by_future[future][2]:
                        for region_name in block_regions[g0]:
                            outstanding[region_name] -= 1
                            if outstanding[region_name] == 0:
                                release(region_name)
                    freed += collect_released()
                    elapsed = (time.time() - started) / 60
                    print(f'{done:,}/{todo:,} rivers in {elapsed:.1f} min, '
                          f'{elapsed / done * (todo - done):.1f} min left', flush=True)
        # Q_timesteps spans every river in a chunk, so it can only be written once every block has landed.
        for name in pending_timesteps:
            t0 = time.time()
            rfs_spec.write_timesteps(extras[name], args.workers)
            with open(progress_file, 'a') as log:
                log.write(f'timesteps {name}\n')
            print(f'{name} Q_timesteps written in {(time.time() - t0) / 60:.1f} min', flush=True)
        rfs_spec.consolidate(out)
        for path in extras.values():
            rfs_spec.consolidate(path)
    for future in delete_futures:
        name, size = future.result()
        freed += size
        print(f'released {name}: {size / 1e9:,.1f} GB of quarters deleted', flush=True)
    delete_pool.shutdown()
    left = sum(1 for region in regions if (in_dir / region['name']).exists())
    print(f'deleted {freed / 1e12:,.2f} TB of quarter stores, {left} of {len(regions)} regions still on disk')
    written = ', '.join([str(out), *(str(p) for p in extras.values())])
    print(f'wrote {written} in {(time.time() - started) / 60:.1f} min')
