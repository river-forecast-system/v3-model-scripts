"""
The parts of the RFS v3 retrospective specification (river-forecast-system/rfs-specification-documents,
docs/specs/rfs-v3.md, "Dataset Structure and Schematics") that more than one script in here writes: the codec, the
encoding and chunking of every array, the layout of each store, the period cascade and the code that creates a store.

This module is the source of truth the spec points at for encodings. Nothing else in here picks a dtype, a
compressor, a chunk shape or a keepbits value -- a writer that needs one imports it from here -- so every published
store is written the same way and a change lands in one place.

``3_build_hourly_daily_maximums.py`` writes hourly.zarr and, out of the same blocks, daily.zarr and maximums.zarr;
``4_build_monthly_yearly.py`` writes whatever periods are left from either of them.

A level is aggregated from the level above it in ``LEVELS``. Every level carries the count of *source* steps behind
each period, not a count of the level above, so a period is always the mean of the source values under it rather
than a mean of means: from an hourly source a month is the mean of its hours, and from a daily source it is the
mean of its days, which is the same number because every day holds the same 24 hours.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import zarr.codecs as zc
from numcodecs import BitRound

TITLE = 'River Forecast System v3 Retrospective Simulation {}'
LICENSE = 'CC-BY-NC-SA 4.0'

# One keepbits for every discharge array in every product. The router rounds the quarters it writes to this grid and
# every store built from them is rounded to the same grid again, which leaves a value that is already on it
# untouched: that is what makes a peak found in maximums.zarr the same float as the peak in hourly.zarr, and what
# makes re-rounding on an append idempotent. Bounds relative error at 2^-14, about 6.1e-05.
KEEP_BITS = 13

# The only codec the spec permits: browser clients bundle the blosc wasm build alone, so an array written with bare
# zstd, lz4 or anything else fails to decode there. One blosc entry (it shuffles inside its own frame) and typesize
# left unset so every array takes it from its own dtype. Coordinate arrays get it too, never zarr's default.
COMPRESSOR = zc.BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')

# Every array is (riverId, time): the river axis leads, which is the layout river-route's kernels produce and its
# zarr_writer stores, so nothing between the router and a published store transposes anything.
#
# The dimension order is not what decides the bytes -- one river per chunk does. A (1, n_time) chunk and an
# (n_time, 1) one both hold a single river's whole series contiguously, so the two orders shard byte for byte
# identically: same compression, same file count, same cost to read one river. What river major buys is on the
# write side, where a time major store made every writer transpose a buffer to fill it.
#
# Chunking and sharding, the same for every timeseries product:
#   one river per chunk, because the read that matters is one river's whole series and a wider chunk would make it
#   decompress its neighbours. Chunks are stored C-order, so a chunk holding several rivers also interleaves them in
#   the byte stream and breaks up the temporal autocorrelation zstd feeds on.
RIVERS_PER_CHUNK = 1
#   250 rivers per shard, so one file on disk answers a 250 river bulk read and a global store is a manageable
#   number of objects instead of one per river.
RIVERS_PER_SHARD = 250
#   The time axis is cut at CHUNK_SPLIT_DATE: the first chunk covers the 85 years the record was built with
#   (1940-01-01 .. 2024-12-31) and everything from 2025 on falls in the next one. The historical chunk is then
#   immutable, and a daily append rewrites only the chunk -- and only the shards -- the new steps land in.
CHUNK_SPLIT_DATE = pd.Timestamp('2025-01-01')
#   Q_timesteps answers the opposite read, every river at one timestep, so it is chunked across time instead and is
#   not sharded: one chunk is already a whole, cacheable GET.
TIMESTEPS_CHUNK = (250_000, 1)

# Per store layout.
#   freq        the pandas period the level is aggregated to
#   variables   the discharge arrays the store holds
#   timesteps   whether it also holds Q_timesteps, the same values chunked for map styling
#   aggregation the aggregation_method attribute; "mean" unless stated
LAYOUTS = {
    'hourly': {'freq': 'h', 'variables': ('Q',), 'timesteps': False},
    'daily': {'freq': 'D', 'variables': ('Q',), 'timesteps': False},
    'monthly': {'freq': 'M', 'variables': ('Q',), 'timesteps': True},
    'yearly': {'freq': 'Y', 'variables': ('Q',), 'timesteps': True},
    # The annual maximum series, one value a river a year, laid out like yearly because it has the same shape. It
    # holds the maximum of the hourly series and the maximum of the daily series in separate arrays, because the
    # hourly maximum of a year is by definition greater than or equal to the daily one and return-periods.zarr fits
    # a distribution against each. A maximum does not cascade -- a daily mean has already flattened the peak -- so
    # it is not a level in LEVELS and is written only where the hourly values are, by 3_build_hourly_daily_maximums.py.
    'maximums': {'freq': 'Y', 'variables': ('hourly', 'daily'), 'timesteps': False, 'aggregation': 'max'},
}
LEVELS = ('hourly', 'daily', 'monthly', 'yearly')
STEPS = {'hourly': pd.Timedelta(hours=1), 'daily': pd.Timedelta(days=1)}


def store_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / f'{name}.zarr'


def store_dates(time_var) -> pd.DatetimeIndex:
    """A store's time axis as naive UTC timestamps, from any '<unit> since <reference>' units string."""
    unit, reference = (s.strip() for s in str(time_var.attrs['units']).split('since', 1))
    epoch = pd.Timestamp(reference)
    if epoch.tzinfo is not None:
        epoch = epoch.tz_convert('UTC').tz_localize(None)
    return epoch + pd.to_timedelta(time_var[:].astype(np.int64), unit={'seconds': 's', 'minutes': 'min',
                                                                      'hours': 'h', 'days': 'D'}[unit])


def source_level(dates: pd.DatetimeIndex) -> str:
    """Which level a store's own values are, from the spacing of its time axis."""
    step = dates[1] - dates[0]
    for name, size in STEPS.items():
        if step == size:
            return name
    raise SystemExit(f'a source stepping {step} is not a level this cascades from: {", ".join(STEPS)}')


def levels_after(source: str, wanted: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """The levels of ``wanted`` that can be built from ``source``, in cascade order; every one of them if None."""
    below = [name for name in LEVELS[LEVELS.index(source) + 1:] if name in LAYOUTS]
    if wanted is None:
        return below
    if impossible := [name for name in wanted if name not in below]:
        raise SystemExit(f'{", ".join(impossible)} cannot be built from a {source} source -- it holds '
                         f'{", ".join(below) or "nothing"}')
    return [name for name in below if name in wanted]


def _link(parent_times: pd.DatetimeIndex, parent_counts: np.ndarray, name: str, step: pd.Timedelta) -> dict:
    """
    One link of the chain: where each ``name`` period begins in ``parent_times``, and how many source steps are
    behind it. ``times`` holds the start of every period, complete or not; ``keep`` slices out the complete ones.
    """
    labels = parent_times.to_period(LAYOUTS[name]['freq'])
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    counts = np.add.reduceat(parent_counts, starts)
    first = labels[starts].start_time
    expected = np.array([(p.end_time.ceil('s') - p.start_time) // step for p in labels[starts]])
    complete = counts == expected
    if not complete.any():
        raise SystemExit(f'the source does not cover one complete {name} period')
    lo, hi = np.flatnonzero(complete)[[0, -1]]
    if not complete[lo:hi + 1].all():
        raise SystemExit(f'{name} periods in the middle of the record are incomplete -- the source has a gap')
    return {'starts': starts, 'counts': counts, 'keep': slice(int(lo), int(hi) + 1), 'times': first,
            'dropped': int(len(starts) - complete.sum())}


def _uniform_step(dates: pd.DatetimeIndex) -> pd.Timedelta:
    step = dates[1] - dates[0]
    if not (np.diff(dates.values) == step.to_timedelta64()).all():
        raise SystemExit('the source time axis is not uniformly spaced -- it has a gap or a duplicate')
    return step


def level_periods(dates: pd.DatetimeIndex, name: str) -> dict:
    """
    Where each ``name`` period begins in ``dates`` itself, rather than in the level above it.

    This is what a reduction that cannot be cascaded needs: an annual maximum has to be taken over the hourly values
    of the year in one pass, since the maximum of daily means is not the maximum of the hours.
    """
    period = _link(dates, np.ones(len(dates), dtype=np.int64), name, _uniform_step(dates))
    return {**period, 'times': period['times'][period['keep']]}


def cascade(dates: pd.DatetimeIndex, levels: list[str] | tuple[str, ...]) -> dict:
    """
    The period boundaries of every level of the cascade, each one indexed into the level before it.

    For each level: `starts`, where each period begins as an index into the level before it (the source axis, for
    the first); `counts`, the source values behind each period; `keep`, the slice of complete periods to write;
    `times`, their start times; `dropped`, the partial ones left out.

    Every period between the first level asked for and the source is carried down the chain, partial ones included,
    so a period's count and sum stay exact even when the level above it is incomplete at an end -- but only complete
    periods are written. Raises on a gap in the source, since a mean over a period with missing values would
    otherwise be written as if it were complete.
    """
    if not len(levels):
        raise SystemExit('cascade was asked for no levels')
    step = _uniform_step(dates)
    built = {}
    parent_times, parent_counts = dates, np.ones(len(dates), dtype=np.int64)
    # every level between the source and the deepest one asked for, so the chain is never broken
    chain = LEVELS[LEVELS.index(source_level(dates)) + 1:LEVELS.index(levels[-1]) + 1]
    for name in chain:
        period = _link(parent_times, parent_counts, name, step)
        parent_times, parent_counts = period['times'], period['counts']
        built[name] = {**period, 'times': period['times'][period['keep']]}
    return built


def time_chunk(times: pd.DatetimeIndex) -> int:
    """
    Steps in one time chunk: everything before ``CHUNK_SPLIT_DATE``, so the historical record is one chunk and
    2025-present is the next. A record that ends before the split, or begins after it, is one chunk.
    """
    n = int(np.searchsorted(times, CHUNK_SPLIT_DATE))
    return n if 0 < n < len(times) else len(times)


def create_store(path: Path, name: str, period: dict, river_ids: np.ndarray, timesteps: bool | None = None) -> None:
    """
    Create one store, empty, with the layout, encoding and metadata the spec fixes, and consolidate its metadata.

    ``timesteps`` overrides whether the store carries Q_timesteps. The per region stores 2_rfs_v3_retro_router.py
    writes are intermediates that one published store is assembled from, and Q_timesteps is chunked across every
    river at one timestep -- a chunking that only means anything once every region is in one store -- so they pass
    False and the assembling step writes it on the global store instead.
    """
    layout = LAYOUTS[name]
    if timesteps is None:
        timesteps = layout['timesteps']
    times = period['times']
    n_time, n_rivers = len(times), len(river_ids)
    epoch = times[0]
    group = zarr.create_group(str(path), overwrite=True,
                              attributes={'title': TITLE.format(name.capitalize()), 'license': LICENSE})
    q_attrs = {'long_name': 'Discharge at catchment outlet', 'standard_name': 'discharge', 'units': 'm3 s-1',
               'aggregation_method': layout.get('aggregation', 'mean'), 'keepbits': KEEP_BITS}
    # One chunk is one river over one half of the time axis, and one shard file is 250 of those chunks. A shard that
    # stopped at the chunk boundary would make an append rewrite the historical chunk along with the new one, so the
    # shard is the chunk's width on the time axis, not the whole axis.
    t_chunk = time_chunk(times)
    for variable in layout['variables']:
        group.create_array(
            variable, shape=(n_rivers, n_time), dtype='float32', fill_value=np.nan,
            chunks=(RIVERS_PER_CHUNK, t_chunk), shards=(RIVERS_PER_SHARD, t_chunk),
            compressors=[COMPRESSOR], config={'write_empty_chunks': True},
            dimension_names=('riverId', 'time'), attributes=q_attrs,
        )
    if timesteps:
        # not bitrounded again: these are the already rounded values of Q, so keepbits describes them as it is
        group.create_array(
            'Q_timesteps', shape=(n_rivers, n_time), dtype='float32', fill_value=np.nan,
            chunks=TIMESTEPS_CHUNK, shards=None,
            compressors=[COMPRESSOR], config={'write_empty_chunks': True},
            dimension_names=('riverId', 'time'), attributes=q_attrs,
        )
    hours = ((times - epoch) // pd.Timedelta(hours=1)).to_numpy()
    if hours.max() >= 2 ** 31:
        raise SystemExit(f'{name}: hours since {epoch} overflow int32')
    time_var = group.create_array(
        'time', shape=(n_time,), dtype='int32', chunks=(n_time,), compressors=[COMPRESSOR],
        dimension_names=('time',),
        attributes={'units': f'hours since {epoch.strftime("%Y-%m-%dT%H:%M:%S")}+00:00',
                    'calendar': 'proleptic_gregorian'},
    )
    time_var[:] = hours.astype(np.int32)
    if river_ids.min() < 0 or river_ids.max() >= 2 ** 31:
        raise SystemExit(f'{name}: river ids do not fit the spec\'s int32 riverId')
    id_var = group.create_array('riverId', shape=(n_rivers,), dtype='int32', chunks=(n_rivers,),
                                compressors=[COMPRESSOR], dimension_names=('riverId',))
    id_var[:] = river_ids.astype(np.int32)
    consolidate(path)


def consolidate(path: Path) -> None:
    """Consolidate a store's metadata. Every published store carries it: a browser client that had to list the
    store and fetch a zarr.json per array pays a round trip for each one before it can read a value."""
    zarr.consolidate_metadata(str(path))


def write_timesteps(path: Path, threads: int) -> None:
    """Q_timesteps: Q read back in full and rewritten one timestep chunk at a time, in parallel over timesteps."""
    group = zarr.open_group(str(path), mode='r+')
    with zarr.config.set({'async.concurrency': threads, 'threading.max_workers': threads}):
        q = group['Q'][:]
        dst = group['Q_timesteps']
        step = dst.chunks[1]  # (riverId, time): a Q_timesteps chunk is every river over `step` timesteps

        def put(t0: int) -> None:
            dst[:, t0:t0 + step] = q[:, t0:t0 + step]

        # each call writes whole chunks of its own, so the threads never share one
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(put, range(0, q.shape[1], step)))


def store_shape(path: Path, name: str = 'daily') -> tuple[int, int]:
    """The (riverId, time) shape of a store, read off its first discharge array -- maximums.zarr has no 'Q'."""
    return zarr.open_array(str(Path(path) / LAYOUTS[name]['variables'][0]), mode='r').shape


def bitrounder():
    """A reusable BitRound. Rounding hands back an integer array; its bits are viewed as float32, never cast."""
    codec = BitRound(keepbits=KEEP_BITS)

    def round_to_keepbits(values: np.ndarray) -> np.ndarray:
        return np.asarray(codec.encode(values)).view(np.float32).reshape(values.shape)

    return round_to_keepbits
