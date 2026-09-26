"""
The layout of the published retrospective stores, from the RFS v3 specification ("Dataset Structure and
Schematics"): encodings, chunking, time axes, and writing a block of rivers into a store.

Every discharge array is (riverId, time) float32, one river per chunk, 250 rivers per shard, and its time axis is cut
at CHUNK_SPLIT_DATE so the historical record is one immutable chunk and an append only rewrites the chunk after it.
Nothing else in this folder picks a dtype, codec, chunk shape or keepbits value.

A store is written one calendar year at a time, each year its own zarr at ``store_path``, holding every river in
riverIndex order. A year is therefore whole in the river dimension the moment its regions are written, and the
published record is those years merged along time, which no longer has to happen while the simulation runs. For a
one-year store the CHUNK_SPLIT_DATE cut never lands inside the time axis, so a chunk is one river's whole year.
"""

import fcntl
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
import zarr.codecs as zc

TITLE = 'River Forecast System v3 Retrospective Simulation {}'
LICENSE = 'CC-BY-NC-SA 4.0'
KEEP_BITS = 13
# blosc is the only codec the browser clients can decode
COMPRESSOR = zc.BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')
RIVERS_PER_SHARD = 250
CHUNK_SPLIT_DATE = pd.Timestamp('2025-01-01')
# Q_timesteps is every river at one step, for map styling: chunked across rivers and not sharded
TIMESTEPS_CHUNK = (250_000, 1)

# name: (time axis frequency, discharge variables, aggregation_method, has Q_timesteps)
STORES = {
    'hourly': ('h', ('Q',), 'mean', False),
    'daily': ('D', ('Q',), 'mean', False),
    'monthly': ('MS', ('Q',), 'mean', True),
    'yearly': ('YS', ('Q',), 'mean', True),
    'maximums': ('YS', ('hourly', 'daily'), 'max', False),
}

_SHIFT = np.uint32(23 - KEEP_BITS)
_HALF = np.uint32((1 << (22 - KEEP_BITS)) - 1)
_MASK = np.uint32(~((1 << (23 - KEEP_BITS)) - 1) & 0xFFFFFFFF)


def global_layout(
    hydrography: Path, routing: Path, regions: list[str]
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Every riverId in riverIndex order, and each region's [start, end) rows of it, from the order of its
    routing.parquet in ``routing``."""
    table = pq.read_table(hydrography / 'global' / 'metadata.parquet', columns=['riverId', 'riverIndex'])
    ids, index = table.column('riverId').to_numpy(), table.column('riverIndex').to_numpy()
    lookup = pd.Series(index, index=ids)
    offsets = {}
    for region in regions:
        region_ids = pq.read_table(routing / f'region={region}' / 'routing.parquet', columns=['river_id'])
        rows = lookup.reindex(region_ids.column('river_id').to_numpy()).to_numpy()
        start = int(rows[0])
        if not np.array_equal(rows, np.arange(start, start + rows.size)):
            raise SystemExit(f'region={region} is not one contiguous run of riverIndex in routing.parquet order')
        offsets[region] = (start, start + rows.size)
    if sum(end - start for start, end in offsets.values()) != ids.size:
        raise SystemExit('the regions do not cover every river in global/metadata.parquet exactly once')
    river_ids = np.empty_like(ids)
    river_ids[index] = ids
    return river_ids, offsets


def round_keepbits(values: np.ndarray) -> np.ndarray:
    """Round float32 values in place to KEEP_BITS mantissa bits, nearest even. The same bits as numcodecs.BitRound."""
    bits = values.view(np.uint32)
    bits += ((bits >> _SHIFT) & np.uint32(1)) + _HALF
    bits &= _MASK
    return values


def reduce_year(hourly: np.ndarray, year: int) -> dict[str, np.ndarray]:
    """
    Round one calendar year of hourly discharge (river, hour) onto the keepbits grid in place, and reduce it to its
    daily, monthly and yearly means and the annual maxima of the hourly values and the daily means. Every mean is of
    the hours under it, summed in float64 and rounded onto the grid, so each maximum is a value its store holds.

    The year may start late, as ERA5 does at 1940-01-01 07:00. ``hourly`` is then the year's last hours, each mean is
    of the hours present under it, and a day with none is NaN.
    """
    month_days = pd.date_range(f'{year}-01-01', periods=12, freq='MS').days_in_month.to_numpy()
    missing = 24 * int(month_days.sum()) - hourly.shape[1]
    if missing < 0:
        raise ValueError(f'{hourly.shape[1]} hours is more than the year {year} has')
    hourly = round_keepbits(hourly)
    max_hourly = hourly.max(axis=1)
    if missing:
        hourly = np.concatenate((np.zeros((hourly.shape[0], missing), np.float32), hourly), axis=1)
    day_hours = np.clip(24 * np.arange(1, month_days.sum() + 1) - missing, 0, 24)
    month_starts = np.r_[0, np.cumsum(month_days)[:-1]]
    day_sums = hourly.reshape(hourly.shape[0], -1, 24).sum(axis=2, dtype=np.float64)
    with np.errstate(invalid='ignore'):  # 0 / 0 for a day or month before the first hour
        daily = round_keepbits((day_sums / day_hours).astype(np.float32))
        monthly = np.add.reduceat(day_sums, month_starts, axis=1) / np.add.reduceat(day_hours, month_starts)
    return {
        'daily': daily,
        'monthly': round_keepbits(monthly.astype(np.float32)),
        'yearly': round_keepbits((day_sums.sum(axis=1) / day_hours.sum()).astype(np.float32)),
        'max_hourly': max_hourly,
        'max_daily': daily[:, day_hours > 0].max(axis=1),
    }


def record_times(first_year: int, last_year: int) -> dict[str, pd.DatetimeIndex]:
    """Every store's time axis for whole calendar years, each value labeled by the start of its period."""
    start, end = pd.Timestamp(f'{first_year}-01-01'), pd.Timestamp(f'{last_year + 1}-01-01')
    return {name: pd.date_range(start, end, freq=freq, inclusive='left') for name, (freq, *_) in STORES.items()}


def store_path(root: Path, name: str, year: int) -> Path:
    """Where one year of one store lives: ``root/hourly/hourly_1980.zarr``, the years of a store side by side."""
    return Path(root) / name / f'{name}_{year}.zarr'


def create_store(path: Path, name: str, times: pd.DatetimeIndex, river_ids: np.ndarray) -> None:
    """Create one store, empty, with the spec's layout, encoding and metadata."""
    _, variables, aggregation, timesteps = STORES[name]
    n_rivers, n_time = len(river_ids), len(times)
    split = int(np.searchsorted(times, CHUNK_SPLIT_DATE))
    t_chunk = split if 0 < split < n_time else n_time
    title = TITLE.format(name.capitalize())
    if times[0].year == times[-1].year:
        title = f'{title} {times[0].year}'
    group = zarr.create_group(str(path), overwrite=True,
                              attributes={'title': title, 'license': LICENSE})
    attrs = {'long_name': 'Discharge at catchment outlet', 'standard_name': 'discharge', 'units': 'm3 s-1',
             'aggregation_method': aggregation, 'keepbits': KEEP_BITS}
    q = {'shape': (n_rivers, n_time), 'dtype': 'float32', 'fill_value': np.nan, 'compressors': [COMPRESSOR],
         'config': {'write_empty_chunks': True}, 'dimension_names': ('riverId', 'time'), 'attributes': attrs}
    for variable in variables:
        group.create_array(variable, chunks=(1, t_chunk), shards=(RIVERS_PER_SHARD, t_chunk), **q)
    if timesteps:
        group.create_array('Q_timesteps', chunks=TIMESTEPS_CHUNK, **q)

    hours = (times - times[0]) // pd.Timedelta(hours=1)
    time_var = group.create_array(
        'time', shape=(n_time,), dtype='int32', chunks=(n_time,), compressors=[COMPRESSOR], dimension_names=('time',),
        attributes={'units': f'hours since {times[0]:%Y-%m-%dT%H:%M:%S}+00:00', 'calendar': 'proleptic_gregorian'},
    )
    time_var[:] = hours.to_numpy().astype(np.int32)
    id_var = group.create_array('riverId', shape=(n_rivers,), dtype='int32', chunks=(n_rivers,),
                                compressors=[COMPRESSOR], dimension_names=('riverId',))
    id_var[:] = river_ids.astype(np.int32)
    zarr.consolidate_metadata(str(path))


@contextmanager
def shard_lock(locks: Path, shard: int):
    """flock one shard's lock file. It belongs to the open file, so it also excludes threads of the same process."""
    Path(locks).mkdir(parents=True, exist_ok=True)
    with open(Path(locks) / f'{shard}.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_rows(array, start: int, values: np.ndarray, locks: Path, write_bytes: int) -> None:
    """
    Write ``values`` into rows [start, start + len(values)) of an array sharded along rivers.

    Writing part of a shard rewrites the whole shard, so the first and last shard, which may hold a neighboring
    region's rivers, are written under a lock. The shards between belong to this region alone and are written whole,
    as many at a time as fit in ``write_bytes``.
    """
    width = array.shards[0]
    stop = start + values.shape[0]
    head = min(-(-start // width) * width, stop)
    tail = max(stop // width * width, head)

    def put(g0: int, g1: int) -> None:
        array[g0:g1] = np.asarray(values[g0 - start:g1 - start], dtype=np.float32)

    if head > start:
        with shard_lock(locks, start // width):
            put(start, head)
    step = width * max(1, write_bytes // (width * array.shape[1] * 4))
    for g0 in range(head, tail, step):
        put(g0, min(g0 + step, tail))
    if stop > tail:
        with shard_lock(locks, tail // width):
            put(tail, stop)


def write_timesteps(path: Path, threads: int) -> None:
    """Fill Q_timesteps from Q, one block of TIMESTEPS_CHUNK[0] rivers at a time. Blocks share no objects."""
    group = zarr.open_group(str(path), mode='r+')
    q, dst = group['Q'], group['Q_timesteps']
    width = TIMESTEPS_CHUNK[0]

    def put(r0: int) -> None:
        dst[r0:r0 + width] = q[r0:r0 + width]

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(put, range(0, q.shape[0], width)))
