"""
The layout of the published RFS v3 stores, from the RFS v3 specification (rfs-specification-documents,
docs/specs/rfs-v3.md, "Zarr structuring"): encodings, chunking, time axes, coordinates, and writing a block of rivers
into a store. The specification names this module the source of truth for encodings, and where the two disagree the
specification is right and this module is wrong. Nothing else in this repository picks a dtype, codec, chunk shape or
keepbits value.

Every store is Zarr v3 with consolidated metadata, riverId the first dimension of every array, every array compressed
with blosc zstd alone, and discharge float32 rounded to KEEP_BITS before it is written, never by a codec filter.

The retrospective stores: every discharge array is (riverId, time) float32, chunked and sharded along riverId as
LAYOUT says, and its time axis is cut at CHUNK_SPLIT_DATE so the historical record is one immutable chunk and an append
only rewrites the chunk after it. A chunk is one river where a river's series in the array is at least a few KB, and
many rivers where it is shorter; a shard holds about 5 to 750 MB. So no array of the 4.9 million rivers has more than
20,000 shards, and no shard index, 16 bytes per chunk, is more than 64 KB.
"""

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
import zarr.codecs as zc

TITLE = 'River Forecast System v3 Retrospective Simulation {}'
LICENSE = 'CC-BY-NC-SA 4.0'
DISCHARGE_ATTRS = {'long_name': 'Discharge at catchment outlet', 'standard_name': 'discharge', 'units': 'm3 s-1'}
KEEP_BITS = 13
# blosc is the only codec the browser clients can decode
COMPRESSOR = zc.BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')
CHUNK_SPLIT_DATE = pd.Timestamp('2025-01-01')
# Q_timesteps is every river at one step, for map styling: chunked across rivers, and on the monthly store sharded a
# year at a time. The yearly store's 85 steps are left unsharded, 20 files each.
TIMESTEPS_CHUNK = (250_000, 1)
TIMESTEPS_SHARD = {'monthly': (250_000, 12), 'yearly': None}

# name: (time axis frequency, discharge variables, aggregation_method, has Q_timesteps)
STORES = {
    'hourly': ('h', ('Q',), 'mean', False),
    'daily': ('D', ('Q',), 'mean', False),
    'monthly': ('MS', ('Q',), 'mean', True),
    'yearly': ('YS', ('Q',), 'mean', True),
    'maximums': ('YS', ('hourly', 'daily'), 'max', False),
}

# each retrospective store's (rivers per chunk, rivers per shard) of its discharge arrays. The values of one river:
# hourly 745,128 over 85 years, daily 31,047, monthly 1,020, yearly and maximums 85
LAYOUT = {
    'hourly': (1, 250),
    'daily': (1, 1_000),
    'monthly': (1, 4_000),
    'yearly': (250, 50_000),
    'maximums': (250, 50_000),
}
# return-periods.zarr's curves, 7 values a river, and fdc.zarr's, 101. The max_simulated arrays are unsharded chunks.
RETURN_PERIODS_LAYOUT = (1_000, 250_000)
FDC_LAYOUT = (250, 50_000)
MAX_SIMULATED_CHUNK = 250_000

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


def create_store(path: Path, name: str, times: pd.DatetimeIndex, river_ids: np.ndarray) -> None:
    """Create one store, empty, with the spec's layout, encoding and metadata."""
    _, variables, aggregation, timesteps = STORES[name]
    n_rivers, n_time = len(river_ids), len(times)
    split = int(np.searchsorted(times, CHUNK_SPLIT_DATE))
    t_chunk = split if 0 < split < n_time else n_time
    title = TITLE.format(name.capitalize())
    if times[0].year == times[-1].year:
        title = f'{title} {times[0].year}'
    group = zarr.create_group(str(path), attributes={'title': title, 'license': LICENSE})  # raises if it exists
    attrs = {**DISCHARGE_ATTRS, 'aggregation_method': aggregation, 'keepbits': KEEP_BITS}
    q = {'shape': (n_rivers, n_time), 'dtype': 'float32', 'fill_value': np.nan, 'compressors': [COMPRESSOR],
         'config': {'write_empty_chunks': True}, 'dimension_names': ('riverId', 'time'), 'attributes': attrs}
    per_chunk, per_shard = LAYOUT[name]
    for variable in variables:
        group.create_array(variable, chunks=(per_chunk, t_chunk), shards=(per_shard, t_chunk), **q)
    if timesteps:
        group.create_array('Q_timesteps', chunks=TIMESTEPS_CHUNK, shards=TIMESTEPS_SHARD[name], **q)

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


# The 15 day forecast, forecasts15/year=YYYY/month=MM/day=DD/discharge.zarr: Q (riverId, member, time), Qpercentiles
# (riverId, percentiles, time) and Qmean (riverId, time), each chunk one river's whole ensemble over the whole horizon
FORECAST_TITLE = 'River Forecast System v3 15 Day Forecast'
IFS_GRID = 'O1280'  # the IFS resolution the forecasts are routed on, and the weight tables they are routed with
FORECAST_STEP_HOURS = 3
FORECAST_STEPS = 120  # 15 days of 3 hour intervals, left aligned: lead times 0, 3, ... 357
MEMBERS = np.arange(0, 51, dtype=np.int32)  # the control, 0, and the 50 perturbed, 1..50
PERCENTILES = np.arange(0, 101, 10, dtype=np.int32)  # deciles: 0 is the ensemble minimum, 50 the median, 100 the max
# ECMWF's own numbering: the control forecast is 0 and each perturbed forecast keeps its number. The specification says
# 1..51 and must be updated to match.
# each forecast array's (rivers per chunk, rivers per shard): Q is 6,120 values a river, Qpercentiles 1,320, Qmean 120
FORECAST_LAYOUT = {'Q': (1, 250), 'Qpercentiles': (1, 1_000), 'Qmean': (250, 10_000)}
# the rivers the forecast scripts read and write at once: whole shards of every forecast array, so no two processes
# ever write one shard
FORECAST_BLOCK = 10_000
MEMBER_DESCRIPTION = 'IFS ensemble member: 0 the control forecast, 1..50 the perturbed forecasts of the same number'


def member_number(forecast_type: str, number: int) -> int:
    """The member of an IFS forecast, from its GRIB type ('cf' or 'pf') and ECMWF member number."""
    if forecast_type == 'cf':
        return 0
    if forecast_type == 'pf' and 1 <= number <= 50:
        return number
    raise ValueError(f'no member for IFS forecast type {forecast_type!r} number {number}')


def forecast_store_path(root: Path, initialization: pd.Timestamp) -> Path:
    """Where a forecast's discharge store lives under the forecasts15/ root: ``year=YYYY/month=MM/day=DD/``."""
    return Path(root) / f'year={initialization:%Y}' / f'month={initialization:%m}' / f'day={initialization:%d}' / \
        'discharge.zarr'


def forecast_times(initialization: pd.Timestamp) -> pd.DatetimeIndex:
    """The start of each forecast interval: the initialization, then every FORECAST_STEP_HOURS for FORECAST_STEPS."""
    return pd.date_range(initialization, periods=FORECAST_STEPS, freq=f'{FORECAST_STEP_HOURS}h')


def ensemble_summaries(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Qpercentiles (river, percentiles, time) and Qmean (river, time) of ``q`` (river, member, time), which must already
    be rounded to KEEP_BITS. With 51 members every decile falls on a member, so the percentiles are members' values and
    already on the keepbits grid; the mean is summed in float64 and rounded onto it.
    """
    percentiles = np.percentile(q, PERCENTILES, axis=1, method='linear').astype(np.float32).transpose(1, 0, 2)
    mean = round_keepbits(q.mean(axis=1, dtype=np.float64).astype(np.float32))
    return round_keepbits(np.ascontiguousarray(percentiles)), mean


def create_forecast_store(path: Path, initialization: pd.Timestamp, river_ids: np.ndarray) -> None:
    """Create one forecast discharge store, empty, with the spec's layout, encoding and metadata."""
    n_rivers, n_members, n_percentiles = len(river_ids), MEMBERS.size, PERCENTILES.size
    # path is the temporary name 2_discharge_zarr.py renames into place, so what is replaced is an unfinished store
    group = zarr.create_group(str(path), overwrite=True, attributes={
        'title': FORECAST_TITLE, 'license': LICENSE, 'initialization_time': f'{initialization:%Y-%m-%dT%H:%M:%SZ}'})
    # coordinates is CF's way to name lead_time a coordinate on the time dimension rather than a variable of its own
    attrs = {**DISCHARGE_ATTRS, 'aggregation_method': 'mean', 'keepbits': KEEP_BITS, 'coordinates': 'lead_time'}
    q = {'dtype': 'float32', 'fill_value': np.nan, 'compressors': [COMPRESSOR], 'config': {'write_empty_chunks': True}}
    # name: the axes after riverId, each whole in one chunk
    arrays = {'Q': ('member', n_members), 'Qpercentiles': ('percentiles', n_percentiles), 'Qmean': None}
    for name, axis in arrays.items():
        inner = ((axis[1],) if axis else ()) + (FORECAST_STEPS,)
        dims = ('riverId', *((axis[0],) if axis else ()), 'time')
        per_chunk, per_shard = FORECAST_LAYOUT[name]
        group.create_array(name, shape=(n_rivers, *inner), chunks=(per_chunk, *inner), shards=(per_shard, *inner),
                           dimension_names=dims, attributes=attrs, **q)

    lead_time = np.arange(FORECAST_STEPS, dtype=np.int32) * FORECAST_STEP_HOURS
    coordinates = {
        'riverId': (river_ids.astype(np.int32), ('riverId',), {}),
        'member': (MEMBERS, ('member',), {'description': MEMBER_DESCRIPTION}),
        'time': (lead_time, ('time',), {'units': f'hours since {initialization:%Y-%m-%dT%H:%M:%S}+00:00',
                                         'calendar': 'proleptic_gregorian'}),
        'lead_time': (lead_time, ('time',), {'units': 'hours', 'long_name': 'time since the forecast initialization'}),
        'percentiles': (PERCENTILES, ('percentiles',), {}),
    }
    for name, (values, dims, attributes) in coordinates.items():
        array = group.create_array(name, shape=values.shape, dtype='int32', chunks=values.shape,
                                   compressors=[COMPRESSOR], dimension_names=dims, attributes=attributes)
        array[:] = values
    zarr.consolidate_metadata(str(path))


# The forecast map stylesets, forecasts15/year=YYYY/month=MM/day=DD/maps/<styleset>/styles.{json,bin}: one byte per
# river per step, in riverIndex order, read by the web app to color and size the stream tiles. The specification's
# "Forecast map stylesets" section documents the format; write_styles is its reference implementation.
STYLESETS = ('timeseries', 'max-flow', 'time-to-peak', 'below-q95')
STYLE_RETURN_PERIODS = (2, 5, 10, 25, 50, 100)  # the recurrence intervals of the color classes 1..6; 0 is below 2
STYLE_THICKNESS_EDGES = (1, 10, 100, 1_000, 10_000)  # m3 s-1: thickness class 1 below 1, 6 at or above 10,000
STYLE_NO_DATA = 255  # time-to-peak and below-q95: no value for the river
STYLE_TIMESTAMP = '%Y-%m-%d-%H'  # the label of each step, the UTC start of its interval


def return_period_class(q: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    The number of STYLE_RETURN_PERIODS thresholds each flow meets or exceeds, 0..6. ``q`` is (river, ...) and
    ``thresholds`` (river, 6), each river's flows at those recurrence intervals. A river without thresholds is 0.
    """
    shape = (thresholds.shape[0],) + (1,) * (q.ndim - 1)
    classes = np.zeros(q.shape, dtype=np.uint8)
    for k in range(thresholds.shape[1]):
        classes += q >= thresholds[:, k].reshape(shape)  # NaN compares False either way
    return classes


def thickness_class(q: np.ndarray) -> np.ndarray:
    """1 plus the number of STYLE_THICKNESS_EDGES each flow meets or exceeds, 1..6. NaN is 1."""
    return (1 + np.searchsorted(np.asarray(STYLE_THICKNESS_EDGES, np.float32), np.nan_to_num(q, nan=0),
                                side='right')).astype(np.uint8)


def return_period_byte(q: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """The timeseries and max-flow byte: return period class in the high 5 bits, thickness class - 1 in the low 3."""
    return (return_period_class(q, thresholds) << 3) | (thickness_class(q) - 1)


def write_styles(directory: Path, styleset: str, cube: np.ndarray, initialization: pd.Timestamp,
                 times: pd.DatetimeIndex, description: str) -> None:
    """
    Write one styleset: ``cube`` is (river, step) uint8 in riverIndex order, ``times`` the start of each step. The
    bytes are delta encoded along each river's steps, modulo 256, and zlib compressed, into styles.bin, and
    styles.json describes them. Each file is written to a temporary name and renamed into place.
    """
    if styleset not in STYLESETS or cube.dtype != np.uint8 or cube.ndim != 2 or cube.shape[1] != len(times):
        raise ValueError(f'{styleset}: expected (river, {len(times)}) uint8, got {cube.shape} {cube.dtype}')
    delta = cube.copy()
    delta[:, 1:] -= cube[:, :-1]  # uint8 arithmetic wraps, which is the modulo 256
    meta = {
        'styleset': styleset,
        'description': description,
        'initialization_time': f'{initialization:%Y-%m-%dT%H:%M:%SZ}',
        'n_reaches': int(cube.shape[0]),
        'n_steps': int(cube.shape[1]),
        'step_hours': FORECAST_STEP_HOURS,
        'timestamps': [f'{t:{STYLE_TIMESTAMP}}' for t in times],
        'ret_per_values': [0, *STYLE_RETURN_PERIODS],
        'thickness_edges': list(STYLE_THICKNESS_EDGES),
        'no_data': STYLE_NO_DATA,
        'encoding': 'uint8 (n_reaches, n_steps) C order, delta along steps modulo 256, zlib',
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    compressor = zlib.compressobj(level=6)
    partial = directory / 'styles.partial.bin'
    with open(partial, 'wb') as file:
        flat = delta.reshape(-1)
        for start in range(0, flat.size, 1 << 26):
            file.write(compressor.compress(flat[start:start + (1 << 26)].tobytes()))
        file.write(compressor.flush())
    os.replace(partial, directory / 'styles.bin')
    partial = directory / 'styles.partial.json'
    partial.write_text(json.dumps(meta, indent=1))
    os.replace(partial, directory / 'styles.json')


# The forecast alerts, forecasts15/year=YYYY/month=MM/day=DD/alerts.csv: one row per river where at least
# ALERT_PROBABILITY of the members exceed a return period flow, at the largest recurrence interval they do, with the
# fields of a CAP alert. The specification's "Forecast alerts" section documents the columns.
ALERT_PROBABILITY = 0.3  # the share of members that must exceed a return period flow at one step
ALERT_SEVERITY = {1.5: 'Minor', 2: 'Minor', 5: 'Moderate', 10: 'Moderate', 25: 'Severe', 50: 'Severe', 100: 'Extreme'}
ALERT_URGENCY_HOURS = ((24, 'Immediate'), (72, 'Expected'))  # onset within these hours of the start, else Future
ALERT_LIKELY = 0.5  # certainty is Likely at or above this share of members, else Possible
ALERT_COLUMNS = ('identifier', 'sent', 'event', 'urgency', 'severity', 'certainty', 'onset', 'expires', 'riverId',
                 'riverIndex', 'lat', 'lon', 'recurrence_interval', 'return_period_flow', 'exceedance_probability',
                 'peak_time', 'peak_median_flow')
