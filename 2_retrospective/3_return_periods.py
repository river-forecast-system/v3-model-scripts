"""
Fit return periods to the annual maxima of maximums.zarr, which 2_concatenate.py writes, and write the published
return-periods.zarr beside it:

    <out-dir>/return-periods.zarr
        riverId, recurrence_interval [1.5, 2, 5, 10, 25, 50, 100], annual_exceedance_probability 1 / interval
        {gumbel,logpearson3,lognormal,weibull}_{hourly,daily}   (riverId, recurrence_interval) the flow of each interval
        max_simulated_{hourly,daily}                            (riverId,) the largest annual maximum of the record

laid out as rfs_spec.py and the specification say: a chunk is the whole curves of 1,000 rivers and a shard 250,000,
rfs_spec.RETURN_PERIODS_LAYOUT, the max_simulated arrays are unsharded chunks of 250,000 rivers, and every discharge
value is rounded to rfs_spec.KEEP_BITS.

Every distribution is fit to each river's own maxima by moments, with each annual maximum the series value of one
year, and evaluated at the non-exceedance probability 1 - 1 / interval:

    gumbel       Gumbel (GEV type 1), method of moments: mean + K * std, K = -sqrt(6) / pi * (0.5772 + ln ln(T / (T-1)))
    lognormal    normal on ln Q: exp(mean + z * std)
    logpearson3  Pearson III on log10 Q with the station skew, the frequency factor from scipy.stats.pearson3
    weibull      two parameter Weibull, shape from the coefficient of variation and scale from the mean

Standard deviations use n - 1 and the skew the adjusted n / ((n - 1)(n - 2)) estimate. The log fits read maxima below
LOG_FLOOR as LOG_FLOOR. A river whose maxima are all equal, often all 0, gets that value at every interval, and no
flow is below 0.

The rivers are cut into blocks of one shard each, 20 for the world, and --processes fit and write them at once, each
reading and writing only its own shards.
"""

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import scipy.special
import scipy.stats
import zarr
from tqdm import tqdm

import rfs_spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

INTERVALS = np.array([1.5, 2, 5, 10, 25, 50, 100], dtype=np.float32)
DISTRIBUTIONS = ('gumbel', 'logpearson3', 'lognormal', 'weibull')
SERIES = ('hourly', 'daily')  # the variables of maximums.zarr each distribution is fit to
LOG_FLOOR = 1e-4  # m3 s-1: the smallest maximum the log fits take a logarithm of
PER_CHUNK, BLOCK = rfs_spec.RETURN_PERIODS_LAYOUT  # BLOCK, one shard, is the rivers a process fits and writes at once

P = 1 - 1 / INTERVALS.astype(np.float64)  # the non-exceedance probability of each interval
Z = scipy.stats.norm.ppf(P)
GUMBEL_K = -np.sqrt(6) / np.pi * (np.euler_gamma + np.log(np.log(INTERVALS / (INTERVALS - 1.0))))
# the Weibull shape k for a coefficient of variation, CV^2 = G(1 + 2/k) / G(1 + 1/k)^2 - 1, decreasing in k
WEIBULL_K = np.geomspace(0.1, 200, 4000)
WEIBULL_CV = np.sqrt(np.exp(scipy.special.gammaln(1 + 2 / WEIBULL_K) - 2 * scipy.special.gammaln(1 + 1 / WEIBULL_K))
                     - 1)


def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's mean and standard deviation with n - 1."""
    return x.mean(axis=1), x.std(axis=1, ddof=1)


def fit(maxima: np.ndarray) -> dict[str, np.ndarray]:
    """Each distribution's flow at every interval for (river, year) annual maxima, as float32 (river, interval)."""
    q = maxima.astype(np.float64)
    n = q.shape[1]
    mean, std = moments(q)
    out = {'gumbel': mean[:, None] + GUMBEL_K * std[:, None]}

    ln_mean, ln_std = moments(np.log(np.maximum(q, LOG_FLOOR)))
    out['lognormal'] = np.exp(ln_mean[:, None] + Z * ln_std[:, None])

    logs = np.log10(np.maximum(q, LOG_FLOOR))
    lg_mean, lg_std = moments(logs)
    with np.errstate(invalid='ignore', divide='ignore'):
        skew = n / ((n - 1) * (n - 2)) * (((logs - lg_mean[:, None]) / lg_std[:, None]) ** 3).sum(axis=1)
    skew = np.nan_to_num(skew)  # 0 where every maximum is equal
    factor = scipy.stats.pearson3.ppf(P[None, :], skew[:, None])
    out['logpearson3'] = 10 ** (lg_mean[:, None] + factor * lg_std[:, None])

    with np.errstate(invalid='ignore', divide='ignore'):
        cv = np.nan_to_num(std / mean)  # 0 where every maximum is 0
    shape = np.interp(cv, WEIBULL_CV[::-1], WEIBULL_K[::-1])  # past the table, the nearest end
    scale = mean / np.exp(scipy.special.gammaln(1 + 1 / shape))
    out['weibull'] = scale[:, None] * np.log(INTERVALS.astype(np.float64))[None, :] ** (1 / shape[:, None])

    flat = std == 0  # every maximum equal: no spread to fit, so every interval is that value
    for name, values in out.items():
        values[flat] = q[flat, :1]
        out[name] = rfs_spec.round_keepbits(np.maximum(values, 0).astype(np.float32))
    return out


def fit_block(source: Path, target: Path, r0: int, r1: int) -> tuple[int, dict[str, np.ndarray]]:
    """Fit rivers [r0, r1) of both series, write their curves, and return each series' largest annual maximum."""
    with zarr.config.set({'async.concurrency': 4, 'threading.max_workers': 1}):
        maximums = zarr.open_group(str(source), mode='r')
        group = zarr.open_group(str(target), mode='r+')
        largest = {}
        for series in SERIES:
            maxima = maximums[series][r0:r1]
            if np.isnan(maxima).any():
                raise ValueError(f'maximums.zarr/{series} has NaN in rivers {r0}..{r1}')
            for name, values in fit(maxima).items():
                group[f'{name}_{series}'][r0:r1] = values
            largest[series] = maxima.max(axis=1)
    return r0, largest


def create_store(path: Path, river_ids: np.ndarray) -> None:
    """Create return-periods.zarr under its temporary name, empty but for its coordinates, with the specification's
    layout. A temporary store an interrupted run left behind is unfinished, and replaced."""
    n = river_ids.size
    group = zarr.create_group(str(path), overwrite=True, attributes={
        'title': rfs_spec.TITLE.format('Return Periods'),
        'description': 'The flow of each recurrence interval, from each distribution fit by moments to the annual '
                       'maxima of the hourly series and of the daily means in maximums.zarr. gumbel is Gumbel '
                       '(GEV type 1) by the method of moments.',
        'license': rfs_spec.LICENSE,
    })
    attrs = {**rfs_spec.DISCHARGE_ATTRS, 'keepbits': rfs_spec.KEEP_BITS}
    common = {'dtype': 'float32', 'fill_value': np.nan, 'compressors': [rfs_spec.COMPRESSOR],
              'config': {'write_empty_chunks': True}}
    width = INTERVALS.size
    for name in DISTRIBUTIONS:
        for series in SERIES:
            group.create_array(f'{name}_{series}', shape=(n, width), chunks=(PER_CHUNK, width), shards=(BLOCK, width),
                               dimension_names=('riverId', 'recurrence_interval'),
                               attributes={**attrs, 'distribution': name, 'series': f'maximums.zarr/{series}'},
                               **common)
    for series in SERIES:
        group.create_array(f'max_simulated_{series}', shape=(n,), chunks=(min(rfs_spec.MAX_SIMULATED_CHUNK, n),),
                           dimension_names=('riverId',), attributes={**attrs, 'aggregation_method': 'max'}, **common)
    coords = {'riverId': (river_ids.astype(np.int32), 'riverId', {}),
              'recurrence_interval': (INTERVALS, 'recurrence_interval', {'units': 'years'}),
              'annual_exceedance_probability': ((1 / INTERVALS).astype(np.float32), 'recurrence_interval', {})}
    for name, (values, dim, attributes) in coords.items():
        array = group.create_array(name, shape=values.shape, dtype=values.dtype, chunks=values.shape,
                                   compressors=[rfs_spec.COMPRESSOR], dimension_names=(dim,), attributes=attributes)
        array[:] = values


if __name__ == '__main__':
    began = time.time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'retrospective',
                        help='where 2_concatenate.py wrote maximums.zarr, and return-periods.zarr goes')
    parser.add_argument('--processes', type=int, default=os.cpu_count() or 1, help='blocks of rivers fit at once')
    args = parser.parse_args()

    maximums = args.out_dir / 'maximums.zarr'
    source = zarr.open_group(str(maximums), mode='r')
    river_ids = source['riverId'][:]
    n_years = source['hourly'].shape[1]
    if n_years < 3:
        raise SystemExit(f'maximums.zarr has {n_years} years; the skew of log-Pearson III needs at least 3')
    path = args.out_dir / 'return-periods.zarr'
    if path.exists():  # finished by an earlier run, so a rerun of every step passes over it
        print(f'{path} exists and is kept; delete it to fit the return periods again', flush=True)
        raise SystemExit(0)
    partial = path.with_suffix('.partial.zarr')
    create_store(partial, river_ids)
    print(f'fitting {len(DISTRIBUTIONS)} distributions to {len(SERIES)} series of {n_years} annual maxima for '
          f'{river_ids.size:,} rivers, {args.processes} processes', flush=True)

    largest = {series: np.empty(river_ids.size, np.float32) for series in SERIES}
    blocks = [(r0, min(r0 + BLOCK, river_ids.size)) for r0 in range(0, river_ids.size, BLOCK)]
    with ProcessPoolExecutor(args.processes) as pool:
        futures = [pool.submit(fit_block, maximums, partial, r0, r1) for r0, r1 in blocks]
        for future in tqdm(as_completed(futures), total=len(futures), desc='blocks'):
            r0, values = future.result()
            for series, block in values.items():
                largest[series][r0:r0 + block.size] = block

    group = zarr.open_group(str(partial), mode='r+')
    for series in SERIES:
        group[f'max_simulated_{series}'][:] = largest[series]
    zarr.consolidate_metadata(str(partial))
    os.replace(partial, path)
    print(f'{path} written in {timedelta(seconds=round(time.time() - began))}', flush=True)
