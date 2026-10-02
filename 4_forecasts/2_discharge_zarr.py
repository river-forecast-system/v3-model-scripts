"""
Publish one forecast's routed members, the netCDF files of 1_route_forecast.py, as the forecast discharge store the
specification defines, laid out as rfs_spec.create_forecast_store says:

    <out-root>/year=YYYY/month=MM/day=DD/discharge.zarr    uploaded as is to s3://river-forecast-system-v3/forecasts15/

    Q               (riverId, member, time)         every member, chunks (1, 51, 120) in shards of (250, 51, 120)
    Qpercentiles    (riverId, percentiles, time)    the ensemble's deciles, 0 the minimum and 100 the maximum,
                                                    chunks (1, 11, 120) in shards of (1000, 11, 120)
    Qmean           (riverId, time)                 the ensemble mean, which is not the median, Qpercentiles at 50,
                                                    chunks (250, 120) in shards of (10000, 120)

The members are opened together with xarray.open_mfdataset, concatenated along member, and their labels checked
before anything is written: every member of rfs_spec.MEMBERS once, the riverId axis of hydrography/global/
metadata.parquet in riverIndex order, the order of every published store, and the time axis of
rfs_spec.forecast_times.

Each chunk holds whole ensembles over the whole horizon, so the store is written a block of rivers at a time:
each block of every member is read, stacked into (river, member, time), reduced, and written as whole shards. Blocks
are a whole number of shards, so no two processes write one shard, and --jobs of them run at once.

The members are already rounded to 13 keepbits. Every decile of 51 members falls on a member, so the percentiles are
members' values, and the mean is rounded onto the same grid, as rfs_spec.ensemble_summaries explains.

The store is written to a temporary name and renamed into place when it is complete, so an interrupted run never
leaves a store that looks finished, and its rerun starts the temporary store over. A store that exists is kept, and
nothing is ever deleted: the member files stay in --work-dir, about 120 GB a forecast, for you to remove once the
store is published.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
import zarr

import rfs_spec

_open = {}  # each process's members and store arrays, opened on its first block


def open_members(files: list[Path], block: int) -> xr.Dataset:
    """
    Every member's file as one dataset, concatenated along member in the order given, in dask chunks of ``block``
    rivers. The files are not chunked on disk, so without it each file would be one chunk, and reading any block of
    rivers would read every member's whole file: 28 s a block rather than 0.4 s.
    """
    return xr.open_mfdataset(files, combine='nested', concat_dim='member', engine='netcdf4', chunks={'riverId': block},
                             data_vars='minimal', coords='minimal', compat='override', join='exact')


def check_labels(members: xr.Dataset, river_ids: np.ndarray, initialization: pd.Timestamp) -> None:
    """Refuse members whose labels are not the published store's axes."""
    found = members['member'].to_numpy()
    if not np.array_equal(found, rfs_spec.MEMBERS):
        raise SystemExit(f'the members are {found.tolist()}, expected {rfs_spec.MEMBERS.tolist()}')
    if not np.array_equal(members['riverId'].to_numpy(), river_ids):
        raise SystemExit('the members\' riverId axis is not the riverIndex order of the hydrography')
    if not np.array_equal(members['time'].to_numpy(), rfs_spec.forecast_times(initialization).to_numpy()):
        raise SystemExit(f'the time axis is not the {rfs_spec.FORECAST_STEPS} intervals from {initialization}')
    if members['Q'].dims != ('member', 'riverId', 'time'):
        raise SystemExit(f'the members\' Q is {members["Q"].dims}, expected (member, riverId, time)')
    return


def river_axis(hydrography: Path) -> np.ndarray:
    """The riverId of every river in riverIndex order."""
    table = pq.read_table(hydrography / 'global' / 'metadata.parquet', columns=['riverId', 'riverIndex'])
    river_ids = np.empty(table.num_rows, dtype=np.int32)
    river_ids[table.column('riverIndex').to_numpy()] = table.column('riverId').to_numpy()
    return river_ids


def write_block(job: tuple[Path, tuple[Path, ...], int, int, int]) -> int:
    """Stack, reduce and write rivers [r0, r1) of every member. Returns the rivers written."""
    store, files, block, r0, r1 = job
    if store not in _open:
        dask.config.set(scheduler='synchronous')  # the processes are the parallelism
        group = zarr.open_group(str(store), mode='r+')
        _open.clear()
        _open[store] = (open_members(list(files), block)['Q'], {n: group[n] for n in ('Q', 'Qpercentiles', 'Qmean')})
    members, arrays = _open[store]
    q = np.ascontiguousarray(members[:, r0:r1].to_numpy().transpose(1, 0, 2))  # (river, member, time)
    percentiles, mean = rfs_spec.ensemble_summaries(q)
    arrays['Q'][r0:r1] = q
    arrays['Qpercentiles'][r0:r1] = percentiles
    arrays['Qmean'][r0:r1] = mean
    return r1 - r0


def latest_initialization(work_dir: Path) -> pd.Timestamp:
    stamps = sorted(d.name for d in work_dir.glob('[0-9]' * 10) if d.is_dir())
    if not stamps:
        raise SystemExit(f'no routed forecast in {work_dir}')
    return pd.Timestamp(f'{stamps[-1][:8]}T{stamps[-1][8:]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--initialization', type=pd.Timestamp, default=None,
                        help='the forecast initialization, e.g. 2026-09-27T00; default the newest in --work-dir')
    parser.add_argument('--work-dir', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'forecasts-work',
                        help='where 1_route_forecast.py saved each member, a folder per initialization')
    parser.add_argument('--hydrography', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'hydrography',
                        help='the published hydrography, whose riverIndex order the store follows')
    parser.add_argument('--out-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'forecasts15',
                        help='the forecasts15/ tree the store is written into')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 8, help='processes writing blocks at once')
    parser.add_argument('--block', type=int, default=rfs_spec.FORECAST_BLOCK,
                        help='rivers each process writes at once, a multiple of rfs_spec.FORECAST_BLOCK')
    args = parser.parse_args()

    began = time.time()
    initialization = args.initialization or latest_initialization(args.work_dir)
    path = rfs_spec.forecast_store_path(args.out_root, initialization)
    if path.exists():  # finished by an earlier run, so a rerun of every step passes over it
        print(f'{path} exists and is kept; delete it to write it again', flush=True)
        raise SystemExit(0)
    member_dir = args.work_dir / f'{initialization:%Y%m%d%H}'
    files = tuple(member_dir / f'member_{m:02d}.nc' for m in rfs_spec.MEMBERS)
    if missing := [f.name for f in files if not f.exists()]:
        raise SystemExit(f'{len(missing)} members are not routed in {member_dir}, e.g. {missing[0]}')

    river_ids = river_axis(args.hydrography)
    block = args.block
    if block < 1 or block % rfs_spec.FORECAST_BLOCK:
        raise SystemExit(f'--block {block} must be a multiple of {rfs_spec.FORECAST_BLOCK}, whole shards of each array')
    with open_members(list(files), block) as members:
        check_labels(members, river_ids, initialization)

    partial = path.with_name('discharge.partial.zarr')
    partial.parent.mkdir(parents=True, exist_ok=True)
    rfs_spec.create_forecast_store(partial, initialization, river_ids)
    jobs = [(partial, files, block, r0, min(r0 + block, river_ids.size)) for r0 in range(0, river_ids.size, block)]
    print(f'{initialization:%Y-%m-%d %H}z: {river_ids.size:,} rivers x {len(files)} members in {len(jobs)} blocks, '
          f'{args.jobs} at a time, into {path}', flush=True)
    written = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        for n, rivers in enumerate(pool.map(write_block, jobs, chunksize=4), start=1):
            written += rivers
            if n % 100 == 0 or n == len(jobs):
                print(f'[{n}/{len(jobs)}] {written:,} rivers, {(time.time() - began) / 60:.1f} min', flush=True)

    os.replace(partial, path)
    print(f'{path} written in {(time.time() - began) / 60:.1f} min. The member files in {member_dir} are kept',
          flush=True)
