"""
Reduce the hourly discharge 2_route_regions.py writes, one zarr per region per year, to the published daily, monthly,
yearly and maximums stores, one of each per calendar year:

    <out-dir>/daily/daily_1940.zarr          Q: the daily means
    <out-dir>/monthly/monthly_1940.zarr      Q: the monthly means, and Q_timesteps
    <out-dir>/yearly/yearly_1940.zarr        Q: the yearly mean, and Q_timesteps
    <out-dir>/maximums/maximums_1940.zarr    hourly and daily: the year's largest hourly value and largest daily mean

Each holds every river in riverIndex order, laid out as spec.py says, and the values are spec.reduce_year's: every mean
is of the hours under it, and each maximum is a value its store holds. ERA5 begins at 1940-01-01 07:00, so the first
day, month and year of 1940 are means of the hours it has.

The work is one region's year at a time, --processes at once, the years in order and each year's regions biggest
first. A region's year is read and reduced ROWS_PER_BLOCK rivers at a time, and its reductions, 0.44 GB for the
largest region, are then written into its rows of that year's stores. Only its first and last shard can hold a
neighbor's rivers, so only those two writes take a file lock.

A year is reduced only once its discharge is finished, which 2_route_regions.py marks by saving the year's channel
state after it, so this can run while routing is still going. Each region's year is marked done in <out-dir>/.progress
once it is written, and a rerun skips it. When every region of a year is done, Q_timesteps is filled for monthly and
yearly. --overwrite deletes the stores and starts over.
"""

import argparse
import os
import re
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

import spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

ROWS_PER_BLOCK = 5_000  # rivers read and reduced at once: 176 MB of a leap year, ten of river-route's chunks
STORES = ('daily', 'monthly', 'yearly', 'maximums')
# spec.reduce_year's name for each reduction: the store and variable it is written to
REDUCED = {'daily': ('daily', 'Q'), 'monthly': ('monthly', 'Q'), 'yearly': ('yearly', 'Q'),
           'max_hourly': ('maximums', 'hourly'), 'max_daily': ('maximums', 'daily')}
# river-route names the discharge of runoff file era5_194001_194012.zarr discharge_era5_194001_194012.zarr
DISCHARGE = re.compile(r'discharge_(?P<stem>.*_(?P<year>\d{4})01_(?P=year)12)\.zarr')


def done_file(progress: Path, year: int, region: str) -> Path:
    """Exists once the region's rows of every store of the year are written."""
    return progress / str(year) / region


def read_dates(time_var: zarr.Array) -> pd.DatetimeIndex:
    """The dates of a time variable as river-route's writers encode them, with units '<unit> since <date>'."""
    unit, origin = time_var.attrs['units'].split(' since ')
    return pd.Timestamp(origin) + pd.to_timedelta(time_var[:], unit=unit)


def reduce_region_year(job: dict) -> None:
    """Reduce one region's year of hourly discharge and write it into the region's rows of that year's stores."""
    region, year, (start, end) = job['region'], job['year'], job['rows']
    with zarr.config.set({'async.concurrency': job['threads'], 'threading.max_workers': job['threads']}):
        source = zarr.open_group(str(job['discharge']), mode='r')
        q = source['Q']
        if not np.array_equal(source['river_id'][:], job['river_ids']):
            raise ValueError(f'{region} {year}: {job["discharge"]} does not hold the region in riverIndex order')
        dates = read_dates(source['time'])
        hours = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left')
        if q.shape[1] != dates.size or not np.array_equal(dates, hours[hours.size - dates.size:]):
            raise ValueError(f'{region} {year}: {dates[0]}..{dates[-1]} is not every hour of {year} from its first')

        targets = {name: zarr.open_array(str(spec.store_path(job['out_dir'], store, year) / variable), mode='r+')
                   for name, (store, variable) in REDUCED.items()}
        reduced = {name: np.empty((end - start, array.shape[1]), np.float32) for name, array in targets.items()}
        for r0 in range(0, end - start, ROWS_PER_BLOCK):
            r1 = min(r0 + ROWS_PER_BLOCK, end - start)
            for name, values in spec.reduce_year(q[r0:r1], year).items():
                reduced[name][r0:r1] = values.reshape(r1 - r0, -1)
        for name, (store, variable) in REDUCED.items():
            locks = job['progress'] / 'locks' / f'{store}_{year}' / variable
            spec.write_rows(targets[name], start, reduced[name], locks, job['write_bytes'])
    done_file(job['progress'], year, region).touch()


def fill_timesteps(out_dir: Path, progress: Path, year: int, threads: int) -> None:
    """Fill Q_timesteps in each store of the year that has one. Every region of the year must be written first."""
    with zarr.config.set({'async.concurrency': threads, 'threading.max_workers': threads}):
        for name in STORES:
            if spec.STORES[name][3]:
                spec.write_timesteps(spec.store_path(out_dir, name, year), threads)
    (progress / str(year) / 'Q_timesteps').touch()


if __name__ == '__main__':
    began = time.time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hydrography', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'hydrography')
    parser.add_argument('--routing', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'routing',
                        help='where 1_prepare_hydrography.py wrote each region\'s routing files')
    parser.add_argument('--discharge-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'discharge',
                        help='where 2_route_regions.py wrote the hourly discharge')
    parser.add_argument('--out-dir', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'retrospective')
    parser.add_argument('--processes', type=int, default=os.cpu_count() or 1, help='region years reduced at once')
    parser.add_argument('--threads', type=int, default=1, help='threads zarr reads and writes on in each process')
    parser.add_argument('--write-mb', type=int, default=256, help='rows handed to zarr per write, in MB')
    parser.add_argument('--overwrite', action='store_true', help='delete the stores and start over')
    args = parser.parse_args()

    cores = os.cpu_count() or 1
    if args.processes < 1 or args.threads < 1 or args.processes * args.threads > cores:
        raise SystemExit(f'--processes {args.processes} x --threads {args.threads} must be at least 1 and at most '
                         f'the {cores} cores of this machine')

    regions = sorted(d.name.split('=')[1] for d in args.routing.glob('region=*'))
    regions = [r for r in regions if r != 'global']  # region=global is every river as one network
    river_ids, offsets = spec.global_layout(args.hydrography, args.routing, regions)
    regions.sort(key=lambda r: offsets[r][0] - offsets[r][1])  # biggest first

    routed = {}  # year: {region: its discharge zarr}, for every year a region has finished routing
    for region in regions:
        for path in sorted((args.discharge_root / f'region={region}').glob('discharge_*.zarr')):
            if not (match := DISCHARGE.fullmatch(path.name)):
                raise SystemExit(f'{path} is not the discharge of one calendar year of runoff')
            if (path.parent / f'channel_state_{match["stem"]}.parquet').exists():
                routed.setdefault(int(match['year']), {})[region] = path
    if not routed:
        raise SystemExit(f'no finished years of discharge in {args.discharge_root}')

    progress = args.out_dir / '.progress'
    if args.overwrite:
        for directory in (progress, *(args.out_dir / name for name in STORES)):
            shutil.rmtree(directory, ignore_errors=True)
    jobs = []
    for year, found in sorted(routed.items()):
        (progress / str(year)).mkdir(parents=True, exist_ok=True)
        todo = [r for r in regions if r in found and not done_file(progress, year, r).exists()]
        if not todo:
            continue
        times = spec.record_times(year, year)
        for name in STORES:
            path = spec.store_path(args.out_dir, name, year)
            if not path.exists():  # created under another name and renamed, so a store that exists is whole
                partial = path.with_suffix('.partial.zarr')
                spec.create_store(partial, name, times[name], river_ids)
                os.replace(partial, path)
            elif zarr.open_array(str(path / spec.STORES[name][1][0]), mode='r').shape[0] != river_ids.size:
                raise SystemExit(f'{path} holds other rivers than {args.hydrography}: pass --overwrite')
        jobs += [{'region': r, 'year': year, 'rows': offsets[r], 'river_ids': river_ids[slice(*offsets[r])],
                  'discharge': found[r], 'out_dir': args.out_dir, 'progress': progress, 'threads': args.threads,
                  'write_bytes': args.write_mb * 1_000_000} for r in todo]
    print(f'{len(regions)} regions, {len(routed)} years routed by at least one. Reducing {len(jobs)} region years, '
          f'{args.processes} at a time on {args.threads} threads each, into {args.out_dir}', flush=True)

    failed = []
    with ProcessPoolExecutor(args.processes) as pool:
        futures = {pool.submit(reduce_region_year, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc='region years'):
            try:
                future.result()
            except Exception as error:  # report it now, rather than after every queued region year has run
                job = futures[future]
                failed.append(f'{job["region"]} {job["year"]}')
                tqdm.write(f'{job["region"]} {job["year"]}: failed: {error!r}')

        # a year's Q_timesteps spans every region, so it waits until all of them are written
        complete = [y for y in sorted(routed) if all(done_file(progress, y, r).exists() for r in regions)]
        unfilled = [y for y in complete if not (progress / str(y) / 'Q_timesteps').exists()]
        futures = [pool.submit(fill_timesteps, args.out_dir, progress, y, args.threads) for y in unfilled]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Q_timesteps'):
            future.result()

    print(f'{len(jobs) - len(failed)} region years reduced in {timedelta(seconds=round(time.time() - began))}. '
          f'{len(complete)} of {len(routed)} years have every region', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} region years failed: {", ".join(failed)}')
