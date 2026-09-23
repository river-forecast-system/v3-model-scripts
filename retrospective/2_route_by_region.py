"""
Route every region of the v3 hydrography over whole calendar years and write each region's rivers straight into the
published stores: hourly.zarr, daily.zarr, monthly.zarr, yearly.zarr and maximums.zarr.

Every published array is chunked as one river over the whole time axis, so a river can only be written once its
whole record exists. Each region therefore routes its years in order, and as each year lands it:

    rounds the year onto the keepbits grid, in place
    reduces it to daily, monthly and yearly means and the annual maxima of the hourly and daily values
    copies it into the region's hourly record

The record and the reductions are memory mapped .npy files under --out-dir. The record is the largest thing in the
build (446 GB for the largest region over 1979-2020), so the operating system pages it to local disk instead of
holding it in memory. After the last year the region writes its rows of every store once and deletes its files.

Each region's rivers are one contiguous run of riverIndex in hydrography/global/metadata.parquet, so its rows
[start, end) are known before anything routes. Only its first and last shard can also hold a neighbor's rivers, and
those two writes take a file lock.

Resuming: after each year the record and reductions are flushed and only then is the channel state saved, so a state
file means that year is complete. A rerun continues each region after its last saved year and skips finished
regions. Once every region is finished, Q_timesteps is filled for monthly and yearly and every store is consolidated.
"""

import argparse
import numpy as np
import os
import pandas as pd
import river_route as rr
import shutil
import time
import warnings
import zarr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

ROWS_PER_BLOCK = 2_000  # rivers reduced together, which bounds the temporaries of a fold
REDUCED = {'daily': ('daily', 'Q'), 'monthly': ('monthly', 'Q'), 'yearly': ('yearly', 'Q'),
           'max_hourly': ('maximums', 'hourly'), 'max_daily': ('maximums', 'daily')}


def open_npy(path: Path, shape: tuple[int, int]) -> np.memmap:
    """Reopen a working file left by an earlier run, or create it sparse."""
    if path.exists():
        array = np.lib.format.open_memmap(path, mode='r+')
        if array.shape != shape:
            raise SystemExit(f'{path} is {array.shape}, expected {shape}: it is from another run, pass --overwrite')
        return array
    return np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)


def route_region(job: dict) -> str:
    """Route one region over every year, then write its rows of every published store."""
    region, work, years, threads = job['region'], job['work_dir'], job['years'], job['threads']
    start, end = job['rows']
    n = end - start
    times = spec.record_times(years[0], years[-1])
    work.mkdir(parents=True, exist_ok=True)
    record = open_npy(work / 'hourly.npy', (n, len(times['hourly'])))
    reduced = {name: open_npy(work / f'{name}.npy', (n, len(times[store])))
               for name, (store, _) in REDUCED.items()}

    def state(year: int) -> Path:
        return work / f'state_{year}.parquet'

    done = 0
    while done < len(years) and state(years[done]).exists():
        done += 1
    began = time.time()

    def write_year(router, dates, discharge, discharge_file, runoff_file='') -> None:
        dates = pd.DatetimeIndex(dates)
        year = dates[0].year
        if dates[0] != pd.Timestamp(f'{year}-01-01') or len(dates) != 24 * (365 + dates[0].is_leap_year):
            raise ValueError(f'{region}: {dates[0]}..{dates[-1]} is not one whole calendar year of hours')
        if discharge.dtype != np.float32:
            raise TypeError(f'{region}: expected float32 discharge, got {discharge.dtype}')
        if year == years[done] and not np.array_equal(np.asarray(router.network.river_ids), job['river_ids']):
            raise ValueError(f'{region}: the router orders its rivers differently than riverIndex')
        t0 = times['hourly'].searchsorted(dates[0])
        d0 = times['daily'].searchsorted(dates[0])
        y = year - years[0]

        def fold(r0: int) -> None:
            r1 = min(r0 + ROWS_PER_BLOCK, n)
            values = spec.reduce_year(discharge[r0:r1], year)
            record[r0:r1, t0:t0 + len(dates)] = discharge[r0:r1]
            reduced['daily'][r0:r1, d0:d0 + values['daily'].shape[1]] = values['daily']
            reduced['monthly'][r0:r1, y * 12:y * 12 + 12] = values['monthly']
            for name in ('yearly', 'max_hourly', 'max_daily'):
                reduced[name][r0:r1, y] = values[name]

        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(fold, range(0, n, ROWS_PER_BLOCK)))
        for array in (record, *reduced.values()):
            array.flush()
        pd.DataFrame({'Q': router.channel_state}).to_parquet(state(year))

    if done < len(years):
        region_dir = job['hydrography'] / f'region={region}'
        todo = years[done:]
        router = rr.Router(rr.Configs(
            params_file=region_dir / 'routing.parquet',
            grid_runoff_files=[job['runoff'][y] for y in todo],
            grid_weights_file=region_dir / f'gridweights_ERA5_{region}.nc',
            # one path per runoff file is required, but write_year replaces the writer that would use them
            discharge_files=[work / f'unused_{y}.zarr' for y in todo],
            channel_state_init_file=state(years[done - 1]) if done else None,
            forcing='vlateral', coeff='static', dt_routing=job['dt_routing'], discharge_dtype='float32',
            runoff_processing_mode='sequential', var_grid_runoff='ro', var_x='longitude', var_y='latitude',
            var_t='valid_time', progress_bar=job['progress'], log_level='ERROR', unstable_coefficients='ignore',
        ))
        router.set_discharge_writer(write_year)
        with ThreadPoolExecutor(threads) as pool:
            router.route(thread_pool=pool, threads=threads)
    routed = time.time()

    locks, published = job['out_dir'] / 'locks', job['global_dir']
    with zarr.config.set({'async.concurrency': threads, 'threading.max_workers': threads}):
        for (store, variable), values in [(('hourly', 'Q'), record), *((REDUCED[k], v) for k, v in reduced.items())]:
            array = zarr.open_array(str(published / f'{store}.zarr' / variable), mode='r+')
            spec.write_rows(array, start, values, locks / store, job['write_bytes'])
    (job['out_dir'] / 'finished' / region).touch()
    shutil.rmtree(work)
    return f'{region}: routed in {(routed - began) / 60:.1f} min, written in {(time.time() - routed) / 60:.1f} min'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Route every region and write the published retrospective stores.')
    parser.add_argument('--hydrography', type=Path, default=Path('/Users/rchales/data/rfsv3/hydrography'))
    parser.add_argument('--runoff', type=Path, default=Path('/Users/rchales/data/era5_zarr_16x16_12month'),
                        help='the one zarr per year written by 1_prepare_runoff.py')
    parser.add_argument('--out-dir', type=Path, default=Path('/Users/rchales/data/discharge_work'),
                        help='working files on fast local disk: records, reductions, channel states, locks')
    parser.add_argument('--global-dir', type=Path, default=Path('/Users/rchales/data/rfsv3/retrospective'),
                        help='where the published stores are written')
    parser.add_argument('--start-year', type=int, default=1979)
    parser.add_argument('--end-year', type=int, default=2020)
    parser.add_argument('--regions', nargs='+', help='route only these regions; the stores still hold every region')
    parser.add_argument('--threads', type=int, default=6, help='threads per region')
    parser.add_argument('--jobs', type=int, default=None, help='regions routed at once, default cores // threads')
    parser.add_argument('--dt-routing', type=int, default=3600)
    parser.add_argument('--write-mb', type=int, default=1024, help='rows handed to zarr per write, in MB')
    parser.add_argument('--overwrite', action='store_true', help='delete the stores and working files and start over')
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    runoff = {y: next(iter(sorted((args.runoff / f'year={y}').glob('*.zarr'))), None) for y in years}
    if missing := [y for y, path in runoff.items() if path is None]:
        raise SystemExit(f'no runoff zarr in {args.runoff} for {missing}')

    all_regions = sorted(d.name.split('=')[1] for d in args.hydrography.glob('region=*'))
    all_regions = [r for r in all_regions if r != 'global']  # region=global is every river as one network
    river_ids, offsets = spec.global_layout(args.hydrography, all_regions)
    regions = [r for r in all_regions if not args.regions or r in args.regions]
    regions.sort(key=lambda r: offsets[r][0] - offsets[r][1])  # biggest first
    times = spec.record_times(years[0], years[-1])
    finished = args.out_dir / 'finished'

    stores = {name: args.global_dir / f'{name}.zarr' for name in spec.STORES}
    if args.overwrite:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    if args.overwrite or not all(path.exists() for path in stores.values()):
        shutil.rmtree(finished, ignore_errors=True)
        for name, path in stores.items():
            spec.create_store(path, name, times[name], river_ids)
    elif zarr.open_array(str(stores['hourly'] / 'Q'), mode='r').shape != (river_ids.size, len(times['hourly'])):
        raise SystemExit(f'the stores in {args.global_dir} are for other years or rivers: pass --overwrite')
    finished.mkdir(parents=True, exist_ok=True)
    for name in spec.STORES:
        (args.out_dir / 'locks' / name).mkdir(parents=True, exist_ok=True)

    regions = [r for r in regions if not (finished / r).exists()]
    jobs = args.jobs or max(1, (os.cpu_count() or 8) // args.threads)
    peak_disk = sum(offsets[r][1] - offsets[r][0] for r in regions[:jobs]) * len(times['hourly']) * 4
    print(f'{len(regions)} regions to route, {jobs} at a time on {args.threads} threads, {years[0]}..{years[-1]}. '
          f'The {jobs} largest need {peak_disk / 1e9:,.0f} GB of working disk; '
          f'{shutil.disk_usage(args.out_dir).free / 1e9:,.0f} GB is free', flush=True)

    with ProcessPoolExecutor(jobs) as pool:
        futures = [pool.submit(route_region, {
            'region': r, 'rows': offsets[r], 'river_ids': river_ids[slice(*offsets[r])], 'years': years,
            'runoff': runoff, 'hydrography': args.hydrography, 'work_dir': args.out_dir / f'region={r}',
            'out_dir': args.out_dir, 'global_dir': args.global_dir, 'threads': args.threads,
            'dt_routing': args.dt_routing, 'write_bytes': args.write_mb * 1_000_000, 'progress': jobs == 1,
        }) for r in regions]
        for future in as_completed(futures):
            print(future.result(), flush=True)

    if unfinished := [r for r in all_regions if not (finished / r).exists()]:
        raise SystemExit(f'{len(unfinished)} regions are not written yet, so the stores are not finished')
    for name, path in stores.items():
        if spec.STORES[name][3]:
            spec.write_timesteps(path, os.cpu_count() or 8)
        zarr.consolidate_metadata(str(path))
    shutil.rmtree(args.out_dir / 'locks')
    print(f'published stores finished in {args.global_dir}')
