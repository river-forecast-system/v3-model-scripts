"""
Route every region of the v3 hydrography one calendar year at a time, concatenate the regions in memory, and write
each year into the published stores: hourly.zarr, daily.zarr, monthly.zarr, yearly.zarr and maximums.zarr.

For each year, in order:

    route every region concurrently, each from its channel state after the previous year, each writing its rivers
        into its rows of one global (river, hour) buffer in shared memory, which is the concatenation
    round the global year onto the keepbits grid and reduce it to daily, monthly and yearly means and annual maxima
    write the year's columns of every store

The global buffer is every river for one year: 173 GB float32 for 4,917,183 rivers in a leap year, plus what the
routers hold while they run. Each region's rivers are one contiguous run of riverIndex in
hydrography/global/metadata.parquet, so its rows [start, end) of the buffer are known before anything routes.

Resuming: a year counts once its columns of every store are written and it is recorded in --out-dir/last_year. A rerun
starts at the year after it, from the channel states saved after that year. After the last year, Q_timesteps is
filled for monthly and yearly and every store is consolidated.
"""

import argparse
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pandas as pd
import river_route as rr
import zarr

import spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

ROWS_PER_BLOCK = 2_000  # rivers reduced together, which bounds the temporaries of a reduction


def state_path(out_dir: Path, region: str, year: int) -> Path:
    return out_dir / 'states' / f'region={region}' / f'state_{year}.parquet'


def route_region_year(job: dict) -> str:
    """Route one region over one year into its rows of the shared global buffer, and save its channel state."""
    region, year, threads = job['region'], job['year'], job['threads']
    start, end = job['rows']
    region_dir = job['hydrography'] / f'region={region}'
    previous = state_path(job['out_dir'], region, year - 1) if not job['first'] else None
    if previous is not None and not previous.exists():
        raise FileNotFoundError(f'{region}: no channel state after {year - 1} to route {year} from')
    shm = SharedMemory(name=job['shm'], track=False)
    try:
        buffer = np.ndarray(job['shape'], dtype=np.float32, buffer=shm.buf)

        def write_year(router, dates, discharge, discharge_file, runoff_file='') -> None:
            if len(dates) != job['shape'][1]:
                raise ValueError(f'{region}: routed {len(dates)} steps for a year of {job["shape"][1]} hours')
            if not np.array_equal(np.asarray(router.network.river_ids), job['river_ids']):
                raise ValueError(f'{region}: the router orders its rivers differently than riverIndex')
            buffer[start:end] = discharge

        router = rr.Router(rr.Configs(
            params_file=region_dir / 'routing.parquet',
            grid_runoff_files=[job['runoff']],
            grid_weights_file=region_dir / f'gridweights_ERA5_{region}.nc',
            # a path is required, but write_year replaces the writer that would use it
            discharge_files=[job['out_dir'] / f'unused_{region}_{year}.zarr'],
            channel_state_init_file=previous,
            forcing='vlateral', coeff='static', dt_routing=job['dt_routing'], discharge_dtype='float32',
            runoff_processing_mode='sequential', var_grid_runoff='ro', var_x='longitude', var_y='latitude',
            var_t='valid_time', progress_bar=False, log_level='ERROR', unstable_coefficients='ignore',
        ))
        router.set_discharge_writer(write_year)
        with ThreadPoolExecutor(threads) as pool:
            router.route(thread_pool=pool, threads=threads)
        del buffer
    finally:
        shm.close()
    path = state_path(job['out_dir'], region, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Q': router.channel_state}).to_parquet(path)
    return region


def write_columns(array, t0: int, values: np.ndarray, write_bytes: int) -> None:
    """Write ``values`` into columns [t0, t0 + width) of every row, a block of whole shards of rivers at a time."""
    values = values.reshape(values.shape[0], -1)
    width = array.shards[0]
    rows = width * max(1, write_bytes // (width * array.shape[1] * 4))
    for r0 in range(0, values.shape[0], rows):
        array[r0:r0 + rows, t0:t0 + values.shape[1]] = values[r0:r0 + rows]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Route every region a year at a time and write the published stores.')
    parser.add_argument('--hydrography', type=Path, default=Path('/Users/rchales/data/rfsv3/hydrography'))
    parser.add_argument('--runoff', type=Path, default=Path('/Users/rchales/data/era5_zarr_16x16_12month'),
                        help='the one zarr per year written by 1_prepare_runoff.py')
    parser.add_argument('--out-dir', type=Path, default=Path('/Users/rchales/data/discharge_work'),
                        help='channel states and the last year written')
    parser.add_argument('--global-dir', type=Path, default=Path('/Users/rchales/data/rfsv3/retrospective'),
                        help='where the published stores are written')
    parser.add_argument('--start-year', type=int, default=1979)
    parser.add_argument('--end-year', type=int, default=2020)
    parser.add_argument('--threads', type=int, default=4, help='threads per region')
    parser.add_argument('--jobs', type=int, default=None, help='regions routed at once, default cores // threads')
    parser.add_argument('--dt-routing', type=int, default=3600)
    parser.add_argument('--write-mb', type=int, default=1024, help='rows handed to zarr per write, in MB')
    parser.add_argument('--overwrite', action='store_true', help='delete the stores and working files and start over')
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    runoff = {y: next(iter(sorted((args.runoff / f'year={y}').glob('*.zarr'))), None) for y in years}
    if missing := [y for y, path in runoff.items() if path is None]:
        raise SystemExit(f'no runoff zarr in {args.runoff} for {missing}')

    regions = [d.name.split('=')[1] for d in args.hydrography.glob('region=*') if d.name != 'region=global']
    river_ids, offsets = spec.global_layout(args.hydrography, regions)
    regions.sort(key=lambda r: offsets[r][0] - offsets[r][1])  # biggest first
    n_rivers = river_ids.size
    times = spec.record_times(years[0], years[-1])
    last_year = args.out_dir / 'last_year'

    stores = {name: args.global_dir / f'{name}.zarr' for name in spec.STORES}
    if args.overwrite:
        shutil.rmtree(args.out_dir, ignore_errors=True)
    if args.overwrite or not all(path.exists() for path in stores.values()):
        last_year.unlink(missing_ok=True)
        for name, path in stores.items():
            spec.create_store(path, name, times[name], river_ids)
    elif zarr.open_array(str(stores['hourly'] / 'Q'), mode='r').shape != (n_rivers, len(times['hourly'])):
        raise SystemExit(f'the stores in {args.global_dir} are for other years or rivers: pass --overwrite')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [y for y in years if y > int(last_year.read_text())] if last_year.exists() else years
    if not todo:
        print(f'every year is already written; finishing the stores in {args.global_dir}')

    cores = os.cpu_count() or 8
    jobs = args.jobs or max(1, cores // args.threads)
    write_bytes = args.write_mb * 1_000_000
    print(f'{len(regions)} regions, {n_rivers:,} rivers, {len(todo)} years to route, '
          f'{jobs} regions at a time on {args.threads} threads. The global year buffer is '
          f'{n_rivers * 8784 * 4 / 1e9:,.0f} GB', flush=True)

    shm = SharedMemory(create=True, size=n_rivers * 8784 * 4)
    try:
        with ProcessPoolExecutor(jobs) as pool:
            for year in todo:
                began = time.time()
                hours = 24 * (365 + pd.Timestamp(f'{year}-01-01').is_leap_year)
                buffer = np.ndarray((n_rivers, hours), dtype=np.float32, buffer=shm.buf)
                buffer[:] = np.nan
                list(pool.map(route_region_year, [{
                    'region': r, 'year': year, 'rows': offsets[r], 'river_ids': river_ids[slice(*offsets[r])],
                    'runoff': runoff[year], 'hydrography': args.hydrography, 'out_dir': args.out_dir,
                    'shm': shm.name, 'shape': (n_rivers, hours), 'threads': args.threads,
                    'dt_routing': args.dt_routing, 'first': year == years[0],
                } for r in regions]))
                if np.isnan(buffer[:, -1]).any():
                    raise SystemExit(f'{year}: some rivers were not routed')
                routed = time.time()

                def reduce(r0: int) -> dict[str, np.ndarray]:
                    return spec.reduce_year(buffer[r0:r0 + ROWS_PER_BLOCK], year)

                with ThreadPoolExecutor(cores) as threads:
                    blocks = list(threads.map(reduce, range(0, n_rivers, ROWS_PER_BLOCK)))
                reduced = {name: np.concatenate([b[name] for b in blocks]) for name in blocks[0]}
                del blocks

                t0 = times['hourly'].searchsorted(pd.Timestamp(f'{year}-01-01'))
                d0 = times['daily'].searchsorted(pd.Timestamp(f'{year}-01-01'))
                y = year - years[0]
                columns = [
                    ('hourly', 'Q', t0, buffer), ('daily', 'Q', d0, reduced['daily']),
                    ('monthly', 'Q', y * 12, reduced['monthly']), ('yearly', 'Q', y, reduced['yearly']),
                    ('maximums', 'hourly', y, reduced['max_hourly']), ('maximums', 'daily', y, reduced['max_daily']),
                ]
                with zarr.config.set({'async.concurrency': cores, 'threading.max_workers': cores}):
                    for store, variable, start, values in columns:
                        array = zarr.open_array(str(stores[store] / variable), mode='r+')
                        write_columns(array, int(start), values, write_bytes)
                del buffer, reduced
                last_year.write_text(str(year))
                for r in regions:
                    state_path(args.out_dir, r, year - 1).unlink(missing_ok=True)
                print(f'{year}: routed in {(routed - began) / 60:.1f} min, written in '
                      f'{(time.time() - routed) / 60:.1f} min', flush=True)
    finally:
        shm.close()
        shm.unlink()

    for name, path in stores.items():
        if spec.STORES[name][3]:
            spec.write_timesteps(path, cores)
        zarr.consolidate_metadata(str(path))
    print(f'published stores finished in {args.global_dir}')
