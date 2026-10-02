"""
Route every region of the v3 hydrography over the ERA5 record, one calendar year at a time, with river-route's stock
Router and zarr writer, the way river-route/examples/example_usage.py routes one region.

Each region routes the years --first-year..--last-year in order, one Router per year, each starting from the channel
state the year before it ended with. A year's runoff is --runoff-root/<yyyy>.zarr, or every zarr in
--runoff-root/year=<yyyy>/, one per quarter or one for the year, routed in order by the one Router, and its discharge
is written as one store for the whole year:

    <discharge-root>/region=<id>/discharge_era5_194001_194012.zarr         one per year: 85 for 1940..2024
    <discharge-root>/region=<id>/channel_state_era5_194001_194012.parquet  the channel state at the end of that year

Each region is warmed up first: its first --warm-up years are routed from empty channels, their discharge discarded,
and the channel state they end with saved as

    <discharge-root>/region=<id>/channel_state_warm_up.parquet

The record then starts at --first-year from that state, so its first year starts from full channels. The warm-up ends
on 31 December and the record starts on 1 January, so the state is of the same season, and no runoff before the record
is needed, so a record can start with ERA5 in 1940.

--processes regions route at once, biggest first, each on --threads threads, so processes * threads may not exceed
the machine's cores. A year's state is written only after its discharge, so a rerun resumes each region after its
last saved state and skips any region that has the last year's.
"""

import argparse
import contextlib
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import river_route as rr
from tqdm import tqdm

import rfs_spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

dt_routing = 3600
network_conditioning = 'stabilized'
# what river-route's zarr writer compresses and rounds with, set on import so every worker process sets it too
rr.router.writers.ZARR_COMPRESSOR = rfs_spec.COMPRESSOR
rr.router.writers.ZARR_KEEPBITS = rfs_spec.KEEP_BITS


def year_stem(year: int) -> str:
    """The name of one year's outputs, as if routed from one yearly runoff zarr named era5_<yyyy>01_<yyyy>12."""
    return f'era5_{year}01_{year}12'


def state_file(discharge_dir: Path, year: int) -> Path:
    """The channel state at the end of one year, named to pair with the discharge_<stem>.zarr it wrote."""
    return discharge_dir / f'channel_state_{year_stem(year)}.parquet'


def discharge_file(discharge_dir: Path, year: int) -> Path:
    return discharge_dir / f'discharge_{year_stem(year)}.zarr'


def warm_up_file(discharge_dir: Path) -> Path:
    """The channel state the region's warm-up ends with, which its first year starts from."""
    return discharge_dir / 'channel_state_warm_up.parquet'


def make_config(region_dir: Path, discharge_dir: Path, runoff_files: list[Path], init_state: Path | None,
                final_state: Path) -> rr.Configs:
    region = region_dir.name.split('=')[1]
    return rr.Configs(
        # the network/routing parameters file
        params_file=region_dir / 'routing.parquet',
        network_type=network_conditioning,
        # primary modeling choices
        coefficients='static',
        forcing='grid',
        transform='uniform',
        dt_routing=dt_routing,
        # model state files: start where the year before ended, and save where this one ends
        channel_state_init_file=init_state,
        channel_state_final_file=final_state,
        # the Router names an output per runoff file here; the year's writer writes discharge_file() instead
        discharge_dir=discharge_dir,
        # the runoff forcing data and how to read it, one year at a time
        runoff_files=runoff_files,
        grid_weights_file=region_dir / f'gridweights_ERA5_{region}.nc',
        var_grid_runoff='ro',
        var_x='longitude',
        var_y='latitude',
        var_t='valid_time',
        # logging and validation options
        progress_bar=False,
        log_level='ERROR',
        unstable_coefficients='ignore',
    )


def year_writer(path: Path | None, n_files: int):
    """
    A discharge writer that holds each runoff file's discharge until the year's last and then writes them as one
    store at ``path`` with river-route's zarr writer. With ``path`` None it writes nothing, for a warm-up year.
    """
    parts = []

    def write(router, dates, discharge_array, _discharge_file, runoff_file=''):
        if path is None:
            return
        parts.append((dates, discharge_array))  # the Router allocates a new array for every file
        if len(parts) < n_files:
            return
        all_dates = np.concatenate([d for d, _ in parts])
        all_q = parts[0][1] if n_files == 1 else np.concatenate([q for _, q in parts], axis=1)
        parts.clear()
        rr.router.writers.zarr_writer(router, all_dates, all_q, path, Path(runoff_file).parent)

    return write


def route_region(region: str, args: argparse.Namespace, runoff_files: dict[int, list[Path]]) -> float:
    """
    Warm one region up, unless a year is already routed or its warm-up state is saved, then route its record a year
    at a time after its last saved year, on args.threads threads. Returns the minutes it took.
    """
    began = time.time()
    region_dir = args.routing / f'region={region}'
    discharge_dir = args.discharge_root / f'region={region}'
    discharge_dir.mkdir(parents=True, exist_ok=True)
    years = list(runoff_files)
    done = next((i for i, y in enumerate(years) if not state_file(discharge_dir, y).exists()), len(years))
    warm_up = [] if done or warm_up_file(discharge_dir).exists() else years[:args.warm_up]
    print(f'{region}: warming up {len(warm_up)} years, then routing {len(years) - done} of {len(years)}', flush=True)

    network, runoff = None, None  # parsed by the first year's Router and reused by every year after it

    def route(year: int, init_state: Path | None, final_state: Path, path: Path | None) -> None:
        nonlocal network, runoff
        files = runoff_files[year]
        conf = make_config(region_dir, discharge_dir, files, init_state, final_state)
        router = rr.Router(conf, network=network, runoff=runoff).set_discharge_writer(year_writer(path, len(files)))
        router.route(thread_pool=pool, threads=args.threads)
        network, runoff = router.network, router.runoff

    pool_context = ThreadPoolExecutor(args.threads) if args.threads > 1 else contextlib.nullcontext()
    with pool_context as pool:
        # each warm-up year starts from the state the one before saved, under a temporary name until the last
        warming = discharge_dir / 'channel_state_warm_up.partial.parquet'
        for i, year in enumerate(warm_up):
            route(year, warming if i else None, warming, None)
        if warm_up:
            os.replace(warming, warm_up_file(discharge_dir))

        previous = state_file(discharge_dir, years[done - 1]) if done else \
            warm_up_file(discharge_dir) if args.warm_up else None
        bar = tqdm(years[done:], desc=region, initial=done, total=len(years),
                   disable=args.processes > 1)  # a bar per process would garble the terminal
        for year in bar:
            route(year, previous, state_file(discharge_dir, year), discharge_file(discharge_dir, year))
            previous = state_file(discharge_dir, year)
    return (time.time() - began) / 60


if __name__ == '__main__':
    began = time.time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--routing', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'routing',
                        help='where 1_prepare_inputs/ wrote each region\'s routing files')
    parser.add_argument('--runoff-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'forcings' / 'era5',
                        help='ERA5 runoff zarrs, <yyyy>.zarr or year=<yyyy>/ folders of one or several each')
    parser.add_argument('--discharge-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'discharge')
    parser.add_argument('--first-year', type=int, default=1940)
    parser.add_argument('--last-year', type=int, default=2024)
    parser.add_argument('--warm-up', type=int, default=2,
                        help='first years of the record each region routes from empty channels, discharge discarded, '
                             'to start the record from the state they end with; 0 starts it from empty channels')
    parser.add_argument('--regions', nargs='*', help='route only these region ids')
    parser.add_argument('--processes', type=int, default=16, help='regions routed at once')
    parser.add_argument('--threads', type=int, default=2, help='threads each region routes on')
    args = parser.parse_args()

    cores = os.cpu_count() or 1
    if args.processes < 1 or args.threads < 1 or args.processes * args.threads > cores:
        raise SystemExit(f'--processes {args.processes} x --threads {args.threads} must be at least 1 and at most '
                         f'the {cores} cores of this machine')

    # the runoff zarrs of each year, oldest first
    runoff_files = {y: sorted(args.runoff_root.glob(f'year={y}/*.zarr')) or sorted(args.runoff_root.glob(f'{y}.zarr'))
                    for y in range(args.first_year, args.last_year + 1)}
    if not 0 <= args.warm_up <= len(runoff_files):
        raise SystemExit(f'--warm-up {args.warm_up} must be 0..{len(runoff_files)}, the years of the record')
    if missing := [y for y, files in runoff_files.items() if not files]:
        raise SystemExit(f'no runoff zarrs for {missing[0]} in {args.runoff_root}')
    regions = sorted(d.name.split('=')[1] for d in args.routing.glob('region=*'))
    regions = [r for r in regions if r != 'global']  # region=global is every river as one network
    if args.regions:
        regions = [r for r in regions if r in args.regions]
    todo = [r for r in regions if not state_file(args.discharge_root / f'region={r}', args.last_year).exists()]
    # biggest first, so the longest regions are not the last ones still running
    todo.sort(key=lambda r: -pq.read_metadata(args.routing / f'region={r}' / 'routing.parquet').num_rows)
    print(f'{len(regions)} regions, {len(regions) - len(todo)} already routed, {args.first_year}..{args.last_year} '
          f'after a {args.warm_up} year warm-up. Routing {len(todo)}, {args.processes} at a time on {args.threads} '
          f'threads each', flush=True)

    failed = []
    with ProcessPoolExecutor(args.processes) as pool:
        futures = {pool.submit(route_region, region, args, runoff_files): region for region in todo}
        for future in as_completed(futures):
            region = futures[future]
            try:
                print(f'{region}: routed in {future.result():.1f} min', flush=True)
            except Exception as error:  # report it now, rather than after every queued region has run
                failed.append(region)
                print(f'{region}: failed: {error!r}', flush=True)
    summary = (f'{len(todo) - len(failed)} regions routed in {timedelta(seconds=round(time.time() - began))}, '
               f'{args.first_year}..{args.last_year} after a {args.warm_up} year warm-up, '
               f'{args.processes} processes x {args.threads} threads')
    print(summary, flush=True)
    # appended, so a rerun that resumes adds its own line rather than replacing the first run's
    args.discharge_root.mkdir(parents=True, exist_ok=True)
    with open(args.discharge_root / 'routing_time.txt', 'a') as log:
        log.write(f'{datetime.now():%Y-%m-%d %H:%M:%S}  {summary}\n')
    if failed:
        raise SystemExit(f'{len(failed)} regions failed: {" ".join(failed)}')
