"""
Route every region of the v3 hydrography over the ERA5 record, one calendar year at a time, with river-route's stock
Router and zarr writer, the way river-route/examples/example_usage.py routes one region.

Each region routes the yearly runoff zarrs in order, one Router per year, each starting from the channel state the
year before it ended with. Every year writes its discharge and the channel state it ended with:

    <discharge-root>/region=<id>/discharge_era5_194001_194012.zarr         one per runoff zarr: 85 for 1940..2024
    <discharge-root>/region=<id>/channel_state_era5_194001_194012.parquet  the channel state at the end of that year

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

import pyarrow.parquet as pq
import river_route as rr
from tqdm import tqdm

import spec

warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)

dt_routing = 3600
network_conditioning = 'stabilized'
# what river-route's zarr writer compresses with, set on import so every worker process sets it too
rr.router.writers.ZARR_COMPRESSOR = spec.COMPRESSOR


def state_file(discharge_dir: Path, runoff_file: Path) -> Path:
    """The channel state at the end of one runoff file, named to pair with the discharge_<runoff file> it wrote."""
    return discharge_dir / f'channel_state_{runoff_file.stem}.parquet'


def make_config(region_dir: Path, discharge_dir: Path, runoff_file: Path, init_state: Path | None) -> rr.Configs:
    region = region_dir.name.split('=')[1]
    return rr.Configs(
        # the network/routing parameters file
        params_file=region_dir / 'routing.parquet',
        network_type=network_conditioning,
        # primary modeling choices
        coefficients='static',
        forcing='runoff',
        transform='uniform',
        dt_routing=dt_routing,
        # model state files: start where the year before ended, and save where this one ends
        channel_state_init_file=init_state,
        channel_state_final_file=state_file(discharge_dir, runoff_file),
        # where to place the results: discharge_<runoff file name>
        discharge_dir=discharge_dir,
        # the runoff forcing data and how to read it, one year at a time
        runoff_type='gaussian_grid',
        runoff_files=[runoff_file],
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


def route_region(region: str, args: argparse.Namespace, runoff_files: list[Path]) -> float:
    """Route one region a year at a time on args.threads threads, after its last saved year, and return the minutes."""
    began = time.time()
    region_dir = args.routing / f'region={region}'
    discharge_dir = args.discharge_root / f'region={region}'
    discharge_dir.mkdir(parents=True, exist_ok=True)
    done = next((i for i, f in enumerate(runoff_files) if not state_file(discharge_dir, f).exists()), len(runoff_files))
    previous = state_file(discharge_dir, runoff_files[done - 1]) if done else None
    print(f'{region}: routing {len(runoff_files) - done} of {len(runoff_files)} years', flush=True)

    network, runoff = None, None  # parsed by the first year's Router and reused by every year after it
    pool_context = ThreadPoolExecutor(args.threads) if args.threads > 1 else contextlib.nullcontext()
    with pool_context as pool:
        years = tqdm(runoff_files[done:], desc=region, initial=done, total=len(runoff_files),
                     disable=args.processes > 1)  # a bar per process would garble the terminal
        for runoff_file in years:
            conf = make_config(region_dir, discharge_dir, runoff_file, previous)
            router = rr.Router(conf, network=network, runoff=runoff).route(thread_pool=pool, threads=args.threads)
            network, runoff, previous = router.network, router.runoff, conf.channel_state_final_file
    return (time.time() - began) / 60


if __name__ == '__main__':
    began = time.time()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--routing', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'routing',
                        help='where 1_prepare_hydrography.py wrote each region\'s routing files')
    parser.add_argument('--runoff-root', type=Path, default=Path.home() / 'data' / 'era5_zarr_16x16_12month')
    parser.add_argument('--discharge-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'discharge')
    parser.add_argument('--processes', type=int, default=16, help='regions routed at once')
    parser.add_argument('--threads', type=int, default=2, help='threads each region routes on')
    args = parser.parse_args()

    cores = os.cpu_count() or 1
    if args.processes < 1 or args.threads < 1 or args.processes * args.threads > cores:
        raise SystemExit(f'--processes {args.processes} x --threads {args.threads} must be at least 1 and at most '
                         f'the {cores} cores of this machine')

    runoff_files = sorted(args.runoff_root.glob('year=194[0-4]/*.zarr'))  # one zarr per calendar year, oldest first
    if not runoff_files:
        raise SystemExit(f'no runoff zarrs in {args.runoff_root}')
    regions = sorted(d.name.split('=')[1] for d in args.routing.glob('region=*'))
    regions = [r for r in regions if r != 'global']  # region=global is every river as one network
    todo = [r for r in regions if not state_file(args.discharge_root / f'region={r}', runoff_files[-1]).exists()]
    # biggest first, so the longest regions are not the last ones still running
    todo.sort(key=lambda r: -pq.read_metadata(args.routing / f'region={r}' / 'routing.parquet').num_rows)
    print(f'{len(regions)} regions, {len(regions) - len(todo)} already routed, {len(runoff_files)} years of runoff '
          f'each. Routing {len(todo)}, {args.processes} at a time on {args.threads} threads each', flush=True)

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
               f'{len(runoff_files)} years each, {args.processes} processes x {args.threads} threads')
    print(summary, flush=True)
    # appended, so a rerun that resumes adds its own line rather than replacing the first run's
    args.discharge_root.mkdir(parents=True, exist_ok=True)
    with open(args.discharge_root / 'routing_time.txt', 'a') as log:
        log.write(f'{datetime.now():%Y-%m-%d %H:%M:%S}  {summary}\n')
    if failed:
        raise SystemExit(f'{len(failed)} regions failed: {" ".join(failed)}')
