"""
Route every member of one IFS ensemble forecast over the whole world at once, and save each member's discharge in the
work directory for 2_discharge_zarr.py to publish:

    <work-dir>/<YYYYMMDDHH>/member_<00..50>.nc    Q (member, riverId, time) float32, every river in riverIndex order

Each is a netCDF file labeled with its coordinates, so it says what it holds without this script: member, its number
in rfs_spec.MEMBERS, riverId, and time, the start of each 3 hour interval. They are uncompressed, since they are only
read once, by 2_discharge_zarr.py, which always deletes them when the store is written.

The forecast is the 51 GRIB files of one initialization in --ifs, <cf_00|pf_01..pf_50>_<YYYYMMDD>.grib, the runoff
of the control and 50 perturbed forecasts of the 00 UTC run on the O1280 grid, accumulated from the initialization. They are read as
they are, by river-route's ecmwf_grib forcing, with the global routing.parquet and gridweights_O1280_global.nc of
1_prepare_inputs. Routing the world at once reads each file once, where routing by region would read each file once
per region, which is what costs the most on networked storage.

IFS runoff is accumulated from the initialization at steps of 1 hour to 90, 3 hours to 144 and 6 hours to 360. The
message at step 0 accumulates nothing and is dropped, and the accumulations are interpolated linearly onto every hour,
which spreads each longer step's runoff evenly over its hours. river-route then de-accumulates them, so every member is
routed hourly, as the retrospective is, on the same stabilized network at the same dt_routing, and a retrospective
channel state can initialize it. The hourly discharge is averaged over 3 hours, the forecast's step.

river-route would resample irregular steps itself, the same way, but after aggregating to catchments, one river at a
time in pandas: 130 s of the 150 s a member took. Every river shares one time axis and aggregation is linear, so
interpolating the grid cells' accumulations first, in one vectorized pass, gives the same runoff.

river-route labels a step by the time it ends, the GRIB validity time. The published forecast labels an interval by
the time it starts: Q at time t is the mean over [t, t + 3 h). Each member's 120 averages, labeled by river-route
1, 4, ... 358 hours after the initialization, are checked to be exactly those and saved in that order, so the first
column is the interval starting at the initialization, and labeled with those times. They are rounded to 13 keepbits
before they are saved.

Every member starts from --init-state, a channel state parquet with one Q per river, or per sub-reach of the stabilized
network, in the order of routing.parquet. The specification initializes the forecasts from the retrospective; without
--init-state the channels start empty, which is right only for testing.

A member whose file exists is not routed again; delete it to route it again. Each file is written to a temporary
name and renamed into place, so an interrupted run never leaves a file that looks finished.
"""

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import river_route as rr
import xarray as xr

import rfs_spec

GRIB_NAME = re.compile(r'(cf|pf)_(\d{2})_(\d{8})\.grib')  # every file is of the 00 UTC run
DT_ROUTING = 3600  # the retrospective's, so the two share channel states
NETWORK_TYPE = 'stabilized'


@dataclass(eq=False, repr=False)
class ForecastRunoff(rr.runoff.ECMWFGribReducedGrid):
    """ECMWFGribReducedGrid for accumulations from ``initialization``, interpolated onto every hour after it."""

    initialization: np.datetime64 | None = field(default=None, init=False)

    def read_runoff(self, runoff_data):
        runoff, dates, conversion_factor = super().read_runoff(runoff_data)
        hours = (dates - self.initialization) / np.timedelta64(1, 'h')
        keep = hours != 0  # step 0, which accumulates nothing
        runoff, hours = runoff[keep], hours[keep]
        if hours[0] <= 0 or np.any(np.diff(hours) <= 0) or np.any(hours % 1):
            raise ValueError(f'{runoff_data} is not accumulated at whole hours after {self.initialization}')
        target = np.arange(hours[0], hours[-1] + 1)
        after = np.searchsorted(hours, target)  # the first step at or after each hour
        before = np.maximum(after - 1, 0)
        span = hours[after] - hours[before]
        share = np.divide(target - hours[before], span, out=np.ones_like(span), where=span > 0).astype(np.float32)
        hourly = np.empty((target.size, runoff.shape[1]), dtype=np.float32)
        for t, (b, a, w) in enumerate(zip(before, after, share, strict=True)):
            np.subtract(runoff[a], runoff[b], out=hourly[t])
            hourly[t] *= w
            hourly[t] += runoff[b]
        return hourly, self.initialization + target.astype('timedelta64[h]'), conversion_factor


def forecast_files(ifs: Path, initialization: pd.Timestamp) -> dict[int, Path]:
    """Each member's GRIB file of one initialization, by member number."""
    if initialization != initialization.normalize():
        raise SystemExit(f'{initialization} is not a 00 UTC run, the only run whose GRIB files are downloaded')
    files = {}
    for path in ifs.glob(f'*_{initialization:%Y%m%d}.grib'):
        if match := GRIB_NAME.fullmatch(path.name):
            files[rfs_spec.member_number(match[1], int(match[2]))] = path
    return dict(sorted(files.items()))


def latest_initialization(ifs: Path) -> pd.Timestamp:
    """The newest initialization with any GRIB file in ``ifs``."""
    found = [GRIB_NAME.fullmatch(p.name) for p in ifs.glob('*.grib')]
    stamps = sorted({pd.Timestamp(m[3]) for m in found if m})
    if not stamps:
        raise SystemExit(f'no IFS runoff files <cf|pf>_<member>_<YYYYMMDD>.grib in {ifs}')
    return stamps[-1]


def check_grid(path: Path) -> None:
    """Refuse a GRIB file that is not on the grid the forecasts are routed on."""
    with open(path, 'rb') as file:
        handle = eccodes.codes_grib_new_from_file(file, headers_only=True)
    try:
        grid = eccodes.codes_get(handle, 'gridName')
    finally:
        eccodes.codes_release(handle)
    if grid != rfs_spec.IFS_GRID:
        raise SystemExit(f'{path} is on the {grid} grid, the forecasts are routed on {rfs_spec.IFS_GRID}')


def member_file(work_dir: Path, member: int) -> Path:
    return work_dir / f'member_{member:02d}.nc'


def write_member(path: Path, member: int, river_ids: np.ndarray, times: pd.DatetimeIndex, values: np.ndarray) -> None:
    """Write one member's (riverId, time) discharge as a labeled netCDF file, in place."""
    dataset = xr.Dataset(
        {'Q': (('member', 'riverId', 'time'), values[np.newaxis], {**rfs_spec.DISCHARGE_ATTRS,
                                                                  'aggregation_method': 'mean',
                                                                  'keepbits': rfs_spec.KEEP_BITS})},
        coords={'member': ('member', np.array([member], dtype=np.int32), {'description': rfs_spec.MEMBER_DESCRIPTION}),
                'riverId': ('riverId', river_ids.astype(np.int32)), 'time': ('time', times)},
    )
    encoding = {'Q': {'dtype': 'float32', 'zlib': False, '_FillValue': np.float32(np.nan)},
                'time': {'dtype': 'int32', 'units': f'hours since {times[0]:%Y-%m-%dT%H:%M:%S}+00:00',
                         'calendar': 'proleptic_gregorian'}}
    partial = path.with_suffix('.partial.nc')
    dataset.to_netcdf(partial, engine='netcdf4', encoding=encoding)
    os.replace(partial, path)
    return


def make_writer(initialization: pd.Timestamp, members: dict[Path, int], river_ids: np.ndarray, began: float):
    """The discharge writer: check the labels are the ends of the first hour of each 3 hour interval, round the values,
    relabel them by the start of each interval, and save them in place."""
    step = np.timedelta64(rfs_spec.FORECAST_STEP_HOURS, 'h')
    times = rfs_spec.forecast_times(initialization)
    first_hour_ends = (times + pd.Timedelta(hours=1)).to_numpy()

    def write(router, dates, discharge, path, runoff_file=''):
        if discharge.shape[1] != rfs_spec.FORECAST_STEPS or not np.array_equal(
                dates.astype('datetime64[s]'), first_hour_ends.astype('datetime64[s]')):
            raise ValueError(f'{runoff_file} routed to {discharge.shape[1]} steps from {dates[0]}, expected '
                             f'{rfs_spec.FORECAST_STEPS} steps of {step} labeled from {first_hour_ends[0]}')
        values = rfs_spec.round_keepbits(np.ascontiguousarray(discharge, dtype=np.float32))
        if nans := int(np.isnan(values).sum()):
            raise ValueError(f'{runoff_file} routed to {nans:,} NaN values')
        if values.shape[0] != river_ids.size:
            raise ValueError(f'{runoff_file} routed {values.shape[0]:,} rivers, routing.parquet has {river_ids.size:,}')
        write_member(Path(path), members[Path(path).resolve()], river_ids, times, values)
        print(f'{Path(runoff_file).name}: {values.shape[0]:,} rivers, max {values.max():,.0f} m3/s, '
              f'{(time.time() - began) / 60:.1f} min', flush=True)

    return write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--initialization', type=pd.Timestamp, default=None,
                        help='the forecast initialization, e.g. 2026-09-27T00; default the newest in --ifs')
    parser.add_argument('--ifs', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'forcings' / 'ifs',
                        help='the IFS runoff GRIB files, <cf_00|pf_NN>_<YYYYMMDD>.grib')
    parser.add_argument('--routing', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'routing',
                        help='the routing files of 1_prepare_inputs, whose global/ is routed')
    parser.add_argument('--work-dir', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'forecasts-work',
                        help='where each member\'s discharge is saved, a folder per initialization')
    parser.add_argument('--init-state', type=Path, default=None,
                        help='channel state parquet every member starts from; default empty channels')
    parser.add_argument('--members', type=int, nargs='+', help='route only these members, 0 the control and 1..50')
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 8, help='routing threads')
    args = parser.parse_args()

    began = time.time()
    initialization = args.initialization or latest_initialization(args.ifs)
    files = forecast_files(args.ifs, initialization)
    wanted = args.members or rfs_spec.MEMBERS.tolist()
    if missing := sorted(set(wanted) - set(files)):
        raise SystemExit(f'{initialization:%Y-%m-%d %H}z has no GRIB file in {args.ifs} for members {missing}')
    work_dir = args.work_dir / f'{initialization:%Y%m%d%H}'
    work_dir.mkdir(parents=True, exist_ok=True)
    todo = [m for m in wanted if not member_file(work_dir, m).exists()]
    print(f'{initialization:%Y-%m-%d %H}z: {len(wanted) - len(todo)} of {len(wanted)} members already routed, '
          f'into {work_dir}', flush=True)
    if not todo:
        raise SystemExit(0)
    check_grid(files[todo[0]])
    if args.init_state is None:
        print('no --init-state: every member starts from empty channels', flush=True)

    configs = rr.Configs(
        params_file=args.routing / 'global' / 'routing.parquet',
        network_type=NETWORK_TYPE,
        coefficients='static',
        forcing='ecmwf_grib',
        transform='uniform',
        dt_routing=DT_ROUTING,
        dt_discharge=rfs_spec.FORECAST_STEP_HOURS * 3600,
        # every member starts from the same state, and none is saved: the next forecast starts from the retrospective
        runoff_processing_mode='ensemble',
        channel_state_init_file=args.init_state,
        runoff_files=[files[m] for m in todo],
        discharge_files=[member_file(work_dir, m) for m in todo],
        grid_weights_file=args.routing / 'global' / f'gridweights_{rfs_spec.IFS_GRID}_global.nc',
        grid_accumulation_type='cumulative',
        var_grid_runoff='ro',
        progress_bar=False,
        log_level='WARNING',
        unstable_coefficients='ignore',
    )
    runoff = ForecastRunoff.from_configs(configs)
    runoff.initialization = initialization.to_datetime64()
    river_ids = pd.read_parquet(configs.params_file, columns=['river_id'])['river_id'].to_numpy()
    members = {member_file(work_dir, m).resolve(): m for m in todo}
    writer = make_writer(initialization, members, river_ids, began)
    router = rr.Router(configs, runoff=runoff).set_discharge_writer(writer)
    print(f'routing {len(todo)} members of {runoff.river_ids.size:,} rivers on {args.threads} threads, weights read '
          f'in {time.time() - began:.0f} s', flush=True)
    with ThreadPoolExecutor(args.threads) as pool:
        router.route(thread_pool=pool, threads=args.threads)
    print(f'{len(todo)} members routed in {(time.time() - began) / 60:.1f} min', flush=True)
