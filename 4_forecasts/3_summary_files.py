"""
Compare one forecast's discharge.zarr, from 2_discharge_zarr.py, to the retrospective's return periods and flow
duration curves, and write the summary files the web app and the alert feeds read, beside the store:

    <out-root>/year=YYYY/month=MM/day=DD/maps/timeseries/styles.{json,bin}      return period and size, every step
    <out-root>/year=YYYY/month=MM/day=DD/maps/max-flow/styles.{json,bin}        the same at each river's forecast peak
    <out-root>/year=YYYY/month=MM/day=DD/maps/time-to-peak/styles.{json,bin}    the step of each river's forecast peak
    <out-root>/year=YYYY/month=MM/day=DD/maps/below-q95/styles.{json,bin}       whether the forecast falls below Q95
    <out-root>/year=YYYY/month=MM/day=DD/alerts.csv                            rivers likely to exceed a return period

The format of the stylesets is the specification's "Forecast map stylesets", written by rfs_spec.write_styles, and
the alerts are its "Forecast alerts", with the rules in rfs_spec. The web app reads each styleset's bytes by
riverIndex, so every array here is in that order, the riverId axis of the store.

The stylesets read the ensemble median, Qpercentiles at 50, which is what the web app's own forecast classification
reads. Its return period classes are those of gumbel_hourly in retrospective/return-periods.zarr, and below-q95
compares the median's lowest value over the horizon with hourly_annual at p_exceed 95 in retrospective/fdc.zarr, the
flow a river exceeds 95% of the time. The alerts read every member: a return period is likely exceeded at a step when
at least rfs_spec.ALERT_PROBABILITY of the members exceed it then, as the web app's exceedance tables count.

The rivers are processed a block of whole shards at a time, --jobs blocks at once, and only the bytes and the alert
rows are kept, about 0.6 GB for the world. Every file is written to a temporary name and renamed into place.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr

import rfs_spec

_open = {}  # each process's arrays, opened on its first block


def open_arrays(store: Path, retrospective: Path) -> dict:
    """The arrays every block reads, and the columns of the thresholds they compare with."""
    forecast = zarr.open_group(str(store), mode='r')
    periods = zarr.open_group(str(retrospective / 'return-periods.zarr'), mode='r')
    fdc = zarr.open_group(str(retrospective / 'fdc.zarr'), mode='r')
    intervals = periods['recurrence_interval'][:]
    if np.any(np.diff(intervals) <= 0):
        raise ValueError(f'recurrence_interval must ascend, got {intervals.tolist()}')
    style_columns = [int(np.flatnonzero(np.isclose(intervals, ri))[0]) for ri in rfs_spec.STYLE_RETURN_PERIODS]
    median = int(np.flatnonzero(forecast['percentiles'][:] == 50)[0])
    q95 = int(np.flatnonzero(fdc['p_exceed'][:] == 95)[0])
    return {'Q': forecast['Q'], 'Qpercentiles': forecast['Qpercentiles'], 'median': median,
            'gumbel': periods['gumbel_hourly'], 'intervals': intervals, 'style_columns': style_columns,
            'fdc': fdc['hourly_annual'], 'q95': q95}


def summarize_block(job: tuple[Path, Path, int, int]) -> tuple[int, dict, dict]:
    """The styleset bytes and the alert rows of rivers [r0, r1)."""
    store, retrospective, r0, r1 = job
    if store not in _open:
        _open.clear()
        _open[store] = open_arrays(store, retrospective)
    a = _open[store]
    median = a['Qpercentiles'][r0:r1, a['median'], :]  # (river, time)
    members = a['Q'][r0:r1]  # (river, member, time)
    periods = a['gumbel'][r0:r1].astype(np.float32)  # (river, recurrence_interval)
    q95 = a['fdc'][r0:r1, a['q95']]

    style_thresholds = periods[:, a['style_columns']]
    peak = np.nanmax(median, axis=1)
    has_flow = np.isfinite(peak) & (peak > 0)
    peak_step = np.argmax(np.nan_to_num(median, nan=-1), axis=1)  # the first step of the median's largest value
    time_to_peak = np.where(has_flow, peak_step, rfs_spec.STYLE_NO_DATA)
    lowest = np.nanmin(median, axis=1)
    below = np.where(np.isfinite(q95) & np.isfinite(lowest), (lowest < q95).astype(np.uint8), rfs_spec.STYLE_NO_DATA)
    styles = {
        'timeseries': rfs_spec.return_period_byte(median, style_thresholds),
        'max-flow': rfs_spec.return_period_byte(peak, style_thresholds)[:, np.newaxis],
        'time-to-peak': time_to_peak.astype(np.uint8)[:, np.newaxis],
        'below-q95': below.astype(np.uint8)[:, np.newaxis],
    }

    # the share of members above each return period flow at each step, (river, recurrence_interval, time)
    share = np.stack([(members > periods[:, [k], np.newaxis]).mean(axis=1) for k in range(periods.shape[1])], axis=1)
    likely = share >= rfs_spec.ALERT_PROBABILITY
    reached = likely.any(axis=2)  # (river, recurrence_interval)
    alerting = np.flatnonzero(reached.any(axis=1))
    level = periods.shape[1] - 1 - np.argmax(reached[alerting, ::-1], axis=1)  # the largest interval reached
    alerts = {
        'riverIndex': r0 + alerting,
        'recurrence_interval': a['intervals'][level],
        'return_period_flow': periods[alerting, level],
        'exceedance_probability': share[alerting, level].max(axis=1),
        'onset_step': np.argmax(likely[alerting, level], axis=1),
        'peak_step': peak_step[alerting],
        'peak_median_flow': peak[alerting],
    }
    return r0, styles, alerts


def alert_table(alerts: pd.DataFrame, initialization: pd.Timestamp, hydrography: Path) -> pd.DataFrame:
    """The alert rows with the rest of their CAP fields, the river's id and outlet, and times as UTC strings."""
    metadata = pq.read_table(hydrography / 'global' / 'metadata.parquet',
                             columns=['riverId', 'riverIndex', 'lat', 'lon']).to_pandas().set_index('riverIndex')
    alerts = alerts.join(metadata, on='riverIndex')
    times = rfs_spec.forecast_times(initialization)
    horizon = pd.Timedelta(hours=rfs_spec.FORECAST_STEPS * rfs_spec.FORECAST_STEP_HOURS)
    onset_hours = alerts['onset_step'] * rfs_spec.FORECAST_STEP_HOURS
    urgency = np.full(len(alerts), 'Future', dtype=object)
    for hours, name in reversed(rfs_spec.ALERT_URGENCY_HOURS):
        urgency[onset_hours.to_numpy() <= hours] = name
    stamp = '%Y-%m-%dT%H:%M:%SZ'
    alerts = alerts.assign(
        identifier=[f'rfs-v3-{initialization:%Y%m%d%H}-{river}' for river in alerts['riverId']],
        sent=f'{initialization:{stamp}}',
        event='River flood',
        urgency=urgency,
        severity=alerts['recurrence_interval'].map(lambda ri: rfs_spec.ALERT_SEVERITY[float(ri)]),
        certainty=np.where(alerts['exceedance_probability'] >= rfs_spec.ALERT_LIKELY, 'Likely', 'Possible'),
        onset=times[alerts['onset_step']].strftime(stamp),
        expires=f'{initialization + horizon:{stamp}}',
        peak_time=times[alerts['peak_step']].strftime(stamp),
    )
    return alerts[list(rfs_spec.ALERT_COLUMNS)].sort_values('riverIndex', ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--initialization', type=pd.Timestamp, required=True,
                        help='the forecast initialization, e.g. 2026-09-27T00')
    parser.add_argument('--out-root', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'forecasts15',
                        help='the forecasts15/ tree the store is in and the summary files are written into')
    parser.add_argument('--retrospective', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'retrospective',
                        help='the retrospective stores, whose return-periods.zarr and fdc.zarr are read')
    parser.add_argument('--hydrography', type=Path, default=Path.home() / 'data' / 'rfsv3' / 'hydrography',
                        help='the published hydrography, for each alert\'s river id and outlet')
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 8, help='processes reading blocks at once')
    parser.add_argument('--shards-per-block', type=int, default=20, help='250 river shards each process reads at once')
    args = parser.parse_args()

    began = time.time()
    initialization = args.initialization
    store = rfs_spec.forecast_store_path(args.out_root, initialization)
    for needed in (store, args.retrospective / 'return-periods.zarr', args.retrospective / 'fdc.zarr'):
        if not needed.exists():
            raise SystemExit(f'{needed} does not exist')
    n_rivers = zarr.open_array(str(store / 'riverId'), mode='r').shape[0]
    for name in ('return-periods.zarr', 'fdc.zarr'):
        river_ids = zarr.open_array(str(args.retrospective / name / 'riverId'), mode='r')[:]
        if not np.array_equal(river_ids, zarr.open_array(str(store / 'riverId'), mode='r')[:]):
            raise SystemExit(f'{name} is not on the riverId axis of {store}')

    block = rfs_spec.RIVERS_PER_SHARD * args.shards_per_block
    jobs = [(store, args.retrospective, r0, min(r0 + block, n_rivers)) for r0 in range(0, n_rivers, block)]
    styles = {name: np.empty((n_rivers, rfs_spec.FORECAST_STEPS if name == 'timeseries' else 1), dtype=np.uint8)
              for name in rfs_spec.STYLESETS}
    alerts = []
    print(f'{initialization:%Y-%m-%d %H}z: summarizing {n_rivers:,} rivers in {len(jobs)} blocks, {args.jobs} at a '
          f'time', flush=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        for n, (r0, block_styles, block_alerts) in enumerate(pool.map(summarize_block, jobs, chunksize=4), start=1):
            for name, values in block_styles.items():
                styles[name][r0:r0 + values.shape[0]] = values
            alerts.append(pd.DataFrame(block_alerts))
            if n % 100 == 0 or n == len(jobs):
                print(f'[{n}/{len(jobs)}] {(time.time() - began) / 60:.1f} min', flush=True)

    day = store.parent
    times = rfs_spec.forecast_times(initialization)
    descriptions = {
        'timeseries': 'each step: return period class of the ensemble median in the high 5 bits, thickness class - 1 '
                      'in the low 3',
        'max-flow': 'the return period class and thickness class, as timeseries, of the ensemble median\'s peak',
        'time-to-peak': 'the step of the ensemble median\'s peak, 0 the first; no_data where it has no flow',
        'below-q95': '1 where the ensemble median falls below Q95 at any step, 0 where it does not; no_data where '
                     'there is no Q95',
    }
    for name, cube in styles.items():
        rfs_spec.write_styles(day / 'maps' / name, name, cube, initialization,
                              times if cube.shape[1] > 1 else times[:1], descriptions[name])
    table = alert_table(pd.concat(alerts, ignore_index=True), initialization, args.hydrography)
    partial = day / 'alerts.partial.csv'
    table.to_csv(partial, index=False)
    os.replace(partial, day / 'alerts.csv')
    print(f'{len(table):,} alerts; summary files in {day} in {(time.time() - began) / 60:.1f} min', flush=True)
