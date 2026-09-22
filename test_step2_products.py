"""
Checks for the parts of ``2_rfs_v3_retro_router.py`` that are not routing: the writer it hands river-route, the four
derived stores a region writes once its years are done, and the re-fold a resume rebuilds its accumulators with.

Routing itself is not exercised here -- it needs a hydrography and a year of ERA5 -- so the Router is stubbed down
to the handful of attributes ``zarr_writer`` actually reads off one. What that leaves is everything a routed year
passes through between the kernel and the disk, checked for both discharge dtypes. Run with
``python test_step2_products.py`` or under pytest.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

import rfs_aggregate
import rfs_spec
import test_rfs_aggregate as helpers

# the step scripts are numbered, so they are not importable by name
step2 = importlib.import_module('2_rfs_v3_retro_router')

N_RIVERS = 300
YEARS = [2019, 2020]
RIVER_IDS = np.arange(5_000_000, 5_000_000 + N_RIVERS, dtype=np.int64)


def stub_router() -> types.SimpleNamespace:
    """The attributes river-route's zarr_writer and the year writer actually read off a Router."""
    return types.SimpleNamespace(
        configs=types.SimpleNamespace(var_river_id='river_id', var_discharge='Q'),
        network=types.SimpleNamespace(river_ids=RIVER_IDS),
        threads=2,
        channel_state=np.zeros(N_RIVERS, dtype=np.float32),
    )


def route_a_region(region: Path, dtype: str) -> rfs_aggregate.RegionAccumulators:
    """Drive the year writer over two synthetic years the way a Router would, then write the region's products."""
    accumulators = rfs_aggregate.RegionAccumulators(N_RIVERS, YEARS)
    with rfs_aggregate.WriteQueue(1, 1) as write_queue:
        write_year = step2.make_year_writer(accumulators, write_queue, region)
        for year in YEARS:
            dates = helpers.hours_of(year)
            hourly = helpers.synthetic(N_RIVERS, dates, seed=year, scale=200.0)
            if dtype == 'float16':
                hourly = np.ascontiguousarray(hourly.astype(np.float16).view(np.uint16))
            write_year(stub_router(), dates.to_numpy(), hourly, step2.hourly_path(region, year))
    step2.write_region_products(region, accumulators, RIVER_IDS, threads=2)
    return accumulators


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_step2_products(tmp_path, dtype: str) -> None:
    region = Path(tmp_path) / f'region=test_{dtype}'
    shutil.rmtree(region, ignore_errors=True)
    region.mkdir(parents=True)
    accumulators = route_a_region(region, dtype)

    # a run that finished leaves every store consolidated and a state file beside every year
    assert all(step2.is_complete(step2.hourly_path(region, y)) for y in YEARS), 'an hourly store is not consolidated'
    assert all(step2.is_complete(region / f'{n}.zarr') for n in step2.DERIVED_STORES), 'a product is not consolidated'
    assert all(step2.state_path(region, y).exists() for y in YEARS), 'a channel state was not checkpointed'

    # what a resume depends on: re-folding the years from disk rebuilds the accumulators bit for bit, because the
    # rounding the fold does is idempotent and the store holds exactly the values it rounded
    again = rfs_aggregate.RegionAccumulators(N_RIVERS, YEARS)
    for year in YEARS:
        step2.refold_year(again, step2.hourly_path(region, year))
    for name in ('daily', 'monthly', 'yearly', 'max_hourly', 'max_daily'):
        before, after = getattr(accumulators, name), getattr(again, name)
        assert np.array_equal(before.view(np.uint32), after.view(np.uint32)), f'{dtype}: re-folding changed {name}'

    # the hourly store keeps the dtype the region was routed at, and the spec's promise holds against it
    hourly = zarr.open_group(str(step2.hourly_path(region, 2020)), mode='r')['Q']
    assert hourly.dtype == np.dtype(dtype), f'hourly stored as {hourly.dtype}, wanted {dtype}'
    stored = np.asarray(hourly[:]).astype(np.float32)
    assert np.isin(accumulators.max_hourly[:, 1], stored).all(), 'an annual maximum is not a value hourly.zarr holds'
    assert (accumulators.max_hourly >= accumulators.max_daily).all(), 'an hourly maximum is below the daily one'

    # each product has the spec's shape, dtype, chunking and metadata, and no gap
    products = (
        ('daily', 365 + 366, ('Q',)),
        ('monthly', 24, ('Q',)),
        ('yearly', 2, ('Q',)),
        ('maximums', 2, ('hourly', 'daily')),
    )
    for name, n_time, variables in products:
        group = zarr.open_group(str(region / f'{name}.zarr'), mode='r')
        for variable in variables:
            array = group[variable]
            assert array.shape == (N_RIVERS, n_time), f'{name}/{variable} is {array.shape}, wanted {(N_RIVERS, n_time)}'
            assert array.dtype == np.float32, f'{name}/{variable} is {array.dtype}, wanted float32'
            assert array.chunks[0] == rfs_spec.RIVERS_PER_CHUNK and array.shards[0] == rfs_spec.RIVERS_PER_SHARD
            assert array.attrs['keepbits'] == rfs_spec.KEEP_BITS
            assert not np.isnan(array[:]).any(), f'{name}/{variable} has a gap after a complete run'
        # Q_timesteps is chunked across every river at one timestep, which only means anything once every region is
        # in one store, so a per region product must not carry it
        assert 'Q_timesteps' not in group, f'{name} carries Q_timesteps, which belongs on the assembled store'
        assert np.array_equal(group['riverId'][:], RIVER_IDS.astype(np.int32)), f'{name} river ids differ'

    assert np.array_equal(zarr.open_group(str(region / 'daily.zarr'), mode='r')['Q'][:], accumulators.daily)
    assert np.array_equal(zarr.open_group(str(region / 'maximums.zarr'), mode='r')['hourly'][:],
                          accumulators.max_hourly)

    daily_times = rfs_spec.store_dates(zarr.open_group(str(region / 'daily.zarr'), mode='r')['time'])
    assert daily_times[0] == pd.Timestamp('2019-01-01') and daily_times[-1] == pd.Timestamp('2020-12-31')


if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    for discharge_dtype in ('float32', 'float16'):
        test_step2_products(root, discharge_dtype)
        print(f'ok  {discharge_dtype}')
    print('ok  step 2 products')
