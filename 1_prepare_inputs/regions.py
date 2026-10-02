"""
What every script in 1_prepare_inputs shares: the regions of the hydrography, writing a file so an interrupted run
never leaves one that looks finished, preparing the regions in parallel, biggest first, and concatenating the regions'
files into the global files that route the whole world at once.

A global file is its regions' files concatenated in riverIndex order, into routing/global/, which mirrors the
hydrography's global/. Nothing is recomputed. Each region is a closed network and one contiguous run of riverIndex, so
ordering the regions by their first river keeps the global routing table topological, and its rows are the rows of
hydrography/global/metadata.parquet by riverIndex, which is checked. A catchment's weights do not depend on the other
catchments of its region, so the regions' weight tables concatenated in the same order are the weights of the world.
"""

import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

HYDROGRAPHY = Path('/Users/rchales/data/rfsv3/hydrography')
ROUTING = Path('/Users/rchales/data/rfsv3/routing')


def list_regions(hydrography: Path, only: list[str] | None = None) -> list[str]:
    """The region ids of the hydrography's region=<id> partition, or of ``only``, which must all be in it."""
    regions = sorted(d.name.split('=')[1] for d in hydrography.glob('region=*'))
    regions = [r for r in regions if r != 'global' and (not only or r in only)]
    if only and (unknown := set(only) - set(regions)):
        raise SystemExit(f'no region={sorted(unknown)[0]} in {hydrography}')
    return regions


def catchments_file(hydrography: Path, region: str) -> Path:
    return hydrography / f'region={region}' / f'catchments_{region}.geo.parquet'


def write_in_place(path: Path, write: Callable[[Path], object]) -> None:
    """Write to a temporary name beside ``path`` and rename it into place."""
    partial = path.with_suffix('.partial' + path.suffix)
    write(partial)
    os.replace(partial, path)
    return


def prepare_all(prepare: Callable[[dict], str], jobs: list[dict], processes: int | None, what: str) -> None:
    """
    Run ``prepare`` on each job in a pool of ``processes``, default every core, printing what each returns. Each job
    names its region with 'region' and the hydrography with 'hydrography'; the biggest regions start first, so they are
    not left running alone at the end.
    """
    jobs = sorted(jobs, key=lambda j: -catchments_file(j['hydrography'], j['region']).stat().st_size)
    processes = min(processes or os.cpu_count() or 8, max(len(jobs), 1))
    print(f'{len(jobs)} regions to {what}, {processes} at a time', flush=True)
    began = time.time()
    with ProcessPoolExecutor(processes) as pool:
        futures = [pool.submit(prepare, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), start=1):
            print(f'[{n}/{len(futures)}] {future.result()}', flush=True)
    print(f'{len(jobs)} regions done in {(time.time() - began) / 60:.1f} min', flush=True)
    return


def region_order(hydrography: Path, routing: Path, regions: list[str]) -> list[str]:
    """The regions in riverIndex order, checked to cover every river of the global metadata once, each region one
    contiguous run of it in the order of its routing.parquet."""
    table = pq.read_table(hydrography / 'global' / 'metadata.parquet', columns=['riverId', 'riverIndex'])
    lookup = pd.Series(table.column('riverIndex').to_numpy(), index=table.column('riverId').to_numpy())
    starts, size = {}, 0
    for region in regions:
        ids = pq.read_table(routing / f'region={region}' / 'routing.parquet', columns=['river_id'])
        rows = lookup.reindex(ids.column('river_id').to_numpy())
        if rows.isna().any():
            raise SystemExit(f'region={region} has rivers that are not in global/metadata.parquet')
        rows = rows.to_numpy()
        if not np.array_equal(rows, np.arange(rows[0], rows[0] + rows.size)):
            raise SystemExit(f'region={region} is not one contiguous run of riverIndex in routing.parquet order')
        starts[region], size = int(rows[0]), size + rows.size
    ordered = sorted(regions, key=starts.get)
    if size != len(lookup) or starts[ordered[0]] != 0:
        raise SystemExit(f'the regions hold {size:,} rivers, global/metadata.parquet {len(lookup):,}')
    return ordered


def concatenate_routing(files: list[Path], path: Path) -> int:
    """Write the routing tables in ``files`` as one. Returns the rows written."""
    routing = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    write_in_place(path, lambda p: routing.to_parquet(p, index=False))
    return len(routing)


def concatenate_netcdf(files: list[Path], path: Path) -> int:
    """Write the weight tables in ``files`` as one along their index, with the attributes of the first but its
    catchments path. Returns the rows written."""
    datasets = [xr.open_dataset(file) for file in files]
    try:
        names = list(datasets[0].data_vars)
        attrs = {k: v for k, v in datasets[0].attrs.items() if k != 'catchments_path'}
        attrs['regions'] = 'the region=<id> weight tables concatenated in riverIndex order'
        columns = {name: ('index', np.concatenate([ds[name].to_numpy() for ds in datasets])) for name in names}
        merged = xr.Dataset(columns, attrs=attrs)
    finally:
        for ds in datasets:
            ds.close()
    write_in_place(path, merged.to_netcdf)
    return merged.sizes['index']


def concatenate_parquet(files: list[Path], path: Path) -> int:
    """Write the parquet tables in ``files`` as one, snappy compressed, without dictionary encoding, in one row group,
    the layout jsrr reads fastest. Returns the rows written."""
    table = pa.concat_tables([pq.read_table(file) for file in files])
    write_in_place(path, lambda p: pq.write_table(table, p, compression='snappy', use_dictionary=False,
                                                 row_group_size=len(table)))
    return len(table)


def write_global(hydrography: Path, routing: Path, name: str) -> None:
    """
    Concatenate every region's file ``name``, where {region} stands for the region's id, into routing/global/ as
    ``name`` with {region} as global, by concatenate_routing, concatenate_netcdf or concatenate_parquet as the name
    says. An existing global file is kept, and none is written until every region has its file.
    """
    global_name = name.format(region='global')
    path = routing / 'global' / global_name
    if path.exists():
        print(f'global/{global_name}: exists, kept', flush=True)
        return
    regions = list_regions(hydrography)
    needed = {'routing.parquet', name}
    if absent := [f'region={r}/{n.format(region=r)}' for r in regions for n in sorted(needed)
                  if not (routing / f'region={r}' / n.format(region=r)).exists()]:
        print(f'global/{global_name}: not written, {len(absent)} region files are missing, e.g. {absent[0]}',
              flush=True)
        return
    files = [routing / f'region={r}' / name.format(region=r) for r in region_order(hydrography, routing, regions)]
    concatenate = concatenate_routing if name == 'routing.parquet' else \
        concatenate_netcdf if path.suffix == '.nc' else concatenate_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    rows = concatenate(files, path)
    print(f'global/{global_name}: {rows:,} rows from {len(files)} regions in {time.time() - began:.1f} s', flush=True)
    return
