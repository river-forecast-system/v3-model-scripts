"""
Write the routing table of each region of the v3 hydrography into the routing directory, which mirrors the
hydrography's region=<id> partition and global/ and is published beside it as routing/:

    routing/region=<id>/routing.parquet    river_id, next_river_id, k, x: river-route's Muskingum network
    routing/global/routing.parquet         every region's, in riverIndex order, to route the whole world at once

Routing the world at once, in one process on many threads, is the way to route when reading the forcing costs more
than routing it: a GRIB forecast on networked storage is read once for the world, rather than once per region. The
global table is the regions' concatenated, as regions.py describes.

The hydrography is only read, so the pipeline that builds it holds no routing configuration, except the Muskingum
musk_k and musk_x it computes so they are attributes of the GIS files.

The table lists the region's rivers in riverIndex order. That order is topological, every river before the river it
drains into, which river-route requires of the network. Every grid weight table, from 2_gridweights_era5.py and
3_gridweights_ifs.py, follows it exactly, because nothing in the router matches them by id: lateral inflow column i is
routed into river i. spec.global_layout also finds a region's rows of the published stores by that order.

The routing table is the region's metadata renamed: riverId, nextRiverId, musk_k and musk_x become river_id,
next_river_id, k and x. Nothing is recomputed. Changing k or x is an option of the routing step.

Regions are independent and run in parallel, biggest first. A region whose routing.parquet exists is skipped unless
--overwrite is passed. The global table is written once every region has one, and rewritten when any region was.
Each file is written to a temporary name and renamed into place, so an interrupted run never leaves a file that looks
finished.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from regions import HYDROGRAPHY, ROUTING, list_regions, prepare_all, write_global, write_in_place

# metadata column -> routing.parquet column
ROUTING_COLUMNS = {'riverId': 'river_id', 'nextRiverId': 'next_river_id', 'musk_k': 'k', 'musk_x': 'x'}


def routing_table(metadata: pd.DataFrame, region: str) -> pd.DataFrame:
    """The region's metadata as river-route's routing parameters, checked to be a closed, topologically sorted
    network."""
    metadata = metadata.sort_values('riverIndex', kind='stable')
    index = metadata['riverIndex'].to_numpy()
    if not np.array_equal(index, np.arange(index[0], index[0] + index.size)):
        raise ValueError(f'{region}: riverIndex is not one contiguous run')
    routing = pd.DataFrame({new: metadata[old].to_numpy() for old, new in ROUTING_COLUMNS.items()}).astype(
        {'river_id': np.int32, 'next_river_id': np.int32, 'k': np.float64, 'x': np.float64})

    ids = pd.Index(routing['river_id'])
    if ids.has_duplicates:
        raise ValueError(f'{region}: river ids repeat')
    downstream = ids.get_indexer(routing['next_river_id'])
    flows_on = routing['next_river_id'].to_numpy() != -1
    if np.any(downstream[flows_on] < 0):
        raise ValueError(f'{region}: {int(np.sum(downstream[flows_on] < 0)):,} rivers drain out of the region')
    if np.any(downstream[flows_on] <= np.flatnonzero(flows_on)):
        raise ValueError(f'{region}: riverIndex is not topological, some river comes after the river it drains into')
    k, x = routing['k'].to_numpy(), routing['x'].to_numpy()
    if not np.all(np.isfinite(k) & (k > 0)) or not np.all((x >= 0) & (x <= 0.5)):
        raise ValueError(f'{region}: k must be positive and x within [0, 0.5]')
    return routing


def prepare_region(job: dict) -> str:
    """Write one region's routing.parquet."""
    region, began = job['region'], time.time()
    metadata_file = job['hydrography'] / f'region={region}' / f'metadata_{region}.parquet'
    routing = routing_table(pd.read_parquet(metadata_file, columns=['riverIndex', *ROUTING_COLUMNS]), region)
    routing_dir = job['routing'] / f'region={region}'
    routing_dir.mkdir(parents=True, exist_ok=True)
    write_in_place(routing_dir / 'routing.parquet', lambda path: routing.to_parquet(path, index=False))
    return f'{region}: {len(routing):>9,} rivers  {time.time() - began:6.1f} s'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hydrography', type=Path, default=HYDROGRAPHY, help='the published hydrography, only read')
    parser.add_argument('--routing', type=Path, default=ROUTING,
                        help='where the routing files are written, a region=<id> folder per region')
    parser.add_argument('--regions', nargs='+', help='prepare only these regions')
    parser.add_argument('--jobs', type=int, default=None, help='regions prepared at once, default every core')
    parser.add_argument('--overwrite', action='store_true',
                        help='rewrite regions whose routing.parquet exists, and the global one')
    args = parser.parse_args()

    regions = list_regions(args.hydrography, args.regions)
    todo = [r for r in regions if args.overwrite or not (args.routing / f'region={r}' / 'routing.parquet').exists()]
    print(f'{len(regions) - len(todo)} of {len(regions)} regions already have routing.parquet', flush=True)
    prepare_all(prepare_region, [{'region': r, 'hydrography': args.hydrography, 'routing': args.routing} for r in todo],
                args.jobs, 'write routing.parquet for')
    write_global(args.hydrography, args.routing, 'routing.parquet', args.overwrite or bool(todo))
