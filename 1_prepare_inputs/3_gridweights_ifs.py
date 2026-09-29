"""
Write the weights of each region of the v3 hydrography on the reduced gaussian grid of ECMWF IFS forecast GRIB files
into the routing directory, beside the routing.parquet 1_routing_files.py wrote there, which must exist first:

    routing/region=<id>/gridweights_<grid>_<id>.nc    the cells each catchment overlaps, and the share of it in each

where <grid> is the grid's ECMWF name, O1280 for the octahedral grid of the 9 km IFS, from the GRIB file's metadata.
This is the grid_weights_file of river-route's ecmwf_grib forcing, which reads the forecast GRIB files as they are.
The weights of the whole world, every region's concatenated in riverIndex order as regions.py describes, go beside
the global routing.parquet of 1_routing_files.py:

    routing/global/gridweights_<grid>_global.nc

The weights are river_route.runoff.reduced_grid_weights's: each cell is a box of longitude and latitude halfway to the
cells beside it, between latitude edges that give its row the area of its gaussian quadrature weight, and a
catchment's share of a cell is the area of it inside the box, measured in cylindrical equal-area. A cell is located by
cell_index, its position in the values of a GRIB message, in place of the x_index and y_index of a regular grid. The
cells centered on 180 degrees are in two parts, one on either edge of the map, so no catchment is lost there.

The rows follow routing.parquet, riverIndex order, because nothing in the router matches them by id: lateral inflow
column i is routed into river i. Every river must have a weight and every weight a river, which is checked before the
file is renamed into place.

reduced_grid_weights reads a region's catchments whole, not in batches as 2_gridweights_era5.py does: region
1020000010 peaks at 11.7 GB and the largest, 3020000010, at 31 GB, so by default only 4 regions run at once. Regions
are prepared biggest first. A region whose weights exist is skipped unless --overwrite is passed. The global file
is written once every region has its own, and rewritten when any region was. Each file is written to a temporary
name and renamed into place, so an interrupted run never leaves a file that looks finished.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river_route.runoff import ReducedGaussianGrid, reduced_grid_weights

from regions import HYDROGRAPHY, ROUTING, catchments_file, list_regions, prepare_all, write_global


def grid_name(grid: ReducedGaussianGrid) -> str:
    """ECMWF's name of the grid: O<N> when it is octahedral, 20 cells on the first row and 4 more on each row toward
    the equator, else N<N>."""
    octahedral = np.array_equal(grid.cells_per_row[:grid.N], 20 + 4 * np.arange(grid.N))
    return f'{"O" if octahedral else "N"}{grid.N}'


def prepare_region(job: dict) -> str:
    """Write one region's weight table."""
    region, began = job['region'], time.time()
    routing_dir = job['routing'] / f'region={region}'
    routing_file = routing_dir / 'routing.parquet'
    path = routing_dir / f'gridweights_{job["name"]}_{region}.nc'
    partial = path.with_suffix('.partial' + path.suffix)
    try:
        weights = reduced_grid_weights(job['grib'], catchments_file(job['hydrography'], region),
                                       var_river_id='river_id', var_catchment_id='riverId',
                                       save_weights_path=partial, routing_params_path=routing_file)
        rivers = pd.read_parquet(routing_file, columns=['river_id'])['river_id'].to_numpy()
        weighted = weights['river_id'].to_numpy()
        # a river's rows are together, so its first row starts wherever the river id changes
        in_order = weighted[np.flatnonzero(np.diff(weighted, prepend=weighted[0] - 1))]
        if not np.array_equal(in_order, rivers):
            missing = np.setdiff1d(rivers, weighted).size
            extra = np.setdiff1d(weighted, rivers).size
            raise ValueError(f'{region}: the weights do not follow routing.parquet: {missing:,} rivers have no weight, '
                             f'{extra:,} weighted rivers are not in it')
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)

    cells = weights['cell_index'].nunique()
    return (f'{region}: {len(rivers):>9,} rivers  {len(weights):>9,} weights  {cells:>6,} cells  '
            f'{time.time() - began:6.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grib', type=Path, required=True,
                        help='any IFS GRIB file on the forecast grid; only the grid of its first message is read')
    parser.add_argument('--hydrography', type=Path, default=HYDROGRAPHY, help='the published hydrography, only read')
    parser.add_argument('--routing', type=Path, default=ROUTING,
                        help='where 1_routing_files.py wrote routing.parquet, and the weights are written')
    parser.add_argument('--regions', nargs='+', help='prepare only these regions')
    parser.add_argument('--jobs', type=int, default=4, help='regions prepared at once, each up to 31 GB, default 4')
    parser.add_argument('--overwrite', action='store_true',
                        help='rewrite regions whose weights already exist, and the global ones')
    args = parser.parse_args()

    grid = ReducedGaussianGrid.from_grib(args.grib)
    name = grid_name(grid)
    regions = list_regions(args.hydrography, args.regions)
    if missing := [r for r in regions if not (args.routing / f'region={r}' / 'routing.parquet').exists()]:
        raise SystemExit(f'{len(missing)} regions have no routing.parquet, run 1_routing_files.py first: {missing[0]}')
    todo = [r for r in regions
            if args.overwrite or not (args.routing / f'region={r}' / f'gridweights_{name}_{r}.nc').exists()]
    print(f'{len(regions) - len(todo)} of {len(regions)} regions already have {name} weights, on the '
          f'{grid.n_cells:,} cells ({grid.spacing_km:.1f} km) of {args.grib}', flush=True)
    prepare_all(prepare_region, [{'region': r, 'hydrography': args.hydrography, 'routing': args.routing,
                                  'grib': args.grib, 'name': name} for r in todo], args.jobs, f'weight on {name}')
    write_global(args.hydrography, args.routing, f'gridweights_{name}_{{region}}.nc', args.overwrite or bool(todo))
