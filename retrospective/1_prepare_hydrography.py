"""
Write the two files each region of the v3 hydrography needs before it can be routed on ERA5 into the routing
directory, which mirrors the hydrography's region=<id> partition and is published beside it as routing/:

    routing/region=<id>/routing.parquet             river_id, next_river_id, k, x: river-route's Muskingum network
    routing/region=<id>/gridweights_ERA5_<id>.nc    the ERA5 cells each catchment overlaps, and the share of it in each

and an extra copy of the weights for jsrr, the browser router, which does not replace the netCDF, the weights' standard
format and the one the routing step reads:

    routing/region=<id>/gridweights_ERA5_<id>.parquet    the netCDF's weights, in the layout jsrr reads fastest

The hydrography is only read, so the pipeline that builds it holds no routing configuration, except the Muskingum
musk_k and musk_x it computes so they are attributes of the GIS files.

All list the region's rivers in riverIndex order. That order is topological, every river before the river it drains
into, which river-route requires of the network. The weight table's rivers must follow it exactly, because nothing in
the router matches them by id: lateral inflow column i is routed into river i. spec.global_layout also finds a
region's rows of the published stores by that order.

The routing table is the region's metadata renamed: riverId, nextRiverId, musk_k and musk_x become river_id,
next_river_id, k and x. Nothing is recomputed. Changing k or x is an option of the routing step.

The weights are what river_route.runoff.grid_weights computes, built without its Voronoi diagram, which takes 34 s and
6 GB to build for every region. On a rectilinear latitude-longitude grid the Voronoi cell of a grid point is the box
between the midpoints to its neighbors, so each cell is built as that box, and only the cells under the catchments
are built. The boxes stop at the antimeridian. No v3 catchment comes within 1.5 degrees of it, and one that reaches
past the last box, into the half of the cell centered on 180 east of it, stops the run instead of losing that area.

The catchments are published in web mercator, where meridians and parallels are still straight lines, so the boxes
are built there and the catchments are cut where they lie. geopandas' spatial index pairs each catchment with the
boxes its bounds overlap. A catchment paired with one box lies inside it and is not cut, as 56% of region
1020000010's do. The rest are cut by shapely.clip_by_rect, a box at a time, every catchment touching that box at
once. geopandas.overlay would find the same areas, to 1e-10, but it cuts with a general intersection, 21 s for that
region where clip_by_rect takes 1.2 s, and first checks the validity of every catchment, 8 s more. The pieces are
measured in cylindrical equal-area on WGS84. A river's proportions are its pieces' areas over their sum, and that sum
is the catchment area the router turns runoff depths into volumes with.

The parquet copy is the netCDF's rows without x and y, which jsrr does not read: river_id, x_index and y_index as
int32, area_sqm and proportion as float32, snappy compressed, without dictionary encoding, in one row group, since jsrr
fetches and reads the whole file anyway. Measured on region 7020014250's 95,030 weights with jsrr's parquet reader,
that is 1.14 MB read in 12.0 ms, against 1.31 MB and 14.5 ms keeping x and y, 1.90 MB and 10.8 ms uncompressed, whose
extra bytes cost more over the network than they save in parsing, and 1.26 MB and 12.7 ms with river ids dictionary
encoded. jsrr refuses a region unless every river in routing.parquet has a weight, which weight_table checks, and every
weight names one of its rivers, which holds because the weights take their river ids from routing.parquet by position.

Regions are independent and run in parallel, biggest first. A region whose three files exist is skipped unless
--overwrite is passed. Each file is written to a temporary name and renamed into place, so an interrupted run never
leaves a file that looks finished.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import shapely
import xarray as xr

MERCATOR_EPSG = 3857
MERCATOR_MAX_LAT = 85.051_128_779_806_59  # where its square extent ends
EQUAL_AREA = '+proj=cea +ellps=WGS84'
# metadata column -> routing.parquet column
ROUTING = {'riverId': 'river_id', 'nextRiverId': 'next_river_id', 'musk_k': 'k', 'musk_x': 'x'}
WEIGHTS = ['river_id', 'x_index', 'y_index', 'x', 'y', 'area_sqm', 'proportion']
# the weight columns jsrr reads, and their types
JSRR_WEIGHTS = {'river_id': np.int32, 'x_index': np.int32, 'y_index': np.int32, 'area_sqm': np.float32,
                'proportion': np.float32}
CATCHMENT_BATCH = 20_000  # catchments decoded and cut at once, which bounds a region's memory


def cell_edges(centers: np.ndarray) -> np.ndarray:
    """The n + 1 edges of n ascending cell centers: the midpoints between them, and half a spacing past each end."""
    middle = (centers[1:] + centers[:-1]) / 2
    return np.concatenate(([2 * centers[0] - middle[0]], middle, [2 * centers[-1] - middle[-1]]))


def read_grid(path: Path) -> dict[str, np.ndarray]:
    """
    The edges of the grid's cells in web mercator, ascending, with ``x_index`` and ``y_index`` to map a position in
    them back to the grid's own index. Longitudes are shifted onto -180..180 first, so the grid may run 0..360.
    """
    with xr.open_dataset(path) as ds:
        lon = ds['longitude'].to_numpy().astype(np.float64)
        lat = ds['latitude'].to_numpy().astype(np.float64)
    if lon.ndim != 1 or lat.ndim != 1:
        raise SystemExit(f'{path} is not a rectilinear grid: its longitude and latitude must be 1D')

    shifted = (lon + 180) % 360 - 180
    x_order, y_order = np.argsort(shifted), np.argsort(lat)
    if np.any(np.diff(shifted[x_order]) <= 0) or np.any(np.diff(lat[y_order]) <= 0):
        raise SystemExit(f'{path} repeats a longitude or latitude')
    # PROJ wraps a longitude past 180 around to the other side, so the edges stop at 180. No catchment reaches past
    # MERCATOR_MAX_LAT, so stopping there loses nothing.
    lon_edges = np.clip(cell_edges(shifted[x_order]), -180, 180)
    lat_edges = np.clip(cell_edges(lat[y_order]), -MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
    to_mercator = pyproj.Transformer.from_crs(4326, MERCATOR_EPSG, always_xy=True)
    mx, _ = to_mercator.transform(lon_edges, np.zeros_like(lon_edges))
    _, my = to_mercator.transform(np.zeros_like(lat_edges), lat_edges)
    return {'lon': lon, 'lat': lat, 'x_index': x_order, 'y_index': y_order, 'mx': np.asarray(mx), 'my': np.asarray(my)}


def routing_table(metadata: pd.DataFrame, region: str) -> pd.DataFrame:
    """The region's metadata as river-route's routing parameters, checked to be a closed, topologically sorted
    network."""
    metadata = metadata.sort_values('riverIndex', kind='stable')
    index = metadata['riverIndex'].to_numpy()
    if not np.array_equal(index, np.arange(index[0], index[0] + index.size)):
        raise ValueError(f'{region}: riverIndex is not one contiguous run')
    routing = pd.DataFrame({new: metadata[old].to_numpy() for old, new in ROUTING.items()}).astype(
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


def cells_under(bounds: np.ndarray, grid: dict[str, np.ndarray]) -> gpd.GeoDataFrame:
    """The boxes of the grid cells that overlap ``bounds``, (minx, miny, maxx, maxy) in web mercator."""
    mx, my = grid['mx'], grid['my']
    cols = np.flatnonzero((mx[1:] > bounds[0]) & (mx[:-1] < bounds[2]))
    rows = np.flatnonzero((my[1:] > bounds[1]) & (my[:-1] < bounds[3]))
    col, row = (a.ravel() for a in np.meshgrid(cols, rows))
    return gpd.GeoDataFrame(
        {'x_index': grid['x_index'][col], 'y_index': grid['y_index'][row]},
        geometry=shapely.box(mx[col], my[row], mx[col + 1], my[row + 1]), crs=MERCATOR_EPSG,
    )


def cut_into_cells(catchments: gpd.GeoSeries, cells: gpd.GeoDataFrame) -> pd.DataFrame:
    """Each catchment's area in each cell it overlaps, as (river, x_index, y_index, area_sqm) rows, where river is
    the catchment's position in ``catchments``."""
    river, cell = cells.sindex.query(catchments)  # every box each catchment's bounds overlap
    pieces = catchments.to_numpy()[river]

    # a catchment paired with one box lies inside it. clip_by_rect takes one box per call, so the pieces of the
    # rest are clipped a box at a time
    shared = np.flatnonzero(np.bincount(river, minlength=len(catchments))[river] > 1)
    box = cells.bounds.to_numpy()
    for c, group in pd.Series(shared).groupby(cell[shared]):
        pieces[group] = shapely.clip_by_rect(pieces[group], *box[c])

    area = gpd.GeoSeries(pieces, crs=catchments.crs).to_crs(EQUAL_AREA).area.to_numpy()
    keep = area > 0  # a box the bounds overlap but the catchment does not
    return pd.DataFrame({
        'river': river[keep], 'x_index': cells['x_index'].to_numpy()[cell[keep]],
        'y_index': cells['y_index'].to_numpy()[cell[keep]], 'area_sqm': area[keep],
    })


def weight_table(table: pd.DataFrame, n_rivers: int, grid: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    The pieces of every river, from cut_into_cells, as the weight table river-route reads, with river still a
    position in routing order. Rows follow that order, and a river's cells run largest first.
    """
    if (missing := n_rivers - table['river'].nunique()) > 0:
        raise ValueError(f'{missing:,} catchments have no area in any cell')
    table['proportion'] = table['area_sqm'] / table.groupby('river')['area_sqm'].transform('sum')
    table = table.sort_values(['river', 'area_sqm'], ascending=[True, False], kind='stable', ignore_index=True)
    table['x'] = grid['lon'][table['x_index']]
    table['y'] = grid['lat'][table['y_index']]
    return table.astype({'x_index': np.int32, 'y_index': np.int32, 'x': np.float32, 'y': np.float32,
                         'area_sqm': np.float32, 'proportion': np.float32})


def write_in_place(path: Path, write) -> None:
    """Write to a temporary name beside ``path`` and rename it into place."""
    partial = path.with_suffix('.partial' + path.suffix)
    write(partial)
    os.replace(partial, path)
    return


def write_jsrr_weights(weights: pd.DataFrame, path: Path) -> None:
    """Write the weights as parquet in the layout jsrr reads fastest: its columns only, snappy compressed, without
    dictionary encoding, in one row group, rows in the order given."""
    table = pa.table({name: weights[name].to_numpy().astype(dtype, copy=False) for name, dtype in JSRR_WEIGHTS.items()})
    pq.write_table(table, path, compression='snappy', use_dictionary=False, row_group_size=len(table))
    return


def prepare_region(job: dict) -> str:
    """Write one region's routing.parquet and weight tables."""
    region, began = job['region'], time.time()
    region_dir = job['hydrography'] / f'region={region}'
    routing_dir = job['routing'] / f'region={region}'
    catchments_file = region_dir / f'catchments_{region}.geo.parquet'

    metadata = pd.read_parquet(region_dir / f'metadata_{region}.parquet', columns=['riverIndex', *ROUTING])
    routing = routing_table(metadata, region)

    # a batch at a time: decoding the whole file at once takes 15 GB for the largest region, a batch at a time 3 GB
    rivers, parts, seen = pd.Index(routing['river_id']), [], np.zeros(len(routing), dtype=np.int64)
    for batch in pq.ParquetFile(catchments_file).iter_batches(CATCHMENT_BATCH, columns=['riverId', 'geometry']):
        catchments = gpd.GeoDataFrame.from_arrow(pa.Table.from_batches([batch]))
        if catchments.crs is None or catchments.crs.to_epsg() != MERCATOR_EPSG:
            raise ValueError(f'{region}: expected catchments in EPSG:{MERCATOR_EPSG}, got {catchments.crs}')
        position = rivers.get_indexer(catchments['riverId'])
        if np.any(position < 0):
            raise ValueError(f'{region}: a catchment has no river in the metadata')
        seen[position] += 1
        bounds, grid = catchments.total_bounds, job['grid']
        if bounds[0] < grid['mx'][0] or bounds[2] > grid['mx'][-1] or bounds[1] < grid['my'][0] or \
                bounds[3] > grid['my'][-1]:
            raise ValueError(f'{region}: a catchment reaches past the grid, maybe across the antimeridian')
        part = cut_into_cells(catchments.geometry, cells_under(bounds, grid))
        part['river'] = position[part['river'].to_numpy()]
        parts.append(part)
    if np.any(seen != 1):
        raise ValueError(f'{region}: the catchments are not one per river of the metadata')

    try:
        weights = weight_table(pd.concat(parts, ignore_index=True), len(routing), job['grid'])
    except ValueError as error:
        raise ValueError(f'{region}: {error}') from None
    weights.insert(0, 'river_id', routing['river_id'].to_numpy()[weights.pop('river').to_numpy()])
    dataset = xr.Dataset(
        {name: ('index', weights[name].to_numpy()) for name in WEIGHTS},
        attrs={
            'description': 'proportions of runoff cells that intersect river catchments',
            'grid_path': str(job['grid_path']),
            'grid_shape': f'{job["grid"]["lat"].size} latitude x {job["grid"]["lon"].size} longitude',
            'catchments_path': str(catchments_file),
            'row_order': 'rivers in the order of routing.parquet, then cells by area_sqm descending',
            'x_y': 'the longitude and latitude of the cell at x_index and y_index, as the grid gives them',
        },
    )
    routing_dir.mkdir(parents=True, exist_ok=True)
    write_in_place(routing_dir / 'routing.parquet', lambda path: routing.to_parquet(path, index=False))
    write_in_place(routing_dir / f'gridweights_ERA5_{region}.nc', dataset.to_netcdf)
    write_in_place(routing_dir / f'gridweights_ERA5_{region}.parquet', lambda path: write_jsrr_weights(weights, path))

    cells = weights[['x_index', 'y_index']].drop_duplicates().shape[0]
    return (f'{region}: {len(routing):>9,} rivers  {len(weights):>9,} weights  {cells:>6,} cells  '
            f'{time.time() - began:6.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hydrography', type=Path, default=Path('/Users/rchales/data/rfsv3/hydrography'),
                        help='the published hydrography, only read')
    parser.add_argument('--routing', type=Path, default=Path('/Users/rchales/data/rfsv3/routing'),
                        help='where the routing files are written, a region=<id> folder per region')
    parser.add_argument('--grid', type=Path, default=Path('/Users/rchales/data/era5/year=1940/era5_194001.nc'),
                        help='any file on the ERA5 grid, netCDF or zarr; only its longitude and latitude are read')
    parser.add_argument('--regions', nargs='+', help='prepare only these regions')
    parser.add_argument('--jobs', type=int, default=None, help='regions prepared at once, default every core')
    parser.add_argument('--overwrite', action='store_true', help='rewrite regions whose files already exist')
    args = parser.parse_args()

    grid = read_grid(args.grid)
    regions = sorted(d.name.split('=')[1] for d in args.hydrography.glob('region=*'))
    regions = [r for r in regions if r != 'global' and (not args.regions or r in args.regions)]
    if args.regions and (unknown := set(args.regions) - set(regions)):
        raise SystemExit(f'no region={sorted(unknown)[0]} in {args.hydrography}')


    def done(region: str) -> bool:
        routing_dir = args.routing / f'region={region}'
        files = ['routing.parquet', f'gridweights_ERA5_{region}.nc', f'gridweights_ERA5_{region}.parquet']
        return all((routing_dir / name).exists() for name in files)


    todo = [r for r in regions if args.overwrite or not done(r)]
    # biggest first, so the largest regions are not left running alone at the end
    todo.sort(key=lambda r: -(args.hydrography / f'region={r}' / f'catchments_{r}.geo.parquet').stat().st_size)
    jobs = min(args.jobs or os.cpu_count() or 8, max(len(todo), 1))
    print(f'{len(todo)} of {len(regions)} regions to prepare, {jobs} at a time, on the '
          f'{grid["lat"].size} x {grid["lon"].size} grid of {args.grid}', flush=True)

    began = time.time()
    with ProcessPoolExecutor(jobs) as pool:
        futures = [pool.submit(prepare_region, {'region': r, 'hydrography': args.hydrography, 'grid': grid,
                                                'routing': args.routing, 'grid_path': args.grid}) for r in todo]
        for n, future in enumerate(as_completed(futures), start=1):
            print(f'[{n}/{len(futures)}] {future.result()}', flush=True)
    print(f'{len(todo)} regions prepared in {(time.time() - began) / 60:.1f} min', flush=True)
