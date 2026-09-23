"""
Convert the monthly ERA5 runoff netCDFs into one zarr per calendar year, chunked for reading one region's grid cells:
small latitude/longitude chunks and the whole year in one time chunk.

    era5/year=2024/era5_202401.nc ... era5_202412.nc  ->  era5_zarr_16x16_12month/year=2024/era5_202401_202412.zarr
"""

import argparse
import re
import shutil
import warnings
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def convert(files: list[Path], output: Path, xy_chunk: int) -> Path:
    """Load the year into memory and write it to a temporary path, then rename it into place."""
    warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)
    tmp = output.with_name(output.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    with xr.open_mfdataset(files, combine='nested', concat_dim='valid_time', data_vars='minimal', coords='minimal',
                           compat='override', join='exact') as ds:
        ds = ds.load()
    for var in ds.variables.values():
        var.encoding = {}  # drop the netCDF encodings so they do not conflict with the zarr one
    chunks = {'valid_time': ds.sizes['valid_time'], 'latitude': xy_chunk, 'longitude': xy_chunk}
    encoding = {name: {'chunks': tuple(min(chunks[d], ds.sizes[d]) for d in da.dims)}
                for name, da in ds.data_vars.items() if set(da.dims) == set(chunks)}
    ds.to_zarr(tmp, mode='w', encoding=encoding)
    shutil.rmtree(output, ignore_errors=True)
    tmp.rename(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--era5-root', type=Path, default=Path('/Users/rchales/data/era5'))
    parser.add_argument('--zarr-root', type=Path, default=Path('/Users/rchales/data/era5_zarr_16x16_12month'))
    parser.add_argument('--xy-chunk', type=int, default=16)
    parser.add_argument('--years', type=int, nargs='*', help='only convert these years')
    parser.add_argument('--workers', type=int, default=4, help='years converted at once, ~40 GB of memory each')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    todo = {}
    for year_dir in sorted(args.era5_root.glob('year=*')):
        year = int(year_dir.name.split('=')[1])
        files = sorted(f for f in year_dir.glob('era5_*.nc') if re.fullmatch(rf'era5_{year}\d\d\.nc', f.name))
        output = args.zarr_root / f'year={year}' / f'era5_{year}01_{year}12.zarr'
        if (args.years and year not in args.years) or (output.exists() and not args.overwrite):
            continue
        if len(files) != 12:
            print(f'skipping {year}: {len(files)} of 12 months')
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        todo[output] = files

    print(f'writing {len(todo)} zarrs to {args.zarr_root}')
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert, files, output, args.xy_chunk) for output, files in todo.items()]
        for n, future in enumerate(as_completed(futures), start=1):
            print(f'[{n}/{len(futures)}] {future.result().name}', flush=True)
