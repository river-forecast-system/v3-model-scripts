"""
Convert the monthly ERA5 runoff netCDFs into zarrs spanning several months each, chunked for reading the grid cells of
one region: small latitude/longitude chunks and the whole time axis in one chunk.

    ~/data/era5/year=2024/era5_202401.nc ... era5_202403.nc
        -> ~/data/era5_zarr_16x16_3month/year=2024/era5_202401_202403.zarr

Step 1 of the retrospective build: the runoff grids 2_rfs_v3_retro_router.py routes.
"""

import argparse
import re
import shutil
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import xarray as xr

VAR_T, VAR_Y, VAR_X = 'valid_time', 'latitude', 'longitude'
FILE_PATTERN = re.compile(r'era5_(\d{4})(\d{2})\.nc$')


def group_files(era5_root: Path, months: int, years: set[int] | None) -> dict[Path, list[Path]]:
    """Group the monthly files by year and block of months, keyed by the output zarr's path relative to the root."""
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for f in era5_root.glob('year=*/era5_*.nc'):
        match = FILE_PATTERN.search(f.name)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        if years and year not in years:
            continue
        groups[(year, (month - 1) // months)].append(f)

    outputs = {}
    for (year, block), files in sorted(groups.items()):
        files = sorted(files)
        first, last = block * months + 1, block * months + months
        if len(files) != months:
            print(f'Skipping incomplete {year} months {first:02d}-{last:02d}: found {[f.name for f in files]}')
            continue
        outputs[Path(f'year={year}') / f'era5_{year}{first:02d}_{year}{last:02d}.zarr'] = files
    return outputs


def convert(files: list[Path], output: Path, xy_chunk: int) -> tuple[Path, float]:
    """Load the months into memory and write them as one zarr. Written to a temporary path then renamed into place."""
    warnings.filterwarnings('ignore', message='Consolidated metadata', category=UserWarning)
    start = time.perf_counter()
    tmp = output.with_name(output.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_mfdataset(
        files, combine='nested', concat_dim=VAR_T, data_vars='minimal', coords='minimal',
        compat='override', join='exact'
    ) as ds:
        ds = ds.load()
    # drop the netCDF compression and chunk encodings so they do not conflict with the zarr encoding
    for var in ds.variables.values():
        var.encoding = {}
    chunks = {VAR_T: ds.sizes[VAR_T], VAR_Y: xy_chunk, VAR_X: xy_chunk}
    encoding = {
        name: {'chunks': tuple(min(chunks[d], ds.sizes[d]) for d in da.dims)}
        for name, da in ds.data_vars.items()
        if set(da.dims) == set(chunks)
    }
    ds.to_zarr(tmp, mode='w', encoding=encoding)
    shutil.rmtree(output, ignore_errors=True)  # only exists when overwriting
    tmp.rename(output)
    return output, time.perf_counter() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--era5-root', type=Path, default=Path('/Users/rchales/data/era5'))
    parser.add_argument('--zarr-root', type=Path, default=None, help='default: era5_zarr_{xy}x{xy}_{months}month')
    parser.add_argument('--xy-chunk', type=int, default=16, help='latitude and longitude chunk size')
    parser.add_argument('--months', type=int, default=3, help='months per zarr, must divide 12')
    parser.add_argument('--years', type=int, nargs='*', help='only convert these years')
    parser.add_argument('--workers', type=int, default=8, help='groups converted concurrently, ~10GB RAM each')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if 12 % args.months:
        raise ValueError(f'--months must divide 12, got {args.months}')
    zarr_root = args.zarr_root or args.era5_root.with_name(
        f'era5_zarr_{args.xy_chunk}x{args.xy_chunk}_{args.months}month'
    )

    todo = {
        zarr_root / rel: files
        for rel, files in group_files(args.era5_root, args.months, set(args.years or [])).items()
        if args.overwrite or not (zarr_root / rel).exists()
    }
    print(f'Writing {len(todo)} zarrs to {zarr_root}')

    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert, files, output, args.xy_chunk) for output, files in todo.items()]
        for n, future in enumerate(as_completed(futures), start=1):
            output, seconds = future.result()
            print(f'[{n}/{len(futures)}] {output.relative_to(zarr_root)} in {seconds:.0f}s')
