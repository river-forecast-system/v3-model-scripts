"""
Which regions of the v3 hydrography can be routed at ``discharge_dtype='float16'``, and the derivation behind it.

float16 stores no value above 65,504 m3 s-1, so a region can only be narrowed if its discharge never gets there.
Routing narrowed halves the per year hourly stores on disk and the buffer held in memory, and costs nothing
downstream: the reductions are taken and written float32 either way, a float16 value already sits on the spec's
keepbits grid, and 3_build_hourly_daily_maximums.py promotes the values on the way into hourly.zarr.

The bound comes from the RFS v2 retrospective's own annual maximum series, which is 85 years of routed hourly
maxima over 6,838,900 rivers -- the same question already answered on the previous hydrography. Each river's
maximum across every year is taken, then grouped to the HydroBASINS level 2 basin the region is, and a region
qualifies when that basin's peak is under ``LIMIT_FRACTION`` of float16's ceiling.

Grouping works because v2 and v3 river ids share an encoding: both are nine digits whose leading two identify the
TDXHydro region, which is the HydroBASINS level 2 basin a v3 ``region=`` directory is named for. Every v3 region's
rivers carry exactly one prefix and every one of those prefixes exists in v2, so the join is exact rather than
approximate. v2 carries three basins v3 does not (46, 48, 49), which are ignored.

Checked against what v3 actually routes: over 1940-1944 on this hydrography, every region's peak came in *below*
its v2 85 year peak -- a median of 0.45x and at most 0.83x -- so the v2 bound is conservative in the direction
that matters. It is still only a bound, so 2_rfs_v3_retro_router.py keeps a backstop: a region that overflows
anyway raises on that year before writing it and falls back to float32 from there, losing one year rather than a
record.

Regenerating
------------
    s5cmd --no-sign-request cp -c 32 's3://geoglows-v2/retrospective/maximums.zarr/*' v2_maximums.zarr/
    python fp16_regions.py --maximums v2_maximums.zarr

which prints the table and the constant below. 35 of the 47 regions qualify, covering 2,920,749 of 4,917,183
rivers, or 59.4% of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

__all__ = ['FLOAT16_MAX', 'FP16_REGIONS', 'LIMIT_FRACTION', 'region_prefix']

FLOAT16_MAX = 65504.0
# How much of float16's range a basin's all-time peak must stay under. The peak is v2's over 85 years and v3 routes
# below it, so this is margin on top of margin rather than the only thing standing between a region and an overflow.
LIMIT_FRACTION = 0.95

# Derived by the __main__ below from s3://geoglows-v2/retrospective/maximums.zarr. Do not hand edit: rerun it.
FP16_REGIONS = frozenset({
    '1020000010', '1020021940', '1020027430', '1020035180', '1020040190',
    '2020000010', '2020003440', '2020018240', '2020024230', '2020033490',
    '2020065840', '2020071190', '3020008670', '3020024310', '4020000010',
    '4020034510', '4020050220', '5020000010', '5020015660', '5020037270',
    '5020049720', '5020082270', '6020017370', '6020021870', '6020029280',
    '7020000010', '7020014250', '7020021430', '7020024600', '7020038340',
    '7020046750', '7020047840', '7020065090', '8020000010', '8020008900',
})


def region_prefix(region_dir: Path) -> int:
    """The TDXHydro region a v3 region's rivers carry, the leading two digits of any nine digit river id."""
    ids = pq.read_table(region_dir / 'routing.parquet', columns=['river_id']).column('river_id').to_numpy()
    prefixes = np.unique(ids // 10**7)
    if prefixes.size != 1:
        raise SystemExit(f'{region_dir.name} spans river id prefixes {prefixes.tolist()}, so it is not one basin')
    return int(prefixes[0])


def basin_peaks(maximums: Path) -> dict[int, float]:
    """Each TDXHydro region's highest hourly discharge over every year of the v2 retrospective."""
    import zarr

    # the store is zarr 2 with consolidated metadata; a partial download has no per array .zarray beside the chunks
    consolidated = json.loads((maximums / '.zmetadata').read_text())['metadata']
    for array in ('hourly', 'river_id'):
        target = maximums / array / '.zarray'
        if not target.exists():
            target.write_text(json.dumps(consolidated[f'{array}/.zarray']))
    river_ids = zarr.open_array(str(maximums / 'river_id'), mode='r')[:]
    annual = zarr.open_array(str(maximums / 'hourly'), mode='r')[:]  # (year, river)
    if annual.shape[1] != river_ids.shape[0]:
        raise SystemExit(f'{maximums} has {annual.shape[1]} columns of discharge for {river_ids.shape[0]} rivers')
    all_time = np.nanmax(np.where(np.isfinite(annual), annual, -np.inf), axis=0)

    prefixes = (river_ids // 10**7).astype(np.int64)
    order = np.argsort(prefixes, kind='stable')
    sorted_prefixes, sorted_peaks = prefixes[order], all_time[order]
    starts = np.flatnonzero(np.r_[True, sorted_prefixes[1:] != sorted_prefixes[:-1]])
    return dict(zip(sorted_prefixes[starts].tolist(), np.maximum.reduceat(sorted_peaks, starts).tolist(), strict=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--maximums', required=True, type=Path, help='a local copy of v2 retrospective maximums.zarr')
    parser.add_argument('--regions-root', type=Path, default=Path('/Users/rchales/data/rfsv3/hydrography'))
    args = parser.parse_args()

    limit = FLOAT16_MAX * LIMIT_FRACTION
    peaks = basin_peaks(args.maximums)
    rows = []
    for region_dir in sorted(args.regions_root.glob('region=*')):
        region = region_dir.name.split('=')[1]
        if region == 'global':
            continue
        prefix = region_prefix(region_dir)
        if prefix not in peaks:
            raise SystemExit(f'{region} is basin {prefix}, which the v2 maximums do not cover')
        rivers = pq.ParquetFile(region_dir / 'routing.parquet').metadata.num_rows
        rows.append((region, prefix, peaks[prefix], rivers))

    rows.sort(key=lambda row: -row[2])
    print(f'{"region":>12} {"basin":>6} {"v2 85yr peak":>13} {"rivers":>9}  dtype')
    for region, prefix, peak, rivers in rows:
        print(f'{region:>12} {prefix:>6} {peak:>13,.0f} {rivers:>9,}  {"float16" if peak < limit else "float32"}')

    narrow = sorted(region for region, _, peak, _ in rows if peak < limit)
    narrowed_rivers = sum(rivers for _, _, peak, rivers in rows if peak < limit)
    total = sum(rivers for *_, rivers in rows)
    print(f'\nunder {LIMIT_FRACTION:.0%} of {FLOAT16_MAX:,.0f} m3 s-1 = {limit:,.1f}: {len(narrow)} of {len(rows)} '
          f'regions, {narrowed_rivers:,} of {total:,} rivers ({100 * narrowed_rivers / total:.1f}%)')
    if set(narrow) != set(FP16_REGIONS):
        print(f'\nFP16_REGIONS differs from the constant in this file; replace it with:\n{narrow}')
    else:
        print('\nFP16_REGIONS in this file is up to date')
