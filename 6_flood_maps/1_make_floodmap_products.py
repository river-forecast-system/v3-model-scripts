"""
Map the return period floods of one DEM tile with nencarta, then copy them to --copy-dir.

A tile named like N05W097_FABDEM_V1-2.tif is written to --output-dir/lon_-097/lat_05 and copied to
--copy-dir/lon=-097/lat=05. The DEMs, land cover and streamlines nencarta reads around it are every
--dem-dir/*/*.tif, --landcover-dir/*.tif and --streamlines-dir/*.parquet.

This script is designed to run in an HPC environment. The copy step takes the outputs off of the 
volatile scratch space and onto a more permanent storage location.
"""

import argparse
import os
import glob
import shutil
from pathlib import Path

for _var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS",
             "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS", "ARROW_IO_THREADS", "OMP_THREAD_LIMIT"):
    os.environ[_var] = "1"

from nencarta import process_watershed, set_log_level
set_log_level("WARNING")


def compute(args: argparse.Namespace) -> None:
    stem = args.dem.stem
    # get the lat on lon from the name like this: N40W100_FABDEM_V1-2
    lat = int(stem[1:3])
    lon = int(stem[4:7])
    lat = lat if stem[0] == "N" else -lat
    lon = lon if stem[3] == "E" else -lon
    # latitude as two digits and longitude as three, after a minus sign south and west: lat_05, lon_-097
    lat_str = f"{'-' if lat < 0 else ''}{abs(lat):02d}"
    lon_str = f"{'-' if lon < 0 else ''}{abs(lon):03d}"
    output_dir = args.output_dir / f"lon_{lon_str}"
    name = f"lat_{lat_str}"
    
    # Let us filter the source DEMs to just those that are adjacent to this tile
    adjacent_dems = []
    for dem in glob.glob(str(args.dem_dir / "*/*.tif")):
        dem_stem = Path(dem).stem
        dem_lat = int(dem_stem[1:3])
        dem_lon = int(dem_stem[4:7])
        dem_lat = dem_lat if dem_stem[0] == "N" else -dem_lat
        dem_lon = dem_lon if dem_stem[3] == "E" else -dem_lon
        if abs(dem_lat - lat) <= 1 and abs(dem_lon - lon) <= 1:
            adjacent_dems.append(dem)

    watershed_dict = {
        "output_dir": output_dir,
        "dem": str(args.dem),
        "source_dems": adjacent_dems,
        "land_cover_cache": glob.glob(str(args.landcover_dir / "*.tif")),
        "buffer": True,
        "buffer_distance": 0.1,
        "use_vrt": True,
        "use_warning_flags_to_download_dem": False,
        "source_flowlines": glob.glob(str(args.streamlines_dir / "*.parquet")),
        "mapper": "Curve2Flood-FLDPLNpy",
        "streamflow_source": "GEOGLOWS",
        "floodmap_mode": "return_period",
        "return_periods": [
            2,
            5,
            10,
            25,
            50,
            100
        ],
        "stream_id_field": "riverId",
        "downstream_id_field": "nextRiverId",
        "parallel": False,
        "profile": False,
        "quiet": True,
        "use_yaml": True,
        "clean_dem": False,
        "overwrite": False, 
        "move_stream_network_to_thalweg": True,
        "burn_streams": True,
        "project_to_utm": False,
        "streams_as_parquet": True,
        "use_parquet": True,
        "new_strm_threshold_km2": 1,
        "slope_low_percentile": 60,
        "slope_high_percentile": 74,
        "StrmOrder_Field": "strahlerOrder",
        "lakes": args.lakes,
        "land_watervalue": 80,
        "flood_waterlc_and_strm_cells": True,
        "disable_bathymetry": False,
        "bathy_use_banks": True,
        "find_banks_based_on_landcover": False,
        "use_specified_depth_for_bathy_mask": False,
        "specify_depths_for_bathy_mask": [
        1.0
        ],
        "specified_bathyflow_field": "rp2",
        "specified_highflow_field": "rp100_premium",
        "create_reach_average_curve_file": False,
        "use_dem_derived_channel_mask": 0,
        "use_power_laws_for_bathymetry": True,
        "area_m2_field": "DSContArea",
        "area_km2_field": "DSContArea_km2",
        "coefficient_depth": 0.689,
        "exponent_depth": 0.153,
        "coefficient_width": 2.32,
        "exponent_width": 0.335,
        "bathy_args": {
            "VDT_Database_NumIterations": 30,
            "Make_Output_GPKG": False,
            "X_Section_Dist": 5000.0,
            "Degree_Manip": 35,
            "Degree_Interval": 3.45,
            "Low_Spot_Range": 0,
            "Gen_Dir_Dist": 1,
            "Gen_Slope_Dist": 30,
            "Bathy_Trap_H": 0.46,
            "Stream_Slope_Method": "local_average_corrected"
        },
        "make_fist_inputs": False,
        "make_vdt": True,
        "make_curvefile": False,
        "make_ap_database": False,
        "make_cross_section_file": False,
        "make_depth_maps": False,
        "make_velocity_maps": False,
        "make_wse_maps": False,
        "make_representative_cross_section_file": True,
        "estimate_consequences": False,
        "overwrite_floodmaps": False, 
        "vdt_file_extension": "parquet",
        "floodmap_args": {
            "Make_Output_GPKG": False,
            "FS_ADJUST_FLOW_BY_FRACTION": 1.0,
            "TW_MultFact": 6,
            "TopWidthPlausibleLimit": 6000,
            "FLDPLN_Median_Filter_Size": 26,
            "FLDPLN_Max_Drop_Below_Source": 4
        },
        "fldpln_min_depth": 0.1,
        "fldpln_max_depth": 25.0,
        "fldpln_max_wse_rise": 1.47,
        "compression": "ZSTD",
        "raise_errors_if_nothing_in_domain": False,
        "name": name,
        "drop_multilinestrings": True,
        "mannings_text_file": args.n_file,
        "minimize_output_files": True,
        "return_period_file": str(args.return_period_file),
        "return_period_variables": "Q_lp3",
        "include_fdc": False,
        "raise_errors_if_river_ids_missing": False,
        "folder_paths": {
            'DEM': '{name}/nencarta_outputs',
            'ARC_InputFiles': '{name}/nencarta_outputs',
            'FloodMap': '{name}/return_period_maps',
            'Bathymetry': '{name}/nencarta_outputs',
            'DEM_Updated': '{name}/nencarta_outputs',
            'STRM': '{name}/nencarta_outputs',
            'LAND': '{name}/nencarta_outputs',
            'FLOW': '{name}/nencarta_outputs',
            'VDT': '{name}/arc',
            'ESA_LC': '{name}/nencarta_outputs',
            'FlowDirection': '{name}/nencarta_outputs',
            "FLDPLN": '{name}/fldpln'
        },
        "file_names": {
            "ARC_Flood": "",
            "VDT_Database_Bathy": "vdt",
            "Representative_XS": "representative_xsections",
            "ARC_Flood": "",
        },
        "short_file_names": True,
        "fldpln_max_drop_below_source": 8,
        "fldpln_max_cells_per_segment": 1e6,
    }
    process_watershed(watershed_dict)

    # Copy everything from the output dir to a new dir after processing is complete,
    # only if every return period has its flood map (indicating success for a tile)
    maps_dir = output_dir / watershed_dict["folder_paths"]["FloodMap"].format(name=name)
    if args.copy_dir and len(glob.glob(str(maps_dir / "*.tif"))) == len(watershed_dict["return_periods"]):
        new_output_dir = args.copy_dir / f"lon={lon_str}" / name.replace('_', '=')
        shutil.copytree(str(output_dir / name), str(new_output_dir), dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dem', type=Path, required=True, help='the DEM tile to map')
    parser.add_argument('--dem-dir', type=Path, required=True, help='folders of DEM tiles, <10x10 degree dir>/*.tif')
    parser.add_argument('--landcover-dir', type=Path, required=True, help='land cover tiles, *.tif')
    parser.add_argument('--streamlines-dir', type=Path, required=True, help='streamlines, *.parquet')
    parser.add_argument('--lakes', type=Path, required=True, help='lake polygons parquet')
    parser.add_argument('--n-file', type=Path, required=True, help="Manning's n by land cover class, tsv")
    parser.add_argument('--return-period-file', type=Path, required=True, help='return period flows netcdf')
    parser.add_argument('--output-dir', type=Path, required=True, help='where nencarta writes each tile')
    parser.add_argument('--copy-dir', type=Path, help='where each finished tile is copied; omitted copies nothing')
    args = parser.parse_args()

    compute(args)
