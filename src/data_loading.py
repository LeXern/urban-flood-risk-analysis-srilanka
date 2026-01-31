"""
Data Loading Module

Functions for loading various geospatial datasets:
- CHIRPS rainfall data (NetCDF)
- SRTM DEM (GeoTIFF)
- Google Building footprints (GeoJSON)
- OpenStreetMap features (Shapefile/GeoPackage)
"""

import xarray as xr
import rioxarray  # noqa: F401 - needed for xarray rio accessor
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Union, Optional


def load_chirps_data(
    filepath: Union[str, Path],
    variable: str = "precip",
    time_slice: Optional[tuple] = None
) -> xr.DataArray:
    """
    Load CHIRPS rainfall data from NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CHIRPS NetCDF file
    variable : str
        Variable name in the NetCDF (default: 'precip')
    time_slice : tuple, optional
        Start and end dates for temporal subset, e.g., ('2020-01-01', '2020-12-31')
    
    Returns
    -------
    xr.DataArray
        Rainfall data as xarray DataArray with dimensions (time, lat, lon)
    
    Example
    -------
    >>> rainfall = load_chirps_data('data/chirps_2020.nc')
    >>> print(rainfall.dims)
    ('time', 'latitude', 'longitude')
    """
    # open the netcdf file
    ds = xr.open_dataset(filepath)
    data = ds[variable]
    
    # apply time slice if provided
    if time_slice is not None:
        start_date, end_date = time_slice
        data = data.sel(time=slice(start_date, end_date))
    
    return data


def load_dem(
    filepath: Union[str, Path],
    clip_bounds: Optional[tuple] = None
) -> xr.DataArray:
    """
    Load Digital Elevation Model from GeoTIFF.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the DEM GeoTIFF file
    clip_bounds : tuple, optional
        Bounding box to clip (minx, miny, maxx, maxy)
    
    Returns
    -------
    xr.DataArray
        Elevation data with CRS information
    
    Example
    -------
    >>> dem = load_dem('data/srtm_sri_lanka.tif')
    >>> print(dem.rio.crs)
    EPSG:4326
    """
    # load raster using rioxarray
    dem = xr.open_dataarray(filepath, engine='rasterio')
    
    # clip to bounds if provided
    if clip_bounds is not None:
        minx, miny, maxx, maxy = clip_bounds
        dem = dem.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    
    return dem


def load_buildings(
    filepath: Union[str, Path],
    bbox: Optional[tuple] = None,
    confidence_threshold: float = 0.7
) -> gpd.GeoDataFrame:
    """
    Load Google Building footprints from GeoJSON.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the buildings GeoJSON file
    bbox : tuple, optional
        Bounding box filter (minx, miny, maxx, maxy)
    confidence_threshold : float
        Minimum confidence score to include (default: 0.7)
    
    Returns
    -------
    gpd.GeoDataFrame
        Building footprints as polygons
    
    Example
    -------
    >>> buildings = load_buildings('data/google_buildings_lk.geojson')
    >>> print(f"Loaded {len(buildings)} buildings")
    """
    # load with optional bbox filter
    if bbox is not None:
        buildings = gpd.read_file(filepath, bbox=bbox)
    else:
        buildings = gpd.read_file(filepath)
    
    # filter by confidence if column exists
    if 'confidence' in buildings.columns:
        buildings = buildings[buildings['confidence'] >= confidence_threshold]
    
    return buildings


def load_osm_roads(
    filepath: Union[str, Path],
    road_types: Optional[list] = None
) -> gpd.GeoDataFrame:
    """
    Load road network from OpenStreetMap data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the OSM roads shapefile or geopackage
    road_types : list, optional
        Filter by highway types, e.g., ['primary', 'secondary', 'trunk']
    
    Returns
    -------
    gpd.GeoDataFrame
        Road network as LineStrings
    
    Example
    -------
    >>> roads = load_osm_roads('data/osm_roads.shp', road_types=['primary', 'secondary'])
    >>> print(f"Loaded {len(roads)} road segments")
    """
    roads = gpd.read_file(filepath)
    
    # filter by road type if specified
    if road_types is not None and 'highway' in roads.columns:
        roads = roads[roads['highway'].isin(road_types)]
    
    return roads


def load_admin_boundaries(
    filepath: Union[str, Path],
    level: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load administrative boundaries.
    
    Parameters
    ----------
    filepath : str or Path
        Path to boundaries shapefile or geopackage
    level : str, optional
        Admin level to filter (e.g., 'district', 'division')
    
    Returns
    -------
    gpd.GeoDataFrame
        Administrative boundary polygons
    """
    boundaries = gpd.read_file(filepath)
    
    # filter by admin level if column exists
    if level is not None and 'admin_level' in boundaries.columns:
        boundaries = boundaries[boundaries['admin_level'] == level]
    
    return boundaries


def validate_crs_match(
    *datasets: Union[xr.DataArray, gpd.GeoDataFrame]
) -> bool:
    """
    Check if all datasets have matching CRS.
    
    Parameters
    ----------
    *datasets : xr.DataArray or gpd.GeoDataFrame
        Variable number of datasets to compare
    
    Returns
    -------
    bool
        True if all CRS match
    
    Example
    -------
    >>> if not validate_crs_match(buildings, admin_boundaries):
    ...     admin_boundaries = admin_boundaries.to_crs(buildings.crs)
    """
    crs_list = []
    
    for ds in datasets:
        if hasattr(ds, 'rio') and hasattr(ds.rio, 'crs'):
            # rioxarray raster
            crs_list.append(str(ds.rio.crs))
        elif hasattr(ds, 'crs'):
            # geopandas geodataframe
            crs_list.append(str(ds.crs))
        else:
            continue
    
    # check if all CRS are the same
    return len(set(crs_list)) <= 1


if __name__ == "__main__":
    # quick test
    print("Data loading module loaded successfully")
    print("Available functions:")
    print("  - load_chirps_data()")
    print("  - load_dem()")
    print("  - load_buildings()")
    print("  - load_osm_roads()")
    print("  - load_admin_boundaries()")
    print("  - validate_crs_match()")
