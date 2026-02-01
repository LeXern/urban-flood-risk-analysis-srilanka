"""
Raster Analysis Module

NumPy and Xarray operations for rainfall and elevation data analysis.
Implements array-based masking, normalization, and temporal statistics.
"""

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter, gaussian_filter, distance_transform_edt
from typing import Union, Optional, Tuple, Dict
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping


def mask_raster_with_vector(
    raster: xr.DataArray,
    vector_gdf: gpd.GeoDataFrame,
    nodata: float = np.nan
) -> xr.DataArray:
    """
    Mask raster data using vector polygons.
    Requires 'rioxarray' extension to be loaded.
    
    Parameters
    ----------
    raster : xr.DataArray
        Input raster
    vector_gdf : gpd.GeoDataFrame
        Vector mask (polygons)
    nodata : float
        Value to fill outside mask
    
    Returns
    -------
    xr.DataArray
        Masked raster
    """
    # ensure crs match
    if not hasattr(raster, 'rio'):
        raise AttributeError("DataArray doesn't have rio accesssor. Did you import rioxarray?")
        
    if raster.rio.crs != vector_gdf.crs:
        vector_gdf = vector_gdf.to_crs(raster.rio.crs)
        
    # mask
    masked = raster.rio.clip(
        vector_gdf.geometry.apply(mapping),
        vector_gdf.crs,
        drop=True,
        invert=False,
        all_touched=True
    )
    
    return masked


def create_extreme_rainfall_mask(
    rainfall: Union[np.ndarray, xr.DataArray],
    threshold: float = 100.0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Create a boolean mask for extreme rainfall events.
    
    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        Rainfall data (can be 2D or 3D with time dimension)
    threshold : float
        Rainfall threshold in mm (default: 100mm)
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Boolean mask where True = extreme rainfall
    """
    # simple element-wise comparison - no loops needed
    mask = rainfall > threshold
    return mask


def count_extreme_events(
    rainfall: Union[np.ndarray, xr.DataArray],
    threshold: float = 100.0,
    time_axis: int = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Count the number of extreme rainfall days per pixel.
    
    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        3D rainfall data (time, lat, lon)
    threshold : float
        Rainfall threshold in mm
    time_axis : int
        Axis representing time dimension
    
    Returns
    -------
    np.ndarray or xr.DataArray
        2D array with count of extreme events per pixel
    """
    extreme_mask = rainfall > threshold
    
    if isinstance(extreme_mask, xr.DataArray):
        return extreme_mask.sum(dim='time')
    else:
        return np.sum(extreme_mask, axis=time_axis)


def calculate_percentile_rainfall(
    rainfall: Union[np.ndarray, xr.DataArray],
    percentile: float = 95.0,
    time_axis: int = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate percentile rainfall value for each pixel.
    
    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        3D rainfall data (time, lat, lon)
    percentile : float
        Percentile to calculate (e.g., 95 for 95th percentile)
    time_axis : int
        Axis representing time dimension
    
    Returns
    -------
    np.ndarray or xr.DataArray
        2D array with percentile values
    """
    if isinstance(rainfall, xr.DataArray):
        return rainfall.quantile(percentile / 100.0, dim='time')
    else:
        return np.percentile(rainfall, percentile, axis=time_axis)


def normalize_array(
    data: Union[np.ndarray, xr.DataArray],
    method: str = 'minmax'
) -> Union[np.ndarray, xr.DataArray]:
    """
    Normalize array values to 0-1 range.
    
    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Input data to normalize
    method : str
        Normalization method: 'minmax' or 'zscore'
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Normalized data
    """
    if method == 'minmax':
        # min-max normalization to [0, 1]
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'zscore':
        # z-score normalization
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax' or 'zscore'")


def identify_low_elevation_areas(
    dem: Union[np.ndarray, xr.DataArray],
    percentile: float = 25.0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Identify low-lying areas based on elevation percentile.
    
    Parameters
    ----------
    dem : np.ndarray or xr.DataArray
        Digital Elevation Model data
    percentile : float
        Percentile threshold (areas below this are 'low')
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Boolean mask where True = low elevation area
    """
    threshold_value = np.nanpercentile(dem, percentile)
    low_mask = dem < threshold_value
    return low_mask


def calculate_annual_maximum(
    rainfall: xr.DataArray,
    dim: str = 'time'
) -> xr.DataArray:
    """
    Calculate annual maximum daily rainfall.
    
    Parameters
    ----------
    rainfall : xr.DataArray
        Daily rainfall data with time dimension
    dim : str
        Name of time dimension
    
    Returns
    -------
    xr.DataArray
        Annual maximum rainfall per year and location
    """
    # Handle files with no 'year' group (e.g. single year data)
    try:
        if 'year' in rainfall.indexes[dim].names: # already multi-indexed
           pass 
        elif pd.api.types.is_datetime64_any_dtype(rainfall.indexes[dim]):
           return rainfall.groupby(f'{dim}.year').max(dim=dim)
        
        # Fallback for simple index
        return rainfall.max(dim=dim)
    except:
        return rainfall.max(dim=dim)


def smooth_raster(
    data: Union[np.ndarray, xr.DataArray],
    method: str = 'gaussian',
    size: int = 3
) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply spatial smoothing to raster data.
    
    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        2D raster data
    method : str
        Smoothing method: 'gaussian' or 'uniform'
    size : int
        Kernel size (for uniform) or sigma (for gaussian)
    
    Returns
    -------
    np.ndarray or xr.DataArray
        Smoothed raster
    """
    # extract numpy array if xarray
    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        values = data.values
        coords = data.coords
        dims = data.dims
    else:
        values = data
    
    # apply filter
    if method == 'gaussian':
        smoothed = gaussian_filter(values.astype(float), sigma=size)
    elif method == 'uniform':
        smoothed = uniform_filter(values.astype(float), size=size)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # convert back to xarray if needed
    if is_xarray:
        return xr.DataArray(smoothed, coords=coords, dims=dims)
    return smoothed


def calculate_standardized_anomalies(
    data: xr.DataArray,
    dim: str = 'time'
) -> xr.DataArray:
    """
    Calculate standardized anomalies (Z-scores) along a dimension.
    z = (x - mean) / std
    
    Parameters
    ----------
    data : xr.DataArray
        Input data
    dim : str
        Dimension to calculate statistics over
        
    Returns
    -------
    xr.DataArray
        Data converted to Z-scores
    """
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)
    # Avoid division by zero
    std = std.where(std != 0, 1.0)
    anomalies = (data - mean) / std
    return anomalies


def calculate_slope(
    dem: xr.DataArray,
    pixel_size: float = 30.0
) -> xr.DataArray:
    """
    Calculate terrain slope from DEM (simplified gradient method).
    
    Parameters
    ----------
    dem : xr.DataArray
        Elevation data
    pixel_size : float
        Pixel size in meters (e.g. 30m for SRTM)
        
    Returns
    -------
    xr.DataArray
        Slope in degrees
    """
    # Guard for very small rasters (need at least 2 cells per dimension)
    if dem.ndim != 2 or min(dem.shape) < 2:
        return xr.zeros_like(dem, dtype=float)

    # Calculate gradients in Y and X directions
    # Note: np.gradient returns [dY, dX] for 2D array
    dy, dx = np.gradient(dem.values, pixel_size)
    
    # Slope = arctan(sqrt(dx^2 + dy^2)) * (180/pi)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return xr.DataArray(slope_deg, coords=dem.coords, dims=dem.dims)


def calculate_euclidean_distance(
    raster_shape: Tuple[int, int],
    target_indices: Tuple[np.ndarray, np.ndarray],
    pixel_size: float = 30.0
) -> np.ndarray:
    """
    Calculate Euclidean distance from target pixels to all other pixels.
    Uses SciPy's optimized distance_transform_edt.
    
    Parameters
    ----------
    raster_shape : tuple
        Shape of the raster (height, width)
    target_indices : tuple of arrays
        (row_indices, col_indices) of the target feature (e.g. river pixels)
    pixel_size : float
        Resolution of pixels in meters (to convert pixel distance to meters)
        
    Returns
    -------
    np.ndarray
        Distance grid in meters
    """
    # Create boolean mask (0 = feature, 1 = background) for transform
    # Note: distance_transform_edt calculates distance to nearest ZERO
    mask = np.ones(raster_shape, dtype=int)
    mask[target_indices] = 0
    
    # Calculate distance in pixel units
    dist_pixels = distance_transform_edt(mask)
    
    # Convert to meters
    dist_meters = dist_pixels * pixel_size
    
    return dist_meters


def ahp_weighted_overlay(
    layers: Dict[str, xr.DataArray],
    weights: Dict[str, float]
) -> xr.DataArray:
    """
    Perform Multi-Criteria Decision Analysis using Weighted Linear Combination.
    Each layer is normalized (0-1) before weighting.
    
    Parameters
    ----------
    layers : dict
        Dictionary of input rasters (name -> DataArray)
    weights : dict
        Dictionary of weights (name -> float). Should sum to 1.0.
        
    Returns
    -------
    xr.DataArray
        Combined suitability/risk map (0-1 range)
    """
    combined = None
    
    for name, layer in layers.items():
        if name not in weights:
            continue
            
        # Normalize layer to 0-1
        norm_layer = normalize_array(layer, method='minmax')
        
        weight = weights[name]
        
        if combined is None:
            combined = norm_layer * weight
        else:
            combined += norm_layer * weight
            
    return combined


def calculate_vulnerability_index(
    rainfall: Union[np.ndarray, xr.DataArray],
    building_density: Union[np.ndarray, xr.DataArray],
    elevation: Union[np.ndarray, xr.DataArray],
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)
) -> Union[np.ndarray, xr.DataArray]:
    """
    Combine normalized hazard/exposure components into a single vulnerability index.

    Parameters
    ----------
    rainfall : np.ndarray or xr.DataArray
        Normalized rainfall intensity (0-1, higher = more hazardous)
    building_density : np.ndarray or xr.DataArray
        Normalized building/urban density (0-1, higher = more exposed)
    elevation : np.ndarray or xr.DataArray
        Normalized elevation (0-1, higher = safer). Will be inverted internally.
    weights : tuple of floats
        Weights for (rainfall, building_density, elevation). Should sum to 1.

    Returns
    -------
    np.ndarray or xr.DataArray
        Vulnerability index in range [0, 1]
    """
    if len(weights) != 3:
        raise ValueError("weights must be a tuple of three values (rainfall, building, elevation)")

    w_rain, w_build, w_elev = weights

    # Invert elevation so low-lying areas get higher vulnerability
    inv_elevation = 1 - elevation

    # Weighted linear combination; works for both numpy arrays and xarray DataArrays
    vulnerability = (
        (rainfall * w_rain) +
        (building_density * w_build) +
        (inv_elevation * w_elev)
    )

    return vulnerability
