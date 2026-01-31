# Data Directory

This directory contains the geospatial data files used in the analysis.

## Datasets

### 1. CHIRPS v2.0 (Rainfall)
- **Source:** https://www.chc.ucsb.edu/data/chirps
- **Resolution:** 0.05° (~5km), Daily
- **Format:** NetCDF (.nc)
- **Download:** Use the script or download manually from UCSB Climate Hazards Center

### 2. SRTM DEM (Elevation)
- **Source:** https://earthexplorer.usgs.gov/
- **Resolution:** 30m
- **Format:** GeoTIFF (.tif)

### 3. Google Open Buildings
- **Source:** https://sites.research.google/open-buildings/
- **Coverage:** Sri Lanka
- **Format:** GeoJSON or CSV

### 4. OpenStreetMap
- **Source:** https://download.geofabrik.de/asia/sri-lanka.html
- **Features:** Roads, administrative boundaries, waterways
- **Format:** Shapefile or GeoPackage

## Directory Structure

```
data/
├── raw/           # Original downloaded data (gitignored)
├── processed/     # Preprocessed data (gitignored)
└── sample/        # Small sample files for testing (tracked)
```

## Note

Large data files are NOT tracked in git. Download them separately using the instructions above.
