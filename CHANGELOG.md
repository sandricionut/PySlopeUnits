# Changelog

## 0.1.0

- Initial public package structure for PySlopeUnits.
- Exposes the hierarchical graph-based pipeline through `SlopeUnits` and the adaptive interface through `AdaptiveSlopeUnits`.
- Uses open-source Fiona + Rasterio for GeoPackage export; no proprietary runtime dependency is required.
- Adds Linux, Windows and macOS GitHub Actions tests.

### Development: universal adaptive execution

- Replaced the dataset-specific `continental` CLI with one command for local, regional and very-large DEMs.
- Added automatic projected-grid resolution handling and explicit reprojection when requested.
- Added automatic CPU/RAM/disk resource detection and hydrological resource planning.
- Added RAM-first temporary allocation with transparent disk spill while keeping shared/persistent arrays file-backed for cross-platform multiprocessing.
- Added 64-bit flat-cell indexing for grids whose total cell count exceeds the signed 32-bit range.
- Added bounded-memory lazy hierarchical graph evaluation: only the active graph frontier is materialized and child statistics are reduced through disk buckets.
- Retained the complete materialized DAG as an optional regression/research mode.
- Added adaptive candidate storage: reusable all-level cache for moderate datasets or one-level-at-a-time streaming for massive grids.
- Preserved exact global fine-grid hydrology so resource-planning domains never introduce artificial slope-unit boundaries.
