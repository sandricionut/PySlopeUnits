# Changelog

## 0.1.0

- Initial public package structure for PySlopeUnits.
- Exposes the hierarchical graph-based pipeline through `SlopeUnits`.
- Adds a command-line interface.
- Uses open-source Fiona + Rasterio for GeoPackage export; no proprietary runtime dependency is required.
- Adds Linux, Windows and macOS GitHub Actions tests.
- Keeps internal research-cache identifiers where required for compatibility; these are not part of the public API.
