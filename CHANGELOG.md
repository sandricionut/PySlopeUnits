# Changelog

## 0.1.8

- Added an adaptive RAM/disk working set for MFD accumulation.
- Uses RAM for the write-intensive accumulation phase when the configured resource budget permits it.
- Retains an exact disk-backed fallback for larger workloads.
- Persists RAM-backed accumulation once using sequential block copies.
- Promotes disk-backed accumulation by file rename rather than a second full copy.
- Defers creation of the adjusted-receiver raster until accumulation is complete.
- Reduces random write pressure and write amplification on memory-mapped storage.
- Preserves the v0.1.7 scientific parameterization and delineation semantics.
