# PySlopeUnits

**A hierarchical graph-based algorithm for scalable slope-unit delineation from massive digital elevation models**

PySlopeUnits is an open-source Python implementation for large-scale delineation of geomorphological slope units from digital elevation models (DEMs). The current pipeline uses reusable memory-mapped hydrological caches, multi-level half-basin candidates, a hierarchical candidate DAG, graph-based selection, residual assignment, and final 4-neighbour connected-component consolidation.

Research software associated with a manuscript in preparation for *Environmental Modelling & Software*.

## Platforms

PySlopeUnits is packaged for Linux, Windows and macOS and is continuously import-tested on all three platforms through GitHub Actions.

## Installation

```bash
python -m pip install -e .
```

## Python API

```python
from pyslopeunits import SlopeUnits

model = SlopeUnits(
    threshold_m2=250_000,
    min_area_m2=100_000,
    cv_min=0.25,
    workers=8,
    numba_threads=8,
)

result = model.run(
    "dem.tif",
    "slope_units.tif",
    work_dir="work",
)
```

## Command line

```bash
pyslopeunits dem.tif slope_units.tif --work-dir work --workers 8
```

GeoPackage export uses only open-source Fiona + Rasterio.

## Workflow

```text
DEM
 │
 ▼
A*-based terrain routing + MFD accumulation
 │
 ▼
Reusable memory-mapped hydrological cache
 │
 ▼
Multi-level half-basin candidates
 │
 ▼
Hierarchical candidate DAG
 │
 ▼
Graph-based selection + residual assignment
 │
 ▼
4-neighbour connected components
 │
 ▼
GeoTIFF + optional GeoPackage
```

## Examples

```bash
python examples/run_10m_20m.py dem_20m.tif dem_10m.tif
python examples/benchmark_cache_modes.py dem_20m.tif
```

Large runtime caches, DEMs and generated geospatial outputs are excluded from version control.

## Citation

See `CITATION.cff`. The article DOI and software DOI will be added after publication/release.

## Licence

A public software licence will be selected after the source-provenance audit documented in `PROVENANCE.md` is complete.
