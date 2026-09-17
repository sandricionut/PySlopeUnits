# PySlopeUnits

**A hierarchical graph-based framework for scalable slope-unit delineation from massive digital elevation models**

PySlopeUnits is an open-source Python package for delineating geomorphological slope units from digital elevation models (DEMs). It is designed to use the same scientific parameterization from local or regional DEMs to grids containing hundreds of millions of cells, while adapting memory use and intermediate storage to the available hardware.

Version **0.1.8** is the software release used for the benchmark experiments associated with the *Environmental Modelling & Software* manuscript.

## Main features

- raster-native slope-unit delineation from a DEM;
- A*-based hydrological routing and multiple-flow-direction (MFD) accumulation;
- multi-threshold stream-derived half-basin candidates;
- lazy hierarchical graph evaluation for bounded-memory execution;
- adaptive RAM / memory-mapped temporary storage;
- process-level parallelism and Numba-compiled raster kernels;
- checkpoint and restart support for long runs;
- optional cleaning, diagnostics and GeoPackage vector export;
- automatic 32/64-bit indexing according to grid size.

The scalable execution path is:

```text
DEM / VRT / mosaic
        │
        ▼
Canonical processing grid + resource planning
        │
        ▼
A* routing + MFD accumulation
        │
        ▼
Multi-threshold stream-derived half-basins
        │
        ▼
Lazy hierarchical graph evaluation
        │
        ▼
Gap completion + 4-neighbour clumping
        │
        ▼
Optional cleaning
        │
        ▼
GeoTIFF + optional GeoPackage / diagnostics
```

## Requirements

- Python >= 3.10
- NumPy >= 1.26
- Numba >= 0.59
- Rasterio >= 1.3
- SciPy >= 1.11
- Fiona >= 1.9
- Shapely >= 2.0
- psutil >= 5.9

Linux, macOS and Windows are supported.

## Installation

Clone the repository and install in editable mode:

```bash
git clone <repository-url>
cd PySlopeUnits
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Verify the installation:

```bash
pyslopeunits --help
python -c "import pyslopeunits; print(pyslopeunits.__version__)"
```

The second command should print:

```text
0.1.8
```

## Quick start

For a projected DEM with metric square cells, the native CRS and resolution can be retained:

```bash
pyslopeunits dem.tif run
```

To define the processing grid explicitly:

```bash
pyslopeunits dem.tif run \
  --target-crs EPSG:3035 \
  --fine-resolution-m 30
```

The same command is used for small and very large datasets. PySlopeUnits detects available resources and selects the execution plan automatically.

## Reference delineation parameters

The following configuration was used in the matched benchmark against GRASS `r.slopeunits`:

```bash
pyslopeunits dem.tif run \
  --target-crs EPSG:3035 \
  --fine-resolution-m 30 \
  --threshold-m2 250000 \
  --min-area-m2 100000 \
  --cv-min 0.25 \
  --reduction-factor 2 \
  --max-iterations 12 \
  --convergence 5 \
  --clean-size-m2 25000 \
  --hierarchy-mode lazy \
  --candidate-cache-mode auto
```

These parameters correspond to:

| Parameter | Meaning | Benchmark value |
|---|---|---:|
| `threshold_m2` | initial contributing-area threshold | 250,000 m² |
| `min_area_m2` | minimum slope-unit area | 100,000 m² |
| `cv_min` | aspect circular-variance threshold | 0.25 |
| `reduction_factor` | threshold reduction factor | 2 |
| `max_iterations` | maximum hierarchy levels | 12 |
| `convergence` | MFD convergence exponent | 5 |
| `clean_size_m2` | post-processing cleaning threshold | 25,000 m² |

Area thresholds are converted internally to integer cell counts from the processing-grid cell area.

## Memory and large DEMs

PySlopeUnits does not require all intermediate arrays to remain in physical RAM. The resource planner uses an adaptive hybrid model:

- reusable arrays are retained in RAM when this is safe and efficient;
- large persistent structures are stored as memory-mapped arrays when necessary;
- temporary working arrays spill to disk when the configured RAM budget would be exceeded;
- the lazy hierarchy keeps only the active graph frontier instead of materializing all descendants;
- candidate rasters can be cached or streamed according to available disk capacity.

Version 0.1.8 further reduces write amplification during MFD accumulation. The mutable accumulation working set is kept in RAM when the resource budget allows it and otherwise uses an exact disk-backed fallback. After accumulation, the result is persisted once, sequentially; the adjusted-receiver array is allocated only when it is needed.

A user-defined upper memory budget can be supplied explicitly:

```bash
pyslopeunits dem.tif run \
  --memory-limit-gb 16 \
  --workers 8 \
  --numba-threads 8
```

For planning without running the full delineation:

```bash
pyslopeunits dem.tif run \
  --target-crs EPSG:3035 \
  --fine-resolution-m 30 \
  --plan-only
```

## Hierarchy and candidate storage

The scalable defaults are:

```text
--hierarchy-mode lazy
--candidate-cache-mode auto
```

`lazy` evaluates only the active hierarchical frontier and is recommended for normal use and large datasets.

`materialized` constructs the complete reusable candidate DAG and is primarily useful for controlled experiments or repeated parameter evaluations where the full hierarchy is intentionally retained.

Candidate storage modes are:

- `auto`: choose according to disk capacity;
- `all`: retain all threshold candidate rasters;
- `stream`: retain only the working threshold.

## Checkpoint and restart

Checkpointing is enabled by default. Long exact routing stages periodically store compact restart metadata while the large state arrays remain in persistent memory-mapped files.

```bash
pyslopeunits dem.tif run --checkpoint-minutes 10
```

Use:

```bash
--no-checkpoint
```

for controlled benchmark runs without checkpoints, or:

```bash
--restart
```

to ignore compatible partial checkpoints and recompute long-running stages.

## NoData

Raster metadata NoData is always respected. Additional sentinel values can be supplied explicitly:

```bash
pyslopeunits dem.tif run --nodata -32767 32767
```

## Python API

```python
from pyslopeunits import AdaptiveSlopeUnits

model = AdaptiveSlopeUnits(
    target_crs="EPSG:3035",
    fine_resolution_m=30,
    memory_limit_gb=16,
    workers=8,
    numba_threads=8,
    threshold_m2=250_000,
    min_area_m2=100_000,
    cv_min=0.25,
    reduction_factor=2,
    max_iterations=12,
    convergence=5,
    clean_size_m2=25_000,
    hierarchy_mode="lazy",
    candidate_cache_mode="auto",
)

result = model.run("dem.tif", "run")
print(result)
```

A complete executable example is provided in [`examples/run_dem.py`](examples/run_dem.py).

## Typical outputs

Depending on the selected options, an output directory may contain:

```text
run/
├── slope_units.tif
├── slope_units_raw.tif
├── slope_units.gpkg
├── processing_plan.json
└── workspace/
```

Vector export and diagnostic products can be disabled for large benchmark runs:

```bash
pyslopeunits dem.tif run --no-vector --no-diagnostics
```

## Tests

The test suite should remain part of the repository. Run it before committing a release:

```bash
python -m pytest -q
```

The release overlay also includes a small `test_release_v018.py` smoke test; it is intended to complement, not replace, the existing project tests.

## Reproducibility

For a manuscript or benchmark, record at least:

- PySlopeUnits version and Git commit/tag;
- DEM source, CRS and processing resolution;
- all delineation parameters;
- hierarchy and candidate-cache modes;
- number of workers and Numba threads;
- memory limit;
- whether checkpointing, vectorization and diagnostics were enabled.

For the manuscript release, tag the exact tested commit (for example `v0.1.8`) instead of citing the moving `main` branch.

## Citation

If you use PySlopeUnits in scientific work, please cite the associated manuscript once published. Until then, cite the software repository and the exact release/tag used in the analysis.

Manuscript title:

> **PySlopeUnits: A scalable graph-based framework for parallel slope-unit delineation from massive digital elevation models**

## Version 0.1.8

Version 0.1.8 introduces adaptive handling of the write-intensive MFD accumulation working set. Relative to v0.1.7, the main changes are:

- RAM-first MFD accumulation when the configured safety budget permits it;
- exact disk-backed fallback for larger workloads;
- one-time sequential persistence of RAM-backed accumulation;
- rename-based promotion of disk-backed accumulation;
- delayed creation of the adjusted-receiver array;
- reduced random write pressure on memory-mapped storage.

These changes target large-grid I/O performance without changing the scientific delineation parameters or output semantics.
