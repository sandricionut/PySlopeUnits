from __future__ import annotations

from .logging_utils import log as print

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterGrid:
    data: np.ndarray
    valid: np.ndarray
    transform: rasterio.Affine
    crs: object
    nodata: float | int | None

    @property
    def cell_area(self) -> float:
        return abs(float(self.transform.a) * float(self.transform.e))

    @property
    def xres(self) -> float:
        return abs(float(self.transform.a))

    @property
    def yres(self) -> float:
        return abs(float(self.transform.e))


@dataclass(frozen=True)
class RasterMeta:
    shape: tuple[int, int]
    transform: rasterio.Affine
    crs: object
    nodata: float | int | None

    @property
    def cell_area(self) -> float:
        return abs(float(self.transform.a) * float(self.transform.e))

    @property
    def xres(self) -> float:
        return abs(float(self.transform.a))

    @property
    def yres(self) -> float:
        return abs(float(self.transform.e))


def normalize_nodata_values(values) -> tuple[float, ...]:
    """Canonicalize user-supplied additional NoData values.

    Raster metadata NoData is always honored separately by Rasterio.  This
    tuple therefore contains *additional* values only.
    """
    if values is None:
        return ()
    out: list[float] = []
    for value in values:
        v = float(value)
        # Preserve NaN only once. All non-finite DEM values are invalid anyway.
        if math.isnan(v):
            if not any(math.isnan(x) for x in out):
                out.append(v)
            continue
        if v not in out:
            out.append(v)
    return tuple(out)


def _apply_extra_nodata_mask(data: np.ndarray, valid: np.ndarray, values) -> None:
    for value in normalize_nodata_values(values):
        if math.isnan(value):
            valid &= ~np.isnan(data)
        else:
            valid &= data != value


def read_dem(path: str | Path, *, nodata_values=None) -> RasterGrid:
    """Read a DEM and build the valid mask.

    Validity is determined from:
      1. the raster mask / metadata NoData,
      2. finite numeric values,
      3. optional user-supplied additional NoData values.

    No DEM-specific sentinel is hard-coded.
    """
    path = Path(path)
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype(np.float64)
        data = np.asarray(arr.filled(np.nan), dtype=np.float64)
        mask = np.ma.getmaskarray(arr)
        valid = ~np.asarray(mask, dtype=bool)
        valid &= np.isfinite(data)
        _apply_extra_nodata_mask(data, valid, nodata_values)
        data[~valid] = np.nan
        return RasterGrid(
            data=data,
            valid=valid,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
        )


def normalize_source_nodata(
    source,
    output: str | Path,
    *,
    nodata_values=None,
    verbose: bool = False,
) -> str:
    """Collapse metadata + additional NoData values to one NaN-masked raster.

    This normalization is needed before reprojection/resampling when the user
    supplies extra sentinel values.  Otherwise bilinear/average resampling can
    mix an undeclared sentinel into neighboring valid elevations.

    When no additional values are supplied, ``source`` is returned unchanged.
    """
    values = normalize_nodata_values(nodata_values)
    if not values:
        return str(source)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".json")

    with rasterio.open(source) as src:
        if src.count < 1:
            raise ValueError("Source DEM contains no raster bands")

        source_stat = None
        try:
            p = Path(source)
            st = p.stat()
            source_stat = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except Exception:
            pass

        expected = {
            "source": str(source),
            "source_stat": source_stat,
            "shape": [int(src.height), int(src.width)],
            "crs": None if src.crs is None else str(src.crs),
            "transform": [float(x) for x in src.transform[:6]],
            "metadata_nodata": None if src.nodata is None else float(src.nodata),
            "extra_nodata_values": [
                "nan" if math.isnan(v) else float(v) for v in values
            ],
            "normalization_version": 1,
        }

        if output.exists() and manifest_path.exists():
            try:
                got = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                got = None
            if got == expected:
                if verbose:
                    print(f"[PySlopeUnits] reusing normalized NoData source: {output}")
                return str(output)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float64",
            nodata=np.nan,
            compress="zstd",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        if verbose:
            metadata_text = "none" if src.nodata is None else repr(src.nodata)
            print(
                "[PySlopeUnits] normalizing NoData before resampling | "
                f"metadata={metadata_text} | extra={list(values)}"
            )

        with rasterio.open(output, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                arr = src.read(1, window=win, masked=True, out_dtype="float64")
                data = np.asarray(arr.filled(np.nan), dtype=np.float64)
                valid = ~np.asarray(np.ma.getmaskarray(arr), dtype=bool)
                valid &= np.isfinite(data)
                _apply_extra_nodata_mask(data, valid, values)
                data[~valid] = np.nan
                dst.write(data, 1, window=win)

        manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")

    return str(output)



def prepare_dem_memmaps_blockwise(
    path: str | Path,
    store,
    meta: RasterMeta,
    *,
    nodata_values=None,
    hydro_scale: int = 1000,
    numba_threads: int = 1,
    max_block_cells: int = 2_000_000,
    verbose: bool = True,
) -> int:
    """Prepare DEM-dependent global arrays without loading the full DEM in RAM.

    The raster is read in row blocks with a one-row halo.  The halo preserves
    the exact central-gradient aspect result at internal block boundaries.
    Output arrays are memory mapped in ``store`` and are therefore suitable for
    datasets larger than physical RAM.

    Returns the number of valid cells.
    """
    from numba import set_num_threads
    from .kernels import aspect_sincos, quantize_dem

    path = Path(path)
    rows, cols = meta.shape
    block_rows = max(1, min(rows, int(max_block_cells) // max(1, cols)))
    values = normalize_nodata_values(nodata_values)

    valid_mm = store.create("valid", meta.shape, np.uint8, fill=0)
    hydro_mm = store.create("hydro_dem", meta.shape, np.int32, fill=0)
    sin_mm = store.create("sin_aspect", meta.shape, np.float32, fill=0.0)
    cos_mm = store.create("cos_aspect", meta.shape, np.float32, fill=0.0)
    good_mm = store.create("aspect_valid", meta.shape, np.uint8, fill=0)

    set_num_threads(max(1, int(numba_threads)))
    nvalid = 0

    if verbose:
        print(
            f"[PySlopeUnits] DEM blockwise -> memmap | "
            f"block_rows={block_rows:,} | max_block_cells={int(max_block_cells):,}"
        )

    with rasterio.open(path) as ds:
        if (ds.height, ds.width) != tuple(meta.shape):
            raise RuntimeError("DEM shape changed during blockwise read")
        if ds.crs != meta.crs or not ds.transform.almost_equals(meta.transform):
            raise RuntimeError("DEM grid changed during blockwise read")

        for r0 in range(0, rows, block_rows):
            r1 = min(rows, r0 + block_rows)
            h0 = max(0, r0 - 1)
            h1 = min(rows, r1 + 1)
            win = Window(0, h0, cols, h1 - h0)

            arr = ds.read(1, window=win, masked=True, out_dtype="float64")
            data = np.asarray(arr.filled(np.nan), dtype=np.float64)
            local_valid = ~np.asarray(np.ma.getmaskarray(arr), dtype=bool)
            local_valid &= np.isfinite(data)
            _apply_extra_nodata_mask(data, local_valid, values)
            data[~local_valid] = np.nan
            local_valid_u8 = local_valid.astype(np.uint8, copy=False)

            lo = r0 - h0
            hi = lo + (r1 - r0)
            core_valid = local_valid_u8[lo:hi]
            valid_mm[r0:r1] = core_valid
            nvalid += int(np.count_nonzero(core_valid))

            q = np.empty(data.shape, dtype=np.int32)
            quantize_dem(data, local_valid_u8, q, int(hydro_scale))
            hydro_mm[r0:r1] = q[lo:hi]

            s = np.empty(data.shape, dtype=np.float32)
            c = np.empty(data.shape, dtype=np.float32)
            g = np.empty(data.shape, dtype=np.uint8)
            aspect_sincos(
                data,
                local_valid_u8,
                float(meta.xres),
                float(meta.yres),
                s,
                c,
                g,
            )
            sin_mm[r0:r1] = s[lo:hi]
            cos_mm[r0:r1] = c[lo:hi]
            good_mm[r0:r1] = g[lo:hi]

    valid_mm.flush()
    hydro_mm.flush()
    sin_mm.flush()
    cos_mm.flush()
    good_mm.flush()
    store.close_many(valid_mm, hydro_mm, sin_mm, cos_mm, good_mm)

    return int(nvalid)

def write_raster_blockwise(
    path: str | Path,
    array: np.ndarray,
    meta: RasterMeta,
    valid: np.ndarray,
    *,
    dtype: str | np.dtype = "int32",
    nodata: float | int = 0,
    block_rows: int = 2048,
) -> None:
    """Write a large ndarray/memmap without creating a full in-memory copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    rows, cols = meta.shape

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": dtype.name,
        "crs": meta.crs,
        "transform": meta.transform,
        "nodata": nodata,
        "compress": "zstd",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(path, "w", **profile) as ds:
        for r0 in range(0, rows, block_rows):
            r1 = min(rows, r0 + block_rows)
            out = np.asarray(array[r0:r1], dtype=dtype).copy()
            vm = np.asarray(valid[r0:r1], dtype=bool)
            out[~vm] = np.asarray(nodata, dtype=dtype)
            ds.write(out, 1, window=Window(0, r0, cols, r1 - r0))


def read_meta(path: str | Path) -> RasterMeta:
    path = Path(path)
    with rasterio.open(path) as ds:
        return RasterMeta(
            shape=(ds.height, ds.width),
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
        )
