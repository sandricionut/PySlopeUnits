from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def read_dem(path: str | Path) -> RasterGrid:
    path = Path(path)
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype(np.float64)
        data = np.asarray(arr.filled(np.nan), dtype=np.float64)
        valid = ~np.asarray(arr.mask, dtype=bool)
        valid &= np.isfinite(data)
        return RasterGrid(
            data=data,
            valid=valid,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
        )


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
