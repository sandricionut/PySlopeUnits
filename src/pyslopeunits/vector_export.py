from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import rasterio
from rasterio.features import shapes


def _remove_existing_gpkg(path: Path, retries: int = 8) -> None:
    if not path.exists():
        return
    for _ in range(retries):
        try:
            path.unlink()
            return
        except PermissionError:
            time.sleep(0.25)
    raise PermissionError(
        f"Cannot overwrite {path}. Close the GeoPackage in any application "
        "that may be using it and rerun the export."
    )


@dataclass(frozen=True)
class VectorExportResult:
    geopackage: Path
    layer: str
    features: int
    backend: str


def _ring_signed_area(coords) -> float:
    if len(coords) < 4:
        return 0.0
    s = 0.0
    x0, y0 = coords[-1][0], coords[-1][1]
    for pt in coords:
        x1, y1 = pt[0], pt[1]
        s += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return 0.5 * s


def _geometry_area(geom: dict) -> float:
    geom_type = geom.get("type")
    coords = geom.get("coordinates", [])

    if geom_type == "Polygon":
        if not coords:
            return 0.0
        area = abs(_ring_signed_area(coords[0]))
        for hole in coords[1:]:
            area -= abs(_ring_signed_area(hole))
        return max(0.0, area)

    if geom_type == "MultiPolygon":
        total = 0.0
        for polygon in coords:
            if not polygon:
                continue
            area = abs(_ring_signed_area(polygon[0]))
            for hole in polygon[1:]:
                area -= abs(_ring_signed_area(hole))
            total += max(0.0, area)
        return total

    return 0.0


def export_geopackage(
    raster_path,
    gpkg_path,
    *,
    layer_name: str = "slope_units",
    backend: str = "auto",
    overwrite: bool = True,
    connectivity: int = 4,
    verbose: bool = True,
) -> VectorExportResult:
    """Export a labelled slope-unit raster to GeoPackage.

    The implementation is fully open source and uses Fiona + Rasterio.
    ``backend`` is retained for API compatibility; accepted values are
    ``"auto"`` and ``"fiona"``.
    """
    backend = str(backend).lower()
    if backend not in {"auto", "fiona"}:
        raise ValueError("Only the open-source Fiona backend is supported.")

    try:
        import fiona
    except ImportError as exc:
        raise RuntimeError(
            "GeoPackage export requires Fiona. Install it with 'pip install fiona'."
        ) from exc

    raster_path = Path(raster_path)
    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    if gpkg_path.exists() and overwrite:
        _remove_existing_gpkg(gpkg_path)

    schema = {
        "geometry": "Polygon",
        "properties": {
            "su_id": "int64",
            "raster_id": "int64",
            "area_map": "float",
        },
    }

    with rasterio.open(raster_path) as src:
        crs_wkt = src.crs.to_wkt() if src.crs else None
        with fiona.open(
            gpkg_path,
            mode="w",
            driver="GPKG",
            layer=layer_name,
            schema=schema,
            crs_wkt=crs_wkt,
        ) as sink:
            count = 0
            for geom, value in shapes(
                rasterio.band(src, 1),
                mask=None,
                connectivity=int(connectivity),
                transform=src.transform,
            ):
                raster_id = int(value)
                if raster_id <= 0:
                    continue

                count += 1
                sink.write(
                    {
                        "geometry": geom,
                        "properties": {
                            "su_id": count,
                            "raster_id": raster_id,
                            "area_map": float(_geometry_area(geom)),
                        },
                    }
                )

                if verbose and count % 50000 == 0:
                    print(f"[PySlopeUnits vector] {count:,} polygons written")

    if verbose:
        print(
            f"[PySlopeUnits vector] GeoPackage complete | "
            f"features={count:,} | {gpkg_path}"
        )

    return VectorExportResult(
        geopackage=gpkg_path,
        layer=layer_name,
        features=count,
        backend="fiona+rasterio",
    )
