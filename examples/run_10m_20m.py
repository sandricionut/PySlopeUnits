"""Run the same PySlopeUnits configuration for 20 m and 10 m DEMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyslopeunits import SlopeUnits


def run_one(name: str, dem: Path, output_dir: Path, work_dir: Path):
    print("\n" + "=" * 88)
    print(f"PySlopeUnits | {name}")
    print("=" * 88)

    model = SlopeUnits(
        threshold_m2=250_000,
        min_area_m2=100_000,
        cv_min=0.25,
        reduction_factor=2,
        max_iterations=12,
        convergence=5,
        workers=8,
        numba_threads=8,
        mfd_block_cells=1_000_000,
        export_vector=True,
        export_diagnostics=True,
        clean_size_m2=25_000,
        clean_method="grass_quick",
        preserve_raw_raster=True,
        verbose=True,
    )

    return model.run(
        dem_path=dem,
        output_path=output_dir / f"pyslopeunits_{name}.tif",
        work_dir=work_dir / name,
        vector_path=output_dir / f"pyslopeunits_{name}.gpkg",
        vector_layer=f"slope_units_{name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dem_20m", type=Path)
    parser.add_argument("dem_10m", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name, dem in (("20m", args.dem_20m), ("10m", args.dem_10m)):
        if not dem.exists():
            raise FileNotFoundError(dem)
        result = run_one(name, dem, args.output_dir, args.work_dir)
        results.append(
            {
                "dataset": name,
                "raster": str(result.output),
                "geopackage": None if result.geopackage is None else str(result.geopackage),
                "valid_cells": result.valid_cells,
                "final_units": result.final_units,
                "dag_nodes": result.dag_nodes,
                "prepare_seconds": result.prepare_seconds,
                "dag_seconds": result.dag_seconds,
                "create_seconds": result.create_seconds,
                "total_seconds": result.total_seconds,
            }
        )

    report = args.output_dir / "pyslopeunits_10m_20m_summary.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary: {report}")


if __name__ == "__main__":
    main()
