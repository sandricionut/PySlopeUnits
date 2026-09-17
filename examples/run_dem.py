#!/usr/bin/env python3
"""Run PySlopeUnits v0.1.8 on a single DEM.

This example uses the public AdaptiveSlopeUnits API and the same scientific
parameterization used in the manuscript benchmark. Resource settings remain
user-configurable so the same example can be used on a workstation or a larger
compute node.

Example
-------
python examples/run_dem.py dem.tif run \
    --target-crs EPSG:3035 \
    --resolution 30 \
    --memory-gb 16 \
    --workers 8 \
    --numba-threads 8 \
    --no-vector
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyslopeunits import AdaptiveSlopeUnits, __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example PySlopeUnits v0.1.8 run on one DEM."
    )
    parser.add_argument("dem", type=Path, help="Input DEM, mosaic or VRT")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--target-crs",
        default=None,
        help="Projected metric target CRS, e.g. EPSG:3035",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Processing resolution in metres; omit to retain a compatible native grid",
    )
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--numba-threads", type=int, default=None)
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable GeoPackage export",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable diagnostic rasters",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore compatible partial checkpoints and recompute long stages",
    )
    args = parser.parse_args()

    print(f"PySlopeUnits {__version__}")

    model = AdaptiveSlopeUnits(
        target_crs=args.target_crs,
        fine_resolution_m=args.resolution,
        memory_limit_gb=args.memory_gb,
        workers=args.workers,
        numba_threads=args.numba_threads,
        checkpoint=True,
        checkpoint_minutes=15.0,
        restart=args.restart,
        # Manuscript/reference scientific parameters
        threshold_m2=250_000.0,
        min_area_m2=100_000.0,
        cv_min=0.25,
        reduction_factor=2,
        max_iterations=12,
        convergence=5,
        clean_size_m2=25_000.0,
        # Scalable defaults
        hierarchy_mode="lazy",
        candidate_cache_mode="auto",
        export_vector=not args.no_vector,
        export_diagnostics=not args.no_diagnostics,
    )

    result = model.run(
        args.dem.expanduser(),
        args.output_dir.expanduser(),
    )
    print(result)


if __name__ == "__main__":
    main()
