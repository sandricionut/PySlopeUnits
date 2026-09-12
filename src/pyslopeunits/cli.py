from __future__ import annotations

import argparse
from pathlib import Path

from .adaptive import AdaptiveSlopeUnits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyslopeunits",
        description=(
            "Memory-adaptive hierarchical graph-based slope-unit delineation from DEMs. "
            "The same command is used for local, regional and very large datasets."
        ),
    )

    parser.add_argument("dem", help="GDAL/Rasterio-readable DEM, mosaic or VRT")
    parser.add_argument("output_dir", type=Path)

    # Canonical processing grid. If both are omitted, a projected metric DEM
    # keeps its native CRS and square pixel size.
    parser.add_argument(
        "--fine-resolution-m",
        type=float,
        default=None,
        help=(
            "Canonical fine-grid resolution in metres. If omitted, PySlopeUnits "
            "uses the native square resolution of a projected metric DEM."
        ),
    )
    parser.add_argument(
        "--target-crs",
        default=None,
        help=(
            "Projected metric target CRS, e.g. EPSG:3035. If omitted, a projected "
            "metric source DEM keeps its native CRS. Geographic/non-metric DEMs "
            "require this option."
        ),
    )

    # Adaptive execution / resource planning.
    parser.add_argument("--memory-limit-gb", type=float, default=None)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument(
        "--scratch-ram-fraction",
        type=float,
        default=0.50,
        help=(
            "Fraction of execution memory used for RAM-first temporary arrays "
            "before disk spill."
        ),
    )
    parser.add_argument("--coarse-memory-fraction", type=float, default=0.18)
    parser.add_argument("--bytes-per-fine-cell", type=int, default=112)
    parser.add_argument("--coarse-bytes-per-cell", type=int, default=72)
    parser.add_argument("--coarse-resolution-m", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto")
    parser.add_argument("--numba-threads", type=int, default=0, help="0 = auto")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")

    # Slope-unit scientific parameters.
    parser.add_argument("--threshold-m2", type=float, default=250000.0)
    parser.add_argument("--min-area-m2", type=float, default=100000.0)
    parser.add_argument("--cv-min", type=float, default=0.25)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--convergence", type=int, default=5)
    parser.add_argument("--max-area-m2", type=float, default=None)
    parser.add_argument("--mfd-block-cells", type=int, default=1_000_000)
    parser.add_argument(
        "--hierarchy-mode",
        choices=("lazy", "materialized"),
        default="lazy",
        help=(
            "Hierarchy execution mode. lazy is the scalable bounded-memory default; "
            "materialized builds the complete reusable DAG and is intended mainly "
            "for regression/benchmarking."
        ),
    )
    parser.add_argument(
        "--candidate-cache-mode",
        choices=("auto", "all", "stream"),
        default="auto",
        help=(
            "Candidate storage. auto selects from disk capacity; all keeps every "
            "threshold for reuse; stream keeps only the working threshold."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        nargs="+",
        action="extend",
        default=None,
        metavar="VALUE",
        help=(
            "Additional NoData value(s); metadata NoData is always read automatically."
        ),
    )
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--clean-size-m2", type=float, default=25000.0)
    parser.add_argument("--quiet", action="store_true")

    return parser


def _run(args) -> None:
    slopeunit_kwargs = {
        "threshold_m2": args.threshold_m2,
        "min_area_m2": args.min_area_m2,
        "cv_min": args.cv_min,
        "reduction_factor": args.reduction_factor,
        "max_iterations": args.max_iterations,
        "convergence": args.convergence,
        "max_area_m2": args.max_area_m2,
        "mfd_block_cells": args.mfd_block_cells,
        "hierarchy_mode": args.hierarchy_mode,
        "candidate_cache_mode": args.candidate_cache_mode,
        "export_vector": not args.no_vector,
        "export_diagnostics": not args.no_diagnostics,
        "clean_size_m2": args.clean_size_m2,
        "scratch_ram_fraction": args.scratch_ram_fraction,
    }

    model = AdaptiveSlopeUnits(
        fine_resolution_m=args.fine_resolution_m,
        target_crs=args.target_crs,
        memory_limit_gb=args.memory_limit_gb,
        memory_fraction=args.memory_fraction,
        coarse_memory_fraction=args.coarse_memory_fraction,
        bytes_per_fine_cell=args.bytes_per_fine_cell,
        coarse_bytes_per_cell=args.coarse_bytes_per_cell,
        coarse_resolution_m=args.coarse_resolution_m,
        workers=None if args.workers <= 0 else args.workers,
        numba_threads=None if args.numba_threads <= 0 else args.numba_threads,
        nodata_values=args.nodata,
        verbose=not args.quiet,
        **slopeunit_kwargs,
    )

    print(
        model.run(
            args.dem,
            args.output_dir.expanduser(),
            plan_only=args.plan_only,
            skip_existing=not args.no_skip_existing,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()
