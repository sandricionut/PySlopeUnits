from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import SlopeUnits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyslopeunits",
        description="Hierarchical graph-based slope-unit delineation from DEMs.",
    )
    parser.add_argument("dem", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--threshold-m2", type=float, default=250000.0)
    parser.add_argument("--min-area-m2", type=float, default=100000.0)
    parser.add_argument("--cv-min", type=float, default=0.25)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--convergence", type=int, default=5)
    parser.add_argument("--max-area-m2", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--numba-threads", type=int, default=8)
    parser.add_argument("--mfd-block-cells", type=int, default=1_000_000)
    parser.add_argument("--no-vector", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--clean-size-m2", type=float, default=25000.0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser()
    dem = args.dem.expanduser()
    work_dir = (
        args.work_dir.expanduser()
        if args.work_dir is not None
        else output.parent / f".{output.stem}_work"
    )

    model = SlopeUnits(
        threshold_m2=args.threshold_m2,
        min_area_m2=args.min_area_m2,
        cv_min=args.cv_min,
        reduction_factor=args.reduction_factor,
        max_iterations=args.max_iterations,
        convergence=args.convergence,
        max_area_m2=args.max_area_m2,
        workers=args.workers,
        numba_threads=args.numba_threads,
        mfd_block_cells=args.mfd_block_cells,
        export_vector=not args.no_vector,
        export_diagnostics=not args.no_diagnostics,
        clean_size_m2=args.clean_size_m2,
        verbose=not args.quiet,
    )
    result = model.run(dem, output, work_dir=work_dir)
    print(result)


if __name__ == "__main__":
    main()
