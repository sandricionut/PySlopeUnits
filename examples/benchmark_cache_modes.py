"""Benchmark cold preparation and repeated warm graph creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyslopeunits import CandidateHierarchyPrecomputer, SlopeUnits


def create(dem: Path, work: Path, out: Path, cvmin: float, suffix: str):
    model = SlopeUnits(
        threshold_m2=250_000,
        min_area_m2=100_000,
        cv_min=cvmin,
        reduction_factor=2,
        max_iterations=12,
        convergence=5,
        workers=8,
        numba_threads=8,
    )
    return model.run(dem, out / f"benchmark_{suffix}.tif", work_dir=work)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dem", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("work/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    prep = CandidateHierarchyPrecomputer(
        threshold_m2=250_000,
        reduction_factor=2,
        max_iterations=12,
        convergence=5,
        workers=8,
        numba_threads=8,
    ).run(args.dem, args.work_dir)

    r1 = create(args.dem, args.work_dir, args.output_dir, 0.25, "cv025")
    r2 = create(args.dem, args.work_dir, args.output_dir, 0.20, "cv020")

    report = {
        "prepare_seconds": prep.total_seconds,
        "create_cv025_seconds": r1.total_seconds,
        "create_cv020_seconds": r2.total_seconds,
    }
    path = args.output_dir / "benchmark_cache_modes.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
