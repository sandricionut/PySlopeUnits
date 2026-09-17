from __future__ import annotations

from .logging_utils import log as print

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

import numpy as np

from .engine import SlopeUnits
from .memmap_store import MemmapStore
from .metrics_v2 import compute_v2_metrics
from .precompute import CandidateHierarchyPrecomputer


@dataclass(frozen=True)
class OptimizationPoint:
    cvmin: float
    areamin_m2: float
    V: float
    I: float
    F: float | None
    units: int
    output: str


@dataclass(frozen=True)
class OptimizationResult:
    best_cvmin: float
    best_areamin_m2: float
    best_F: float
    evaluations: list[OptimizationPoint]


def _five_points(cv_bounds, area_bounds):
    c0, c1 = map(float, cv_bounds)
    a0, a1 = map(float, area_bounds)
    cm = 0.5 * (c0 + c1)
    am = 0.5 * (a0 + a1)
    return [
        (c0, a0), (c0, a1), (c1, a0), (c1, a1), (cm, am)
    ]


class V2Optimizer:
    """Adaptive five-point optimizer inspired by r.slopeunits v2.0.

    The expensive hydrological/half-basin candidate hierarchy is reused across
    all parameter evaluations. ``F`` is normalized from all evaluations
    available at the current step, following the V/I objective concept in the
    GRASS optimizer.

    This implementation intentionally keeps the search strategy transparent
    and deterministic; it is not claimed to be line-for-line identical to the
    current GRASS limit-update cases.
    """

    def __init__(
        self,
        *,
        cv_bounds=(0.05, 0.25),
        area_bounds_m2=(50_000.0, 200_000.0),
        epsilon_cv=0.01,
        epsilon_area_m2=50_000.0,
        max_rounds=8,
        threshold_m2=250_000.0,
        reduction_factor=2,
        max_iterations=12,
        convergence=5,
        workers=8,
        numba_threads=8,
        candidate_cache=True,
        precompute_candidates=True,
        verbose=True,
    ):
        self.cv_bounds = tuple(map(float, cv_bounds))
        self.area_bounds = tuple(map(float, area_bounds_m2))
        self.epsilon_cv = float(epsilon_cv)
        self.epsilon_area = float(epsilon_area_m2)
        self.max_rounds = int(max_rounds)
        self.threshold_m2 = float(threshold_m2)
        self.reduction_factor = int(reduction_factor)
        self.max_iterations = int(max_iterations)
        self.convergence = int(convergence)
        self.workers = int(workers)
        self.numba_threads = int(numba_threads)
        self.candidate_cache = bool(candidate_cache)
        self.precompute_candidates = bool(precompute_candidates)
        self.verbose = bool(verbose)

    @staticmethod
    def _normalize_F(points):
        V = np.array([p["V"] for p in points], dtype=float)
        I = np.array([p["I"] for p in points], dtype=float)
        vmin, vmax = np.nanmin(V), np.nanmax(V)
        imin, imax = np.nanmin(I), np.nanmax(I)

        vden = vmax - vmin
        iden = imax - imin

        for p in points:
            fv = 0.0 if vden == 0 else (vmax - p["V"]) / vden
            fi = 0.0 if iden == 0 else (imax - p["I"]) / iden
            p["F"] = float(fv + fi)

    def run(self, dem_path, *, work_dir, output_dir) -> OptimizationResult:
        dem_path = Path(dem_path)
        work_dir = Path(work_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.precompute_candidates:
            if self.verbose:
                print("[PySlopeUnits optimize] preparing reusable candidate hierarchy")
            CandidateHierarchyPrecomputer(
                threshold_m2=self.threshold_m2,
                reduction_factor=self.reduction_factor,
                max_iterations=self.max_iterations,
                convergence=self.convergence,
                workers=self.workers,
                numba_threads=self.numba_threads,
                verbose=self.verbose,
            ).run(dem_path, work_dir)

        evaluated: dict[tuple[float, float], dict] = {}
        cv_bounds = list(self.cv_bounds)
        area_bounds = list(self.area_bounds)

        for round_no in range(1, self.max_rounds + 1):
            points = _five_points(cv_bounds, area_bounds)

            if self.verbose:
                print(
                    f"[PySlopeUnits optimize] round={round_no} "
                    f"cv={cv_bounds} area={area_bounds}"
                )

            for cvmin, areamin in points:
                key = (round(cvmin, 12), round(areamin, 6))
                if key in evaluated:
                    continue

                stem = f"su_cv{cvmin:.6f}_a{areamin:.0f}".replace(".", "p")
                out = output_dir / f"{stem}.tif"

                model = SlopeUnits(
                    threshold_m2=self.threshold_m2,
                    min_area_m2=areamin,
                    cv_min=cvmin,
                    reduction_factor=self.reduction_factor,
                    max_iterations=self.max_iterations,
                    convergence=self.convergence,
                    workers=self.workers,
                    numba_threads=self.numba_threads,
                    reuse_hydrology=True,
                    reuse_candidates=self.candidate_cache,
                    keep_work=True,
                    verbose=self.verbose,
                )
                result = model.run(
                    dem_path=dem_path,
                    output_path=out,
                    work_dir=work_dir,
                )

                store = MemmapStore(work_dir / "memmap")
                metrics = compute_v2_metrics(
                    store.open("final", "r"),
                    store.open("valid", "r"),
                    store.open("sin_aspect", "r"),
                    store.open("cos_aspect", "r"),
                )

                evaluated[key] = {
                    "cvmin": cvmin,
                    "areamin_m2": areamin,
                    "V": metrics.V,
                    "I": metrics.I,
                    "F": None,
                    "units": metrics.units,
                    "output": str(result.output),
                }

            all_points = list(evaluated.values())
            self._normalize_F(all_points)

            current = [evaluated[(round(c, 12), round(a, 6))] for c, a in points]
            current.sort(key=lambda p: p["F"], reverse=True)
            best = current[0]

            if self.verbose:
                print(
                    f"[PySlopeUnits optimize] best round point: "
                    f"cv={best['cvmin']:.6f} area={best['areamin_m2']:.0f} "
                    f"V={best['V']:.6f} I={best['I']:.6f} F={best['F']:.6f}"
                )

            if (
                (cv_bounds[1] - cv_bounds[0]) <= self.epsilon_cv
                and (area_bounds[1] - area_bounds[0]) <= self.epsilon_area
            ):
                break

            # Deterministic contraction around the best point.
            cspan = (cv_bounds[1] - cv_bounds[0]) / 3.0
            aspan = (area_bounds[1] - area_bounds[0]) / 3.0

            cv_bounds = [
                max(self.cv_bounds[0], best["cvmin"] - cspan),
                min(self.cv_bounds[1], best["cvmin"] + cspan),
            ]
            area_bounds = [
                max(self.area_bounds[0], best["areamin_m2"] - aspan),
                min(self.area_bounds[1], best["areamin_m2"] + aspan),
            ]

        all_points = list(evaluated.values())
        self._normalize_F(all_points)
        all_points.sort(key=lambda p: p["F"], reverse=True)

        result = OptimizationResult(
            best_cvmin=float(all_points[0]["cvmin"]),
            best_areamin_m2=float(all_points[0]["areamin_m2"]),
            best_F=float(all_points[0]["F"]),
            evaluations=[
                OptimizationPoint(**p) for p in all_points
            ],
        )

        (output_dir / "optimization_report.json").write_text(
            json.dumps(
                {
                    "best_cvmin": result.best_cvmin,
                    "best_areamin_m2": result.best_areamin_m2,
                    "best_F": result.best_F,
                    "evaluations": [asdict(p) for p in result.evaluations],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return result
