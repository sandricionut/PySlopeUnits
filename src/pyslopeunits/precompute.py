from __future__ import annotations

from .logging_utils import log as print

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import multiprocessing as mp
import time

import numpy as np

from .candidate_cache import CandidateKey, MultiThresholdCandidateCache
from .engine import SlopeUnits
from .halfbasin_parallel import build_half_basins
from .kernels import stream_mask_grasslike_static
from .memmap_store import MemmapStore
from .raster import read_meta
from .schedule import threshold_schedule


@dataclass(frozen=True)
class CandidateLevelResult:
    level: int
    threshold_cells: int
    threshold_m2: float
    cache_hit: bool
    stream_cells: int
    branch_roots: int
    residual_components: int
    residual_mode: str
    seconds: float


@dataclass(frozen=True)
class CandidatePreparationResult:
    dem: str
    work_dir: str
    cell_area_m2: float
    levels: list[CandidateLevelResult]
    total_seconds: float

    @property
    def cache_hits(self) -> int:
        return sum(int(x.cache_hit) for x in self.levels)


class CandidateHierarchyPrecomputer:
    """
    Build the complete multi-threshold half-basin hierarchy once per DEM.

    This is the key optimization for parameter search: `areamin` and `cvmin`
    do not affect candidate geometry, therefore all subsequent create/optimize
    runs reuse these levels.
    """

    def __init__(
        self,
        *,
        threshold_m2: float = 250_000.0,
        reduction_factor: int = 2,
        max_iterations: int = 12,
        convergence: int = 5,
        workers: int = 8,
        numba_threads: int = 8,
        hydro_scale: int = 1000,
        mfd_block_cells: int = 1_000_000,
        nodata_values=None,
        memory_budget_bytes: int | None = None,
        scratch_ram_fraction: float = 0.50,
        hydrology_domain_raster: str | Path | None = None,
        checkpoint: bool = True,
        checkpoint_minutes: float = 15.0,
        restart: bool = False,
        verbose: bool = True,
    ):
        self.threshold_m2 = float(threshold_m2)
        self.reduction_factor = int(reduction_factor)
        self.max_iterations = int(max_iterations)
        self.convergence = int(convergence)
        self.workers = max(1, int(workers))
        self.numba_threads = max(1, int(numba_threads))
        self.hydro_scale = int(hydro_scale)
        self.mfd_block_cells = max(10_000, int(mfd_block_cells))
        self.nodata_values = None if nodata_values is None else tuple(float(x) for x in nodata_values)
        self.memory_budget_bytes = (
            None if memory_budget_bytes is None else max(0, int(memory_budget_bytes))
        )
        self.scratch_ram_fraction = float(scratch_ram_fraction)
        self.hydrology_domain_raster = (
            None if hydrology_domain_raster is None else Path(hydrology_domain_raster)
        )
        self.checkpoint = bool(checkpoint)
        self.checkpoint_minutes = float(checkpoint_minutes)
        self.restart = bool(restart)
        self.verbose = bool(verbose)

    def run(self, dem_path, work_dir) -> CandidatePreparationResult:
        t0 = time.perf_counter()
        dem_path = Path(dem_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        meta = read_meta(dem_path)
        store = MemmapStore(
            work_dir / "memmap",
            ram_budget_bytes=self.memory_budget_bytes,
            ram_fraction=self.scratch_ram_fraction,
            verbose=self.verbose,
        )
        cache = MultiThresholdCandidateCache(work_dir)

        # Reuse the same tested hydrology implementation.
        model = SlopeUnits(
            threshold_m2=self.threshold_m2,
            min_area_m2=100_000.0,
            cv_min=0.25,
            reduction_factor=self.reduction_factor,
            max_iterations=self.max_iterations,
            convergence=self.convergence,
            workers=self.workers,
            numba_threads=self.numba_threads,
            hydro_scale=self.hydro_scale,
            mfd_block_cells=self.mfd_block_cells,
            resume_partial_hydrology=True,
            reuse_hydrology=True,
            reuse_candidates=True,
            keep_work=True,
            nodata_values=self.nodata_values,
            memory_budget_bytes=self.memory_budget_bytes,
            scratch_ram_fraction=self.scratch_ram_fraction,
            hydrology_domain_raster=self.hydrology_domain_raster,
            checkpoint=self.checkpoint,
            checkpoint_minutes=self.checkpoint_minutes,
            restart=self.restart,
            verbose=self.verbose,
        )
        model._prepare_hydrology(dem_path, store, meta)

        valid = store.open("valid", "r")
        accumulation = store.open("accumulation", "r")
        receiver = store.open("receiver", "r")
        order = store.open("order", "r")
        static_ok = store.open("stream_static_ok", "r")
        up_acc = store.open("stream_up_acc_max", "r")

        stream = store.create("stream", meta.shape, np.uint8, fill=0)

        schedule = threshold_schedule(
            self.threshold_m2,
            meta.cell_area,
            self.reduction_factor,
            self.max_iterations,
        )

        pool = None
        if self.workers > 1:
            pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
            )

        results: list[CandidateLevelResult] = []

        try:
            for level in schedule:
                lt0 = time.perf_counter()
                key = CandidateKey(level.cells)

                if cache.ensure(key, verbose=self.verbose):
                    stats = cache.load_stats(key)
                    result = CandidateLevelResult(
                        level=level.level,
                        threshold_cells=level.cells,
                        threshold_m2=level.area_m2,
                        cache_hit=True,
                        stream_cells=int(stats.get("stream_cells", 0)),
                        branch_roots=int(stats.get("stream_branch_roots", 0)),
                        residual_components=int(stats.get("residual_outlets", 0)),
                        residual_mode=str(stats.get("residual_mode", "cached")),
                        seconds=time.perf_counter() - lt0,
                    )
                    results.append(result)
                    if self.verbose:
                        print(
                            f"[PySlopeUnits prepare] level={level.level:02d} "
                            f"{level.cells:,} cells | candidate cache ready"
                        )
                    continue

                if self.verbose:
                    print(
                        f"[PySlopeUnits prepare] level={level.level:02d} | "
                        f"threshold={level.area_m2:,.1f} m2 "
                        f"({level.cells:,} cells)"
                    )

                stream_mask_grasslike_static(
                    valid,
                    accumulation,
                    receiver,
                    order,
                    float(level.cells),
                    static_ok,
                    up_acc,
                    stream,
                )
                stream.flush()
                stream_cells = int(np.count_nonzero(stream))

                cache.prepare_for_build(store)
                stats = build_half_basins(
                    store,
                    workers=self.workers,
                    executor=pool,
                    verbose=self.verbose,
                )
                stats["stream_cells"] = stream_cells
                stats["threshold_cells"] = level.cells
                stats["threshold_m2"] = level.area_m2

                cache.save_from_working(key, store, stats)

                result = CandidateLevelResult(
                    level=level.level,
                    threshold_cells=level.cells,
                    threshold_m2=level.area_m2,
                    cache_hit=False,
                    stream_cells=stream_cells,
                    branch_roots=int(stats["stream_branch_roots"]),
                    residual_components=int(stats["residual_outlets"]),
                    residual_mode=str(stats.get("residual_mode", "unknown")),
                    seconds=time.perf_counter() - lt0,
                )
                results.append(result)

                if self.verbose:
                    print(
                        f"[PySlopeUnits prepare] cached level {level.level:02d} | "
                        f"streams={stream_cells:,} | "
                        f"branches={result.branch_roots:,} | "
                        f"{result.seconds:.1f}s"
                    )
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        result = CandidatePreparationResult(
            dem=str(dem_path),
            work_dir=str(work_dir),
            cell_area_m2=float(meta.cell_area),
            levels=results,
            total_seconds=time.perf_counter() - t0,
        )

        report = {
            "dem": result.dem,
            "work_dir": result.work_dir,
            "cell_area_m2": result.cell_area_m2,
            "cache_hits": result.cache_hits,
            "levels": [asdict(x) for x in result.levels],
            "total_seconds": result.total_seconds,
        }
        (work_dir / "candidate_precompute_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        store.write_memory_report("memory_allocation_hydrology.json")

        return result
