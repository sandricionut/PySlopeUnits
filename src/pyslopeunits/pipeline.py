from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import shutil

import numpy as np

from .candidate_cache import MultiThresholdCandidateCache
from .dag import CandidateDAG
from .graph_create import create_from_dag
from .lazy_graph import create_lazy_hierarchy
from .stream_candidates import StreamingCandidateProvider
from .engine import SlopeUnits as HydrologyEngine
from .schedule import threshold_schedule
from .memmap_store import MemmapStore
from .precompute import CandidateHierarchyPrecomputer
from .raster import read_meta, write_raster_blockwise, normalize_nodata_values
from .vector_export import export_geopackage
from .diagnostics import write_boundary_raster, write_hashed_display_raster
from .clean import clean_slope_units


@dataclass(frozen=True)
class SlopeUnitsResult:
    output: Path
    geopackage: Path | None
    work_dir: Path
    valid_cells: int
    final_units: int
    dag_nodes: int
    prepare_seconds: float
    dag_seconds: float
    create_seconds: float
    total_seconds: float


class SlopeUnits:
    """PySlopeUnits hierarchical graph-based engine.

    Pipeline:
      hydrology -> stream-only multi-threshold candidates -> nested intersection
      DAG -> graph cut -> last-halfbasin gap fill -> GRASS-like 4-neighbour
      clump -> GeoTIFF -> GeoPackage.
    """

    def __init__(
        self,
        *,
        threshold_m2: float = 250_000.0,
        min_area_m2: float = 100_000.0,
        cv_min: float = 0.25,
        reduction_factor: int = 2,
        max_iterations: int = 12,
        convergence: int = 5,
        max_area_m2: float | None = None,
        workers: int = 8,
        numba_threads: int = 8,
        mfd_block_cells: int = 1_000_000,
        export_vector: bool = True,
        export_diagnostics: bool = True,
        clean_size_m2: float | None = 25_000.0,
        clean_method: str = "grass_quick",
        preserve_raw_raster: bool = True,
        hierarchy_mode: str = "lazy",
        candidate_cache_mode: str = "auto",

        # Backward-compatible research flags. The current engine reuses compatible
        # hydrology/candidate caches internally, so these are accepted to
        # prevent legacy runners from failing.
        reuse_hydrology: bool = True,
        reuse_candidates: bool = True,
        keep_work: bool = True,
        resume_partial_hydrology: bool = True,
        hydro_scale: int = 1000,
        nodata_values=None,
        memory_budget_bytes: int | None = None,
        scratch_ram_fraction: float = 0.50,

        verbose: bool = True,
    ):
        self.threshold_m2 = float(threshold_m2)
        self.min_area_m2 = float(min_area_m2)
        self.cv_min = float(cv_min)
        self.reduction_factor = int(reduction_factor)
        self.max_iterations = int(max_iterations)
        self.convergence = int(convergence)
        self.max_area_m2 = None if max_area_m2 is None else float(max_area_m2)
        self.workers = int(workers)
        self.numba_threads = int(numba_threads)
        self.mfd_block_cells = int(mfd_block_cells)
        self.export_vector = bool(export_vector)
        self.export_diagnostics = bool(export_diagnostics)
        self.clean_size_m2 = (
            None if clean_size_m2 is None else float(clean_size_m2)
        )
        self.clean_method = str(clean_method)
        self.preserve_raw_raster = bool(preserve_raw_raster)
        self.hierarchy_mode = str(hierarchy_mode).strip().lower()
        if self.hierarchy_mode not in {"lazy", "materialized"}:
            raise ValueError("hierarchy_mode must be 'lazy' or 'materialized'")
        self.candidate_cache_mode = str(candidate_cache_mode).strip().lower()
        if self.candidate_cache_mode not in {"auto", "all", "stream"}:
            raise ValueError("candidate_cache_mode must be 'auto', 'all' or 'stream'")

        # Retained for API compatibility and run-report transparency.
        self.reuse_hydrology = bool(reuse_hydrology)
        self.reuse_candidates = bool(reuse_candidates)
        self.keep_work = bool(keep_work)
        self.resume_partial_hydrology = bool(resume_partial_hydrology)
        self.hydro_scale = int(hydro_scale)
        self.nodata_values = normalize_nodata_values(nodata_values)
        self.memory_budget_bytes = (
            None if memory_budget_bytes is None else max(0, int(memory_budget_bytes))
        )
        self.scratch_ram_fraction = float(scratch_ram_fraction)

        self.verbose = bool(verbose)

    def run(self, dem_path, output_path, *, work_dir, vector_path=None, vector_layer="slope_units"):
        t0 = time.perf_counter()
        dem_path = Path(dem_path)
        output_path = Path(output_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta = read_meta(dem_path)
        store = MemmapStore(
            work_dir / "memmap",
            ram_budget_bytes=self.memory_budget_bytes,
            ram_fraction=self.scratch_ram_fraction,
            verbose=self.verbose,
        )

        # 1. Hydrology + candidate storage strategy.
        #
        # For massive DEMs, keeping one dense half-basin raster per threshold
        # can dominate temporary disk.  ``stream`` generates one candidate level
        # at a time.  ``all`` keeps the reusable cache.  ``auto`` chooses from
        # an explicit disk estimate without changing candidate geometry.
        schedule = threshold_schedule(
            self.threshold_m2, meta.cell_area, self.reduction_factor, self.max_iterations
        )
        estimated_candidate_bytes = (
            int(meta.shape[0]) * int(meta.shape[1]) * 4 * len(schedule)
        )
        free_disk = int(shutil.disk_usage(work_dir).free)
        if self.hierarchy_mode == "materialized":
            effective_cache_mode = "all"
        elif self.candidate_cache_mode == "auto":
            effective_cache_mode = (
                "all"
                if estimated_candidate_bytes <= min(64 * 1024**3, int(free_disk * 0.20))
                else "stream"
            )
        else:
            effective_cache_mode = self.candidate_cache_mode

        if self.verbose:
            print(
                f"[PySlopeUnits] candidate storage | mode={effective_cache_mode} | "
                f"all-level estimate={estimated_candidate_bytes / 1024**3:.1f} GB | "
                f"free-disk={free_disk / 1024**3:.1f} GB"
            )

        p0 = time.perf_counter()
        cache = MultiThresholdCandidateCache(work_dir)
        if effective_cache_mode == "all":
            CandidateHierarchyPrecomputer(
                threshold_m2=self.threshold_m2,
                reduction_factor=self.reduction_factor,
                max_iterations=self.max_iterations,
                convergence=self.convergence,
                workers=self.workers,
                numba_threads=self.numba_threads,
                mfd_block_cells=self.mfd_block_cells,
                nodata_values=self.nodata_values,
                memory_budget_bytes=self.memory_budget_bytes,
                scratch_ram_fraction=self.scratch_ram_fraction,
                verbose=self.verbose,
            ).run(dem_path, work_dir)
        else:
            hydro = HydrologyEngine(
                threshold_m2=self.threshold_m2,
                min_area_m2=self.min_area_m2,
                cv_min=self.cv_min,
                reduction_factor=self.reduction_factor,
                max_iterations=self.max_iterations,
                convergence=self.convergence,
                workers=self.workers,
                numba_threads=self.numba_threads,
                hydro_scale=self.hydro_scale,
                mfd_block_cells=self.mfd_block_cells,
                resume_partial_hydrology=True,
                reuse_hydrology=True,
                reuse_candidates=False,
                keep_work=True,
                nodata_values=self.nodata_values,
                memory_budget_bytes=self.memory_budget_bytes,
                scratch_ram_fraction=self.scratch_ram_fraction,
                verbose=self.verbose,
            )
            hydro._prepare_hydrology(dem_path, store, meta)
        prepare_seconds = time.perf_counter() - p0

        # 2-3. Hierarchical graph evaluation.
        #
        # ``lazy`` is the scalable default: only the active graph frontier is
        # materialized and child intersections are externally reduced in bounded
        # disk buckets. ``materialized`` retains the original full-DAG path for
        # regression tests and research workflows that explicitly need a reusable
        # DAG cache.
        if self.hierarchy_mode == "materialized":
            d0 = time.perf_counter()
            dag = CandidateDAG(work_dir)
            dag_result = dag.build(
                store,
                cache,
                threshold_m2=self.threshold_m2,
                cell_area_m2=meta.cell_area,
                reduction_factor=self.reduction_factor,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )
            dag_seconds = time.perf_counter() - d0

            c0 = time.perf_counter()
            create_result = create_from_dag(
                store,
                cache,
                dag,
                threshold_m2=self.threshold_m2,
                cell_area_m2=meta.cell_area,
                min_area_m2=self.min_area_m2,
                cv_min=self.cv_min,
                reduction_factor=self.reduction_factor,
                max_iterations=self.max_iterations,
                max_area_m2=self.max_area_m2,
                verbose=self.verbose,
            )
            create_seconds = time.perf_counter() - c0
            hierarchy_nodes = int(dag_result.nodes)
        else:
            dag_seconds = 0.0
            c0 = time.perf_counter()
            if effective_cache_mode == "stream":
                with StreamingCandidateProvider(
                    store, workers=self.workers, verbose=self.verbose
                ) as provider:
                    create_result = create_lazy_hierarchy(
                        store,
                        None,
                        candidate_provider=provider,
                        threshold_m2=self.threshold_m2,
                        cell_area_m2=meta.cell_area,
                        min_area_m2=self.min_area_m2,
                        cv_min=self.cv_min,
                        reduction_factor=self.reduction_factor,
                        max_iterations=self.max_iterations,
                        max_area_m2=self.max_area_m2,
                        verbose=self.verbose,
                    )
            else:
                create_result = create_lazy_hierarchy(
                    store,
                    cache,
                    threshold_m2=self.threshold_m2,
                    cell_area_m2=meta.cell_area,
                    min_area_m2=self.min_area_m2,
                    cv_min=self.cv_min,
                    reduction_factor=self.reduction_factor,
                    max_iterations=self.max_iterations,
                    max_area_m2=self.max_area_m2,
                    verbose=self.verbose,
                )
            create_seconds = time.perf_counter() - c0
            hierarchy_nodes = int(create_result.visited_nodes)

        valid = store.open("valid", "r")
        final = store.open("final", "r")
        valid_cells = int(np.count_nonzero(valid))

        raw_path = output_path
        if self.clean_size_m2 is not None and self.clean_size_m2 > 0:
            if self.preserve_raw_raster:
                raw_path = output_path.with_name(
                    output_path.stem + "_raw" + output_path.suffix
                )

        if self.verbose:
            print(f"[PySlopeUnits] writing raw GeoTIFF: {raw_path}")
        write_raster_blockwise(
            raw_path, final, meta, valid,
            dtype="int32", nodata=0
        )

        final_output = raw_path
        final_units_for_result = create_result.final_units

        if self.clean_size_m2 is not None and self.clean_size_m2 > 0:
            clean_result = clean_slope_units(
                raw_path,
                output_path,
                work_dir=work_dir / "clean",
                clean_size_m2=self.clean_size_m2,
                method=self.clean_method,
                reclump=True,
                memory_budget_bytes=self.memory_budget_bytes,
                scratch_ram_fraction=self.scratch_ram_fraction,
                verbose=self.verbose,
            )
            final_output = output_path
            final_units_for_result = clean_result.final_units

        gpkg = None
        if self.export_vector:
            if vector_path is None:
                vector_path = final_output.with_suffix(".gpkg")
            gpkg = Path(vector_path)
            export_geopackage(
                final_output,
                gpkg,
                layer_name=vector_layer,
                backend="auto",
                overwrite=True,
                connectivity=4,
                verbose=self.verbose,
            )

        if self.export_diagnostics:
            write_boundary_raster(
                final_output,
                final_output.with_name(
                    final_output.stem + "_boundaries.tif"
                )
            )
            write_hashed_display_raster(
                final_output,
                final_output.with_name(
                    final_output.stem + "_display.tif"
                )
            )

        total = time.perf_counter() - t0
        result = SlopeUnitsResult(
            output=final_output,
            geopackage=gpkg,
            work_dir=work_dir,
            valid_cells=valid_cells,
            final_units=final_units_for_result,
            dag_nodes=hierarchy_nodes,
            prepare_seconds=prepare_seconds,
            dag_seconds=dag_seconds,
            create_seconds=create_seconds,
            total_seconds=total,
        )
        (work_dir / "run_report.json").write_text(
            json.dumps({
                **asdict(result),
                "output": str(result.output),
                "geopackage": None if result.geopackage is None else str(result.geopackage),
                "work_dir": str(result.work_dir),
                "parameters": {
                    "threshold_m2": self.threshold_m2,
                    "min_area_m2": self.min_area_m2,
                    "cv_min": self.cv_min,
                    "reduction_factor": self.reduction_factor,
                    "max_iterations": self.max_iterations,
                    "convergence": self.convergence,
                    "max_area_m2": self.max_area_m2,
                    "clean_size_m2": self.clean_size_m2,
                    "clean_method": self.clean_method,
                    "preserve_raw_raster": self.preserve_raw_raster,
                    "hierarchy_mode": self.hierarchy_mode,
                    "candidate_cache_mode": self.candidate_cache_mode,
                    "effective_candidate_cache_mode": effective_cache_mode,
                    "estimated_candidate_cache_bytes": int(estimated_candidate_bytes),
                    "reuse_hydrology": self.reuse_hydrology,
                    "reuse_candidates": self.reuse_candidates,
                    "keep_work": self.keep_work,
                    "resume_partial_hydrology": self.resume_partial_hydrology,
                    "hydro_scale": self.hydro_scale,
                    "memory_budget_bytes": self.memory_budget_bytes,
                    "scratch_ram_fraction": self.scratch_ram_fraction,
                    "nodata_values": [
                        "nan" if np.isnan(v) else float(v)
                        for v in self.nodata_values
                    ],
                },
            }, indent=2), encoding="utf-8"
        )
        store.write_memory_report("memory_allocation_graph.json")
        if self.verbose:
            print(f"[PySlopeUnits] final slope units: {result.final_units:,}")
            print(f"[PySlopeUnits] total time: {total/60:.2f} min")
        return result

