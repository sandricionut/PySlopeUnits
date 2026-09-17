from __future__ import annotations

from .logging_utils import log as print

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from .halfbasin_parallel import build_half_basins
from .candidate_cache import MultiThresholdCandidateCache, CandidateKey
from .kernels import (
    astar_route_preallocated,
    mfd_accumulation,
    stream_mask_grasslike,
    precompute_stream_seed_static,
    stream_mask_grasslike_static,
    drainage_components,
    parent_bboxes,
    apply_decisions,
    apply_parent_child_decisions,
    finalize_remaining,
    count_unlabeled,
)
from .memmap_store import MemmapStore
from .parent_eval import evaluate_parent, evaluate_parent_batch, ParentDecision
from .raster import (
    read_meta, RasterMeta, write_raster_blockwise,
    normalize_nodata_values, prepare_dem_memmaps_blockwise,
)
from .schedule import threshold_schedule
from .vector_export import export_geopackage
from .mfd_parallel import run_mfd_block_parallel
from .indexing import index_dtype_for_shape, dtype_name
from .fine_hydrology import run_domain_sharded_astar


@dataclass
class IterationInfo:
    iteration: int
    threshold_m2: float
    active_cells: int
    stream_cells: int
    stream_branch_roots: int
    half_basin_label_max: int
    active_parents: int
    rejected_parents: int
    split_children: int
    finalized_children: int
    finalized_cells: int
    unresolved_cells: int
    halfbasin_worker_pids: list[int]
    parent_worker_pids: list[int]
    candidate_cache_hit: bool
    candidate_seconds: float
    parent_seconds: float
    seconds: float


@dataclass
class SlopeUnitsResult:
    output: Path
    work_dir: Path
    valid_cells: int
    final_units: int
    iterations: list[IterationInfo]
    total_seconds: float


def _manifest_for(
    dem_path: Path, meta: RasterMeta, convergence: int, scale: int,
    nodata_values=None, hydrology_domain_raster=None,
) -> dict:
    st = dem_path.stat()
    domain_path = None
    domain_stat = None
    if hydrology_domain_raster is not None:
        dp = Path(hydrology_domain_raster)
        domain_path = str(dp.resolve())
        try:
            dst = dp.stat()
            domain_stat = {"size": int(dst.st_size), "mtime_ns": int(dst.st_mtime_ns)}
        except OSError:
            domain_stat = None
    return {
        "dem": str(dem_path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "shape": list(meta.shape),
        "xres": float(meta.xres),
        "yres": float(meta.yres),
        "convergence": int(convergence),
        "hydro_scale": int(scale),
        "nodata_values": [
            "nan" if np.isnan(v) else float(v)
            for v in normalize_nodata_values(nodata_values)
        ],
        "hydrology_io_version": 4,
        "index_dtype": dtype_name(index_dtype_for_shape(meta.shape)),
        "routing_mode": (
            "domain-sharded-global-astar" if hydrology_domain_raster is not None
            else "global-astar"
        ),
        "hydrology_domain_raster": domain_path,
        "hydrology_domain_raster_stat": domain_stat,
        "engine": "PySlope NumPy+Numba multires 1.0",
    }


def _manifest_matches_dataset(got: dict, expected: dict) -> bool:
    # Engine-version changes should not invalidate expensive A*/MFD arrays.
    keys = [
        "dem", "size", "mtime_ns", "shape", "xres", "yres",
        "convergence", "hydro_scale", "nodata_values",
        "hydrology_io_version", "index_dtype", "routing_mode",
    ]
    return all(got.get(k) == expected.get(k) for k in keys)


def _hydrology_core_ok(store: MemmapStore, expected: dict) -> bool:
    manifest_path = store.root / "hydrology_manifest.json"
    needed = [
        "valid", "receiver", "accumulation", "order",
        "sin_aspect", "cos_aspect", "aspect_valid",
    ]
    if not manifest_path.exists() or not all(store.exists(n) for n in needed):
        return False
    try:
        got = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _manifest_matches_dataset(got, expected)


def _hydrology_cache_ok(store: MemmapStore, expected: dict) -> bool:
    if not _hydrology_core_ok(store, expected):
        return False
    optimized = [
        "stream_static_ok", "stream_up_acc_max", "drainage_component",
    ]
    return all(store.exists(n) for n in optimized)


def _round_robin_job_batches(jobs, workers: int):
    if not jobs:
        return []
    workers = max(1, min(int(workers), len(jobs)))
    batches = [[] for _ in range(workers)]
    # jobs already sorted largest first; round-robin balances spatial workload.
    for i, job in enumerate(jobs):
        batches[i % workers].append(job)
    return [b for b in batches if b]


def _evaluate_parents_parallel(
    store: MemmapStore,
    jobs: list[tuple[int, tuple[int, int, int, int], int]],
    *,
    workers: int,
    cell_area: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None,
    max_child_label: int,
    dense_parent_threshold_cells: int,
    executor=None,
    verbose: bool = True,
):
    if not jobs:
        return [], []

    # Sort by parent cell count so heavy windows are distributed first.
    jobs = sorted(jobs, key=lambda x: x[2], reverse=True)
    simple = jobs

    if workers <= 1 or len(simple) == 1:
        results = [
            evaluate_parent(
                str(store.root), pid, bbox, cell_area,
                min_area_m2, cv_min, max_area_m2,
                max_child_label, dense_parent_threshold_cells, parent_cells,
            )
            for pid, bbox, parent_cells in simple
        ]
        return results, [os.getpid()]

    batches = _round_robin_job_batches(simple, workers)
    ctx = mp.get_context("spawn")
    decisions: list[ParentDecision] = []
    pids: set[int] = set()

    owns_executor = executor is None
    pool = executor if executor is not None else ProcessPoolExecutor(max_workers=len(batches), mp_context=ctx)
    try:
        futures = [
            pool.submit(
                evaluate_parent_batch,
                str(store.root), batch, cell_area,
                min_area_m2, cv_min, max_area_m2,
                max_child_label, dense_parent_threshold_cells,
            )
            for batch in batches
        ]
        for f in as_completed(futures):
            pid, result_batch = f.result()
            pids.add(int(pid))
            decisions.extend(result_batch)
            if verbose:
                print(f"           parent worker pid={pid} parents={len(result_batch):,}")
    finally:
        if owns_executor:
            pool.shutdown(wait=True)

    decisions.sort(key=lambda d: d.parent_id)
    return decisions, sorted(pids)



def _partial_astar_cache_ok(store: MemmapStore, meta: RasterMeta) -> bool:
    """Conservative structural check for a completed A* routing checkpoint.

    This also recognizes an interrupted Stage-10 run, where A* arrays were
    fully flushed before entering the serial MFD kernel but no final
    hydrology manifest had yet been written.
    """
    needed_grid = [
        "valid", "hydro_dem", "astar_receiver",
        "sin_aspect", "cos_aspect", "aspect_valid",
    ]
    if not all(store.exists(n) for n in needed_grid):
        return False
    if not store.exists("order"):
        return False

    try:
        for name in needed_grid:
            arr = store.open(name, "r")
            try:
                if tuple(arr.shape) != tuple(meta.shape):
                    return False
            finally:
                store.close_array(arr)

        order = store.open("order", "r")
        try:
            if order.ndim != 1 or order.size <= 0:
                return False
        finally:
            store.close_array(order)
    except Exception:
        return False

    return True



def _astar_checkpoint_matches_routing(
    store: MemmapStore,
    expected: dict,
) -> bool:
    path = store.root / "astar_checkpoint.json"
    if not path.exists():
        return False
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # A completed exact global A* routing is independent of how its queues
    # were sharded into hydrological computational domains.  Domain identity
    # matters only for an *in-stage* frontier checkpoint, validated inside
    # fine_hydrology.py.  Once A* is complete, routing_mode + DEM identity are
    # sufficient and a rewritten/repartitioned domain raster must not force an
    # expensive A* recomputation.
    return info.get("routing_mode") == expected.get("routing_mode")

def _array_shape_ok(store: MemmapStore, name: str, shape) -> bool:
    if not store.exists(name):
        return False
    arr = None
    try:
        arr = store.open(name, "r")
        return tuple(arr.shape) == tuple(shape)
    except Exception:
        return False
    finally:
        store.close_array(arr)


def _mfd_checkpoint_ok(
    store: MemmapStore,
    meta: RasterMeta,
    convergence: int,
) -> bool:
    """Validate a durable Stage-12 MFD completion checkpoint."""
    path = store.root / "mfd_complete_checkpoint.json"
    if not path.exists():
        return False

    if not (
        _array_shape_ok(store, "accumulation", meta.shape)
        and _array_shape_ok(store, "receiver", meta.shape)
        and store.exists("order")
    ):
        return False

    order = None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        order = store.open("order", "r")
        return (
            int(info.get("valid_cells", -1)) == int(order.size)
            and int(info.get("convergence", -1)) == int(convergence)
        )
    except Exception:
        return False
    finally:
        store.close_array(order)


def _recover_stage11_windows_cleanup_failure(
    store: MemmapStore,
    meta: RasterMeta,
) -> bool:
    """Recognize the exact Stage-11 WinError 32 post-MFD failure state.

    In Stage 11, `mfd_rank.npy` was deleted only AFTER:
      1. all MFD blocks were scattered,
      2. accumulation was flushed,
      3. all adjusted-receiver workers completed,
      4. receiver was flushed,
      5. the worker pool was closed.

    The subsequent deletion of `mfd_weights_0.npy` could fail because the
    parent still held the final `np.load(..., mmap_mode="r")` mapping.

    Therefore:
        accumulation + receiver present
        AND mfd_rank absent
        AND one/both mfd_weights_* still present
    is a safe signature of a computationally completed Stage-11 MFD whose
    only failure was temporary-file cleanup.
    """
    if store.exists("mfd_rank"):
        return False

    if not (
        store.exists("mfd_weights_0")
        or store.exists("mfd_weights_1")
    ):
        return False

    if not (
        _array_shape_ok(store, "accumulation", meta.shape)
        and _array_shape_ok(store, "receiver", meta.shape)
        and store.exists("order")
        and store.exists("astar_receiver")
    ):
        return False

    return True


def _cleanup_stale_mfd_buffers(store: MemmapStore, verbose: bool = False) -> None:
    """Best-effort cleanup after the previous process has exited."""
    for name in ("mfd_rank", "mfd_weights_0", "mfd_weights_1"):
        p = store.path(name)
        if not p.exists():
            continue
        removed = store.cleanup(name)
        if not removed and verbose:
            print(
                f"[PySlopeUnits] warning: stale temporary file remains locked: {p}"
            )


def _recover_completed_hydrology_cleanup_failure(
    store: MemmapStore,
    dem_path: Path,
    meta: RasterMeta,
    convergence: int,
    hydro_scale: int,
    manifest: dict,
    *,
    verbose: bool = False,
) -> bool:
    """Recover a fully computed hydrology cache that failed only during cleanup.

    This covers the V0.1.0 Windows state where:
      * A* is complete,
      * MFD accumulation and adjusted receiver are complete,
      * stream-static arrays are complete,
      * drainage components are complete,
      * but hydrology_manifest.json was not yet written because unlinking
        astar_receiver.npy or edgeflag.npy raised WinError 32.

    The expensive numerical products are validated before the manifest is
    reconstructed. No hydrology is recomputed.
    """
    if (store.root / "hydrology_manifest.json").exists():
        return False

    required_grid = (
        "valid",
        "hydro_dem",
        "receiver",
        "accumulation",
        "sin_aspect",
        "cos_aspect",
        "aspect_valid",
        "stream_static_ok",
        "stream_up_acc_max",
        "drainage_component",
    )
    if not all(_array_shape_ok(store, name, meta.shape) for name in required_grid):
        return False

    if not store.exists("order"):
        return False

    order = None
    try:
        order = store.open("order", "r")
        if order.ndim != 1 or order.size <= 0:
            return False
        valid_cells = int(order.size)
    finally:
        store.close_array(order)

    if not _mfd_checkpoint_ok(store, meta, convergence):
        return False

    astar_checkpoint = store.root / "astar_checkpoint.json"
    drainage_checkpoint = store.root / "drainage_components.json"
    if not astar_checkpoint.exists() or not drainage_checkpoint.exists():
        return False

    try:
        ainfo = json.loads(astar_checkpoint.read_text(encoding="utf-8"))
        dinfo = json.loads(drainage_checkpoint.read_text(encoding="utf-8"))
    except Exception:
        return False

    try:
        checkpoint_dem = str(Path(ainfo.get("dem", "")).resolve())
    except Exception:
        return False

    if checkpoint_dem != str(dem_path.resolve()):
        return False
    if list(ainfo.get("shape", [])) != list(meta.shape):
        return False
    if int(ainfo.get("hydro_scale", -1)) != int(hydro_scale):
        return False
    if int(ainfo.get("valid_cells", -1)) != valid_cells:
        return False
    if int(dinfo.get("components", -1)) < 0:
        return False

    # Durable completion marker BEFORE any optional cleanup.
    store.write_json("hydrology_manifest.json", manifest)

    # These arrays are expendable after the adjusted receiver exists.
    # Failure to delete them is never fatal.
    store.cleanup("astar_receiver")
    store.cleanup("edgeflag")

    if verbose:
        print(
            "[PySlopeUnits] recovered completed hydrology cache "
            "after Windows cleanup failure (no A*/MFD recomputation)"
        )
    return True


class SlopeUnits:
    """Pure-Python package using NumPy + Numba + multiprocessing + memmap.

    Design principles
    -----------------
    * no Cython and no compiled extension build step;
    * graph-heavy kernels JIT-compiled by Numba;
    * large shared rasters stored as .npy memory maps;
    * independent half-basin drainage components processed by spawned workers;
    * active parent-unit statistics evaluated by spawned workers;
    * global hydrology computed once and optionally reused across parameter runs.
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
        hydro_scale: int = 1000,
        reuse_hydrology: bool = True,
        reuse_candidates: bool = True,
        keep_work: bool = True,
        dense_parent_threshold_cells: int = 5_000_000,
        mfd_block_cells: int = 1_000_000,
        resume_partial_hydrology: bool = True,
        nodata_values=None,
        memory_budget_bytes: int | None = None,
        scratch_ram_fraction: float = 0.50,
        hydrology_domain_raster: str | Path | None = None,
        checkpoint: bool = True,
        checkpoint_minutes: float = 15.0,
        restart: bool = False,
        verbose: bool = True,
    ):
        if threshold_m2 <= 0:
            raise ValueError("threshold_m2 must be > 0")
        if min_area_m2 <= 0:
            raise ValueError("min_area_m2 must be > 0")
        if not 0 <= cv_min <= 1:
            raise ValueError("cv_min must be in [0, 1]")
        if reduction_factor <= 1:
            raise ValueError("reduction_factor must be > 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not 1 <= convergence <= 10:
            raise ValueError("convergence must be in [1, 10]")

        self.threshold_m2 = float(threshold_m2)
        self.min_area_m2 = float(min_area_m2)
        self.cv_min = float(cv_min)
        self.reduction_factor = int(reduction_factor)
        self.max_iterations = int(max_iterations)
        self.convergence = int(convergence)
        self.max_area_m2 = None if max_area_m2 is None else float(max_area_m2)
        self.workers = max(1, int(workers))
        self.numba_threads = max(1, int(numba_threads))
        self.hydro_scale = int(hydro_scale)
        self.reuse_hydrology = bool(reuse_hydrology)
        self.reuse_candidates = bool(reuse_candidates)
        self.keep_work = bool(keep_work)
        self.dense_parent_threshold_cells = int(dense_parent_threshold_cells)
        self.mfd_block_cells = max(10_000, int(mfd_block_cells))
        self.resume_partial_hydrology = bool(resume_partial_hydrology)
        self.nodata_values = normalize_nodata_values(nodata_values)
        self.memory_budget_bytes = (
            None if memory_budget_bytes is None else max(0, int(memory_budget_bytes))
        )
        self.scratch_ram_fraction = float(scratch_ram_fraction)
        self.hydrology_domain_raster = (
            None if hydrology_domain_raster is None else Path(hydrology_domain_raster)
        )
        self.checkpoint = bool(checkpoint)
        self.checkpoint_minutes = float(checkpoint_minutes)
        if self.checkpoint_minutes <= 0:
            raise ValueError("checkpoint_minutes must be > 0")
        self.restart = bool(restart)
        self.verbose = bool(verbose)

    def _prepare_hydrology(self, dem_path: Path, store: MemmapStore, meta: RasterMeta):
        manifest = _manifest_for(
            dem_path, meta, self.convergence, self.hydro_scale,
            self.nodata_values, self.hydrology_domain_raster,
        )

        if self.reuse_hydrology and not self.restart and _hydrology_cache_ok(store, manifest):
            if self.verbose:
                print("[PySlopeUnits] reusing optimized memory-mapped hydrology cache")
            return

        if (
            self.reuse_hydrology
            and not self.restart
            and _recover_completed_hydrology_cleanup_failure(
                store,
                dem_path,
                meta,
                self.convergence,
                self.hydro_scale,
                manifest,
                verbose=self.verbose,
            )
        ):
            return

        # Upgrade a Stage-9 cache without recomputing A* + MFD when the
        # expensive core arrays and hydro_dem are already available.
        if (
            self.reuse_hydrology
            and not self.restart
            and _hydrology_core_ok(store, manifest)
            and store.exists("hydro_dem")
        ):
            if self.verbose:
                print("[PySlopeUnits] upgrading existing hydrology cache to Stage 10")

            valid = store.open("valid", "r")
            hydro_dem = store.open("hydro_dem", "r")
            accumulation = store.open("accumulation", "r")
            receiver = store.open("receiver", "r")
            order = store.open("order", "r")

            static_ok = store.create(
                "stream_static_ok", meta.shape, np.uint8, fill=0
            )
            up_acc = store.create(
                "stream_up_acc_max", meta.shape, np.float32, fill=0.0
            )
            if self.verbose:
                print("[PySlopeUnits] precomputing threshold-independent stream tests")
            precompute_stream_seed_static(
                hydro_dem, valid, accumulation, order, static_ok, up_acc
            )
            static_ok.flush(); up_acc.flush()

            component = store.create(
                "drainage_component", meta.shape, index_dtype_for_shape(meta.shape), fill=0
            )
            if self.verbose:
                print("[PySlopeUnits] precomputing global drainage components")
            ncomp = int(
                drainage_components(receiver, valid, order, component)
            )
            component.flush()
            store.write_json(
                "drainage_components.json", {"components": ncomp}
            )

            store.write_json("hydrology_manifest.json", manifest)
            if self.verbose:
                print(
                    f"[PySlopeUnits] optimized hydrology cache ready | "
                    f"drainage_components={ncomp:,}"
                )
            return

        shape = meta.shape

        resumed_astar = (
            self.resume_partial_hydrology
            and not self.restart
            and _partial_astar_cache_ok(store, meta)
            and _astar_checkpoint_matches_routing(store, manifest)
        )
        edgeflag = None

        # Mid-stage v0.1.6 A* checkpoint.  Unlike the completed routing cache
        # above, this preserves the mutable frontier and can resume from inside
        # the long exact domain-sharded A* traversal.
        fine_astar_checkpoint = store.root.parent / "checkpoints" / "fine_astar.json"
        partial_domain_astar = (
            self.checkpoint
            and not self.restart
            and self.hydrology_domain_raster is not None
            and fine_astar_checkpoint.exists()
            and all(
                store.exists(name)
                for name in (
                    "valid", "hydro_dem", "astar_receiver", "edgeflag", "order",
                    "astar_domain_state", "astar_domain_heap_idx",
                    "astar_domain_heap_age", "astar_domain_sizes",
                    "astar_global_heap", "astar_global_pos",
                )
            )
        )

        if resumed_astar:
            if self.verbose:
                print(
                    "[PySlopeUnits] resuming completed A* routing from partial cache "
                    "(no A* recomputation)"
                )
            valid = store.open("valid", "r")
            hydro_dem = store.open("hydro_dem", "r")
            astar_receiver = store.open("astar_receiver", "r")
            order = store.open("order", "r")
            nvalid = int(order.size)

        else:
            if partial_domain_astar:
                if self.verbose:
                    print("[PySlopeUnits] detected resumable in-stage fine A* checkpoint")
                valid = store.open("valid", "r")
                hydro_dem = store.open("hydro_dem", "r")
                astar_receiver = store.open("astar_receiver", "r+")
                edgeflag = store.open("edgeflag", "r+")
                order = store.open("order", "r+")
                nvalid = int(order.size)
                index_dtype = order.dtype
            else:
                if self.verbose:
                    print(
                        "[PySlopeUnits] preparing DEM blockwise | storage=memmap"
                    )
                nvalid = prepare_dem_memmaps_blockwise(
                    dem_path,
                    store,
                    meta,
                    nodata_values=self.nodata_values,
                    hydro_scale=self.hydro_scale,
                    numba_threads=self.numba_threads,
                    max_block_cells=max(250_000, min(2_000_000, self.mfd_block_cells * 2)),
                    verbose=self.verbose,
                )
                valid = store.open("valid", "r")
                hydro_dem = store.open("hydro_dem", "r")

                index_dtype = index_dtype_for_shape(shape)
                if self.verbose:
                    print(
                        f"[PySlopeUnits] index dtype | {dtype_name(index_dtype)} | "
                        f"grid-cells={int(shape[0]) * int(shape[1]):,}"
                    )
                astar_receiver = store.create(
                    "astar_receiver", shape, index_dtype, fill=-1
                )
                edgeflag = store.create(
                    "edgeflag", shape, np.uint8, fill=0
                )
                order = store.create("order", (nvalid,), index_dtype)

            if self.hydrology_domain_raster is not None:
                astar_result = run_domain_sharded_astar(
                    store,
                    meta=meta,
                    domain_raster=self.hydrology_domain_raster,
                    hydro_dem=hydro_dem,
                    valid=valid,
                    receiver=astar_receiver,
                    order=order,
                    edgeflag=edgeflag,
                    nvalid=nvalid,
                    xres_scaled=meta.xres * self.hydro_scale,
                    yres_scaled=meta.yres * self.hydro_scale,
                    index_dtype=index_dtype,
                    checkpoint=self.checkpoint,
                    checkpoint_minutes=self.checkpoint_minutes,
                    restart=self.restart,
                    verbose=self.verbose,
                )
                visited = int(astar_result.visited_cells)
            else:
                # Original exact single-global-heap A* retained for direct and
                # small-dataset runs and as the regression reference.
                astar_inlist = store.create_temp(
                    "astar_inlist_tmp", shape, np.uint8, fill=0
                )
                astar_worked = store.create_temp(
                    "astar_worked_tmp", shape, np.uint8, fill=0
                )
                astar_heap_idx = store.create_temp(
                    "astar_heap_idx_tmp", (nvalid,), index_dtype
                )
                astar_heap_age = store.create_temp(
                    "astar_heap_age_tmp", (nvalid,), index_dtype
                )

                if self.verbose:
                    print(
                        f"[PySlopeUnits] GRASS-like A* routing | "
                        f"hybrid RAM/disk scratch | valid={nvalid:,}"
                    )
                visited = astar_route_preallocated(
                    hydro_dem,
                    valid,
                    meta.xres * self.hydro_scale,
                    meta.yres * self.hydro_scale,
                    astar_receiver,
                    order,
                    edgeflag,
                    astar_inlist,
                    astar_worked,
                    astar_heap_idx,
                    astar_heap_age,
                )

                store.close_many(
                    astar_inlist, astar_worked, astar_heap_idx, astar_heap_age
                )
                del astar_inlist, astar_worked, astar_heap_idx, astar_heap_age
                store.cleanup("astar_inlist_tmp")
                store.cleanup("astar_worked_tmp")
                store.cleanup("astar_heap_idx_tmp")
                store.cleanup("astar_heap_age_tmp")

            if int(visited) != nvalid:
                raise RuntimeError(
                    f"A* visited {visited:,} cells, expected {nvalid:,}"
                )
            astar_receiver.flush()
            order.flush()
            edgeflag.flush()

            # Completed-stage checkpoint.  If the process is stopped during
            # MFD or any later stage, A* is never repeated.
            store.write_json(
                "astar_checkpoint.json",
                {
                    "dem": str(dem_path.resolve()),
                    "shape": list(meta.shape),
                    "valid_cells": nvalid,
                    "hydro_scale": self.hydro_scale,
                    "routing_mode": manifest["routing_mode"],
                    "hydrology_domain_raster": manifest["hydrology_domain_raster"],
                },
            )

            # The compact completion checkpoint above now owns restartability.
            # Remove the large in-stage frontier arrays only after it is durable.
            for _name in (
                "astar_domain_state", "astar_domain_heap_idx",
                "astar_domain_heap_age", "astar_domain_sizes",
                "astar_global_heap", "astar_global_pos",
            ):
                store.remove(_name, best_effort=True)

        mfd_ready = False

        if not self.restart and _mfd_checkpoint_ok(store, meta, self.convergence):
            mfd_ready = True
            if self.verbose:
                print(
                    "[PySlopeUnits] reusing completed MFD checkpoint "
                    "(no MFD recomputation)"
                )

        elif not self.restart and _recover_stage11_windows_cleanup_failure(store, meta):
            # This is exactly the state produced by the Stage-11 WinError 32
            # reported after adjusted-receiver completion.
            mfd_ready = True
            order_for_checkpoint = store.open("order", "r")
            store.write_json(
                "mfd_complete_checkpoint.json",
                {
                    "valid_cells": int(order_for_checkpoint.size),
                    "convergence": int(self.convergence),
                    "xres_scaled": float(meta.xres * self.hydro_scale),
                    "yres_scaled": float(meta.yres * self.hydro_scale),
                    "recovered_from": "stage11_windows_cleanup_failure",
                },
            )
            if self.verbose:
                print(
                    "[PySlopeUnits] recovered completed Stage-11 MFD after "
                    "Windows temp-file cleanup failure "
                    "(no MFD recomputation)"
                )
            del order_for_checkpoint
            _cleanup_stale_mfd_buffers(store, verbose=self.verbose)

        if not mfd_ready:
            if self.verbose:
                print(
                    f"[PySlopeUnits] MFD accumulation | "
                    f"convergence={self.convergence} | "
                    f"block-parallel weights | "
                    f"Numba threads={self.numba_threads}"
                )

            # Only discard MFD outputs when there is no completed checkpoint.
            store.remove("accumulation")
            store.remove("receiver")
            store.cleanup("mfd_rank")
            _cleanup_stale_mfd_buffers(store, verbose=self.verbose)

            mfd_result = run_mfd_block_parallel(
                store,
                xres_scaled=meta.xres * self.hydro_scale,
                yres_scaled=meta.yres * self.hydro_scale,
                convergence=self.convergence,
                numba_threads=self.numba_threads,
                workers=self.workers,
                block_cells=self.mfd_block_cells,
                verbose=self.verbose,
            )

        accumulation = store.open("accumulation", "r")
        receiver = store.open("receiver", "r")

        # Threshold-independent data used by every candidate level.
        static_ok = store.create(
            "stream_static_ok", shape, np.uint8, fill=0
        )
        up_acc = store.create(
            "stream_up_acc_max", shape, np.float32, fill=0.0
        )
        if self.verbose:
            print("[PySlopeUnits] precomputing threshold-independent stream tests")
        precompute_stream_seed_static(
            hydro_dem, valid, accumulation, order, static_ok, up_acc
        )
        static_ok.flush(); up_acc.flush()

        component = store.create(
            "drainage_component", shape, index_dtype_for_shape(shape), fill=0
        )
        if self.verbose:
            print("[PySlopeUnits] precomputing global drainage components")
        ncomp = int(
            drainage_components(receiver, valid, order, component)
        )
        component.flush()
        store.write_json(
            "drainage_components.json", {"components": ncomp}
        )

        # CRITICAL WINDOWS RULE:
        # The expensive hydrology cache is complete at this point. Persist its
        # manifest BEFORE attempting to delete any expendable temporary files.
        store.write_json("hydrology_manifest.json", manifest)

        # Explicitly release parent-owned mappings. Cleanup is best-effort and
        # can never invalidate the completed computation.
        try:
            store.close_array(astar_receiver)
        except Exception:
            pass
        try:
            store.close_array(edgeflag)
        except Exception:
            pass
        try:
            del astar_receiver
        except Exception:
            pass
        try:
            del edgeflag
        except Exception:
            pass

        store.cleanup("astar_receiver")
        store.cleanup("edgeflag")

        if self.verbose:
            print(f"[PySlopeUnits] optimized hydrology cache ready | drainage_components={ncomp:,}")

    def run(
        self, dem_path, output_path, *, work_dir=None,
        vector_path=None, vector_layer="slope_units", vector_backend="auto",
    ) -> SlopeUnitsResult:
        t0 = time.perf_counter()
        dem_path = Path(dem_path)
        output_path = Path(output_path)
        if work_dir is None:
            work_dir = output_path.parent / "workspace" / "engine"
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        store = MemmapStore(
            work_dir / "memmap",
            ram_budget_bytes=self.memory_budget_bytes,
            ram_fraction=self.scratch_ram_fraction,
            verbose=self.verbose,
        )
        candidate_cache = MultiThresholdCandidateCache(work_dir)
        meta = read_meta(dem_path)

        self._prepare_hydrology(dem_path, store, meta)

        valid = store.open("valid", "r")
        receiver = store.open("receiver", "r")
        accumulation = store.open("accumulation", "r")
        order = store.open("order", "r")
        stream_static_ok = store.open("stream_static_ok", "r")
        stream_up_acc_max = store.open("stream_up_acc_max", "r")
        valid_cells = int(np.count_nonzero(valid))

        # Iterative state is fresh for every parameter run.
        stream = store.create("stream", meta.shape, np.uint8, fill=0)
        todo = store.create("todo", meta.shape, np.int32, fill=0)
        todo_next = store.create("todo_next", meta.shape, np.int32, fill=0)
        final = store.create("final", meta.shape, np.int32, fill=0)
        todo[...] = valid.astype(np.int32)
        todo.flush()

        next_final_id = 1
        schedule = threshold_schedule(
            self.threshold_m2,
            meta.cell_area,
            self.reduction_factor,
            self.max_iterations,
        )
        infos: list[IterationInfo] = []

        process_pool = None
        if self.workers > 1:
            process_pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
            )

        try:
            for level in schedule:
                iteration = int(level.level)
                threshold_cells_int = int(level.cells)
                threshold = float(level.area_m2)

                it0 = time.perf_counter()
                active_cells = int(np.count_nonzero(todo))
                if active_cells == 0:
                    break

                if self.verbose:
                    print(
                        f"[PySlopeUnits] iteration {iteration:02d} | "
                        f"threshold={threshold:,.3f} m2 "
                        f"({threshold_cells_int:,} cells) | active={active_cells:,}"
                    )

                threshold_cells = float(threshold_cells_int)
                candidate_key = CandidateKey(threshold_cells_int)
                candidate_t0 = time.perf_counter()
                candidate_cache_hit = False

                if self.reuse_candidates and candidate_cache.exists(candidate_key):
                    candidate_cache_hit = True
                    link_mode = candidate_cache.activate(candidate_key, store)
                    hb_stats = candidate_cache.load_stats(candidate_key)
                    stream_cells = int(hb_stats.get("stream_cells", 0))
                    if self.verbose:
                        print(
                            f"[PySlopeUnits] reusing half-basin candidate "
                            f"threshold_cells={threshold_cells_int:,} ({link_mode})"
                        )
                else:
                    stream_mask_grasslike_static(
                        valid,
                        accumulation,
                        receiver,
                        order,
                        threshold_cells,
                        stream_static_ok,
                        stream_up_acc_max,
                        stream,
                    )
                    stream.flush()
                    stream_cells = int(np.count_nonzero(stream))

                    candidate_cache.prepare_for_build(store)
                    hb_stats = build_half_basins(
                        store,
                        workers=self.workers,
                        executor=process_pool,
                        verbose=self.verbose,
                    )
                    hb_stats["stream_cells"] = stream_cells
                    hb_stats["threshold_cells"] = threshold_cells_int
                    if self.reuse_candidates:
                        candidate_cache.save_from_working(
                            candidate_key, store, hb_stats
                        )

                hb = store.open("half_basins", "r")
                candidate_seconds = time.perf_counter() - candidate_t0

                parent_t0 = time.perf_counter()
                counts, rmin, rmax, cmin, cmax = parent_bboxes(todo)
                parent_ids = np.flatnonzero(counts > 0)
                parent_ids = parent_ids[parent_ids > 0]
                jobs = []
                for pid0 in parent_ids:
                    pid = int(pid0)
                    jobs.append((
                        pid,
                        (int(rmin[pid]), int(rmax[pid]) + 1, int(cmin[pid]), int(cmax[pid]) + 1),
                        int(counts[pid]),
                    ))

                if self.verbose:
                    print(f"[PySlopeUnits] parent evaluation: parents={len(jobs):,} | workers={min(self.workers, max(1, len(jobs)))}")

                decisions, parent_pids = _evaluate_parents_parallel(
                    store,
                    jobs,
                    workers=self.workers,
                    cell_area=meta.cell_area,
                    min_area_m2=self.min_area_m2,
                    cv_min=self.cv_min,
                    max_area_m2=self.max_area_m2,
                    max_child_label=int(hb_stats["max_half_basin_label"]),
                    dense_parent_threshold_cells=self.dense_parent_threshold_cells,
                    executor=process_pool,
                    verbose=self.verbose,
                )

                nparents = int(counts.size - 1)
                rejected = np.zeros(nparents + 1, dtype=np.uint8)
                parent_final = np.zeros(nparents + 1, dtype=np.int32)
                hb_max = int(hb_stats["max_half_basin_label"])

                # A current hydrological half-basin is not guaranteed to be
                # nested inside exactly one active parent. Therefore decisions
                # are stored by the pair (parent_id, half_basin_id).
                #
                # Each parent's child_ids are sorted already (np.unique or
                # flatnonzero in parent_eval), allowing binary search in the
                # Numba application kernel.
                decision_by_parent = {int(d.parent_id): d for d in decisions}

                pair_counts = np.zeros(nparents + 1, dtype=np.int64)
                for d in decisions:
                    pid = int(d.parent_id)
                    if not d.reject:
                        pair_counts[pid] = int(len(d.child_ids))

                parent_offsets = np.zeros(nparents + 2, dtype=np.int64)
                for pid in range(1, nparents + 1):
                    parent_offsets[pid + 1] = parent_offsets[pid] + pair_counts[pid]

                total_pairs = int(parent_offsets[-1])
                pair_child_ids = np.empty(total_pairs, dtype=np.int32)
                pair_actions = np.empty(total_pairs, dtype=np.int32)

                next_parent_id = 1
                rejected_count = 0
                split_children = 0
                finalized_children = 0
                crossing_pairs = 0

                # Diagnostic only: count half-basin IDs that occur under more
                # than one parent. This is legal and no longer an error.
                child_first_parent = {}

                for d in decisions:
                    pid = int(d.parent_id)

                    if d.reject:
                        rejected[pid] = 1
                        parent_final[pid] = next_final_id
                        next_final_id += 1
                        rejected_count += 1
                        continue

                    lo = int(parent_offsets[pid])
                    for local_idx, (child, split) in enumerate(zip(d.child_ids, d.split)):
                        child = int(child)
                        pos = lo + local_idx
                        pair_child_ids[pos] = child

                        first_pid = child_first_parent.get(child)
                        if first_pid is None:
                            child_first_parent[child] = pid
                        elif first_pid != pid:
                            crossing_pairs += 1

                        if split:
                            pair_actions[pos] = -next_parent_id
                            next_parent_id += 1
                            split_children += 1
                        else:
                            pair_actions[pos] = next_final_id
                            next_final_id += 1
                            finalized_children += 1

                todo_next[...] = 0
                apply_parent_child_decisions(
                    todo,
                    hb,
                    rejected,
                    parent_final,
                    parent_offsets,
                    pair_child_ids,
                    pair_actions,
                    final,
                    todo_next,
                )
                final.flush(); todo_next.flush()
                parent_seconds = time.perf_counter() - parent_t0

                unresolved = int(np.count_nonzero(todo_next))
                finalized_cells = active_cells - unresolved

                # Next iteration: one linear memmap copy, then clear scratch lazily next pass.
                todo[...] = todo_next
                todo.flush()

                info = IterationInfo(
                    iteration=iteration,
                    threshold_m2=float(threshold),
                    active_cells=active_cells,
                    stream_cells=stream_cells,
                    stream_branch_roots=int(hb_stats["stream_branch_roots"]),
                    half_basin_label_max=hb_max,
                    active_parents=len(jobs),
                    rejected_parents=rejected_count,
                    split_children=split_children,
                    finalized_children=finalized_children,
                    finalized_cells=finalized_cells,
                    unresolved_cells=unresolved,
                    halfbasin_worker_pids=list(hb_stats.get("worker_pids", [])),
                    parent_worker_pids=parent_pids,
                    candidate_cache_hit=bool(candidate_cache_hit),
                    candidate_seconds=float(candidate_seconds),
                    parent_seconds=float(parent_seconds),
                    seconds=time.perf_counter() - it0,
                )
                infos.append(info)

                if self.verbose:
                    print(
                        f"           streams={stream_cells:,} | branch_roots={info.stream_branch_roots:,} | "
                        f"parents={info.active_parents:,} | rejected={rejected_count:,} | "
                        f"split_children={split_children:,} | crossings={crossing_pairs:,} | "
                        f"remaining={unresolved:,} | "
                        f"candidate={candidate_seconds:.1f}s | parent={parent_seconds:.1f}s | "
                        f"total={info.seconds:.1f}s"
                    )

                # Release per-iteration Python/NumPy objects before the next
                # half-basin pass; the large persistent rasters remain memmapped.
                del counts, rmin, rmax, cmin, cmax, decisions, rejected, parent_final, parent_offsets, pair_child_ids, pair_actions, decision_by_parent, child_first_parent, hb

                if unresolved == 0:
                    break


        finally:
            if process_pool is not None:
                process_pool.shutdown(wait=True)

        if np.any(todo):
            if self.verbose:
                print("[PySlopeUnits] finalizing remaining active parents at iteration limit")
            next_final_id = finalize_remaining(todo, final, next_final_id)
            final.flush(); todo.flush()

        # IDs were assigned sequentially; final_units is maximum positive ID.
        final_units = int(np.asarray(final).max())
        missing = int(count_unlabeled(valid, final))
        if missing != 0:
            raise RuntimeError(f"final result contains {missing:,} unlabeled valid cells")

        if self.verbose:
            print(f"[PySlopeUnits] writing GeoTIFF blockwise: {output_path}")
        write_raster_blockwise(
            output_path,
            final,
            meta,
            valid,
            dtype="int32",
            nodata=0,
        )

        if vector_path is not None:
            if self.verbose:
                print(f"[PySlopeUnits] exporting GeoPackage: {vector_path}")
            export_geopackage(
                output_path, vector_path, layer_name=vector_layer,
                backend=vector_backend, overwrite=True, connectivity=4,
                verbose=self.verbose,
            )

        total_seconds = time.perf_counter() - t0
        report = {
            "engine": "PySlope 1.3 validation + GeoPackage export engine",
            "version": "1.3.0",
            "dem": str(dem_path),
            "output": str(output_path),
            "work_dir": str(work_dir),
            "parameters": {
                "threshold_m2": self.threshold_m2,
                "min_area_m2": self.min_area_m2,
                "cv_min": self.cv_min,
                "reduction_factor": self.reduction_factor,
                "max_iterations": self.max_iterations,
                "convergence": self.convergence,
                "max_area_m2": self.max_area_m2,
                "workers": self.workers,
                "numba_threads": self.numba_threads,
                "reuse_hydrology": self.reuse_hydrology,
                "reuse_candidates": self.reuse_candidates,
                "mfd_block_cells": self.mfd_block_cells,
                "resume_partial_hydrology": self.resume_partial_hydrology,
                "nodata_values": [
                    "nan" if np.isnan(v) else float(v)
                    for v in self.nodata_values
                ],
                "out_of_core_dem_prepare": True,
                "out_of_core_astar_scratch": True,
                "dense_parent_threshold_cells": self.dense_parent_threshold_cells,
                "memory_budget_bytes": self.memory_budget_bytes,
                "scratch_ram_fraction": self.scratch_ram_fraction,
            },
            "valid_cells": valid_cells,
            "cell_area_m2": meta.cell_area,
            "resolution_x": meta.xres,
            "resolution_y": meta.yres,
            "final_units": final_units,
            "candidate_cache_hits": int(sum(1 for x in infos if x.candidate_cache_hit)),
            "candidate_cache_misses": int(sum(1 for x in infos if not x.candidate_cache_hit)),
            "candidate_seconds_total": float(sum(x.candidate_seconds for x in infos)),
            "parent_seconds_total": float(sum(x.parent_seconds for x in infos)),
            "total_seconds": total_seconds,
            "iterations": [asdict(x) for x in infos],
        }
        store.write_json("run_report.json", report)

        if not self.keep_work:
            # Keep report beside output, then remove large cache.
            report_copy = output_path.with_suffix(".run_report.json")
            report_copy.write_text(json.dumps(report, indent=2), encoding="utf-8")
            shutil.rmtree(store.root, ignore_errors=True)

        if self.verbose:
            print(f"[PySlopeUnits] final slope units: {final_units:,}")
            print(f"[PySlopeUnits] total time: {total_seconds / 60.0:.2f} min")

        return SlopeUnitsResult(
            output=output_path,
            work_dir=work_dir,
            valid_cells=valid_cells,
            final_units=final_units,
            iterations=infos,
            total_seconds=total_seconds,
        )
