from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window

from .hydro_domains import (
    HydrologicalDomainPlan,
    build_coarse_dem,
    estimate_valid_fine_cells_from_coarse,
    partition_coarse_hydrology,
    projected_source_bounds,
    save_plan,
    source_grid_info,
    write_domain_products,
)
from .pipeline import SlopeUnits
from .raster import normalize_nodata_values, normalize_source_nodata
from .resources import detect_system_resources, make_resource_plan


@dataclass(frozen=True)
class AdaptiveRunResult:
    output_dir: Path
    plan_file: Path
    processed_domains: int
    skipped_domains: int
    failed_domains: int
    total_seconds: float
    execution_mode: str = "unknown"


def _aligned_target_grid(source, target_crs: str, resolution_m: float):
    """Return one deterministic fine grid, independent of HDD planning resolution."""
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError("Source DEM has no CRS")

        target = rasterio.crs.CRS.from_user_input(target_crs)
        same_crs = src.crs == target
        xres = abs(float(src.transform.a))
        yres = abs(float(src.transform.e))
        tol = max(1e-6, float(resolution_m) * 1e-6)

        # Preserve the exact source grid whenever CRS + resolution already match.
        # This is essential for SINGLE/HDD equivalence tests.
        if (
            same_crs
            and abs(xres - resolution_m) <= tol
            and abs(yres - resolution_m) <= tol
        ):
            return src.transform, int(src.width), int(src.height)

        left, bottom, right, top = transform_bounds(
            src.crs, target, *src.bounds, densify_pts=21
        )
        left = math.floor(left / resolution_m) * resolution_m
        bottom = math.floor(bottom / resolution_m) * resolution_m
        right = math.ceil(right / resolution_m) * resolution_m
        top = math.ceil(top / resolution_m) * resolution_m
        width = max(1, int(round((right - left) / resolution_m)))
        height = max(1, int(round((top - bottom) / resolution_m)))
        return from_origin(left, top, resolution_m, resolution_m), width, height


def materialize_target_dem(
    source,
    output: str | Path,
    *,
    target_crs: str,
    fine_resolution_m: float,
    max_block_cells: int = 2_000_000,
    verbose: bool = True,
) -> Path:
    """Materialize the canonical fine DEM used by both SINGLE and HDD.

    The coarse HDD grid is deliberately *not* used to mask or align this DEM.
    Therefore changing the memory limit or coarse resolution cannot change the
    fine DEM presented to the slope-unit engine.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".json")

    transform, width, height = _aligned_target_grid(
        source, target_crs, float(fine_resolution_m)
    )

    with rasterio.open(source) as src:
        source_stat = None
        try:
            st = Path(source).stat()
            source_stat = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except Exception:
            pass

        expected = {
            "source": str(source),
            "source_stat": source_stat,
            "source_crs": None if src.crs is None else str(src.crs),
            "source_shape": [int(src.height), int(src.width)],
            "source_transform": [float(x) for x in src.transform[:6]],
            "target_crs": str(rasterio.crs.CRS.from_user_input(target_crs)),
            "fine_resolution_m": float(fine_resolution_m),
            "target_shape": [int(height), int(width)],
            "target_transform": [float(x) for x in transform[:6]],
            "resampling": "bilinear",
            "materialization_version": 2,
        }

        if output.exists() and manifest_path.exists():
            try:
                got = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                got = None
            if got == expected:
                if verbose:
                    print(f"[PySlopeUnits adaptive] reusing canonical fine DEM: {output}")
                return output

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": target_crs,
            "transform": transform,
            "nodata": np.nan,
            "compress": "zstd",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
        block_rows = max(1, min(height, max_block_cells // max(1, width)))

        if verbose:
            print(
                f"[PySlopeUnits adaptive] canonical fine DEM | "
                f"{width:,} x {height:,} | {fine_resolution_m:g} m"
            )

        same_grid = (
            src.crs == rasterio.crs.CRS.from_user_input(target_crs)
            and src.width == width
            and src.height == height
            and src.transform.almost_equals(transform)
        )

        with rasterio.open(output, "w", **profile) as dst:
            if same_grid:
                for r0 in range(0, height, block_rows):
                    nr = min(block_rows, height - r0)
                    win = Window(0, r0, width, nr)
                    arr = src.read(1, window=win, masked=True, out_dtype="float32")
                    dst.write(
                        np.asarray(arr.filled(np.nan), dtype=np.float32),
                        1,
                        window=win,
                    )
            else:
                with WarpedVRT(
                    src,
                    crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                ) as vrt:
                    for r0 in range(0, height, block_rows):
                        nr = min(block_rows, height - r0)
                        win = Window(0, r0, width, nr)
                        arr = vrt.read(
                            1, window=win, masked=True, out_dtype="float32"
                        )
                        dst.write(
                            np.asarray(arr.filled(np.nan), dtype=np.float32),
                            1,
                            window=win,
                        )

        manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")

    return output


class AdaptiveSlopeUnits:
    """Dataset-agnostic, memory-adaptive orchestration.

    The coarse hydrological decomposition is used for resource planning only;
    it never clips or defines scientific slope-unit boundaries.  The canonical
    fine DEM is processed with exact global out-of-core hydrology.  The default
    lazy hierarchy materializes only the active graph frontier and can stream
    candidate thresholds one at a time when full candidate caching would be
    too expensive.
    """

    def __init__(
        self,
        *,
        fine_resolution_m: float | None = None,
        target_crs: str | None = None,
        memory_limit_gb: float | None = None,
        memory_fraction: float = 0.65,
        coarse_memory_fraction: float = 0.18,
        bytes_per_fine_cell: int = 112,
        coarse_bytes_per_cell: int = 72,
        coarse_resolution_m: float | None = None,
        workers: int | None = None,
        numba_threads: int | None = None,
        nodata_values=None,
        verbose: bool = True,
        **slopeunit_kwargs,
    ):
        self.fine_resolution_m = (
            None if fine_resolution_m is None else float(fine_resolution_m)
        )
        self.target_crs = (
            None if target_crs is None else str(target_crs)
        )
        self.memory_limit_gb = memory_limit_gb
        self.memory_fraction = float(memory_fraction)
        self.coarse_memory_fraction = float(coarse_memory_fraction)
        self.bytes_per_fine_cell = int(bytes_per_fine_cell)
        self.coarse_bytes_per_cell = int(coarse_bytes_per_cell)
        self.coarse_resolution_m = coarse_resolution_m
        self.workers = workers
        self.numba_threads = numba_threads
        self.nodata_values = normalize_nodata_values(nodata_values)
        self.verbose = bool(verbose)
        self.slopeunit_kwargs = dict(slopeunit_kwargs)
        self.slopeunit_kwargs["nodata_values"] = self.nodata_values

    def _effective_source(self, source, output_dir: Path) -> str:
        return normalize_source_nodata(
            source,
            output_dir / "_work" / "source_nodata_normalized.tif",
            nodata_values=self.nodata_values,
            verbose=self.verbose,
        )

    def _resolve_target_grid(self, source) -> tuple[str, float]:
        """Resolve a metric projected CRS and square fine-grid resolution.

        General policy:
        - projected DEMs in metres keep their native CRS by default;
        - their native square resolution is reused when --fine-resolution-m is omitted;
        - geographic or non-metric DEMs require an explicit --target-crs;
        - if reprojection is requested, --fine-resolution-m must be explicit.
        """
        with rasterio.open(source) as src:
            if src.crs is None:
                raise ValueError(
                    "Source DEM has no CRS. Define a valid CRS before running PySlopeUnits."
                )

            source_crs = src.crs
            target = (
                source_crs
                if self.target_crs is None
                else rasterio.crs.CRS.from_user_input(self.target_crs)
            )

            if not target.is_projected:
                raise ValueError(
                    "PySlopeUnits requires a projected metric target CRS because area and "
                    "distance parameters are expressed in metres. Use --target-crs."
                )

            try:
                units_name, units_factor = target.linear_units_factor
            except Exception as exc:
                raise ValueError(
                    "Unable to determine linear units for the target CRS. "
                    "Use a projected CRS with metre units."
                ) from exc

            if not math.isclose(float(units_factor), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Target CRS linear units are {units_name!r}, not metres. "
                    "Use a projected CRS with metre units."
                )

            same_crs = source_crs == target

            if self.fine_resolution_m is None:
                if not same_crs:
                    raise ValueError(
                        "--fine-resolution-m is required when --target-crs differs from "
                        "the source DEM CRS."
                    )

                xres = abs(float(src.transform.a))
                yres = abs(float(src.transform.e))
                tol = max(1e-9, max(xres, yres) * 1e-6)
                if abs(xres - yres) > tol:
                    raise ValueError(
                        "Source DEM pixels are not square. Specify --fine-resolution-m "
                        "for the canonical square processing grid."
                    )
                resolution = xres
            else:
                resolution = float(self.fine_resolution_m)

            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError("fine_resolution_m must be > 0")

        self.target_crs = str(target)
        self.fine_resolution_m = float(resolution)

        if self.verbose:
            print(
                f"[PySlopeUnits adaptive] target grid | CRS={self.target_crs} | "
                f"resolution={self.fine_resolution_m:g} m"
            )

        return self.target_crs, self.fine_resolution_m

    def plan(self, source, output_dir: str | Path) -> HydrologicalDomainPlan:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        effective_source = self._effective_source(source, output_dir)
        self._resolve_target_grid(effective_source)
        resources = detect_system_resources(work_dir)
        left, bottom, right, top = projected_source_bounds(
            effective_source, self.target_crs
        )
        source_fine_cells_est, source_matches_target = source_grid_info(
            effective_source, self.target_crs, self.fine_resolution_m
        )
        memory_limit_bytes = (
            None
            if self.memory_limit_gb is None
            else int(self.memory_limit_gb * 1024**3)
        )
        resource_plan = make_resource_plan(
            resources,
            target_width_m=right - left,
            target_height_m=top - bottom,
            fine_resolution_m=self.fine_resolution_m,
            memory_limit_bytes=memory_limit_bytes,
            memory_fraction=self.memory_fraction,
            coarse_memory_fraction=self.coarse_memory_fraction,
            bytes_per_fine_cell=self.bytes_per_fine_cell,
            coarse_bytes_per_cell=self.coarse_bytes_per_cell,
            workers=self.workers,
            numba_threads=self.numba_threads,
            coarse_resolution_m=self.coarse_resolution_m,
        )

        if self.verbose:
            print(
                f"[PySlopeUnits adaptive] RAM total="
                f"{resources.total_ram_bytes / 1024**3:.1f} GB | "
                f"available-now={resources.available_ram_bytes / 1024**3:.1f} GB | "
                f"planning-budget={resource_plan.memory_budget_bytes / 1024**3:.1f} GB"
            )
            print(
                f"[PySlopeUnits adaptive] target fine cells/domain="
                f"{resource_plan.target_fine_cells:,} | source-estimate="
                f"{source_fine_cells_est:,}"
            )

        (output_dir / "resources.json").write_text(
            json.dumps(
                {"system": resources.to_dict(), "plan": resource_plan.to_dict()},
                indent=2,
            ),
            encoding="utf-8",
        )

        if (
            source_matches_target
            and source_fine_cells_est <= resource_plan.target_fine_cells
        ):
            if self.verbose:
                print(
                    "[PySlopeUnits adaptive] source fits memory plan -> "
                    "DIRECT mode (no hydrological decomposition)"
                )
            plan = HydrologicalDomainPlan(
                mode="direct",
                coarse_dem=None,
                domain_raster=None,
                domain_vector=None,
                target_crs=self.target_crs,
                fine_resolution_m=self.fine_resolution_m,
                coarse_resolution_m=resource_plan.coarse_resolution_m,
                max_fine_cells=resource_plan.target_fine_cells,
                max_coarse_cells=0,
                source_fine_cells_est=source_fine_cells_est,
                domains=[],
            )
            save_plan(plan, output_dir / "processing_plan.json")
            return plan

        coarse_dem = work_dir / "coarse_dem.tif"
        build_coarse_dem(
            effective_source,
            coarse_dem,
            target_crs=self.target_crs,
            resolution_m=resource_plan.coarse_resolution_m,
            verbose=self.verbose,
        )
        valid_fine_est, valid_coarse_cells, _ = (
            estimate_valid_fine_cells_from_coarse(
                coarse_dem, fine_resolution_m=self.fine_resolution_m
            )
        )
        if self.verbose:
            print(
                f"[PySlopeUnits adaptive] coarse valid="
                f"{valid_coarse_cells:,} cells -> estimated fine valid="
                f"{valid_fine_est:,}"
            )

        if valid_fine_est <= resource_plan.target_fine_cells:
            if self.verbose:
                print(
                    "[PySlopeUnits adaptive] valid fine footprint fits memory -> "
                    "SINGLE mode (no hydrological decomposition)"
                )
            plan = HydrologicalDomainPlan(
                mode="single",
                coarse_dem=coarse_dem,
                domain_raster=None,
                domain_vector=None,
                target_crs=self.target_crs,
                fine_resolution_m=self.fine_resolution_m,
                coarse_resolution_m=resource_plan.coarse_resolution_m,
                max_fine_cells=resource_plan.target_fine_cells,
                max_coarse_cells=0,
                source_fine_cells_est=valid_fine_est,
                domains=[],
            )
            save_plan(plan, output_dir / "processing_plan.json")
            return plan

        labels, sizes, meta, max_coarse_cells = partition_coarse_hydrology(
            coarse_dem,
            work_dir,
            max_fine_cells=resource_plan.target_fine_cells,
            fine_resolution_m=self.fine_resolution_m,
            workers=resource_plan.workers,
            numba_threads=resource_plan.numba_threads,
            verbose=self.verbose,
        )

        domain_raster = output_dir / "hydro_domains.tif"
        domains = write_domain_products(
            labels,
            sizes,
            meta,
            fine_resolution_m=self.fine_resolution_m,
            raster_path=domain_raster,
            vector_path=None,
            verbose=self.verbose,
        )

        plan = HydrologicalDomainPlan(
            mode="hydrological",
            coarse_dem=coarse_dem,
            domain_raster=domain_raster,
            domain_vector=None,
            target_crs=self.target_crs,
            fine_resolution_m=self.fine_resolution_m,
            coarse_resolution_m=resource_plan.coarse_resolution_m,
            max_fine_cells=resource_plan.target_fine_cells,
            max_coarse_cells=max_coarse_cells,
            source_fine_cells_est=valid_fine_est,
            domains=domains,
        )
        save_plan(plan, output_dir / "processing_plan.json")
        return plan

    def run(
        self,
        source,
        output_dir: str | Path,
        *,
        plan_only: bool = False,
        skip_existing: bool = True,
    ) -> AdaptiveRunResult:
        t0 = time.perf_counter()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_file = output_dir / "processing_plan.json"
        plan = self.plan(source, output_dir)

        if plan_only:
            return AdaptiveRunResult(
                output_dir=output_dir,
                plan_file=plan_file,
                processed_domains=0,
                skipped_domains=0,
                failed_domains=0,
                total_seconds=time.perf_counter() - t0,
                execution_mode="plan-only",
            )

        resources_data = json.loads(
            (output_dir / "resources.json").read_text(encoding="utf-8")
        )
        auto_workers = int(resources_data["plan"]["workers"])
        auto_numba = int(resources_data["plan"]["numba_threads"])
        execution_memory_budget = int(resources_data["plan"]["memory_budget_bytes"])
        effective_source = self._effective_source(source, output_dir)

        out_path = output_dir / "slope_units.tif"
        work_path = output_dir / "work"
        if skip_existing and out_path.exists():
            return AdaptiveRunResult(
                output_dir=output_dir,
                plan_file=plan_file,
                processed_domains=0,
                skipped_domains=1,
                failed_domains=0,
                total_seconds=time.perf_counter() - t0,
                execution_mode=f"{plan.mode}-existing",
            )

        if plan.mode == "direct":
            fine_dem = effective_source
            hierarchy_mode = str(self.slopeunit_kwargs.get("hierarchy_mode", "lazy"))
            execution_mode = f"direct-global-{hierarchy_mode}"
        else:
            fine_dem = materialize_target_dem(
                effective_source,
                output_dir / "_work" / "canonical_fine_dem.tif",
                target_crs=self.target_crs,
                fine_resolution_m=self.fine_resolution_m,
                verbose=self.verbose,
            )
            hierarchy_mode = str(self.slopeunit_kwargs.get("hierarchy_mode", "lazy"))
            if plan.mode == "hydrological":
                execution_mode = f"adaptive-global-ooc-{hierarchy_mode}"
                if self.verbose:
                    print(
                        "[PySlopeUnits adaptive] large-dataset exact mode | "
                        f"planned hydrological domains={len(plan.domains):,} | "
                        "global out-of-core hydrology | "
                        f"{hierarchy_mode} hierarchical graph"
                    )
            else:
                execution_mode = f"single-global-{hierarchy_mode}"

        model = SlopeUnits(
            workers=auto_workers,
            numba_threads=auto_numba,
            memory_budget_bytes=execution_memory_budget,
            verbose=self.verbose,
            **self.slopeunit_kwargs,
        )
        model.run(fine_dem, out_path, work_dir=work_path)

        result = AdaptiveRunResult(
            output_dir=output_dir,
            plan_file=plan_file,
            processed_domains=(len(plan.domains) if plan.mode == "hydrological" else 1),
            skipped_domains=0,
            failed_domains=0,
            total_seconds=time.perf_counter() - t0,
            execution_mode=execution_mode,
        )
        (output_dir / "run_report.json").write_text(
            json.dumps(
                {
                    **asdict(result),
                    "output_dir": str(result.output_dir),
                    "plan_file": str(result.plan_file),
                    "mode": plan.mode,
                    "planned_domains": len(plan.domains),
                    "scientific_execution": (
                        "exact global out-of-core hydrology; hydrological domains are "
                        "resource-planning units and never define slope-unit boundaries; "
                        "hierarchy is evaluated lazily unless materialized mode is requested"
                    ),
                    "hierarchy_mode": str(self.slopeunit_kwargs.get("hierarchy_mode", "lazy")),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
