from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import os
import platform
import shutil

import psutil


@dataclass(frozen=True)
class SystemResources:
    platform: str
    machine: str
    logical_cpus: int
    physical_cpus: int
    total_ram_bytes: int
    available_ram_bytes: int
    disk_free_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResourcePlan:
    memory_budget_bytes: int
    workers: int
    numba_threads: int
    target_fine_cells: int
    bytes_per_fine_cell: int
    coarse_resolution_m: float
    coarse_cell_budget: int
    coarse_bytes_per_cell: int
    safety_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def detect_system_resources(work_dir: str | Path = ".") -> SystemResources:
    """Detect cross-platform CPU, RAM and free-disk resources."""
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(work_dir)
    logical = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    physical = int(psutil.cpu_count(logical=False) or logical)
    return SystemResources(
        platform=platform.system(),
        machine=platform.machine(),
        logical_cpus=max(1, logical),
        physical_cpus=max(1, physical),
        total_ram_bytes=int(vm.total),
        available_ram_bytes=int(vm.available),
        disk_free_bytes=int(disk.free),
    )


def choose_coarse_resolution(
    *,
    target_width_m: float,
    target_height_m: float,
    fine_resolution_m: float,
    coarse_cell_budget: int,
    minimum_factor: int = 4,
    preferred_resolutions_m: tuple[float, ...] = (
        120.0, 180.0, 240.0, 300.0, 360.0, 480.0,
        600.0, 750.0, 900.0, 1200.0, 1500.0, 2000.0,
    ),
) -> float:
    """Choose the finest coarse resolution that fits the coarse-cell budget."""
    if fine_resolution_m <= 0:
        raise ValueError("fine_resolution_m must be positive")
    if target_width_m <= 0 or target_height_m <= 0:
        raise ValueError("target dimensions must be positive")
    if coarse_cell_budget <= 0:
        raise ValueError("coarse_cell_budget must be positive")

    minimum = float(fine_resolution_m) * max(2, int(minimum_factor))
    candidates = [r for r in preferred_resolutions_m if r >= minimum]
    if not candidates:
        candidates = [minimum]

    for res in candidates:
        cells = math.ceil(target_width_m / res) * math.ceil(target_height_m / res)
        if cells <= coarse_cell_budget:
            return float(res)

    required = math.sqrt((target_width_m * target_height_m) / coarse_cell_budget)
    return float(max(minimum, math.ceil(required / 10.0) * 10.0))


def make_resource_plan(
    resources: SystemResources,
    *,
    target_width_m: float,
    target_height_m: float,
    fine_resolution_m: float,
    memory_limit_bytes: int | None = None,
    memory_fraction: float = 0.65,
    coarse_memory_fraction: float = 0.18,
    bytes_per_fine_cell: int = 112,
    coarse_bytes_per_cell: int = 72,
    workers: int | None = None,
    numba_threads: int | None = None,
    coarse_resolution_m: float | None = None,
) -> ResourcePlan:
    """Create a machine-capacity execution plan.

    Planning uses installed RAM rather than momentary ``available`` RAM so that
    the same 24/32/64 GB machine produces a stable domain plan independent of
    transient desktop memory pressure. ``memory_limit_bytes`` can cap this.

    ``bytes_per_fine_cell`` is an empirical peak-pressure parameter and can be
    calibrated from benchmarks without changing the hydrological algorithm.
    """
    if not (0.1 <= memory_fraction <= 0.95):
        raise ValueError("memory_fraction must be between 0.1 and 0.95")
    if not (0.05 <= coarse_memory_fraction <= 0.5):
        raise ValueError("coarse_memory_fraction must be between 0.05 and 0.5")

    capacity = int(resources.total_ram_bytes)
    if memory_limit_bytes is not None:
        capacity = min(capacity, int(memory_limit_bytes))

    memory_budget = max(512 * 1024**2, int(capacity * memory_fraction))
    target_fine_cells = max(
        1_000_000,
        memory_budget // max(1, int(bytes_per_fine_cell)),
    )

    coarse_budget_bytes = max(
        256 * 1024**2,
        int(capacity * coarse_memory_fraction),
    )
    coarse_cell_budget = max(
        1_000_000,
        coarse_budget_bytes // max(1, int(coarse_bytes_per_cell)),
    )

    if coarse_resolution_m is None:
        coarse_resolution_m = choose_coarse_resolution(
            target_width_m=target_width_m,
            target_height_m=target_height_m,
            fine_resolution_m=fine_resolution_m,
            coarse_cell_budget=coarse_cell_budget,
        )

    auto_workers = max(1, min(resources.physical_cpus, resources.logical_cpus))
    workers = auto_workers if workers is None or int(workers) <= 0 else int(workers)
    numba_threads = (
        workers if numba_threads is None or int(numba_threads) <= 0
        else int(numba_threads)
    )

    return ResourcePlan(
        memory_budget_bytes=int(memory_budget),
        workers=max(1, workers),
        numba_threads=max(1, numba_threads),
        target_fine_cells=int(target_fine_cells),
        bytes_per_fine_cell=int(bytes_per_fine_cell),
        coarse_resolution_m=float(coarse_resolution_m),
        coarse_cell_budget=int(coarse_cell_budget),
        coarse_bytes_per_cell=int(coarse_bytes_per_cell),
        safety_fraction=float(memory_fraction),
    )
