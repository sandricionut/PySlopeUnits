from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .raster import read_meta
from .schedule import threshold_schedule


@dataclass(frozen=True)
class StorageEstimate:
    resolution_m: float
    cells: int
    candidate_levels: int
    hydrology_gb: float
    candidates_gb: float
    iterative_state_gb: float
    estimated_total_gb: float


def estimate_work_storage(
    dem_path: str | Path,
    *,
    threshold_m2: float = 250_000.0,
    reduction_factor: int = 2,
    max_iterations: int = 12,
) -> StorageEstimate:
    meta = read_meta(dem_path)
    cells = int(meta.shape[0]) * int(meta.shape[1])
    levels = threshold_schedule(
        threshold_m2, meta.cell_area, reduction_factor, max_iterations
    )

    # Approximate persistent optimized hydrology:
    # valid u1 + receiver i4 + accumulation f8 + order i4 (valid fraction
    # conservatively taken as all cells) + sin/cos f4+f4 + aspectvalid u1 +
    # static_ok u1 + up_acc f4 + drainage_component i4.
    hydro_bpc = 1 + 4 + 8 + 4 + 4 + 4 + 1 + 1 + 4 + 4

    # One int32 half-basin raster per candidate threshold.
    candidate_bpc = 4 * len(levels)

    # stream u1 + todo i4 + todo_next i4 + final i4 + active hardlink does
    # not duplicate candidate bytes.
    state_bpc = 1 + 4 + 4 + 4

    gb = 1024.0 ** 3
    hydro = cells * hydro_bpc / gb
    candidates = cells * candidate_bpc / gb
    state = cells * state_bpc / gb

    return StorageEstimate(
        resolution_m=float(meta.xres),
        cells=cells,
        candidate_levels=len(levels),
        hydrology_gb=hydro,
        candidates_gb=candidates,
        iterative_state_gb=state,
        estimated_total_gb=hydro + candidates + state,
    )
