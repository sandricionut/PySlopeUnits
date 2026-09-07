from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np

from .kernels import dense_parent_stats
from .memmap_store import MemmapStore


@dataclass
class ParentDecision:
    parent_id: int
    reject: bool
    child_ids: np.ndarray
    split: np.ndarray
    parent_cells: int


def _decision_from_stats(
    parent_id: int,
    child_ids: np.ndarray,
    count: np.ndarray,
    aspect_count: np.ndarray,
    sum_sin: np.ndarray,
    sum_cos: np.ndarray,
    parent_cells: int,
    cell_area: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None,
) -> ParentDecision:
    diversity = int(child_ids.size)
    if diversity == 0 or parent_cells == 0:
        return ParentDecision(parent_id, True, child_ids.astype(np.int32), np.zeros(diversity, bool), parent_cells)

    # r.slopeunits-style parent partition rejection.
    avg_child_area = parent_cells * float(cell_area) / diversity
    if avg_child_area < min_area_m2:
        return ParentDecision(parent_id, True, child_ids.astype(np.int32), np.zeros(diversity, bool), parent_cells)

    area = count.astype(np.float64, copy=False) * float(cell_area)
    cv = np.zeros(diversity, dtype=np.float64)
    ok = aspect_count > 0
    cv[ok] = 1.0 - np.hypot(sum_sin[ok], sum_cos[ok]) / aspect_count[ok]

    split = (area > min_area_m2) & (cv > cv_min)
    if max_area_m2 is not None:
        split |= area > float(max_area_m2)

    return ParentDecision(
        parent_id=int(parent_id),
        reject=False,
        child_ids=child_ids.astype(np.int32, copy=False),
        split=split.astype(bool, copy=False),
        parent_cells=int(parent_cells),
    )


def _evaluate_parent_arrays(
    todo,
    hb,
    sin_a,
    cos_a,
    aspect_valid,
    parent_id: int,
    bbox: tuple[int, int, int, int],
    cell_area: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None,
    max_child_label: int,
    dense_threshold_cells: int,
    known_parent_cells: int | None,
) -> ParentDecision:
    r0, r1, c0, c1 = bbox
    parent_cells = int(known_parent_cells) if known_parent_cells is not None else -1

    # Large parents: Numba scans the memmaps directly and allocates only
    # per-label statistics. No full boolean-indexed raster copy is created.
    if parent_cells >= int(dense_threshold_cells):
        counts, acount, ss, cc, scanned = dense_parent_stats(
            todo, hb, sin_a, cos_a, aspect_valid,
            int(parent_id), int(r0), int(r1), int(c0), int(c1), int(max_child_label),
        )
        child_ids = np.flatnonzero(counts > 0).astype(np.int32)
        return _decision_from_stats(
            parent_id,
            child_ids,
            counts[child_ids],
            acount[child_ids].astype(np.float64),
            ss[child_ids],
            cc[child_ids],
            int(scanned),
            cell_area,
            min_area_m2,
            cv_min,
            max_area_m2,
        )

    # Small/medium parent: vectorized NumPy over only its bounding window.
    tw = np.asarray(todo[r0:r1, c0:c1])
    mask = tw == parent_id
    if parent_cells < 0:
        parent_cells = int(mask.sum())
    if parent_cells == 0:
        return ParentDecision(parent_id, True, np.empty(0, np.int32), np.empty(0, bool), 0)

    children = np.asarray(hb[r0:r1, c0:c1])[mask].astype(np.int32, copy=False)
    child_ids, inv = np.unique(children, return_inverse=True)
    if np.any(child_ids <= 0):
        keep_cell = children > 0
        children = children[keep_cell]
        child_ids, inv = np.unique(children, return_inverse=True)
        parent_cells = int(children.size)
    else:
        keep_cell = None

    diversity = int(child_ids.size)
    if diversity == 0:
        return ParentDecision(parent_id, True, child_ids.astype(np.int32), np.zeros(0, bool), parent_cells)

    count = np.bincount(inv, minlength=diversity).astype(np.float64)
    av = np.asarray(aspect_valid[r0:r1, c0:c1])[mask]
    ss0 = np.asarray(sin_a[r0:r1, c0:c1])[mask]
    cc0 = np.asarray(cos_a[r0:r1, c0:c1])[mask]
    if keep_cell is not None:
        av = av[keep_cell]
        ss0 = ss0[keep_cell]
        cc0 = cc0[keep_cell]

    aspect_count = np.bincount(inv, weights=av.astype(np.float64, copy=False), minlength=diversity)
    sum_sin = np.bincount(inv, weights=ss0.astype(np.float64, copy=False), minlength=diversity)
    sum_cos = np.bincount(inv, weights=cc0.astype(np.float64, copy=False), minlength=diversity)

    return _decision_from_stats(
        parent_id,
        child_ids,
        count,
        aspect_count,
        sum_sin,
        sum_cos,
        parent_cells,
        cell_area,
        min_area_m2,
        cv_min,
        max_area_m2,
    )


def evaluate_parent(
    store_dir: str,
    parent_id: int,
    bbox: tuple[int, int, int, int],
    cell_area: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None,
    max_child_label: int,
    dense_threshold_cells: int = 5_000_000,
    known_parent_cells: int | None = None,
) -> ParentDecision:
    store = MemmapStore(store_dir)
    return _evaluate_parent_arrays(
        store.open("todo", "r"),
        store.open("half_basins", "r"),
        store.open("sin_aspect", "r"),
        store.open("cos_aspect", "r"),
        store.open("aspect_valid", "r"),
        parent_id,
        bbox,
        cell_area,
        min_area_m2,
        cv_min,
        max_area_m2,
        max_child_label,
        dense_threshold_cells,
        known_parent_cells,
    )


def evaluate_parent_batch(
    store_dir: str,
    jobs: list[tuple[int, tuple[int, int, int, int], int]],
    cell_area: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None,
    max_child_label: int,
    dense_threshold_cells: int,
):
    """One process opens each shared memmap once, then evaluates its batch."""
    store = MemmapStore(store_dir)
    todo = store.open("todo", "r")
    hb = store.open("half_basins", "r")
    sin_a = store.open("sin_aspect", "r")
    cos_a = store.open("cos_aspect", "r")
    aspect_valid = store.open("aspect_valid", "r")

    results = []
    for pid, bbox, parent_cells in jobs:
        results.append(_evaluate_parent_arrays(
            todo, hb, sin_a, cos_a, aspect_valid,
            pid, bbox, cell_area, min_area_m2, cv_min,
            max_area_m2, max_child_label, dense_threshold_cells, parent_cells,
        ))
    return os.getpid(), results
