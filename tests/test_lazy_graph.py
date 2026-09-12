import json
from pathlib import Path

import numpy as np

from pyslopeunits.candidate_cache import (
    CANDIDATE_SEMANTICS,
    CandidateKey,
    MultiThresholdCandidateCache,
)
from pyslopeunits.dag import CandidateDAG
from pyslopeunits.graph_create import create_from_dag
from pyslopeunits.lazy_graph import create_lazy_hierarchy
from pyslopeunits.memmap_store import MemmapStore


def _save_candidate(cache, cells, arr):
    key = CandidateKey(cells)
    np.save(cache.raster_path(key), np.asarray(arr, dtype=np.int32), allow_pickle=False)
    stats = {
        "candidate_semantics": CANDIDATE_SEMANTICS,
        "max_half_basin_label": int(np.max(arr)),
        "stream_branch_roots": int(np.max(arr) // 2),
        "residual_outlets": 0,
        "residual_mode": "final-fill-only",
    }
    cache.stats_path(key).write_text(json.dumps(stats), encoding="utf-8")


def test_lazy_graph_matches_materialized_dag(tmp_path: Path):
    work = tmp_path / "work"
    store = MemmapStore(work / "memmap")
    shape = (6, 8)

    valid = store.create("valid", shape, np.uint8, fill=1)
    sin_a = store.create("sin_aspect", shape, np.float32, fill=0.0)
    cos_a = store.create("cos_aspect", shape, np.float32, fill=1.0)
    aspect_valid = store.create("aspect_valid", shape, np.uint8, fill=1)
    store.close_many(valid, sin_a, cos_a, aspect_valid)

    # Two intentionally crossing levels: the exact parent/HB intersections
    # matter, so this tests the hierarchy rather than simple nested labels.
    hb8 = np.array([
        [1,1,1,1,2,2,2,2],
        [1,1,1,1,2,2,2,2],
        [3,3,3,3,4,4,4,4],
        [3,3,3,3,4,4,4,4],
        [5,5,5,5,6,6,6,6],
        [5,5,5,5,6,6,6,6],
    ], dtype=np.int32)
    hb4 = np.array([
        [1,1,2,2,3,3,4,4],
        [1,1,2,2,3,3,4,4],
        [5,5,6,6,7,7,8,8],
        [5,5,6,6,7,7,8,8],
        [9,9,10,10,11,11,12,12],
        [9,9,10,10,11,11,12,12],
    ], dtype=np.int32)

    cache = MultiThresholdCandidateCache(work)
    _save_candidate(cache, 8, hb8)
    _save_candidate(cache, 4, hb4)

    dag = CandidateDAG(work)
    dag.build(
        store, cache,
        threshold_m2=8.0,
        cell_area_m2=1.0,
        reduction_factor=2,
        max_iterations=2,
        verbose=False,
    )
    full = create_from_dag(
        store, cache, dag,
        threshold_m2=8.0,
        cell_area_m2=1.0,
        min_area_m2=3.0,
        cv_min=0.1,
        reduction_factor=2,
        max_iterations=2,
        verbose=False,
    )
    full_arr = np.asarray(store.open("final", "r")).copy()
    store.remove("final")

    lazy = create_lazy_hierarchy(
        store, cache,
        threshold_m2=8.0,
        cell_area_m2=1.0,
        min_area_m2=3.0,
        cv_min=0.1,
        reduction_factor=2,
        max_iterations=2,
        verbose=False,
    )
    lazy_arr = np.asarray(store.open("final", "r")).copy()

    assert full.final_units == lazy.final_units
    assert np.array_equal(full_arr, lazy_arr)
