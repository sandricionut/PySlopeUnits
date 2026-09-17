from __future__ import annotations

from .logging_utils import log as print

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import numpy as np

from .candidate_cache import MultiThresholdCandidateCache, CandidateKey
from .clump import clump_equal_categories
from .dag import CandidateDAG
from .memmap_store import MemmapStore
from .schedule import threshold_schedule


@dataclass(frozen=True)
class GraphCreateResult:
    selected_nodes: int
    preclump_categories: int
    final_units: int
    seconds: float


def graph_cut(
    dag: CandidateDAG,
    *,
    cell_area_m2: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None = None,
    verbose: bool = True,
):
    arrays = dag.load()
    parent = arrays["parent"]
    level = arrays["level"]
    count = arrays["count"]
    acount = arrays["aspect_count"]
    ss = arrays["sum_sin"]
    cc = arrays["sum_cos"]
    offsets = arrays["child_offsets"]
    children = arrays["children"]

    n_nodes = int(parent.size - 1)
    max_level = int(level.max()) if n_nodes > 0 else 0
    selected = np.zeros(n_nodes + 1, dtype=np.uint8)
    active = np.array([1], dtype=np.int32)

    for current_level in range(1, max_level + 1):
        next_active: list[int] = []
        rejected = 0
        finalized = 0
        split = 0

        for pid0 in active:
            pid = int(pid0)
            lo = int(offsets[pid])
            hi = int(offsets[pid + 1])
            if hi <= lo:
                # No current half-basin under this active parent. GRASS leaves
                # this area as a hole to be completed by the final last-HB fill.
                continue

            # Only direct children at this threshold belong here.  The DAG is
            # level-ordered by construction, but keep the check defensive.
            ch = children[lo:hi]
            if ch.size and int(level[int(ch[0])]) != current_level:
                ch = ch[level[ch] == current_level]
            diversity = int(ch.size)
            if diversity == 0:
                continue

            avg_child_area = float(count[pid]) * cell_area_m2 / diversity
            if avg_child_area < min_area_m2:
                selected[pid] = 1
                rejected += 1
                continue

            for child0 in ch:
                child = int(child0)
                area = float(count[child]) * cell_area_m2
                if acount[child] > 0:
                    cv = 1.0 - float(np.hypot(ss[child], cc[child])) / float(acount[child])
                else:
                    cv = 0.0

                wants_split = area > min_area_m2 and cv > cv_min
                if max_area_m2 is not None and area > float(max_area_m2):
                    wants_split = True

                if wants_split and current_level < max_level:
                    next_active.append(child)
                    split += 1
                else:
                    selected[child] = 1
                    finalized += 1

        if verbose:
            print(
                f"[PySlopeUnits] graph cut level {current_level:02d} | "
                f"active={active.size:,} | rejected={rejected:,} | "
                f"finalized={finalized:,} | split={split:,}"
            )

        if not next_active:
            break
        active = np.asarray(next_active, dtype=np.int32)

    selected_ids = np.flatnonzero(selected > 0).astype(np.int32)
    own_cat = np.zeros(n_nodes + 1, dtype=np.int32)
    own_cat[selected_ids] = np.arange(1, selected_ids.size + 1, dtype=np.int32)

    # Every terminal node inherits the first selected ancestor. Parents always
    # have lower IDs than children.
    node_category = np.zeros(n_nodes + 1, dtype=np.int32)
    for nid in range(1, n_nodes + 1):
        own = int(own_cat[nid])
        if own > 0:
            node_category[nid] = own
        else:
            pid = int(parent[nid])
            node_category[nid] = node_category[pid] if pid > 0 else 0

    return selected_ids, node_category


def render_and_clump(
    store: MemmapStore,
    candidate_cache: MultiThresholdCandidateCache,
    dag: CandidateDAG,
    node_category: np.ndarray,
    *,
    threshold_m2: float,
    cell_area_m2: float,
    reduction_factor: int,
    max_iterations: int,
    block_rows: int = 1024,
    verbose: bool = True,
):
    valid = store.open("valid", "r")
    terminal = store.open("v14_terminal_node", "r")
    shape = valid.shape
    schedule = threshold_schedule(threshold_m2, cell_area_m2, reduction_factor, max_iterations)
    last_key = CandidateKey(schedule[-1].cells)
    if not candidate_cache.ensure(last_key, verbose=verbose):
        raise RuntimeError("last candidate cache missing")
    last_hb = np.load(candidate_cache.raster_path(last_key), mmap_mode="r", allow_pickle=False)
    last_stats = candidate_cache.load_stats(last_key)
    max_hb = int(last_stats["max_half_basin_label"])

    pre = store.create_temp("v14_preclump", shape, np.int32, fill=0)
    selected_max = int(node_category.max())
    hb_offset = selected_max + 1
    residual_category = hb_offset + max_hb + 1

    rows = shape[0]
    gaps_from_last = 0
    residual_cells = 0
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        vm = np.asarray(valid[r0:r1], dtype=bool)
        tn = np.asarray(terminal[r0:r1])
        out = np.zeros(tn.shape, dtype=np.int32)
        good_tn = vm & (tn > 0)
        out[good_tn] = node_category[tn[good_tn]]

        gap = vm & (out == 0)
        hb = np.asarray(last_hb[r0:r1])
        fill = gap & (hb > 0)
        if np.any(fill):
            out[fill] = hb_offset + hb[fill].astype(np.int32)
            gaps_from_last += int(fill.sum())

        residual = vm & (out == 0)
        if np.any(residual):
            out[residual] = residual_category
            residual_cells += int(residual.sum())
        pre[r0:r1] = out
    pre.flush()

    if verbose:
        print(
            f"[PySlopeUnits] final fill | from-last-halfbasin={gaps_from_last:,} | "
            f"residual-placeholder={residual_cells:,}"
        )
        print("[PySlopeUnits] GRASS-like 4-neighbour categorical clump")

    final, n_units = clump_equal_categories(pre, valid, store, verbose=verbose)
    store.close_array(pre)
    del pre
    store.remove("v14_preclump", best_effort=True)
    pre_categories = int(residual_category)
    return final, int(n_units), pre_categories


def create_from_dag(
    store: MemmapStore,
    candidate_cache: MultiThresholdCandidateCache,
    dag: CandidateDAG,
    *,
    threshold_m2: float,
    cell_area_m2: float,
    min_area_m2: float,
    cv_min: float,
    reduction_factor: int,
    max_iterations: int,
    max_area_m2: float | None = None,
    verbose: bool = True,
) -> GraphCreateResult:
    t0 = time.perf_counter()
    selected, node_category = graph_cut(
        dag,
        cell_area_m2=cell_area_m2,
        min_area_m2=min_area_m2,
        cv_min=cv_min,
        max_area_m2=max_area_m2,
        verbose=verbose,
    )
    final, n_units, pre_categories = render_and_clump(
        store, candidate_cache, dag, node_category,
        threshold_m2=threshold_m2,
        cell_area_m2=cell_area_m2,
        reduction_factor=reduction_factor,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    result = GraphCreateResult(
        selected_nodes=int(selected.size),
        preclump_categories=int(pre_categories),
        final_units=int(n_units),
        seconds=time.perf_counter() - t0,
    )
    store.write_json("v14_graph_create_report.json", asdict(result))
    return result
