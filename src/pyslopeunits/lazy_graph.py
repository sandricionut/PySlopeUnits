from __future__ import annotations

"""Bounded-memory lazy evaluation of the PySlopeUnits candidate hierarchy.

The scientific hierarchy is identical to :mod:`pyslopeunits.dag`, but only
nodes that are actually reached by the parameter-dependent graph cut are
materialized.  Candidate intersections are reduced in hash buckets on disk,
which removes the O(total DAG nodes) Python-list/dict memory requirement.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import json
import math
import shutil
import time

import numpy as np
from numba import njit

from .candidate_cache import CandidateKey, MultiThresholdCandidateCache
from .clump import clump_equal_categories
from .memmap_store import MemmapStore
from .schedule import threshold_schedule


# One record is one block-local contribution to one (parent, half-basin) child.
_STAT_DTYPE = np.dtype([
    ("key", "<u8"),
    ("count", "<i8"),
    ("aspect_count", "<i8"),
    ("sum_sin", "<f8"),
    ("sum_cos", "<f8"),
])

# Final lookup used during the rendering pass for a hierarchy level.
_MAP_DTYPE = np.dtype([
    ("key", "<u8"),
    ("next_id", "<i8"),
    ("category", "<i8"),
])


@dataclass(frozen=True)
class LazyHierarchyResult:
    selected_nodes: int
    visited_nodes: int
    preclump_categories: int
    final_units: int
    levels_processed: int
    seconds: float


def _count_nonzero_blockwise(arr, block_rows: int = 1024) -> int:
    total = 0
    rows = int(arr.shape[0])
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        total += int(np.count_nonzero(np.asarray(arr[r0:r1])))
    return total


def _choose_bucket_count(max_half_basin_label: int) -> int:
    """Choose a bounded number of external-reduction buckets.

    The label maximum is approximately twice the number of stream roots.  The
    selected values are primes to avoid systematic aliasing with even/odd HAF
    labels.  This keeps individual reduction files moderate without creating
    tens of thousands of files for very-large/global runs.
    """
    estimate = max(1, int(max_half_basin_label))
    if estimate <= 250_000:
        return 31
    if estimate <= 1_000_000:
        return 127
    if estimate <= 5_000_000:
        return 509
    if estimate <= 25_000_000:
        return 2039
    return 8191


def _append_records_by_bucket(records: np.ndarray, root: Path, nbuckets: int) -> None:
    if records.size == 0:
        return
    bids = np.remainder(records["key"], np.uint64(nbuckets)).astype(np.int32)
    order = np.argsort(bids, kind="stable")
    bids_sorted = bids[order]
    rec_sorted = records[order]
    starts = np.flatnonzero(
        np.r_[True, bids_sorted[1:] != bids_sorted[:-1]]
    )
    ends = np.r_[starts[1:], bids_sorted.size]
    for s, e in zip(starts, ends):
        b = int(bids_sorted[s])
        with (root / f"bucket_{b:05d}.bin").open("ab") as fh:
            rec_sorted[s:e].tofile(fh)


@njit(cache=True)
def _reduce_sorted_records(keys, counts, acounts, sums_sin, sums_cos):
    n = keys.size
    if n == 0:
        return (
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    out_k = np.empty(n, dtype=np.uint64)
    out_c = np.empty(n, dtype=np.int64)
    out_a = np.empty(n, dtype=np.int64)
    out_s = np.empty(n, dtype=np.float64)
    out_x = np.empty(n, dtype=np.float64)

    q = 0
    key = keys[0]
    c = 0
    a = 0
    ss = 0.0
    cc = 0.0

    for i in range(n):
        k = keys[i]
        if k != key:
            out_k[q] = key
            out_c[q] = c
            out_a[q] = a
            out_s[q] = ss
            out_x[q] = cc
            q += 1
            key = k
            c = 0
            a = 0
            ss = 0.0
            cc = 0.0
        c += counts[i]
        a += acounts[i]
        ss += sums_sin[i]
        cc += sums_cos[i]

    out_k[q] = key
    out_c[q] = c
    out_a[q] = a
    out_s[q] = ss
    out_x[q] = cc
    q += 1

    return out_k[:q], out_c[:q], out_a[:q], out_s[:q], out_x[:q]


def _reduce_bucket_file(path: Path) -> np.ndarray:
    """Stable exact reduction of block-local statistics for equal keys."""
    rec = np.fromfile(path, dtype=_STAT_DTYPE)
    if rec.size == 0:
        return rec
    order = np.argsort(rec["key"], kind="stable")
    rec = rec[order]
    k, c, a, ss, cc = _reduce_sorted_records(
        rec["key"], rec["count"], rec["aspect_count"],
        rec["sum_sin"], rec["sum_cos"],
    )
    out = np.empty(k.size, dtype=_STAT_DTYPE)
    out["key"] = k
    out["count"] = c
    out["aspect_count"] = a
    out["sum_sin"] = ss
    out["sum_cos"] = cc
    return out


def _aggregate_children(
    *,
    level_root: Path,
    active,
    root_mode: bool,
    valid,
    hb,
    aspect_valid,
    sin_a,
    cos_a,
    n_active: int,
    base: int,
    nbuckets: int,
    block_rows: int,
) -> tuple[np.ndarray, list[Path], int]:
    """Build external child statistics and parent diversity counts."""
    raw_root = level_root / "raw"
    reduced_root = level_root / "reduced"
    shutil.rmtree(level_root, ignore_errors=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    reduced_root.mkdir(parents=True, exist_ok=True)

    rows = int(valid.shape[0])
    total_local_records = 0

    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        vm = np.asarray(valid[r0:r1], dtype=bool)
        if root_mode:
            pblock = vm.astype(np.int64, copy=False)
        else:
            pblock = np.asarray(active[r0:r1], dtype=np.int64)
        hblock = np.asarray(hb[r0:r1], dtype=np.int64)
        positive = (pblock > 0) & vm & (hblock > 0)
        if not np.any(positive):
            continue

        pv = pblock[positive].astype(np.uint64, copy=False)
        hv = hblock[positive].astype(np.uint64, copy=False)
        # Generic 64-bit pair encoding.  Unlike the old <<32 packing this does
        # not impose a 32-bit half-basin label limit.
        keys = pv * np.uint64(base) + hv
        uniq, inv = np.unique(keys, return_inverse=True)

        bcount = np.bincount(inv, minlength=uniq.size).astype(np.int64)
        av0 = np.asarray(aspect_valid[r0:r1])[positive].astype(np.float64, copy=False)
        sin0 = np.asarray(sin_a[r0:r1])[positive].astype(np.float64, copy=False)
        cos0 = np.asarray(cos_a[r0:r1])[positive].astype(np.float64, copy=False)
        bac = np.bincount(inv, weights=av0, minlength=uniq.size)
        bss = np.bincount(inv, weights=sin0 * av0, minlength=uniq.size)
        bcc = np.bincount(inv, weights=cos0 * av0, minlength=uniq.size)

        rec = np.empty(uniq.size, dtype=_STAT_DTYPE)
        rec["key"] = uniq
        rec["count"] = bcount
        rec["aspect_count"] = np.rint(bac).astype(np.int64)
        rec["sum_sin"] = bss
        rec["sum_cos"] = bcc
        _append_records_by_bucket(rec, raw_root, nbuckets)
        total_local_records += int(rec.size)

    diversity = np.zeros(n_active + 1, dtype=np.int64)
    reduced_paths: list[Path] = []
    visited_children = 0

    for raw_path in sorted(raw_root.glob("bucket_*.bin")):
        reduced = _reduce_bucket_file(raw_path)
        if reduced.size == 0:
            continue
        parents = (reduced["key"] // np.uint64(base)).astype(np.int64)
        # Each reduced record is exactly one distinct direct child.
        np.add.at(diversity, parents, 1)
        out_path = reduced_root / (raw_path.stem + ".npy")
        np.save(out_path, reduced, allow_pickle=False)
        reduced_paths.append(out_path)
        visited_children += int(reduced.size)

    shutil.rmtree(raw_root, ignore_errors=True)
    return diversity, reduced_paths, visited_children


def _prepare_level_maps(
    *,
    reduced_paths: list[Path],
    level_root: Path,
    base: int,
    parent_count: np.ndarray,
    diversity: np.ndarray,
    cell_area_m2: float,
    min_area_m2: float,
    cv_min: float,
    max_area_m2: float | None,
    current_level: int,
    max_level: int,
    category_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list[Path], int, int, int]:
    """Apply graph-cut decisions and create disk lookup maps."""
    n_active = int(parent_count.size - 1)
    rejected = np.zeros(n_active + 1, dtype=np.uint8)
    parent_category = np.zeros(n_active + 1, dtype=np.int64)

    valid_parent = diversity > 0
    pids = np.flatnonzero(valid_parent)
    if pids.size:
        avg = (
            parent_count[pids].astype(np.float64)
            * float(cell_area_m2)
            / diversity[pids].astype(np.float64)
        )
        rej_ids = pids[avg < float(min_area_m2)]
    else:
        rej_ids = np.empty(0, dtype=np.int64)

    rejected[rej_ids] = 1
    next_category = int(category_start)
    if rej_ids.size:
        cats = np.arange(
            next_category,
            next_category + int(rej_ids.size),
            dtype=np.int64,
        )
        parent_category[rej_ids] = cats
        next_category += int(rej_ids.size)

    # Pass 1: count children that split versus finalize.
    total_split = 0
    total_final = 0
    for path in reduced_paths:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        parents = (arr["key"] // np.uint64(base)).astype(np.int64)
        allowed = rejected[parents] == 0
        ac = arr["aspect_count"].astype(np.float64)
        cv = np.zeros(arr.size, dtype=np.float64)
        nz = ac > 0
        cv[nz] = 1.0 - np.hypot(arr["sum_sin"][nz], arr["sum_cos"][nz]) / ac[nz]
        area = arr["count"].astype(np.float64) * float(cell_area_m2)
        wants = (area > float(min_area_m2)) & (cv > float(cv_min))
        if max_area_m2 is not None:
            wants |= area > float(max_area_m2)
        split = allowed & wants & (current_level < max_level)
        final = allowed & ~split
        total_split += int(np.count_nonzero(split))
        total_final += int(np.count_nonzero(final))

    next_parent_count = np.zeros(total_split + 1, dtype=np.int64)
    map_root = level_root / "maps"
    map_root.mkdir(parents=True, exist_ok=True)
    map_paths: list[Path] = []
    next_id = 1
    finalized_children = 0

    # Pass 2: deterministic IDs/categories and compact lookup files.
    for path in reduced_paths:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        parents = (arr["key"] // np.uint64(base)).astype(np.int64)
        allowed = rejected[parents] == 0
        ac = arr["aspect_count"].astype(np.float64)
        cv = np.zeros(arr.size, dtype=np.float64)
        nz = ac > 0
        cv[nz] = 1.0 - np.hypot(arr["sum_sin"][nz], arr["sum_cos"][nz]) / ac[nz]
        area = arr["count"].astype(np.float64) * float(cell_area_m2)
        wants = (area > float(min_area_m2)) & (cv > float(cv_min))
        if max_area_m2 is not None:
            wants |= area > float(max_area_m2)
        split = allowed & wants & (current_level < max_level)
        final = allowed & ~split

        mapping = np.zeros(arr.size, dtype=_MAP_DTYPE)
        mapping["key"] = arr["key"]

        ns = int(np.count_nonzero(split))
        if ns:
            ids = np.arange(next_id, next_id + ns, dtype=np.int64)
            mapping["next_id"][split] = ids
            next_parent_count[ids] = arr["count"][split]
            next_id += ns

        nf = int(np.count_nonzero(final))
        if nf:
            cats = np.arange(next_category, next_category + nf, dtype=np.int64)
            mapping["category"][final] = cats
            next_category += nf
            finalized_children += nf

        out_path = map_root / path.name
        np.save(out_path, mapping, allow_pickle=False)
        map_paths.append(out_path)

    return (
        rejected,
        parent_category,
        next_parent_count,
        next_category,
        total_split,
        map_paths,
        int(rej_ids.size),
        finalized_children,
        int(np.count_nonzero(valid_parent)),
    )


def _render_level(
    *,
    active,
    root_mode: bool,
    active_next,
    pre,
    valid,
    hb,
    base: int,
    nbuckets: int,
    rejected: np.ndarray,
    parent_category: np.ndarray,
    map_paths: list[Path],
    block_rows: int,
) -> None:
    maps: dict[int, np.ndarray] = {}
    try:
        for path in map_paths:
            # bucket_00012.npy -> 12
            bid = int(path.stem.split("_")[-1])
            maps[bid] = np.load(path, mmap_mode="r", allow_pickle=False)

        rows = int(valid.shape[0])
        for r0 in range(0, rows, block_rows):
            r1 = min(rows, r0 + block_rows)
            vm = np.asarray(valid[r0:r1], dtype=bool)
            if root_mode:
                pblock = vm.astype(np.int64, copy=False)
            else:
                pblock = np.asarray(active[r0:r1], dtype=np.int64)
            hblock = np.asarray(hb[r0:r1], dtype=np.int64)
            out = np.asarray(pre[r0:r1], dtype=np.int64).copy()
            nxt = np.zeros(pblock.shape, dtype=np.int32)

            amask = (pblock > 0) & vm
            if not np.any(amask):
                active_next[r0:r1] = nxt
                continue

            # Parent rejection selects the entire parent, including cells where
            # the current half-basin raster is zero.
            ap = pblock[amask]
            rej = rejected[ap] != 0
            if np.any(rej):
                flat_idx = np.flatnonzero(amask)
                chosen = flat_idx[rej]
                out.ravel()[chosen] = parent_category[ap[rej]]

            positive = amask & (rejected[pblock] == 0) & (hblock > 0)
            if np.any(positive):
                flat_idx = np.flatnonzero(positive)
                pv = pblock[positive].astype(np.uint64, copy=False)
                hv = hblock[positive].astype(np.uint64, copy=False)
                keys = pv * np.uint64(base) + hv
                bids = np.remainder(keys, np.uint64(nbuckets)).astype(np.int32)

                for bid in np.unique(bids):
                    take = bids == bid
                    mapping = maps.get(int(bid))
                    if mapping is None:
                        raise RuntimeError(f"missing lazy hierarchy map bucket {int(bid)}")
                    qkeys = keys[take]
                    pos = np.searchsorted(mapping["key"], qkeys)
                    if np.any(pos >= mapping.size) or np.any(mapping["key"][pos] != qkeys):
                        raise RuntimeError("lazy hierarchy mapping lookup failed")
                    loc = flat_idx[take]
                    cats = mapping["category"][pos]
                    nids = mapping["next_id"][pos]
                    if np.any(cats > 0):
                        m = cats > 0
                        out.ravel()[loc[m]] = cats[m]
                    if np.any(nids > 0):
                        if int(nids.max()) > np.iinfo(np.int32).max:
                            raise OverflowError(
                                "active hierarchy frontier exceeds int32; "
                                "reduce processing scale or enable domain execution"
                            )
                        m = nids > 0
                        nxt.ravel()[loc[m]] = nids[m].astype(np.int32)

            pre[r0:r1] = out.astype(pre.dtype, copy=False)
            active_next[r0:r1] = nxt

        pre.flush()
        active_next.flush()
    finally:
        for arr in maps.values():
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()


def create_lazy_hierarchy(
    store: MemmapStore,
    candidate_cache: MultiThresholdCandidateCache | None,
    *,
    candidate_provider=None,
    threshold_m2: float,
    cell_area_m2: float,
    min_area_m2: float,
    cv_min: float,
    reduction_factor: int,
    max_iterations: int,
    max_area_m2: float | None = None,
    block_rows: int = 512,
    verbose: bool = True,
) -> LazyHierarchyResult:
    """Evaluate the hierarchical graph lazily with bounded memory.

    This is scientifically equivalent to building the complete intersection
    DAG and then cutting it, but descendants of finalized/rejected parents are
    never created.  The large global rasters remain memmapped and child
    statistics are externally reduced in bounded hash buckets.
    """
    t0 = time.perf_counter()
    valid = store.open("valid", "r")
    sin_a = store.open("sin_aspect", "r")
    cos_a = store.open("cos_aspect", "r")
    aspect_valid = store.open("aspect_valid", "r")
    shape = valid.shape
    rows = int(shape[0])

    schedule = threshold_schedule(
        threshold_m2, cell_area_m2, reduction_factor, max_iterations
    )
    max_level = len(schedule)
    valid_cells = _count_nonzero_blockwise(valid, block_rows=1024)

    # The selected-category raster is intentionally disk-backed: it survives
    # all hierarchy levels and may be far larger than RAM on very-large runs.
    pre = store.create("lazy_preclump", shape, np.int32, fill=0)
    active_a = store.create("lazy_active_a", shape, np.int32, fill=0)
    active_b = store.create("lazy_active_b", shape, np.int32, fill=0)

    active = None
    root_mode = True
    parent_count = np.array([0, valid_cells], dtype=np.int64)
    n_active = 1
    next_category = 1
    selected_nodes = 0
    visited_nodes = 1  # implicit global root
    levels_processed = 0

    lazy_root = store.root.parent / "lazy_hierarchy"
    shutil.rmtree(lazy_root, ignore_errors=True)
    lazy_root.mkdir(parents=True, exist_ok=True)

    try:
        for level in schedule:
            levels_processed = int(level.level)
            key = CandidateKey(level.cells)
            if candidate_provider is not None:
                provided = candidate_provider.get(level)
                hb_path = Path(provided.raster_path)
                stats = dict(provided.stats)
            else:
                if candidate_cache is None or not candidate_cache.ensure(key, verbose=verbose):
                    raise RuntimeError(
                        f"candidate cache missing for threshold {level.cells} cells"
                    )
                hb_path = candidate_cache.raster_path(key)
                stats = candidate_cache.load_stats(key)
            hb = np.load(hb_path, mmap_mode="r", allow_pickle=False)
            max_hb = int(stats.get("max_half_basin_label", int(np.max(hb))))
            base = max(2, max_hb + 1)
            # Guard uint64 pair encoding.
            if n_active > (np.iinfo(np.uint64).max // base):
                raise OverflowError(
                    "hierarchy pair key exceeds uint64 capacity; domain execution required"
                )
            nbuckets = _choose_bucket_count(max_hb)
            level_root = lazy_root / f"level_{int(level.level):02d}"

            diversity, reduced_paths, child_nodes = _aggregate_children(
                level_root=level_root,
                active=active,
                root_mode=root_mode,
                valid=valid,
                hb=hb,
                aspect_valid=aspect_valid,
                sin_a=sin_a,
                cos_a=cos_a,
                n_active=n_active,
                base=base,
                nbuckets=nbuckets,
                block_rows=block_rows,
            )
            visited_nodes += int(child_nodes)

            (
                rejected,
                parent_category,
                next_parent_count,
                next_category_after,
                nsplit,
                map_paths,
                nrejected,
                nfinalized,
                parents_with_children,
            ) = _prepare_level_maps(
                reduced_paths=reduced_paths,
                level_root=level_root,
                base=base,
                parent_count=parent_count,
                diversity=diversity,
                cell_area_m2=cell_area_m2,
                min_area_m2=min_area_m2,
                cv_min=cv_min,
                max_area_m2=max_area_m2,
                current_level=int(level.level),
                max_level=max_level,
                category_start=next_category,
            )

            selected_nodes += int(nrejected + nfinalized)
            if next_category_after - 1 > np.iinfo(np.int32).max:
                raise OverflowError(
                    "selected hierarchy categories exceed int32 output capacity"
                )

            active_next = active_a if root_mode or active is active_b else active_b
            _render_level(
                active=active,
                root_mode=root_mode,
                active_next=active_next,
                pre=pre,
                valid=valid,
                hb=hb,
                base=base,
                nbuckets=nbuckets,
                rejected=rejected,
                parent_category=parent_category,
                map_paths=map_paths,
                block_rows=block_rows,
            )

            if verbose:
                print(
                    f"[PySlopeUnits] lazy graph level {int(level.level):02d} | "
                    f"active={n_active:,} | children={child_nodes:,} | "
                    f"rejected={nrejected:,} | finalized={nfinalized:,} | "
                    f"split={nsplit:,}"
                )

            mm = getattr(hb, "_mmap", None)
            if mm is not None:
                mm.close()
            del hb
            shutil.rmtree(level_root, ignore_errors=True)
            gc.collect()

            next_category = int(next_category_after)
            if nsplit == 0:
                break

            active = active_next
            root_mode = False
            parent_count = next_parent_count
            n_active = int(nsplit)

        # Same final-fill semantics as graph_create.render_and_clump().
        last_level = schedule[-1]
        last_key = CandidateKey(last_level.cells)
        if candidate_provider is not None:
            provided = candidate_provider.get(last_level)
            last_path = Path(provided.raster_path)
            last_stats = dict(provided.stats)
        else:
            if candidate_cache is None or not candidate_cache.ensure(last_key, verbose=verbose):
                raise RuntimeError("last candidate cache missing")
            last_path = candidate_cache.raster_path(last_key)
            last_stats = candidate_cache.load_stats(last_key)
        last_hb = np.load(last_path, mmap_mode="r", allow_pickle=False)
        max_hb = int(last_stats["max_half_basin_label"])
        selected_max = int(next_category - 1)
        hb_offset = selected_max + 1
        residual_category = hb_offset + max_hb + 1
        if residual_category > np.iinfo(np.int32).max:
            raise OverflowError("preclump categories exceed int32 capacity")

        gaps_from_last = 0
        residual_cells = 0
        for r0 in range(0, rows, 1024):
            r1 = min(rows, r0 + 1024)
            vm = np.asarray(valid[r0:r1], dtype=bool)
            out = np.asarray(pre[r0:r1], dtype=np.int32).copy()
            hb0 = np.asarray(last_hb[r0:r1])
            gap = vm & (out == 0)
            fill = gap & (hb0 > 0)
            if np.any(fill):
                out[fill] = hb_offset + hb0[fill].astype(np.int32)
                gaps_from_last += int(np.count_nonzero(fill))
            residual = vm & (out == 0)
            if np.any(residual):
                out[residual] = residual_category
                residual_cells += int(np.count_nonzero(residual))
            pre[r0:r1] = out
        pre.flush()

        if verbose:
            print(
                f"[PySlopeUnits] final fill | from-last-halfbasin={gaps_from_last:,} | "
                f"residual-placeholder={residual_cells:,}"
            )
            print("[PySlopeUnits] GRASS-like 4-neighbour categorical clump")

        final, n_units = clump_equal_categories(
            pre,
            valid,
            store,
            verbose=verbose,
            output_name="final",
            provisional_name="lazy_clump_provisional",
        )
        store.close_array(final)
        mm = getattr(last_hb, "_mmap", None)
        if mm is not None:
            mm.close()

        result = LazyHierarchyResult(
            selected_nodes=int(selected_nodes),
            visited_nodes=int(visited_nodes),
            preclump_categories=int(residual_category),
            final_units=int(n_units),
            levels_processed=int(levels_processed),
            seconds=time.perf_counter() - t0,
        )
        store.write_json("lazy_hierarchy_report.json", asdict(result))
        return result
    finally:
        store.close_many(valid, sin_a, cos_a, aspect_valid, pre, active_a, active_b)
        store.cleanup("lazy_preclump")
        store.cleanup("lazy_active_a")
        store.cleanup("lazy_active_b")
        shutil.rmtree(lazy_root, ignore_errors=True)
