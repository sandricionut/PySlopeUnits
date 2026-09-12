from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

from .kernels import (
    mark_branch_roots_fast,
    process_stream_outlets,
)
from .memmap_store import MemmapStore


def _split_round_robin(values: np.ndarray, workers: int) -> list[np.ndarray]:
    if values.size == 0:
        return []
    workers = max(1, min(int(workers), int(values.size)))
    return [
        np.ascontiguousarray(values[i::workers], dtype=values.dtype)
        for i in range(workers)
    ]


def _warmup_halfbasin_numba() -> None:
    rec = np.array([[-1]], dtype=np.int32)
    stream = np.array([[1]], dtype=np.uint8)
    acc = np.array([[1.0]], dtype=np.float64)
    valid = np.array([[1]], dtype=np.uint8)
    haf = np.zeros((1, 1), dtype=np.int32)
    roots = np.array([0], dtype=np.int32)
    outlets = np.array([0], dtype=np.int32)
    process_stream_outlets(outlets, roots, rec, stream, acc, valid, haf)


def _stream_worker(
    store_dir: str,
    roots_path: str,
    outlets: np.ndarray,
) -> tuple[int, int]:
    # Keep spawned workers single-threaded; process-level parallelism is explicit.
    from numba import set_num_threads

    set_num_threads(1)
    store = MemmapStore(store_dir)
    arrays = []
    roots = None
    try:
        receiver = store.open("receiver", "r")
        arrays.append(receiver)
        stream = store.open("stream", "r")
        arrays.append(stream)
        accumulation = store.open("accumulation", "r")
        arrays.append(accumulation)
        valid = store.open("valid", "r")
        arrays.append(valid)
        half_basins = store.open("half_basins", "r+")
        arrays.append(half_basins)
        roots = np.load(roots_path, mmap_mode="r", allow_pickle=False)

        process_stream_outlets(
            np.asarray(outlets, dtype=outlets.dtype),
            roots,
            receiver,
            stream,
            accumulation,
            valid,
            half_basins,
        )
        half_basins.flush()
        return os.getpid(), int(outlets.size)
    finally:
        if roots is not None:
            store.close_array(roots)
        for arr in arrays:
            store.close_array(arr)


def build_half_basins(
    store: MemmapStore,
    *,
    workers: int = 8,
    executor=None,
    verbose: bool = True,
) -> dict:
    """Build stream-derived half-basins.

    Persistent/shared inputs and ``half_basins`` stay file-backed because
    spawned workers reopen them through ``MemmapStore``.  The branch-root
    detection buffers are local to the parent process, so V06 allocates those
    RAM-first and transparently spills them to disk under memory pressure.
    """

    receiver = store.open("receiver", "r")
    stream = store.open("stream", "r")
    valid = store.open("valid", "r")
    accumulation = store.open("accumulation", "r")

    # Half-basin label dtype is chosen after branch roots are known.
    # The array itself remains file-backed because spawned workers reopen it.

    # V06+: these two full-raster buffers are local-only scratch.
    mark = store.create_temp(
        "branch_mark", receiver.shape, np.uint8, fill=0
    )
    donor_count = store.create_temp(
        "stream_donor_count", receiver.shape, np.uint8, fill=0
    )

    # Fast O(N) root detection instead of an 8-neighbour donor search for
    # every stream cell.
    mark_branch_roots_fast(receiver, stream, valid, mark, donor_count)
    mark.flush()

    index_dtype = receiver.dtype
    roots = np.flatnonzero(np.asarray(mark).ravel()).astype(index_dtype, copy=False)
    roots_path = store.path("branch_roots")
    np.save(roots_path, roots, allow_pickle=False)

    max_label = 2 * int(roots.size)
    label_dtype = np.int32 if max_label <= np.iinfo(np.int32).max else np.int64
    if store.exists("half_basins"):
        half_basins = store.open("half_basins", "r+")
        if half_basins.dtype != np.dtype(label_dtype):
            store.close_array(half_basins)
            store.remove("half_basins")
            half_basins = store.create_shared(
                "half_basins", receiver.shape, label_dtype, fill=0
            )
        else:
            half_basins[...] = 0
            half_basins.flush()
    else:
        half_basins = store.create_shared(
            "half_basins", receiver.shape, label_dtype, fill=0
        )

    # Explicitly release RAM budget or remove spill files before workers start.
    store.close_many(mark, donor_count)
    store.release_temp("branch_mark")
    store.release_temp("stream_donor_count")
    del mark, donor_count
    gc.collect()

    # Stream outlets are the subset of branch roots that drain outside stream.
    recf = receiver.ravel()
    sf = stream.ravel()
    vf = valid.ravel()
    if roots.size:
        rr = recf[roots]
        safe = np.maximum(rr, 0)
        is_outlet = (rr < 0) | (vf[safe] == 0) | (sf[safe] == 0)
        outlets = roots[is_outlet]
    else:
        outlets = np.empty(0, dtype=index_dtype)

    worker_pids: set[int] = set()
    ctx = mp.get_context("spawn")
    chunks = _split_round_robin(outlets, workers)

    if verbose:
        print(
            f"[PySlopeUnits] half-basins: stream roots={roots.size:,} | "
            f"outlets={outlets.size:,} | workers={len(chunks) or 1}"
        )

    _warmup_halfbasin_numba()

    if len(chunks) <= 1:
        if outlets.size:
            process_stream_outlets(
                outlets,
                roots,
                receiver,
                stream,
                accumulation,
                valid,
                half_basins,
            )
    else:
        owns_executor = executor is None
        pool = (
            executor
            if executor is not None
            else ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx)
        )
        try:
            futures = [
                pool.submit(
                    _stream_worker,
                    str(store.root),
                    str(roots_path),
                    ch,
                )
                for ch in chunks
            ]
            for future in as_completed(futures):
                pid, n = future.result()
                worker_pids.add(int(pid))
                if verbose:
                    print(f"           stream worker pid={pid} outlets={n:,}")
        finally:
            if owns_executor:
                pool.shutdown(wait=True)

    half_basins.flush()

    # Preserve the tested candidate semantics: only SWALE/stream pour points
    # initiate half-basins. Components without stream stay 0 until final fill.
    nbranch = int(roots.size)
    max_label = 2 * nbranch
    positive_cells = 0
    valid_cells = 0
    block_rows = max(1, min(int(valid.shape[0]), 1024))
    for r0 in range(0, int(valid.shape[0]), block_rows):
        r1 = min(int(valid.shape[0]), r0 + block_rows)
        positive_cells += int(np.count_nonzero(np.asarray(half_basins[r0:r1])))
        valid_cells += int(np.count_nonzero(np.asarray(valid[r0:r1])))
    missing_cells = valid_cells - positive_cells

    if verbose:
        print(
            f"           stream-derived cells={positive_cells:,} | "
            f"unassigned-for-final-fill={missing_cells:,}"
        )

    result = {
        "stream_branch_roots": int(roots.size),
        "stream_outlets": int(outlets.size),
        "residual_outlets": 0,
        "residual_mode": "final-fill-only",
        "defensive_components": 0,
        "candidate_positive_cells": positive_cells,
        "candidate_unassigned_cells": missing_cells,
        "max_half_basin_label": int(max_label),
        "worker_pids": sorted(worker_pids),
    }

    # The ProcessPoolExecutor is reused across levels on Windows/macOS. No
    # parent mapping may survive into the next candidate level.
    store.close_many(receiver, stream, valid, accumulation, half_basins)
    del receiver, stream, valid, accumulation, half_basins
    gc.collect()
    return result
