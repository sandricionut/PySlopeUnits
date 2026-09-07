from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import multiprocessing as mp
from pathlib import Path
import time
import gc

import numpy as np
from numba import set_num_threads

from .kernels import (
    build_rank_serial,
    init_mfd_valid_serial,
    mfd_weights_slice_serial,
    mfd_scatter_block_serial,
    adjusted_receiver_slice_serial,
)
from .memmap_store import MemmapStore
from .mfd_workers import compute_weight_slice_worker, adjusted_receiver_worker


@dataclass(frozen=True)
class MFDParallelResult:
    valid_cells: int
    blocks: int
    block_cells: int
    workers: int
    weight_seconds_waited: float
    scatter_seconds: float
    adjusted_receiver_seconds: float
    total_seconds: float


def _split_ranges(n: int, workers: int):
    workers = max(1, min(int(workers), int(n)))
    q, r = divmod(int(n), workers)
    out = []
    start = 0
    for w in range(workers):
        length = q + (1 if w < r else 0)
        if length:
            out.append((start, start + length))
            start += length
    return out


def _submit_weight_block(pool, store, weight_path, reverse_offset, block_len,
                         xres_scaled, yres_scaled, convergence, workers):
    futures = []
    for k0, k1 in _split_ranges(block_len, workers):
        futures.append(pool.submit(
            compute_weight_slice_worker,
            str(store.root), str(weight_path),
            int(reverse_offset), int(k0), int(k1),
            float(xres_scaled), float(yres_scaled), int(convergence),
        ))
    return futures


def _wait_all(futures):
    pids = set()
    cells = 0
    for f in futures:
        pid, n = f.result()
        pids.add(int(pid))
        cells += int(n)
    return sorted(pids), cells



def _close_memmap(arr) -> None:
    """Flush and explicitly close a NumPy memmap on Windows."""
    if arr is None:
        return
    try:
        arr.flush()
    except Exception:
        pass
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:
            pass


def _remove_temp_best_effort(
    store: MemmapStore,
    name: str,
    *,
    retries: int = 8,
    delay: float = 0.25,
    verbose: bool = False,
) -> bool:
    """Remove an expendable temp memmap without failing a completed MFD."""
    path = store.path(name)
    if not path.exists():
        return True

    for _ in range(max(1, retries)):
        try:
            path.unlink()
            return True
        except PermissionError:
            gc.collect()
            time.sleep(delay)
        except FileNotFoundError:
            return True

    if verbose:
        print(
            f"[PySlopeUnits] warning: temporary file still locked; "
            f"leaving it for next run: {path}"
        )
    return False


def run_mfd_block_parallel(
    store: MemmapStore,
    *,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
    numba_threads: int,
    workers: int = 8,
    block_cells: int = 1_000_000,
    progress_percent: int = 5,
    verbose: bool = True,
) -> MFDParallelResult:
    """Exact MFD with double-buffered multiprocessing weight pipeline.

    For block b, worker processes compute only routing weights, which are
    independent of accumulation. While the parent process scatters block b in
    exact topological order, workers already compute weights for block b+1 in
    the second buffer. This overlaps the expensive parallelizable work with
    the unavoidable dependency-sensitive scatter.
    """
    set_num_threads(max(1, int(numba_threads)))
    workers = max(1, int(workers))

    dem = store.open("hydro_dem", "r")
    valid = store.open("valid", "r")
    astar = store.open("astar_receiver", "r")
    order = store.open("order", "r")
    nv = int(order.size)

    rank = store.create("mfd_rank", dem.shape, np.int32)
    accumulation = store.create("accumulation", dem.shape, np.float64, fill=0.0)
    receiver = store.create("receiver", dem.shape, np.int32, fill=-1)

    if verbose:
        print(f"[PySlopeUnits] MFD rank map | valid={nv:,}")
    t0 = time.perf_counter()
    build_rank_serial(order, rank)
    rank.flush()
    init_mfd_valid_serial(order, astar, accumulation, receiver)
    accumulation.flush(); receiver.flush()

    block_cells = max(50_000, int(block_cells))
    nblocks = (nv + block_cells - 1) // block_cells
    capacity = min(block_cells, nv)

    # Two disk-backed buffers. On a modern NVMe they are typically cached in
    # RAM by Windows, while remaining shareable by spawned worker processes.
    w0 = store.create("mfd_weights_0", (capacity, 9), np.float64)
    w1 = store.create("mfd_weights_1", (capacity, 9), np.float64)
    wpaths = [store.path("mfd_weights_0"), store.path("mfd_weights_1")]

    # Compile serial kernels in parent before spawn; workers then normally load
    # Numba's on-disk cache instead of all compiling simultaneously.
    warm = min(1, nv)
    if warm:
        mfd_weights_slice_serial(
            dem, valid, astar, order, rank, 0, 0, 0,
            xres_scaled, yres_scaled, convergence, w0,
        )
        adjusted_receiver_slice_serial(
            valid, astar, order, rank, accumulation, receiver, 0, 0,
        )

    weight_wait = 0.0
    scatter_seconds = 0.0
    next_report = max(1, int(progress_percent))
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        # Prime first block.
        cur_b = 0
        cur_offset = 0
        cur_len = min(block_cells, nv)
        cur_buf = 0
        cur_futures = _submit_weight_block(
            pool, store, wpaths[cur_buf], cur_offset, cur_len,
            xres_scaled, yres_scaled, convergence, workers,
        )

        for b in range(nblocks):
            # Ensure current weights are complete.
            wt0 = time.perf_counter()
            pids, _ = _wait_all(cur_futures)
            weight_wait += time.perf_counter() - wt0

            # Start next weight block before scatter -> double buffering.
            next_futures = None
            next_buf = 1 - cur_buf
            next_offset = (b + 1) * block_cells
            next_len = min(block_cells, max(0, nv - next_offset))
            if b + 1 < nblocks:
                next_futures = _submit_weight_block(
                    pool, store, wpaths[next_buf], next_offset, next_len,
                    xres_scaled, yres_scaled, convergence, workers,
                )

            current_weights = np.load(
                wpaths[cur_buf], mmap_mode="r", allow_pickle=False
            )
            try:
                st0 = time.perf_counter()
                mfd_scatter_block_serial(
                    valid, astar, order,
                    cur_offset, cur_len,
                    accumulation, current_weights,
                )
                scatter_seconds += time.perf_counter() - st0
            finally:
                # Critical on Windows: np.load(..., mmap_mode="r") keeps an
                # OS file handle open until the mapping itself is closed.
                _close_memmap(current_weights)
                del current_weights

            if verbose:
                pct = int(((b + 1) * 100) / nblocks)
                if pct >= next_report or b == nblocks - 1:
                    elapsed = time.perf_counter() - t0
                    done = cur_offset + cur_len
                    rate = (done / 1_000_000.0) / max(elapsed, 1e-9)
                    print(
                        f"           MFD {pct:3d}% | {done:,}/{nv:,} cells | "
                        f"{rate:.2f} M cells/s | {elapsed/60:.1f} min | "
                        f"weight_pids={pids}"
                    )
                    while next_report <= pct:
                        next_report += max(1, int(progress_percent))

            if next_futures is not None:
                cur_futures = next_futures
                cur_buf = next_buf
                cur_offset = next_offset
                cur_len = next_len

        accumulation.flush()

        # Final adjusted receiver is fully read-only with respect to completed
        # accumulation, so split it across the same worker pool.
        if verbose:
            print(
                f"[PySlopeUnits] adjusted drainage receiver | "
                f"multiprocessing workers={workers}"
            )
        ar0 = time.perf_counter()
        futures = []
        for p0, p1 in _split_ranges(nv, workers):
            futures.append(pool.submit(
                adjusted_receiver_worker,
                str(store.root), p0, p1,
            ))
        adjusted_pids, _ = _wait_all(futures)
        adjusted_seconds = time.perf_counter() - ar0
        receiver.flush()

    total = time.perf_counter() - t0

    result = MFDParallelResult(
        valid_cells=nv,
        blocks=nblocks,
        block_cells=block_cells,
        workers=workers,
        weight_seconds_waited=weight_wait,
        scatter_seconds=scatter_seconds,
        adjusted_receiver_seconds=adjusted_seconds,
        total_seconds=total,
    )

    # Write a durable completion marker BEFORE touching expendable temporary
    # buffers. A Windows cleanup failure must never invalidate completed MFD.
    store.write_json(
        "mfd_complete_checkpoint.json",
        {
            "valid_cells": nv,
            "convergence": int(convergence),
            "xres_scaled": float(xres_scaled),
            "yres_scaled": float(yres_scaled),
            "blocks": int(nblocks),
            "block_cells": int(block_cells),
            "workers": int(workers),
        },
    )
    store.write_json("mfd_parallel_report.json", asdict(result))

    # Explicitly close all parent-owned mappings before unlinking on Windows.
    _close_memmap(rank)
    _close_memmap(w0)
    _close_memmap(w1)
    del rank, w0, w1
    gc.collect()

    _remove_temp_best_effort(
        store, "mfd_rank", verbose=verbose
    )
    _remove_temp_best_effort(
        store, "mfd_weights_0", verbose=verbose
    )
    _remove_temp_best_effort(
        store, "mfd_weights_1", verbose=verbose
    )

    # Release every parent-owned mapping before returning to engine.py.
    # The files themselves remain; only the OS mappings are closed.
    _close_memmap(dem)
    _close_memmap(valid)
    _close_memmap(astar)
    _close_memmap(order)
    _close_memmap(accumulation)
    _close_memmap(receiver)
    del dem, valid, astar, order, accumulation, receiver
    gc.collect()

    if verbose:
        print(
            f"[PySlopeUnits] MFD complete | total={total/60:.2f} min | "
            f"scatter={scatter_seconds/60:.2f} min | "
            f"weight_wait={weight_wait/60:.2f} min | "
            f"adjusted={adjusted_seconds/60:.2f} min | "
            f"adjust_pids={adjusted_pids}"
        )

    return result
