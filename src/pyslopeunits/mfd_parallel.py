from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import gc
import multiprocessing as mp
import os
import time

import numpy as np
from numba import njit, set_num_threads

from .kernels import (
    DR,
    DC,
    init_mfd_valid_serial,
    mfd_scatter_block_serial,
)
from .memmap_store import MemmapStore


# ============================================================================
# V09
#
# Exact rank-free MFD.
#
# Key idea:
#   old:
#       global mfd_rank[cell] -> topological rank
#
#   v09:
#       uint8 state raster
#       + temporary hash only for the current order block
#
# During reverse traversal:
#       state == 1  -> cell belongs to an already processed upstream block
#       state == 0  -> downstream/future block or current block
#
# Cells inside the current block are distinguished by a compact hash:
#       global cell id -> local block position
#
# This reproduces exactly:
#       rank[j] < rank[i]
#
# without a full-grid int32/int64 rank raster.
#
# During the final adjusted-receiver pass:
#       state == 2 -> lower-rank cell already processed
#
# No hydrological domain is treated as an independent hydrology.
# Global A* order and global MFD semantics remain unchanged.
# ============================================================================


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


# ============================================================================
# BASIC UTILITIES
# ============================================================================


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


def _close_memmap(arr) -> None:
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


def _cleanup_temp(store: MemmapStore, name: str) -> None:
    try:
        store.cleanup(name)
    except Exception:
        pass


def _next_power_of_two(v: int) -> int:
    v = max(8, int(v))

    out = 1

    while out < v:
        out <<= 1

    return out


# ============================================================================
# HASH FOR ONE ORDER BLOCK
# ============================================================================


@njit(cache=True, inline="always")
def _hash_slot(key: int, mask: int) -> int:
    # Signed 64-bit multiplicative hash.
    #
    # We only use the low bits because capacity is always power-of-two.
    x = key * 6364136223846793005

    return int(x & mask)


@njit(cache=True, inline="always")
def _hash_lookup(
    key: int,
    keys: np.ndarray,
    values: np.ndarray,
    tags: np.ndarray,
    generation: int,
) -> int:

    mask = keys.size - 1

    slot = _hash_slot(key, mask)

    while True:
        if tags[slot] != generation:
            return -1

        if keys[slot] == key:
            return int(values[slot])

        slot = (slot + 1) & mask


@njit(cache=True)
def _build_reverse_block_hash(
    order: np.ndarray,
    reverse_offset: int,
    block_len: int,
    keys: np.ndarray,
    values: np.ndarray,
    tags: np.ndarray,
    generation: int,
):
    """
    Hash:
        cell index -> local k

    where:
        k=0 is the highest-rank cell of this reverse block.
    """

    nv = order.size
    mask = keys.size - 1

    for k in range(block_len):
        pos = nv - 1 - (reverse_offset + k)
        idx = int(order[pos])

        # 0 is reserved as empty key.
        key = idx + 1

        slot = _hash_slot(key, mask)

        while tags[slot] == generation:
            slot = (slot + 1) & mask

        keys[slot] = key
        values[slot] = k
        tags[slot] = generation


@njit(cache=True)
def _build_forward_block_hash(
    order: np.ndarray,
    forward_offset: int,
    block_len: int,
    keys: np.ndarray,
    values: np.ndarray,
    tags: np.ndarray,
    generation: int,
):
    """
    Hash:
        cell index -> local k

    where:
        k=0 is the lowest-rank cell of this forward block.
    """

    mask = keys.size - 1

    for k in range(block_len):
        pos = forward_offset + k
        idx = int(order[pos])
        key = idx + 1

        slot = _hash_slot(key, mask)

        while tags[slot] == generation:
            slot = (slot + 1) & mask

        keys[slot] = key
        values[slot] = k
        tags[slot] = generation


# ============================================================================
# STATE
# ============================================================================


@njit(cache=True)
def _mark_reverse_block(
    order: np.ndarray,
    reverse_offset: int,
    block_len: int,
    state: np.ndarray,
    value: int,
):
    nv = order.size
    sf = state.reshape(state.size)

    for k in range(block_len):
        pos = nv - 1 - (reverse_offset + k)
        i = int(order[pos])
        sf[i] = value


@njit(cache=True)
def _mark_forward_block(
    order: np.ndarray,
    forward_offset: int,
    block_len: int,
    state: np.ndarray,
    value: int,
):
    sf = state.reshape(state.size)

    for k in range(block_len):
        i = int(order[forward_offset + k])
        sf[i] = value


# ============================================================================
# MFD WEIGHTS
# ============================================================================


@njit(cache=True, inline="always")
def _conv_power(x: float, convergence: int) -> float:
    if convergence == 1:
        return x

    if convergence == 2:
        return x * x

    if convergence == 3:
        return x * x * x

    if convergence == 4:
        xx = x * x
        return xx * xx

    if convergence == 5:
        xx = x * x
        return xx * xx * x

    if convergence == 6:
        xx = x * x
        return xx * xx * xx

    out = 1.0

    for _ in range(convergence):
        out *= x

    return out


@njit(cache=True)
def _mfd_weights_rankfree_slice(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    state: np.ndarray,
    hash_keys: np.ndarray,
    hash_values: np.ndarray,
    hash_tags: np.ndarray,
    generation: int,
    reverse_offset: int,
    k0: int,
    k1: int,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
    weights_out: np.ndarray,
):
    """
    Exact replacement for mfd_weights_slice_serial().

    Old test:
        rank[j] < rank[i]

    V09 equivalent during reverse traversal:
        previous reverse block -> state[j] == 1 -> NOT eligible
        current block:
            local_j > local_i -> eligible
            local_j < local_i -> not eligible
        future block -> state[j] == 0 and not in current hash -> eligible
    """

    rows, cols = dem_scaled.shape

    zf = dem_scaled.reshape(dem_scaled.size)
    vf = valid.reshape(valid.size)
    rec0 = astar_receiver.reshape(astar_receiver.size)
    sf = state.reshape(state.size)
    nv = order.size

    diag = np.sqrt(xres_scaled * xres_scaled + yres_scaled * yres_scaled)

    for k in range(k0, k1):
        for q in range(9):
            weights_out[k, q] = 0.0

        pos = nv - 1 - (reverse_offset + k)
        i = int(order[pos])

        r = i // cols
        c = i - r * cols

        # Same edge semantics as V08.
        if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
            continue

        edge = False

        for ct in range(8):
            rr = r + DR[ct]
            cc = c + DC[ct]

            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                edge = True
                break

            j = rr * cols + cc

            if vf[j] == 0:
                edge = True
                break

        if edge:
            continue

        zi = float(zf[i])
        astar_j = int(rec0[i])
        astar_present = False
        maxw = 0.0
        sw = 0.0

        for ct in range(8):
            rr = r + DR[ct]
            cc = c + DC[ct]
            j = rr * cols + cc

            # ------------------------------------------------------------
            # Exact rank[j] < rank[i] test without global rank raster.
            # ------------------------------------------------------------
            if sf[j] != 0:
                # j belongs to a previous reverse block:
                # rank[j] > rank[i]
                continue

            local_j = _hash_lookup(
                int(j) + 1,
                hash_keys,
                hash_values,
                hash_tags,
                generation,
            )

            if local_j >= 0 and local_j <= k:
                # Same block, but j is upstream/higher-rank.
                continue

            # Otherwise j is either:
            #   * later in this reverse block; or
            #   * in a future reverse block.
            #
            # Therefore rank[j] < rank[i].

            dz = zi - float(zf[j])

            if ct < 2:
                dist = yres_scaled
            elif ct < 4:
                dist = xres_scaled
            else:
                dist = diag

            if dz > 0.0:
                w = _conv_power(dz / dist, convergence)
            elif dz == 0.0:
                w = _conv_power(0.5 / dist, convergence)
            else:
                continue

            weights_out[k, ct] = w
            sw += w

            if w > maxw:
                maxw = w

            if j == astar_j:
                astar_present = True

        # Preserve V08 A* fallback semantics exactly.
        if astar_j >= 0 and not astar_present:
            if maxw <= 0.0:
                maxw = 1.0

            ar = astar_j // cols
            ac = astar_j - ar * cols

            adr = ar - r
            adc = ac - c

            represented = False

            for ct in range(8):
                if DR[ct] == adr and DC[ct] == adc:
                    weights_out[k, ct] += maxw
                    sw += maxw
                    represented = True
                    break

            if not represented:
                weights_out[k, 8] = maxw
                sw += maxw

        if sw <= 0.0:
            if astar_j >= 0:
                weights_out[k, 8] = 1.0
            continue

        inv = 1.0 / sw

        for q in range(9):
            weights_out[k, q] *= inv


# ============================================================================
# ADJUSTED RECEIVER
# ============================================================================


@njit(cache=True)
def _adjusted_receiver_rankfree_slice(
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    state: np.ndarray,
    hash_keys: np.ndarray,
    hash_values: np.ndarray,
    hash_tags: np.ndarray,
    generation: int,
    accumulation: np.ndarray,
    adjusted_receiver: np.ndarray,
    forward_offset: int,
    k0: int,
    k1: int,
):
    """
    Exact rank-free equivalent of adjusted_receiver_slice_serial().

    During forward traversal:
        state == 2 -> lower-rank cell in a previous block

    Inside the current block:
        local_j < local_i -> lower rank
    """

    rows, cols = valid.shape

    vf = valid.reshape(valid.size)
    rec0 = astar_receiver.reshape(astar_receiver.size)
    sf = state.reshape(state.size)
    acc = accumulation.reshape(accumulation.size)
    adj = adjusted_receiver.reshape(adjusted_receiver.size)

    for k in range(k0, k1):
        pos = forward_offset + k
        i = int(order[pos])

        r = i // cols
        c = i - r * cols

        if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
            continue

        edge = False

        for ct in range(8):
            rr = r + DR[ct]
            cc = c + DC[ct]

            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                edge = True
                break

            j = rr * cols + cc

            if vf[j] == 0:
                edge = True
                break

        if edge:
            continue

        best = int(rec0[i])
        best_acc = acc[best] if best >= 0 else -1.0

        for ct in range(8):
            rr = r + DR[ct]
            cc = c + DC[ct]
            j = rr * cols + cc

            lower_rank = sf[j] == 2

            if not lower_rank:
                local_j = _hash_lookup(
                    int(j) + 1,
                    hash_keys,
                    hash_values,
                    hash_tags,
                    generation,
                )

                if local_j >= 0 and local_j < k:
                    lower_rank = True

            if not lower_rank:
                continue

            if acc[j] > best_acc:
                best = j
                best_acc = acc[j]

        adj[i] = best


# ============================================================================
# PROCESS WORKERS
# ============================================================================


def _compute_weight_slice_worker(
    root: str,
    weight_name: str,
    generation: int,
    reverse_offset: int,
    k0: int,
    k1: int,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
):
    store = MemmapStore(root)

    dem = None
    valid = None
    astar = None
    order = None
    state = None
    keys = None
    values = None
    tags = None
    weights = None

    try:
        dem = store.open("hydro_dem", "r")
        valid = store.open("valid", "r")
        astar = store.open("astar_receiver", "r")
        order = store.open("order", "r")
        state = store.open("mfd_v09_state", "r")
        keys = store.open("mfd_v09_hash_keys", "r")
        values = store.open("mfd_v09_hash_values", "r")
        tags = store.open("mfd_v09_hash_tags", "r")
        weights = store.open(weight_name, "r+")

        _mfd_weights_rankfree_slice(
            dem,
            valid,
            astar,
            order,
            state,
            keys,
            values,
            tags,
            generation,
            reverse_offset,
            k0,
            k1,
            xres_scaled,
            yres_scaled,
            convergence,
            weights,
        )

        weights.flush()

        return (os.getpid(), int(k1 - k0))

    finally:
        for arr in (
            dem,
            valid,
            astar,
            order,
            state,
            keys,
            values,
            tags,
            weights,
        ):
            _close_memmap(arr)


def _adjusted_receiver_worker_v09(
    root: str,
    generation: int,
    forward_offset: int,
    k0: int,
    k1: int,
):
    store = MemmapStore(root)

    valid = None
    astar = None
    order = None
    state = None
    keys = None
    values = None
    tags = None
    accumulation = None
    receiver = None

    try:
        valid = store.open("valid", "r")
        astar = store.open("astar_receiver", "r")
        order = store.open("order", "r")
        state = store.open("mfd_v09_state", "r")
        keys = store.open("mfd_v09_hash_keys", "r")
        values = store.open("mfd_v09_hash_values", "r")
        tags = store.open("mfd_v09_hash_tags", "r")
        accumulation = store.open("accumulation", "r")
        receiver = store.open("receiver", "r+")

        _adjusted_receiver_rankfree_slice(
            valid,
            astar,
            order,
            state,
            keys,
            values,
            tags,
            generation,
            accumulation,
            receiver,
            forward_offset,
            k0,
            k1,
        )

        receiver.flush()

        return (os.getpid(), int(k1 - k0))

    finally:
        for arr in (
            valid,
            astar,
            order,
            state,
            keys,
            values,
            tags,
            accumulation,
            receiver,
        ):
            _close_memmap(arr)


def _wait_all(futures):
    pids = set()
    cells = 0

    for f in futures:
        pid, n = f.result()
        pids.add(int(pid))
        cells += int(n)

    return (sorted(pids), cells)


def _submit_weight_block(
    pool,
    store: MemmapStore,
    weight_name: str,
    generation: int,
    reverse_offset: int,
    block_len: int,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
    workers: int,
):
    futures = []

    for k0, k1 in _split_ranges(block_len, workers):
        futures.append(
            pool.submit(
                _compute_weight_slice_worker,
                str(store.root),
                weight_name,
                int(generation),
                int(reverse_offset),
                int(k0),
                int(k1),
                float(xres_scaled),
                float(yres_scaled),
                int(convergence),
            )
        )

    return futures


def _submit_adjusted_block(
    pool,
    store: MemmapStore,
    generation: int,
    forward_offset: int,
    block_len: int,
    workers: int,
):
    futures = []

    for k0, k1 in _split_ranges(block_len, workers):
        futures.append(
            pool.submit(
                _adjusted_receiver_worker_v09,
                str(store.root),
                int(generation),
                int(forward_offset),
                int(k0),
                int(k1),
            )
        )

    return futures


# ============================================================================
# MAIN API
# ============================================================================


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
    """
    V09 exact rank-free block-parallel MFD.

    Public function name and signature intentionally remain unchanged so
    engine.py does not need a hydrology API change.
    """

    set_num_threads(max(1, int(numba_threads)))
    workers = max(1, int(workers))

    # ------------------------------------------------------------
    # Clear only V09 temporary arrays.
    #
    # We intentionally do NOT use the old mfd_weights_* names because
    # engine.py recognizes those names as a legacy Windows Stage-11
    # recovery signature.
    # ------------------------------------------------------------
    for name in (
        "mfd_v09_state",
        "mfd_v09_hash_keys",
        "mfd_v09_hash_values",
        "mfd_v09_hash_tags",
        "mfd_v09_weights_0",
        "mfd_v09_weights_1",
    ):
        _cleanup_temp(store, name)

    dem = store.open("hydro_dem", "r")
    valid = store.open("valid", "r")
    astar = store.open("astar_receiver", "r")
    order = store.open("order", "r")

    nv = int(order.size)

    # Preserve the index dtype chosen by V08.
    index_dtype = np.dtype(astar.dtype)

    accumulation = store.create(
        "accumulation",
        dem.shape,
        np.float64,
        fill=0.0,
    )

    receiver = store.create(
        "receiver",
        dem.shape,
        index_dtype,
        fill=-1,
    )

    # One byte/cell instead of global int32/int64 rank.
    state = store.create(
        "mfd_v09_state",
        dem.shape,
        np.uint8,
        fill=0,
    )

    init_mfd_valid_serial(order, astar, accumulation, receiver)

    accumulation.flush()
    receiver.flush()

    block_cells = max(50_000, int(block_cells))
    nblocks = (nv + block_cells - 1) // block_cells

    capacity = min(block_cells, nv)

    # ------------------------------------------------------------
    # Compact current-block hash.
    #
    # At load factor <= 0.5 lookup remains cheap.
    # No full-grid rank raster is required.
    # ------------------------------------------------------------
    hash_capacity = _next_power_of_two(max(8, capacity * 2))

    hash_keys = store.create(
        "mfd_v09_hash_keys",
        (hash_capacity,),
        np.int64,
        fill=0,
    )

    hash_values = store.create(
        "mfd_v09_hash_values",
        (hash_capacity,),
        np.int32,
        fill=0,
    )

    hash_tags = store.create(
        "mfd_v09_hash_tags",
        (hash_capacity,),
        np.int32,
        fill=0,
    )

    w0 = store.create(
        "mfd_v09_weights_0",
        (capacity, 9),
        np.float64,
    )

    w1 = store.create(
        "mfd_v09_weights_1",
        (capacity, 9),
        np.float64,
    )

    weight_names = (
        "mfd_v09_weights_0",
        "mfd_v09_weights_1",
    )

    if verbose:
        old_rank_bytes = int(np.prod(dem.shape)) * np.dtype(np.int32).itemsize
        state_bytes = int(np.prod(dem.shape)) * np.dtype(np.uint8).itemsize

        print(
            "[PySlopeUnits] MFD V09 | exact rank-free block-parallel | "
            f"valid={nv:,}"
        )

        print(
            "[PySlopeUnits] MFD V09 storage | "
            f"old-rank={old_rank_bytes / 1024**3:.2f} GB | "
            f"state={state_bytes / 1024**3:.2f} GB | "
            f"block-hash={hash_capacity:,} slots"
        )

    # Warm-up in parent.
    _mfd_weights_rankfree_slice(
        dem,
        valid,
        astar,
        order,
        state,
        hash_keys,
        hash_values,
        hash_tags,
        1,
        0,
        0,
        0,
        xres_scaled,
        yres_scaled,
        convergence,
        w0,
    )

    _adjusted_receiver_rankfree_slice(
        valid,
        astar,
        order,
        state,
        hash_keys,
        hash_values,
        hash_tags,
        1,
        accumulation,
        receiver,
        0,
        0,
        0,
    )

    t0 = time.perf_counter()

    weight_wait = 0.0
    scatter_seconds = 0.0
    adjusted_seconds = 0.0

    next_report = max(1, int(progress_percent))
    generation = 1

    ctx = mp.get_context("spawn")
    adjusted_pids = []

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        # ========================================================
        # PHASE 1:
        # MFD accumulation in exact reverse global order.
        # ========================================================

        cur_offset = 0
        cur_len = min(block_cells, nv)
        cur_buf = 0

        _build_reverse_block_hash(
            order,
            cur_offset,
            cur_len,
            hash_keys,
            hash_values,
            hash_tags,
            generation,
        )

        cur_generation = generation

        cur_futures = _submit_weight_block(
            pool,
            store,
            weight_names[cur_buf],
            cur_generation,
            cur_offset,
            cur_len,
            xres_scaled,
            yres_scaled,
            convergence,
            workers,
        )

        for b in range(nblocks):
            wt0 = time.perf_counter()

            pids, _ = _wait_all(cur_futures)

            weight_wait += time.perf_counter() - wt0

            # Current weights are final.
            #
            # Mark this whole block as belonging to the processed
            # upstream portion BEFORE launching the next block.
            _mark_reverse_block(order, cur_offset, cur_len, state, 1)

            # ----------------------------------------------------
            # Prepare next weight calculation before scattering the
            # current block. This retains V08's pipeline overlap.
            # ----------------------------------------------------

            next_futures = None
            next_buf = 1 - cur_buf
            next_offset = cur_offset + cur_len
            next_len = min(block_cells, max(0, nv - next_offset))
            next_generation = generation + 1

            if b + 1 < nblocks:
                _build_reverse_block_hash(
                    order,
                    next_offset,
                    next_len,
                    hash_keys,
                    hash_values,
                    hash_tags,
                    next_generation,
                )

                next_futures = _submit_weight_block(
                    pool,
                    store,
                    weight_names[next_buf],
                    next_generation,
                    next_offset,
                    next_len,
                    xres_scaled,
                    yres_scaled,
                    convergence,
                    workers,
                )

            # ----------------------------------------------------
            # Exact accumulation scatter.
            #
            # We deliberately keep the V08 kernel here.
            # ----------------------------------------------------

            current_weights = store.open(weight_names[cur_buf], "r")

            try:
                st0 = time.perf_counter()

                mfd_scatter_block_serial(
                    valid,
                    astar,
                    order,
                    cur_offset,
                    cur_len,
                    accumulation,
                    current_weights,
                )

                scatter_seconds += time.perf_counter() - st0

            finally:
                _close_memmap(current_weights)
                del current_weights

            if verbose:
                pct = int(((b + 1) * 100) / nblocks)

                if pct >= next_report or b == nblocks - 1:
                    elapsed = time.perf_counter() - t0
                    done = cur_offset + cur_len
                    rate = (done / 1_000_000.0) / max(elapsed, 1e-9)

                    print(
                        f"           MFD {pct:3d}% | "
                        f"{done:,}/{nv:,} cells | "
                        f"{rate:.2f} M cells/s | "
                        f"{elapsed / 60:.1f} min | "
                        f"weight_pids={pids}"
                    )

                    while next_report <= pct:
                        next_report += max(1, int(progress_percent))

            if next_futures is not None:
                cur_futures = next_futures
                cur_buf = next_buf
                cur_offset = next_offset
                cur_len = next_len
                generation = next_generation

        accumulation.flush()

        # ========================================================
        # PHASE 2:
        # adjusted drainage receiver
        #
        # state currently == 1 for all valid order cells.
        #
        # We now walk forward.
        #
        # state == 2 means:
        #     already processed lower-rank block.
        # ========================================================

        if verbose:
            print(
                "[PySlopeUnits] MFD V09 adjusted receiver | "
                f"rank-free block-parallel | workers={workers}"
            )

        ar0 = time.perf_counter()
        next_report = max(1, int(progress_percent))
        forward_offset = 0
        generation += 1

        for b in range(nblocks):
            block_len = min(block_cells, nv - forward_offset)

            _build_forward_block_hash(
                order,
                forward_offset,
                block_len,
                hash_keys,
                hash_values,
                hash_tags,
                generation,
            )

            futures = _submit_adjusted_block(
                pool,
                store,
                generation,
                forward_offset,
                block_len,
                workers,
            )

            adjusted_pids, _ = _wait_all(futures)

            _mark_forward_block(order, forward_offset, block_len, state, 2)

            forward_offset += block_len

            if verbose:
                pct = int(((b + 1) * 100) / nblocks)

                if pct >= next_report or b == nblocks - 1:
                    print(
                        f"           ADJ {pct:3d}% | "
                        f"{forward_offset:,}/{nv:,} cells"
                    )

                    while next_report <= pct:
                        next_report += max(1, int(progress_percent))

            generation += 1

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

    # ------------------------------------------------------------
    # Durable checkpoint only after BOTH accumulation and adjusted
    # receiver are complete.
    # ------------------------------------------------------------

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
            "algorithm": "v09-rankfree-block-hash",
        },
    )

    store.write_json("mfd_parallel_report.json", asdict(result))

    # ------------------------------------------------------------
    # Close parent mappings before temporary cleanup.
    # ------------------------------------------------------------

    for arr in (
        hash_keys,
        hash_values,
        hash_tags,
        w0,
        w1,
        state,
    ):
        _close_memmap(arr)

    del (
        hash_keys,
        hash_values,
        hash_tags,
        w0,
        w1,
        state,
    )

    gc.collect()

    for name in (
        "mfd_v09_state",
        "mfd_v09_hash_keys",
        "mfd_v09_hash_values",
        "mfd_v09_hash_tags",
        "mfd_v09_weights_0",
        "mfd_v09_weights_1",
    ):
        _cleanup_temp(store, name)

    # Persistent products remain on disk; close only mappings.
    for arr in (
        dem,
        valid,
        astar,
        order,
        accumulation,
        receiver,
    ):
        _close_memmap(arr)

    del (
        dem,
        valid,
        astar,
        order,
        accumulation,
        receiver,
    )

    gc.collect()

    if verbose:
        print(
            "[PySlopeUnits] MFD V09 complete | "
            f"total={total / 60:.2f} min | "
            f"scatter={scatter_seconds / 60:.2f} min | "
            f"weight_wait={weight_wait / 60:.2f} min | "
            f"adjusted={adjusted_seconds / 60:.2f} min | "
            f"adjust_pids={adjusted_pids}"
        )

    return result
