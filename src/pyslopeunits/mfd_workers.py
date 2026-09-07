from __future__ import annotations

import os
import numpy as np

from .kernels import (
    mfd_weights_slice_serial,
    adjusted_receiver_slice_serial,
)
from .memmap_store import MemmapStore


def compute_weight_slice_worker(
    store_dir: str,
    weights_path: str,
    reverse_offset: int,
    k0: int,
    k1: int,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
):
    from numba import set_num_threads
    set_num_threads(1)
    store = MemmapStore(store_dir)
    arrays = []
    weights = None
    try:
        dem = store.open("hydro_dem", "r"); arrays.append(dem)
        valid = store.open("valid", "r"); arrays.append(valid)
        astar = store.open("astar_receiver", "r"); arrays.append(astar)
        order = store.open("order", "r"); arrays.append(order)
        rank = store.open("mfd_rank", "r"); arrays.append(rank)
        weights = np.load(weights_path, mmap_mode="r+", allow_pickle=False)

        mfd_weights_slice_serial(
            dem, valid, astar, order, rank,
            int(reverse_offset), int(k0), int(k1),
            float(xres_scaled), float(yres_scaled), int(convergence),
            weights,
        )
        weights.flush()
        return os.getpid(), int(k1-k0)
    finally:
        if weights is not None:
            store.close_array(weights)
        for arr in arrays:
            store.close_array(arr)


def adjusted_receiver_worker(store_dir: str, p0: int, p1: int):
    from numba import set_num_threads
    set_num_threads(1)
    store = MemmapStore(store_dir)
    arrays = []
    try:
        valid = store.open("valid", "r"); arrays.append(valid)
        astar = store.open("astar_receiver", "r"); arrays.append(astar)
        order = store.open("order", "r"); arrays.append(order)
        rank = store.open("mfd_rank", "r"); arrays.append(rank)
        accumulation = store.open("accumulation", "r"); arrays.append(accumulation)
        receiver = store.open("receiver", "r+"); arrays.append(receiver)

        adjusted_receiver_slice_serial(
            valid, astar, order, rank, accumulation, receiver,
            int(p0), int(p1),
        )
        receiver.flush()
        return os.getpid(), int(p1-p0)
    finally:
        for arr in arrays:
            store.close_array(arr)
