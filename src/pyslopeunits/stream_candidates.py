from __future__ import annotations

from .logging_utils import log as print

"""On-demand candidate generation for bounded-disk lazy hierarchy runs."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .halfbasin_parallel import build_half_basins
from .kernels import stream_mask_grasslike_static
from .memmap_store import MemmapStore


@dataclass(frozen=True)
class StreamingCandidate:
    raster_path: Path
    stats: dict


class StreamingCandidateProvider:
    """Generate one half-basin threshold at a time from prepared hydrology.

    Only ``stream.npy`` and ``half_basins.npy`` are retained as working rasters.
    This avoids storing one full-size candidate raster per threshold, which is
    critical for very-large/global DEMs.  Candidate geometry is unchanged.
    """

    def __init__(
        self,
        store: MemmapStore,
        *,
        workers: int = 1,
        verbose: bool = True,
    ):
        self.store = store
        self.workers = max(1, int(workers))
        self.verbose = bool(verbose)
        self._pool = None
        self._current_cells: int | None = None
        self._current_stats: dict | None = None

        self.valid = store.open("valid", "r")
        self.accumulation = store.open("accumulation", "r")
        self.receiver = store.open("receiver", "r")
        self.order = store.open("order", "r")
        self.static_ok = store.open("stream_static_ok", "r")
        self.up_acc = store.open("stream_up_acc_max", "r")

        if store.exists("stream"):
            self.stream = store.open("stream", "r+")
        else:
            self.stream = store.create_shared(
                "stream", self.valid.shape, np.uint8, fill=0
            )

        if self.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
            )

    def get(self, level) -> StreamingCandidate:
        cells = int(level.cells)
        if self._current_cells == cells and self._current_stats is not None:
            return StreamingCandidate(
                raster_path=self.store.path("half_basins"),
                stats=dict(self._current_stats),
            )

        if self.verbose:
            print(
                f"[PySlopeUnits stream] candidate level={int(level.level):02d} | "
                f"threshold={float(level.area_m2):,.1f} m2 ({cells:,} cells)"
            )

        stream_mask_grasslike_static(
            self.valid,
            self.accumulation,
            self.receiver,
            self.order,
            float(cells),
            self.static_ok,
            self.up_acc,
            self.stream,
        )
        self.stream.flush()
        stream_cells = int(np.count_nonzero(self.stream))

        # build_half_basins recreates/zeros this shared working raster.
        stats = build_half_basins(
            self.store,
            workers=self.workers,
            executor=self._pool,
            verbose=self.verbose,
        )
        stats["stream_cells"] = stream_cells
        stats["threshold_cells"] = cells
        stats["threshold_m2"] = float(level.area_m2)
        stats["candidate_storage"] = "streaming-working-raster"

        self._current_cells = cells
        self._current_stats = dict(stats)
        return StreamingCandidate(
            raster_path=self.store.path("half_basins"),
            stats=dict(stats),
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.store.close_many(
            self.valid,
            self.accumulation,
            self.receiver,
            self.order,
            self.static_ok,
            self.up_acc,
            self.stream,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
