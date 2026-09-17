from __future__ import annotations

from .logging_utils import log as print

from dataclasses import dataclass
from pathlib import Path
import json
import gc
import os
import shutil

import numpy as np

from .memmap_store import MemmapStore


CANDIDATE_SEMANTICS = "grass_stream_halfbasin_v14"


@dataclass(frozen=True)
class CandidateKey:
    threshold_cells: int

    @property
    def stem(self) -> str:
        return f"hb_t{self.threshold_cells:012d}"


class MultiThresholdCandidateCache:
    """Persistent cache of stream-derived GRASS-like half-basin candidates.

    The current pipeline deliberately excludes drainage components that contain no SWALE/stream
    outlet. This matches the semantics of ``r.watershed hbasin`` more closely:
    half-basin construction starts only at stream pour points. Remaining DEM
    cells are handled only by the final fill + clump stage.

    Existing Stage 9--13 candidate rasters can be migrated without rebuilding
    hydrology or tracing streams: their stream-derived labels occupy
    ``1..2*stream_branch_roots`` and the legacy residual labels are larger.
    """

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.root = self.work_dir / "candidates_v14"
        self.legacy_root = self.work_dir / "candidates"
        self.root.mkdir(parents=True, exist_ok=True)

    def raster_path(self, key: CandidateKey) -> Path:
        return self.root / f"{key.stem}.npy"

    def stats_path(self, key: CandidateKey) -> Path:
        return self.root / f"{key.stem}.json"

    def legacy_raster_path(self, key: CandidateKey) -> Path:
        return self.legacy_root / f"{key.stem}.npy"

    def legacy_stats_path(self, key: CandidateKey) -> Path:
        return self.legacy_root / f"{key.stem}.json"

    def exists(self, key: CandidateKey) -> bool:
        rp = self.raster_path(key)
        sp = self.stats_path(key)
        if not (rp.exists() and sp.exists()):
            return False
        try:
            stats = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            return False
        return stats.get("candidate_semantics") == CANDIDATE_SEMANTICS

    def legacy_exists(self, key: CandidateKey) -> bool:
        return self.legacy_raster_path(key).exists() and self.legacy_stats_path(key).exists()

    def load_stats(self, key: CandidateKey) -> dict:
        return json.loads(self.stats_path(key).read_text(encoding="utf-8"))

    def prepare_for_build(self, store: MemmapStore) -> None:
        # Never let build_half_basins zero a hard-linked cached candidate.
        # All V0.1.0 STABLE workers explicitly release this mapping after each
        # threshold. A stale file from a previous process can therefore be
        # removed safely.
        gc.collect()
        if not store.remove("half_basins", best_effort=True):
            raise RuntimeError(
                "half_basins.npy is still locked by another process. "
                "Close any other PySlope Python process using this work_dir "
                "and rerun; the completed caches will be reused."
            )

    def save_from_working(self, key: CandidateKey, store: MemmapStore, stats: dict) -> None:
        src = store.path("half_basins")
        dst = self.raster_path(key)
        tmp = dst.with_suffix(".tmp.npy")
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        payload = dict(stats)
        payload["candidate_semantics"] = CANDIDATE_SEMANTICS
        payload["residual_outlets"] = 0
        payload["residual_mode"] = "final-fill-only"
        self.stats_path(key).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def migrate_legacy(self, key: CandidateKey, *, block_rows: int = 2048, verbose: bool = True) -> bool:
        """Strip legacy residual labels into a new stream-only candidate.

        Returns True if migration was performed.
        """
        if self.exists(key):
            return False
        if not self.legacy_exists(key):
            return False

        old_stats = json.loads(self.legacy_stats_path(key).read_text(encoding="utf-8"))
        nroots = int(old_stats.get("stream_branch_roots", 0))
        if nroots <= 0:
            return False

        legacy_path = self.legacy_raster_path(key)
        src = np.load(legacy_path, mmap_mode="r+", allow_pickle=False)
        dst_path = self.raster_path(key)

        # In-place migration avoids duplicating tens of GB for the 10 m run.
        # Labels above 2*N_stream_roots were introduced only by the legacy
        # residual-component completion and are reset to 0.
        max_stream_label = 2 * nroots
        positive = 0
        rows = src.shape[0]
        for r0 in range(0, rows, block_rows):
            r1 = min(rows, r0 + block_rows)
            block = np.asarray(src[r0:r1])
            bad = block > max_stream_label
            if np.any(bad):
                block[bad] = 0
                src[r0:r1] = block
            positive += int(np.count_nonzero(block))
        src.flush()
        mm = getattr(src, "_mmap", None)
        if mm is not None:
            mm.close()
        del src
        os.replace(legacy_path, dst_path)

        stats = dict(old_stats)
        stats.update(
            {
                "candidate_semantics": CANDIDATE_SEMANTICS,
                "migrated_from_legacy": True,
                "max_half_basin_label": int(max_stream_label),
                "residual_outlets": 0,
                "residual_mode": "final-fill-only",
                "defensive_components": 0,
                "candidate_positive_cells": positive,
            }
        )
        self.stats_path(key).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        try:
            self.legacy_stats_path(key).unlink()
        except FileNotFoundError:
            pass
        if verbose:
            print(
                f"[PySlopeUnits] migrated candidate {key.threshold_cells:,} cells | "
                f"stream labels <= {max_stream_label:,} | positive={positive:,}"
            )
        return True

    def ensure(self, key: CandidateKey, *, verbose: bool = True) -> bool:
        """Return True when a valid candidate is available, migrating if possible."""
        if self.exists(key):
            return True
        self.migrate_legacy(key, verbose=verbose)
        return self.exists(key)

    def activate(self, key: CandidateKey, store: MemmapStore) -> str:
        cached = self.raster_path(key)
        active = store.path("half_basins")
        if active.exists():
            try:
                active.unlink()
            except PermissionError:
                # An old process may have left a stale active mapping. The
                # cached source is authoritative; use a distinct copy below.
                stale = active.with_name(active.stem + "_stale.npy")
                try:
                    os.replace(active, stale)
                except OSError:
                    pass

        try:
            os.link(cached, active)
            return "hardlink"
        except OSError:
            shutil.copyfile(cached, active)
            return "copy"
