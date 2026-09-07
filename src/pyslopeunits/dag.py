from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import gc

import numpy as np

from .candidate_cache import MultiThresholdCandidateCache, CandidateKey, CANDIDATE_SEMANTICS
from .memmap_store import MemmapStore
from .schedule import threshold_schedule


DAG_VERSION = 2


@dataclass(frozen=True)
class DAGBuildResult:
    nodes: int
    levels: int
    terminal_alive_cells: int
    seconds: float
    cache_hit: bool


class CandidateDAG:
    """Nested intersection DAG built from non-nested half-basin candidates.

    A node is the exact raster intersection of one parent node and one current
    threshold half-basin.  This makes the hierarchy nested by construction even
    when independent r.watershed half-basin maps cross each other.
    """

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.root = self.work_dir / "dag_v14"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.npy"

    def metadata_path(self) -> Path:
        return self.root / "dag.json"

    def _signature(self, shape, schedule_cells):
        return {
            "dag_version": DAG_VERSION,
            "candidate_semantics": CANDIDATE_SEMANTICS,
            "shape": list(shape),
            "schedule_cells": list(map(int, schedule_cells)),
        }

    def valid(self, shape, schedule_cells) -> bool:
        needed = [
            "parent", "level", "hb_id", "count", "aspect_count",
            "sum_sin", "sum_cos", "child_offsets", "children",
        ]
        if not self.metadata_path().exists() or not all(self.path(n).exists() for n in needed):
            return False
        try:
            got = json.loads(self.metadata_path().read_text(encoding="utf-8"))
        except Exception:
            return False
        return got.get("signature") == self._signature(shape, schedule_cells)

    def load(self):
        return {
            n: np.load(self.path(n), mmap_mode="r", allow_pickle=False)
            for n in (
                "parent", "level", "hb_id", "count", "aspect_count",
                "sum_sin", "sum_cos", "child_offsets", "children",
            )
        }

    def _array_set_complete(self, shape) -> tuple[bool, int]:
        """Validate node arrays written at the end of a V14.1 DAG build."""
        node_names = (
            "parent", "level", "hb_id", "count",
            "aspect_count", "sum_sin", "sum_cos",
        )
        if not all(self.path(n).exists() for n in node_names):
            return False, 0
        if not self.path("child_offsets").exists() or not self.path("children").exists():
            return False, 0

        try:
            arrays = {
                n: np.load(self.path(n), mmap_mode="r", allow_pickle=False)
                for n in node_names
            }
            lengths = {int(a.size) for a in arrays.values()}
            if len(lengths) != 1:
                return False, 0

            size = lengths.pop()
            if size < 2:
                return False, 0
            n_nodes = size - 1

            offsets = np.load(
                self.path("child_offsets"), mmap_mode="r", allow_pickle=False
            )
            children = np.load(
                self.path("children"), mmap_mode="r", allow_pickle=False
            )

            ok = (
                offsets.ndim == 1
                and offsets.size == n_nodes + 2
                and children.ndim == 1
                and children.size == max(0, n_nodes - 1)
                and int(offsets[-1]) == int(children.size)
                and int(arrays["parent"][1]) == 0
            )

            for a in arrays.values():
                mm = getattr(a, "_mmap", None)
                if mm is not None:
                    mm.close()
            mm = getattr(offsets, "_mmap", None)
            if mm is not None:
                mm.close()
            mm = getattr(children, "_mmap", None)
            if mm is not None:
                mm.close()

            return bool(ok), int(n_nodes)
        except Exception:
            return False, 0

    def _recover_finished_v14_1_build(
        self,
        store: MemmapStore,
        shape,
        schedule_cells,
        *,
        verbose: bool,
    ) -> DAGBuildResult | None:
        """Recover the exact V14.1 WinError32 state.

        V14.1 wrote all node arrays and child adjacency arrays before it tried
        to delete ``v14_dag_alive.npy``. Metadata was written only after that
        deletion. Therefore a complete array set + terminal raster + alive
        raster, but no valid metadata, is a completed DAG whose only failure
        was cleanup.
        """
        if self.valid(shape, schedule_cells):
            return None

        if not store.exists("v14_terminal_node"):
            return None

        complete, n_nodes = self._array_set_complete(shape)
        if not complete:
            return None

        alive_cells = 0
        if store.exists("v14_dag_alive"):
            alive = store.open("v14_dag_alive", "r")
            try:
                alive_cells = int(np.count_nonzero(alive))
            finally:
                store.close_array(alive)
                del alive

        level_arr = np.load(
            self.path("level"), mmap_mode="r", allow_pickle=False
        )
        try:
            level_ranges = []
            for lev in range(1, len(schedule_cells) + 1):
                ids = np.flatnonzero(level_arr == lev)
                if ids.size:
                    level_ranges.append(
                        (int(ids[0]), int(ids[-1]) + 1)
                    )
                else:
                    level_ranges.append((0, 0))
        finally:
            mm = getattr(level_arr, "_mmap", None)
            if mm is not None:
                mm.close()
            del level_arr

        metadata = {
            "signature": self._signature(shape, schedule_cells),
            "nodes": int(n_nodes),
            "levels": len(schedule_cells),
            "level_ranges": level_ranges,
            "terminal_alive_cells": int(alive_cells),
            "seconds": 0.0,
            "recovered_from": "v14.1_windows_alive_cleanup_failure",
        }

        # Durable metadata first. Cleanup is never allowed to invalidate this.
        self.metadata_path().write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        store.remove("v14_dag_alive", best_effort=True)

        if verbose:
            print(
                f"[PySlopeUnits] recovered completed V14.1 DAG | "
                f"nodes={n_nodes:,} | no DAG rebuild"
            )

        return DAGBuildResult(
            nodes=int(n_nodes),
            levels=len(schedule_cells),
            terminal_alive_cells=int(alive_cells),
            seconds=0.0,
            cache_hit=True,
        )

    def build(
        self,
        store: MemmapStore,
        candidate_cache: MultiThresholdCandidateCache,
        *,
        threshold_m2: float,
        cell_area_m2: float,
        reduction_factor: int,
        max_iterations: int,
        block_rows: int = 512,
        verbose: bool = True,
        force: bool = False,
    ) -> DAGBuildResult:
        t0 = time.perf_counter()
        valid = store.open("valid", "r")
        sin_a = store.open("sin_aspect", "r")
        cos_a = store.open("cos_aspect", "r")
        aspect_valid = store.open("aspect_valid", "r")
        shape = valid.shape
        schedule = threshold_schedule(threshold_m2, cell_area_m2, reduction_factor, max_iterations)
        cells = [x.cells for x in schedule]

        if not force and self.valid(shape, cells) and store.exists("v14_terminal_node"):
            meta = json.loads(self.metadata_path().read_text(encoding="utf-8"))
            if verbose:
                print(
                    f"[PySlopeUnits] reusing DAG cache | "
                    f"nodes={int(meta['nodes']):,}"
                )
            return DAGBuildResult(
                nodes=int(meta["nodes"]), levels=len(cells),
                terminal_alive_cells=int(meta.get("terminal_alive_cells", 0)),
                seconds=time.perf_counter() - t0, cache_hit=True,
            )

        if not force:
            recovered = self._recover_finished_v14_1_build(
                store, shape, cells, verbose=verbose
            )
            if recovered is not None:
                return recovered

        if verbose:
            print("[PySlopeUnits] building nested candidate intersection DAG")

        # Root node 1 is the full valid DEM domain.
        terminal = store.create("v14_terminal_node", shape, np.int32, fill=0)
        alive = store.create("v14_dag_alive", shape, np.uint8, fill=0)

        root_count = 0
        root_acount = 0
        root_ss = 0.0
        root_cc = 0.0
        rows = shape[0]
        for r0 in range(0, rows, block_rows):
            r1 = min(rows, r0 + block_rows)
            vm = np.asarray(valid[r0:r1], dtype=bool)
            terminal[r0:r1][vm] = 1
            alive[r0:r1][vm] = 1
            root_count += int(vm.sum())
            av = np.asarray(aspect_valid[r0:r1], dtype=bool) & vm
            root_acount += int(av.sum())
            root_ss += float(np.asarray(sin_a[r0:r1])[av].sum(dtype=np.float64))
            root_cc += float(np.asarray(cos_a[r0:r1])[av].sum(dtype=np.float64))
        terminal.flush(); alive.flush()

        parent = [0, 0]
        level_arr = [0, 0]
        hb_id = [0, 0]
        count = [0, root_count]
        acount = [0, root_acount]
        ss = [0.0, root_ss]
        cc = [0.0, root_cc]
        next_node = 1
        level_ranges = []

        for level in schedule:
            key = CandidateKey(level.cells)
            if not candidate_cache.ensure(key, verbose=verbose):
                raise RuntimeError(f"candidate cache missing for threshold {level.cells} cells")
            hb = np.load(candidate_cache.raster_path(key), mmap_mode="r", allow_pickle=False)
            pair_to_node: dict[int, int] = {}
            level_start = next_node + 1
            alive_before = int(np.count_nonzero(alive))

            for r0 in range(0, rows, block_rows):
                r1 = min(rows, r0 + block_rows)
                pblock = np.asarray(terminal[r0:r1]).copy()
                ablock = np.asarray(alive[r0:r1], dtype=bool).copy()
                hblock = np.asarray(hb[r0:r1])

                positive = ablock & (hblock > 0)
                dropped = ablock & ~positive
                if np.any(dropped):
                    alive[r0:r1][dropped] = 0

                if not np.any(positive):
                    continue

                pv = pblock[positive].astype(np.uint64, copy=False)
                hv = hblock[positive].astype(np.uint64, copy=False)
                keys = (pv << np.uint64(32)) | hv
                uniq, inv = np.unique(keys, return_inverse=True)

                bcount = np.bincount(inv, minlength=uniq.size).astype(np.int64)
                av0 = np.asarray(aspect_valid[r0:r1])[positive].astype(np.float64, copy=False)
                sin0 = np.asarray(sin_a[r0:r1])[positive].astype(np.float64, copy=False)
                cos0 = np.asarray(cos_a[r0:r1])[positive].astype(np.float64, copy=False)
                bac = np.bincount(inv, weights=av0, minlength=uniq.size)
                bss = np.bincount(inv, weights=sin0 * av0, minlength=uniq.size)
                bcc = np.bincount(inv, weights=cos0 * av0, minlength=uniq.size)

                gids = np.empty(uniq.size, dtype=np.int32)
                for q, u0 in enumerate(uniq):
                    u = int(u0)
                    gid = pair_to_node.get(u)
                    if gid is None:
                        next_node += 1
                        gid = next_node
                        pair_to_node[u] = gid
                        par = int(u >> 32)
                        h = int(u & 0xFFFFFFFF)
                        parent.append(par)
                        level_arr.append(int(level.level))
                        hb_id.append(h)
                        count.append(0)
                        acount.append(0)
                        ss.append(0.0)
                        cc.append(0.0)
                    gids[q] = gid
                    count[gid] += int(bcount[q])
                    acount[gid] += int(round(float(bac[q])))
                    ss[gid] += float(bss[q])
                    cc[gid] += float(bcc[q])

                newids = gids[inv]
                out = pblock
                out[positive] = newids
                terminal[r0:r1] = out

            terminal.flush(); alive.flush()
            level_end = next_node + 1
            level_ranges.append((level_start, level_end))
            alive_after = int(np.count_nonzero(alive))
            if verbose:
                print(
                    f"[PySlopeUnits] DAG level {level.level:02d} | "
                    f"threshold={level.cells:,} cells | nodes={level_end-level_start:,} | "
                    f"alive={alive_after:,}/{alive_before:,}"
                )

        # Save node arrays.
        arrays = {
            "parent": np.asarray(parent, dtype=np.int32),
            "level": np.asarray(level_arr, dtype=np.uint8),
            "hb_id": np.asarray(hb_id, dtype=np.int32),
            "count": np.asarray(count, dtype=np.int64),
            "aspect_count": np.asarray(acount, dtype=np.int64),
            "sum_sin": np.asarray(ss, dtype=np.float64),
            "sum_cos": np.asarray(cc, dtype=np.float64),
        }
        for name, arr in arrays.items():
            np.save(self.path(name), arr, allow_pickle=False)

        n_nodes = next_node
        child_count = np.zeros(n_nodes + 1, dtype=np.int64)
        par = arrays["parent"]
        for nid in range(2, n_nodes + 1):
            child_count[int(par[nid])] += 1
        offsets = np.zeros(n_nodes + 2, dtype=np.int64)
        np.cumsum(child_count, out=offsets[1:n_nodes+2])
        children = np.empty(n_nodes - 1, dtype=np.int32)
        cursor = offsets[:-1].copy()
        for nid in range(2, n_nodes + 1):
            pid = int(par[nid])
            pos = int(cursor[pid])
            children[pos] = nid
            cursor[pid] += 1
        np.save(self.path("child_offsets"), offsets, allow_pickle=False)
        np.save(self.path("children"), children, allow_pickle=False)

        terminal_alive = int(np.count_nonzero(alive))
        metadata = {
            "signature": self._signature(shape, cells),
            "nodes": n_nodes,
            "levels": len(cells),
            "level_ranges": level_ranges,
            "terminal_alive_cells": terminal_alive,
            "seconds": time.perf_counter() - t0,
        }

        # Checkpoint FIRST. A Windows cleanup failure must never invalidate a
        # completed 7-million-node DAG.
        self.metadata_path().write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        store.close_array(alive)
        del alive
        gc.collect()
        removed = store.remove("v14_dag_alive", best_effort=True)
        if verbose and not removed:
            print(
                "[PySlopeUnits] warning: v14_dag_alive.npy remains locked; "
                "DAG is complete and the stale temp file will be ignored"
            )

        if verbose:
            print(
                f"[PySlopeUnits] DAG complete | nodes={n_nodes:,} | "
                f"{metadata['seconds']/60:.2f} min"
            )

        return DAGBuildResult(
            nodes=n_nodes, levels=len(cells), terminal_alive_cells=terminal_alive,
            seconds=float(metadata["seconds"]), cache_hit=False,
        )
