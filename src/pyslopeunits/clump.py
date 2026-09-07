from __future__ import annotations

from pathlib import Path
import numpy as np
from numba import njit

from .memmap_store import MemmapStore


@njit(cache=True)
def _tile_clump(values: np.ndarray):
    rows, cols = values.shape
    n = rows * cols
    labels = np.zeros(n, dtype=np.int32)
    parent = np.arange(n + 1, dtype=np.int32)
    next_label = 0
    vf = values.ravel()

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            v = int(vf[i])
            if v <= 0:
                continue
            left = 0
            up = 0
            if c > 0 and int(vf[i - 1]) == v:
                left = int(labels[i - 1])
            if r > 0 and int(vf[i - cols]) == v:
                up = int(labels[i - cols])

            if left == 0 and up == 0:
                next_label += 1
                labels[i] = next_label
            elif left != 0 and up == 0:
                labels[i] = left
            elif left == 0 and up != 0:
                labels[i] = up
            else:
                a = left
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                b = up
                while parent[b] != b:
                    parent[b] = parent[parent[b]]
                    b = parent[b]
                root = a if a < b else b
                other = b if a < b else a
                parent[other] = root
                labels[i] = root

    # Compact local roots.
    lut = np.zeros(next_label + 1, dtype=np.int32)
    nroots = 0
    for lab in range(1, next_label + 1):
        x = lab
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        if lut[x] == 0:
            nroots += 1
            lut[x] = nroots
        lut[lab] = lut[x]

    for i in range(n):
        if labels[i] > 0:
            labels[i] = lut[labels[i]]

    return labels.reshape((rows, cols)), nroots


def _uf_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return x


def _uf_union(parent: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if ra < rb:
        parent[rb] = ra
    else:
        parent[ra] = rb


def clump_equal_categories(
    categories: np.ndarray,
    valid: np.ndarray,
    store: MemmapStore,
    *,
    tile_rows: int = 1024,
    tile_cols: int = 1024,
    verbose: bool = True,
    output_name: str = "final",
    provisional_name: str = "v14_clump_provisional",
):
    """GRASS-like 4-neighbour connected clump, out-of-core.

    Category distinctions are preserved.  Local tile components are labelled
    independently and only equal-valued components touching across tile seams
    are unioned globally.
    """
    rows, cols = categories.shape
    provisional = store.create(provisional_name, categories.shape, np.int32, fill=0)
    next_global = 0
    equivalence_chunks: list[np.ndarray] = []

    for r0 in range(0, rows, tile_rows):
        r1 = min(rows, r0 + tile_rows)
        for c0 in range(0, cols, tile_cols):
            c1 = min(cols, c0 + tile_cols)
            vals = np.asarray(categories[r0:r1, c0:c1], dtype=np.int64).copy()
            vm = np.asarray(valid[r0:r1, c0:c1], dtype=bool)
            vals[~vm] = 0
            local, nlocal = _tile_clump(vals)
            if nlocal == 0:
                continue
            mask = local > 0
            local[mask] += next_global
            provisional[r0:r1, c0:c1] = local

            pairs = []
            if r0 > 0:
                top_lab = np.asarray(provisional[r0 - 1, c0:c1])
                top_val = np.asarray(categories[r0 - 1, c0:c1])
                cur_lab = local[0, :]
                cur_val = vals[0, :]
                ok = (cur_lab > 0) & (top_lab > 0) & (cur_val == top_val) & (cur_val > 0)
                if np.any(ok):
                    a = cur_lab[ok].astype(np.uint64)
                    b = top_lab[ok].astype(np.uint64)
                    lo = np.minimum(a, b); hi = np.maximum(a, b)
                    pairs.append(np.unique((lo << np.uint64(32)) | hi))
            if c0 > 0:
                left_lab = np.asarray(provisional[r0:r1, c0 - 1])
                left_val = np.asarray(categories[r0:r1, c0 - 1])
                cur_lab = local[:, 0]
                cur_val = vals[:, 0]
                ok = (cur_lab > 0) & (left_lab > 0) & (cur_val == left_val) & (cur_val > 0)
                if np.any(ok):
                    a = cur_lab[ok].astype(np.uint64)
                    b = left_lab[ok].astype(np.uint64)
                    lo = np.minimum(a, b); hi = np.maximum(a, b)
                    pairs.append(np.unique((lo << np.uint64(32)) | hi))
            if pairs:
                equivalence_chunks.append(np.unique(np.concatenate(pairs)))
            next_global += int(nlocal)

        provisional.flush()
        if verbose and (r0 // tile_rows) % 8 == 0:
            print(f"[PySlopeUnits] clump pass 1 | row {r1:,}/{rows:,} | provisional={next_global:,}")

    parent = np.arange(next_global + 1, dtype=np.int32)
    for chunk in equivalence_chunks:
        left = (chunk >> np.uint64(32)).astype(np.int64)
        right = (chunk & np.uint64(0xFFFFFFFF)).astype(np.int64)
        for a, b in zip(left, right):
            _uf_union(parent, int(a), int(b))

    root_to_compact: dict[int, int] = {}
    mapping = np.zeros(next_global + 1, dtype=np.int32)
    nxt = 0
    for lab in range(1, next_global + 1):
        root = _uf_find(parent, lab)
        cid = root_to_compact.get(root)
        if cid is None:
            nxt += 1
            cid = nxt
            root_to_compact[root] = cid
        mapping[lab] = cid

    result = store.create(output_name, categories.shape, np.int32, fill=0)
    for r0 in range(0, rows, tile_rows):
        r1 = min(rows, r0 + tile_rows)
        block = np.asarray(provisional[r0:r1])
        result[r0:r1] = mapping[block]
    result.flush()

    # Windows-safe cleanup. The provisional raster is expendable after final
    # labels are flushed, so cleanup may not abort a valid computation.
    store.close_array(provisional)
    del provisional
    store.remove(provisional_name, best_effort=True)

    if verbose:
        print(f"[PySlopeUnits] 4-neighbour clump complete | units={nxt:,}")
    return result, int(nxt)
