from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

import numpy as np


@dataclass(frozen=True)
class SlopeUnitMetrics:
    """Raster-native aspect segmentation metrics.

    ``V`` follows the area-weighted circular-variance metric used by
    r.slopeunits.metrics.

    ``I`` follows the same angular transformation but evaluates adjacency
    directly from raster label pairs rather than converting polygons to a
    GRASS vector topology/database. One unique neighbouring SU pair is counted
    once. This is the PySlope raster-native form and should be benchmarked
    against the GRASS vector implementation.
    """

    V: float
    I: float
    units: int
    adjacency_pairs: int
    valid_cells: int

    def to_dict(self) -> dict:
        return asdict(self)


def _label_stats_blockwise(
    labels: np.ndarray,
    valid: np.ndarray,
    sin_aspect: np.ndarray,
    cos_aspect: np.ndarray,
    *,
    block_rows: int = 512,
):
    max_label = int(np.asarray(labels).max())
    count = np.zeros(max_label + 1, dtype=np.int64)
    aspect_count = np.zeros(max_label + 1, dtype=np.int64)
    sum_sin = np.zeros(max_label + 1, dtype=np.float64)
    sum_cos = np.zeros(max_label + 1, dtype=np.float64)

    rows = labels.shape[0]
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        lab = np.asarray(labels[r0:r1]).ravel().astype(np.int64, copy=False)
        v = np.asarray(valid[r0:r1]).ravel() != 0
        if not np.any(v):
            continue

        lab = lab[v]
        good = lab > 0
        if not np.any(good):
            continue
        lab = lab[good]

        count += np.bincount(lab, minlength=max_label + 1)

        ss = np.asarray(sin_aspect[r0:r1]).ravel()[v][good]
        cc = np.asarray(cos_aspect[r0:r1]).ravel()[v][good]
        av = np.isfinite(ss) & np.isfinite(cc)

        if np.any(av):
            lav = lab[av]
            aspect_count += np.bincount(lav, minlength=max_label + 1)
            sum_sin += np.bincount(lav, weights=ss[av], minlength=max_label + 1)
            sum_cos += np.bincount(lav, weights=cc[av], minlength=max_label + 1)

    return count, aspect_count, sum_sin, sum_cos


def _unique_adjacency_pairs(
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    block_rows: int = 256,
) -> np.ndarray:
    """Return unique undirected 4-neighbour label pairs as uint64 keys."""
    rows, cols = labels.shape
    chunks: list[np.ndarray] = []

    # Include one extra row so vertical boundaries at block seams are retained.
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        rr1 = min(rows, r1 + 1)

        lab = np.asarray(labels[r0:rr1], dtype=np.int64)
        val = np.asarray(valid[r0:rr1]) != 0

        block_keys = []

        # Horizontal neighbours within current rows.
        a = lab[: r1 - r0, :-1]
        b = lab[: r1 - r0, 1:]
        ok = val[: r1 - r0, :-1] & val[: r1 - r0, 1:]
        ok &= (a > 0) & (b > 0) & (a != b)
        if np.any(ok):
            aa = a[ok].astype(np.uint64, copy=False)
            bb = b[ok].astype(np.uint64, copy=False)
            lo = np.minimum(aa, bb)
            hi = np.maximum(aa, bb)
            block_keys.append((lo << np.uint64(32)) | hi)

        # Vertical neighbours; only pairs whose upper row belongs to this block.
        if rr1 - r0 >= 2:
            a = lab[:-1, :]
            b = lab[1:, :]
            ok = val[:-1, :] & val[1:, :]
            ok &= (a > 0) & (b > 0) & (a != b)
            if np.any(ok):
                aa = a[ok].astype(np.uint64, copy=False)
                bb = b[ok].astype(np.uint64, copy=False)
                lo = np.minimum(aa, bb)
                hi = np.maximum(aa, bb)
                block_keys.append((lo << np.uint64(32)) | hi)

        if block_keys:
            chunks.append(np.unique(np.concatenate(block_keys)))

    if not chunks:
        return np.empty(0, dtype=np.uint64)

    return np.unique(np.concatenate(chunks))


def compute_v2_metrics(
    labels: np.ndarray,
    valid: np.ndarray,
    sin_aspect: np.ndarray,
    cos_aspect: np.ndarray,
    *,
    block_rows: int = 512,
    adjacency_block_rows: int = 256,
) -> SlopeUnitMetrics:
    """Calculate PySlope raster-native V and I metrics."""
    count, aspect_count, sum_sin, sum_cos = _label_stats_blockwise(
        labels, valid, sin_aspect, cos_aspect, block_rows=block_rows
    )

    ids = np.flatnonzero(count > 0)
    ids = ids[ids > 0]
    if ids.size == 0:
        return SlopeUnitMetrics(float("nan"), float("nan"), 0, 0, 0)

    cv = np.zeros_like(sum_sin)
    ok = aspect_count > 0
    cv[ok] = 1.0 - np.hypot(sum_sin[ok], sum_cos[ok]) / aspect_count[ok]

    total_cells = int(count[ids].sum())
    V = float(np.sum(count[ids] * cv[ids]) / total_cells)

    # Mean aspect of each SU and the whole valid surface.
    ai = np.arctan2(sum_sin, sum_cos)
    global_sin = float(sum_sin[ids].sum())
    global_cos = float(sum_cos[ids].sum())
    a_all = math.atan2(global_sin, global_cos)

    # Same angular transformation used by the current GRASS metrics module.
    beta = np.arctan2(
        np.sin(ai) + math.sin(a_all),
        np.cos(ai) + math.cos(a_all),
    )

    pairs = _unique_adjacency_pairs(
        labels, valid, block_rows=adjacency_block_rows
    )
    if pairs.size:
        left = (pairs >> np.uint64(32)).astype(np.int64)
        right = (pairs & np.uint64(0xFFFFFFFF)).astype(np.int64)
        I = float(np.mean(np.cos(beta[left] - beta[right])))
    else:
        I = float("nan")

    return SlopeUnitMetrics(
        V=V,
        I=I,
        units=int(ids.size),
        adjacency_pairs=int(pairs.size),
        valid_cells=total_cells,
    )


def write_metrics_json(path: str | Path, metrics: SlopeUnitMetrics, **extra) -> Path:
    path = Path(path)
    payload = metrics.to_dict()
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
