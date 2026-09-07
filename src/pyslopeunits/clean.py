from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import time

import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from scipy.ndimage import (
        distance_transform_edt,
        minimum_filter,
        maximum_filter,
        binary_dilation,
    )
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

from .clump import clump_equal_categories
from .memmap_store import MemmapStore
from .raster import RasterMeta, read_meta, write_raster_blockwise


@dataclass(frozen=True)
class CleanResult:
    input_raster: Path
    output_raster: Path
    method: str
    clean_size_m2: float
    cell_area_m2: float
    clean_size_cells: float
    grow_radius_cells: int
    initial_units: int
    removed_small_units: int
    removed_small_cells: int
    cells_unfilled_after_basic: int
    cells_unfilled_after_quick: int
    final_units: int
    total_seconds: float


def _profile_meta(src: rasterio.DatasetReader) -> RasterMeta:
    return RasterMeta(
        shape=(src.height, src.width),
        transform=src.transform,
        crs=src.crs,
        nodata=src.nodata,
    )


def _count_labels(src: rasterio.DatasetReader) -> tuple[np.ndarray, int]:
    """Two-pass out-of-core label counts."""
    max_id = 0
    for _, win in src.block_windows(1):
        a = src.read(1, window=win)
        if a.size:
            m = int(np.max(a))
            if m > max_id:
                max_id = m

    counts = np.zeros(max_id + 1, dtype=np.int64)
    for _, win in src.block_windows(1):
        a = src.read(1, window=win).astype(np.int64, copy=False)
        good = a > 0
        if np.any(good):
            counts += np.bincount(a[good], minlength=max_id + 1)
    return counts, max_id


def _window_with_halo(
    rows: int,
    cols: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    halo: int,
):
    rr0 = max(0, r0 - halo)
    rr1 = min(rows, r1 + halo)
    cc0 = max(0, c0 - halo)
    cc1 = min(cols, c1 + halo)
    cr0 = r0 - rr0
    cc = c0 - cc0
    return rr0, rr1, cc0, cc1, cr0, cc


def _grow_from_seed_blockwise(
    source,
    out,
    valid_source,
    *,
    radius: float,
    tile_rows: int = 1536,
    tile_cols: int = 1536,
    only_fill_zeros: bool = False,
    verbose: bool = True,
) -> int:
    """Euclidean nearest-label growth with bounded radius, out-of-core.

    `source` and `valid_source` may be rasterio datasets or 2-D arrays/memmaps.
    The halo equals ceil(radius), so nearest allocation inside the central tile
    is exact for all cells whose nearest seed is within that radius.
    """
    if not HAVE_SCIPY:
        raise RuntimeError(
            "PySlope clean requires scipy for bounded Euclidean growth. "
            "Install with: pip install scipy"
        )

    if hasattr(source, "height"):
        rows, cols = source.height, source.width
        def read_source(rr0, rr1, cc0, cc1):
            return source.read(
                1, window=Window(cc0, rr0, cc1-cc0, rr1-rr0)
            )
    else:
        rows, cols = source.shape
        def read_source(rr0, rr1, cc0, cc1):
            return np.asarray(source[rr0:rr1, cc0:cc1])

    if hasattr(valid_source, "height"):
        def read_valid(rr0, rr1, cc0, cc1):
            a = valid_source.read(
                1, window=Window(cc0, rr0, cc1-cc0, rr1-rr0)
            )
            return a > 0
    else:
        def read_valid(rr0, rr1, cc0, cc1):
            return np.asarray(valid_source[rr0:rr1, cc0:cc1]) > 0

    halo = int(math.ceil(radius))
    unfilled = 0
    tile_no = 0
    ntiles = math.ceil(rows/tile_rows) * math.ceil(cols/tile_cols)

    for r0 in range(0, rows, tile_rows):
        r1 = min(rows, r0 + tile_rows)
        for c0 in range(0, cols, tile_cols):
            c1 = min(cols, c0 + tile_cols)
            rr0, rr1, cc0, cc1, cr0, ccentral = _window_with_halo(
                rows, cols, r0, r1, c0, c1, halo
            )
            src = read_source(rr0, rr1, cc0, cc1).astype(np.int32, copy=False)
            valid = read_valid(rr0, rr1, cc0, cc1)

            seeds = (src > 0) & valid
            center_h = r1-r0
            center_w = c1-c0
            center_valid = valid[
                cr0:cr0+center_h,
                ccentral:ccentral+center_w
            ]
            center_src = src[
                cr0:cr0+center_h,
                ccentral:ccentral+center_w
            ]

            result = np.zeros((center_h, center_w), dtype=np.int32)

            if np.any(seeds):
                dist, inds = distance_transform_edt(
                    ~seeds,
                    return_distances=True,
                    return_indices=True,
                )
                nearest = src[inds[0], inds[1]]
                center_dist = dist[
                    cr0:cr0+center_h,
                    ccentral:ccentral+center_w
                ]
                center_nearest = nearest[
                    cr0:cr0+center_h,
                    ccentral:ccentral+center_w
                ]
                take = center_valid & (center_dist <= float(radius))
                result[take] = center_nearest[take]

            if only_fill_zeros:
                existing = np.asarray(out[r0:r1, c0:c1])
                keep = existing > 0
                result[keep] = existing[keep]

            out[r0:r1, c0:c1] = result
            unfilled += int(np.count_nonzero(center_valid & (result == 0)))

            tile_no += 1
            if verbose and tile_no % 50 == 0:
                print(
                    f"[PySlopeUnits clean] grow tiles {tile_no:,}/{ntiles:,} | "
                    f"unfilled-so-far={unfilled:,}"
                )

    out.flush()
    return int(unfilled)


def _make_seed_from_kept_labels(
    src: rasterio.DatasetReader,
    kept: np.ndarray,
    target,
    *,
    block_rows: int = 2048,
) -> int:
    rows, cols = src.height, src.width
    removed_cells = 0
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        a = src.read(1, window=Window(0, r0, cols, r1-r0))
        out = np.zeros(a.shape, dtype=np.int32)
        good = a > 0
        kg = np.zeros(a.shape, dtype=bool)
        if np.any(good):
            kg[good] = kept[a[good].astype(np.int64)]
            out[kg] = a[kg]
            removed_cells += int(np.count_nonzero(good & ~kg))
        target[r0:r1] = out
    target.flush()
    return removed_cells


def _quick_interior_seed(
    labels,
    target,
    *,
    tile_rows: int = 1536,
    tile_cols: int = 1536,
    verbose: bool = True,
) -> int:
    """GRASS -m analogue: diversity==1 in 5x5, then grow by radius 1.01.

    Diversity==1 is exactly equivalent to min==max among positive values in
    the 5x5 window. Nodata 0 is ignored.
    """
    if not HAVE_SCIPY:
        raise RuntimeError("scipy is required for grass_quick cleaning")

    rows, cols = labels.shape
    halo = 3
    retained = 0

    cross = np.array(
        [[0,1,0],
         [1,1,1],
         [0,1,0]], dtype=bool
    )

    for r0 in range(0, rows, tile_rows):
        r1 = min(rows, r0 + tile_rows)
        for c0 in range(0, cols, tile_cols):
            c1 = min(cols, c0 + tile_cols)
            rr0, rr1, cc0, cc1, cr0, cc = _window_with_halo(
                rows, cols, r0, r1, c0, c1, halo
            )
            a = np.asarray(labels[rr0:rr1, cc0:cc1], dtype=np.int32)

            pos = a > 0
            amin = a.copy()
            amin[~pos] = np.iinfo(np.int32).max
            amax = a.copy()
            amax[~pos] = 0

            mn = minimum_filter(
                amin, size=5, mode="constant",
                cval=np.iinfo(np.int32).max
            )
            mx = maximum_filter(
                amax, size=5, mode="constant", cval=0
            )
            interior = pos & (mn == mx) & (mx > 0)

            # r.grow radius=1.01: cardinal one-cell expansion, no diagonals.
            interior_grow = binary_dilation(
                interior,
                structure=cross,
                iterations=1,
                border_value=0,
            )

            h, w = r1-r0, c1-c0
            central_a = a[cr0:cr0+h, cc:cc+w]
            central_mask = interior_grow[cr0:cr0+h, cc:cc+w]
            out = np.zeros((h, w), dtype=np.int32)
            take = central_mask & (central_a > 0)
            out[take] = central_a[take]
            target[r0:r1, c0:c1] = out
            retained += int(np.count_nonzero(take))

    target.flush()
    if verbose:
        print(f"[PySlopeUnits clean] quick interior seed cells={retained:,}")
    return retained


def _adaptive_fill_all(
    seed,
    valid,
    out,
    *,
    step_radius: int = 32,
    max_radius: int = 1000,
    tile_rows: int = 1536,
    tile_cols: int = 1536,
    verbose: bool = True,
) -> int:
    """Fill gaps by repeated bounded Euclidean growth.

    GRASS quick mode uses `r.grow radius=1000` after border stripping.
    A single 1000-cell halo is impractical for >400M cells. Repeated exact
    bounded EDT passes use newly allocated cells as seeds and preserve the same
    intent while remaining out-of-core. In typical slope-unit maps gaps are
    only a few cells wide and the first pass fills them completely.
    """
    store_arr = seed
    rows, cols = seed.shape

    # First copy seed to output.
    for r0 in range(0, rows, 2048):
        r1 = min(rows, r0+2048)
        out[r0:r1] = np.asarray(seed[r0:r1])
    out.flush()

    total_radius = 0
    unfilled = int(np.count_nonzero((np.asarray(valid) > 0) & (np.asarray(out) == 0))) \
        if rows * cols <= 20_000_000 else -1

    while total_radius < max_radius:
        radius = min(step_radius, max_radius-total_radius)
        unfilled = _grow_from_seed_blockwise(
            out,
            out,
            valid,
            radius=radius,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            only_fill_zeros=True,
            verbose=verbose,
        )
        total_radius += radius

        if verbose:
            print(
                f"[PySlopeUnits clean] quick regrow cumulative-radius="
                f"{total_radius} cells | unfilled={unfilled:,}"
            )

        if unfilled == 0:
            break

    return int(unfilled)


def clean_slope_units(
    input_raster: str | Path,
    output_raster: str | Path,
    *,
    work_dir: str | Path,
    clean_size_m2: float = 25_000.0,
    method: str = "grass_quick",
    tile_rows: int = 1536,
    tile_cols: int = 1536,
    reclump: bool = True,
    verbose: bool = True,
) -> CleanResult:
    """Clean a PySlope categorical raster.

    Methods
    -------
    grass_basic
        Reproduces the main raster logic of r.slopeunits.clean:
        clump/count -> remove <= cleansize -> bounded nearest growth.

    grass_quick
        Adds the `-m` concept: 5x5 single-category interiors, one-cell
        cardinal restoration, then large-radius regrowth. The large GRASS
        radius is implemented as repeated bounded EDT passes for out-of-core
        scalability.

    The final raster is optionally reclumped so every disconnected polygon has
    a unique integer slope-unit ID, which is important for metrics and GPKG.
    """
    t0 = time.perf_counter()
    input_raster = Path(input_raster)
    output_raster = Path(output_raster)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    store = MemmapStore(work_dir / "clean_memmap")

    method = method.lower()
    if method not in ("grass_basic", "grass_quick"):
        raise ValueError("method must be 'grass_basic' or 'grass_quick'")
    if clean_size_m2 <= 0:
        raise ValueError("clean_size_m2 must be > 0")

    with rasterio.open(input_raster) as src:
        meta = _profile_meta(src)
        cell_area = float(meta.cell_area)
        clean_cells = float(clean_size_m2) / cell_area

        # GRASS code:
        # growdist = int((10 * cleansize_cells / pi) ** 0.5)
        grow_radius = max(
            1, int(math.sqrt(10.0 * clean_cells / math.pi))
        )

        if verbose:
            print(
                f"[PySlopeUnits clean] {method} | cleansize={clean_size_m2:,.0f} m² "
                f"| {clean_cells:.2f} cells | grow={grow_radius} cells"
            )

        counts, max_id = _count_labels(src)
        initial_units = int(np.count_nonzero(counts[1:] > 0))

        keep = counts.astype(np.float64) * cell_area > float(clean_size_m2)
        if keep.size:
            keep[0] = False

        removed_ids = (counts > 0) & ~keep
        if removed_ids.size:
            removed_ids[0] = False
        removed_units = int(np.count_nonzero(removed_ids))
        removed_small_cells = int(counts[removed_ids].sum())

        if verbose:
            print(
                f"[PySlopeUnits clean] initial={initial_units:,} | "
                f"small-units={removed_units:,} | "
                f"small-cells={removed_small_cells:,}"
            )

        seed = store.create(
            "clean_seed", meta.shape, np.int32, fill=0
        )
        _make_seed_from_kept_labels(src, keep, seed)

        basic = store.create(
            "clean_basic", meta.shape, np.int32, fill=0
        )

        # valid target is the original positive slope-unit domain.
        valid = store.create(
            "clean_valid", meta.shape, np.uint8, fill=0
        )
        rows, cols = meta.shape
        for r0 in range(0, rows, 2048):
            r1 = min(rows, r0+2048)
            a = src.read(
                1, window=Window(0, r0, cols, r1-r0)
            )
            valid[r0:r1] = (a > 0).astype(np.uint8)
        valid.flush()

        unfilled_basic = _grow_from_seed_blockwise(
            seed,
            basic,
            valid,
            radius=float(grow_radius),
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            verbose=verbose,
        )

    store.close_array(seed)
    del seed
    store.remove("clean_seed", best_effort=True)

    current = basic
    unfilled_quick = unfilled_basic

    if method == "grass_quick":
        stripe_seed = store.create(
            "clean_stripe_seed", current.shape, np.int32, fill=0
        )
        _quick_interior_seed(
            current,
            stripe_seed,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            verbose=verbose,
        )

        quick = store.create(
            "clean_quick", current.shape, np.int32, fill=0
        )
        unfilled_quick = _adaptive_fill_all(
            stripe_seed,
            valid,
            quick,
            step_radius=max(8, min(64, grow_radius * 2)),
            max_radius=1000,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            verbose=verbose,
        )

        store.close_array(stripe_seed)
        del stripe_seed
        store.remove("clean_stripe_seed", best_effort=True)

        store.close_array(current)
        del current
        store.remove("clean_basic", best_effort=True)
        current = quick

    # Any cell still not allocated after the maximum GRASS-equivalent grow
    # retains its original category rather than becoming a hole.
    if unfilled_quick > 0:
        if verbose:
            print(
                f"[PySlopeUnits clean] fallback: restoring {unfilled_quick:,} "
                f"unfilled valid cells from raw raster"
            )
        with rasterio.open(input_raster) as src:
            rows, cols = current.shape
            for r0 in range(0, rows, 2048):
                r1 = min(rows, r0+2048)
                raw = src.read(
                    1, window=Window(0, r0, cols, r1-r0)
                )
                block = np.asarray(current[r0:r1])
                holes = (block == 0) & (raw > 0)
                if np.any(holes):
                    block = block.copy()
                    block[holes] = raw[holes]
                    current[r0:r1] = block
        current.flush()

    if reclump:
        if verbose:
            print("[PySlopeUnits clean] final 4-neighbour reclump")
        cleaned, final_units = clump_equal_categories(
            current,
            valid,
            store,
            tile_rows=1024,
            tile_cols=1024,
            verbose=verbose,
            output_name="clean_final",
            provisional_name="clean_clump_provisional",
        )
    else:
        cleaned = current
        final_units = -1

    write_raster_blockwise(
        output_raster,
        cleaned,
        meta,
        valid,
        dtype="int32",
        nodata=0,
    )

    total = time.perf_counter() - t0
    result = CleanResult(
        input_raster=input_raster,
        output_raster=output_raster,
        method=method,
        clean_size_m2=float(clean_size_m2),
        cell_area_m2=cell_area,
        clean_size_cells=clean_cells,
        grow_radius_cells=grow_radius,
        initial_units=initial_units,
        removed_small_units=removed_units,
        removed_small_cells=removed_small_cells,
        cells_unfilled_after_basic=int(unfilled_basic),
        cells_unfilled_after_quick=int(unfilled_quick),
        final_units=int(final_units),
        total_seconds=total,
    )

    report_path = output_raster.with_suffix(
        output_raster.suffix + ".clean.json"
    )
    payload = asdict(result)
    payload["input_raster"] = str(input_raster)
    payload["output_raster"] = str(output_raster)
    report_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(
            f"[PySlopeUnits clean] complete | final-units={final_units:,} | "
            f"{total/60:.2f} min"
        )

    return result
