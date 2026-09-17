from __future__ import annotations

from .logging_utils import log as print

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import numpy as np
import rasterio
from numba import njit
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window

from .engine import SlopeUnits as ResearchSlopeUnits
from .memmap_store import MemmapStore
from .raster import read_meta


@dataclass(frozen=True)
class HydrologicalDomain:
    domain_id: int
    coarse_cells: int
    estimated_fine_cells: int
    area_m2: float
    bounds: tuple[float, float, float, float]
    row_min: int
    row_max: int
    col_min: int
    col_max: int


@dataclass(frozen=True)
class HydrologicalDomainPlan:
    mode: str
    coarse_dem: Path | None
    domain_raster: Path | None
    domain_vector: Path | None
    target_crs: str
    fine_resolution_m: float
    coarse_resolution_m: float
    max_fine_cells: int
    max_coarse_cells: int
    source_fine_cells_est: int
    domains: list[HydrologicalDomain]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["coarse_dem"] = None if self.coarse_dem is None else str(self.coarse_dem)
        data["domain_raster"] = None if self.domain_raster is None else str(self.domain_raster)
        data["domain_vector"] = None if self.domain_vector is None else str(self.domain_vector)
        return data


def projected_source_bounds(source, target_crs: str) -> tuple[float, float, float, float]:
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError("Source DEM has no CRS")
        return transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)


def source_grid_info(source, target_crs: str, fine_resolution_m: float) -> tuple[int, bool]:
    """Estimate target fine cells and report whether source already matches target grid."""
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError("Source DEM has no CRS")

        same_crs = str(src.crs) == str(rasterio.crs.CRS.from_user_input(target_crs))
        xres = abs(float(src.transform.a))
        yres = abs(float(src.transform.e))
        tol = max(1e-6, fine_resolution_m * 1e-6)
        same_res = abs(xres - fine_resolution_m) <= tol and abs(yres - fine_resolution_m) <= tol

        if same_crs and same_res:
            return int(src.width) * int(src.height), True

        left, bottom, right, top = transform_bounds(
            src.crs, target_crs, *src.bounds, densify_pts=21
        )
        width = max(1, int(math.ceil((right - left) / fine_resolution_m)))
        height = max(1, int(math.ceil((top - bottom) / fine_resolution_m)))
        return int(width) * int(height), False



def estimate_valid_fine_cells_from_coarse(
    coarse_dem: str | Path,
    *,
    fine_resolution_m: float,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Estimate valid fine-grid cells from the valid coarse DEM footprint.

    Returns ``(estimated_fine_cells, valid_coarse_cells, extent)`` where
    extent is ``(row_min, row_max_exclusive, col_min, col_max_exclusive)``.
    The coarse DEM is small enough to scan block-wise without material RAM use.
    """
    coarse_dem = Path(coarse_dem)
    valid_count = 0
    row_min = None
    row_max = None
    col_min = None
    col_max = None

    with rasterio.open(coarse_dem) as src:
        coarse_cell_area = abs(float(src.transform.a) * float(src.transform.e))
        for _, window in src.block_windows(1):
            mask = src.read_masks(1, window=window) > 0
            if not np.any(mask):
                continue
            valid_count += int(np.count_nonzero(mask))
            rr, cc = np.nonzero(mask)
            r0 = int(window.row_off) + int(rr.min())
            r1 = int(window.row_off) + int(rr.max()) + 1
            c0 = int(window.col_off) + int(cc.min())
            c1 = int(window.col_off) + int(cc.max()) + 1
            row_min = r0 if row_min is None else min(row_min, r0)
            row_max = r1 if row_max is None else max(row_max, r1)
            col_min = c0 if col_min is None else min(col_min, c0)
            col_max = c1 if col_max is None else max(col_max, c1)

    if valid_count == 0:
        raise ValueError(f"Coarse DEM contains no valid cells: {coarse_dem}")

    scale = coarse_cell_area / (float(fine_resolution_m) ** 2)
    estimated_fine = int(math.ceil(valid_count * scale))
    return estimated_fine, valid_count, (row_min, row_max, col_min, col_max)

def build_coarse_dem(
    source,
    output: str | Path,
    *,
    target_crs: str,
    resolution_m: float = 480.0,
    max_block_cells: int = 2_000_000,
    resampling: Resampling = Resampling.average,
    verbose: bool = True,
) -> Path:
    """Build a projected coarse DEM using bounded block reads.

    The coarse DEM is used only to derive processing domains. It never defines
    final slope-unit boundaries.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".json")

    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError("Source DEM has no CRS")
        left, bottom, right, top = transform_bounds(
            src.crs, target_crs, *src.bounds, densify_pts=21
        )
        width = max(1, int(math.ceil((right - left) / resolution_m)))
        height = max(1, int(math.ceil((top - bottom) / resolution_m)))
        transform = from_origin(left, top, resolution_m, resolution_m)
        resampling_name = getattr(resampling, "name", str(resampling))
        expected_manifest = {
            "source": str(source),
            "source_crs": str(src.crs),
            "source_width": int(src.width),
            "source_height": int(src.height),
            "source_bounds": [float(x) for x in src.bounds],
            "target_crs": str(target_crs),
            "resolution_m": float(resolution_m),
            "width": int(width),
            "height": int(height),
            "resampling": resampling_name,
        }

        if output.exists() and manifest_path.exists():
            try:
                got = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                got = None
            if got == expected_manifest:
                if verbose:
                    print(f"[PySlopeUnits large-area] reusing coarse DEM: {output}")
                return output

        nodata = -9999.0
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": target_crs,
            "transform": transform,
            "nodata": nodata,
            "compress": "zstd",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }

        block_rows = max(1, min(height, max_block_cells // max(1, width)))
        if verbose:
            print(
                f"[PySlopeUnits large-area] coarse DEM | {width:,} x {height:,} | "
                f"{resolution_m:g} m | block_rows={block_rows:,}"
            )

        with WarpedVRT(
            src,
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=resampling,
            nodata=nodata,
        ) as vrt, rasterio.open(output, "w", **profile) as dst:
            for r0 in range(0, height, block_rows):
                nr = min(block_rows, height - r0)
                window = Window(0, r0, width, nr)
                arr = vrt.read(1, window=window, masked=True, out_dtype="float32")
                dst.write(np.asarray(arr.filled(nodata), dtype=np.float32), 1, window=window)

        manifest_path.write_text(
            json.dumps(expected_manifest, indent=2), encoding="utf-8"
        )

    return output


@njit(cache=True)
def _adaptive_tree_partition(receiver, valid, order, max_domain_cells):
    """Partition a single-receiver drainage forest into large bounded subtrees.

    Unlike the previous implementation, the memory limit is applied directly;
    it is not divided by maximum donor count. A donor subtree is cut only when
    adding it to its receiver would exceed the domain budget. This produces far
    fewer, fuller hydrological processing domains.
    """
    rec = receiver.ravel()
    vf = valid.ravel()
    n = rec.size
    limit = max(2, int(max_domain_cells))

    pending = np.zeros(n, dtype=np.int64)
    cut = np.zeros(n, dtype=np.uint8)

    # Reverse topological order: upstream -> downstream.
    # pending[i] is the size of the not-yet-cut upstream subtree entering i.
    for pos in range(order.size - 1, -1, -1):
        i = int(order[pos])
        if vf[i] == 0:
            continue

        pending[i] += 1  # own cell
        j = int(rec[i])
        is_outlet = j < 0 or vf[j] == 0

        if is_outlet or pending[i] >= limit:
            cut[i] = 1
            continue

        # Keep one cell of headroom for the receiver itself.
        if pending[j] + pending[i] > limit - 1:
            cut[i] = 1
        else:
            pending[j] += pending[i]

    labels = np.zeros(n, dtype=np.int32)
    sizes_tmp = np.zeros(n + 1, dtype=np.int64)
    next_id = 0

    # Downstream -> upstream: receiver label is already known.
    for pos in range(order.size):
        i = int(order[pos])
        if vf[i] == 0:
            continue
        j = int(rec[i])
        if cut[i] != 0 or j < 0 or vf[j] == 0:
            next_id += 1
            labels[i] = next_id
        else:
            labels[i] = labels[j]
        sizes_tmp[labels[i]] += 1

    return labels.reshape(valid.shape), sizes_tmp[1 : next_id + 1], limit, 0


def _merge_small_adjacent_domains(
    labels: np.ndarray,
    sizes: np.ndarray,
    max_domain_cells: int,
    *,
    min_fraction: float = 0.20,
    max_passes: int = 4,
):
    """Pack small adjacent hydrological fragments into larger processing groups.

    This does not define final slope-unit boundaries. It only reduces I/O by
    processing adjacent complete hydrological fragments together. The merged
    processing group never exceeds ``max_domain_cells``.
    """
    if sizes.size <= 1:
        return labels, sizes

    n = int(sizes.size)
    parent = np.arange(n + 1, dtype=np.int32)
    comp_size = np.zeros(n + 1, dtype=np.int64)
    comp_size[1:] = sizes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if comp_size[ra] < comp_size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        comp_size[ra] += comp_size[rb]
        return ra

    # Shared-boundary adjacency. Only 4-neighbour contact is used here.
    pair_chunks = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        if np.any(m):
            lo = np.minimum(a[m], b[m]).astype(np.int64)
            hi = np.maximum(a[m], b[m]).astype(np.int64)
            pair_chunks.append(lo * (n + 1) + hi)

    if not pair_chunks:
        return labels, sizes

    keys = np.concatenate(pair_chunks)
    uniq, counts = np.unique(keys, return_counts=True)
    adjacency: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, n + 1)}
    for key, contact in zip(uniq.tolist(), counts.tolist()):
        a = int(key // (n + 1))
        b = int(key % (n + 1))
        adjacency[a].append((b, int(contact)))
        adjacency[b].append((a, int(contact)))

    min_cells = max(1, int(max_domain_cells * float(min_fraction)))

    for _ in range(max(1, int(max_passes))):
        changed = False
        roots = sorted(
            {find(i) for i in range(1, n + 1)},
            key=lambda r: int(comp_size[r]),
        )
        for r0 in roots:
            r = find(r0)
            if comp_size[r] >= min_cells:
                continue

            candidates: dict[int, int] = {}
            members = [i for i in range(1, n + 1) if find(i) == r]
            for member in members:
                for nb, contact in adjacency.get(member, []):
                    rn = find(nb)
                    if rn == r:
                        continue
                    candidates[rn] = candidates.get(rn, 0) + contact

            best = None
            best_contact = -1
            for rn, contact in candidates.items():
                if comp_size[r] + comp_size[rn] <= max_domain_cells:
                    if contact > best_contact:
                        best = rn
                        best_contact = contact
            if best is not None:
                union(r, best)
                changed = True
        if not changed:
            break

    root_to_new: dict[int, int] = {}
    next_id = 0
    lookup = np.zeros(n + 1, dtype=np.int32)
    for i in range(1, n + 1):
        r = find(i)
        if r not in root_to_new:
            next_id += 1
            root_to_new[r] = next_id
        lookup[i] = root_to_new[r]

    out = lookup[labels]
    new_sizes = np.bincount(out.ravel(), minlength=next_id + 1)[1:].astype(np.int64)
    return out.astype(np.int32, copy=False), new_sizes


@njit(cache=True)
def _domain_extents(labels, n_domains):
    rows, cols = labels.shape
    rmin = np.full(n_domains + 1, rows, dtype=np.int32)
    rmax = np.full(n_domains + 1, -1, dtype=np.int32)
    cmin = np.full(n_domains + 1, cols, dtype=np.int32)
    cmax = np.full(n_domains + 1, -1, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            d = int(labels[r, c])
            if d <= 0:
                continue
            if r < rmin[d]: rmin[d] = r
            if r > rmax[d]: rmax[d] = r
            if c < cmin[d]: cmin[d] = c
            if c > cmax[d]: cmax[d] = c
    return rmin, rmax, cmin, cmax


def partition_coarse_hydrology(
    coarse_dem: str | Path,
    work_dir: str | Path,
    *,
    max_fine_cells: int,
    fine_resolution_m: float,
    workers: int,
    numba_threads: int,
    convergence: int = 5,
    merge_small_fraction: float = 0.20,
    verbose: bool = True,
):
    """Run coarse hydrology and derive memory-bounded processing domains."""
    coarse_dem = Path(coarse_dem)
    work_dir = Path(work_dir)
    hydro_work = work_dir / "coarse_hydrology"
    hydro_work.mkdir(parents=True, exist_ok=True)

    meta = read_meta(coarse_dem)
    coarse_resolution = math.sqrt(meta.cell_area)
    scale = (coarse_resolution / float(fine_resolution_m)) ** 2
    max_coarse_cells = max(32, int(max_fine_cells / max(1.0, scale)))

    store = MemmapStore(hydro_work / "memmap")
    model = ResearchSlopeUnits(
        threshold_m2=250_000.0,
        min_area_m2=100_000.0,
        cv_min=0.25,
        convergence=convergence,
        workers=workers,
        numba_threads=numba_threads,
        mfd_block_cells=1_000_000,
        reuse_hydrology=True,
        reuse_candidates=True,
        keep_work=True,
        verbose=verbose,
    )
    model._prepare_hydrology(coarse_dem, store, meta)

    valid = store.open("valid", "r")
    receiver = store.open("receiver", "r")
    order = store.open("order", "r")

    labels, sizes, cut_threshold, _ = _adaptive_tree_partition(
        receiver, valid, order, max_coarse_cells
    )
    before = int(sizes.size)
    labels, sizes = _merge_small_adjacent_domains(
        labels,
        sizes,
        max_coarse_cells,
        min_fraction=merge_small_fraction,
    )

    if verbose:
        print(
            f"[PySlopeUnits large-area] hydro domains={sizes.size:,} "
            f"(raw={before:,}) | max_coarse_cells={max_coarse_cells:,} | "
            f"cut_threshold={cut_threshold:,}"
        )
        if sizes.size:
            print(
                f"[PySlopeUnits large-area] largest coarse domain="
                f"{int(sizes.max()):,} cells | median={int(np.median(sizes)):,}"
            )

    return labels, sizes, meta, max_coarse_cells


def write_domain_products(
    labels: np.ndarray,
    sizes: np.ndarray,
    meta,
    *,
    fine_resolution_m: float,
    raster_path: str | Path,
    vector_path: str | Path | None = None,
    verbose: bool = True,
) -> list[HydrologicalDomain]:
    """Write only the coarse domain-ID raster and lightweight metadata.

    No polygonization is performed. This removes a major planning-stage I/O and
    CPU cost for very-large-area runs. ``vector_path`` is accepted only for
    API compatibility and is intentionally ignored.
    """
    raster_path = Path(raster_path)
    raster_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": meta.shape[0],
        "width": meta.shape[1],
        "count": 1,
        "dtype": "int32",
        "crs": meta.crs,
        "transform": meta.transform,
        "nodata": 0,
        "compress": "zstd",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(labels.astype(np.int32, copy=False), 1)

    n = int(sizes.size)
    rmin, rmax, cmin, cmax = _domain_extents(labels, n)
    scale = meta.cell_area / (fine_resolution_m * fine_resolution_m)
    records: list[HydrologicalDomain] = []

    for did in range(1, n + 1):
        if rmax[did] < 0:
            continue
        row0, row1 = int(rmin[did]), int(rmax[did]) + 1
        col0, col1 = int(cmin[did]), int(cmax[did]) + 1
        win = Window(col0, row0, col1 - col0, row1 - row0)
        left, bottom, right, top = rasterio.windows.bounds(win, meta.transform)
        coarse_n = int(sizes[did - 1])
        fine_n = int(math.ceil(coarse_n * scale))
        records.append(
            HydrologicalDomain(
                domain_id=did,
                coarse_cells=coarse_n,
                estimated_fine_cells=fine_n,
                area_m2=float(coarse_n * meta.cell_area),
                bounds=(float(left), float(bottom), float(right), float(top)),
                row_min=row0,
                row_max=row1,
                col_min=col0,
                col_max=col1,
            )
        )

    if verbose:
        print(
            f"[PySlopeUnits large-area] domain raster written | "
            f"domains={len(records):,} | no vectorization"
        )
    return records


def save_plan(plan: HydrologicalDomainPlan, path: str | Path) -> None:
    Path(path).write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
