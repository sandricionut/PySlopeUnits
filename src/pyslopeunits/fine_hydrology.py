from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time

import numpy as np
import rasterio
from numba import njit

from .kernels import DR, DC, NBR_EW, NBR_NS
from .memmap_store import MemmapStore
from .raster import RasterMeta
from .indexing import dtype_name


@dataclass(frozen=True)
class DomainShardedAStarResult:
    valid_cells: int
    domains: int
    fallback_cells: int
    max_domain_cells: int
    visited_cells: int
    total_seconds: float


@njit(cache=True, inline="always")
def _slope2(ele: float, up_ele: float, dist: float) -> float:
    if ele >= up_ele:
        return 0.0
    return (up_ele - ele) / dist


@njit(cache=True, inline="always")
def _domain_for_cell(
    idx: int,
    fine_cols: int,
    fine_c: float,
    fine_f: float,
    fine_xres: float,
    fine_yres: float,
    coarse_labels: np.ndarray,
    coarse_c: float,
    coarse_f: float,
    coarse_xres: float,
    coarse_yres: float,
    fallback_domain: int,
) -> int:
    r = idx // fine_cols
    c = idx - r * fine_cols

    x = fine_c + (c + 0.5) * fine_xres
    y = fine_f - (r + 0.5) * fine_yres

    cc = int(math.floor((x - coarse_c) / coarse_xres))
    rr = int(math.floor((coarse_f - y) / coarse_yres))

    if 0 <= rr < coarse_labels.shape[0] and 0 <= cc < coarse_labels.shape[1]:
        d = int(coarse_labels[rr, cc])
        if d > 0:
            return d
    return fallback_domain


@njit(cache=True)
def _count_domains(
    valid: np.ndarray,
    fine_c: float,
    fine_f: float,
    fine_xres: float,
    fine_yres: float,
    coarse_labels: np.ndarray,
    coarse_c: float,
    coarse_f: float,
    coarse_xres: float,
    coarse_yres: float,
    fallback_domain: int,
    counts: np.ndarray,
) -> int:
    vf = valid.reshape(valid.size)
    cols = valid.shape[1]
    fallback = 0
    for i in range(vf.size):
        if vf[i] == 0:
            continue
        d = _domain_for_cell(
            i, cols,
            fine_c, fine_f, fine_xres, fine_yres,
            coarse_labels,
            coarse_c, coarse_f, coarse_xres, coarse_yres,
            fallback_domain,
        )
        counts[d] += 1
        if d == fallback_domain:
            fallback += 1
    return fallback


@njit(cache=True, inline="always")
def _heap_less(idx_a: int, age_a: int, idx_b: int, age_b: int, zf: np.ndarray) -> bool:
    za = zf[idx_a]
    zb = zf[idx_b]
    if za < zb:
        return True
    if za > zb:
        return False
    return age_a < age_b


@njit(cache=True, inline="always")
def _local_heap_push(
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    sizes: np.ndarray,
    domain: int,
    idx: int,
    age: int,
    zf: np.ndarray,
) -> None:
    base = int(offsets[domain])
    size = int(sizes[domain])
    pos = size
    abs_pos = base + pos
    heap_idx[abs_pos] = idx
    heap_age[abs_pos] = age
    sizes[domain] = size + 1

    while pos > 0:
        parent = (pos - 1) // 2
        a = base + pos
        b = base + parent
        if not _heap_less(heap_idx[a], heap_age[a], heap_idx[b], heap_age[b], zf):
            break
        ti = heap_idx[b]
        ta = heap_age[b]
        heap_idx[b] = heap_idx[a]
        heap_age[b] = heap_age[a]
        heap_idx[a] = ti
        heap_age[a] = ta
        pos = parent


@njit(cache=True, inline="always")
def _local_heap_pop(
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    sizes: np.ndarray,
    domain: int,
    zf: np.ndarray,
):
    base = int(offsets[domain])
    size = int(sizes[domain])
    idx = int(heap_idx[base])
    age = int(heap_age[base])
    size -= 1
    sizes[domain] = size

    if size > 0:
        heap_idx[base] = heap_idx[base + size]
        heap_age[base] = heap_age[base + size]
        pos = 0
        while True:
            left = 2 * pos + 1
            if left >= size:
                break
            right = left + 1
            best = left
            if right < size:
                a = base + right
                b = base + left
                if _heap_less(heap_idx[a], heap_age[a], heap_idx[b], heap_age[b], zf):
                    best = right
            a = base + best
            b = base + pos
            if not _heap_less(heap_idx[a], heap_age[a], heap_idx[b], heap_age[b], zf):
                break
            ti = heap_idx[b]
            ta = heap_age[b]
            heap_idx[b] = heap_idx[a]
            heap_age[b] = heap_age[a]
            heap_idx[a] = ti
            heap_age[a] = ta
            pos = best

    return idx, age


@njit(cache=True, inline="always")
def _domain_less(
    da: int,
    db: int,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    zf: np.ndarray,
) -> bool:
    pa = int(offsets[da])
    pb = int(offsets[db])
    return _heap_less(
        int(heap_idx[pa]), int(heap_age[pa]),
        int(heap_idx[pb]), int(heap_age[pb]),
        zf,
    )


@njit(cache=True, inline="always")
def _global_swap(gheap: np.ndarray, gpos: np.ndarray, a: int, b: int) -> None:
    da = int(gheap[a])
    db = int(gheap[b])
    gheap[a] = db
    gheap[b] = da
    gpos[da] = b
    gpos[db] = a


@njit(cache=True, inline="always")
def _global_insert(
    gheap: np.ndarray,
    gpos: np.ndarray,
    gsize: int,
    domain: int,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    zf: np.ndarray,
) -> int:
    pos = gsize
    gheap[pos] = domain
    gpos[domain] = pos
    gsize += 1
    while pos > 0:
        parent = (pos - 1) // 2
        if not _domain_less(int(gheap[pos]), int(gheap[parent]), heap_idx, heap_age, offsets, zf):
            break
        _global_swap(gheap, gpos, pos, parent)
        pos = parent
    return gsize


@njit(cache=True, inline="always")
def _global_fix(
    gheap: np.ndarray,
    gpos: np.ndarray,
    gsize: int,
    domain: int,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    zf: np.ndarray,
) -> None:
    pos = int(gpos[domain])
    if pos < 0:
        return

    while pos > 0:
        parent = (pos - 1) // 2
        if not _domain_less(int(gheap[pos]), int(gheap[parent]), heap_idx, heap_age, offsets, zf):
            break
        _global_swap(gheap, gpos, pos, parent)
        pos = parent

    while True:
        left = 2 * pos + 1
        if left >= gsize:
            break
        right = left + 1
        best = left
        if right < gsize and _domain_less(
            int(gheap[right]), int(gheap[left]), heap_idx, heap_age, offsets, zf
        ):
            best = right
        if not _domain_less(
            int(gheap[best]), int(gheap[pos]), heap_idx, heap_age, offsets, zf
        ):
            break
        _global_swap(gheap, gpos, best, pos)
        pos = best


@njit(cache=True, inline="always")
def _global_remove(
    gheap: np.ndarray,
    gpos: np.ndarray,
    gsize: int,
    domain: int,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    zf: np.ndarray,
) -> int:
    pos = int(gpos[domain])
    if pos < 0:
        return gsize
    gsize -= 1
    gpos[domain] = -1
    if pos == gsize:
        return gsize

    moved = int(gheap[gsize])
    gheap[pos] = moved
    gpos[moved] = pos
    _global_fix(gheap, gpos, gsize, moved, heap_idx, heap_age, offsets, zf)
    return gsize


@njit(cache=True)
def _astar_domain_sharded_impl(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    xres_scaled: float,
    yres_scaled: float,
    receiver: np.ndarray,
    order: np.ndarray,
    edgeflag: np.ndarray,
    state: np.ndarray,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
    offsets: np.ndarray,
    sizes: np.ndarray,
    coarse_labels: np.ndarray,
    fine_c: float,
    fine_f: float,
    fine_xres: float,
    fine_yres: float,
    coarse_c: float,
    coarse_f: float,
    coarse_xres: float,
    coarse_yres: float,
    fallback_domain: int,
    gheap: np.ndarray,
    gpos: np.ndarray,
) -> int:
    rows, cols = dem_scaled.shape
    zf = dem_scaled.reshape(dem_scaled.size)
    vf = valid.reshape(valid.size)
    rec = receiver.reshape(receiver.size)
    ef = edgeflag.reshape(edgeflag.size)
    st = state.reshape(state.size)

    for i in range(st.size):
        st[i] = 0

    for d in range(sizes.size):
        sizes[d] = 0
        gpos[d] = -1

    age = 0

    # Exact V08/V09 seed order: row-major external edge + valid cells adjacent to nodata.
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if vf[i] == 0:
                continue
            seed = r == 0 or c == 0 or r == rows - 1 or c == cols - 1
            if not seed:
                for ct in range(8):
                    rr = r + DR[ct]
                    cc = c + DC[ct]
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        continue
                    j = rr * cols + cc
                    if vf[j] == 0:
                        seed = True
                        break
            if seed:
                st[i] = 1
                ef[i] = 1
                d = _domain_for_cell(
                    i, cols,
                    fine_c, fine_f, fine_xres, fine_yres,
                    coarse_labels,
                    coarse_c, coarse_f, coarse_xres, coarse_yres,
                    fallback_domain,
                )
                _local_heap_push(heap_idx, heap_age, offsets, sizes, d, i, age, zf)
                age += 1

    gsize = 0
    for d in range(1, sizes.size):
        if sizes[d] > 0:
            gsize = _global_insert(
                gheap, gpos, gsize, d,
                heap_idx, heap_age, offsets, zf,
            )

    diag = math.sqrt(xres_scaled * xres_scaled + yres_scaled * yres_scaled)
    dist = np.empty(8, dtype=np.float64)
    dist[0] = yres_scaled
    dist[1] = yres_scaled
    dist[2] = xres_scaled
    dist[3] = xres_scaled
    for ct in range(4, 8):
        dist[ct] = diag

    slopes = np.empty(8, dtype=np.float64)
    nbr_e = np.empty(8, dtype=np.float64)
    nbr_ids = np.empty(8, dtype=np.int64)

    k = 0
    while gsize > 0:
        d = int(gheap[0])
        i, _ = _local_heap_pop(heap_idx, heap_age, offsets, sizes, d, zf)

        if sizes[d] == 0:
            gsize = _global_remove(
                gheap, gpos, gsize, d,
                heap_idx, heap_age, offsets, zf,
            )
        else:
            _global_fix(
                gheap, gpos, gsize, d,
                heap_idx, heap_age, offsets, zf,
            )

        order[k] = i
        k += 1
        r = i // cols
        c = i - r * cols
        ele = float(zf[i])

        for ct in range(8):
            slopes[ct] = -1.0
            nbr_e[ct] = 0.0
            nbr_ids[ct] = -1
            rr = r + DR[ct]
            cc = c + DC[ct]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            j = rr * cols + cc
            if vf[j] == 0:
                continue
            nbr_ids[ct] = j
            if st[j] != 2:
                nbr_e[ct] = float(zf[j])
                slopes[ct] = _slope2(ele, nbr_e[ct], dist[ct])

        for ct in range(8):
            j = int(nbr_ids[ct])
            if j < 0:
                continue

            eligible = st[j] == 0 or (st[j] != 2 and ef[j] != 0)
            skip_diag = False
            if eligible and ct > 3 and slopes[ct] > 0.0:
                ew = NBR_EW[ct]
                ns = NBR_NS[ct]
                if slopes[ew] >= 0.0:
                    if slopes[ct] < _slope2(nbr_e[ew], nbr_e[ct], xres_scaled):
                        skip_diag = True
                if not skip_diag and slopes[ns] >= 0.0:
                    if slopes[ct] < _slope2(nbr_e[ns], nbr_e[ct], yres_scaled):
                        skip_diag = True
            if skip_diag:
                continue

            if st[j] == 0:
                rec[j] = i
                st[j] = 1
                jd = _domain_for_cell(
                    j, cols,
                    fine_c, fine_f, fine_xres, fine_yres,
                    coarse_labels,
                    coarse_c, coarse_f, coarse_xres, coarse_yres,
                    fallback_domain,
                )
                was_empty = sizes[jd] == 0
                old_top_idx = -1
                old_top_age = -1
                if not was_empty:
                    p = int(offsets[jd])
                    old_top_idx = int(heap_idx[p])
                    old_top_age = int(heap_age[p])

                _local_heap_push(
                    heap_idx, heap_age, offsets, sizes, jd, j, age, zf
                )
                age += 1

                if was_empty:
                    gsize = _global_insert(
                        gheap, gpos, gsize, jd,
                        heap_idx, heap_age, offsets, zf,
                    )
                else:
                    p = int(offsets[jd])
                    if int(heap_idx[p]) != old_top_idx or int(heap_age[p]) != old_top_age:
                        _global_fix(
                            gheap, gpos, gsize, jd,
                            heap_idx, heap_age, offsets, zf,
                        )

            elif st[j] != 2 and ef[j] != 0 and slopes[ct] > 0.0:
                rec[j] = i

        st[i] = 2

    return k


def _north_up_grid(ds, label: str) -> None:
    t = ds.transform
    tol = 1e-12
    if abs(float(t.b)) > tol or abs(float(t.d)) > tol:
        raise ValueError(f"{label} must be a north-up raster without rotation")
    if float(t.a) <= 0 or float(t.e) >= 0:
        raise ValueError(f"{label} must have positive x resolution and negative y resolution")


def _load_domain_map(
    domain_raster: str | Path,
    store: MemmapStore,
    fine_meta: RasterMeta,
):
    domain_raster = Path(domain_raster)
    if not domain_raster.exists():
        raise FileNotFoundError(domain_raster)

    with rasterio.open(domain_raster) as ds:
        if ds.crs != fine_meta.crs:
            raise ValueError(
                "Hydrological domain raster and fine DEM must use the same CRS"
            )
        _north_up_grid(ds, "Hydrological domain raster")
        coarse = store.create_temp(
            "fine_hydro_coarse_domains_tmp",
            (int(ds.height), int(ds.width)),
            np.int32,
            fill=0,
        )
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win, masked=True, out_dtype="int32")
            data = np.asarray(arr.filled(0), dtype=np.int32)
            r0 = int(win.row_off)
            c0 = int(win.col_off)
            coarse[r0:r0 + int(win.height), c0:c0 + int(win.width)] = data

        max_domain = int(np.max(coarse)) if coarse.size else 0
        if max_domain <= 0:
            raise ValueError("Hydrological domain raster contains no positive domain IDs")

        return coarse, ds.transform, max_domain


def run_domain_sharded_astar(
    store: MemmapStore,
    *,
    meta: RasterMeta,
    domain_raster: str | Path,
    hydro_dem: np.ndarray,
    valid: np.ndarray,
    receiver: np.ndarray,
    order: np.ndarray,
    edgeflag: np.ndarray,
    nvalid: int,
    xres_scaled: float,
    yres_scaled: float,
    index_dtype,
    verbose: bool = True,
) -> DomainShardedAStarResult:
    """Exact global A* with the priority queue sharded by hydrological domains.

    Domains are only queue/storage shards. A global heap of domain minima always
    selects the same next cell as the original single global priority queue, so
    routing across domain boundaries is not approximated and domains never act
    as independent hydrological analyses.
    """
    t0 = time.perf_counter()

    fine_t = meta.transform
    if abs(float(fine_t.b)) > 1e-12 or abs(float(fine_t.d)) > 1e-12:
        raise ValueError("Fine DEM must be north-up for domain-sharded hydrology")
    if float(fine_t.a) <= 0 or float(fine_t.e) >= 0:
        raise ValueError("Fine DEM must have positive x and negative y resolution")

    coarse_labels, coarse_t, max_domain = _load_domain_map(
        domain_raster, store, meta
    )
    fallback_domain = max_domain + 1
    domain_slots = fallback_domain + 1

    counts = np.zeros(domain_slots, dtype=np.int64)
    fallback_cells = int(
        _count_domains(
            valid,
            float(fine_t.c), float(fine_t.f),
            abs(float(fine_t.a)), abs(float(fine_t.e)),
            coarse_labels,
            float(coarse_t.c), float(coarse_t.f),
            abs(float(coarse_t.a)), abs(float(coarse_t.e)),
            int(fallback_domain),
            counts,
        )
    )

    counted = int(counts.sum())
    if counted != int(nvalid):
        raise RuntimeError(
            f"Domain assignment counted {counted:,} valid cells, expected {int(nvalid):,}"
        )

    offsets = np.zeros(domain_slots + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    sizes = np.zeros(domain_slots, dtype=np.int64)

    state = store.create_temp(
        "astar_domain_state_tmp", meta.shape, np.uint8, fill=0
    )
    heap_idx = store.create_temp(
        "astar_domain_heap_idx_tmp", (int(nvalid),), index_dtype
    )
    heap_age = store.create_temp(
        "astar_domain_heap_age_tmp", (int(nvalid),), index_dtype
    )

    gheap = np.zeros(domain_slots, dtype=np.int64)
    gpos = np.full(domain_slots, -1, dtype=np.int64)

    nonempty = int(np.count_nonzero(counts[1:]))
    max_domain_cells = int(counts.max()) if counts.size else 0

    if verbose:
        print(
            "[PySlopeUnits] fine hydrology | exact domain-sharded global A* | "
            f"domains={nonempty:,} | fallback-cells={fallback_cells:,} | "
            f"largest-domain={max_domain_cells:,}"
        )
        print(
            "[PySlopeUnits] fine hydrology queue | "
            f"index={dtype_name(index_dtype)} | total-capacity={int(nvalid):,} cells | "
            "global arbitration preserves V08/V09 A* order"
        )

    visited = int(
        _astar_domain_sharded_impl(
            hydro_dem,
            valid,
            float(xres_scaled),
            float(yres_scaled),
            receiver,
            order,
            edgeflag,
            state,
            heap_idx,
            heap_age,
            offsets,
            sizes,
            coarse_labels,
            float(fine_t.c), float(fine_t.f),
            abs(float(fine_t.a)), abs(float(fine_t.e)),
            float(coarse_t.c), float(coarse_t.f),
            abs(float(coarse_t.a)), abs(float(coarse_t.e)),
            int(fallback_domain),
            gheap,
            gpos,
        )
    )

    receiver.flush()
    order.flush()
    edgeflag.flush()

    result = DomainShardedAStarResult(
        valid_cells=int(nvalid),
        domains=nonempty,
        fallback_cells=fallback_cells,
        max_domain_cells=max_domain_cells,
        visited_cells=visited,
        total_seconds=time.perf_counter() - t0,
    )
    store.write_json("fine_hydrology_report.json", asdict(result))

    store.close_many(state, heap_idx, heap_age, coarse_labels)
    del state, heap_idx, heap_age, coarse_labels
    store.cleanup("astar_domain_state_tmp")
    store.cleanup("astar_domain_heap_idx_tmp")
    store.cleanup("astar_domain_heap_age_tmp")
    store.cleanup("fine_hydro_coarse_domains_tmp")

    if verbose:
        print(
            "[PySlopeUnits] fine hydrology A* complete | "
            f"visited={visited:,} | {result.total_seconds / 60:.2f} min"
        )

    return result
