from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

# GRASS-like neighbour order used by the A* / MFD translation:
# S, N, W, E, NE, SW, SE, NW
DR = np.array([1, -1, 0, 0, -1, 1, 1, -1], dtype=np.int32)
DC = np.array([0, 0, -1, 1, 1, -1, 1, -1], dtype=np.int32)
NBR_EW = np.array([0, 1, 2, 3, 1, 0, 0, 1], dtype=np.int32)
NBR_NS = np.array([0, 1, 2, 3, 3, 2, 3, 2], dtype=np.int32)

# Scan order used by half-basin donor collection: NW,N,NE,W,E,SW,S,SE
HBR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
HBC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)


@njit(cache=True, parallel=True)
def quantize_dem(dem: np.ndarray, valid: np.ndarray, out: np.ndarray, scale: int = 1000):
    """Quantize floating DEM to the 0.001-unit convention used by r.watershed."""
    n = dem.size
    df = dem.ravel()
    vf = valid.ravel()
    of = out.ravel()
    for i in prange(n):
        if vf[i] != 0 and math.isfinite(df[i]):
            of[i] = int(math.floor(df[i] * scale + 0.5))
        else:
            of[i] = 0


@njit(cache=True, parallel=True)
def aspect_sincos(dem: np.ndarray, valid: np.ndarray, xres: float, yres: float,
                  sin_out: np.ndarray, cos_out: np.ndarray, good_out: np.ndarray):
    """Memory-efficient central-gradient aspect components.

    This preserves the Stage-6 PySlope aspect convention while avoiding a
    full aspect raster in RAM. Cells without a complete N/S/E/W neighbourhood
    are marked invalid for circular statistics.
    """
    rows, cols = dem.shape
    for r in prange(rows):
        for c in range(cols):
            if valid[r, c] == 0:
                sin_out[r, c] = 0.0
                cos_out[r, c] = 0.0
                good_out[r, c] = 0
                continue
            if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                sin_out[r, c] = 0.0
                cos_out[r, c] = 0.0
                good_out[r, c] = 0
                continue
            if (valid[r-1, c] == 0 or valid[r+1, c] == 0 or
                    valid[r, c-1] == 0 or valid[r, c+1] == 0):
                sin_out[r, c] = 0.0
                cos_out[r, c] = 0.0
                good_out[r, c] = 0
                continue

            dz_drow = (dem[r+1, c] - dem[r-1, c]) / (2.0 * yres)
            dz_dx = (dem[r, c+1] - dem[r, c-1]) / (2.0 * xres)
            dz_dnorth = -dz_drow
            east = -dz_dx
            north = -dz_dnorth
            norm = math.sqrt(east * east + north * north)
            if norm <= 1e-15:
                sin_out[r, c] = 0.0
                cos_out[r, c] = 0.0
                good_out[r, c] = 0
            else:
                # sin(aspect)=east/norm, cos(aspect)=north/norm
                sin_out[r, c] = east / norm
                cos_out[r, c] = north / norm
                good_out[r, c] = 1


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
def _heap_push(heap_idx, heap_age, size: int, idx: int, age: int, zf) -> int:
    pos = size
    heap_idx[pos] = idx
    heap_age[pos] = age
    size += 1
    while pos > 0:
        parent = (pos - 1) // 2
        if not _heap_less(heap_idx[pos], heap_age[pos], heap_idx[parent], heap_age[parent], zf):
            break
        ti = heap_idx[parent]
        ta = heap_age[parent]
        heap_idx[parent] = heap_idx[pos]
        heap_age[parent] = heap_age[pos]
        heap_idx[pos] = ti
        heap_age[pos] = ta
        pos = parent
    return size


@njit(cache=True, inline="always")
def _heap_pop(heap_idx, heap_age, size: int, zf):
    idx = heap_idx[0]
    age = heap_age[0]
    size -= 1
    if size > 0:
        heap_idx[0] = heap_idx[size]
        heap_age[0] = heap_age[size]
        pos = 0
        while True:
            left = 2 * pos + 1
            if left >= size:
                break
            right = left + 1
            best = left
            if right < size and _heap_less(
                    heap_idx[right], heap_age[right], heap_idx[left], heap_age[left], zf):
                best = right
            if not _heap_less(heap_idx[best], heap_age[best], heap_idx[pos], heap_age[pos], zf):
                break
            ti = heap_idx[pos]
            ta = heap_age[pos]
            heap_idx[pos] = heap_idx[best]
            heap_age[pos] = heap_age[best]
            heap_idx[best] = ti
            heap_age[best] = ta
            pos = best
    return idx, age, size


@njit(cache=True, inline="always")
def _slope2(ele: float, up_ele: float, dist: float) -> float:
    if ele >= up_ele:
        return 0.0
    return (up_ele - ele) / dist


@njit(cache=True)
def _astar_route_impl(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    xres_scaled: float,
    yres_scaled: float,
    receiver: np.ndarray,
    order: np.ndarray,
    edgeflag: np.ndarray,
    inlist: np.ndarray,
    worked: np.ndarray,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
) -> int:
    """Shared A* implementation with caller-owned scratch arrays."""
    rows, cols = dem_scaled.shape
    n = dem_scaled.size
    zf = dem_scaled.ravel()
    vf = valid.ravel()
    rec = receiver.ravel()
    ef = edgeflag.ravel()
    il = inlist.ravel()
    wk = worked.ravel()

    # Scratch arrays may be reused after an interrupted run.
    for i in range(n):
        il[i] = 0
        wk[i] = 0

    nvalid = 0
    for i in range(n):
        if vf[i] != 0:
            nvalid += 1

    if heap_idx.size < nvalid or heap_age.size < nvalid:
        return -1

    heap_size = 0
    age = 0

    # Seeds: row-major external edge and valid cells adjacent to nodata.
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
                il[i] = 1
                ef[i] = 1
                heap_size = _heap_push(heap_idx, heap_age, heap_size, i, age, zf)
                age += 1

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
    while heap_size > 0:
        i, _, heap_size = _heap_pop(heap_idx, heap_age, heap_size, zf)
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
            if wk[j] == 0:
                nbr_e[ct] = float(zf[j])
                slopes[ct] = _slope2(ele, nbr_e[ct], dist[ct])

        for ct in range(8):
            j = nbr_ids[ct]
            if j < 0:
                continue
            eligible = il[j] == 0 or (wk[j] == 0 and ef[j] != 0)
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

            if il[j] == 0:
                rec[j] = i
                il[j] = 1
                heap_size = _heap_push(heap_idx, heap_age, heap_size, j, age, zf)
                age += 1
            elif wk[j] == 0 and ef[j] != 0 and slopes[ct] > 0.0:
                rec[j] = i

        wk[i] = 1

    return k


@njit(cache=True)
def astar_route(dem_scaled: np.ndarray, valid: np.ndarray,
                xres_scaled: float, yres_scaled: float,
                receiver: np.ndarray, order: np.ndarray, edgeflag: np.ndarray) -> int:
    """Backward-compatible in-RAM scratch A* routing."""
    n = dem_scaled.size
    nvalid = order.size
    inlist = np.zeros(n, dtype=np.uint8)
    worked = np.zeros(n, dtype=np.uint8)
    heap_idx = np.empty(nvalid, dtype=np.int64)
    heap_age = np.empty(nvalid, dtype=np.int64)
    return _astar_route_impl(
        dem_scaled, valid, xres_scaled, yres_scaled,
        receiver, order, edgeflag,
        inlist, worked, heap_idx, heap_age,
    )


@njit(cache=True)
def astar_route_preallocated(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    xres_scaled: float,
    yres_scaled: float,
    receiver: np.ndarray,
    order: np.ndarray,
    edgeflag: np.ndarray,
    inlist: np.ndarray,
    worked: np.ndarray,
    heap_idx: np.ndarray,
    heap_age: np.ndarray,
) -> int:
    """A* routing with disk-backed/memmap scratch supplied by the caller.

    Numerical and tie-breaking semantics are identical to :func:`astar_route`;
    only ownership of the large scratch arrays changes.
    """
    return _astar_route_impl(
        dem_scaled, valid, xres_scaled, yres_scaled,
        receiver, order, edgeflag,
        inlist, worked, heap_idx, heap_age,
    )


@njit(cache=True)
def mfd_accumulation(dem_scaled: np.ndarray, valid: np.ndarray,
                     astar_receiver: np.ndarray, order: np.ndarray,
                     xres_scaled: float, yres_scaled: float, convergence: int,
                     accumulation: np.ndarray, adjusted_receiver: np.ndarray):
    """Holmgren MFD accumulation followed by a single adjusted receiver."""
    rows, cols = dem_scaled.shape
    n = dem_scaled.size
    zf = dem_scaled.ravel()
    vf = valid.ravel()
    rec0 = astar_receiver.ravel()
    acc = accumulation.ravel()
    adj = adjusted_receiver.ravel()
    nv = order.size

    rank = np.full(n, -1, dtype=np.int64)
    for pos in range(nv):
        rank[order[pos]] = pos
    for i in range(n):
        if vf[i] != 0:
            acc[i] = 1.0
            adj[i] = rec0[i]
        else:
            acc[i] = 0.0
            adj[i] = -1

    diag = math.sqrt(xres_scaled * xres_scaled + yres_scaled * yres_scaled)
    dist = np.empty(8, dtype=np.float64)
    dist[0] = yres_scaled
    dist[1] = yres_scaled
    dist[2] = xres_scaled
    dist[3] = xres_scaled
    for ct in range(4, 8):
        dist[ct] = diag

    ids = np.empty(9, dtype=np.int64)
    weights = np.empty(9, dtype=np.float64)

    for pos in range(nv - 1, -1, -1):
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

        zi = float(zf[i])
        count = 0
        maxw = 0.0
        astar_j = int(rec0[i])
        astar_present = False

        for ct in range(8):
            rr = r + DR[ct]
            cc = c + DC[ct]
            j = rr * cols + cc
            if rank[j] >= rank[i]:
                continue
            dz = zi - float(zf[j])
            if dz > 0.0:
                w = (dz / dist[ct]) ** convergence
            elif dz == 0.0:
                w = (0.5 / dist[ct]) ** convergence
            else:
                continue
            ids[count] = j
            weights[count] = w
            count += 1
            if w > maxw:
                maxw = w
            if j == astar_j:
                astar_present = True

        if astar_j >= 0 and not astar_present:
            if maxw <= 0.0:
                maxw = 1.0
            ids[count] = astar_j
            weights[count] = maxw
            count += 1

        if count == 0:
            if astar_j >= 0:
                acc[astar_j] += acc[i]
            continue

        sw = 0.0
        for q in range(count):
            sw += weights[q]
        if sw <= 0.0:
            continue
        value = acc[i]
        for q in range(count):
            acc[ids[q]] += value * (weights[q] / sw)

    # Select single adjusted drainage receiver by highest downstream accumulation.
    for pos in range(nv - 1, -1, -1):
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
            if rank[j] >= rank[i]:
                continue
            if acc[j] > best_acc:
                best = j
                best_acc = acc[j]
        adj[i] = best



# ============================================================================
# Stage 11 multiprocessing-pipelined exact MFD
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
def build_rank_serial(order: np.ndarray, rank_out: np.ndarray):
    rf = rank_out.reshape(rank_out.size)
    for pos in range(order.size):
        rf[int(order[pos])] = pos


@njit(cache=True)
def init_mfd_valid_serial(
    order: np.ndarray,
    astar_receiver: np.ndarray,
    accumulation: np.ndarray,
    adjusted_receiver: np.ndarray,
):
    rf = astar_receiver.reshape(astar_receiver.size)
    af = accumulation.reshape(accumulation.size)
    df = adjusted_receiver.reshape(adjusted_receiver.size)
    for pos in range(order.size):
        i = int(order[pos])
        af[i] = 1.0
        df[i] = rf[i]


@njit(cache=True)
def mfd_weights_slice_serial(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    rank: np.ndarray,
    reverse_offset: int,
    k0: int,
    k1: int,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
    weights_out: np.ndarray,
):
    """Compute normalized MFD weights for rows k0:k1 of one order block."""
    rows, cols = dem_scaled.shape
    zf = dem_scaled.reshape(dem_scaled.size)
    vf = valid.reshape(valid.size)
    rec0 = astar_receiver.reshape(astar_receiver.size)
    rk = rank.reshape(rank.size)
    nv = order.size
    diag = math.sqrt(xres_scaled * xres_scaled + yres_scaled * yres_scaled)

    for k in range(k0, k1):
        for q in range(9):
            weights_out[k, q] = 0.0

        pos = nv - 1 - (reverse_offset + k)
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

        zi = float(zf[i])
        astar_j = int(rec0[i])
        astar_present = False
        maxw = 0.0
        sw = 0.0

        for ct in range(8):
            rr = r + DR[ct]
            cc = c + DC[ct]
            j = rr * cols + cc

            if rk[j] >= rk[i]:
                continue

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


@njit(cache=True)
def mfd_scatter_block_serial(
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    reverse_offset: int,
    block_len: int,
    accumulation: np.ndarray,
    weights: np.ndarray,
):
    rows, cols = valid.shape
    vf = valid.reshape(valid.size)
    rec0 = astar_receiver.reshape(astar_receiver.size)
    acc = accumulation.reshape(accumulation.size)
    nv = order.size

    for k in range(block_len):
        pos = nv - 1 - (reverse_offset + k)
        i = int(order[pos])
        value = acc[i]
        r = i // cols
        c = i - r * cols

        for ct in range(8):
            w = weights[k, ct]
            if w == 0.0:
                continue
            rr = r + DR[ct]
            cc = c + DC[ct]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            j = rr * cols + cc
            if vf[j] != 0:
                acc[j] += value * w

        w = weights[k, 8]
        if w != 0.0:
            j = int(rec0[i])
            if j >= 0:
                acc[j] += value * w


@njit(cache=True)
def adjusted_receiver_slice_serial(
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    rank: np.ndarray,
    accumulation: np.ndarray,
    adjusted_receiver: np.ndarray,
    pos0: int,
    pos1: int,
):
    rows, cols = valid.shape
    vf = valid.reshape(valid.size)
    rec0 = astar_receiver.reshape(astar_receiver.size)
    rk = rank.reshape(rank.size)
    acc = accumulation.reshape(accumulation.size)
    adj = adjusted_receiver.reshape(adjusted_receiver.size)

    for pos in range(pos0, pos1):
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
            if rk[j] >= rk[i]:
                continue
            if acc[j] > best_acc:
                best = j
                best_acc = acc[j]

        adj[i] = best

@njit(cache=True)
def stream_mask_grasslike(dem_scaled: np.ndarray, valid: np.ndarray,
                          accumulation: np.ndarray, receiver: np.ndarray,
                          order: np.ndarray, threshold_cells: float,
                          stream_out: np.ndarray):
    """Build GRASS-like SWALEFLAG stream network for a threshold."""
    rows, cols = dem_scaled.shape
    n = dem_scaled.size
    zf = dem_scaled.ravel()
    vf = valid.ravel()
    af = accumulation.ravel()
    rec = receiver.ravel()
    sw = stream_out.ravel()
    worked = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        sw[i] = 0

    for pos in range(order.size - 1, -1, -1):
        i = int(order[pos])
        if vf[i] == 0:
            continue
        worked[i] = 1
        r = i // cols
        c = i - r * cols
        edge = False
        stream_cells = 0
        swale_cells = 0
        flat = True

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    edge = True
                    break
                j = rr * cols + cc
                if vf[j] == 0:
                    edge = True
                    break
                if sw[j] != 0:
                    swale_cells += 1
                if (af[j] + 0.5) >= threshold_cells and zf[j] > zf[i]:
                    stream_cells += 1
                if worked[j] == 0 and zf[j] != zf[i]:
                    flat = False
            if edge:
                break

        if edge:
            continue
        if (sw[i] == 0 and af[i] >= threshold_cells and
                stream_cells < 1 and swale_cells < 1 and not flat):
            sw[i] = 1
        if sw[i] != 0:
            j = int(rec[i])
            if j >= 0 and vf[j] != 0:
                sw[j] = 1


@njit(cache=True, inline="always")
def _dir_between(i: int, j: int, cols: int) -> int:
    if j < 0:
        return 0
    ri = i // cols
    ci = i - ri * cols
    rj = j // cols
    cj = j - rj * cols
    dr = rj - ri
    dc = cj - ci
    if dr == -1 and dc == 1:
        return 1
    if dr == -1 and dc == 0:
        return 2
    if dr == -1 and dc == -1:
        return 3
    if dr == 0 and dc == -1:
        return 4
    if dr == 1 and dc == -1:
        return 5
    if dr == 1 and dc == 0:
        return 6
    if dr == 1 and dc == 1:
        return 7
    if dr == 0 and dc == 1:
        return 8
    return 0


@njit(cache=True, inline="always")
def _opposite_dir(code: int) -> int:
    if code <= 0:
        return 0
    return ((code - 1 + 4) % 8) + 1


@njit(cache=True, inline="always")
def _haf_side(updir: int, downdir: int, thisdir: int) -> int:
    if updir <= 0 or downdir <= 0 or thisdir <= 0:
        return 0
    newup = updir - downdir
    if newup < 0:
        newup += 8
    newthis = thisdir - downdir
    if newthis < 0:
        newthis += 8
    if newthis < newup:
        return 2
    if newthis > newup:
        return 1
    return 0


@njit(cache=True, inline="always")
def _collect_donors(current: int, rows: int, cols: int,
                    rec: np.ndarray, valid: np.ndarray, donors: np.ndarray) -> int:
    r = current // cols
    c = current - r * cols
    count = 0
    for k in range(8):
        rr = r + HBR[k]
        cc = c + HBC[k]
        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
            continue
        j = rr * cols + cc
        if valid[j] != 0 and rec[j] == current:
            donors[count] = j
            count += 1
    return count


@njit(cache=True, inline="always")
def _downstream_dir(current: int, fallback_updir: int, cols: int, rec: np.ndarray) -> int:
    d = _dir_between(current, int(rec[current]), cols)
    if d == 0 and fallback_updir > 0:
        return _opposite_dir(fallback_updir)
    return d


@njit(cache=True)
def mark_branch_roots(receiver: np.ndarray, stream: np.ndarray, valid: np.ndarray,
                      mark: np.ndarray):
    """Mark stream outlets and stream-donor roots above junctions."""
    rows, cols = receiver.shape
    n = receiver.size
    rec = receiver.ravel()
    sf = stream.ravel()
    vf = valid.ravel()
    mf = mark.ravel()
    for i in range(n):
        mf[i] = 0
    donors = np.empty(8, dtype=np.int64)
    for i in range(n):
        if vf[i] == 0 or sf[i] == 0:
            continue
        j = int(rec[i])
        if j < 0 or vf[j] == 0 or sf[j] == 0:
            mf[i] = 1

        nd = _collect_donors(i, rows, cols, rec, vf, donors)
        nsd = 0
        for k in range(nd):
            d = donors[k]
            if sf[d] != 0:
                nsd += 1
        if nsd >= 2:
            for k in range(nd):
                d = donors[k]
                if sf[d] != 0:
                    mf[d] = 1


@njit(cache=True, inline="always")
def _root_pair(root: int, roots: np.ndarray) -> int:
    # Manual searchsorted, roots sorted ascending.
    lo = 0
    hi = roots.size
    while lo < hi:
        mid = (lo + hi) // 2
        if roots[mid] < root:
            lo = mid + 1
        else:
            hi = mid
    if lo >= roots.size or roots[lo] != root:
        return 0
    return 2 * (lo + 1)


@njit(cache=True)
def _label_overland(root: int, hlabel: int, rows: int, cols: int,
                    rec: np.ndarray, valid: np.ndarray, haf: np.ndarray):
    stack = [root]
    while len(stack) > 0:
        i = stack.pop()
        if haf[i] != 0:
            continue
        haf[i] = hlabel
        r = i // cols
        c = i - r * cols
        for k in range(8):
            rr = r + HBR[k]
            cc = c + HBC[k]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            j = rr * cols + cc
            if valid[j] != 0 and rec[j] == i and haf[j] == 0:
                stack.append(j)


@njit(cache=True)
def process_stream_outlets(outlets: np.ndarray, roots: np.ndarray,
                           receiver: np.ndarray, stream: np.ndarray,
                           accumulation: np.ndarray, valid: np.ndarray,
                           half_basins: np.ndarray):
    """Process independent stream-outlet drainage components in-place."""
    rows, cols = receiver.shape
    rec = receiver.ravel()
    sf = stream.ravel()
    af = accumulation.ravel()
    vf = valid.ravel()
    haf = half_basins.ravel()
    donors = np.empty(8, dtype=np.int64)
    stream_donors = np.empty(8, dtype=np.int64)
    stream_dirs = np.empty(8, dtype=np.int32)

    for oi in range(outlets.size):
        start = int(outlets[oi])
        start_pair = _root_pair(start, roots)
        if start_pair <= 0:
            continue
        task_cell = [start]
        task_basin = [start_pair]

        while len(task_cell) > 0:
            current = task_cell.pop()
            basin_even = task_basin.pop()

            while True:
                if haf[current] != 0:
                    break

                nd = _collect_donors(current, rows, cols, rec, vf, donors)
                nsd = 0
                for k in range(nd):
                    j = donors[k]
                    if sf[j] != 0:
                        stream_donors[nsd] = j
                        nsd += 1

                if nsd == 0:
                    # no_stream: follow donor with maximum accumulation.
                    while True:
                        nd = _collect_donors(current, rows, cols, rec, vf, donors)
                        if nd == 0:
                            haf[current] = basin_even
                            break
                        main = int(donors[0])
                        best_acc = af[main]
                        for k in range(1, nd):
                            j = int(donors[k])
                            if af[j] > best_acc:
                                main = j
                                best_acc = af[j]

                        updir = _dir_between(current, main, cols)
                        downdir = _downstream_dir(current, updir, cols, rec)
                        left = 0
                        right = 0
                        for k in range(nd):
                            j = int(donors[k])
                            thisdir = _dir_between(current, j, cols)
                            side = _haf_side(updir, downdir, thisdir)
                            if side == 2:
                                _label_overland(j, basin_even - 1, rows, cols, rec, vf, haf)
                                left += 1
                            elif side == 1:
                                _label_overland(j, basin_even, rows, cols, rec, vf, haf)
                                right += 1
                        if haf[current] == 0:
                            haf[current] = basin_even - 1 if left > right else basin_even
                        current = main
                    break

                downdir = _downstream_dir(
                    current, _dir_between(current, int(stream_donors[0]), cols), cols, rec
                )

                if nsd == 1:
                    main = int(stream_donors[0])
                    updir = _dir_between(current, main, cols)
                    left = 0
                    right = 0
                    for k in range(nd):
                        j = int(donors[k])
                        thisdir = _dir_between(current, j, cols)
                        side = _haf_side(updir, downdir, thisdir)
                        if side == 2:
                            _label_overland(j, basin_even - 1, rows, cols, rec, vf, haf)
                            left += 1
                        elif side == 1:
                            _label_overland(j, basin_even, rows, cols, rec, vf, haf)
                            right += 1
                    if haf[current] == 0:
                        haf[current] = basin_even - 1 if left > right else basin_even
                    current = main
                    continue

                # Junction.
                updir = _dir_between(current, int(stream_donors[0]), cols)
                for q in range(nsd):
                    stream_dirs[q] = _dir_between(current, int(stream_donors[q]), cols)
                left = 0
                right = 0
                for k in range(nd):
                    j = int(donors[k])
                    thisdir = _dir_between(current, j, cols)
                    is_stream_dir = False
                    for q in range(nsd):
                        if thisdir == stream_dirs[q]:
                            is_stream_dir = True
                            break
                    if is_stream_dir:
                        continue
                    side = _haf_side(updir, downdir, thisdir)
                    if side == 2:
                        _label_overland(j, basin_even - 1, rows, cols, rec, vf, haf)
                        left += 1
                    elif side == 1:
                        _label_overland(j, basin_even, rows, cols, rec, vf, haf)
                        right += 1
                haf[current] = basin_even - 1 if left >= right else basin_even

                for k in range(nsd - 1, -1, -1):
                    j = int(stream_donors[k])
                    pair = _root_pair(j, roots)
                    if pair > 0:
                        task_cell.append(j)
                        task_basin.append(pair)
                break


@njit(cache=True)
def find_residual_outlets(receiver: np.ndarray, valid: np.ndarray,
                          half_basins: np.ndarray) -> np.ndarray:
    rec = receiver.ravel()
    vf = valid.ravel()
    haf = half_basins.ravel()
    count = 0
    for i in range(rec.size):
        if vf[i] != 0 and haf[i] == 0:
            j = int(rec[i])
            if j < 0 or vf[j] == 0:
                count += 1
    out = np.empty(count, dtype=np.int64)
    p = 0
    for i in range(rec.size):
        if vf[i] != 0 and haf[i] == 0:
            j = int(rec[i])
            if j < 0 or vf[j] == 0:
                out[p] = i
                p += 1
    return out


@njit(cache=True)
def process_residual_outlets(outlets: np.ndarray, labels: np.ndarray,
                             receiver: np.ndarray, valid: np.ndarray,
                             half_basins: np.ndarray):
    rows, cols = receiver.shape
    rec = receiver.ravel()
    vf = valid.ravel()
    haf = half_basins.ravel()
    for k in range(outlets.size):
        root = int(outlets[k])
        _label_overland(root, int(labels[k]), rows, cols, rec, vf, haf)


@njit(cache=True)
def fill_unassigned(receiver: np.ndarray, valid: np.ndarray, half_basins: np.ndarray,
                    label_offset_pairs: int) -> int:
    """Defensive completion with compact labels. Returns components filled."""
    rows, cols = receiver.shape
    rec = receiver.ravel()
    vf = valid.ravel()
    haf = half_basins.ravel()
    count = 0
    for i in range(rec.size):
        if vf[i] != 0 and haf[i] == 0:
            count += 1
            label = 2 * (label_offset_pairs + count)
            _label_overland(i, label, rows, cols, rec, vf, haf)
    return count



# ============================================================================
# Stage 10 reusable stream / drainage topology kernels
# ============================================================================

@njit(cache=True)
def precompute_stream_seed_static(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    accumulation: np.ndarray,
    order: np.ndarray,
    static_ok_out: np.ndarray,
    higher_acc_max_out: np.ndarray,
):
    """
    Precompute threshold-independent parts of the GRASS-like SWALE seed test.

    For every valid cell we store:
      * static_ok: not on a DEM/nodata edge and not a flat according to the
        original processing-order rule;
      * higher_acc_max: max(acc[j] + 0.5) among higher neighbouring cells.

    At a later threshold T, the expensive condition
        stream_cells < 1
    becomes simply
        higher_acc_max[i] < T.

    This is computed once per DEM and reused at all thresholds.
    """
    rows, cols = dem_scaled.shape
    n = dem_scaled.size
    zf = dem_scaled.ravel()
    vf = valid.ravel()
    af = accumulation.ravel()
    okf = static_ok_out.ravel()
    upf = higher_acc_max_out.ravel()

    worked = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        okf[i] = 0
        upf[i] = 0.0

    for pos in range(order.size - 1, -1, -1):
        i = int(order[pos])
        if vf[i] == 0:
            continue

        worked[i] = 1
        r = i // cols
        c = i - r * cols

        edge = False
        flat = True
        max_up = 0.0

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc

                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    edge = True
                    break

                j = rr * cols + cc
                if vf[j] == 0:
                    edge = True
                    break

                if zf[j] > zf[i]:
                    candidate = af[j] + 0.5
                    if candidate > max_up:
                        max_up = candidate

                if worked[j] == 0 and zf[j] != zf[i]:
                    flat = False

            if edge:
                break

        if not edge and not flat:
            okf[i] = 1
        upf[i] = max_up



# ============================================================================
# Stage 11 block-parallel exact MFD
# ============================================================================

@njit(cache=True, inline="always")
def _conv_power(x: float, convergence: int) -> float:
    """Fast small-integer power used by Holmgren MFD."""
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
    # Generic path retained for allowed convergence values 7..10.
    out = 1.0
    for _ in range(convergence):
        out *= x
    return out


@njit(cache=True, parallel=True)
def build_rank_parallel(
    order: np.ndarray,
    rank_out: np.ndarray,
):
    """Build raster-index -> A* rank map in parallel.

    Every order position writes to a unique raster cell, so this loop has no
    data races. Invalid raster cells are never queried by MFD and do not need
    initialization.
    """
    for pos in prange(order.size):
        rank_out[int(order[pos])] = pos


@njit(cache=True, parallel=True)
def init_mfd_valid_parallel(
    order: np.ndarray,
    astar_receiver: np.ndarray,
    accumulation: np.ndarray,
    adjusted_receiver: np.ndarray,
):
    """Initialize only valid cells listed in `order` in parallel.

    The output arrays are created pre-filled with 0 / -1, avoiding a Numba
    full-grid validity scan on very sparse rectangular DEM extents.
    """
    rf = astar_receiver.reshape(astar_receiver.size)
    af = accumulation.reshape(accumulation.size)
    df = adjusted_receiver.reshape(adjusted_receiver.size)

    for pos in prange(order.size):
        i = int(order[pos])
        af[i] = 1.0
        df[i] = rf[i]


@njit(cache=True, parallel=True)
def mfd_weights_block_parallel(
    dem_scaled: np.ndarray,
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    rank: np.ndarray,
    reverse_offset: int,
    block_len: int,
    xres_scaled: float,
    yres_scaled: float,
    convergence: int,
    weights_out: np.ndarray,
):
    """
    Compute normalized MFD routing weights for a contiguous block of the
    high->low A* processing order.

    This phase contains the expensive neighbourhood scans and powers but does
    not depend on current accumulation values, so it is safe to parallelize.

    weights_out[k, 0:8] correspond to the fixed DR/DC neighbour order.
    weights_out[k, 8] is an exact A* fallback contribution used only if the
    fallback cannot be represented by one of those eight neighbour slots.
    """
    rows, cols = dem_scaled.shape
    zf = dem_scaled.ravel()
    vf = valid.ravel()
    rec0 = astar_receiver.ravel()
    rk = rank.ravel()
    nv = order.size

    diag = math.sqrt(xres_scaled * xres_scaled + yres_scaled * yres_scaled)

    for k in prange(block_len):
        for q in range(9):
            weights_out[k, q] = 0.0

        pos = nv - 1 - (reverse_offset + k)
        i = int(order[pos])

        r = i // cols
        c = i - r * cols

        # Match the reference MFD edge behaviour exactly.
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

            if rk[j] >= rk[i]:
                continue

            dz = zi - float(zf[j])
            if ct < 2:
                dist = yres_scaled
            elif ct < 4:
                dist = xres_scaled
            else:
                dist = diag

            if dz > 0.0:
                x = dz / dist
                w = _conv_power(x, convergence)
            elif dz == 0.0:
                x = 0.5 / dist
                w = _conv_power(x, convergence)
            else:
                continue

            weights_out[k, ct] = w
            sw += w
            if w > maxw:
                maxw = w
            if j == astar_j:
                astar_present = True

        if astar_j >= 0 and not astar_present:
            if maxw <= 0.0:
                maxw = 1.0

            represented = False
            ar = astar_j // cols
            ac = astar_j - ar * cols
            adr = ar - r
            adc = ac - c

            for ct in range(8):
                if DR[ct] == adr and DC[ct] == adc:
                    weights_out[k, ct] += maxw
                    sw += maxw
                    represented = True
                    break

            if not represented:
                weights_out[k, 8] = maxw
                sw += maxw

        # Reference special case: if there are no natural/fallback weights,
        # route the whole contribution to the A* receiver.
        if sw <= 0.0:
            if astar_j >= 0:
                weights_out[k, 8] = 1.0
            continue

        inv = 1.0 / sw
        for q in range(9):
            weights_out[k, q] *= inv


@njit(cache=True)
def mfd_scatter_block_serial(
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    reverse_offset: int,
    block_len: int,
    accumulation: np.ndarray,
    weights: np.ndarray,
):
    """
    Exact dependency-preserving MFD scatter for one precomputed weight block.

    This part remains serial because multiple upstream cells may update the
    same downstream cell and CPU Numba has no efficient portable atomic
    floating-point add. The expensive weight calculation is already done in
    parallel.
    """
    rows, cols = valid.shape
    vf = valid.ravel()
    rec0 = astar_receiver.ravel()
    acc = accumulation.ravel()
    nv = order.size

    for k in range(block_len):
        pos = nv - 1 - (reverse_offset + k)
        i = int(order[pos])
        value = acc[i]

        r = i // cols
        c = i - r * cols

        any_weight = False

        for ct in range(8):
            w = weights[k, ct]
            if w == 0.0:
                continue

            rr = r + DR[ct]
            cc = c + DC[ct]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            j = rr * cols + cc
            if vf[j] == 0:
                continue

            acc[j] += value * w
            any_weight = True

        w = weights[k, 8]
        if w != 0.0:
            j = int(rec0[i])
            if j >= 0:
                acc[j] += value * w
                any_weight = True


@njit(cache=True, parallel=True)
def adjusted_receiver_parallel(
    valid: np.ndarray,
    astar_receiver: np.ndarray,
    order: np.ndarray,
    rank: np.ndarray,
    accumulation: np.ndarray,
    adjusted_receiver: np.ndarray,
):
    """Parallel final adjusted-receiver selection from completed accumulation."""
    rows, cols = valid.shape
    vf = valid.ravel()
    rec0 = astar_receiver.ravel()
    rk = rank.ravel()
    acc = accumulation.ravel()
    adj = adjusted_receiver.ravel()
    nv = order.size

    for pos in prange(nv):
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

            if rk[j] >= rk[i]:
                continue

            if acc[j] > best_acc:
                best = j
                best_acc = acc[j]

        adj[i] = best

@njit(cache=True)
def stream_mask_grasslike_static(
    valid: np.ndarray,
    accumulation: np.ndarray,
    receiver: np.ndarray,
    order: np.ndarray,
    threshold_cells: float,
    static_ok: np.ndarray,
    higher_acc_max: np.ndarray,
    stream_out: np.ndarray,
):
    """
    GRASS-like SWALE network using precomputed threshold-independent tests.

    Unlike stream_mask_grasslike(), this function no longer revisits DEM
    elevation comparisons or creates the large `worked` array at every
    threshold. Neighbour inspection is only required for cells that can
    actually become new stream seeds at the current threshold.
    """
    rows, cols = valid.shape
    vf = valid.ravel()
    af = accumulation.ravel()
    rec = receiver.ravel()
    sf = static_ok.ravel()
    uf = higher_acc_max.ravel()
    sw = stream_out.ravel()

    for i in range(sw.size):
        sw[i] = 0

    for pos in range(order.size - 1, -1, -1):
        i = int(order[pos])
        if vf[i] == 0:
            continue

        if (
            sw[i] == 0
            and sf[i] != 0
            and af[i] >= threshold_cells
            and uf[i] < threshold_cells
        ):
            r = i // cols
            c = i - r * cols
            has_swale_neighbor = False

            # static_ok guarantees a complete valid 3x3 neighbourhood.
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    j = (r + dr) * cols + (c + dc)
                    if sw[j] != 0:
                        has_swale_neighbor = True
                        break
                if has_swale_neighbor:
                    break

            if not has_swale_neighbor:
                sw[i] = 1

        if sw[i] != 0:
            j = int(rec[i])
            if j >= 0 and vf[j] != 0:
                sw[j] = 1


@njit(cache=True)
def drainage_components(
    receiver: np.ndarray,
    valid: np.ndarray,
    order: np.ndarray,
    component_out: np.ndarray,
) -> int:
    """
    Dense global drainage-component labels for the adjusted receiver graph.

    Receiver cells have a lower A* rank, therefore processing `order` from
    low to high propagates the already-known outlet component upstream.
    """
    rec = receiver.ravel()
    vf = valid.ravel()
    cf = component_out.ravel()

    for i in range(cf.size):
        cf[i] = 0

    ncomp = 0

    for pos in range(order.size):
        i = int(order[pos])
        if vf[i] == 0:
            continue

        j = int(rec[i])
        if j < 0 or vf[j] == 0:
            ncomp += 1
            cf[i] = ncomp
        else:
            cid = int(cf[j])
            if cid <= 0:
                # Defensive fallback; should not occur for an acyclic
                # adjusted receiver ordered by the A* rank.
                ncomp += 1
                cid = ncomp
            cf[i] = cid

    return ncomp


@njit(cache=True)
def mark_branch_roots_fast(
    receiver: np.ndarray,
    stream: np.ndarray,
    valid: np.ndarray,
    mark: np.ndarray,
    stream_donor_count: np.ndarray,
):
    """
    O(N) stream-root detection.

    The original reference kernel checks all eight neighbours for every stream
    cell. Here the number of stream donors is first accumulated directly onto
    receiver cells. A stream cell is a branch root iff:
      * it drains outside the current stream network, or
      * its receiver has >=2 stream donors.

    This is algebraically the same criterion used by mark_branch_roots().
    """
    rec = receiver.ravel()
    sf = stream.ravel()
    vf = valid.ravel()
    mf = mark.ravel()
    dc = stream_donor_count.ravel()

    for i in range(rec.size):
        mf[i] = 0
        dc[i] = 0

    for i in range(rec.size):
        if vf[i] == 0 or sf[i] == 0:
            continue
        j = int(rec[i])
        if j >= 0 and vf[j] != 0 and sf[j] != 0:
            if dc[j] < 255:
                dc[j] += 1

    for i in range(rec.size):
        if vf[i] == 0 or sf[i] == 0:
            continue

        j = int(rec[i])
        if j < 0 or vf[j] == 0 or sf[j] == 0:
            mf[i] = 1
        elif dc[j] >= 2:
            mf[i] = 1


@njit(cache=True)
def residual_component_status(
    drainage_component: np.ndarray,
    valid: np.ndarray,
    half_basins: np.ndarray,
    has_label: np.ndarray,
    has_unlabeled: np.ndarray,
):
    """Classify global drainage components after stream-half-basin tracing."""
    cf = drainage_component.ravel()
    vf = valid.ravel()
    hf = half_basins.ravel()

    for k in range(has_label.size):
        has_label[k] = 0
        has_unlabeled[k] = 0

    for i in range(cf.size):
        if vf[i] == 0:
            continue
        cid = int(cf[i])
        if cid <= 0:
            continue
        if hf[i] == 0:
            has_unlabeled[cid] = 1
        else:
            has_label[cid] = 1


@njit(cache=True)
def assign_residual_components(
    drainage_component: np.ndarray,
    valid: np.ndarray,
    half_basins: np.ndarray,
    residual_index: np.ndarray,
    branch_count: int,
):
    """
    Fill wholly streamless drainage components in one linear scan.

    residual_index[cid] is 1..R for residual components and 0 otherwise.
    """
    cf = drainage_component.ravel()
    vf = valid.ravel()
    hf = half_basins.ravel()

    for i in range(cf.size):
        if vf[i] == 0 or hf[i] != 0:
            continue
        cid = int(cf[i])
        if cid <= 0:
            continue
        rid = int(residual_index[cid])
        if rid > 0:
            hf[i] = 2 * (branch_count + rid)

@njit(cache=True)
def parent_bboxes(todo_parent: np.ndarray):
    """Dense parent IDs 1..N -> counts and bounding boxes."""
    rows, cols = todo_parent.shape
    tf = todo_parent.ravel()
    max_id = 0
    for i in range(tf.size):
        if tf[i] > max_id:
            max_id = int(tf[i])
    counts = np.zeros(max_id + 1, dtype=np.int64)
    rmin = np.full(max_id + 1, rows, dtype=np.int32)
    rmax = np.full(max_id + 1, -1, dtype=np.int32)
    cmin = np.full(max_id + 1, cols, dtype=np.int32)
    cmax = np.full(max_id + 1, -1, dtype=np.int32)
    for r in range(rows):
        base = r * cols
        for c in range(cols):
            pid = int(tf[base + c])
            if pid <= 0:
                continue
            counts[pid] += 1
            if r < rmin[pid]: rmin[pid] = r
            if r > rmax[pid]: rmax[pid] = r
            if c < cmin[pid]: cmin[pid] = c
            if c > cmax[pid]: cmax[pid] = c
    return counts, rmin, rmax, cmin, cmax


@njit(cache=True)
def apply_decisions(todo_parent: np.ndarray, half_basins: np.ndarray,
                    rejected_parent: np.ndarray, parent_final_id: np.ndarray,
                    child_action: np.ndarray, final: np.ndarray,
                    new_todo: np.ndarray):
    """Apply parent/child decisions in a single memory-linear pass.

    child_action[child] > 0 -> final slope-unit ID
    child_action[child] < 0 -> new dense parent ID = -value
    """
    tf = todo_parent.ravel()
    hf = half_basins.ravel()
    ff = final.ravel()
    nf = new_todo.ravel()
    for i in range(tf.size):
        pid = int(tf[i])
        nf[i] = 0
        if pid <= 0:
            continue
        if rejected_parent[pid] != 0:
            ff[i] = parent_final_id[pid]
            continue
        child = int(hf[i])
        if child <= 0 or child >= child_action.size:
            continue
        action = int(child_action[child])
        if action > 0:
            ff[i] = action
        elif action < 0:
            nf[i] = -action



@njit(cache=True, inline="always")
def _binary_search_child(child_ids: np.ndarray, lo: int, hi: int, target: int) -> int:
    """Return index of target in sorted child_ids[lo:hi], or -1."""
    left = lo
    right = hi - 1
    while left <= right:
        mid = (left + right) // 2
        value = int(child_ids[mid])
        if value < target:
            left = mid + 1
        elif value > target:
            right = mid - 1
        else:
            return mid
    return -1


@njit(cache=True)
def apply_parent_child_decisions(
    todo_parent: np.ndarray,
    half_basins: np.ndarray,
    rejected_parent: np.ndarray,
    parent_final_id: np.ndarray,
    parent_offsets: np.ndarray,
    pair_child_ids: np.ndarray,
    pair_actions: np.ndarray,
    final: np.ndarray,
    new_todo: np.ndarray,
):
    """
    Apply refinement decisions using the pair (parent_id, half_basin_id).

    This is intentionally different from a global child_action[half_basin_id].
    A half-basin generated at the current threshold can intersect more than
    one active parent. The intersection with each parent is treated as a
    distinct hierarchical child.

    pair_actions[k] > 0 -> final slope-unit ID
    pair_actions[k] < 0 -> new dense parent ID = -value
    """
    tf = todo_parent.ravel()
    hf = half_basins.ravel()
    ff = final.ravel()
    nf = new_todo.ravel()

    for i in range(tf.size):
        pid = int(tf[i])
        nf[i] = 0

        if pid <= 0:
            continue

        if rejected_parent[pid] != 0:
            ff[i] = int(parent_final_id[pid])
            continue

        child = int(hf[i])
        if child <= 0:
            continue

        lo = int(parent_offsets[pid])
        hi = int(parent_offsets[pid + 1])

        pos = _binary_search_child(pair_child_ids, lo, hi, child)
        if pos < 0:
            continue

        action = int(pair_actions[pos])
        if action > 0:
            ff[i] = action
        elif action < 0:
            nf[i] = -action

@njit(cache=True)
def compact_labels_inplace(labels: np.ndarray, valid: np.ndarray, mapping: np.ndarray):
    lf = labels.ravel()
    vf = valid.ravel()
    for i in range(lf.size):
        if vf[i] != 0 and lf[i] > 0:
            lf[i] = mapping[lf[i]]

@njit(cache=True)
def finalize_remaining(todo_parent: np.ndarray, final: np.ndarray, start_id: int) -> int:
    tf = todo_parent.ravel()
    ff = final.ravel()
    maxpid = 0
    for i in range(tf.size):
        if tf[i] > maxpid:
            maxpid = int(tf[i])
    counts = np.zeros(maxpid + 1, dtype=np.int64)
    for i in range(tf.size):
        pid = int(tf[i])
        if pid > 0:
            counts[pid] += 1
    lut = np.zeros(maxpid + 1, dtype=np.int32)
    nxt = start_id
    for pid in range(1, maxpid + 1):
        if counts[pid] > 0:
            lut[pid] = nxt
            nxt += 1
    for i in range(tf.size):
        pid = int(tf[i])
        if pid > 0:
            ff[i] = lut[pid]
            tf[i] = 0
    return nxt

@njit(cache=True)
def dense_parent_stats(todo: np.ndarray, hb: np.ndarray,
                       sin_a: np.ndarray, cos_a: np.ndarray, aspect_valid: np.ndarray,
                       parent_id: int, r0: int, r1: int, c0: int, c1: int,
                       max_child_label: int):
    """Memory-bounded parent statistics for very large parent windows."""
    counts = np.zeros(max_child_label + 1, dtype=np.int64)
    aspect_count = np.zeros(max_child_label + 1, dtype=np.int64)
    sum_sin = np.zeros(max_child_label + 1, dtype=np.float64)
    sum_cos = np.zeros(max_child_label + 1, dtype=np.float64)
    parent_cells = 0
    for r in range(r0, r1):
        for c in range(c0, c1):
            if todo[r, c] != parent_id:
                continue
            child = int(hb[r, c])
            if child <= 0 or child > max_child_label:
                continue
            parent_cells += 1
            counts[child] += 1
            if aspect_valid[r, c] != 0:
                aspect_count[child] += 1
                sum_sin[child] += float(sin_a[r, c])
                sum_cos[child] += float(cos_a[r, c])
    return counts, aspect_count, sum_sin, sum_cos, parent_cells

@njit(cache=True)
def count_unlabeled(valid: np.ndarray, labels: np.ndarray) -> int:
    vf = valid.ravel()
    lf = labels.ravel()
    n = 0
    for i in range(vf.size):
        if vf[i] != 0 and lf[i] <= 0:
            n += 1
    return n
