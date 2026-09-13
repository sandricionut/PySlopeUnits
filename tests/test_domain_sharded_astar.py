import numpy as np

from pyslopeunits.fine_hydrology import _astar_domain_sharded_impl, _count_domains
from pyslopeunits.kernels import astar_route_preallocated


def test_domain_sharded_astar_is_pixel_exact_to_global_heap():
    rng = np.random.default_rng(42)
    rows, cols = 80, 93
    dem = (rng.normal(100.0, 20.0, (rows, cols)) * 1000.0).astype(np.int32)
    valid = np.ones((rows, cols), dtype=np.uint8)
    valid[rng.random((rows, cols)) < 0.08] = 0
    dem[valid == 0] = 0
    nvalid = int(valid.sum())

    rec_ref = np.full((rows, cols), -1, dtype=np.int64)
    order_ref = np.empty(nvalid, dtype=np.int64)
    edge_ref = np.zeros((rows, cols), dtype=np.uint8)
    inlist = np.zeros((rows, cols), dtype=np.uint8)
    worked = np.zeros((rows, cols), dtype=np.uint8)
    heap_idx_ref = np.empty(nvalid, dtype=np.int64)
    heap_age_ref = np.empty(nvalid, dtype=np.int64)

    visited_ref = astar_route_preallocated(
        dem, valid, 1000.0, 1000.0,
        rec_ref, order_ref, edge_ref,
        inlist, worked, heap_idx_ref, heap_age_ref,
    )

    coarse_rows, coarse_cols = 8, 10
    coarse = np.empty((coarse_rows, coarse_cols), dtype=np.int32)
    for r in range(coarse_rows):
        for c in range(coarse_cols):
            coarse[r, c] = 1 + ((r * 3 + c * 5) % 7)
    coarse[2, 3] = 0
    coarse[5, 8] = 0

    fallback_domain = int(coarse.max()) + 1
    counts = np.zeros(fallback_domain + 1, dtype=np.int64)

    fine_c = 0.0
    fine_f = float(rows)
    fine_xres = 1.0
    fine_yres = 1.0
    coarse_c = 0.0
    coarse_f = float(rows)
    coarse_xres = cols / coarse_cols
    coarse_yres = rows / coarse_rows

    _count_domains(
        valid,
        fine_c, fine_f, fine_xres, fine_yres,
        coarse,
        coarse_c, coarse_f, coarse_xres, coarse_yres,
        fallback_domain,
        counts,
    )

    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    sizes = np.zeros(counts.size, dtype=np.int64)

    rec_new = np.full((rows, cols), -1, dtype=np.int64)
    order_new = np.empty(nvalid, dtype=np.int64)
    edge_new = np.zeros((rows, cols), dtype=np.uint8)
    state = np.zeros((rows, cols), dtype=np.uint8)
    heap_idx = np.empty(nvalid, dtype=np.int64)
    heap_age = np.empty(nvalid, dtype=np.int64)
    gheap = np.zeros(counts.size, dtype=np.int64)
    gpos = np.full(counts.size, -1, dtype=np.int64)

    visited_new = _astar_domain_sharded_impl(
        dem, valid, 1000.0, 1000.0,
        rec_new, order_new, edge_new,
        state, heap_idx, heap_age,
        offsets, sizes, coarse,
        fine_c, fine_f, fine_xres, fine_yres,
        coarse_c, coarse_f, coarse_xres, coarse_yres,
        fallback_domain, gheap, gpos,
    )

    assert visited_ref == visited_new == nvalid
    assert np.array_equal(order_ref, order_new)
    assert np.array_equal(rec_ref, rec_new)
    assert np.array_equal(edge_ref, edge_new)
