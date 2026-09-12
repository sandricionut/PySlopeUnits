from pathlib import Path

import numpy as np

from pyslopeunits.hydro_domains import _adaptive_tree_partition
from pyslopeunits.resources import choose_coarse_resolution, make_resource_plan, SystemResources


def test_adaptive_tree_partition_is_hydrological_and_bounded():
    # Root 0, two tributary subtrees rooted at 1 and 2.
    receiver = np.array([-1, 0, 0, 1, 1, 2, 2], dtype=np.int32).reshape(1, -1)
    valid = np.ones_like(receiver, dtype=np.uint8)
    order = np.arange(7, dtype=np.int32)  # downstream -> upstream

    labels, sizes, _, _ = _adaptive_tree_partition(receiver, valid, order, 6)
    assert int(sizes.max()) <= 6
    assert labels[0, 3] == labels[0, 1]
    assert labels[0, 5] == labels[0, 2]
    assert labels[0, 1] != labels[0, 2]


def test_resource_plan_is_memory_adaptive(tmp_path: Path):
    resources = SystemResources(
        platform="test",
        machine="test",
        logical_cpus=8,
        physical_cpus=4,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        disk_free_bytes=100 * 1024**3,
    )
    plan = make_resource_plan(
        resources,
        target_width_m=5_000_000,
        target_height_m=4_000_000,
        fine_resolution_m=30.0,
    )
    assert plan.workers == 4
    assert plan.target_fine_cells > 1_000_000
    assert plan.coarse_resolution_m >= 120.0


def test_choose_coarse_resolution_respects_budget():
    res = choose_coarse_resolution(
        target_width_m=1_000_000,
        target_height_m=1_000_000,
        fine_resolution_m=30.0,
        coarse_cell_budget=10_000_000,
    )
    cells = int(np.ceil(1_000_000 / res)) ** 2
    assert cells <= 10_000_000
