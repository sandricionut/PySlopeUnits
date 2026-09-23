"""Lightweight release smoke tests for PySlopeUnits v0.1.8.

These tests complement the existing project test suite; they are not intended
to replace algorithmic regression tests already present in the repository.
"""

from pyslopeunits import __version__, threshold_schedule
from pyslopeunits.cli import build_parser


def test_release_version():
    assert __version__ == "0.1.8"


def test_reference_threshold_schedule_30m():
    levels = threshold_schedule(
        threshold_m2=250_000.0,
        cell_area_m2=30.0 * 30.0,
        reduction_factor=2,
        max_iterations=12,
    )
    assert levels[0].cells == 277
    assert levels[1].cells == 138
    assert all(a.cells > b.cells for a, b in zip(levels, levels[1:]))


def test_cli_reference_defaults():
    args = build_parser().parse_args(["dem.tif", "run"])
    assert args.threshold_m2 == 250_000.0
    assert args.min_area_m2 == 100_000.0
    assert args.cv_min == 0.25
    assert args.reduction_factor == 2
    assert args.max_iterations == 12
    assert args.convergence == 5
    assert args.clean_size_m2 == 25_000.0
    assert args.hierarchy_mode == "lazy"
    assert args.candidate_cache_mode == "auto"
