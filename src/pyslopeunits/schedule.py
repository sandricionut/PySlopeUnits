from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdLevel:
    level: int
    cells: int
    area_m2: float


def threshold_schedule(
    threshold_m2: float,
    cell_area_m2: float,
    reduction_factor: int = 2,
    max_iterations: int = 12,
) -> list[ThresholdLevel]:
    """Exact integer-cell schedule used by the iterative r.slopeunits logic."""
    if threshold_m2 <= 0 or cell_area_m2 <= 0:
        raise ValueError("threshold and cell area must be positive")
    if reduction_factor <= 1:
        raise ValueError("reduction_factor must be > 1")

    cells = max(1, int(float(threshold_m2) / float(cell_area_m2)))
    out: list[ThresholdLevel] = []

    for level in range(1, int(max_iterations) + 1):
        out.append(
            ThresholdLevel(
                level=level,
                cells=int(cells),
                area_m2=float(cells) * float(cell_area_m2),
            )
        )

        new_cells = int(cells - cells / int(reduction_factor))
        if new_cells < int(reduction_factor):
            break
        if new_cells == cells:
            break
        cells = new_cells

    return out
