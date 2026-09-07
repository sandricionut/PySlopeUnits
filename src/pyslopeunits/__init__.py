"""PySlopeUnits public API."""

from .__about__ import __version__
from .pipeline import SlopeUnits, SlopeUnitsResult
from .precompute import CandidateHierarchyPrecomputer, CandidatePreparationResult
from .experiments import StorageEstimate, estimate_work_storage
from .schedule import ThresholdLevel, threshold_schedule
from .vector_export import export_geopackage, VectorExportResult
from .clean import clean_slope_units, CleanResult
from .diagnostics import write_boundary_raster, write_hashed_display_raster
from .dag import CandidateDAG, DAGBuildResult

__all__ = [
    "__version__",
    "SlopeUnits",
    "SlopeUnitsResult",
    "CandidateHierarchyPrecomputer",
    "CandidatePreparationResult",
    "StorageEstimate",
    "estimate_work_storage",
    "ThresholdLevel",
    "threshold_schedule",
    "export_geopackage",
    "VectorExportResult",
    "clean_slope_units",
    "CleanResult",
    "write_boundary_raster",
    "write_hashed_display_raster",
    "CandidateDAG",
    "DAGBuildResult",
]
