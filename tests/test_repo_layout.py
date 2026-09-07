from pathlib import Path


def test_core_files_present():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src/pyslopeunits/pipeline.py",
        root / "src/pyslopeunits/engine.py",
        root / "src/pyslopeunits/precompute.py",
        root / "src/pyslopeunits/mfd_parallel.py",
        root / "src/pyslopeunits/memmap_store.py",
        root / "examples/run_10m_20m.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    assert not missing, f"Missing PySlopeUnits core files: {missing}"
