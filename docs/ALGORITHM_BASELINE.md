# Algorithm baseline

The 0.1.0 release packages the validated hierarchical graph-based PySlopeUnits research pipeline without changing its numerical delineation logic.

The public pipeline is implemented in `src/pyslopeunits/pipeline.py`. Internal cache identifiers retained from earlier research iterations are implementation details and are not part of the public API.

Core stages:

1. terrain routing and MFD accumulation;
2. reusable memory-mapped hydrological state;
3. multi-threshold stream half-basin candidates;
4. nested candidate intersection DAG;
5. graph-based candidate selection;
6. residual-cell assignment;
7. four-neighbour connected-component consolidation;
8. GeoTIFF and optional GeoPackage output.
