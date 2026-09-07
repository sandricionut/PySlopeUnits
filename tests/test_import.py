def test_package_imports():
    import pyslopeunits

    assert pyslopeunits.__version__ == "0.1.0"
    assert pyslopeunits.SlopeUnits is not None


def test_cli_parser_imports():
    from pyslopeunits.cli import build_parser

    assert build_parser() is not None
