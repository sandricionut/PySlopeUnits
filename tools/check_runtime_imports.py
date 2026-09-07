#!/usr/bin/env python3
"""List imported top-level modules in the exact local PySlopeUnits source tree.

This intentionally does not modify pyproject.toml automatically; import-name to
PyPI-package mappings are not always one-to-one and should be reviewed.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pyslopeunits"
STDLIB = set(getattr(sys, "stdlib_module_names", ()))
LOCAL = {p.stem for p in SRC.glob("*.py")}

mods: set[str] = set()
for py in SRC.rglob("*.py"):
    try:
        tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        print(f"WARNING: could not parse {py}: {exc}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])

third_party = sorted(m for m in mods if m not in STDLIB and m not in LOCAL and m != "pyslopeunits")
print("Third-party import names detected in PySlopeUnits:")
for mod in third_party:
    print(f"  - {mod}")
