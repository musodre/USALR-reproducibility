#!/usr/bin/env python3
"""Check repository path configuration without running scientific analyses."""
from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import (
    REPO_ROOT,
    RAW_DATA_ROOT,
    DERIVED_DATA_ROOT,
    FIGURE_OUTPUT_ROOT,
    TABLE_OUTPUT_ROOT,
)

print("REPO_ROOT       =", REPO_ROOT)
print("RAW_DATA_ROOT   =", RAW_DATA_ROOT)
print("DERIVED_DATA    =", DERIVED_DATA_ROOT)
print("FIGURE_OUTPUT   =", FIGURE_OUTPUT_ROOT)
print("TABLE_OUTPUT    =", TABLE_OUTPUT_ROOT)
print()
print("Raw data exists:", RAW_DATA_ROOT.exists())
print("Derived exists :", DERIVED_DATA_ROOT.exists())
