"""Central filesystem configuration for the USALR reproducibility repository."""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent

RAW_DATA_ROOT = Path(
    os.environ.get("USALR_RAW_DATA", REPO_ROOT / "data" / "raw")
).expanduser().resolve()

DERIVED_DATA_ROOT = Path(
    os.environ.get("USALR_DERIVED_DATA", REPO_ROOT / "data" / "derived")
).expanduser().resolve()

FIGURE_OUTPUT_ROOT = Path(
    os.environ.get("USALR_FIGURES", REPO_ROOT / "outputs" / "figures")
).expanduser().resolve()

TABLE_OUTPUT_ROOT = Path(
    os.environ.get("USALR_TABLES", REPO_ROOT / "outputs" / "tables")
).expanduser().resolve()

def ensure_output_dirs():
    for p in (DERIVED_DATA_ROOT, FIGURE_OUTPUT_ROOT, TABLE_OUTPUT_ROOT):
        p.mkdir(parents=True, exist_ok=True)
