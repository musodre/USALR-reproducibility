#!/usr/bin/env python3
"""Display the intended analysis order.

This helper intentionally does not launch expensive LAMMPS or analysis jobs.
It prevents accidental multi-day computation and serves as a machine-readable
workflow index.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ORDER = [
    "02_data_analysis/01_diffusion",
    "02_data_analysis/02_thermodynamics",
    "02_data_analysis/03_tmd",
    "02_data_analysis/04_pair_order",
    "02_data_analysis/05_shell_structure",
    "02_data_analysis/06_structure_factor",
    "02_data_analysis/07_structure_dynamics",
    "02_data_analysis/08_derivative_analysis",
    "02_data_analysis/09_clusters",
    "03_figures",
]

print("USALR reproducibility workflow")
print("=" * 40)
for i, rel in enumerate(ORDER, 1):
    folder = ROOT / rel
    scripts = sorted(p.name for p in folder.glob("*.py")) if folder.exists() else []
    print(f"{i:02d}. {rel}")
    for name in scripts:
        print(f"    - {name}")
