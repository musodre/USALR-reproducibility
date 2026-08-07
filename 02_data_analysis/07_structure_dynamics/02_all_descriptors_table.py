#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete structure--dynamics hierarchy table
=============================================

Descriptors compared on EXACTLY THE SAME (P,T) STATES:

    -s2*
    tau
    g2/g1
    n2/n1
    A_SALR

Metrics:

    Pearson r_P
    Spearman r_S
    R2 linear
    R2 quadratic

The script starts from the data set already used in the
three-panel structure--dynamics figure and intersects it
with the available s2 and tau states.

Outputs
-------
structure_dynamics_hierarchy/

    complete_hierarchy_statistics.dat
    complete_hierarchy_table.tex
    complete_hierarchy_common_states.dat
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from pathlib import Path
from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import DERIVED_DATA_ROOT



# =============================================================================
# 1. INPUT FILES
# =============================================================================

# Data already used in Fig. 11:
#
# expected to contain:
# P, T, lnD, g2_g1, n2_n1, A_SALR
#
HIERARCHY_FILE = DERIVED_DATA_ROOT / "structure_dynamics_hierarchy" / "structure_dynamics_hierarchy_data.dat"


# -------------------------------------------------------------------------
# Global pair-order files
#
# Change only these paths if needed.
# -------------------------------------------------------------------------

S2_FILE = DERIVED_DATA_ROOT / "s2" / "s2_global.dat"

TAU_FILE = DERIVED_DATA_ROOT / "tau" / "tau_global.dat"


# =============================================================================
# 2. OUTPUT
# =============================================================================

OUTDIR = DERIVED_DATA_ROOT / "structure_dynamics_hierarchy"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


OUT_STATS = (
    OUTDIR
    /
    "complete_hierarchy_statistics.dat"
)


OUT_COMMON = (
    OUTDIR
    /
    "complete_hierarchy_common_states.dat"
)


OUT_LATEX = (
    OUTDIR
    /
    "complete_hierarchy_table.tex"
)


# =============================================================================
# 3. THERMODYNAMIC DOMAIN
# =============================================================================

T_MIN = 0.20
T_MAX = 0.60

P_MIN = 0.10
P_MAX = 6.00


# =============================================================================
# 4. CHECK FILES
# =============================================================================

for filename, label in [
    (HIERARCHY_FILE, "Hierarchy data"),
    (S2_FILE, "s2 data"),
    (TAU_FILE, "tau data"),
]:

    if not filename.exists():

        raise FileNotFoundError(
            f"\n{label} not found:\n"
            f"{filename.resolve()}"
        )


# =============================================================================
# 5. READ FIGURE-11 DATA
# =============================================================================

hier = pd.read_csv(
    HIERARCHY_FILE,
    sep=r"\s+",
    comment="#"
)


required_hier = [
    "P",
    "T",
    "lnD",
    "g2_g1",
    "n2_n1",
    "A_SALR",
]


missing = [
    col
    for col in required_hier
    if col not in hier.columns
]


if missing:

    raise RuntimeError(
        "Missing columns in hierarchy data:\n"
        +
        "\n".join(missing)
    )


# =============================================================================
# 6. READ s2
# =============================================================================
#
# Expected:
#
#     P  T  rho  s2
#
# =============================================================================

s2_raw = pd.read_csv(
    S2_FILE,
    sep=r"\s+",
    comment="#",
    header=None,
    engine="python"
)


if s2_raw.shape[1] < 4:

    raise RuntimeError(
        "s2_global.dat must contain at least four columns:\n"
        "P T rho s2"
    )


s2 = pd.DataFrame({
    "P": pd.to_numeric(
        s2_raw.iloc[:, 0],
        errors="coerce"
    ),

    "T": pd.to_numeric(
        s2_raw.iloc[:, 1],
        errors="coerce"
    ),

    "s2": pd.to_numeric(
        s2_raw.iloc[:, 3],
        errors="coerce"
    ),
})


s2["minus_s2"] = (
    -s2["s2"]
)


# =============================================================================
# 7. READ tau
# =============================================================================
#
# Expected:
#
#     P  T  rho  tau
#
# Duplicates are averaged before merging.
# =============================================================================

tau_raw = pd.read_csv(
    TAU_FILE,
    sep=r"\s+",
    comment="#",
    header=None,
    engine="python"
)


if tau_raw.shape[1] < 4:

    raise RuntimeError(
        "tau_global.dat must contain at least four columns:\n"
        "P T rho tau"
    )


tau = pd.DataFrame({
    "P": pd.to_numeric(
        tau_raw.iloc[:, 0],
        errors="coerce"
    ),

    "T": pd.to_numeric(
        tau_raw.iloc[:, 1],
        errors="coerce"
    ),

    "tau": pd.to_numeric(
        tau_raw.iloc[:, 3],
        errors="coerce"
    ),
})


# =============================================================================
# 8. CLEAN INPUT DATA
# =============================================================================

hier = hier.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=required_hier
)


s2 = s2.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=[
        "P",
        "T",
        "minus_s2"
    ]
)


tau = tau.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=[
        "P",
        "T",
        "tau"
    ]
)


# =============================================================================
# 9. THERMODYNAMIC FILTER
# =============================================================================

def domain_filter(df):

    return df[
        (df["P"] >= P_MIN)
        &
        (df["P"] <= P_MAX)
        &
        (df["T"] >= T_MIN)
        &
        (df["T"] <= T_MAX)
    ].copy()


hier = domain_filter(
    hier
)

s2 = domain_filter(
    s2
)

tau = domain_filter(
    tau
)


# =============================================================================
# 10. MERGE KEYS
# =============================================================================

def add_keys(df):

    out = df.copy()

    out["P_key"] = np.round(
        out["P"].to_numpy(dtype=float),
        6
    )

    out["T_key"] = np.round(
        out["T"].to_numpy(dtype=float),
        6
    )

    return out


hier = add_keys(
    hier
)

s2 = add_keys(
    s2
)

tau = add_keys(
    tau
)


# =============================================================================
# 11. AVERAGE DUPLICATES
# =============================================================================

s2 = (
    s2
    .groupby(
        [
            "P_key",
            "T_key"
        ],
        as_index=False
    )
    .agg({
        "minus_s2": "mean"
    })
)


tau = (
    tau
    .groupby(
        [
            "P_key",
            "T_key"
        ],
        as_index=False
    )
    .agg({
        "tau": "mean"
    })
)


# =============================================================================
# 12. EXACT COMMON-STATE INTERSECTION
# =============================================================================

common = pd.merge(
    hier,
    s2,
    on=[
        "P_key",
        "T_key"
    ],
    how="inner"
)


common = pd.merge(
    common,
    tau,
    on=[
        "P_key",
        "T_key"
    ],
    how="inner"
)


common["P"] = (
    common["P_key"]
)

common["T"] = (
    common["T_key"]
)


required_common = [
    "P",
    "T",
    "lnD",
    "minus_s2",
    "tau",
    "g2_g1",
    "n2_n1",
    "A_SALR",
]


common = common.replace(
    [np.inf, -np.inf],
    np.nan
)


common = common.dropna(
    subset=required_common
).copy()


common = common.sort_values(
    [
        "T",
        "P"
    ]
).reset_index(
    drop=True
)


# =============================================================================
# 13. REPORT DATA AVAILABILITY
# =============================================================================

print()

print(
    "=" * 92
)

print(
    "COMMON-STATE STRUCTURE--DYNAMICS HIERARCHY"
)

print(
    "=" * 92
)


print(
    f"Figure-11 states = {len(hier)}"
)

print(
    f"s2 states        = {len(s2)}"
)

print(
    f"tau states       = {len(tau)}"
)

print()

print(
    f"FINAL common states = {len(common)}"
)


print(
    f"P range = "
    f"{common['P'].min():.3f}"
    f" -- "
    f"{common['P'].max():.3f}"
)


print(
    f"T range = "
    f"{common['T'].min():.3f}"
    f" -- "
    f"{common['T'].max():.3f}"
)


# =============================================================================
# 14. SAVE COMMON STATES
# =============================================================================

common[
    [
        "P",
        "T",
        "lnD",
        "minus_s2",
        "tau",
        "g2_g1",
        "n2_n1",
        "A_SALR",
    ]
].to_csv(
    OUT_COMMON,
    sep=" ",
    index=False,
    float_format="%.10e"
)


# =============================================================================
# 15. STATISTICAL ANALYSIS
# =============================================================================

def calculate_statistics(
    dataframe,
    column,
    label
):

    x = dataframe[
        column
    ].to_numpy(
        dtype=float
    )

    y = dataframe[
        "lnD"
    ].to_numpy(
        dtype=float
    )


    # ------------------------------------------------------------------
    # Pearson / Spearman
    # ------------------------------------------------------------------

    rP = pearsonr(
        x,
        y
    )[0]


    rS = spearmanr(
        x,
        y
    )[0]


    # ------------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------------

    coef_lin = np.polyfit(
        x,
        y,
        1
    )


    y_lin = np.polyval(
        coef_lin,
        x
    )


    ss_tot = np.sum(
        (
            y
            -
            np.mean(y)
        ) ** 2
    )


    ss_lin = np.sum(
        (
            y
            -
            y_lin
        ) ** 2
    )


    R2_lin = (
        1.0
        -
        ss_lin / ss_tot
    )


    # ------------------------------------------------------------------
    # Quadratic regression
    # ------------------------------------------------------------------

    coef_quad = np.polyfit(
        x,
        y,
        2
    )


    y_quad = np.polyval(
        coef_quad,
        x
    )


    ss_quad = np.sum(
        (
            y
            -
            y_quad
        ) ** 2
    )


    R2_quad = (
        1.0
        -
        ss_quad / ss_tot
    )


    return {
        "descriptor": label,
        "column": column,
        "N": len(x),
        "Pearson": rP,
        "Spearman": rS,
        "R2_linear": R2_lin,
        "R2_quadratic": R2_quad,
    }


# =============================================================================
# 16. CALCULATE ALL FIVE
# =============================================================================

results = [

    calculate_statistics(
        common,
        "minus_s2",
        r"$-s_2^*$"
    ),

    calculate_statistics(
        common,
        "tau",
        r"$\tau$"
    ),

    calculate_statistics(
        common,
        "n2_n1",
        r"$n_2/n_1$"
    ),

    calculate_statistics(
        common,
        "g2_g1",
        r"$g_2/g_1$"
    ),

    calculate_statistics(
        common,
        "A_SALR",
        r"$A_{\mathrm{SALR}}$"
    ),
]


# =============================================================================
# 17. PRINT RESULTS
# =============================================================================

print()

print(
    "=" * 92
)

print(
    "COMPLETE HIERARCHY"
)

print(
    "=" * 92
)


for result in results:

    print()

    print(
        f"Descriptor = "
        f"{result['descriptor']}"
    )

    print(
        f"N = "
        f"{result['N']}"
    )

    print(
        f"Pearson      = "
        f"{result['Pearson']: .6f}"
    )

    print(
        f"Spearman     = "
        f"{result['Spearman']: .6f}"
    )

    print(
        f"R2 linear    = "
        f"{result['R2_linear']: .6f}"
    )

    print(
        f"R2 quadratic = "
        f"{result['R2_quadratic']: .6f}"
    )


# =============================================================================
# 18. SAVE NUMERIC STATISTICS
# =============================================================================

with open(
    OUT_STATS,
    "w"
) as handle:

    handle.write(
        "# descriptor N Pearson Spearman "
        "R2_linear R2_quadratic\n"
    )


    for result in results:

        descriptor_plain = (
            result[
                "column"
            ]
        )


        handle.write(
            f"{descriptor_plain} "
            f"{result['N']} "
            f"{result['Pearson']:.10e} "
            f"{result['Spearman']:.10e} "
            f"{result['R2_linear']:.10e} "
            f"{result['R2_quadratic']:.10e}\n"
        )


# =============================================================================
# 19. GENERATE LATEX TABLE
# =============================================================================

latex_lines = []


latex_lines.append(
    r"\begin{table}[htbp]"
)

latex_lines.append(
    r"    \centering"
)

latex_lines.append(
    r"    \caption{"
)

latex_lines.append(
    r"    Quantitative comparison of the global, shell-resolved, and "
)

latex_lines.append(
    r"    intermediate-range structural descriptors with the reduced "
)

latex_lines.append(
    r"    diffusion coefficient. All statistics are evaluated over the "
)

latex_lines.append(
    rf"    same set of ${len(common)}$ thermodynamic states in the domain "
)

latex_lines.append(
    rf"    ${T_MIN:.2f}\leq T^*\leq{T_MAX:.2f}$ and "
)

latex_lines.append(
    rf"    ${P_MIN:.2f}\leq P^*\leq{P_MAX:.2f}$. "
)

latex_lines.append(
    r"    The Pearson coefficient $r_{\rm P}$ measures linear association, "
)

latex_lines.append(
    r"    whereas the Spearman coefficient $r_{\rm S}$ quantifies monotonic "
)

latex_lines.append(
    r"    correspondence. $R_{\rm lin}^2$ and $R_{\rm quad}^2$ denote the "
)

latex_lines.append(
    r"    coefficients of determination for the linear and quadratic "
)

latex_lines.append(
    r"    regressions of $\ln D^*$ against each structural descriptor."
)

latex_lines.append(
    r"    }"
)

latex_lines.append(
    r"    \label{tab:structure_dynamics_hierarchy}"
)

latex_lines.append(
    r"    \begin{tabular}{lccccc}"
)

latex_lines.append(
    r"        \hline\hline"
)

latex_lines.append(
    r"        Descriptor"
    r" & $N$"
    r" & $r_{\rm P}$"
    r" & $r_{\rm S}$"
    r" & $R_{\rm lin}^2$"
    r" & $R_{\rm quad}^2$ \\"
)

latex_lines.append(
    r"        \hline"
)


for result in results:

    line = (
        "        "
        f"{result['descriptor']}"
        " & "
        f"{result['N']}"
        " & "
        f"{result['Pearson']:.3f}"
        " & "
        f"{result['Spearman']:.3f}"
        " & "
        f"{result['R2_linear']:.3f}"
        " & "
        f"{result['R2_quadratic']:.3f}"
        r" \\"
    )

    latex_lines.append(
        line
    )


latex_lines.append(
    r"        \hline\hline"
)

latex_lines.append(
    r"    \end{tabular}"
)

latex_lines.append(
    r"\end{table}"
)


latex_code = "\n".join(
    latex_lines
)


with open(
    OUT_LATEX,
    "w"
) as handle:

    handle.write(
        latex_code
    )

    handle.write(
        "\n"
    )


# =============================================================================
# 20. SHOW LATEX TABLE IN TERMINAL
# =============================================================================

print()

print(
    "=" * 92
)

print(
    "LATEX TABLE"
)

print(
    "=" * 92
)

print()

print(
    latex_code
)


# =============================================================================
# 21. FINAL REPORT
# =============================================================================

print()

print(
    "=" * 92
)

print(
    "OUTPUT"
)

print(
    "=" * 92
)


print(
    f"Common states:\n"
    f"  {OUT_COMMON.resolve()}"
)


print()


print(
    f"Statistics:\n"
    f"  {OUT_STATS.resolve()}"
)


print()


print(
    f"LaTeX table:\n"
    f"  {OUT_LATEX.resolve()}"
)


print()

print(
    "✓ All five descriptors evaluated on exactly the same states."
)

print(
    "✓ tau duplicates averaged before the merge."
)

print(
    "✓ s2 read from the fourth column."
)

print(
    "✓ tau read from the fourth column."
)

print(
    "✓ Pearson and Spearman recalculated."
)

print(
    "✓ Linear and quadratic R2 recalculated."
)

print(
    "✓ Publication-ready LaTeX table generated."
)

print()

print(
    "=" * 92
)
