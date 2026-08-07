#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure--dynamics hierarchy
=============================

Figure:
    (a) ln D* versus g2/g1
    (b) ln D* versus n2/n1
    (c) ln D* versus A_SALR

All panels use the same thermodynamic domain:

        T* >= 0.20
        P* <= 6.00

and the same temperature color scale.

The script:
    1. reads the original A_SALR table;
    2. extracts shell-resolved descriptors directly from RDF files;
    3. merges all descriptors with diffusion through (P,T);
    4. performs linear and quadratic regressions;
    5. reports Pearson, Spearman, R2_linear and R2_quadratic;
    6. produces the three-panel hierarchy figure.

No smoothing is applied to the scatter data.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from numpy.polynomial import Polynomial

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pathlib import Path
from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT



# =============================================================================
# 1. PATHS
# =============================================================================

# -------------------------------------------------------------------------
# Main simulation tree
# -------------------------------------------------------------------------

SIM_ROOT = RAW_DATA_ROOT / "allpress"


# -------------------------------------------------------------------------
# Original A_SALR table
# -------------------------------------------------------------------------

ASALR_FILE = DERIVED_DATA_ROOT / "static_structure_factor" / "Asalr_analysis" / "SALR_area_summary.dat"


# -------------------------------------------------------------------------
# Optional pre-existing shell table.
#
# If this file does not exist, shell descriptors are extracted directly
# from the RDF files.
# -------------------------------------------------------------------------

SHELL_TABLE = DERIVED_DATA_ROOT / "shell_resolved_structure" / "shell_descriptors_global.dat"


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

OUTDIR = DERIVED_DATA_ROOT / "structure_dynamics_hierarchy"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_PDF = OUTDIR / "structure_dynamics_hierarchy.pdf"
OUT_PNG = OUTDIR / "structure_dynamics_hierarchy.png"

OUT_DATA = OUTDIR / "structure_dynamics_hierarchy_data.dat"
OUT_STATS = OUTDIR / "structure_dynamics_hierarchy_statistics.dat"

OUT_SHELL = OUTDIR / "shell_resolved_global.dat"


# =============================================================================
# 2. COMMON THERMODYNAMIC DOMAIN
# =============================================================================

T_MIN = 0.20
T_MAX = 0.60

P_MIN = 0.10
P_MAX = 6.00

PT_TOL = 5.0e-4


# =============================================================================
# 3. RDF WINDOWS
# =============================================================================

R1_MIN = 1.10
R1_MAX = 1.35

R2_MIN = 1.85
R2_MAX = 2.15


# =============================================================================
# 4. GRAPHICAL SETTINGS
# =============================================================================

FIGSIZE = (14.0, 5.0)

CMAP = "plasma"

POINT_SIZE = 22
POINT_ALPHA = 0.72
POINT_EDGEWIDTH = 0.20

FIT_LW = 2.0

LABEL_FS = 18
TICK_FS = 11.5
PANEL_FS = 16

STATS_FS = 10.5

CBAR_LABEL_FS = 18
CBAR_TICK_FS = 11.5

GRID_ALPHA = 0.20
GRID_LW = 0.55


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "axes.linewidth": 1.15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


# =============================================================================
# 5. AUXILIARY FUNCTIONS
# =============================================================================

def parse_PT_from_rdf_filename(filename):
    """
    Extract P and T from filenames such as:

        P_1.000_T_0.20.rdf
    """

    pattern = (
        r"P_([0-9]+(?:\.[0-9]+)?)"
        r"_T_([0-9]+(?:\.[0-9]+)?)"
        r"\.rdf$"
    )

    match = re.search(
        pattern,
        filename.name
    )

    if match is None:
        return None

    pressure = float(
        match.group(1)
    )

    temperature = float(
        match.group(2)
    )

    return pressure, temperature


def read_last_rdf_block(filename):
    """
    Read the last RDF block from a LAMMPS fix ave/time-style RDF file.

    Expected numerical structure:

        timestep  number_of_rows
        index  r  g(r)  coord(r)
        ...

    The last complete block is returned.
    """

    blocks = []

    with open(
        filename,
        "r"
    ) as handle:

        lines = handle.readlines()


    i = 0
    nlines = len(lines)

    while i < nlines:

        line = lines[i].strip()

        if (
            not line
            or
            line.startswith("#")
        ):
            i += 1
            continue


        parts = line.split()


        # -------------------------------------------------------------
        # Header of an RDF block:
        #
        # timestep  Nrows
        # -------------------------------------------------------------

        if len(parts) == 2:

            try:
                timestep = int(
                    float(parts[0])
                )

                nrows = int(
                    float(parts[1])
                )

            except ValueError:
                i += 1
                continue


            data = []

            for j in range(
                i + 1,
                min(
                    i + 1 + nrows,
                    nlines
                )
            ):

                row = lines[j].strip()

                if (
                    not row
                    or
                    row.startswith("#")
                ):
                    continue

                values = row.split()

                if len(values) < 3:
                    continue

                try:
                    numbers = [
                        float(x)
                        for x in values
                    ]

                except ValueError:
                    continue

                data.append(
                    numbers
                )


            if len(data) == nrows:

                arr = np.asarray(
                    data,
                    dtype=float
                )

                blocks.append(
                    (
                        timestep,
                        arr
                    )
                )


            i += (
                nrows
                +
                1
            )

        else:

            i += 1


    if len(blocks) == 0:

        raise RuntimeError(
            f"No complete RDF block found in:\n"
            f"{filename}"
        )


    timestep, arr = blocks[-1]


    if arr.shape[1] < 3:

        raise RuntimeError(
            f"RDF block has fewer than 3 columns:\n"
            f"{filename}"
        )


    # LAMMPS RDF:
    #
    # col 0 = bin index
    # col 1 = r
    # col 2 = g(r)
    # col 3 = coordination number, when available

    rvals = arr[:, 1]
    gvals = arr[:, 2]

    coord = None

    if arr.shape[1] >= 4:
        coord = arr[:, 3]


    return timestep, rvals, gvals, coord


def extract_shell_descriptors(
    rvals,
    gvals,
    coord
):
    """
    Extract:
        g1
        g2
        g2/g1
        n1
        n2
        n2/n1

    Peak heights:
        g1 = max g(r) in first window
        g2 = max g(r) in second window

    Shell populations:
        n1 = coordination number accumulated across first window
        n2 = coordination-number increment across second window

    If coordination data are unavailable, n1 and n2 are returned as NaN.
    """

    mask1 = (
        (rvals >= R1_MIN)
        &
        (rvals <= R1_MAX)
    )

    mask2 = (
        (rvals >= R2_MIN)
        &
        (rvals <= R2_MAX)
    )


    if np.sum(mask1) < 2:
        raise RuntimeError(
            "Too few RDF points in first radial window."
        )

    if np.sum(mask2) < 2:
        raise RuntimeError(
            "Too few RDF points in second radial window."
        )


    g1 = np.nanmax(
        gvals[mask1]
    )

    g2 = np.nanmax(
        gvals[mask2]
    )


    if (
        not np.isfinite(g1)
        or
        g1 <= 0.0
    ):
        ratio_g = np.nan

    else:
        ratio_g = g2 / g1


    n1 = np.nan
    n2 = np.nan
    ratio_n = np.nan


    if coord is not None:

        # -------------------------------------------------------------
        # Coordination number evaluated at shell boundaries.
        #
        # np.interp is used only to evaluate the cumulative
        # coordination curve at the predefined boundaries.
        # -------------------------------------------------------------

        c_r1_min = np.interp(
            R1_MIN,
            rvals,
            coord
        )

        c_r1_max = np.interp(
            R1_MAX,
            rvals,
            coord
        )

        c_r2_min = np.interp(
            R2_MIN,
            rvals,
            coord
        )

        c_r2_max = np.interp(
            R2_MAX,
            rvals,
            coord
        )


        n1 = (
            c_r1_max
            -
            c_r1_min
        )

        n2 = (
            c_r2_max
            -
            c_r2_min
        )


        if (
            np.isfinite(n1)
            and
            n1 > 0.0
        ):
            ratio_n = n2 / n1


    return {
        "g1": g1,
        "g2": g2,
        "g2_g1": ratio_g,
        "n1": n1,
        "n2": n2,
        "n2_n1": ratio_n,
    }


def extract_shell_table():
    """
    Extract shell-resolved descriptors from all RDF files.
    """

    rdf_files = sorted(
        SIM_ROOT.glob(
            "P_*/rdfs/*.rdf"
        )
    )


    if len(rdf_files) == 0:

        raise FileNotFoundError(
            f"No RDF files found under:\n"
            f"{SIM_ROOT.resolve()}"
        )


    rows = []

    bad_name = 0
    bad_rdf = 0


    print()
    print("=" * 92)
    print("EXTRACTING SHELL-RESOLVED DESCRIPTORS")
    print("=" * 92)

    print(
        f"RDF files found = {len(rdf_files)}"
    )


    for count, filename in enumerate(
        rdf_files,
        start=1
    ):

        state = parse_PT_from_rdf_filename(
            filename
        )

        if state is None:

            bad_name += 1
            continue


        pressure, temperature = state


        if temperature < T_MIN - PT_TOL:
            continue

        if temperature > T_MAX + PT_TOL:
            continue

        if pressure < P_MIN - PT_TOL:
            continue

        if pressure > P_MAX + PT_TOL:
            continue


        try:

            timestep, rvals, gvals, coord = (
                read_last_rdf_block(
                    filename
                )
            )

            desc = extract_shell_descriptors(
                rvals,
                gvals,
                coord
            )

        except Exception as exc:

            bad_rdf += 1

            print(
                f"WARNING: {filename}\n"
                f"  {exc}"
            )

            continue


        rows.append({
            "P": pressure,
            "T": temperature,
            "timestep": timestep,
            "g1": desc["g1"],
            "g2": desc["g2"],
            "g2_g1": desc["g2_g1"],
            "n1": desc["n1"],
            "n2": desc["n2"],
            "n2_n1": desc["n2_n1"],
        })


        if count % 250 == 0:

            print(
                f"processed "
                f"{count}/{len(rdf_files)} "
                f"RDF files..."
            )


    shell = pd.DataFrame(
        rows
    )


    if len(shell) == 0:

        raise RuntimeError(
            "No valid shell-resolved states extracted."
        )


    # If duplicate files/states exist, average them.
    shell = (
        shell
        .groupby(
            ["P", "T"],
            as_index=False
        )
        .agg({
            "timestep": "max",
            "g1": "mean",
            "g2": "mean",
            "g2_g1": "mean",
            "n1": "mean",
            "n2": "mean",
            "n2_n1": "mean",
        })
    )


    shell = shell.sort_values(
        ["P", "T"]
    ).reset_index(
        drop=True
    )


    shell.to_csv(
        OUT_SHELL,
        sep=" ",
        index=False,
        float_format="%.10e"
    )


    print()
    print(
        f"Valid unique states = {len(shell)}"
    )

    print(
        f"Bad filenames       = {bad_name}"
    )

    print(
        f"Unreadable RDFs     = {bad_rdf}"
    )

    print()
    print(
        f"Shell table:\n"
        f"  {OUT_SHELL.resolve()}"
    )


    return shell


# =============================================================================
# 6. LOAD / EXTRACT SHELL DESCRIPTORS
# =============================================================================

if SHELL_TABLE.exists():

    print()
    print("=" * 92)
    print("READING EXISTING SHELL TABLE")
    print("=" * 92)

    print(
        SHELL_TABLE.resolve()
    )

    shell = pd.read_csv(
        SHELL_TABLE,
        sep=r"\s+",
        comment="#"
    )


    required_shell = {
        "P",
        "T",
        "g2_g1",
        "n2_n1",
    }


    if not required_shell.issubset(
        shell.columns
    ):

        print()
        print(
            "Existing shell table does not contain "
            "the required columns."
        )

        print(
            "Re-extracting directly from RDF files."
        )

        shell = extract_shell_table()

else:

    shell = extract_shell_table()


# =============================================================================
# 7. READ ORIGINAL A_SALR TABLE
# =============================================================================

if not ASALR_FILE.exists():

    raise FileNotFoundError(
        f"A_SALR file not found:\n"
        f"{ASALR_FILE.resolve()}"
    )


asalr = pd.read_csv(
    ASALR_FILE,
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "P",
        "T",
        "A_SALR",
        "D",
        "lnD",
    ]
)


for col in asalr.columns:

    asalr[col] = pd.to_numeric(
        asalr[col],
        errors="coerce"
    )


asalr = asalr.replace(
    [np.inf, -np.inf],
    np.nan
)


asalr = asalr.dropna(
    subset=[
        "P",
        "T",
        "A_SALR",
        "D",
        "lnD",
    ]
).copy()


asalr = asalr[
    (asalr["T"] >= T_MIN - PT_TOL)
    &
    (asalr["T"] <= T_MAX + PT_TOL)
    &
    (asalr["P"] >= P_MIN - PT_TOL)
    &
    (asalr["P"] <= P_MAX + PT_TOL)
].copy()


# =============================================================================
# 8. ROUND STATE VARIABLES FOR SAFE MERGE
# =============================================================================

def add_merge_keys(df):

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


shell = add_merge_keys(
    shell
)

asalr = add_merge_keys(
    asalr
)


# =============================================================================
# 9. MERGE SHELL DESCRIPTORS WITH ORIGINAL D AND A_SALR
# =============================================================================

merged = pd.merge(
    shell,
    asalr[
        [
            "P_key",
            "T_key",
            "A_SALR",
            "D",
            "lnD",
        ]
    ],
    on=[
        "P_key",
        "T_key"
    ],
    how="inner"
)


merged["P"] = merged["P_key"]
merged["T"] = merged["T_key"]


merged = merged.replace(
    [np.inf, -np.inf],
    np.nan
)


merged = merged.dropna(
    subset=[
        "P",
        "T",
        "lnD",
        "g2_g1",
        "n2_n1",
        "A_SALR",
    ]
).copy()


merged = merged.sort_values(
    ["T", "P"]
).reset_index(
    drop=True
)


merged.to_csv(
    OUT_DATA,
    sep=" ",
    index=False,
    float_format="%.10e"
)


# =============================================================================
# 10. INPUT REPORT
# =============================================================================

print()
print("=" * 92)
print("STRUCTURE--DYNAMICS HIERARCHY DATA")
print("=" * 92)

print(
    f"Shell states available = {len(shell)}"
)

print(
    f"A_SALR/D states        = {len(asalr)}"
)

print(
    f"Common states          = {len(merged)}"
)

print()

print(
    f"P range = "
    f"{merged['P'].min():.3f} -- "
    f"{merged['P'].max():.3f}"
)

print(
    f"T range = "
    f"{merged['T'].min():.3f} -- "
    f"{merged['T'].max():.3f}"
)


# =============================================================================
# 11. STATISTICAL ANALYSIS
# =============================================================================

def analyze_descriptor(
    dataframe,
    column,
    label
):

    data = dataframe[
        [
            column,
            "lnD"
        ]
    ].dropna().copy()


    x = data[
        column
    ].to_numpy(
        dtype=float
    )

    y = data[
        "lnD"
    ].to_numpy(
        dtype=float
    )


    if len(x) < 5:

        raise RuntimeError(
            f"Too few states for {label}"
        )


    pearson = pearsonr(
        x,
        y
    )[0]


    spearman = spearmanr(
        x,
        y
    )[0]


    # -----------------------------------------------------------------
    # Linear regression
    # -----------------------------------------------------------------

    coef_lin = np.polyfit(
        x,
        y,
        1
    )

    pred_lin = np.polyval(
        coef_lin,
        x
    )


    ss_res_lin = np.sum(
        (y - pred_lin) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )


    r2_lin = (
        1.0
        -
        ss_res_lin / ss_tot
    )


    # -----------------------------------------------------------------
    # Quadratic regression
    # -----------------------------------------------------------------

    coef_quad = np.polyfit(
        x,
        y,
        2
    )

    pred_quad = np.polyval(
        coef_quad,
        x
    )


    ss_res_quad = np.sum(
        (y - pred_quad) ** 2
    )


    r2_quad = (
        1.0
        -
        ss_res_quad / ss_tot
    )


    return {
        "label": label,
        "column": column,
        "N": len(x),
        "pearson": pearson,
        "spearman": spearman,
        "r2_lin": r2_lin,
        "r2_quad": r2_quad,
        "coef_lin": coef_lin,
        "coef_quad": coef_quad,
    }


stats_g = analyze_descriptor(
    merged,
    "g2_g1",
    "g2/g1"
)

stats_n = analyze_descriptor(
    merged,
    "n2_n1",
    "n2/n1"
)

stats_a = analyze_descriptor(
    merged,
    "A_SALR",
    "A_SALR"
)


stats_all = [
    stats_g,
    stats_n,
    stats_a,
]


print()
print("=" * 92)
print("GLOBAL STRUCTURE--DYNAMICS STATISTICS")
print("=" * 92)


for result in stats_all:

    print()
    print(
        f"Descriptor: {result['label']}"
    )

    print(
        f"N = {result['N']}"
    )

    print(
        f"Pearson      = "
        f"{result['pearson']: .6f}"
    )

    print(
        f"Spearman     = "
        f"{result['spearman']: .6f}"
    )

    print(
        f"R2 linear    = "
        f"{result['r2_lin']: .6f}"
    )

    print(
        f"R2 quadratic = "
        f"{result['r2_quad']: .6f}"
    )


with open(
    OUT_STATS,
    "w"
) as handle:

    handle.write(
        "# descriptor N Pearson Spearman R2_linear R2_quadratic\n"
    )

    for result in stats_all:

        handle.write(
            f"{result['label']} "
            f"{result['N']} "
            f"{result['pearson']:.10e} "
            f"{result['spearman']:.10e} "
            f"{result['r2_lin']:.10e} "
            f"{result['r2_quad']:.10e}\n"
        )


# =============================================================================
# 12. FIGURE SETTINGS
# =============================================================================

temperature_norm = Normalize(
    vmin=T_MIN,
    vmax=T_MAX
)

cmap = plt.get_cmap(
    CMAP
)


# =============================================================================
# 13. CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=FIGSIZE,
    sharey=True
)


panel_info = [
    (
        axes[0],
        "g2_g1",
        r"$g_2/g_1$",
        stats_g,
        r"\textbf{(a)}"
    ),

    (
        axes[1],
        "n2_n1",
        r"$n_2/n_1$",
        stats_n,
        r"\textbf{(b)}"
    ),

    (
        axes[2],
        "A_SALR",
        r"$A_{\mathrm{SALR}}$",
        stats_a,
        r"\textbf{(c)}"
    ),
]


# =============================================================================
# 14. DRAW PANELS
# =============================================================================

for ax, column, xlabel, result, panel_label in panel_info:

    data = merged[
        [
            column,
            "lnD",
            "T"
        ]
    ].dropna().copy()


    x = data[
        column
    ].to_numpy(
        dtype=float
    )

    y = data[
        "lnD"
    ].to_numpy(
        dtype=float
    )

    temp = data[
        "T"
    ].to_numpy(
        dtype=float
    )


    # -----------------------------------------------------------------
    # Scatter
    # -----------------------------------------------------------------

    ax.scatter(
        x,
        y,
        c=temp,
        cmap=cmap,
        norm=temperature_norm,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="black",
        linewidths=POINT_EDGEWIDTH,
        rasterized=True,
        zorder=2,
    )


    # -----------------------------------------------------------------
    # Quadratic regression
    # -----------------------------------------------------------------

    xmin = np.nanmin(
        x
    )

    xmax = np.nanmax(
        x
    )


    xfit = np.linspace(
        xmin,
        xmax,
        500
    )


    yfit = np.polyval(
        result[
            "coef_quad"
        ],
        xfit
    )


    ax.plot(
        xfit,
        yfit,
        color="black",
        linestyle="--",
        linewidth=FIT_LW,
        zorder=4,
    )


    # -----------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------

    ax.set_xlabel(
        xlabel
    )


    ax.text(
        0.035,
        0.965,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_FS,
        zorder=10,
    )


    # -----------------------------------------------------------------
    # Statistical annotation
    # -----------------------------------------------------------------

    stats_text = (
        rf"$r_{{\rm P}}={result['pearson']:.2f}$"
        "\n"
        rf"$r_{{\rm S}}={result['spearman']:.2f}$"
        "\n"
        rf"$R^2_{{\rm quad}}={result['r2_quad']:.2f}$"
    )


    ax.text(
        0.965,
        0.045,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=STATS_FS,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="0.65",
            alpha=0.90,
            linewidth=0.7,
        ),
        zorder=10,
    )


    # -----------------------------------------------------------------
    # Cosmetics
    # -----------------------------------------------------------------

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1.0,
    )


    ax.grid(
        True,
        linestyle=":",
        linewidth=GRID_LW,
        alpha=GRID_ALPHA,
        zorder=0,
    )


axes[0].set_ylabel(
    r"$\ln D^*$"
)


# =============================================================================
# 15. LAYOUT
# =============================================================================

FIG_LEFT = 0.075
FIG_RIGHT = 0.895
FIG_BOTTOM = 0.155
FIG_TOP = 0.970


fig.subplots_adjust(
    left=FIG_LEFT,
    right=FIG_RIGHT,
    bottom=FIG_BOTTOM,
    top=FIG_TOP,
    wspace=0.13,
)


# =============================================================================
# 16. EXTERNAL TEMPERATURE COLORBAR
# =============================================================================

sm = ScalarMappable(
    norm=temperature_norm,
    cmap=cmap
)

sm.set_array([])


CBAR_LEFT = 0.915
CBAR_WIDTH = 0.026


cax = fig.add_axes([
    CBAR_LEFT,
    FIG_BOTTOM,
    CBAR_WIDTH,
    FIG_TOP - FIG_BOTTOM,
])


cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation="vertical"
)


cbar.set_label(
    r"$T^*$",
    fontsize=CBAR_LABEL_FS,
    labelpad=10,
)


cbar.set_ticks(
    [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    ]
)


cbar.ax.tick_params(
    direction="in",
    length=4.5,
    width=0.9,
    labelsize=CBAR_TICK_FS,
    pad=5,
)


cbar.outline.set_linewidth(
    1.0
)


# =============================================================================
# 17. SAVE
# =============================================================================

fig.savefig(
    OUT_PDF,
    dpi=600
)

fig.savefig(
    OUT_PNG,
    dpi=600
)

plt.show()


# =============================================================================
# 18. FINAL REPORT
# =============================================================================

print()
print("=" * 92)
print("STRUCTURE--DYNAMICS HIERARCHY FIGURE")
print("=" * 92)

print(
    f"Common states = {len(merged)}"
)

print(
    f"T domain      = "
    f"{T_MIN:.2f} -- {T_MAX:.2f}"
)

print(
    f"P domain      = "
    f"{P_MIN:.2f} -- {P_MAX:.2f}"
)

print()

print(
    f"PDF:\n"
    f"  {OUT_PDF.resolve()}"
)

print()

print(
    f"PNG:\n"
    f"  {OUT_PNG.resolve()}"
)

print()

print(
    f"Data:\n"
    f"  {OUT_DATA.resolve()}"
)

print()

print(
    f"Statistics:\n"
    f"  {OUT_STATS.resolve()}"
)

print()

print(
    "✓ Same (P,T) states used in all three panels."
)

print(
    "✓ Common thermal domain T* >= 0.20."
)

print(
    "✓ Original A_SALR values used."
)

print(
    "✓ Shell descriptors extracted from RDFs when needed."
)

print(
    "✓ Same T* color normalization used in all panels."
)

print(
    "✓ Quadratic global regressions shown as dashed curves."
)

print(
    "✓ Pearson, Spearman, and quadratic R2 reported."
)

print(
    "✓ No smoothing applied to simulation data."
)

print("=" * 92)
