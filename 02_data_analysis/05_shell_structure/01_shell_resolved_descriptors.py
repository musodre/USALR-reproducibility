#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shell-resolved structural reorganization
========================================

This script performs TWO tasks:

1. GLOBAL EXTRACTION
   -----------------
   It scans all RDF files in

       ../allpress/P_*/rdfs/*.rdf

   finds the corresponding thermodynamic file in

       ../allpress/P_*/thermo/

   and calculates, for every (P*,T*) state,

       g1
       g2
       g2/g1

       n1
       n2
       n2/n1

   using the operational radial windows

       W1 = [1.10, 1.35]
       W2 = [1.85, 2.15]

   with

       n_i = 4*pi*rho* integral_Wi r*^2 g(r*) dr*

   The complete table is saved as

       shell_resolved_structure/
           shell_descriptors_global.dat


2. STRUCTURAL-REORGANIZATION FIGURE
   --------------------------------
   Four panels:

       (a) g1(P*)
       (b) g2(P*)
       (c) Rg = g2/g1
       (d) Rn = n2/n1

   along representative isotherms.

IMPORTANT
---------
W1 and W2 are operational characteristic radial windows.
They should not automatically be interpreted as complete coordination
shells bounded by successive minima of g(r).

The connecting lines are visual guides joining actual simulated states.
No smoothing, spline, polynomial fit, or extrapolation is used.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import trapezoid
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
# 1. USER SETTINGS
# =============================================================================

ROOT = RAW_DATA_ROOT / "allpress"


# -----------------------------------------------------------------------------
# Rebuild complete descriptor table?
#
# FIRST RUN:
#     True
#
# Subsequent runs, if RDF files have not changed:
#     False
# -----------------------------------------------------------------------------

REBUILD_GLOBAL_TABLE = True


# -----------------------------------------------------------------------------
# Operational radial windows
# -----------------------------------------------------------------------------

R1_MIN = 1.10
R1_MAX = 1.35

R2_MIN = 1.85
R2_MAX = 2.15


# -----------------------------------------------------------------------------
# Representative isotherms shown in the figure
# -----------------------------------------------------------------------------

T_SELECTED = np.array(
    [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    ],
    dtype=float
)


# -----------------------------------------------------------------------------
# Thermodynamic domain shown
# -----------------------------------------------------------------------------

P_MIN = 0.10
P_MAX = 6.00


# -----------------------------------------------------------------------------
# Matching tolerance
# -----------------------------------------------------------------------------

T_TOL = 1.0e-6


# =============================================================================
# 2. OUTPUT
# =============================================================================

OUTDIR = DERIVED_DATA_ROOT / "shell_resolved_structure"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


GLOBAL_TABLE = (
    OUTDIR
    /
    "shell_descriptors_global.dat"
)


OUT_PDF = (
    OUTDIR
    /
    "shell_reorganization.pdf"
)


OUT_PNG = (
    OUTDIR
    /
    "shell_reorganization.png"
)


SELECTED_TABLE = (
    OUTDIR
    /
    "shell_reorganization_selected.dat"
)


# =============================================================================
# 3. GRAPHICAL SETTINGS
# =============================================================================

FIGSIZE = (
    10.4,
    8.0
)


LINEWIDTH = 1.40

MARKER_SIZE = 34

MARKER_EDGEWIDTH = 0.50


LABEL_FS = 18

TICK_FS = 11

PANEL_FS = 15

LEGEND_FS = 10.5

CBAR_FS = 17


CMAP = "cividis"


T_COLOR_MIN = float(
    T_SELECTED.min()
)

T_COLOR_MAX = float(
    T_SELECTED.max()
)


# -----------------------------------------------------------------------------
# Legend
#
# Only one legend is drawn, because the same temperature/color mapping is
# used in all four panels.
# -----------------------------------------------------------------------------

SHOW_LEGEND = True


# =============================================================================
# 4. MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

    "text.usetex": True,

    "font.family": "serif",

    "font.size": 12,

    "axes.labelsize": LABEL_FS,

    "xtick.labelsize": TICK_FS,

    "ytick.labelsize": TICK_FS,

    "axes.linewidth": 1.10,

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
# 5. RDF FILENAME PARSER
# =============================================================================

RDF_PATTERN = re.compile(

    r"P_(?P<P>[0-9]+(?:\.[0-9]+)?)"
    r"_T_(?P<T>[0-9]+(?:\.[0-9]+)?)"
    r"\.rdf$"
)


def parse_state_from_rdf_filename(filename):

    match = RDF_PATTERN.search(
        filename.name
    )


    if match is None:

        return None


    try:

        P = float(
            match.group("P")
        )

        T = float(
            match.group("T")
        )

    except ValueError:

        return None


    return (
        P,
        T
    )


# =============================================================================
# 6. READ THERMODYNAMIC DENSITY
# =============================================================================

def read_density(filename):

    """
    Thermodynamic file expected columns:

        timestep  T  P  V  rho  U  K  H

    rho is column index 4.

    The mean of all valid recorded density values is used, consistent with
    the representative-state analysis.
    """

    rho_values = []


    with open(
        filename,
        "r"
    ) as f:

        for line in f:

            line = line.strip()


            if not line:

                continue


            if line.startswith("#"):

                continue


            parts = line.split()


            if len(parts) < 5:

                continue


            try:

                rho = float(
                    parts[4]
                )

            except ValueError:

                continue


            if np.isfinite(
                rho
            ):

                rho_values.append(
                    rho
                )


    if len(
        rho_values
    ) == 0:

        return np.nan


    return float(

        np.mean(
            rho_values
        )
    )


# =============================================================================
# 7. READ FINAL COMPLETE RDF BLOCK
# =============================================================================

def read_lammps_rdf_last_block(filename):

    """
    Read the LAST complete RDF block from a LAMMPS fix ave/time output.

    Expected block form:

        timestep number_of_bins
        bin r g(r) coordination
        ...

    Only:

        column 1 -> r
        column 2 -> g(r)

    are used.
    """

    with open(
        filename,
        "r"
    ) as f:

        lines = f.readlines()


    blocks = []

    i = 0


    while i < len(lines):

        line = lines[
            i
        ].strip()


        if (
            not line
            or
            line.startswith("#")
        ):

            i += 1

            continue


        parts = line.split()


        if len(
            parts
        ) != 2:

            i += 1

            continue


        try:

            timestep = int(
                float(
                    parts[0]
                )
            )

            nrows = int(
                float(
                    parts[1]
                )
            )

        except ValueError:

            i += 1

            continue


        if nrows <= 0:

            i += 1

            continue


        data = []


        start = (
            i + 1
        )


        stop = min(

            start + nrows,

            len(lines)
        )


        for j in range(
            start,
            stop
        ):

            row = lines[
                j
            ].strip()


            if (
                not row
                or
                row.startswith("#")
            ):

                continue


            values = row.split()


            if len(
                values
            ) < 3:

                continue


            try:

                numeric = [

                    float(x)

                    for x in values
                ]

            except ValueError:

                continue


            data.append(
                numeric
            )


        if len(
            data
        ) == nrows:

            blocks.append(

                (

                    timestep,

                    np.asarray(
                        data,
                        dtype=float
                    )
                )
            )


        i += (
            nrows + 1
        )


    if len(
        blocks
    ) == 0:

        return (
            np.nan,
            None,
            None
        )


    timestep, array = blocks[-1]


    if array.shape[1] < 3:

        return (
            timestep,
            None,
            None
        )


    r = array[
        :,
        1
    ]


    g = array[
        :,
        2
    ]


    good = (

        np.isfinite(r)

        &

        np.isfinite(g)
    )


    r = r[
        good
    ]


    g = g[
        good
    ]


    if len(
        r
    ) < 3:

        return (
            timestep,
            None,
            None
        )


    order = np.argsort(
        r
    )


    r = r[
        order
    ]


    g = g[
        order
    ]


    return (
        timestep,
        r,
        g
    )


# =============================================================================
# 8. LOCATE CORRESPONDING THERMO FILE
# =============================================================================

def find_thermo_file(
    rdf_file,
    P,
    T
):

    """
    RDF:
        ../allpress/P_1.000/rdfs/P_1.00_T_0.20.rdf

    Thermo:
        ../allpress/P_1.000/thermo/
        outvars_P_1.00_T_0.20.profile
    """

    pressure_dir = rdf_file.parent.parent


    thermo_dir = (
        pressure_dir
        /
        "thermo"
    )


    expected = (

        thermo_dir
        /
        f"outvars_P_{P:.2f}_T_{T:.2f}.profile"
    )


    if expected.exists():

        return expected


    # ------------------------------------------------------------------
    # Fallback search if formatting differs slightly
    # ------------------------------------------------------------------

    candidates = list(

        thermo_dir.glob(
            f"outvars_P_*_T_{T:.2f}.profile"
        )
    )


    if len(
        candidates
    ) == 1:

        return candidates[0]


    # ------------------------------------------------------------------
    # More robust fallback: parse filenames numerically
    # ------------------------------------------------------------------

    pattern = re.compile(

        r"outvars_P_"
        r"(?P<P>[0-9]+(?:\.[0-9]+)?)"
        r"_T_"
        r"(?P<T>[0-9]+(?:\.[0-9]+)?)"
        r"\.profile$"
    )


    for filename in thermo_dir.glob(
        "outvars_P_*_T_*.profile"
    ):

        match = pattern.search(
            filename.name
        )


        if match is None:

            continue


        try:

            Pf = float(
                match.group("P")
            )

            Tf = float(
                match.group("T")
            )

        except ValueError:

            continue


        if (
            abs(Pf-P) < 1.0e-6
            and
            abs(Tf-T) < 1.0e-6
        ):

            return filename


    return None


# =============================================================================
# 9. CALCULATE DESCRIPTORS FOR ONE STATE
# =============================================================================

def calculate_shell_descriptors(
    r,
    g,
    rho
):

    """
    Calculate:

        r_peak1
        g1
        r_peak2
        g2
        Rg = g2/g1

        n1
        n2
        Rn = n2/n1
    """

    mask1 = (

        (r >= R1_MIN)

        &

        (r <= R1_MAX)
    )


    mask2 = (

        (r >= R2_MIN)

        &

        (r <= R2_MAX)
    )


    if np.count_nonzero(
        mask1
    ) < 3:

        return None


    if np.count_nonzero(
        mask2
    ) < 3:

        return None


    r1 = r[
        mask1
    ]


    g_region1 = g[
        mask1
    ]


    r2 = r[
        mask2
    ]


    g_region2 = g[
        mask2
    ]


    # ------------------------------------------------------------------
    # Peak heights
    # ------------------------------------------------------------------

    idx1 = np.argmax(
        g_region1
    )


    idx2 = np.argmax(
        g_region2
    )


    r_peak1 = float(
        r1[idx1]
    )


    g1 = float(
        g_region1[idx1]
    )


    r_peak2 = float(
        r2[idx2]
    )


    g2 = float(
        g_region2[idx2]
    )


    if (
        not np.isfinite(g1)
        or
        not np.isfinite(g2)
        or
        g1 <= 0
    ):

        return None


    Rg = (
        g2/g1
    )


    # ------------------------------------------------------------------
    # Coordination populations
    # ------------------------------------------------------------------

    integrand1 = (

        4.0
        *
        np.pi
        *
        rho
        *
        r1**2
        *
        g_region1
    )


    integrand2 = (

        4.0
        *
        np.pi
        *
        rho
        *
        r2**2
        *
        g_region2
    )


    n1 = float(

        trapezoid(
            integrand1,
            r1
        )
    )


    n2 = float(

        trapezoid(
            integrand2,
            r2
        )
    )


    if (
        not np.isfinite(n1)
        or
        not np.isfinite(n2)
        or
        n1 <= 0
    ):

        return None


    Rn = (
        n2/n1
    )


    return {

        "r_peak1":
            r_peak1,

        "g1":
            g1,

        "r_peak2":
            r_peak2,

        "g2":
            g2,

        "Rg":
            Rg,

        "n1":
            n1,

        "n2":
            n2,

        "Rn":
            Rn,
    }


# =============================================================================
# 10. BUILD GLOBAL TABLE
# =============================================================================

def build_global_table():

    print()

    print(
        "=" * 96
    )

    print(
        "BUILDING GLOBAL SHELL-DESCRIPTOR TABLE"
    )

    print(
        "=" * 96
    )


    rdf_files = sorted(

        ROOT.glob(
            "P_*/rdfs/*.rdf"
        )
    )


    print(
        f"RDF files found = {len(rdf_files)}"
    )


    if len(
        rdf_files
    ) == 0:

        raise RuntimeError(

            "No RDF files found under:\n"
            f"{ROOT.resolve()}/P_*/rdfs/"
        )


    rows = []


    n_bad_name = 0

    n_missing_thermo = 0

    n_bad_density = 0

    n_bad_rdf = 0

    n_bad_descriptor = 0


    for count, rdf_file in enumerate(
        rdf_files,
        start=1
    ):

        state = parse_state_from_rdf_filename(
            rdf_file
        )


        if state is None:

            n_bad_name += 1

            continue


        P, T = state


        # --------------------------------------------------------------
        # Corresponding thermo
        # --------------------------------------------------------------

        thermo_file = find_thermo_file(

            rdf_file,

            P,

            T
        )


        if thermo_file is None:

            n_missing_thermo += 1

            continue


        # --------------------------------------------------------------
        # Density
        # --------------------------------------------------------------

        rho = read_density(
            thermo_file
        )


        if not np.isfinite(
            rho
        ):

            n_bad_density += 1

            continue


        # --------------------------------------------------------------
        # RDF
        # --------------------------------------------------------------

        timestep, r, g = read_lammps_rdf_last_block(
            rdf_file
        )


        if (
            r is None
            or
            g is None
        ):

            n_bad_rdf += 1

            continue


        # --------------------------------------------------------------
        # Structural descriptors
        # --------------------------------------------------------------

        descriptors = calculate_shell_descriptors(

            r,

            g,

            rho
        )


        if descriptors is None:

            n_bad_descriptor += 1

            continue


        rows.append({

            "P":
                float(P),

            "T":
                float(T),

            "rho":
                float(rho),

            "rdf_timestep":
                float(timestep),

            "r_peak1":
                descriptors[
                    "r_peak1"
                ],

            "g1":
                descriptors[
                    "g1"
                ],

            "r_peak2":
                descriptors[
                    "r_peak2"
                ],

            "g2":
                descriptors[
                    "g2"
                ],

            "Rg":
                descriptors[
                    "Rg"
                ],

            "n1":
                descriptors[
                    "n1"
                ],

            "n2":
                descriptors[
                    "n2"
                ],

            "Rn":
                descriptors[
                    "Rn"
                ],
        })


        if (
            count % 100 == 0
            or
            count == len(rdf_files)
        ):

            print(

                f"processed "
                f"{count}/{len(rdf_files)} RDF files..."
            )


    if len(
        rows
    ) == 0:

        raise RuntimeError(

            "No valid states could be extracted."
        )


    df = pd.DataFrame(
        rows
    )


    # ------------------------------------------------------------------
    # Average duplicate (P,T) states if they exist
    # ------------------------------------------------------------------

    df = (

        df

        .groupby(
            [
                "P",
                "T"
            ],
            as_index=False
        )

        .agg({

            "rho":
                "mean",

            "rdf_timestep":
                "max",

            "r_peak1":
                "mean",

            "g1":
                "mean",

            "r_peak2":
                "mean",

            "g2":
                "mean",

            "Rg":
                "mean",

            "n1":
                "mean",

            "n2":
                "mean",

            "Rn":
                "mean",
        })
    )


    # ------------------------------------------------------------------
    # Recalculate ratios AFTER duplicate averaging
    # ------------------------------------------------------------------

    df[
        "Rg"
    ] = (

        df["g2"]

        /

        df["g1"]
    )


    df[
        "Rn"
    ] = (

        df["n2"]

        /

        df["n1"]
    )


    df = df.sort_values(

        [
            "T",
            "P"
        ]

    ).reset_index(
        drop=True
    )


    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    df.to_csv(

        GLOBAL_TABLE,

        sep=" ",

        index=False,

        float_format="%.10e"
    )


    print()

    print(
        "=" * 96
    )

    print(
        "GLOBAL EXTRACTION SUMMARY"
    )

    print(
        "=" * 96
    )


    print(
        f"RDF files found        = {len(rdf_files)}"
    )


    print(
        f"Valid unique states    = {len(df)}"
    )


    print(
        f"Bad RDF filenames      = {n_bad_name}"
    )


    print(
        f"Missing thermo files   = {n_missing_thermo}"
    )


    print(
        f"Invalid densities      = {n_bad_density}"
    )


    print(
        f"Unreadable RDFs        = {n_bad_rdf}"
    )


    print(
        f"Invalid descriptors    = {n_bad_descriptor}"
    )


    print()

    print(
        f"P range = "
        f"{df['P'].min():.3f}"
        f" -- "
        f"{df['P'].max():.3f}"
    )


    print(
        f"T range = "
        f"{df['T'].min():.3f}"
        f" -- "
        f"{df['T'].max():.3f}"
    )


    print()

    print(
        f"Global table:\n"
        f"  {GLOBAL_TABLE.resolve()}"
    )


    return df


# =============================================================================
# 11. LOAD OR REBUILD GLOBAL TABLE
# =============================================================================

if (
    REBUILD_GLOBAL_TABLE
    or
    not GLOBAL_TABLE.exists()
):

    data = build_global_table()


else:

    print()

    print(
        "=" * 96
    )

    print(
        "USING EXISTING GLOBAL SHELL-DESCRIPTOR TABLE"
    )

    print(
        "=" * 96
    )


    print(
        GLOBAL_TABLE.resolve()
    )


    data = pd.read_csv(

        GLOBAL_TABLE,

        sep=r"\s+"
    )


# =============================================================================
# 12. CHECK GLOBAL DATA
# =============================================================================

required_columns = [

    "P",
    "T",
    "rho",

    "r_peak1",
    "g1",

    "r_peak2",
    "g2",

    "Rg",

    "n1",
    "n2",

    "Rn",
]


missing_columns = [

    col

    for col in required_columns

    if col not in data.columns
]


if missing_columns:

    raise RuntimeError(

        "Missing columns in global descriptor table:\n"
        +
        "\n".join(
            missing_columns
        )
    )


for col in required_columns:

    data[col] = pd.to_numeric(

        data[col],

        errors="coerce"
    )


data = data.replace(

    [
        np.inf,
        -np.inf
    ],

    np.nan
)


data = data.dropna(

    subset=required_columns
)


# =============================================================================
# 13. PRESSURE FILTER
# =============================================================================

data = data[

    (data["P"] >= P_MIN)

    &

    (data["P"] <= P_MAX)

].copy()


# =============================================================================
# 14. SELECT REPRESENTATIVE ISOTHERMS
# =============================================================================

selected_frames = []


available_T = np.sort(

    data[
        "T"
    ].unique()
)


print()

print(
    "=" * 96
)

print(
    "SELECTED ISOTHERMS"
)

print(
    "=" * 96
)


for target_T in T_SELECTED:

    if len(
        available_T
    ) == 0:

        continue


    nearest_T = available_T[

        np.argmin(

            np.abs(
                available_T-target_T
            )
        )
    ]


    if abs(
        nearest_T-target_T
    ) > T_TOL:

        print(

            f"WARNING: T*={target_T:.2f} "
            f"not available."
        )

        continue


    subset = data[

        np.isclose(

            data["T"],

            nearest_T,

            atol=T_TOL
        )

    ].copy()


    subset = subset.sort_values(
        "P"
    )


    subset[
        "T_selected"
    ] = target_T


    selected_frames.append(
        subset
    )


    print(

        f"T*={target_T:.2f}: "
        f"{len(subset)} states"
    )


if len(
    selected_frames
) == 0:

    raise RuntimeError(

        "None of the selected isotherms were found."
    )


plotdata = pd.concat(

    selected_frames,

    ignore_index=True
)


# =============================================================================
# 15. SAVE SELECTED DATA
# =============================================================================

plotdata[

    [
        "P",
        "T",
        "rho",

        "r_peak1",
        "g1",

        "r_peak2",
        "g2",

        "Rg",

        "n1",
        "n2",

        "Rn"
    ]

].to_csv(

    SELECTED_TABLE,

    sep=" ",

    index=False,

    float_format="%.10e"
)


# =============================================================================
# 16. COLOR SCALE
# =============================================================================

norm = Normalize(

    vmin=T_COLOR_MIN,

    vmax=T_COLOR_MAX
)


cmap = plt.get_cmap(
    CMAP
)


# =============================================================================
# 17. CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=FIGSIZE,

    sharex=True
)


ax_a = axes[
    0,
    0
]


ax_b = axes[
    0,
    1
]


ax_c = axes[
    1,
    0
]


ax_d = axes[
    1,
    1
]


# =============================================================================
# 18. PLOT HELPER
# =============================================================================

def plot_descriptor(
    ax,
    ycolumn
):

    for T0 in T_SELECTED:

        subset = plotdata[

            np.isclose(

                plotdata[
                    "T_selected"
                ],

                T0,

                atol=T_TOL
            )

        ].copy()


        if len(
            subset
        ) == 0:

            continue


        subset = subset.sort_values(
            "P"
        )


        color = cmap(
            norm(T0)
        )


        # -------------------------------------------------------------
        # Straight connections between actual simulation states
        # -------------------------------------------------------------

        ax.plot(

            subset["P"],

            subset[ycolumn],

            color=color,

            linewidth=LINEWIDTH,

            alpha=0.90,

            zorder=2
        )


        # -------------------------------------------------------------
        # Actual simulation points
        # -------------------------------------------------------------

        ax.scatter(

            subset["P"],

            subset[ycolumn],

            s=MARKER_SIZE,

            facecolor=color,

            edgecolor="black",

            linewidth=MARKER_EDGEWIDTH,

            alpha=0.97,

            zorder=3
        )


# =============================================================================
# 19. DRAW PANELS
# =============================================================================

plot_descriptor(
    ax_a,
    "g1"
)


plot_descriptor(
    ax_b,
    "g2"
)


plot_descriptor(
    ax_c,
    "Rg"
)


plot_descriptor(
    ax_d,
    "Rn"
)


# =============================================================================
# 20. AXIS LABELS
# =============================================================================

ax_a.set_ylabel(
    r"$g_1$"
)


ax_b.set_ylabel(
    r"$g_2$"
)


ax_c.set_ylabel(
    r"$R_g=g_2/g_1$"
)


ax_d.set_ylabel(
    r"$R_n=n_2/n_1$"
)


ax_c.set_xlabel(
    r"$P^*$"
)


ax_d.set_xlabel(
    r"$P^*$"
)


# =============================================================================
# 21. PANEL LABELS
# =============================================================================

panel_labels = {

    ax_a:
        r"\textbf{(a)}",

    ax_b:
        r"\textbf{(b)}",

    ax_c:
        r"\textbf{(c)}",

    ax_d:
        r"\textbf{(d)}",
}


for ax, label in panel_labels.items():

    ax.text(

        0.035,

        0.955,

        label,

        transform=ax.transAxes,

        ha="left",

        va="top",

        fontsize=PANEL_FS,

        zorder=20
    )


# =============================================================================
# 22. COMMON AXIS FORMATTING
# =============================================================================

for ax in axes.flat:

    ax.set_xlim(
        P_MIN,
        P_MAX
    )


    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

        length=5,

        width=1.0
    )


    ax.grid(

        True,

        which="major",

        linestyle=":",

        linewidth=0.55,

        alpha=0.27,

        zorder=0
    )


# =============================================================================
# 23. DISCRETE TEMPERATURE LEGEND
# =============================================================================

if SHOW_LEGEND:

    handles = []


    for T0 in T_SELECTED:

        color = cmap(
            norm(T0)
        )


        handle = plt.Line2D(

            [],

            [],

            color=color,

            linewidth=LINEWIDTH,

            marker="o",

            markersize=6.2,

            markerfacecolor=color,

            markeredgecolor="black",

            markeredgewidth=0.5,

            label=rf"$T^*={T0:.2f}$"
        )


        handles.append(
            handle
        )


    ax_a.legend(

        handles=handles,

        loc="best",

        frameon=True,

        framealpha=0.94,

        facecolor="white",

        edgecolor="0.70",

        fontsize=LEGEND_FS,

        ncol=1,

        labelspacing=0.35,

        borderpad=0.45,

        handletextpad=0.55
    )


# =============================================================================
# 24. LAYOUT
# =============================================================================

fig.subplots_adjust(

    left=0.095,

    right=0.865,

    bottom=0.10,

    top=0.975,

    wspace=0.24,

    hspace=0.12
)


# =============================================================================
# 25. EXTERNAL COLORBAR
# =============================================================================

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cax = fig.add_axes([0.908, 0.18, 0.022, 0.82])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label(r"$T^*$", fontsize=CBAR_FS, labelpad=10)
cbar.set_ticks(T_SELECTED)
cbar.ax.tick_params(direction="in", length=4, width=0.9,labelsize=TICK_FS)
cbar.outline.set_linewidth(1.0)


# =============================================================================
# 26. SAVE FIGURE
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
# 27. FINAL REPORT
# =============================================================================

print()

print(
    "=" * 96
)

print(
    "SHELL-RESOLVED STRUCTURAL REORGANIZATION"
)

print(
    "=" * 96
)


print(
    f"Global states available = {len(data)}"
)


print(
    f"States plotted          = {len(plotdata)}"
)


print()


print(
    f"Global descriptor table:\n"
    f"  {GLOBAL_TABLE.resolve()}"
)


print()


print(
    f"Selected-state table:\n"
    f"  {SELECTED_TABLE.resolve()}"
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
    "Definitions:"
)

print(
    f"  W1 = [{R1_MIN:.2f}, {R1_MAX:.2f}]"
)

print(
    f"  W2 = [{R2_MIN:.2f}, {R2_MAX:.2f}]"
)

print()

print(
    "  Rg = g2/g1"
)

print(
    "  Rn = n2/n1"
)

print()

print(
    "  ni = 4*pi*rho integral_Wi r^2 g(r) dr"
)

print()

print(
    "✓ Descriptor extraction performed directly from the original RDFs."
)

print(
    "✓ Thermodynamic density read from the corresponding outvars file."
)

print(
    "✓ Final complete RDF block used for every state."
)

print(
    "✓ g1 and g2 evaluated independently inside W1 and W2."
)

print(
    "✓ n1 and n2 calculated using the proper radial population integral."
)

print(
    "✓ No old peak-ratio or population-summary file is required."
)

print(
    "✓ No smoothing or regression applied to the plotted structural curves."
)

print(
    "✓ Lines merely connect the actual simulated state points."
)

print()

print(
    "=" * 96
)
