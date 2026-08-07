#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig12_derivative_response.py

Publication-quality Figure 12 for the USALR manuscript.

Run from:
    USALR/python-inputs
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

from pathlib import Path
from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import DERIVED_DATA_ROOT, FIGURE_OUTPUT_ROOT



# =============================================================================
# PATHS
# =============================================================================

DERIV_FILE = (
    DERIVED_DATA_ROOT
    / "derivative_structure_dynamics"
    / "derivative_master.dat"
)

BOUNDARY_FILE = (
    DERIVED_DATA_ROOT
    / "derivative_boundary_alignment"
    / "dynamic_boundaries.dat"
)

OUTDIR = (
    FIGURE_OUTPUT_ROOT
    / "fig12_derivative_response"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_PDF = OUTDIR / "fig12_derivative_response.pdf"
OUT_PNG = OUTDIR / "fig12_derivative_response.png"


# =============================================================================
# SELECTED ISOTHERMS
# =============================================================================

T_SELECTED = [
    0.26,
    0.30,
    0.40,
    0.50,
    0.60,
]


# =============================================================================
# PUBLICATION PALETTE
# =============================================================================

# Paul-Tol-inspired scientific palette:
# high contrast, colorblind-friendly and suitable for print.

COLORS = {

    0.26: "#4477AA",   # blue
    0.30: "#228833",   # green
    0.40: "#CCBB44",   # ochre
    0.50: "#EE6677",   # coral
    0.60: "#AA3377",   # purple

}


# =============================================================================
# FIGURE SETTINGS
# =============================================================================

FIGSIZE = (
    7.25,
    5.70
)

# Curves
LINEWIDTH = 1.65

MARKER_SIZE = 4.0
MARKER_EDGEWIDTH = 0.45

# Dynamic boundary
BOUNDARY_MARKER_SIZE = 8.5
BOUNDARY_MARKER_EDGEWIDTH = 1.55

# Reference y=0
ZERO_LW = 0.90
ZERO_ALPHA = 0.58

# Main-text pressure range
PLOT_P_MIN = 0.0
PLOT_P_MAX = 3.0


# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 10.5,

    "axes.labelsize": 14.0,

    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    "legend.fontsize": 9.8,

    "axes.linewidth": 1.05,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 4.8,
    "ytick.major.size": 4.8,

    "xtick.major.width": 0.90,
    "ytick.major.width": 0.90,

    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,

    "xtick.minor.width": 0.65,
    "ytick.minor.width": 0.65,

    "legend.frameon": False,

    "savefig.transparent": False,
})


# =============================================================================
# CHECK FILES
# =============================================================================

if not DERIV_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{DERIV_FILE.resolve()}\n"
    )


if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{BOUNDARY_FILE.resolve()}\n"
    )


# =============================================================================
# READ DERIVATIVES
# =============================================================================

df = pd.read_csv(
    DERIV_FILE,
    sep=r"\s+",
    engine="python"
)


required = [

    "T",
    "P",

    "dlnD_dP",
    "dRg_dP",
    "dRn_dP",
    "dA_SALR_dP",

]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise RuntimeError(
        "\nMissing columns:\n"
        +
        "\n".join(missing)
    )


for c in required:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


df = df.sort_values(
    ["T", "P"]
).reset_index(
    drop=True
)


# =============================================================================
# READ BOUNDARIES
# =============================================================================

boundaries = pd.read_csv(
    BOUNDARY_FILE,
    sep=r"\s+",
    engine="python"
)


for c in [
    "T",
    "P_boundary"
]:

    boundaries[c] = pd.to_numeric(
        boundaries[c],
        errors="coerce"
    )


boundaries = boundaries.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(
    subset=[
        "T",
        "P_boundary"
    ]
)


# =============================================================================
# INTERPOLATION AT DYNAMIC BOUNDARY
# =============================================================================

def interpolate_at_boundary(
    sub,
    column,
    P0
):

    x = pd.to_numeric(
        sub["P"],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        sub[column],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    x = x[mask]
    y = y[mask]


    if len(x) < 4:

        return np.nan


    order = np.argsort(
        x
    )

    x = x[order]
    y = y[order]


    xu, index = np.unique(
        x,
        return_index=True
    )

    yu = y[index]


    if len(xu) < 4:

        return np.nan


    if not (
        xu.min()
        <=
        P0
        <=
        xu.max()
    ):

        return np.nan


    try:

        interp = PchipInterpolator(
            xu,
            yu,
            extrapolate=False
        )

        return float(
            interp(P0)
        )

    except Exception:

        return np.nan


# =============================================================================
# DIAGNOSTICS
# =============================================================================

print()
print("=" * 82)
print("FIGURE 12")
print("=" * 82)

print(
    f"Derivative states  = {len(df)}"
)

print(
    f"Dynamic boundaries = {len(boundaries)}"
)

print()

for T in T_SELECTED:

    sub = df[
        np.isclose(
            df["T"],
            T
        )
    ]


    bsub = boundaries[
        np.isclose(
            boundaries["T"],
            T
        )
    ]


    if len(bsub) > 0:

        boundary_string = ", ".join(
            f"{x:.4f}"
            for x in bsub[
                "P_boundary"
            ]
        )

    else:

        boundary_string = "none"


    print(
        f"T*={T:.2f}: "
        f"N={len(sub):3d}, "
        f"P_boundary={boundary_string}"
    )


# =============================================================================
# CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=FIGSIZE,

    sharex=True

)


ax_a = axes[0, 0]
ax_b = axes[0, 1]
ax_c = axes[1, 0]
ax_d = axes[1, 1]


# =============================================================================
# PANEL SPECS
# =============================================================================

panel_specs = [(ax_a, "dlnD_dP", r"$"
        r"\left("r"\partial\ln D^*/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",),

    (ax_b,

        "dRg_dP",

        r"$"
        r"\left("
        r"\partial R_g/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",),

    (
        ax_c,

        "dRn_dP",

        r"$"
        r"\left("
        r"\partial R_n/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",),
    (
        ax_d,

        "dA_SALR_dP",

        r"$"
        r"\left("
        r"\partial A_{\rm SALR}/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",),]


# =============================================================================
# CURVES
# =============================================================================

for ax, column, ylabel in panel_specs:

    for T in T_SELECTED:

        sub = df[
            np.isclose(
                df["T"],
                T
            )
        ].copy()


        sub = sub.sort_values(
            "P"
        )


        sub = sub[
            (
                sub["P"]
                >=
                PLOT_P_MIN
            )
            &
            (
                sub["P"]
                <=
                PLOT_P_MAX
            )
        ]


        mask = (

            np.isfinite(
                sub["P"]
            )

            &

            np.isfinite(
                sub[column]
            )

        )


        sub = sub.loc[
            mask
        ]


        if len(sub) == 0:

            continue


        color = COLORS[T]


        ax.plot(

            sub["P"],

            sub[column],

            color=color,

            marker="o",

            markersize=MARKER_SIZE,

            markerfacecolor=color,

            markeredgecolor=color,

            markeredgewidth=MARKER_EDGEWIDTH,

            linewidth=LINEWIDTH,

            label=(
                rf"$T^*={T:.2f}$"
            ),

            zorder=3

        )


    ax.set_ylabel(
        ylabel
    )


# =============================================================================
# ZERO REFERENCES
# =============================================================================

# Only where the sign relative to zero is physically central.

for ax in [
    ax_a,
    ax_d
]:

    ax.axhline(

        0.0,

        color="0.35",

        linestyle="--",

        linewidth=ZERO_LW,

        alpha=ZERO_ALPHA,

        zorder=1

    )


# =============================================================================
# DYNAMIC BOUNDARY MARKERS
# =============================================================================

structural_panels = [

    (
        ax_b,
        "dRg_dP"
    ),

    (
        ax_c,
        "dRn_dP"
    ),

    (
        ax_d,
        "dA_SALR_dP"
    ),

]


for T in T_SELECTED:

    color = COLORS[T]


    bsub = boundaries[
        np.isclose(
            boundaries["T"],
            T
        )
    ]


    if len(bsub) == 0:

        continue


    full_sub = df[
        np.isclose(
            df["T"],
            T
        )
    ].sort_values(
        "P"
    )


    for P0 in bsub[
        "P_boundary"
    ]:

        if not (
            PLOT_P_MIN
            <=
            P0
            <=
            PLOT_P_MAX
        ):

            continue


        # panel (a)

        ax_a.plot(

            P0,
            0.0,

            marker="o",

            markersize=BOUNDARY_MARKER_SIZE,

            markerfacecolor="white",

            markeredgecolor=color,

            markeredgewidth=BOUNDARY_MARKER_EDGEWIDTH,

            linestyle="None",

            zorder=15

        )


        # structural panels

        for ax, column in structural_panels:

            y0 = interpolate_at_boundary(

                full_sub,
                column,
                P0

            )


            if not np.isfinite(
                y0
            ):

                continue


            ax.plot(

                P0,
                y0,

                marker="o",

                markersize=BOUNDARY_MARKER_SIZE,

                markerfacecolor="white",

                markeredgecolor=color,

                markeredgewidth=BOUNDARY_MARKER_EDGEWIDTH,

                linestyle="None",

                zorder=15

            )


# =============================================================================
# AXES
# =============================================================================

for ax in axes.ravel():

    ax.set_xlim(
        PLOT_P_MIN,
        PLOT_P_MAX
    )

    ax.margins(
        x=0.01
    )


ax_c.set_xlabel(
    r"$P^*$"
)

ax_d.set_xlabel(
    r"$P^*$"
)


xticks = np.arange(
    0.0,
    3.01,
    0.5
)


ax_c.set_xticks(
    xticks
)

ax_d.set_xticks(
    xticks
)


# =============================================================================
# COMMON TEMPERATURE LEGEND
# =============================================================================

# Line-only legend:
# the boundary symbols remain visually unique inside the panels.

handles = []
labels = []


for T in T_SELECTED:

    color = COLORS[T]


    handle, = ax_a.plot(

        [], [],

        color=color,

        linewidth=2.5

    )


    handles.append(
        handle
    )


    labels.append(
        rf"$T^*={T:.2f}$"
    )


fig.legend(

    handles,
    labels,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.995
    ),

    ncol=5,

    frameon=False,

    columnspacing=1.55,

    handlelength=2.25,

    handletextpad=0.50

)


# =============================================================================
# BOUNDARY KEY
# =============================================================================

boundary_handle, = ax_a.plot(

    [], [],

    marker="o",

    markersize=7.0,

    markerfacecolor="white",

    markeredgecolor="0.25",

    markeredgewidth=1.35,

    linestyle="None"

)


ax_a.legend(

    [boundary_handle],

    [
        r"$"
        r"\partial_{P^*}\ln D^*=0"
        r"$"
    ],

    loc="lower right",

    frameon=False,

    fontsize=9.3,

    handletextpad=0.35,

    borderaxespad=0.50

)


# =============================================================================
# LAYOUT
# =============================================================================

fig.subplots_adjust(

    left=0.105,

    right=0.985,

    bottom=0.10,

    top=0.90,

    wspace=0.28,

    hspace=0.14

)


# =============================================================================
# SAVE
# =============================================================================

fig.savefig(

    OUT_PDF,

    bbox_inches="tight"

)


fig.savefig(

    OUT_PNG,

    dpi=500,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 82)
print("OUTPUT")
print("=" * 82)

print(
    OUT_PDF
)

print(
    OUT_PNG
)

print()
print(
    "Palette:"
)

for T in T_SELECTED:

    print(
        f"T*={T:.2f}  {COLORS[T]}"
    )

print()
print("Done.")
