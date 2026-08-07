#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global pair-order descriptors
=============================

Panel (a): -s2* versus P*
Panel (b): tau versus P*

Selected isotherms:
    T* = 0.10, 0.20, 0.30, 0.40, 0.50, 0.60

Input files
-----------
somepress/analysis/s2/s2_global.dat
    columns:
    P   T   rho   s2

somepress/analysis/tau/tau_global.dat
    columns:
    P   T   rho   tau

Important
---------
s2_global.dat contains conventional negative s2 values.
Therefore the plotted quantity is:

    -s2* = -s2

tau_global.dat contains duplicated states in some cases.
Duplicated (P,T) states are averaged before plotting.

PCHIP is used ONLY as a visual guide.
The circles are the actual state-point data.

Outputs
-------
plots/global_pair_order/global_pair_order_isotherms.pdf
plots/global_pair_order/global_pair_order_isotherms.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize



from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import DERIVED_DATA_ROOT, FIGURE_OUTPUT_ROOT

# ======================================================================
# 1. INPUT FILES
# ======================================================================

FILE_S2 = DERIVED_DATA_ROOT / "s2" / "s2_global.dat"

FILE_TAU = DERIVED_DATA_ROOT / "tau" / "tau_global.dat"


# ======================================================================
# 2. OUTPUT
# ======================================================================

OUTDIR = FIGURE_OUTPUT_ROOT / "fig06_global_pair_order"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_PDF = (
    OUTDIR
    /
    "global_pair_order_isotherms.pdf"
)

OUT_PNG = (
    OUTDIR
    /
    "global_pair_order_isotherms.png"
)


# ======================================================================
# 3. SELECTED ISOTHERMS
# ======================================================================

SELECTED_T = np.array(
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


# ======================================================================
# 4. PRESSURE DOMAIN
# ======================================================================

P_MIN = 0.10
P_MAX = 6.00


# ======================================================================
# 5. GRAPHICAL PARAMETERS
# ======================================================================

FIGSIZE = (
    10.3,
    5.2
)

MARKER_SIZE = 6.3

MARKER_EDGE_WIDTH = 0.50

MARKER_ALPHA = 0.97

GUIDE_LINE_WIDTH = 1.25

GUIDE_LINE_ALPHA = 0.88

PANEL_LABEL_SIZE = 15


# ----------------------------------------------------------------------
# Legend
# ----------------------------------------------------------------------

LEGEND_FONT_SIZE = 10.5

LEGEND_MARKER_SCALE = 1.15


# ----------------------------------------------------------------------
# Colorbar
# ----------------------------------------------------------------------

COLORBAR_WIDTH = 0.020

COLORBAR_GAP = 0.018

COLORBAR_BOTTOM = 0.16

COLORBAR_HEIGHT = 0.78


# ======================================================================
# 6. MATPLOTLIB STYLE
# ======================================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.size": 12,

    "axes.labelsize": 19,

    "xtick.labelsize": 12,

    "ytick.labelsize": 12,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": True,

    "ytick.right": True,

    "xtick.major.size": 4.5,

    "ytick.major.size": 4.5,

    "xtick.major.width": 0.9,

    "ytick.major.width": 0.9,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "text.usetex": True,
})


# ======================================================================
# 7. CHECK FILES
# ======================================================================

if not FILE_S2.exists():

    raise FileNotFoundError(
        f"s2 file not found:\n{FILE_S2.resolve()}"
    )


if not FILE_TAU.exists():

    raise FileNotFoundError(
        f"tau file not found:\n{FILE_TAU.resolve()}"
    )


print()
print("=" * 88)
print("INPUT FILES")
print("=" * 88)

print(
    f"s2:\n  {FILE_S2.resolve()}"
)

print()

print(
    f"tau:\n  {FILE_TAU.resolve()}"
)


# ======================================================================
# 8. READ s2 FILE
# ======================================================================

def read_s2_file(filename):

    df = pd.read_csv(

        filename,

        sep=r"\s+",

        comment="#",

        header=None,

        usecols=[
            0,
            1,
            2,
            3
        ],

        names=[
            "P",
            "T",
            "rho",
            "s2"
        ]
    )


    for col in [
        "P",
        "T",
        "rho",
        "s2"
    ]:

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"
        )


    df = df[

        np.isfinite(
            df["P"]
        )

        &

        np.isfinite(
            df["T"]
        )

        &

        np.isfinite(
            df["rho"]
        )

        &

        np.isfinite(
            df["s2"]
        )

    ].copy()


    df = df[

        (df["P"] >= P_MIN)

        &

        (df["P"] <= P_MAX)

    ].copy()


    # ------------------------------------------------------------------
    # Average duplicated states, if present
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
        .agg(

            rho=(
                "rho",
                "mean"
            ),

            s2=(
                "s2",
                "mean"
            )
        )
    )


    # ------------------------------------------------------------------
    # Quantity actually plotted
    # ------------------------------------------------------------------

    df[
        "minus_s2"
    ] = (

        -df[
            "s2"
        ]
    )


    return (

        df
        .sort_values(
            [
                "T",
                "P"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# 9. READ tau FILE
# ======================================================================

def read_tau_file(filename):

    df = pd.read_csv(

        filename,

        sep=r"\s+",

        comment="#",

        header=None,

        usecols=[
            0,
            1,
            2,
            3
        ],

        names=[
            "P",
            "T",
            "rho",
            "tau"
        ]
    )


    for col in [
        "P",
        "T",
        "rho",
        "tau"
    ]:

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"
        )


    df = df[

        np.isfinite(
            df["P"]
        )

        &

        np.isfinite(
            df["T"]
        )

        &

        np.isfinite(
            df["rho"]
        )

        &

        np.isfinite(
            df["tau"]
        )

    ].copy()


    df = df[

        (df["P"] >= P_MIN)

        &

        (df["P"] <= P_MAX)

    ].copy()


    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # tau_global.dat contains duplicated (P,T) states.
    # Average them before plotting.
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
        .agg(

            rho=(
                "rho",
                "mean"
            ),

            tau=(
                "tau",
                "mean"
            )
        )
    )


    return (

        df
        .sort_values(
            [
                "T",
                "P"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# 10. LOAD DATA
# ======================================================================

df_s2 = read_s2_file(
    FILE_S2
)

df_tau = read_tau_file(
    FILE_TAU
)


# ======================================================================
# 11. INPUT DIAGNOSTIC
# ======================================================================

print()
print("=" * 88)
print("DATA SUMMARY")
print("=" * 88)


print(
    f"s2 states = "
    f"{len(df_s2)}"
)

print(
    f"s2 P range = "
    f"{df_s2['P'].min():.3f}"
    f" -- "
    f"{df_s2['P'].max():.3f}"
)

print(
    f"s2 T range = "
    f"{df_s2['T'].min():.3f}"
    f" -- "
    f"{df_s2['T'].max():.3f}"
)

print(
    f"s2 range = "
    f"{df_s2['s2'].min():.6f}"
    f" -- "
    f"{df_s2['s2'].max():.6f}"
)

print(
    f"-s2 range = "
    f"{df_s2['minus_s2'].min():.6f}"
    f" -- "
    f"{df_s2['minus_s2'].max():.6f}"
)


print()


print(
    f"tau states = "
    f"{len(df_tau)}"
)

print(
    f"tau P range = "
    f"{df_tau['P'].min():.3f}"
    f" -- "
    f"{df_tau['P'].max():.3f}"
)

print(
    f"tau T range = "
    f"{df_tau['T'].min():.3f}"
    f" -- "
    f"{df_tau['T'].max():.3f}"
)

print(
    f"tau range = "
    f"{df_tau['tau'].min():.6f}"
    f" -- "
    f"{df_tau['tau'].max():.6f}"
)


# ======================================================================
# 12. NEAREST AVAILABLE TEMPERATURE
# ======================================================================

def nearest_temperature(
    dataframe,
    target
):

    available = np.sort(

        dataframe[
            "T"
        ].unique()
    )


    index = np.argmin(

        np.abs(
            available-target
        )
    )


    return float(
        available[
            index
        ]
    )


# ======================================================================
# 13. MATCH REQUESTED ISOTHERMS
# ======================================================================

T_S2 = [

    nearest_temperature(
        df_s2,
        T
    )

    for T in SELECTED_T
]


T_TAU = [

    nearest_temperature(
        df_tau,
        T
    )

    for T in SELECTED_T
]


print()
print("=" * 88)
print("SELECTED ISOTHERMS")
print("=" * 88)


for target, Ts2, Ttau in zip(
    SELECTED_T,
    T_S2,
    T_TAU
):

    print(

        f"requested T*={target:.2f}   "
        f"s2 -> {Ts2:.3f}   "
        f"tau -> {Ttau:.3f}"
    )


# ======================================================================
# 14. TEMPERATURE COLOR SCALE
# ======================================================================

norm = Normalize(

    vmin=float(
        SELECTED_T.min()
    ),

    vmax=float(
        SELECTED_T.max()
    )
)


cmap = plt.get_cmap(
    "cividis"
)


# ======================================================================
# 15. SHAPE-PRESERVING PCHIP GUIDE
# ======================================================================

def pchip_guide(
    P,
    Y,
    ngrid=700
):

    P = np.asarray(
        P,
        dtype=float
    )


    Y = np.asarray(
        Y,
        dtype=float
    )


    good = (

        np.isfinite(
            P
        )

        &

        np.isfinite(
            Y
        )
    )


    P = P[
        good
    ]

    Y = Y[
        good
    ]


    if len(
        P
    ) < 3:

        return (
            P,
            Y
        )


    order = np.argsort(
        P
    )


    P = P[
        order
    ]

    Y = Y[
        order
    ]


    P_unique, unique_index = np.unique(

        P,

        return_index=True
    )


    Y_unique = Y[
        unique_index
    ]


    if len(
        P_unique
    ) < 3:

        return (
            P_unique,
            Y_unique
        )


    interpolator = PchipInterpolator(

        P_unique,

        Y_unique,

        extrapolate=False
    )


    P_grid = np.linspace(

        P_unique.min(),

        P_unique.max(),

        ngrid
    )


    Y_grid = interpolator(
        P_grid
    )


    return (
        P_grid,
        Y_grid
    )


# ======================================================================
# 16. CREATE FIGURE
# ======================================================================

fig, (
    ax1,
    ax2
) = plt.subplots(

    1,
    2,

    figsize=FIGSIZE
)


# ======================================================================
# 17. PANEL (a): -s2*
# ======================================================================

for target_T, actual_T in zip(
    SELECTED_T,
    T_S2
):

    iso = df_s2[

        np.isclose(

            df_s2[
                "T"
            ],

            actual_T,

            atol=1.0e-8
        )

    ].sort_values(
        "P"
    )


    if len(
        iso
    ) == 0:

        continue


    color = cmap(
        norm(
            target_T
        )
    )


    P_data = iso[
        "P"
    ].to_numpy(
        dtype=float
    )


    minus_s2_data = iso[
        "minus_s2"
    ].to_numpy(
        dtype=float
    )


    # ==============================================================
    # PCHIP guide
    # ==============================================================

    P_guide, S2_guide = pchip_guide(

        P_data,

        minus_s2_data
    )


    ax1.plot(

        P_guide,

        S2_guide,

        color=color,

        lw=GUIDE_LINE_WIDTH,

        alpha=GUIDE_LINE_ALPHA,

        zorder=2
    )


    # ==============================================================
    # Simulation points
    # ==============================================================

    ax1.plot(

        P_data,

        minus_s2_data,

        linestyle="none",

        marker="o",

        ms=MARKER_SIZE,

        markerfacecolor=color,

        markeredgecolor="black",

        markeredgewidth=(
            MARKER_EDGE_WIDTH
        ),

        alpha=MARKER_ALPHA,

        label=(
            rf"$T^*={target_T:.2f}$"
        ),

        zorder=3
    )


# ======================================================================
# 18. PANEL (b): tau
# ======================================================================

for target_T, actual_T in zip(
    SELECTED_T,
    T_TAU
):

    iso = df_tau[

        np.isclose(

            df_tau[
                "T"
            ],

            actual_T,

            atol=1.0e-8
        )

    ].sort_values(
        "P"
    )


    if len(
        iso
    ) == 0:

        continue


    color = cmap(
        norm(
            target_T
        )
    )


    P_data = iso[
        "P"
    ].to_numpy(
        dtype=float
    )


    tau_data = iso[
        "tau"
    ].to_numpy(
        dtype=float
    )


    # ==============================================================
    # PCHIP guide
    # ==============================================================

    P_guide, TAU_guide = pchip_guide(

        P_data,

        tau_data
    )


    ax2.plot(

        P_guide,

        TAU_guide,

        color=color,

        lw=GUIDE_LINE_WIDTH,

        alpha=GUIDE_LINE_ALPHA,

        zorder=2
    )


    # ==============================================================
    # Simulation points
    # ==============================================================

    ax2.plot(

        P_data,

        tau_data,

        linestyle="none",

        marker="o",

        ms=MARKER_SIZE,

        markerfacecolor=color,

        markeredgecolor="black",

        markeredgewidth=(
            MARKER_EDGE_WIDTH
        ),

        alpha=MARKER_ALPHA,

        label=(
            rf"$T^*={target_T:.2f}$"
        ),

        zorder=3
    )


# ======================================================================
# 19. COMMON AXIS APPEARANCE
# ======================================================================

for ax in [
    ax1,
    ax2
]:

    ax.set_xlim(
        P_MIN,
        P_MAX
    )


    ax.grid(

        True,

        linestyle=":",

        linewidth=0.65,

        alpha=0.24,

        color="0.35"
    )


    ax.tick_params(

        direction="in",

        top=True,

        right=True,

        length=4.5,

        width=0.9
    )


    ax.set_xlabel(
        r"$P^{\ast}$"
    )


ax1.set_ylabel(
    r"$-s_{2}^{\ast}$"
)


ax2.set_ylabel(
    r"$\tau$"
)


# ======================================================================
# 20. PANEL LABELS
# ======================================================================

ax1.text(

    0.035,
    0.955,

    r"\textbf{(a)}",

    transform=ax1.transAxes,

    fontsize=PANEL_LABEL_SIZE,

    ha="left",

    va="top",

    zorder=10
)


ax2.text(

    0.035,
    0.955,

    r"\textbf{(b)}",

    transform=ax2.transAxes,

    fontsize=PANEL_LABEL_SIZE,

    ha="left",

    va="top",

    zorder=10
)


# ======================================================================
# 21. LEGENDS
#
# Larger and in ONE COLUMN
# ======================================================================

legend_kwargs = dict(

    frameon=True,

    framealpha=0.94,

    facecolor="white",

    edgecolor="0.70",

    fontsize=(
        LEGEND_FONT_SIZE
    ),

    ncol=1,

    handletextpad=0.55,

    labelspacing=0.45,

    borderpad=0.55,

    borderaxespad=0.50,

    markerscale=(
        LEGEND_MARKER_SCALE
    ),

    handlelength=1.20
)


ax1.legend(

    loc="lower right",

    bbox_to_anchor=(
        0.975,
        0.025
    ),

    **legend_kwargs
)


ax2.legend(

    loc="lower right",

    bbox_to_anchor=(
        0.975,
        0.025
    ),

    **legend_kwargs
)


# ======================================================================
# 22. Y-MARGINS
# ======================================================================

for ax in [
    ax1,
    ax2
]:

    ymin, ymax = ax.get_ylim()


    yrange = (
        ymax-ymin
    )


    ax.set_ylim(

        ymin
        -
        0.025*yrange,

        ymax
        +
        0.045*yrange
    )


# ======================================================================
# 23. RESERVE SPACE FOR COLORBAR
# ======================================================================

fig.subplots_adjust(

    left=0.085,

    right=0.875,

    bottom=0.155,

    top=0.965,

    wspace=0.26
)


# ======================================================================
# 24. EXPLICIT COLORBAR AXIS
# ======================================================================

cax_left = (

    0.875

    +

    COLORBAR_GAP
)


cax = fig.add_axes(

    [
        cax_left,
        COLORBAR_BOTTOM,
        COLORBAR_WIDTH,
        COLORBAR_HEIGHT
    ]
)


sm = ScalarMappable(

    norm=norm,

    cmap=cmap
)


sm.set_array(
    []
)


cbar = fig.colorbar(

    sm,

    cax=cax
)


cbar.set_label(

    r"$T^{\ast}$",

    fontsize=17,

    labelpad=10
)


cbar.set_ticks(
    SELECTED_T
)


cbar.ax.tick_params(

    direction="in",

    length=4,

    width=0.8,

    labelsize=11
)


cbar.outline.set_linewidth(
    1.0
)


# ======================================================================
# 25. EXPORT
# ======================================================================

fig.savefig(

    OUT_PDF,

    bbox_inches="tight"
)


fig.savefig(

    OUT_PNG,

    dpi=600,

    bbox_inches="tight"
)


plt.show()


# ======================================================================
# 26. FINAL REPORT
# ======================================================================

print()
print("=" * 88)
print("GLOBAL PAIR-ORDER FIGURE")
print("=" * 88)


print(
    "✓ Correct s2 column used."
)


print(
    "✓ Correct tau column used."
)


print(
    "✓ -s2 calculated from conventional negative s2."
)


print(
    "✓ Duplicate tau states averaged by (P,T)."
)


print(
    "✓ Panel (a): -s2* versus P*."
)


print(
    "✓ Panel (b): tau versus P*."
)


print(
    "✓ Six representative isotherms."
)


print(
    "✓ Large one-column legends."
)


print(
    "✓ Shared continuous temperature colorbar."
)


print(
    "✓ PCHIP used only as visual guide."
)


print(
    "✓ No extrapolation."
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


print("=" * 88)
