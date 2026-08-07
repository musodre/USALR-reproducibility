#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
figure13_cluster_dynamics.py

Final publication figure for cluster reorganization across the
diffusion-anomaly boundary in the USALR fluid.

Run from:
    USALR/python-inputs

Figure layout
-------------
(a) particle-weighted cluster size <s>_w vs P*
    + low-pressure inset

(b) clustered fraction f_cl vs P*

(c) particle-weighted cluster-size distribution P_p(s), T*=0.30

(d) particle-weighted cluster-size distribution P_p(s), T*=0.50

Inputs
------
plots/cluster_analysis/cluster_state_summary.dat

plots/derivative_boundary_alignment/dynamic_boundaries.dat

plots/cluster_size_distributions/selected_cluster_states.dat

plots/cluster_analysis/distributions/
    cluster_distribution_P_..._T_....dat

Outputs
-------
plots/figure13_cluster_dynamics/
    figure13_cluster_dynamics.pdf
    figure13_cluster_dynamics.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator



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
# 1. PATHS
# =============================================================================

CLUSTER_FILE = DERIVED_DATA_ROOT / "cluster_analysis" / "cluster_state_summary.dat"

BOUNDARY_FILE = DERIVED_DATA_ROOT / "derivative_boundary_alignment" / "dynamic_boundaries.dat"

SELECTED_FILE = DERIVED_DATA_ROOT / "cluster_size_distributions" / "selected_cluster_states.dat"

DISTDIR = DERIVED_DATA_ROOT / "cluster_analysis" / "distributions"

OUTDIR = FIGURE_OUTPUT_ROOT / "figure13_cluster_dynamics"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. THERMODYNAMIC STATES
# =============================================================================

# Isotherms shown in panels (a,b)

T_CURVES = [
    0.30,
    0.40,
    0.50,
    0.60,
]


# Temperatures used for representative P_p(s) distributions

T_DIST = [
    0.30,
    0.50,
]


# =============================================================================
# 3. PUBLICATION PALETTE
# =============================================================================
#
# Chosen to remain visually close to the palette used in Fig. 12,
# while preserving good separation among neighboring isotherms.

TCOLORS = {

    0.30: "#3B6EA8",   # deep blue
    0.40: "#2B8C4B",   # restrained green
    0.50: "#D0B832",   # muted yellow-gold
    0.60: "#B12F72",   # magenta

}


# The before/near/after colors deliberately differ from the
# temperature palette to avoid confusing thermodynamic and
# dynamic classifications.

REGION_COLORS = {

    "before": "#3B6EA8",
    "near":   "#C28B22",
    "after":  "#A83F68",

}


REGION_MARKERS = {

    "before": "o",
    "near": "s",
    "after": "^",

}


REGION_LABELS = {

    "before": "before",
    "near": "near",
    "after": "after",

}


# =============================================================================
# 4. LATEX / PUBLICATION STYLE
# =============================================================================
#
# This assumes a working LaTeX installation, which is already used in
# the other publication plots.

plt.rcParams.update({

    # -------------------------------------------------------------------------
    # LaTeX rendering
    # -------------------------------------------------------------------------

    "text.usetex": True,

    "font.family": "serif",

    "font.serif": [
        "Computer Modern Roman"
    ],

    "mathtext.fontset": "cm",

    # -------------------------------------------------------------------------
    # Font sizes
    # -------------------------------------------------------------------------

    "font.size": 10.0,

    "axes.labelsize": 12.5,

    "axes.titlesize": 11.0,

    "xtick.labelsize": 9.5,

    "ytick.labelsize": 9.5,

    "legend.fontsize": 8.5,

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    "axes.linewidth": 1.0,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": True,

    "ytick.right": True,

    "xtick.major.size": 4.2,

    "ytick.major.size": 4.2,

    "xtick.minor.size": 2.2,

    "ytick.minor.size": 2.2,

    "xtick.major.width": 0.9,

    "ytick.major.width": 0.9,

    "xtick.minor.width": 0.7,

    "ytick.minor.width": 0.7,

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    "legend.frameon": False,

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    "savefig.transparent": False,

})


# =============================================================================
# 5. VALIDATION
# =============================================================================

for path in [

    CLUSTER_FILE,
    BOUNDARY_FILE,
    SELECTED_FILE,

]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing input file:\n"
            f"{path.resolve()}\n"
        )


if not DISTDIR.exists():

    raise FileNotFoundError(
        f"\nMissing distribution directory:\n"
        f"{DISTDIR.resolve()}\n"
    )


# =============================================================================
# 6. LOAD CLUSTER MASTER DATA
# =============================================================================

cluster = pd.read_csv(

    CLUSTER_FILE,

    sep=r"\s+",

    engine="python"

)


required_cluster_columns = [

    "P",
    "T",
    "weighted_cluster",
    "clustered_fraction",

]


for column in required_cluster_columns:

    if column not in cluster.columns:

        raise RuntimeError(
            f"\nMissing column in cluster file: "
            f"{column}\n"
        )


    cluster[
        column
    ] = pd.to_numeric(

        cluster[
            column
        ],

        errors="coerce"

    )


cluster = cluster.dropna(
    subset=required_cluster_columns
)


cluster = (

    cluster
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


# =============================================================================
# 7. LOAD DYNAMIC BOUNDARIES
# =============================================================================

boundaries = pd.read_csv(

    BOUNDARY_FILE,

    sep=r"\s+",

    engine="python"

)


for column in [

    "T",
    "P_boundary",

]:

    boundaries[
        column
    ] = pd.to_numeric(

        boundaries[
            column
        ],

        errors="coerce"

    )


boundaries = boundaries.dropna(

    subset=[
        "T",
        "P_boundary",
    ]

)


# =============================================================================
# 8. LOAD REPRESENTATIVE STATES
# =============================================================================

selected = pd.read_csv(

    SELECTED_FILE,

    sep=r"\s+",

    engine="python"

)


required_selected_columns = [

    "T",
    "region",
    "P_selected",

]


for column in required_selected_columns:

    if column not in selected.columns:

        raise RuntimeError(
            f"\nMissing column in selected-state file: "
            f"{column}\n"
        )


selected[
    "T"
] = pd.to_numeric(

    selected[
        "T"
    ],

    errors="coerce"

)


selected[
    "P_selected"
] = pd.to_numeric(

    selected[
        "P_selected"
    ],

    errors="coerce"

)


# =============================================================================
# 9. INTERPOLATION AT THE DYNAMIC BOUNDARY
# =============================================================================

def interpolate_at(
    x,
    y,
    x0
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )


    mask = (

        np.isfinite(x)
        &
        np.isfinite(y)

    )


    x = x[
        mask
    ]

    y = y[
        mask
    ]


    if len(
        x
    ) < 4:

        return np.nan


    order = np.argsort(
        x
    )


    x = x[
        order
    ]

    y = y[
        order
    ]


    xu, idx = np.unique(

        x,

        return_index=True

    )


    yu = y[
        idx
    ]


    if len(
        xu
    ) < 4:

        return np.nan


    if not (

        xu.min()
        <=
        x0
        <=
        xu.max()

    ):

        return np.nan


    try:

        interpolator = PchipInterpolator(

            xu,
            yu,

            extrapolate=False

        )


        return float(
            interpolator(
                x0
            )
        )


    except Exception:

        return np.nan


# =============================================================================
# 10. READ ONE CLUSTER-SIZE DISTRIBUTION
# =============================================================================

def read_distribution(
    path
):

    data = np.loadtxt(
        path
    )


    data = np.atleast_2d(
        data
    )


    if data.shape[
        1
    ] < 4:

        raise RuntimeError(
            f"\nUnexpected distribution format:\n"
            f"{path}\n"
        )


    return {

        "s":
        data[
            :,
            0
        ],

        "Pparticle":
        data[
            :,
            3
        ],

    }


# =============================================================================
# 11. DISTRIBUTION FILE NAME
# =============================================================================

def distribution_path(
    P,
    T
):

    path = (

        DISTDIR
        /
        (
            f"cluster_distribution_"
            f"P_{P:.3f}_"
            f"T_{T:.2f}.dat"
        )

    )


    if not path.exists():

        raise FileNotFoundError(
            f"\nDistribution not found:\n"
            f"{path}\n"
        )


    return path


# =============================================================================
# 12. FIGURE CANVAS
# =============================================================================
#
# Approximately full-width APS figure* dimensions.

fig = plt.figure(

    figsize=(
        7.25,
        6.15
    )

)


gs = fig.add_gridspec(

    2,
    2,

    left=0.095,
    right=0.985,

    bottom=0.095,
    top=0.975,

    wspace=0.29,
    hspace=0.30

)


ax_a = fig.add_subplot(
    gs[
        0,
        0
    ]
)

ax_b = fig.add_subplot(
    gs[
        0,
        1
    ]
)

ax_c = fig.add_subplot(
    gs[
        1,
        0
    ]
)

ax_d = fig.add_subplot(
    gs[
        1,
        1
    ]
)


# =============================================================================
# 13. PANEL (a): PARTICLE-WEIGHTED CLUSTER SIZE
# =============================================================================

for T in T_CURVES:

    sub = cluster[

        np.isclose(

            cluster[
                "T"
            ],

            T,

            atol=1e-8

        )

    ].sort_values(
        "P"
    )


    if len(
        sub
    ) == 0:

        continue


    ax_a.plot(

        sub[
            "P"
        ],

        sub[
            "weighted_cluster"
        ],

        marker="o",

        markersize=3.6,

        linewidth=1.55,

        markeredgewidth=0.4,

        color=TCOLORS[
            T
        ],

        label=(
            rf"$T^*={T:.2f}$"
        ),

        zorder=4

    )


    # -------------------------------------------------------------------------
    # Dynamic boundary
    # -------------------------------------------------------------------------

    boundary = boundaries[

        np.isclose(

            boundaries[
                "T"
            ],

            T,

            atol=1e-8

        )

    ]


    if len(
        boundary
    ) > 0:

        P0 = float(

            boundary
            .iloc[
                0
            ][
                "P_boundary"
            ]

        )


        y0 = interpolate_at(

            sub[
                "P"
            ],

            sub[
                "weighted_cluster"
            ],

            P0

        )


        if np.isfinite(
            y0
        ):

            ax_a.plot(

                P0,
                y0,

                marker="o",

                markersize=7.2,

                markerfacecolor="white",

                markeredgecolor=TCOLORS[
                    T
                ],

                markeredgewidth=1.5,

                linestyle="None",

                zorder=20

            )


ax_a.set_xlabel(
    r"$P^*$"
)


ax_a.set_ylabel(
    r"$\langle s\rangle_{\rm w}$"
)


ax_a.set_xlim(
    0.0,
    6.05
)


# -------------------------------------------------------------------------
# Legend
# -------------------------------------------------------------------------

ax_a.legend(

    loc="upper center",

    bbox_to_anchor=(
        0.58,
        0.995
    ),

    ncol=2,

    handlelength=1.45,

    columnspacing=0.9,

    handletextpad=0.45,

    labelspacing=0.28

)


# =============================================================================
# 14. LOW-PRESSURE INSET IN PANEL (a)
# =============================================================================

axins = ax_a.inset_axes([0.1, 0.3, 0.42, 0.36])

for T in T_CURVES:

    sub = cluster[

        np.isclose(

            cluster[
                "T"
            ],

            T,

            atol=1e-8

        )

    ].sort_values(
        "P"
    )


    sub_low = sub[
        sub[
            "P"
        ]
        <=
        1.20
    ]


    if len(
        sub_low
    ) == 0:

        continue


    axins.plot(

        sub_low[
            "P"
        ],

        sub_low[
            "weighted_cluster"
        ],

        marker="o",

        markersize=2.7,

        linewidth=1.10,

        markeredgewidth=0.3,

        color=TCOLORS[
            T
        ]

    )


    boundary = boundaries[

        np.isclose(

            boundaries[
                "T"
            ],

            T,

            atol=1e-8

        )

    ]


    if len(
        boundary
    ) > 0:

        P0 = float(

            boundary
            .iloc[
                0
            ][
                "P_boundary"
            ]

        )


        y0 = interpolate_at(

            sub[
                "P"
            ],

            sub[
                "weighted_cluster"
            ],

            P0

        )


        if np.isfinite(
            y0
        ):

            axins.plot(

                P0,
                y0,

                marker="o",

                markersize=5.1,

                markerfacecolor="white",

                markeredgecolor=TCOLORS[
                    T
                ],

                markeredgewidth=1.1,

                linestyle="None",

                zorder=20

            )


axins.set_xlim(
    0.0,
    1.20
)

low_mask = np.logical_and(
    cluster["P"].to_numpy(dtype=float) <= 1.20,
    cluster["T"].isin(T_CURVES).to_numpy(dtype=bool)
)


low_values = cluster.loc[
    low_mask,
    "weighted_cluster"
]


if len(low_values) > 0:
    ymax_low = (

        1.12
        *
        low_values.max())


    axins.set_ylim(

        0.0,
        ymax_low

    )


axins.set_xlabel(

    r"$P^*$",

    fontsize=7.8,

    labelpad=1.0

)


axins.set_ylabel(

    r"$\langle s\rangle_{\rm w}$",

    fontsize=7.8,

    labelpad=1.0

)


axins.tick_params(

    axis="both",

    which="major",

    labelsize=7.0,

    length=2.8,

    width=0.7

)


axins.tick_params(

    axis="both",

    which="minor",

    length=1.5,

    width=0.6

)


for spine in axins.spines.values():

    spine.set_linewidth(
        0.8
    )


# Subtle indication of enlarged region

try:

    ax_a.indicate_inset_zoom(axins, edgecolor="0.45", linewidth=0.55, alpha=0.5)

except Exception:

    pass


# =============================================================================
# 15. PANEL (b): CLUSTERED FRACTION
# =============================================================================

for T in T_CURVES:

    sub = cluster[

        np.isclose(

            cluster[
                "T"
            ],

            T,

            atol=1e-8

        )

    ].sort_values(
        "P"
    )


    if len(
        sub
    ) == 0:

        continue


    ax_b.plot(

        sub[
            "P"
        ],

        sub[
            "clustered_fraction"
        ],

        marker="o",

        markersize=3.6,

        linewidth=1.55,

        markeredgewidth=0.4,

        color=TCOLORS[
            T
        ],

        zorder=4

    )


    boundary = boundaries[

        np.isclose(

            boundaries[
                "T"
            ],

            T,

            atol=1e-8

        )

    ]


    if len(
        boundary
    ) > 0:

        P0 = float(

            boundary
            .iloc[
                0
            ][
                "P_boundary"
            ]

        )


        y0 = interpolate_at(

            sub[
                "P"
            ],

            sub[
                "clustered_fraction"
            ],

            P0

        )


        if np.isfinite(
            y0
        ):

            ax_b.plot(

                P0,
                y0,

                marker="o",

                markersize=7.2,

                markerfacecolor="white",

                markeredgecolor=TCOLORS[
                    T
                ],

                markeredgewidth=1.5,

                linestyle="None",

                zorder=20

            )


ax_b.set_xlabel(
    r"$P^*$"
)


ax_b.set_ylabel(
    r"$f_{\rm cl}$"
)


ax_b.set_xlim(
    0.0,
    6.05
)


ax_b.set_ylim(
    0.44,
    0.92
)


# =============================================================================
# 16. DISTRIBUTION PANEL FUNCTION
# =============================================================================

def plot_distribution_panel(
    ax,
    T
):

    sub = selected[

        np.isclose(

            selected[
                "T"
            ],

            T,

            atol=1e-8

        )

    ]


    if len(
        sub
    ) == 0:

        raise RuntimeError(
            f"\nNo selected states for T*={T:.2f}\n"
        )


    for region in [

        "before",
        "near",
        "after",

    ]:

        row = sub[
            sub[
                "region"
            ]
            ==
            region
        ]


        if len(
            row
        ) == 0:

            continue


        row = row.iloc[
            0
        ]


        P = float(
            row[
                "P_selected"
            ]
        )


        path = distribution_path(

            P,
            T

        )


        dist = read_distribution(
            path
        )


        s = np.asarray(

            dist[
                "s"
            ],

            dtype=float

        )


        y = np.asarray(

            dist[
                "Pparticle"
            ],

            dtype=float

        )


        mask = (

            np.isfinite(
                s
            )

            &

            np.isfinite(
                y
            )

            &

            (
                s
                >
                0
            )

            &

            (
                y
                >
                0
            )

        )


        s = s[
            mask
        ]


        y = y[
            mask
        ]


        ax.plot(

            s,
            y,

            marker=REGION_MARKERS[
                region
            ],

            markersize=4.0,

            linewidth=1.45,

            markeredgewidth=0.45,

            color=REGION_COLORS[
                region
            ],

            label=(

                rf"{REGION_LABELS[region]}, "
                rf"$P^*={P:.2f}$"

            ),

            zorder=4

        )


    ax.set_xscale(
        "log"
    )


    ax.set_yscale(
        "log"
    )


    ax.set_xlabel(
        r"$s$"
    )


    ax.set_ylabel(
        r"$P_{\rm p}(s)$"
    )


    # Temperature annotation

    ax.text(

        0.965,
        0.945,

        rf"$T^*={T:.2f}$",

        transform=ax.transAxes,

        ha="right",
        va="top",

        fontsize=10.5

    )


    ax.legend(

        loc="lower left",

        handlelength=1.45,

        handletextpad=0.45,

        labelspacing=0.35

    )


# =============================================================================
# 17. PANELS (c,d)
# =============================================================================

plot_distribution_panel(

    ax_c,
    0.30

)


plot_distribution_panel(

    ax_d,
    0.50

)


# =============================================================================
# 18. PANEL LABELS
# =============================================================================
#
# Use ordinary strings + fontweight instead of \textbf so the labels
# render correctly regardless of the Matplotlib/LaTeX backend.

#for ax, label in [(ax_a, "(a)" ), (ax_b,"(b)"), (ax_c, "(c)"), (ax_d,"(d)"),]:
#    ax.text(0.035, 0.955, label, transform=ax.transAxes, ha="left",
#        va="top", fontsize=12.0, fontweight="bold",zorder=50)


# =============================================================================
# 19. REMOVE GRID COMPLETELY
# =============================================================================
#
# Intentionally no grid: this matches the cleaner style of Fig. 12.

for ax in [

    ax_a,
    ax_b,
    ax_c,
    ax_d,

]:

    ax.grid(
        False
    )


# =============================================================================
# 20. SAVE
# =============================================================================

PDF = (

    OUTDIR
    /
    "figure13_cluster_dynamics.pdf"

)


PNG = (

    OUTDIR
    /
    "figure13_cluster_dynamics.png"

)


fig.savefig(

    PDF,

    bbox_inches="tight",

    pad_inches=0.03

)


fig.savefig(

    PNG,

    dpi=500,

    bbox_inches="tight",

    pad_inches=0.03

)


plt.close(
    fig
)


# =============================================================================
# 21. TERMINAL SUMMARY
# =============================================================================

print()
print("=" * 92)
print("FIGURE 13 — FINAL CLUSTER-DYNAMICS PLOT")
print("=" * 92)

print()
print("Panel (a): <s>_w vs P*")
print("           low-pressure inset included")

print()
print("Panel (b): f_cl vs P*")

print()
print("Panel (c): P_p(s), T*=0.30")

print()
print("Panel (d): P_p(s), T*=0.50")

print()
print(
    "Open circles in panels (a,b) mark "
    "(d ln D*/dP*)_T* = 0."
)

print()
print("LaTeX text rendering: ON")

print()
print("Files written:")

print(PDF)
print(PNG)
print()
plt.show()
print("Done.")
