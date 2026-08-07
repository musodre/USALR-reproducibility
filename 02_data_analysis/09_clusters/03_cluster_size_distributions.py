#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_size_distributions.py

Analyze cluster-size distributions relative to the diffusion-anomaly
boundary.

Run from:
    USALR/python-inputs

Inputs
------
plots/cluster_analysis/distributions/
    cluster_distribution_P_..._T_....dat

plots/derivative_boundary_alignment/
    dynamic_boundaries.dat

Outputs
-------
plots/cluster_size_distributions/

    selected_cluster_states.dat

    cluster_number_distribution.pdf/png
    particle_weighted_distribution.pdf/png
    cluster_distribution_combined.pdf/png
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
# 1. PATHS
# =============================================================================

DISTDIR = DERIVED_DATA_ROOT / "cluster_analysis" / "distributions"

BOUNDARY_FILE = DERIVED_DATA_ROOT / "derivative_boundary_alignment" / "dynamic_boundaries.dat"

OUTDIR = DERIVED_DATA_ROOT / "cluster_size_distributions"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

T_SELECTED = [
    0.30,
    0.40,
    0.50,
]

DELTA_P = 0.30


# =============================================================================
# 3. COLORS / STYLES
# =============================================================================

COLORS = {

    "before": "#4477AA",
    "near":   "#CCBB44",
    "after":  "#AA3377",

}

LABELS = {

    "before": "before boundary",
    "near":   "near boundary",
    "after":  "after boundary",

}

MARKERS = {

    "before": "o",
    "near": "s",
    "after": "^",

}


# =============================================================================
# 4. MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 11,

    "axes.labelsize": 14,

    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    "legend.fontsize": 9.2,

    "axes.linewidth": 1.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "legend.frameon": False,

})


# =============================================================================
# 5. VALIDATION
# =============================================================================

if not DISTDIR.exists():

    raise FileNotFoundError(
        f"\nDistribution directory not found:\n"
        f"{DISTDIR.resolve()}\n"
    )


if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"\nBoundary file not found:\n"
        f"{BOUNDARY_FILE.resolve()}\n"
    )


# =============================================================================
# 6. PARSE DISTRIBUTION FILENAMES
# =============================================================================

def parse_PT(path):

    match = re.search(
        r"P_([0-9]+(?:\.[0-9]+)?)_T_([0-9]+(?:\.[0-9]+)?)",
        path.name
    )

    if match is None:

        return None

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


# =============================================================================
# 7. BUILD AVAILABLE-STATE DATABASE
# =============================================================================

files = sorted(
    DISTDIR.glob(
        "cluster_distribution_P_*_T_*.dat"
    )
)


rows = []


for path in files:

    parsed = parse_PT(
        path
    )

    if parsed is None:
        continue


    P, T = parsed


    rows.append({

        "P": P,
        "T": T,
        "file": path,

    })


states = pd.DataFrame(
    rows
)


if len(states) == 0:

    raise RuntimeError(
        "No cluster distribution files were found."
    )


# =============================================================================
# 8. LOAD BOUNDARIES
# =============================================================================

boundaries = pd.read_csv(

    BOUNDARY_FILE,

    sep=r"\s+",

    engine="python"

)


for c in [
    "T",
    "P_boundary",
]:

    boundaries[c] = pd.to_numeric(
        boundaries[c],
        errors="coerce"
    )


boundaries = boundaries.dropna(
    subset=[
        "T",
        "P_boundary",
    ]
)


# =============================================================================
# 9. FIND NEAREST AVAILABLE STATE
# =============================================================================

def nearest_state(
    T0,
    Ptarget
):

    sub = states[
        np.isclose(
            states["T"],
            T0,
            atol=1e-8
        )
    ].copy()


    if len(sub) == 0:

        return None


    sub[
        "distance"
    ] = np.abs(
        sub["P"]
        -
        Ptarget
    )


    sub = sub.sort_values(
        "distance"
    )


    return sub.iloc[0]


# =============================================================================
# 10. SELECT STATES
# =============================================================================

selected_rows = []


for T0 in T_SELECTED:

    b = boundaries[
        np.isclose(
            boundaries["T"],
            T0,
            atol=1e-8
        )
    ]


    if len(b) == 0:

        print(
            f"No boundary found for T*={T0:.2f}"
        )

        continue


    Pboundary = float(
        b.iloc[0][
            "P_boundary"
        ]
    )


    targets = {

        "before":
        max(
            0.0,
            Pboundary - DELTA_P
        ),

        "near":
        Pboundary,

        "after":
        Pboundary + DELTA_P,

    }


    for region, Ptarget in targets.items():

        state = nearest_state(

            T0,
            Ptarget

        )


        if state is None:

            continue


        selected_rows.append({

            "T":
            T0,

            "P_boundary":
            Pboundary,

            "region":
            region,

            "P_target":
            Ptarget,

            "P_selected":
            float(
                state["P"]
            ),

            "distance":
            float(
                state["distance"]
            ),

            "file":
            str(
                state["file"]
            ),

        })


selected = pd.DataFrame(
    selected_rows
)


if len(selected) == 0:

    raise RuntimeError(
        "No representative states could be selected."
    )


selected.to_csv(

    OUTDIR /
    "selected_cluster_states.dat",

    sep=" ",

    index=False,

    float_format="%.10e"

)


# =============================================================================
# 11. PRINT SELECTED STATES
# =============================================================================

print()
print("=" * 90)
print("SELECTED STATES")
print("=" * 90)


print(

    selected[
        [
            "T",
            "P_boundary",
            "region",
            "P_target",
            "P_selected",
            "distance",
        ]
    ].to_string(
        index=False
    )

)


# =============================================================================
# 12. READ DISTRIBUTION
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


    if data.shape[1] < 4:

        raise RuntimeError(
            f"\nUnexpected distribution format:\n"
            f"{path}\n"
        )


    return {

        "s":
        data[:, 0],

        "Ns":
        data[:, 1],

        "Pcluster":
        data[:, 2],

        "Pparticle":
        data[:, 3],

    }


# =============================================================================
# 13. PLOT FUNCTION
# =============================================================================

def make_distribution_plot(
    ykey,
    ylabel,
    stem,
    logy=True
):

    fig, axes = plt.subplots(

        1,
        len(T_SELECTED),

        figsize=(
            10.5,
            3.8
        ),

        sharey=True

    )


    if len(T_SELECTED) == 1:

        axes = [axes]


    for ax, T0 in zip(
        axes,
        T_SELECTED
    ):

        sub = selected[
            np.isclose(
                selected["T"],
                T0,
                atol=1e-8
            )
        ]


        for region in [
            "before",
            "near",
            "after",
        ]:

            row = sub[
                sub["region"]
                ==
                region
            ]


            if len(row) == 0:

                continue


            row = row.iloc[0]


            dist = read_distribution(
                row["file"]
            )


            s = dist[
                "s"
            ]

            y = dist[
                ykey
            ]


            mask = (
                np.isfinite(s)
                &
                np.isfinite(y)
                &
                (y > 0)
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

                marker=MARKERS[
                    region
                ],

                markersize=4.2,

                linewidth=1.5,

                color=COLORS[
                    region
                ],

                label=(
                    LABELS[
                        region
                    ]
                    +
                    rf" ($P^*={row['P_selected']:.2f}$)"
                )

            )


        ax.set_xlabel(
            r"$s$"
        )


        ax.set_title(
            rf"$T^*={T0:.2f}$"
        )


        if logy:

            ax.set_yscale(
                "log"
            )


        ax.set_xscale(
            "log"
        )


    axes[0].set_ylabel(
        ylabel
    )


    handles,labels = axes[0].get_legend_handles_labels()


    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)


    fig.subplots_adjust(

        left=0.08,

        right=0.985,

        bottom=0.16,

        top=0.80,

        wspace=0.10

    )


    fig.savefig(

        OUTDIR /
        f"{stem}.pdf",

        bbox_inches="tight"

    )


    fig.savefig(

        OUTDIR /
        f"{stem}.png",

        dpi=400,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )


# =============================================================================
# 14. CLUSTER-NUMBER DISTRIBUTION
# =============================================================================

make_distribution_plot("Pcluster", r"$P_{\rm cl}(s)$", "cluster_number_distribution", logy=True)


# =============================================================================
# 15. PARTICLE-WEIGHTED DISTRIBUTION
# =============================================================================

make_distribution_plot(

    "Pparticle",

    r"$P_{\rm p}(s)$",

    "particle_weighted_distribution",

    logy=True

)


# =============================================================================
# 16. COMBINED FIGURE
# =============================================================================

fig, axes = plt.subplots(

    2,
    len(T_SELECTED),

    figsize=(
        10.5,
        6.6
    ),

    sharex="col",

    sharey="row"

)


for col, T0 in enumerate(
    T_SELECTED
):

    sub = selected[
        np.isclose(
            selected["T"],
            T0,
            atol=1e-8
        )
    ]


    for region in [
        "before",
        "near",
        "after",
    ]:

        row = sub[
            sub["region"]
            ==
            region
        ]


        if len(row) == 0:

            continue


        row = row.iloc[0]


        dist = read_distribution(
            row["file"]
        )


        s = dist[
            "s"
        ]


        for row_index, ykey in enumerate(
            [
                "Pcluster",
                "Pparticle",
            ]
        ):

            y = dist[
                ykey
            ]


            mask = (
                np.isfinite(s)
                &
                np.isfinite(y)
                &
                (y > 0)
            )


            axes[
                row_index,
                col
            ].plot(

                s[
                    mask
                ],

                y[
                    mask
                ],

                marker=MARKERS[
                    region
                ],

                markersize=3.8,

                linewidth=1.4,

                color=COLORS[
                    region
                ],

                label=(
                    LABELS[
                        region
                    ]
                )

            )


    axes[
        0,
        col
    ].set_title(
        rf"$T^*={T0:.2f}$"
    )


    axes[
        1,
        col
    ].set_xlabel(
        r"$s$"
    )


    for row_index in [
        0,
        1
    ]:

        axes[
            row_index,
            col
        ].set_xscale(
            "log"
        )

        axes[
            row_index,
            col
        ].set_yscale(
            "log"
        )


axes[
    0,
    0
].set_ylabel(
    r"$P_{\rm cl}(s)$"
)


axes[
    1,
    0
].set_ylabel(
    r"$P_{\rm p}(s)$"
)


handles, labels = axes[0, 0].get_legend_handles_labels()


fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3)


fig.subplots_adjust(

    left=0.085,

    right=0.985,

    bottom=0.10,

    top=0.89,

    wspace=0.10,

    hspace=0.12

)


fig.savefig(

    OUTDIR /
    "cluster_distribution_combined.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "cluster_distribution_combined.png",

    dpi=400,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 17. FINAL
# =============================================================================

print()
print("=" * 90)
print("OUTPUT")
print("=" * 90)


for f in sorted(
    OUTDIR.iterdir()
):

    print(
        f
    )


print()
print("Done.")
