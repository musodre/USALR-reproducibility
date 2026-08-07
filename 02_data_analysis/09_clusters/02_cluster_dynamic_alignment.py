#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_dynamic_alignment.py

Alignment between cluster growth and the diffusion-anomaly boundary
for the USALR system.

Run from:
    USALR/python-inputs

Inputs
------
plots/cluster_analysis/cluster_state_summary.dat

plots/derivative_boundary_alignment/dynamic_boundaries.dat

Scientific questions
--------------------
1. Does the diffusion minimum coincide with rapid cluster growth?

2. What happens to:
       <s>_cl
       <s>_w
       s_max/N
       f_cl
   when crossing

       (d ln D*/dP*)_T = 0 ?

3. Does the cluster response distinguish the normal and anomalous
   diffusion regimes?

Outputs
-------
plots/cluster_dynamic_alignment/

    cluster_dynamic_master.dat
    cluster_response_at_boundaries.dat
    cluster_boundary_summary.dat

    cluster_observables_vs_pressure.pdf/png
    cluster_derivatives_vs_pressure.pdf/png
    cluster_boundary_alignment.pdf/png
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

from usalr_paths import DERIVED_DATA_ROOT

# =============================================================================
# 1. INPUT / OUTPUT
# =============================================================================

CLUSTER_FILE = DERIVED_DATA_ROOT / "cluster_analysis" / "cluster_state_summary.dat"

BOUNDARY_FILE = DERIVED_DATA_ROOT / "derivative_boundary_alignment" / "dynamic_boundaries.dat"

OUTDIR = DERIVED_DATA_ROOT / "cluster_dynamic_alignment"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

T_SELECTED = [
    0.26,
    0.30,
    0.40,
    0.50,
    0.60,
]


# Same local-derivative philosophy used previously

WINDOWS = (
    5,
    7,
    9,
)

MIN_R2 = 0.65

MIN_VALID_FITS = 2

CONSENSUS = 2.0 / 3.0

SIGMA_FACTOR = 1.0


# local interval around dynamic boundary

LOCAL_DP = 0.30


# =============================================================================
# 3. PUBLICATION PALETTE
# =============================================================================

COLORS = {

    0.26: "#4477AA",
    0.30: "#228833",
    0.40: "#CCBB44",
    0.50: "#EE6677",
    0.60: "#AA3377",

}


# =============================================================================
# 4. MATPLOTLIB
# =============================================================================

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 11,

    "axes.labelsize": 14,

    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,

    "legend.fontsize": 9.5,

    "axes.linewidth": 1.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "legend.frameon": False,

})


# =============================================================================
# 5. LOAD CLUSTER DATABASE
# =============================================================================

if not CLUSTER_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{CLUSTER_FILE.resolve()}\n"
    )


df = pd.read_csv(

    CLUSTER_FILE,

    sep=r"\s+",

    engine="python"

)


required = [

    "P",
    "T",

    "mean_cluster",
    "weighted_cluster",
    "largest_fraction",
    "clustered_fraction",

    "macrocluster_flag",

    "D",
    "lnD",

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
    [
        "T",
        "P"
    ]
).reset_index(
    drop=True
)


# =============================================================================
# 6. LOAD DYNAMIC BOUNDARIES
# =============================================================================

if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{BOUNDARY_FILE.resolve()}\n"
    )


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
        "P_boundary"
    ]
)


# =============================================================================
# 7. LOCAL WINDOWS
# =============================================================================

def get_local_window(
    n,
    i,
    window
):

    if n < window:

        return None


    half = window // 2

    start = (
        i - half
    )


    start = max(
        0,
        start
    )


    start = min(
        start,
        n - window
    )


    return np.arange(

        start,
        start + window,

        dtype=int

    )


# =============================================================================
# 8. LOCAL QUADRATIC FIT
# =============================================================================

def quadratic_derivative_fit(
    P,
    Y,
    i,
    window
):

    idx = get_local_window(

        len(P),

        i,

        window

    )


    if idx is None:

        return None


    x = np.asarray(
        P[idx],
        dtype=float
    )

    y = np.asarray(
        Y[idx],
        dtype=float
    )


    good = (

        np.isfinite(x)
        &
        np.isfinite(y)

    )


    x = x[good]
    y = y[good]


    if len(x) < 5:

        return None


    P0 = float(
        P[i]
    )


    xc = (
        x
        -
        P0
    )


    try:

        coeff, cov = np.polyfit(
            xc, y, 2, cov=True)

    except Exception:

        return None


    a, b, c = coeff


    predicted = np.polyval(
        coeff,
        xc
    )


    ss_res = np.sum(
        (
            y
            -
            predicted
        )**2
    )


    ss_tot = np.sum(
        (
            y
            -
            np.mean(y)
        )**2
    )


    if (
        not np.isfinite(ss_tot)
        or
        ss_tot <= 0
    ):

        return None


    R2 = (

        1.0
        -
        ss_res
        /
        ss_tot

    )


    if not np.isfinite(R2):

        return None


    if R2 < MIN_R2:

        return None


    sigma = np.sqrt(
        max(
            float(
                cov[1, 1]
            ),
            0.0
        )
    )


    return {

        "slope":
        float(b),

        "sigma":
        float(sigma),

        "R2":
        float(R2),

    }


# =============================================================================
# 9. MULTI-WINDOW DERIVATIVE
# =============================================================================

def evaluate_derivative(
    P,
    Y,
    i
):

    fits = []


    for window in WINDOWS:

        result = quadratic_derivative_fit(

            P,
            Y,
            i,
            window

        )


        if result is not None:

            fits.append(
                result
            )


    if len(fits) == 0:

        return {

            "derivative":
            np.nan,

            "sigma":
            np.nan,

            "R2":
            np.nan,

            "nfits":
            0,

            "sign":
            0,

            "class":
            "unresolved",

        }


    slopes = np.asarray(
        [
            f["slope"]
            for f in fits
        ],
        dtype=float
    )


    sigmas = np.asarray(
        [
            f["sigma"]
            for f in fits
        ],
        dtype=float
    )


    r2s = np.asarray(
        [
            f["R2"]
            for f in fits
        ],
        dtype=float
    )


    derivative = np.median(
        slopes
    )


    sigma = np.median(
        sigmas
    )


    median_R2 = np.median(
        r2s
    )


    signs = []


    for slope, error in zip(
        slopes,
        sigmas
    ):

        threshold = (

            SIGMA_FACTOR
            *
            error

        )


        if slope > threshold:

            signs.append(
                +1
            )


        elif slope < -threshold:

            signs.append(
                -1
            )


        else:

            signs.append(
                0
            )


    signs = np.asarray(
        signs,
        dtype=int
    )


    frac_positive = np.mean(
        signs == +1
    )


    frac_negative = np.mean(
        signs == -1
    )


    nfits = len(
        fits
    )


    if nfits < MIN_VALID_FITS:

        sign = 0

        classification = (
            "unresolved"
        )


    elif frac_positive >= CONSENSUS:

        sign = +1

        classification = (
            "positive"
        )


    elif frac_negative >= CONSENSUS:

        sign = -1

        classification = (
            "negative"
        )


    else:

        sign = 0

        classification = (
            "uncertain"
        )


    return {

        "derivative":
        float(
            derivative
        ),

        "sigma":
        float(
            sigma
        ),

        "R2":
        float(
            median_R2
        ),

        "nfits":
        int(
            nfits
        ),

        "sign":
        int(
            sign
        ),

        "class":
        classification,

    }


# =============================================================================
# 10. COMPUTE CLUSTER DERIVATIVES
# =============================================================================

OBSERVABLES = {

    "mean_cluster":
    "mean_cluster",

    "weighted_cluster":
    "weighted_cluster",

    "largest_fraction":
    "largest_fraction",

    "clustered_fraction":
    "clustered_fraction",

}


rows = []


for T, group in df.groupby(
    "T"
):

    group = (

        group
        .sort_values("P")
        .reset_index(
            drop=True
        )

    )


    P = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    for i in range(
        len(group)
    ):

        row = group.loc[
            i
        ].to_dict()


        for name, column in (
            OBSERVABLES.items()
        ):

            Y = group[
                column
            ].to_numpy(
                dtype=float
            )


            result = evaluate_derivative(

                P,
                Y,
                i

            )


            prefix = (
                f"d{name}_dP"
            )


            row[
                prefix
            ] = result[
                "derivative"
            ]


            row[
                prefix
                +
                "_sigma"
            ] = result[
                "sigma"
            ]


            row[
                prefix
                +
                "_R2"
            ] = result[
                "R2"
            ]


            row[
                prefix
                +
                "_nfits"
            ] = result[
                "nfits"
            ]


            row[
                prefix
                +
                "_sign"
            ] = result[
                "sign"
            ]


            row[
                prefix
                +
                "_class"
            ] = result[
                "class"
            ]


        rows.append(
            row
        )


master = pd.DataFrame(
    rows
)


# =============================================================================
# 11. DYNAMIC CLASS FROM EXISTING D(P)
# =============================================================================

master[
    "dynamic_class"
] = "unknown"


# Use each previously determined boundary.
#
# For these isotherms:
#
#     P < P_boundary  -> normal side
#     P > P_boundary  -> anomalous side
#
# because the observed transition is from decreasing D(P)
# to increasing D(P).

for _, b in boundaries.iterrows():

    T0 = b["T"]

    P0 = b[
        "P_boundary"
    ]


    maskT = np.isclose(
        master["T"],
        T0
    )


    master.loc[
        maskT
        &
        (
            master["P"]
            <
            P0
        ),
        "dynamic_class"
    ] = "normal"


    master.loc[
        maskT
        &
        (
            master["P"]
            >
            P0
        ),
        "dynamic_class"
    ] = "anomalous"


# =============================================================================
# 12. INTERPOLATION
# =============================================================================

def interpolate_value(
    P,
    Y,
    P0
):

    P = np.asarray(
        P,
        dtype=float
    )

    Y = np.asarray(
        Y,
        dtype=float
    )


    mask = (

        np.isfinite(P)
        &
        np.isfinite(Y)

    )


    P = P[mask]
    Y = Y[mask]


    if len(P) < 4:

        return np.nan


    order = np.argsort(
        P
    )


    P = P[order]
    Y = Y[order]


    Pu,index = np.unique(P,return_index=True)

    Yu = Y[index]


    if not (
        Pu.min()
        <=
        P0
        <=
        Pu.max()
    ):

        return np.nan


    try:

        f = PchipInterpolator(

            Pu,
            Yu,

            extrapolate=False

        )


        return float(
            f(P0)
        )


    except Exception:

        return np.nan


# =============================================================================
# 13. LEFT/RIGHT MEANS AROUND BOUNDARY
# =============================================================================

def side_mean(
    P,
    Y,
    P0,
    side
):

    P = np.asarray(
        P,
        dtype=float
    )

    Y = np.asarray(
        Y,
        dtype=float
    )


    if side == "left":

        mask = (

            (P >= P0 - LOCAL_DP)
            &
            (P < P0)

        )


    else:

        mask = (

            (P > P0)
            &
            (P <= P0 + LOCAL_DP)

        )


    mask &= np.isfinite(
        Y
    )


    if np.sum(
        mask
    ) == 0:

        return np.nan


    return float(
        np.mean(
            Y[mask]
        )
    )


# =============================================================================
# 14. RESPONSE AT DYNAMIC BOUNDARY
# =============================================================================

boundary_rows = []


for _, b in boundaries.iterrows():

    T0 = b[
        "T"
    ]

    P0 = b[
        "P_boundary"
    ]


    group = master[
        np.isclose(
            master["T"],
            T0
        )
    ].sort_values(
        "P"
    )


    if len(group) == 0:

        continue


    P = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    row = {

        "T":
        T0,

        "P_boundary":
        P0,

    }


    for name in OBSERVABLES:

        Y = group[
            name
        ].to_numpy(
            dtype=float
        )


        value = interpolate_value(

            P,
            Y,
            P0

        )


        left = side_mean(

            P,
            Y,
            P0,
            "left"

        )


        right = side_mean(

            P,
            Y,
            P0,
            "right"

        )


        row[
            f"{name}_boundary"
        ] = value


        row[
            f"{name}_left"
        ] = left


        row[
            f"{name}_right"
        ] = right


        row[
            f"delta_{name}"
        ] = (

            right
            -
            left

            if
            np.isfinite(left)
            and
            np.isfinite(right)

            else
            np.nan

        )


        # -------------------------------------------------------------
        # derivative at boundary
        # -------------------------------------------------------------

        dcolumn = (
            f"d{name}_dP"
        )


        dY = group[
            dcolumn
        ].to_numpy(
            dtype=float
        )


        row[
            f"{dcolumn}_boundary"
        ] = interpolate_value(

            P,
            dY,
            P0

        )


    boundary_rows.append(
        row
    )


boundary_response = pd.DataFrame(
    boundary_rows
)


# =============================================================================
# 15. GLOBAL SUMMARY AROUND BOUNDARY
# =============================================================================

summary_rows = []


print()
print("=" * 90)
print("CLUSTER RESPONSE ACROSS DYNAMIC BOUNDARY")
print("=" * 90)


for name in OBSERVABLES:

    left = boundary_response[
        f"{name}_left"
    ].to_numpy(
        dtype=float
    )


    right = boundary_response[
        f"{name}_right"
    ].to_numpy(
        dtype=float
    )


    mask = (

        np.isfinite(left)
        &
        np.isfinite(right)

    )


    if np.sum(
        mask
    ) == 0:

        continue


    delta = (

        right[mask]
        -
        left[mask]

    )


    print()
    print(
        name
    )

    print(
        f"N boundaries       = "
        f"{np.sum(mask)}"
    )

    print(
        f"median left        = "
        f"{np.median(left[mask]): .6e}"
    )

    print(
        f"median right       = "
        f"{np.median(right[mask]): .6e}"
    )

    print(
        f"median right-left  = "
        f"{np.median(delta): .6e}"
    )

    print(
        f"fraction increase  = "
        f"{np.mean(delta > 0):.3f}"
    )

    print(
        f"fraction decrease  = "
        f"{np.mean(delta < 0):.3f}"
    )


    summary_rows.append({

        "observable":
        name,

        "N":
        np.sum(mask),

        "median_left":
        np.median(
            left[mask]
        ),

        "median_right":
        np.median(
            right[mask]
        ),

        "median_delta":
        np.median(
            delta
        ),

        "fraction_increase":
        np.mean(
            delta > 0
        ),

        "fraction_decrease":
        np.mean(
            delta < 0
        ),

    })


summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# 16. MACROCLUSTER LOCATION
# =============================================================================

macro = master[
    master[
        "macrocluster_flag"
    ] == 1
]


print()
print("=" * 90)
print("MACROCLUSTER STATES")
print("=" * 90)

print(
    f"N macrocluster states = "
    f"{len(macro)}"
)


if len(macro) > 0:

    print(
        f"P range = "
        f"{macro['P'].min():.3f} -- "
        f"{macro['P'].max():.3f}"
    )

    print(
        f"T range = "
        f"{macro['T'].min():.3f} -- "
        f"{macro['T'].max():.3f}"
    )


    first_macro = (

        macro
        .groupby(
            "T"
        )["P"]
        .min()
        .reset_index(
            name="P_first_macrocluster"
        )

    )


    print()
    print(
        "First macrocluster pressure by T:"
    )

    print(
        first_macro.to_string(
            index=False
        )
    )


else:

    first_macro = pd.DataFrame(
        columns=[
            "T",
            "P_first_macrocluster"
        ]
    )


# =============================================================================
# 17. SAVE
# =============================================================================

master.to_csv(

    OUTDIR /
    "cluster_dynamic_master.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


boundary_response.to_csv(

    OUTDIR /
    "cluster_response_at_boundaries.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


summary.to_csv(

    OUTDIR /
    "cluster_boundary_summary.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


first_macro.to_csv(

    OUTDIR /
    "first_macrocluster_pressure.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


# =============================================================================
# 18. FIGURE — RAW CLUSTER OBSERVABLES
# =============================================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=(
        7.25,
        5.7
    ),

    sharex=True

)


plot_specs = [

    (
        axes[0, 0],

        "mean_cluster",

        r"$\langle s\rangle_{\rm cl}$",

        "(a)",
    ),

    (
        axes[0, 1],

        "weighted_cluster",

        r"$\langle s\rangle_{\rm w}$",

        "(b)",
    ),

    (
        axes[1, 0],

        "largest_fraction",

        r"$\langle s_{\max}/N\rangle$",

        "(c)",
    ),

    (
        axes[1, 1],

        "clustered_fraction",

        r"$f_{\rm cl}$",

        "(d)",
    ),

]


for ax, column, ylabel, label in plot_specs:

    for T in T_SELECTED:

        group = master[
            np.isclose(
                master["T"],
                T
            )
        ].sort_values(
            "P"
        )


        if len(group) == 0:

            continue


        ax.plot(

            group[
                "P"
            ],

            group[
                column
            ],

            marker="o",

            markersize=3.7,

            linewidth=1.6,

            color=COLORS[T],

            label=(
                rf"$T^*={T:.2f}$"
            )

        )


        b = boundaries[
            np.isclose(
                boundaries["T"],
                T
            )
        ]


        for P0 in b[
            "P_boundary"
        ]:

            y0 = interpolate_value(

                group["P"],
                group[column],
                P0

            )


            if np.isfinite(
                y0
            ):

                ax.plot(

                    P0,
                    y0,

                    marker="o",

                    markersize=7.5,

                    markerfacecolor="white",

                    markeredgecolor=COLORS[T],

                    markeredgewidth=1.4,

                    linestyle="None",

                    zorder=10

                )


    ax.set_ylabel(
        ylabel
    )


    ax.text(

        0.04,
        0.94,

        label,

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontweight="bold"

    )


axes[1, 0].set_xlabel(
    r"$P^*$"
)

axes[1, 1].set_xlabel(
    r"$P^*$"
)


handles, labels = axes[
    0,
    0
].get_legend_handles_labels()


fig.legend(handles, labels,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        0.995
    ),

    ncol=5

)


fig.subplots_adjust(

    left=0.105,

    right=0.985,

    bottom=0.10,

    top=0.90,

    wspace=0.28,

    hspace=0.16

)


fig.savefig(

    OUTDIR /
    "cluster_observables_vs_pressure.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "cluster_observables_vs_pressure.png",

    dpi=400,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 19. FIGURE — CLUSTER DERIVATIVES
# =============================================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=(
        7.25,
        5.7
    ),

    sharex=True

)


derivative_specs = [

    (
        axes[0, 0],

        "dmean_cluster_dP",

        r"$\partial_{P^*}\langle s\rangle_{\rm cl}$",

        "(a)",
    ),

    (
        axes[0, 1],

        "dweighted_cluster_dP",

        r"$\partial_{P^*}\langle s\rangle_{\rm w}$",

        "(b)",
    ),

    (
        axes[1, 0],

        "dlargest_fraction_dP",

        r"$\partial_{P^*}\langle s_{\max}/N\rangle$",

        "(c)",
    ),

    (
        axes[1, 1],

        "dclustered_fraction_dP",

        r"$\partial_{P^*}f_{\rm cl}$",

        "(d)",
    ),

]


for ax, column, ylabel, label in derivative_specs:

    for T in T_SELECTED:

        group = master[
            np.isclose(
                master["T"],
                T
            )
        ].sort_values(
            "P"
        )


        if len(group) == 0:

            continue


        ax.plot(

            group["P"],

            group[column],

            marker="o",

            markersize=3.3,

            linewidth=1.5,

            color=COLORS[T]

        )


    ax.axhline(

        0.0,

        linestyle="--",

        linewidth=0.8

    )


    ax.set_ylabel(
        ylabel
    )


    ax.text(

        0.04,
        0.94,

        label,

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontweight="bold"

    )


axes[1, 0].set_xlabel(
    r"$P^*$"
)

axes[1, 1].set_xlabel(
    r"$P^*$"
)


fig.subplots_adjust(

    left=0.105,

    right=0.985,

    bottom=0.10,

    top=0.97,

    wspace=0.28,

    hspace=0.16

)


fig.savefig(

    OUTDIR /
    "cluster_derivatives_vs_pressure.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "cluster_derivatives_vs_pressure.png",

    dpi=400,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 20. FINAL
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
