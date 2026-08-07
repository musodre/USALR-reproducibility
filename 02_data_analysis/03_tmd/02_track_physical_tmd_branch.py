#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import DERIVED_DATA_ROOT

"""
Conservative TMD branch tracking
================================

INPUT
-----
thermo_response/TMD_local_quadratic/
    TMD_local_quadratic_all.dat

This script DOES NOT recalculate density maxima.

It uses the pressure-by-pressure local quadratic results already
obtained and classifies them as:

    validated
        robust local maximum; used as an anchor

    branch_supported
        weaker local maximum, but quantitatively compatible with
        the branch defined by neighboring validated maxima

    disconnected
        mathematically possible local maximum, but not compatible
        with the physical TMD branch

    rejected
        no acceptable local density maximum

Important methodological point
------------------------------
There is NO greedy branch construction.

Candidates between two validated anchors are tested independently
against the interpolation defined by those anchors.

Candidates beyond the final validated anchor may be accepted only
over a SHORT pressure extension and only if they agree with a local
regression through the last validated anchors.

Therefore, rejecting one candidate cannot cause a cascade of
rejections at higher pressure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

INPUT = (
    DERIVED_DATA_ROOT / "thermo_response" / "TMD_local_quadratic"
    / "TMD_local_quadratic_all.dat"
)

OUTDIR = DERIVED_DATA_ROOT / "thermo_response" / "TMD_branch_tracking"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_ALL = (
    OUTDIR
    /
    "TMD_branch_classification.dat"
)

OUT_BRANCH = (
    OUTDIR
    /
    "TMD_physical_branch.dat"
)

OUT_ANCHORS = (
    OUTDIR
    /
    "TMD_validated_anchors.dat"
)

OUT_PDF = (
    OUTDIR
    /
    "TMD_physical_branch.pdf"
)

OUT_PNG = (
    OUTDIR
    /
    "TMD_physical_branch.png"
)


# ============================================================
# 2. LOCAL QUALITY REQUIREMENTS FOR CANDIDATES
#
# These criteria DO NOT create a branch.
# They only decide whether a locally weaker maximum is good
# enough to be tested against the established branch.
# ============================================================

CANDIDATE_MIN_R2 = 0.60

CANDIDATE_MIN_SUCCESS = 2

CANDIDATE_MAX_STD = 0.025

CANDIDATE_MAX_SPREAD = 0.055


# ============================================================
# 3. INTERIOR BRANCH SUPPORT
#
# If a candidate lies between two validated anchors,
#
#      (P1,T1) ------------ candidate ------------ (P2,T2)
#
# its expected TMD temperature is obtained by interpolation.
# ============================================================

# Do not bridge two validated anchors separated by an
# excessively large pressure interval.
MAX_ANCHOR_GAP = 0.50


# Base tolerance in T*
BRANCH_BASE_TOL = 0.015


# Additional tolerance proportional to uncertainty.
SIGMA_FACTOR = 1.50


# Absolute upper limit. Even a very uncertain candidate cannot
# be accepted if it lies farther than this from the local branch.
MAX_BRANCH_RESIDUAL = 0.030


# ============================================================
# 4. TERMINAL EXTENSION
#
# After the final strict validated anchor, allow only a SHORT
# continuation.
#
# This is useful if the density maximum becomes broader and
# therefore fails the strict criterion near the end of the TMD.
#
# Crucially, this prevents isolated maxima at P* ~ 2.2--3.0
# from being interpreted as continuation of the TMD.
# ============================================================

MAX_HIGH_P_EXTENSION = 0.40

MAX_LOW_P_EXTENSION = 0.20


# Number of validated anchors used to estimate the local
# terminal trend.
N_ANCHORS_FOR_EXTRAPOLATION = 3


# Maximum residual from the terminal regression.
END_BASE_TOL = 0.018

MAX_END_RESIDUAL = 0.030


# ============================================================
# 5. OPTIONAL PHYSICAL TREND CHECK
#
# We do NOT force monotonicity globally.
#
# This only prevents a terminal extrapolation from suddenly
# reversing strongly relative to the validated branch.
# ============================================================

MAX_BACKWARD_DEVIATION = 0.020


# ============================================================
# 6. PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 13,

    "axes.labelsize": 18,

    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    "legend.fontsize": 10.5,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


# ============================================================
# 7. LOAD LOCAL-QUADRATIC RESULTS
# ============================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT.resolve()}"
    )


df = pd.read_csv(
    INPUT,
    sep=r"\s+"
)


required = {
    "P",
    "status",
    "T_TMD",
    "T_std",
    "T_spread",
    "R2_median",
    "n_success",
}


missing = (
    required
    -
    set(df.columns)
)


if missing:

    raise KeyError(
        "\nMissing required columns:\n"
        f"{sorted(missing)}"
    )


df = df.sort_values(
    "P"
).reset_index(
    drop=True
)


print()
print("=" * 88)
print("INPUT")
print("=" * 88)

print(
    f"States/pressures = {len(df)}"
)

print(
    f"P range = "
    f"{df['P'].min():.3f} -- "
    f"{df['P'].max():.3f}"
)


# ============================================================
# 8. INITIAL FINAL CLASS
# ============================================================

df["final_class"] = "rejected"

df["branch_method"] = "none"

df["T_expected"] = np.nan

df["branch_residual"] = np.nan

df["branch_tolerance"] = np.nan


# ============================================================
# 9. STRICT VALIDATED POINTS = ANCHORS
# ============================================================

anchor_mask = (
    df["status"]
    ==
    "validated"
)


df.loc[
    anchor_mask,
    "final_class"
] = "validated"


df.loc[
    anchor_mask,
    "branch_method"
] = "local_strict"


anchors = df[
    anchor_mask
].copy()


anchors = anchors[
    np.isfinite(
        anchors["T_TMD"]
    )
].copy()


anchors = anchors.sort_values(
    "P"
).reset_index(
    drop=True
)


if len(anchors) < 2:

    raise RuntimeError(
        "Fewer than two validated TMD anchors were found."
    )


print()
print("=" * 88)
print("STRICT VALIDATED ANCHORS")
print("=" * 88)

print(
    anchors[
        [
            "P",
            "T_TMD",
            "T_std",
            "R2_median",
            "n_success",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. CANDIDATE LOCAL-QUALITY FILTER
# ============================================================

def locally_usable_candidate(row):

    if row["status"] != "candidate":

        return False


    values = [
        row["T_TMD"],
        row["T_std"],
        row["T_spread"],
        row["R2_median"],
        row["n_success"],
    ]


    if not all(
        np.isfinite(values)
    ):

        return False


    if (
        row["R2_median"]
        <
        CANDIDATE_MIN_R2
    ):

        return False


    if (
        row["n_success"]
        <
        CANDIDATE_MIN_SUCCESS
    ):

        return False


    if (
        row["T_std"]
        >
        CANDIDATE_MAX_STD
    ):

        return False


    if (
        row["T_spread"]
        >
        CANDIDATE_MAX_SPREAD
    ):

        return False


    return True


candidate_usable = df.apply(
    locally_usable_candidate,
    axis=1
)


# Candidates failing even this relaxed local test remain rejected
df.loc[
    (
        (df["status"] == "candidate")
        &
        (~candidate_usable)
    ),
    "final_class"
] = "disconnected"


df.loc[
    (
        (df["status"] == "candidate")
        &
        (~candidate_usable)
    ),
    "branch_method"
] = "local_quality_failed"


# ============================================================
# 11. HELPER: COMBINED TOLERANCE
# ============================================================

def branch_tolerance(
    sigma_candidate,
    sigma_reference
):

    if not np.isfinite(
        sigma_candidate
    ):

        sigma_candidate = 0.0


    if not np.isfinite(
        sigma_reference
    ):

        sigma_reference = 0.0


    uncertainty = np.sqrt(
        sigma_candidate**2
        +
        sigma_reference**2
    )


    tolerance = (
        BRANCH_BASE_TOL
        +
        SIGMA_FACTOR*uncertainty
    )


    return min(
        tolerance,
        MAX_BRANCH_RESIDUAL
    )


# ============================================================
# 12. INTERPOLATED REFERENCE BETWEEN TWO ANCHORS
# ============================================================

def interpolate_between_anchors(
    P,
    left,
    right
):

    P1 = float(
        left["P"]
    )

    P2 = float(
        right["P"]
    )


    T1 = float(
        left["T_TMD"]
    )

    T2 = float(
        right["T_TMD"]
    )


    if P2 <= P1:

        return None


    if (
        P2-P1
        >
        MAX_ANCHOR_GAP
    ):

        return None


    fraction = (
        (P-P1)
        /
        (P2-P1)
    )


    T_expected = (
        T1
        +
        fraction*(T2-T1)
    )


    # --------------------------------------------------------
    # Interpolate anchor uncertainty conservatively
    # --------------------------------------------------------

    s1 = (
        float(left["T_std"])
        if np.isfinite(left["T_std"])
        else 0.0
    )


    s2 = (
        float(right["T_std"])
        if np.isfinite(right["T_std"])
        else 0.0
    )


    sigma_ref = np.sqrt(
        (
            (1.0-fraction)*s1
        )**2
        +
        (
            fraction*s2
        )**2
    )


    return (
        T_expected,
        sigma_ref
    )


# ============================================================
# 13. CLASSIFY INTERIOR CANDIDATES
#
# This is NOT greedy:
#
# every candidate is compared directly with the two validated
# anchors surrounding it.
# ============================================================

for idx, row in df.iterrows():

    if not candidate_usable.iloc[idx]:

        continue


    P = float(
        row["P"]
    )


    T = float(
        row["T_TMD"]
    )


    lower = anchors[
        anchors["P"]
        <
        P
    ]


    upper = anchors[
        anchors["P"]
        >
        P
    ]


    # Not interior
    if (
        len(lower) == 0
        or
        len(upper) == 0
    ):

        continue


    left = lower.iloc[-1]

    right = upper.iloc[0]


    reference = interpolate_between_anchors(
        P,
        left,
        right
    )


    if reference is None:

        df.loc[
            idx,
            "final_class"
        ] = "disconnected"

        df.loc[
            idx,
            "branch_method"
        ] = "anchor_gap_too_large"

        continue


    T_expected, sigma_ref = (
        reference
    )


    tolerance = branch_tolerance(
        float(
            row["T_std"]
        ),
        sigma_ref
    )


    residual = abs(
        T-T_expected
    )


    df.loc[
        idx,
        "T_expected"
    ] = T_expected


    df.loc[
        idx,
        "branch_residual"
    ] = residual


    df.loc[
        idx,
        "branch_tolerance"
    ] = tolerance


    if residual <= tolerance:

        df.loc[
            idx,
            "final_class"
        ] = "branch_supported"

        df.loc[
            idx,
            "branch_method"
        ] = "interpolated_between_anchors"


    else:

        df.loc[
            idx,
            "final_class"
        ] = "disconnected"

        df.loc[
            idx,
            "branch_method"
        ] = "interpolation_residual_too_large"


# ============================================================
# 14. TERMINAL REGRESSION
# ============================================================

def terminal_regression(
    anchor_table,
    side="high"
):

    if len(anchor_table) < 2:

        return None


    if side == "high":

        use = anchor_table.tail(
            min(
                N_ANCHORS_FOR_EXTRAPOLATION,
                len(anchor_table)
            )
        )


    else:

        use = anchor_table.head(
            min(
                N_ANCHORS_FOR_EXTRAPOLATION,
                len(anchor_table)
            )
        )


    P = use["P"].values.astype(
        float
    )

    T = use["T_TMD"].values.astype(
        float
    )


    if len(P) < 2:

        return None


    # --------------------------------------------------------
    # Weighted fit when meaningful uncertainties exist
    # --------------------------------------------------------

    sigma = use[
        "T_std"
    ].values.astype(float)


    good_sigma = (
        np.isfinite(sigma)
        &
        (sigma > 1.0e-6)
    )


    if np.sum(
        good_sigma
    ) >= 2:

        weights = np.ones(
            len(P)
        )


        weights[
            good_sigma
        ] = (
            1.0
            /
            sigma[
                good_sigma
            ]
        )


        coeff = np.polyfit(
            P,
            T,
            1,
            w=weights
        )


    else:

        coeff = np.polyfit(
            P,
            T,
            1
        )


    slope = float(
        coeff[0]
    )


    intercept = float(
        coeff[1]
    )


    fitted = np.polyval(
        coeff,
        P
    )


    scatter = float(
        np.sqrt(
            np.mean(
                (T-fitted)**2
            )
        )
    )


    return {
        "slope": slope,
        "intercept": intercept,
        "scatter": scatter,
        "Pmin": float(P.min()),
        "Pmax": float(P.max()),
        "T_terminal":
            float(
                T[-1]
                if side == "high"
                else T[0]
            ),
    }


high_model = terminal_regression(
    anchors,
    side="high"
)


low_model = terminal_regression(
    anchors,
    side="low"
)


# ============================================================
# 15. TERMINAL TOLERANCE
# ============================================================

def terminal_tolerance(
    sigma_candidate,
    model_scatter
):

    if not np.isfinite(
        sigma_candidate
    ):

        sigma_candidate = 0.0


    if not np.isfinite(
        model_scatter
    ):

        model_scatter = 0.0


    uncertainty = np.sqrt(
        sigma_candidate**2
        +
        model_scatter**2
    )


    tol = (
        END_BASE_TOL
        +
        SIGMA_FACTOR*uncertainty
    )


    return min(
        tol,
        MAX_END_RESIDUAL
    )


# ============================================================
# 16. HIGH-P TERMINAL CANDIDATES
# ============================================================

P_last_anchor = float(
    anchors["P"].max()
)


T_last_anchor = float(
    anchors.loc[
        anchors["P"].idxmax(),
        "T_TMD"
    ]
)


if high_model is not None:

    for idx, row in df.iterrows():

        if not candidate_usable.iloc[idx]:

            continue


        if (
            df.loc[
                idx,
                "final_class"
            ]
            ==
            "branch_supported"
        ):

            continue


        P = float(
            row["P"]
        )


        T = float(
            row["T_TMD"]
        )


        if P <= P_last_anchor:

            continue


        dP = (
            P
            -
            P_last_anchor
        )


        if (
            dP
            >
            MAX_HIGH_P_EXTENSION
        ):

            df.loc[
                idx,
                "final_class"
            ] = "disconnected"

            df.loc[
                idx,
                "branch_method"
            ] = "beyond_high_P_extension"

            continue


        T_expected = (
            high_model["slope"]*P
            +
            high_model["intercept"]
        )


        residual = abs(
            T-T_expected
        )


        tolerance = terminal_tolerance(
            float(
                row["T_std"]
            ),
            high_model["scatter"]
        )


        # ----------------------------------------------------
        # Prevent abrupt backwards turn at the end of branch.
        # ----------------------------------------------------

        backward = (
            T
            <
            T_last_anchor
            -
            MAX_BACKWARD_DEVIATION
        )


        df.loc[
            idx,
            "T_expected"
        ] = T_expected


        df.loc[
            idx,
            "branch_residual"
        ] = residual


        df.loc[
            idx,
            "branch_tolerance"
        ] = tolerance


        if (
            residual <= tolerance
            and
            not backward
        ):

            df.loc[
                idx,
                "final_class"
            ] = "branch_supported"

            df.loc[
                idx,
                "branch_method"
            ] = "short_high_P_extrapolation"


        else:

            df.loc[
                idx,
                "final_class"
            ] = "disconnected"

            df.loc[
                idx,
                "branch_method"
            ] = "high_P_extrapolation_failed"


# ============================================================
# 17. LOW-P TERMINAL CANDIDATES
# ============================================================

P_first_anchor = float(
    anchors["P"].min()
)


T_first_anchor = float(
    anchors.loc[
        anchors["P"].idxmin(),
        "T_TMD"
    ]
)


if low_model is not None:

    for idx, row in df.iterrows():

        if not candidate_usable.iloc[idx]:

            continue


        if (
            df.loc[
                idx,
                "final_class"
            ]
            ==
            "branch_supported"
        ):

            continue


        P = float(
            row["P"]
        )


        T = float(
            row["T_TMD"]
        )


        if P >= P_first_anchor:

            continue


        dP = (
            P_first_anchor
            -
            P
        )


        if (
            dP
            >
            MAX_LOW_P_EXTENSION
        ):

            df.loc[
                idx,
                "final_class"
            ] = "disconnected"

            df.loc[
                idx,
                "branch_method"
            ] = "beyond_low_P_extension"

            continue


        T_expected = (
            low_model["slope"]*P
            +
            low_model["intercept"]
        )


        residual = abs(
            T-T_expected
        )


        tolerance = terminal_tolerance(
            float(
                row["T_std"]
            ),
            low_model["scatter"]
        )


        df.loc[
            idx,
            "T_expected"
        ] = T_expected


        df.loc[
            idx,
            "branch_residual"
        ] = residual


        df.loc[
            idx,
            "branch_tolerance"
        ] = tolerance


        if residual <= tolerance:

            df.loc[
                idx,
                "final_class"
            ] = "branch_supported"

            df.loc[
                idx,
                "branch_method"
            ] = "short_low_P_extrapolation"


        else:

            df.loc[
                idx,
                "final_class"
            ] = "disconnected"

            df.loc[
                idx,
                "branch_method"
            ] = "low_P_extrapolation_failed"


# ============================================================
# 18. ANY REMAINING USABLE CANDIDATE = DISCONNECTED
# ============================================================

remaining = (
    candidate_usable
    &
    (
        df["final_class"]
        ==
        "rejected"
    )
)


df.loc[
    remaining,
    "final_class"
] = "disconnected"


df.loc[
    remaining,
    "branch_method"
] = "no_branch_support"


# ============================================================
# 19. PHYSICAL BRANCH
# ============================================================

physical_branch = df[
    df["final_class"].isin(
        [
            "validated",
            "branch_supported",
        ]
    )
].copy()


physical_branch = physical_branch.sort_values(
    "P"
).reset_index(
    drop=True
)


# ============================================================
# 20. SAVE OUTPUT
# ============================================================

df.to_csv(
    OUT_ALL,
    sep=" ",
    index=False,
    float_format="%.10g"
)


physical_branch.to_csv(
    OUT_BRANCH,
    sep=" ",
    index=False,
    float_format="%.10g"
)


anchors.to_csv(
    OUT_ANCHORS,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 21. PRINT FULL CLASSIFICATION
# ============================================================

print()
print("=" * 120)
print("FINAL BRANCH CLASSIFICATION")
print("=" * 120)


columns_to_print = [
    "P",
    "status",
    "final_class",
    "T_TMD",
    "T_std",
    "R2_median",
    "n_success",
    "T_expected",
    "branch_residual",
    "branch_tolerance",
    "branch_method",
]


print(
    df[
        columns_to_print
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print()
print("=" * 88)
print("SUMMARY")
print("=" * 88)


print(
    f"Strict validated anchors : "
    f"{np.sum(df['final_class']=='validated')}"
)


print(
    f"Branch-supported points  : "
    f"{np.sum(df['final_class']=='branch_supported')}"
)


print(
    f"Disconnected candidates  : "
    f"{np.sum(df['final_class']=='disconnected')}"
)


print(
    f"Final branch points       : "
    f"{len(physical_branch)}"
)


if len(physical_branch) > 0:

    print(
        f"\nFinal TMD pressure range:"
        f"\n  {physical_branch['P'].min():.3f}"
        f" <= P* <= "
        f"{physical_branch['P'].max():.3f}"
    )


    print(
        f"\nFinal TMD temperature range:"
        f"\n  {physical_branch['T_TMD'].min():.4f}"
        f" <= T_TMD* <= "
        f"{physical_branch['T_TMD'].max():.4f}"
    )


# ============================================================
# 22. FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.6)
)


# ------------------------------------------------------------
# Disconnected candidates
# ------------------------------------------------------------

disconnected = df[
    (
        df["final_class"]
        ==
        "disconnected"
    )
    &
    np.isfinite(
        df["T_TMD"]
    )
]


if len(disconnected) > 0:

    ax.scatter(
        disconnected["T_TMD"],
        disconnected["P"],

        s=42,

        marker="x",

        color="0.70",

        linewidths=1.1,

        label="disconnected"
    )


# ------------------------------------------------------------
# Branch-supported candidates
# ------------------------------------------------------------

supported = physical_branch[
    physical_branch["final_class"]
    ==
    "branch_supported"
]


if len(supported) > 0:

    ax.errorbar(
        supported["T_TMD"],
        supported["P"],

        xerr=supported["T_std"],

        fmt="s",

        ms=5.5,

        mfc="white",

        mec="tab:orange",

        ecolor="tab:orange",

        capsize=3,

        linestyle="none",

        label="branch-supported"
    )


# ------------------------------------------------------------
# Strict anchors
# ------------------------------------------------------------

strict = physical_branch[
    physical_branch["final_class"]
    ==
    "validated"
]


if len(strict) > 0:

    ax.errorbar(
        strict["T_TMD"],
        strict["P"],

        xerr=strict["T_std"],

        fmt="o",

        ms=6.0,

        capsize=3,

        label="validated"
    )


# ------------------------------------------------------------
# Connect only physical branch
# ------------------------------------------------------------

if len(physical_branch) >= 2:

    branch = physical_branch.sort_values(
        "P"
    )


    ax.plot(
        branch["T_TMD"],
        branch["P"],

        color="black",

        lw=1.35,

        alpha=0.72,

        zorder=1
    )


ax.set_xlabel(
    r"$T_{\mathrm{TMD}}^*$"
)


ax.set_ylabel(
    r"$P^*$"
)


ax.grid(
    alpha=0.12
)


ax.legend(
    frameon=False
)


fig.tight_layout()


fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)


fig.savefig(
    OUT_PNG,
    dpi=500,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. REPORT
# ============================================================

print()
print("=" * 88)
print("OUTPUT")
print("=" * 88)


print(
    f"Complete classification:\n"
    f"  {OUT_ALL}"
)


print(
    f"\nPhysical TMD branch:\n"
    f"  {OUT_BRANCH}"
)


print(
    f"\nStrict anchors:\n"
    f"  {OUT_ANCHORS}"
)


print(
    f"\nFigure:\n"
    f"  {OUT_PDF}"
)


print()
print(
    "For the phase diagram use:"
)

print(
    f"  {OUT_BRANCH}"
)
