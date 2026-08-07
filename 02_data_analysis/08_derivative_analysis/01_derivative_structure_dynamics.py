#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
derivative_structure_dynamics.py

Pressure-derivative analysis for the USALR manuscript.

Run from:
    USALR/python-inputs

Input:
    plots/final_author_table/author_master_table_common.dat

Expected columns:
    T P rho D lnD Rg Rn A_SALR s2 minus_s2 tau

Scientific question
-------------------
Do the pressure derivatives of the structural descriptors distinguish
the diffusion-anomalous region,

    (dD/dP)_T > 0,

from the normal region,

    (dD/dP)_T < 0 ?

Quantities analyzed
-------------------
    dD/dP
    dlnD/dP
    dRg/dP
    dRn/dP
    dA_SALR/dP

with

    Rg = g2/g1
    Rn = n2/n1

Local derivatives are obtained from independent quadratic regressions
using 5, 7 and 9 neighboring pressure points.

The final derivative at each state is the median among the valid local
estimates. The sign is considered resolved when at least two valid
windows are available and at least 2/3 of them agree significantly
on the sign.

The dynamic classification uses dlnD/dP. Since D > 0,

    sign(dlnD/dP) = sign(dD/dP).

Outputs:
    plots/derivative_structure_dynamics/
"""

from pathlib import Path
import sys

# =============================================================================
# REPOSITORY BOOTSTRAP
# =============================================================================

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        if str(_repo_parent) not in sys.path:
            sys.path.insert(0, str(_repo_parent))
        break
else:
    raise RuntimeError(
        "Could not locate USALR repository root containing usalr_paths.py"
    )

from usalr_paths import DERIVED_DATA_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    pearsonr,
    spearmanr,
    mannwhitneyu,
    rankdata,
)


# =============================================================================
# 1. INPUT / OUTPUT
# =============================================================================

INPUT = (
    DERIVED_DATA_ROOT
    / "final_author_table"
    / "author_master_table_common.dat"
)

OUTDIR = (
    DERIVED_DATA_ROOT
    / "derivative_structure_dynamics"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

WINDOWS = (5, 7, 9)

MIN_R2 = 0.65

SIGMA_FACTOR = 1.0

MIN_VALID_FITS = 2

CONSENSUS = 2.0 / 3.0

T_SELECTED = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]

VARIABLES = [
    "D",
    "lnD",
    "Rg",
    "Rn",
    "A_SALR",
]


# =============================================================================
# 3. MATPLOTLIB
# =============================================================================

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 12,

    "axes.labelsize": 16,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 10,

    "axes.linewidth": 1.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


# =============================================================================
# 4. LOAD MASTER TABLE
# =============================================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"\nCould not find input file:\n"
        f"{INPUT.resolve()}\n"
    )


df = pd.read_csv(
    INPUT,
    sep=r"\s+",
    engine="python"
)


required = [
    "T",
    "P",
    "rho",
    "D",
    "lnD",
    "Rg",
    "Rn",
    "A_SALR",
]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise RuntimeError(
        "\nMissing required columns:\n"
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


df = df.dropna(
    subset=required
)


df = (
    df
    .groupby(
        ["T", "P"],
        as_index=False
    )
    .mean(
        numeric_only=True
    )
)


df = df.sort_values(
    ["T", "P"]
).reset_index(
    drop=True
)


print()
print("=" * 88)
print("INPUT")
print("=" * 88)

print(
    f"N states = {len(df)}"
)

print(
    f"N isotherms = {df['T'].nunique()}"
)

print(
    f"T range  = "
    f"{df['T'].min():.3f} -- "
    f"{df['T'].max():.3f}"
)

print(
    f"P range  = "
    f"{df['P'].min():.3f} -- "
    f"{df['P'].max():.3f}"
)

print()
print("States per isotherm:")

print(
    df.groupby("T")
    .size()
    .to_string()
)


# =============================================================================
# 5. LOCAL WINDOW
# =============================================================================

def get_local_window(
    n,
    i,
    window
):

    """
    Return a contiguous local window of exactly `window` state points.

    The selected state i is kept as close as possible to the center.
    At the boundaries, the window is shifted rather than shortened.
    """

    if n < window:

        return None


    half = window // 2

    start = i - half

    start = max(
        0,
        start
    )

    start = min(
        start,
        n - window
    )

    stop = (
        start
        +
        window
    )


    return np.arange(
        start,
        stop,
        dtype=int
    )


# =============================================================================
# 6. SINGLE-WINDOW QUADRATIC DERIVATIVE
# =============================================================================

def quadratic_derivative_fit(
    P,
    Y,
    i,
    window
):

    """
    Local quadratic regression:

        Y(P) = a(P-P0)^2 + b(P-P0) + c

    where

        P0 = P[i].

    Therefore

        dY/dP | P0 = b.

    Returns
    -------
    dict or None
        slope
        sigma
        R2
        window
    """

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


    if len(
        np.unique(x)
    ) < 3:

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

            xc,
            y,

            deg=2,

            cov=True

        )

    except Exception:

        return None


    a, b, c = coeff


    predicted = np.polyval(
        coeff,
        xc
    )


    ss_res = np.sum(
        (y - predicted)**2
    )


    ss_tot = np.sum(
        (y - np.mean(y))**2
    )


    if (
        not np.isfinite(ss_tot)
        or
        ss_tot <= 0.0
    ):

        return None


    R2 = (
        1.0
        -
        ss_res / ss_tot
    )


    if not np.isfinite(R2):

        return None


    if R2 < MIN_R2:

        return None


    try:

        variance_b = float(
            cov[1, 1]
        )


        if variance_b < 0:

            variance_b = 0.0


        sigma_b = np.sqrt(
            variance_b
        )


    except Exception:

        sigma_b = np.nan


    if not np.isfinite(b):

        return None


    return {

        "slope": float(b),

        "sigma": float(sigma_b)
        if np.isfinite(sigma_b)
        else np.nan,

        "R2": float(R2),

        "window": int(window),

    }


# =============================================================================
# 7. MULTI-WINDOW DERIVATIVE
# =============================================================================

def evaluate_derivative(
    P,
    Y,
    i
):

    """
    Estimate derivative from independent local fits using WINDOWS.

    The derivative itself is the median slope among all valid fits.

    A sign is considered resolved only if:
        - at least MIN_VALID_FITS fits survive;
        - each individual sign is significant relative to its slope error;
        - at least CONSENSUS of valid fits agree in sign.
    """

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


    # -------------------------------------------------------------------------
    # No valid fits
    # -------------------------------------------------------------------------

    if len(fits) == 0:

        return {

            "derivative": np.nan,

            "sigma": np.nan,

            "spread": np.nan,

            "median_R2": np.nan,

            "nfits": 0,

            "sign": 0,

            "class": "unresolved",

            "frac_positive": np.nan,

            "frac_negative": np.nan,

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


    r2_values = np.asarray(

        [
            f["R2"]
            for f in fits
        ],

        dtype=float

    )


    derivative = float(
        np.nanmedian(
            slopes
        )
    )


    if np.any(
        np.isfinite(sigmas)
    ):

        sigma = float(
            np.nanmedian(
                sigmas
            )
        )

    else:

        sigma = np.nan


    if len(slopes) >= 2:

        spread = float(
            np.nanstd(
                slopes
            )
        )

    else:

        spread = 0.0


    median_R2 = float(
        np.nanmedian(
            r2_values
        )
    )


    nfits = int(
        len(fits)
    )


    # -------------------------------------------------------------------------
    # Resolve sign independently in each window
    # -------------------------------------------------------------------------

    signs = []


    for slope, sigma_local in zip(
        slopes,
        sigmas
    ):

        if not np.isfinite(
            sigma_local
        ):

            signs.append(0)

            continue


        threshold = (
            SIGMA_FACTOR
            *
            sigma_local
        )


        if slope > threshold:

            signs.append(+1)


        elif slope < -threshold:

            signs.append(-1)


        else:

            signs.append(0)


    signs = np.asarray(
        signs,
        dtype=int
    )


    frac_positive = float(
        np.mean(
            signs == +1
        )
    )


    frac_negative = float(
        np.mean(
            signs == -1
        )
    )


    # -------------------------------------------------------------------------
    # Final consensus
    # -------------------------------------------------------------------------

    if nfits < MIN_VALID_FITS:

        sign_final = 0

        classification = (
            "unresolved"
        )


    elif (
        frac_positive
        >=
        CONSENSUS
    ):

        sign_final = +1

        classification = (
            "positive"
        )


    elif (
        frac_negative
        >=
        CONSENSUS
    ):

        sign_final = -1

        classification = (
            "negative"
        )


    else:

        sign_final = 0

        classification = (
            "uncertain"
        )


    return {

        "derivative": derivative,

        "sigma": sigma,

        "spread": spread,

        "median_R2": median_R2,

        "nfits": nfits,

        "sign": int(
            sign_final
        ),

        "class": classification,

        "frac_positive": frac_positive,

        "frac_negative": frac_negative,

    }


# =============================================================================
# 8. COMPUTE DERIVATIVES ISOTHERM BY ISOTHERM
# =============================================================================

rows = []


for T, group in df.groupby(
    "T"
):

    group = (
        group
        .sort_values("P")
        .reset_index(drop=True)
    )


    P = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    values = {

        var:
        group[var]
        .to_numpy(
            dtype=float
        )

        for var in VARIABLES

    }


    for i in range(
        len(group)
    ):

        row = {

            "T": float(T),

            "P": float(
                group.loc[
                    i,
                    "P"
                ]
            ),

            "rho": float(
                group.loc[
                    i,
                    "rho"
                ]
            ),

        }


        # ---------------------------------------------------------------------
        # Original observables
        # ---------------------------------------------------------------------

        for var in VARIABLES:

            row[var] = float(
                group.loc[
                    i,
                    var
                ]
            )


        # ---------------------------------------------------------------------
        # Derivatives
        # ---------------------------------------------------------------------

        for var in VARIABLES:

            result = (
                evaluate_derivative(

                    P,
                    values[var],
                    i

                )
            )


            prefix = (
                f"d{var}_dP"
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
                "_spread"
            ] = result[
                "spread"
            ]


            row[
                prefix
                +
                "_R2"
            ] = result[
                "median_R2"
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


            row[
                prefix
                +
                "_frac_positive"
            ] = result[
                "frac_positive"
            ]


            row[
                prefix
                +
                "_frac_negative"
            ] = result[
                "frac_negative"
            ]


        rows.append(
            row
        )


deriv = pd.DataFrame(
    rows
)


# =============================================================================
# 9. SANITY CHECK
# =============================================================================

print()
print("=" * 88)
print("DERIVATIVE SANITY CHECK")
print("=" * 88)


for var in [

    "D",
    "lnD",
    "Rg",
    "Rn",
    "A_SALR",

]:

    dcol = (
        f"d{var}_dP"
    )

    ncol = (
        f"d{var}_dP_nfits"
    )

    ccol = (
        f"d{var}_dP_class"
    )


    x = pd.to_numeric(
        deriv[dcol],
        errors="coerce"
    )


    nfinite = int(
        np.isfinite(x).sum()
    )


    print()
    print(
        f"{var}"
    )

    print(
        f"finite derivatives = "
        f"{nfinite}/{len(deriv)}"
    )


    print(
        "nfits:"
    )

    print(

        deriv[ncol]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()

    )


    print(
        "classes:"
    )

    print(

        deriv[ccol]
        .value_counts(
            dropna=False
        )
        .to_string()

    )


# =============================================================================
# 10. DYNAMIC CLASSIFICATION
# =============================================================================

# dlnD/dP is used as the authoritative sign.
#
# Since D > 0:
#
#     dlnD/dP = (1/D) dD/dP
#
# therefore the signs are identical.

deriv[
    "dynamic_class"
] = "uncertain"


deriv.loc[

    deriv[
        "dlnD_dP_sign"
    ] == +1,

    "dynamic_class"

] = "anomalous"


deriv.loc[

    deriv[
        "dlnD_dP_sign"
    ] == -1,

    "dynamic_class"

] = "normal"


deriv.loc[

    deriv[
        "dlnD_dP_class"
    ] == "unresolved",

    "dynamic_class"

] = "unresolved"


# =============================================================================
# 11. CHECK D vs lnD SIGN CONSISTENCY
# =============================================================================

mask_both = (

    (
        deriv[
            "dD_dP_sign"
        ] != 0
    )

    &

    (
        deriv[
            "dlnD_dP_sign"
        ] != 0
    )

)


n_both = int(
    mask_both.sum()
)


n_disagree = int(

    np.sum(

        deriv.loc[
            mask_both,
            "dD_dP_sign"
        ].to_numpy()

        !=

        deriv.loc[
            mask_both,
            "dlnD_dP_sign"
        ].to_numpy()

    )

)


print()
print("=" * 88)
print("D vs lnD SIGN CONSISTENCY")
print("=" * 88)

print(
    f"States with both resolved = "
    f"{n_both}"
)

print(
    f"Sign disagreements        = "
    f"{n_disagree}"
)


# =============================================================================
# 12. DERIVATIVE--DERIVATIVE CORRESPONDENCE
# =============================================================================

descriptor_derivatives = {

    "Rg":
    "dRg_dP",

    "Rn":
    "dRn_dP",

    "A_SALR":
    "dA_SALR_dP",

}


summary_rows = []


print()
print("=" * 88)
print("DERIVATIVE--DERIVATIVE CORRESPONDENCE")
print("=" * 88)


for name, column in (
    descriptor_derivatives.items()
):

    mask = (

        np.isfinite(
            deriv[
                "dlnD_dP"
            ]
        )

        &

        np.isfinite(
            deriv[column]
        )

    )


    x = deriv.loc[
        mask,
        column
    ].to_numpy(
        dtype=float
    )


    y = deriv.loc[
        mask,
        "dlnD_dP"
    ].to_numpy(
        dtype=float
    )


    if len(x) < 5:

        continue


    rP, pP = pearsonr(
        x,
        y
    )


    rS, pS = spearmanr(
        x,
        y
    )


    print()
    print(
        f"Descriptor: {name}"
    )

    print(
        f"N              = "
        f"{len(x)}"
    )

    print(
        f"Pearson  r     = "
        f"{rP: .6f}"
    )

    print(
        f"Pearson  p     = "
        f"{pP:.6e}"
    )

    print(
        f"Spearman r     = "
        f"{rS: .6f}"
    )

    print(
        f"Spearman p     = "
        f"{pS:.6e}"
    )


    summary_rows.append([

        name,

        len(x),

        rP,
        pP,

        rS,
        pS,

    ])


global_summary = pd.DataFrame(

    summary_rows,

    columns=[

        "descriptor",

        "N",

        "Pearson_r",
        "Pearson_p",

        "Spearman_r",
        "Spearman_p",

    ]

)


# =============================================================================
# 13. ROC AUC
# =============================================================================

def auc_rank(
    y_true,
    scores
):

    """
    Rank-based ROC AUC.

    y_true:
        0 = normal
        1 = anomalous
    """

    y_true = np.asarray(
        y_true,
        dtype=int
    )


    scores = np.asarray(
        scores,
        dtype=float
    )


    good = (

        np.isfinite(
            scores
        )

        &

        np.isfinite(
            y_true
        )

    )


    y_true = y_true[good]
    scores = scores[good]


    n1 = int(
        np.sum(
            y_true == 1
        )
    )


    n0 = int(
        np.sum(
            y_true == 0
        )
    )


    if (
        n1 == 0
        or
        n0 == 0
    ):

        return np.nan


    ranks = rankdata(
        scores
    )


    sum_positive_ranks = np.sum(

        ranks[
            y_true == 1
        ]

    )


    auc = (

        sum_positive_ranks

        -

        n1
        *
        (n1 + 1)
        /
        2.0

    ) / (

        n1
        *
        n0

    )


    return float(
        auc
    )


# =============================================================================
# 14. NORMAL vs ANOMALOUS
# =============================================================================

class_rows = []


print()
print("=" * 88)
print("NORMAL vs DIFFUSION-ANOMALOUS STATES")
print("=" * 88)


for name, column in (
    descriptor_derivatives.items()
):

    normal = deriv.loc[

        (
            deriv[
                "dynamic_class"
            ]
            ==
            "normal"
        )

        &

        np.isfinite(
            deriv[column]
        ),

        column

    ].to_numpy(
        dtype=float
    )


    anomalous = deriv.loc[

        (
            deriv[
                "dynamic_class"
            ]
            ==
            "anomalous"
        )

        &

        np.isfinite(
            deriv[column]
        ),

        column

    ].to_numpy(
        dtype=float
    )


    if (
        len(normal) < 3
        or
        len(anomalous) < 3
    ):

        continue


    U, pMW = mannwhitneyu(

        anomalous,
        normal,

        alternative="two-sided"

    )


    y_true = np.concatenate([

        np.zeros(
            len(normal),
            dtype=int
        ),

        np.ones(
            len(anomalous),
            dtype=int
        ),

    ])


    scores = np.concatenate([

        normal,

        anomalous,

    ])


    auc = auc_rank(
        y_true,
        scores
    )


    oriented_auc = max(
        auc,
        1.0 - auc
    )


    if auc >= 0.5:

        direction = (
            "higher_in_anomalous"
        )

    else:

        direction = (
            "lower_in_anomalous"
        )


    print()
    print(
        f"Descriptor: {name}"
    )

    print(
        f"N normal          = "
        f"{len(normal)}"
    )

    print(
        f"N anomalous       = "
        f"{len(anomalous)}"
    )

    print(
        f"mean normal       = "
        f"{np.mean(normal): .6e}"
    )

    print(
        f"mean anomalous    = "
        f"{np.mean(anomalous): .6e}"
    )

    print(
        f"median normal     = "
        f"{np.median(normal): .6e}"
    )

    print(
        f"median anomalous  = "
        f"{np.median(anomalous): .6e}"
    )

    print(
        f"Mann-Whitney p    = "
        f"{pMW:.6e}"
    )

    print(
        f"ROC AUC           = "
        f"{auc:.6f}"
    )

    print(
        f"oriented AUC      = "
        f"{oriented_auc:.6f}"
    )

    print(
        f"direction         = "
        f"{direction}"
    )


    class_rows.append([

        name,

        len(normal),

        len(anomalous),

        np.mean(normal),

        np.median(normal),

        np.std(normal),

        np.mean(anomalous),

        np.median(anomalous),

        np.std(anomalous),

        U,

        pMW,

        auc,

        oriented_auc,

        direction,

    ])


class_summary = pd.DataFrame(

    class_rows,

    columns=[

        "descriptor",

        "N_normal",

        "N_anomalous",

        "mean_dX_normal",

        "median_dX_normal",

        "std_dX_normal",

        "mean_dX_anomalous",

        "median_dX_anomalous",

        "std_dX_anomalous",

        "MannWhitney_U",

        "MannWhitney_p",

        "ROC_AUC",

        "oriented_AUC",

        "direction",

    ]

)


# =============================================================================
# 15. STRUCTURAL DERIVATIVE SIGN BY DYNAMIC CLASS
# =============================================================================

sign_rows = []


print()
print("=" * 88)
print("STRUCTURAL-DERIVATIVE SIGN BY DYNAMIC CLASS")
print("=" * 88)


for name, column in (
    descriptor_derivatives.items()
):

    sign_col = (
        column
        +
        "_sign"
    )


    for dyn_class in [

        "normal",
        "anomalous",

    ]:

        subset = deriv[
            deriv[
                "dynamic_class"
            ]
            ==
            dyn_class
        ]


        signs = pd.to_numeric(
            subset[
                sign_col
            ],
            errors="coerce"
        ).to_numpy()


        n = len(
            signs
        )


        if n == 0:

            continue


        npos = int(
            np.sum(
                signs == +1
            )
        )


        nneg = int(
            np.sum(
                signs == -1
            )
        )


        nzero = int(
            np.sum(
                signs == 0
            )
        )


        frac_pos = (
            npos / n
        )


        frac_neg = (
            nneg / n
        )


        frac_zero = (
            nzero / n
        )


        print()

        print(
            f"{name:8s} | "
            f"{dyn_class:10s}"
        )

        print(
            f"N          = {n}"
        )

        print(
            f"positive   = "
            f"{npos:4d} "
            f"({frac_pos:.3f})"
        )

        print(
            f"negative   = "
            f"{nneg:4d} "
            f"({frac_neg:.3f})"
        )

        print(
            f"uncertain  = "
            f"{nzero:4d} "
            f"({frac_zero:.3f})"
        )


        sign_rows.append([

            name,

            dyn_class,

            n,

            npos,

            nneg,

            nzero,

            frac_pos,

            frac_neg,

            frac_zero,

        ])


sign_summary = pd.DataFrame(

    sign_rows,

    columns=[

        "descriptor",

        "dynamic_class",

        "N",

        "N_positive",

        "N_negative",

        "N_uncertain",

        "frac_positive",

        "frac_negative",

        "frac_uncertain",

    ]

)


# =============================================================================
# 16. DYNAMIC CLASSIFICATION
# =============================================================================

print()
print("=" * 88)
print("DYNAMIC CLASSIFICATION")
print("=" * 88)


print(

    deriv[
        "dynamic_class"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()

)


# =============================================================================
# 17. SAVE DATA
# =============================================================================

deriv = deriv.sort_values(
    ["T", "P"]
).reset_index(
    drop=True
)


deriv.to_csv(

    OUTDIR /
    "derivative_master.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


deriv.to_csv(

    OUTDIR /
    "derivative_master.csv",

    index=False,

    na_rep="NaN",

    float_format="%.10e"

)


global_summary.to_csv(

    OUTDIR /
    "derivative_global_summary.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


class_summary.to_csv(

    OUTDIR /
    "derivative_class_summary.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


sign_summary.to_csv(

    OUTDIR /
    "derivative_sign_summary.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


# =============================================================================
# 18. FIGURE 1 — DERIVATIVES vs PRESSURE
# =============================================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=(11.0, 8.2),

    sharex=True

)


axes = axes.ravel()


plot_specs = [

    (
        "dlnD_dP",

        r"$"
        r"\left("
        r"\partial\ln D^*/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",
    ),

    (
        "dRg_dP",

        r"$"
        r"\left("
        r"\partial R_g/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",
    ),

    (
        "dRn_dP",

        r"$"
        r"\left("
        r"\partial R_n/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",
    ),

    (
        "dA_SALR_dP",

        r"$"
        r"\left("
        r"\partial A_{\rm SALR}/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$",
    ),

]


for ax, (
    column,
    ylabel
) in zip(
    axes,
    plot_specs
):

    for Tsel in T_SELECTED:

        sub = deriv[
            np.isclose(
                deriv[
                    "T"
                ],
                Tsel
            )
        ].sort_values(
            "P"
        )


        if len(sub) == 0:

            continue


        ax.plot(

            sub["P"],

            sub[column],

            marker="o",

            markersize=3.5,

            linewidth=1.3,

            label=(
                rf"$T^*="
                rf"{Tsel:.2f}$"
            )

        )


    ax.axhline(

        0.0,

        linestyle="--",

        linewidth=1.0

    )


    ax.set_xlabel(
        r"$P^*$"
    )


    ax.set_ylabel(
        ylabel
    )


axes[0].legend(

    ncol=2,

    frameon=False,

    loc="best"

)


fig.tight_layout()


fig.savefig(

    OUTDIR /
    "derivatives_vs_pressure.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "derivatives_vs_pressure.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 19. FIGURE 2 — STRUCTURAL DERIVATIVE vs DYNAMIC DERIVATIVE
# =============================================================================

fig, axes = plt.subplots(

    1,
    3,

    figsize=(13.5, 4.3)

)


scatter_specs = [

    (
        "dRg_dP",

        r"$"
        r"\partial R_g/"
        r"\partial P^*"
        r"$",
    ),

    (
        "dRn_dP",

        r"$"
        r"\partial R_n/"
        r"\partial P^*"
        r"$",
    ),

    (
        "dA_SALR_dP",

        r"$"
        r"\partial A_{\rm SALR}/"
        r"\partial P^*"
        r"$",
    ),

]


sc = None


for ax, (
    xcol,
    xlabel
) in zip(
    axes,
    scatter_specs
):

    mask = (

        np.isfinite(
            deriv[
                xcol
            ]
        )

        &

        np.isfinite(
            deriv[
                "dlnD_dP"
            ]
        )

    )


    sc = ax.scatter(

        deriv.loc[
            mask,
            xcol
        ],

        deriv.loc[
            mask,
            "dlnD_dP"
        ],

        c=deriv.loc[
            mask,
            "T"
        ],

        s=16,

        alpha=0.70

    )


    ax.axhline(

        0.0,

        linestyle="--",

        linewidth=0.9

    )


    ax.axvline(

        0.0,

        linestyle="--",

        linewidth=0.9

    )


    ax.set_xlabel(
        xlabel
    )


    ax.set_ylabel(

        r"$"
        r"\partial\ln D^*/"
        r"\partial P^*"
        r"$"

    )


if sc is not None:

    cbar = fig.colorbar(

        sc,

        ax=axes,

        pad=0.02,

        fraction=0.025

    )


    cbar.set_label(
        r"$T^*$"
    )


fig.subplots_adjust(

    left=0.075,

    right=0.91,

    bottom=0.17,

    top=0.96,

    wspace=0.30

)


fig.savefig(

    OUTDIR /
    "derivative_correspondence.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "derivative_correspondence.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 20. FIGURE 3 — NORMAL vs ANOMALOUS
# =============================================================================

fig, axes = plt.subplots(

    1,
    3,

    figsize=(11.5, 4.3)

)


box_specs = [

    (
        "dRg_dP",

        r"$"
        r"\partial R_g/"
        r"\partial P^*"
        r"$",
    ),

    (
        "dRn_dP",

        r"$"
        r"\partial R_n/"
        r"\partial P^*"
        r"$",
    ),

    (
        "dA_SALR_dP",

        r"$"
        r"\partial A_{\rm SALR}/"
        r"\partial P^*"
        r"$",
    ),

]


for ax, (
    column,
    ylabel
) in zip(
    axes,
    box_specs
):

    normal = deriv.loc[

        deriv[
            "dynamic_class"
        ] == "normal",

        column

    ].dropna().to_numpy()


    anomalous = deriv.loc[

        deriv[
            "dynamic_class"
        ] == "anomalous",

        column

    ].dropna().to_numpy()


    if (
        len(normal) > 0
        and
        len(anomalous) > 0
    ):

        ax.boxplot(

            [
                normal,
                anomalous
            ],

            tick_labels=[
                "normal",
                "anomalous"
            ],

            showfliers=False

        )


    ax.axhline(

        0.0,

        linestyle="--",

        linewidth=0.9

    )


    ax.set_ylabel(
        ylabel
    )


fig.tight_layout()


fig.savefig(

    OUTDIR /
    "derivative_class_distributions.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "derivative_class_distributions.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 21. FINAL
# =============================================================================

print()
print("=" * 88)
print("OUTPUT FILES")
print("=" * 88)


for f in sorted(
    OUTDIR.iterdir()
):

    print(
        f
    )


print()
print("Done.")
