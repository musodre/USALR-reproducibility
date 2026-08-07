#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diffusion-anomaly validator
===========================

Physical criterion
------------------
For each isotherm:

    diffusion anomaly <=> (dD/dP)_T > 0

Since D > 0:

    sign(dD/dP) = sign[d ln(D)/dP]

We therefore analyze ln(D), which is numerically more stable.

Main features
-------------
1. No assumption of exactly two boundaries.
2. No global spline defines the anomaly.
3. Local quadratic fits are performed on ln[D(P)] at fixed T.
4. Several local fitting windows are tested.
5. Derivative sign must be stable across windows.
6. Fit covariance gives local derivative uncertainty.
7. Boundary types:

       resolved
       open_low
       open_high
       unresolved

8. A resolved boundary can NEVER have NaN pressure.
9. Multiple anomalous intervals are preserved.
10. The widest interval at each T is saved separately for
    incorporation into the P*-T* phase diagram.

Analysis domain
---------------
    0.10 <= P* <= 6.00
    0.02 <= T* <= 0.60

Input
-----
D_asymptotic_global.dat

Expected first columns:
    P T D

Additional columns are ignored.

Outputs
-------
diffusion_anomaly_validation/

    diffusion_local_derivatives.dat
    diffusion_anomaly_intervals.dat
    diffusion_anomaly_main_region.dat
    diffusion_anomaly_boundary_points.dat

    representative_isotherms.pdf
    representative_isotherms.png

    diffusion_anomaly_PT.pdf
    diffusion_anomaly_PT.png
"""

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# 1. INPUT / OUTPUT
# ======================================================================

INPUT_CANDIDATES = [

    Path("D_asymptotic_global.dat"),

    Path("analysis/D_asymptotic_global.dat"),

    DERIVED_DATA_ROOT / "asymptotic_diffusion" / "D_asymptotic_global.dat",

    Path("../D_asymptotic_global.dat"),
]


OUTDIR = Path(
    "diffusion_anomaly_validation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


OUT_LOCAL = (
    OUTDIR
    /
    "diffusion_local_derivatives.dat"
)


OUT_INTERVALS = (
    OUTDIR
    /
    "diffusion_anomaly_intervals.dat"
)


OUT_MAIN = (
    OUTDIR
    /
    "diffusion_anomaly_main_region.dat"
)


OUT_BOUNDARIES = (
    OUTDIR
    /
    "diffusion_anomaly_boundary_points.dat"
)


# ======================================================================
# 2. ANALYSIS DOMAIN
#
# IMPORTANT:
# the upper pressure limit is now P*=6.00
# ======================================================================

P_MIN = 0.10
P_MAX = 6.00

T_MIN = 0.02
T_MAX = 0.60


# ======================================================================
# 3. LOCAL QUADRATIC DERIVATIVE ANALYSIS
#
# ln D(P) = a(P-P0)^2 + b(P-P0) + c
#
# At P=P0:
#
# d ln(D)/dP = b
# ======================================================================

WINDOW_POINTS = [
    5,
    7,
    9,
]


MIN_POINTS_PER_FIT = 5


# ======================================================================
# 4. MINIMUM QUALITY OF AN INDIVIDUAL LOCAL FIT
# ======================================================================

MIN_INDIVIDUAL_R2 = 0.65


# ======================================================================
# 5. DERIVATIVE SIGNIFICANCE
#
# positive if:
#
# slope > Z sigma_slope
#
# negative if:
#
# slope < -Z sigma_slope
# ======================================================================

DERIVATIVE_Z = 1.00


# ======================================================================
# 6. CONSENSUS BETWEEN FIT WINDOWS
# ======================================================================

MIN_VALID_FITS = 2

POSITIVE_FRACTION_REQUIRED = 2.0 / 3.0

NEGATIVE_FRACTION_REQUIRED = 2.0 / 3.0


# ======================================================================
# 7. OPTIONAL REPAIR OF A SINGLE UNCERTAIN POINT
#
# anomalous -- uncertain -- anomalous
#
# A clearly normal state is NEVER bridged.
# ======================================================================

BRIDGE_SINGLE_UNCERTAIN = True


# ======================================================================
# 8. MINIMUM NUMBER OF ANOMALOUS STATES IN AN INTERVAL
# ======================================================================

MIN_ANOMALOUS_POINTS = 2


# ======================================================================
# 9. REPRESENTATIVE TEMPERATURES
# ======================================================================

REPRESENTATIVE_T = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]


# ======================================================================
# 10. MATPLOTLIB STYLE
# ======================================================================

plt.rcParams.update({

    "font.family": "serif",

    "mathtext.fontset": "cm",

    "font.size": 13,

    "axes.labelsize": 18,

    "axes.titlesize": 13,

    "xtick.labelsize": 12,

    "ytick.labelsize": 12,

    "legend.fontsize": 9,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": True,

    "ytick.right": True,
})


# ======================================================================
# 11. LOCATE INPUT FILE
# ======================================================================

INPUT = None


for candidate in INPUT_CANDIDATES:

    if candidate.exists():

        INPUT = candidate.resolve()

        break


if INPUT is None:

    raise FileNotFoundError(

        "\nCould not find D_asymptotic_global.dat.\n\n"
        "Checked:\n"
        +
        "\n".join(
            str(p)
            for p in INPUT_CANDIDATES
        )
    )


print()
print("=" * 88)
print("INPUT")
print("=" * 88)

print(
    f"Using:\n  {INPUT}"
)


# ======================================================================
# 12. ROBUST INPUT READER
# ======================================================================

def read_diffusion_table(
    filename
):

    # ------------------------------------------------------------------
    # Attempt 1:
    # file with a real header
    # ------------------------------------------------------------------

    try:

        trial = pd.read_csv(
            filename,
            sep=r"\s+",
            comment="#"
        )


        lower_map = {

            str(c).lower():
                c

            for c in trial.columns
        }


        aliases_P = [
            "p",
            "pressure",
            "pressao"
        ]


        aliases_T = [
            "t",
            "temperature",
            "temperatura"
        ]


        aliases_D = [
            "d",
            "diffusion",
            "diffusion_coefficient"
        ]


        def find_column(
            aliases
        ):

            for alias in aliases:

                if alias in lower_map:

                    return lower_map[
                        alias
                    ]

            return None


        Pcol = find_column(
            aliases_P
        )

        Tcol = find_column(
            aliases_T
        )

        Dcol = find_column(
            aliases_D
        )


        if (
            Pcol is not None
            and
            Tcol is not None
            and
            Dcol is not None
        ):

            result = trial[
                [
                    Pcol,
                    Tcol,
                    Dcol
                ]
            ].copy()


            result.columns = [
                "P",
                "T",
                "D"
            ]


            return result


    except Exception:

        pass


    # ------------------------------------------------------------------
    # Attempt 2:
    # headerless file
    #
    # first three columns assumed to be:
    #
    # P T D
    # ------------------------------------------------------------------

    raw = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None
    )


    if raw.shape[1] < 3:

        raise ValueError(

            "D_asymptotic_global.dat must contain "
            "at least three columns: P T D."
        )


    result = raw.iloc[
        :,
        0:3
    ].copy()


    result.columns = [
        "P",
        "T",
        "D"
    ]


    return result


# ======================================================================
# 13. READ AND CLEAN DATA
# ======================================================================

df = read_diffusion_table(
    INPUT
)


for col in [
    "P",
    "T",
    "D"
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
        df["D"]
    )
    &
    (
        df["D"]
        >
        0.0
    )
].copy()


# ======================================================================
# IMPORTANT:
#
# ANALYSIS NOW EXTENDS TO P*=6.00
# ======================================================================

df = df[
    (df["P"] >= P_MIN)
    &
    (df["P"] <= P_MAX)
    &
    (df["T"] >= T_MIN)
    &
    (df["T"] <= T_MAX)
].copy()


# ----------------------------------------------------------------------
# Average duplicated thermodynamic states, if present
# ----------------------------------------------------------------------

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

        D=(
            "D",
            "mean"
        )
    )
)


df[
    "lnD"
] = np.log(
    df[
        "D"
    ]
)


df = df.sort_values(
    [
        "T",
        "P"
    ]
).reset_index(
    drop=True
)


print()

print(
    f"States loaded = "
    f"{len(df)}"
)


print(
    f"Unique temperatures = "
    f"{df['T'].nunique()}"
)


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


# ======================================================================
# 14. R²
# ======================================================================

def calc_r2(
    y,
    yfit
):

    y = np.asarray(
        y,
        dtype=float
    )


    yfit = np.asarray(
        yfit,
        dtype=float
    )


    ss_res = np.sum(
        (
            y-yfit
        )**2
    )


    ss_tot = np.sum(
        (
            y-np.mean(y)
        )**2
    )


    if ss_tot <= 0:

        return np.nan


    return (
        1.0
        -
        ss_res/ss_tot
    )


# ======================================================================
# 15. LOCAL QUADRATIC FIT
# ======================================================================

def local_derivative_fit(
    P,
    lnD,
    center_index,
    npoints
):

    P = np.asarray(
        P,
        dtype=float
    )


    lnD = np.asarray(
        lnD,
        dtype=float
    )


    n = len(
        P
    )


    if (
        npoints > n
        or
        npoints < MIN_POINTS_PER_FIT
    ):

        return None


    P0 = P[
        center_index
    ]


    nearest = np.argsort(
        np.abs(
            P-P0
        )
    )


    use = nearest[
        :npoints
    ]


    Pf = P[
        use
    ]


    yf = lnD[
        use
    ]


    order = np.argsort(
        Pf
    )


    Pf = Pf[
        order
    ]


    yf = yf[
        order
    ]


    # ------------------------------------------------------------------
    # Center local polynomial at P0
    # ------------------------------------------------------------------

    x = (
        Pf-P0
    )


    try:

        coeff, cov = np.polyfit(
            x,
            yf,
            2,
            cov=True
        )

    except Exception:

        return None


    a, b, c = coeff


    yfit = np.polyval(
        coeff,
        x
    )


    R2 = calc_r2(
        yf,
        yfit
    )


    if (
        np.isfinite(
            R2
        )
        and
        R2
        <
        MIN_INDIVIDUAL_R2
    ):

        return None


    # ------------------------------------------------------------------
    # Because x=P-P0:
    #
    # dlnD/dP at P0 = b
    # ------------------------------------------------------------------

    slope = float(
        b
    )


    try:

        sigma_slope = float(

            np.sqrt(

                max(
                    cov[
                        1,
                        1
                    ],
                    0.0
                )
            )
        )

    except Exception:

        sigma_slope = np.nan


    if (
        np.isfinite(
            sigma_slope
        )
        and
        sigma_slope > 0.0
    ):

        significance = (
            slope
            /
            sigma_slope
        )


    else:

        significance = np.nan


    if (
        np.isfinite(
            significance
        )
        and
        significance
        >
        DERIVATIVE_Z
    ):

        vote = "positive"


    elif (
        np.isfinite(
            significance
        )
        and
        significance
        <
        -DERIVATIVE_Z
    ):

        vote = "negative"


    else:

        vote = "uncertain"


    return {

        "slope":
            slope,

        "sigma_slope":
            sigma_slope,

        "significance":
            significance,

        "vote":
            vote,

        "R2":
            R2,

        "window":
            npoints,
    }


# ======================================================================
# 16. CONSENSUS AT ONE STATE
# ======================================================================

def classify_local_state(
    P,
    lnD,
    index
):

    fits = []


    for window in WINDOW_POINTS:

        fit = local_derivative_fit(
            P,
            lnD,
            index,
            window
        )


        if fit is not None:

            fits.append(
                fit
            )


    if len(
        fits
    ) < MIN_VALID_FITS:

        return {

            "class":
                "unresolved",

            "n_valid_fits":
                len(
                    fits
                ),
        }


    slopes = np.array(
        [
            f[
                "slope"
            ]
            for f in fits
        ],
        dtype=float
    )


    sigmas = np.array(
        [
            f[
                "sigma_slope"
            ]
            for f in fits
        ],
        dtype=float
    )


    R2s = np.array(
        [
            f[
                "R2"
            ]
            for f in fits
        ],
        dtype=float
    )


    votes = [
        f[
            "vote"
        ]
        for f in fits
    ]


    n_positive = sum(
        v == "positive"
        for v in votes
    )


    n_negative = sum(
        v == "negative"
        for v in votes
    )


    n_uncertain = sum(
        v == "uncertain"
        for v in votes
    )


    nfits = len(
        fits
    )


    positive_fraction = (
        n_positive
        /
        nfits
    )


    negative_fraction = (
        n_negative
        /
        nfits
    )


    if (
        positive_fraction
        >=
        POSITIVE_FRACTION_REQUIRED
    ):

        state_class = (
            "anomalous"
        )


    elif (
        negative_fraction
        >=
        NEGATIVE_FRACTION_REQUIRED
    ):

        state_class = (
            "normal"
        )


    else:

        state_class = (
            "uncertain"
        )


    return {

        "class":
            state_class,

        "median_slope":
            float(
                np.median(
                    slopes
                )
            ),

        "mean_slope":
            float(
                np.mean(
                    slopes
                )
            ),

        "std_slope":
            (
                float(
                    np.std(
                        slopes,
                        ddof=1
                    )
                )
                if len(
                    slopes
                ) > 1
                else 0.0
            ),

        "median_sigma_slope":
            float(
                np.nanmedian(
                    sigmas
                )
            ),

        "median_R2":
            float(
                np.nanmedian(
                    R2s
                )
            ),

        "n_valid_fits":
            int(
                nfits
            ),

        "n_positive":
            int(
                n_positive
            ),

        "n_negative":
            int(
                n_negative
            ),

        "n_uncertain":
            int(
                n_uncertain
            ),

        "positive_fraction":
            float(
                positive_fraction
            ),

        "negative_fraction":
            float(
                negative_fraction
            ),
    }


# ======================================================================
# 17. ANALYZE ALL THERMODYNAMIC STATES
# ======================================================================

local_rows = []


print()
print("=" * 88)
print("LOCAL DIFFUSION-DERIVATIVE ANALYSIS")
print("=" * 88)


for T, group in df.groupby(
    "T"
):

    group = group.sort_values(
        "P"
    )


    P = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    D = group[
        "D"
    ].to_numpy(
        dtype=float
    )


    lnD = group[
        "lnD"
    ].to_numpy(
        dtype=float
    )


    for i in range(
        len(
            group
        )
    ):

        result = classify_local_state(
            P,
            lnD,
            i
        )


        local_rows.append({

            "P":
                float(
                    P[i]
                ),

            "T":
                float(
                    T
                ),

            "D":
                float(
                    D[i]
                ),

            "lnD":
                float(
                    lnD[i]
                ),

            "class":
                result[
                    "class"
                ],

            "median_slope":
                result.get(
                    "median_slope",
                    np.nan
                ),

            "mean_slope":
                result.get(
                    "mean_slope",
                    np.nan
                ),

            "std_slope":
                result.get(
                    "std_slope",
                    np.nan
                ),

            "median_sigma_slope":
                result.get(
                    "median_sigma_slope",
                    np.nan
                ),

            "median_R2":
                result.get(
                    "median_R2",
                    np.nan
                ),

            "n_valid_fits":
                result.get(
                    "n_valid_fits",
                    0
                ),

            "n_positive":
                result.get(
                    "n_positive",
                    0
                ),

            "n_negative":
                result.get(
                    "n_negative",
                    0
                ),

            "n_uncertain":
                result.get(
                    "n_uncertain",
                    0
                ),

            "positive_fraction":
                result.get(
                    "positive_fraction",
                    np.nan
                ),

            "negative_fraction":
                result.get(
                    "negative_fraction",
                    np.nan
                ),
        })


local = pd.DataFrame(
    local_rows
)


# ======================================================================
# 18. BRIDGE ONE UNCERTAIN POINT
# ======================================================================

local[
    "class_final"
] = local[
    "class"
].copy()


if BRIDGE_SINGLE_UNCERTAIN:

    for T, indices in local.groupby(
        "T"
    ).groups.items():

        indices = list(
            indices
        )


        indices = sorted(

            indices,

            key=lambda idx:
                local.loc[
                    idx,
                    "P"
                ]
        )


        if len(
            indices
        ) < 3:

            continue


        pressures = local.loc[
            indices,
            "P"
        ].to_numpy(
            dtype=float
        )


        dp = np.diff(
            pressures
        )


        median_dp = (
            np.median(
                dp
            )
            if len(
                dp
            ) > 0
            else np.nan
        )


        for j in range(
            1,
            len(
                indices
            )-1
        ):

            idx_prev = indices[
                j-1
            ]

            idx = indices[
                j
            ]

            idx_next = indices[
                j+1
            ]


            class_prev = local.loc[
                idx_prev,
                "class"
            ]


            class_now = local.loc[
                idx,
                "class"
            ]


            class_next = local.loc[
                idx_next,
                "class"
            ]


            if class_now not in [
                "uncertain",
                "unresolved"
            ]:

                continue


            if not (
                class_prev
                ==
                "anomalous"
                and
                class_next
                ==
                "anomalous"
            ):

                continue


            if np.isfinite(
                median_dp
            ):

                gap_left = (

                    local.loc[
                        idx,
                        "P"
                    ]

                    -

                    local.loc[
                        idx_prev,
                        "P"
                    ]
                )


                gap_right = (

                    local.loc[
                        idx_next,
                        "P"
                    ]

                    -

                    local.loc[
                        idx,
                        "P"
                    ]
                )


                if (
                    gap_left
                    >
                    1.75*median_dp
                    or
                    gap_right
                    >
                    1.75*median_dp
                ):

                    continue


            local.loc[
                idx,
                "class_final"
            ] = (
                "anomalous_bridged"
            )


# ======================================================================
# 19. SAVE LOCAL STATE CLASSIFICATION
# ======================================================================

local.to_csv(
    OUT_LOCAL,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ======================================================================
# 20. ROBUST ZERO-SLOPE INTERPOLATION
# ======================================================================

def interpolate_zero_slope(
    P1,
    s1,
    P2,
    s2
):

    values = [
        P1,
        s1,
        P2,
        s2
    ]


    if not all(
        np.isfinite(
            values
        )
    ):

        return (
            np.nan,
            "unresolved"
        )


    if s1 == 0.0:

        return (
            float(
                P1
            ),
            "resolved"
        )


    if s2 == 0.0:

        return (
            float(
                P2
            ),
            "resolved"
        )


    # ------------------------------------------------------------------
    # No derivative sign change
    # ------------------------------------------------------------------

    if (
        s1*s2 > 0.0
    ):

        return (
            np.nan,
            "unresolved"
        )


    denominator = (
        s2-s1
    )


    if abs(
        denominator
    ) < 1.0e-14:

        return (
            np.nan,
            "unresolved"
        )


    root = (

        P1

        -

        s1
        *
        (P2-P1)
        /
        denominator
    )


    low = min(
        P1,
        P2
    )


    high = max(
        P1,
        P2
    )


    if not (
        low
        <=
        root
        <=
        high
    ):

        return (
            np.nan,
            "unresolved"
        )


    return (
        float(
            root
        ),
        "resolved"
    )


# ======================================================================
# 21. FIND CONTIGUOUS ANOMALOUS RUNS
# ======================================================================

def anomalous_runs(
    group
):

    group = group.sort_values(
        "P"
    ).reset_index(
        drop=True
    )


    mask = group[
        "class_final"
    ].isin(
        [
            "anomalous",
            "anomalous_bridged"
        ]
    ).to_numpy()


    runs = []

    start = None


    for i, value in enumerate(
        mask
    ):

        if (
            value
            and
            start is None
        ):

            start = i


        if (
            start is not None
            and
            (
                not value
                or
                i
                ==
                len(
                    mask
                )-1
            )
        ):

            if value:

                end = i

            else:

                end = (
                    i-1
                )


            if (
                end-start+1
                >=
                MIN_ANOMALOUS_POINTS
            ):

                runs.append(
                    (
                        start,
                        end
                    )
                )


            start = None


    return (
        group,
        runs
    )


# ======================================================================
# 22. LOWER BOUNDARY
# ======================================================================

def classify_lower_boundary(
    group,
    i_start
):

    P_all = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    slopes = group[
        "median_slope"
    ].to_numpy(
        dtype=float
    )


    classes = group[
        "class_final"
    ].to_numpy()


    # ------------------------------------------------------------------
    # Anomaly already exists at lowest simulated P
    # ------------------------------------------------------------------

    if i_start == 0:

        return (
            float(
                P_all[
                    i_start
                ]
            ),
            "open_low"
        )


    i_prev = (
        i_start-1
    )


    P1 = P_all[
        i_prev
    ]


    P2 = P_all[
        i_start
    ]


    s1 = slopes[
        i_prev
    ]


    s2 = slopes[
        i_start
    ]


    # ------------------------------------------------------------------
    # Only a previous NORMAL state gives a resolved bracket
    # ------------------------------------------------------------------

    if (
        classes[
            i_prev
        ]
        ==
        "normal"
        and
        np.isfinite(
            s1
        )
        and
        np.isfinite(
            s2
        )
    ):

        root, status = (
            interpolate_zero_slope(
                P1,
                s1,
                P2,
                s2
            )
        )


        if (
            status
            ==
            "resolved"
        ):

            return (
                root,
                "resolved"
            )


    return (
        np.nan,
        "unresolved"
    )


# ======================================================================
# 23. UPPER BOUNDARY
# ======================================================================

def classify_upper_boundary(
    group,
    i_end
):

    P_all = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    slopes = group[
        "median_slope"
    ].to_numpy(
        dtype=float
    )


    classes = group[
        "class_final"
    ].to_numpy()


    # ------------------------------------------------------------------
    # Anomaly reaches maximum simulated pressure
    #
    # With present analysis, this can now be P*=6.00.
    # ------------------------------------------------------------------

    if (
        i_end
        ==
        len(
            group
        )-1
    ):

        return (
            float(
                P_all[
                    i_end
                ]
            ),
            "open_high"
        )


    i_next = (
        i_end+1
    )


    P1 = P_all[
        i_end
    ]


    P2 = P_all[
        i_next
    ]


    s1 = slopes[
        i_end
    ]


    s2 = slopes[
        i_next
    ]


    if (
        classes[
            i_next
        ]
        ==
        "normal"
        and
        np.isfinite(
            s1
        )
        and
        np.isfinite(
            s2
        )
    ):

        root, status = (
            interpolate_zero_slope(
                P1,
                s1,
                P2,
                s2
            )
        )


        if (
            status
            ==
            "resolved"
        ):

            return (
                root,
                "resolved"
            )


    return (
        np.nan,
        "unresolved"
    )


# ======================================================================
# 24. EXTRACT ANOMALOUS INTERVALS
# ======================================================================

interval_rows = []


for T, group0 in local.groupby(
    "T"
):

    group, runs = anomalous_runs(
        group0
    )


    if len(
        runs
    ) == 0:

        continue


    P_all = group[
        "P"
    ].to_numpy(
        dtype=float
    )


    for region_id, (
        i_start,
        i_end
    ) in enumerate(
        runs,
        start=1
    ):

        P_low, low_status = (
            classify_lower_boundary(
                group,
                i_start
            )
        )


        P_high, high_status = (
            classify_upper_boundary(
                group,
                i_end
            )
        )


        P_first = float(
            P_all[
                i_start
            ]
        )


        P_last = float(
            P_all[
                i_end
            ]
        )


        # --------------------------------------------------------------
        # Observed/effective interval width
        # --------------------------------------------------------------

        if (
            np.isfinite(
                P_low
            )
            and
            np.isfinite(
                P_high
            )
        ):

            pressure_width = (
                P_high
                -
                P_low
            )


        elif (
            low_status
            ==
            "open_low"
            and
            np.isfinite(
                P_high
            )
        ):

            pressure_width = (
                P_high
                -
                P_first
            )


        elif (
            high_status
            ==
            "open_high"
            and
            np.isfinite(
                P_low
            )
        ):

            pressure_width = (
                P_last
                -
                P_low
            )


        else:

            pressure_width = (
                P_last
                -
                P_first
            )


        interval_rows.append({

            "T":
                float(
                    T
                ),

            "region_id":
                int(
                    region_id
                ),

            "P_low":
                (
                    float(
                        P_low
                    )
                    if
                    np.isfinite(
                        P_low
                    )
                    else
                    np.nan
                ),

            "P_high":
                (
                    float(
                        P_high
                    )
                    if
                    np.isfinite(
                        P_high
                    )
                    else
                    np.nan
                ),

            "low_status":
                low_status,

            "high_status":
                high_status,

            "P_first_anomalous":
                P_first,

            "P_last_anomalous":
                P_last,

            "n_anomalous_points":
                int(
                    i_end
                    -
                    i_start
                    +
                    1
                ),

            "pressure_width":
                float(
                    pressure_width
                ),
        })


intervals = pd.DataFrame(
    interval_rows
)


if len(
    intervals
) > 0:

    intervals = (

        intervals
        .sort_values(
            [
                "T",
                "region_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )


intervals.to_csv(
    OUT_INTERVALS,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ======================================================================
# 25. MAIN ANOMALOUS INTERVAL AT EACH TEMPERATURE
# ======================================================================

main_rows = []


if len(
    intervals
) > 0:

    for T, group in intervals.groupby(
        "T"
    ):

        group = group.sort_values(
            "pressure_width",
            ascending=False
        )


        best = group.iloc[
            0
        ]


        main_rows.append(
            best.to_dict()
        )


main = pd.DataFrame(
    main_rows
)


if len(
    main
) > 0:

    main = (

        main
        .sort_values(
            "T"
        )
        .reset_index(
            drop=True
        )
    )


main.to_csv(
    OUT_MAIN,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ======================================================================
# 26. LONG-FORMAT BOUNDARY TABLE
# ======================================================================

boundary_rows = []


if len(
    main
) > 0:

    for _, row in main.iterrows():

        boundary_rows.append({

            "T":
                row[
                    "T"
                ],

            "P_boundary":
                row[
                    "P_low"
                ],

            "boundary":
                "low",

            "status":
                row[
                    "low_status"
                ],

            "P_observed":
                row[
                    "P_first_anomalous"
                ],
        })


        boundary_rows.append({

            "T":
                row[
                    "T"
                ],

            "P_boundary":
                row[
                    "P_high"
                ],

            "boundary":
                "high",

            "status":
                row[
                    "high_status"
                ],

            "P_observed":
                row[
                    "P_last_anomalous"
                ],
        })


boundaries = pd.DataFrame(
    boundary_rows
)


boundaries.to_csv(
    OUT_BOUNDARIES,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ======================================================================
# 27. CONSISTENCY CHECK
# ======================================================================

if len(
    main
) > 0:

    bad_low = main[
        (
            main[
                "low_status"
            ]
            ==
            "resolved"
        )
        &
        (
            ~np.isfinite(
                main[
                    "P_low"
                ]
            )
        )
    ]


    bad_high = main[
        (
            main[
                "high_status"
            ]
            ==
            "resolved"
        )
        &
        (
            ~np.isfinite(
                main[
                    "P_high"
                ]
            )
        )
    ]


else:

    bad_low = pd.DataFrame()

    bad_high = pd.DataFrame()


# ======================================================================
# 28. INTERVAL TABLE
# ======================================================================

print()
print("=" * 118)
print("CORRECTED DIFFUSION-ANOMALY INTERVALS")
print("=" * 118)


if len(
    intervals
) == 0:

    print(
        "No anomalous interval found."
    )


else:

    print(

        intervals[
            [
                "T",
                "region_id",
                "P_low",
                "P_high",
                "low_status",
                "high_status",
                "P_first_anomalous",
                "P_last_anomalous",
                "n_anomalous_points",
                "pressure_width",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ======================================================================
# 29. STATE SUMMARY
# ======================================================================

print()
print("=" * 88)
print("STATE CLASSIFICATION")
print("=" * 88)


n_anom = np.sum(

    local[
        "class_final"
    ].isin(
        [
            "anomalous",
            "anomalous_bridged"
        ]
    )
)


n_normal = np.sum(

    local[
        "class_final"
    ]
    ==
    "normal"
)


n_uncertain = (

    len(
        local
    )

    -

    n_anom

    -

    n_normal
)


print(
    f"Anomalous states = "
    f"{n_anom}"
)


print(
    f"Normal states    = "
    f"{n_normal}"
)


print(
    f"Uncertain states = "
    f"{n_uncertain}"
)


# ======================================================================
# 30. BOUNDARY-STATUS SUMMARY
# ======================================================================

print()
print("=" * 88)
print("BOUNDARY STATUS")
print("=" * 88)


if len(
    main
) > 0:

    for boundary_name in [
        "low",
        "high"
    ]:

        column = (
            f"{boundary_name}_status"
        )


        print(
            f"\n{boundary_name.upper()} boundary:"
        )


        counts = (
            main[
                column
            ]
            .value_counts(
                dropna=False
            )
        )


        for status, count in counts.items():

            print(
                f"  {status:12s}: "
                f"{count}"
            )


# ======================================================================
# 31. CONSISTENCY REPORT
# ======================================================================

print()
print("=" * 88)
print("CONSISTENCY CHECK")
print("=" * 88)


if (
    len(
        bad_low
    ) == 0
    and
    len(
        bad_high
    ) == 0
):

    print(
        "PASS: no resolved boundary has NaN pressure."
    )


else:

    print(
        "WARNING: inconsistent boundaries remain."
    )


    print(
        f"Bad lower boundaries = "
        f"{len(bad_low)}"
    )


    print(
        f"Bad upper boundaries = "
        f"{len(bad_high)}"
    )


# ======================================================================
# 32. REPRESENTATIVE ISOTHERMS
# ======================================================================

def nearest_available_temperature(
    target
):

    values = np.sort(
        df[
            "T"
        ].unique()
    )


    return float(

        values[
            np.argmin(
                np.abs(
                    values-target
                )
            )
        ]
    )


fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        11.0,
        6.5
    )
)


axes = axes.ravel()


for ax, T_target in zip(
    axes,
    REPRESENTATIVE_T
):

    Tsel = nearest_available_temperature(
        T_target
    )


    raw = df[
        np.isclose(
            df[
                "T"
            ],
            Tsel
        )
    ].sort_values(
        "P"
    )


    loc = local[
        np.isclose(
            local[
                "T"
            ],
            Tsel
        )
    ].sort_values(
        "P"
    )


    ax.plot(

        raw[
            "P"
        ],

        raw[
            "lnD"
        ],

        "-",

        lw=1.0,

        alpha=0.45
    )


    normal = loc[
        loc[
            "class_final"
        ]
        ==
        "normal"
    ]


    anomalous = loc[
        loc[
            "class_final"
        ].isin(
            [
                "anomalous",
                "anomalous_bridged"
            ]
        )
    ]


    uncertain = loc[
        ~loc[
            "class_final"
        ].isin(
            [
                "normal",
                "anomalous",
                "anomalous_bridged"
            ]
        )
    ]


    if len(
        normal
    ) > 0:

        ax.scatter(

            normal[
                "P"
            ],

            normal[
                "lnD"
            ],

            s=24,

            label="normal"
        )


    if len(
        anomalous
    ) > 0:

        ax.scatter(

            anomalous[
                "P"
            ],

            anomalous[
                "lnD"
            ],

            s=34,

            marker="s",

            label="anomalous"
        )


    if len(
        uncertain
    ) > 0:

        ax.scatter(

            uncertain[
                "P"
            ],

            uncertain[
                "lnD"
            ],

            s=28,

            marker="x",

            label="uncertain"
        )


    # ------------------------------------------------------------------
    # Visualize the directly supported anomaly interval
    # ------------------------------------------------------------------

    if len(
        main
    ) > 0:

        current = main[
            np.isclose(
                main[
                    "T"
                ],
                Tsel
            )
        ]


        if len(
            current
        ) > 0:

            row = current.iloc[
                0
            ]


            if (
                row[
                    "low_status"
                ]
                ==
                "resolved"
            ):

                span_low = row[
                    "P_low"
                ]


            else:

                span_low = row[
                    "P_first_anomalous"
                ]


            if (
                row[
                    "high_status"
                ]
                ==
                "resolved"
            ):

                span_high = row[
                    "P_high"
                ]


            else:

                span_high = row[
                    "P_last_anomalous"
                ]


            if (
                np.isfinite(
                    span_low
                )
                and
                np.isfinite(
                    span_high
                )
            ):

                ax.axvspan(

                    span_low,

                    span_high,

                    alpha=0.08
                )


    ax.set_title(
        rf"$T^*={Tsel:.2f}$"
    )


    ax.set_xlabel(
        r"$P^*$"
    )


    ax.set_ylabel(
        r"$\ln D^*$"
    )


    ax.grid(
        alpha=0.12
    )


handles, labels = (
    axes[
        0
    ]
    .get_legend_handles_labels()
)


if len(
    handles
) > 0:

    fig.legend(

        handles,

        labels,

        frameon=False,

        ncol=3,

        loc="upper center"
    )


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94
    ]
)


fig.savefig(

    OUTDIR
    /
    "representative_isotherms.pdf",

    bbox_inches="tight"
)


fig.savefig(

    OUTDIR
    /
    "representative_isotherms.png",

    dpi=500,

    bbox_inches="tight"
)


plt.close(
    fig
)


# ======================================================================
# 33. P-T DIAGNOSTIC MAP
# ======================================================================

fig, ax = plt.subplots(
    figsize=(
        7.3,
        5.5
    )
)


normal = local[
    local[
        "class_final"
    ]
    ==
    "normal"
]


anom = local[
    local[
        "class_final"
    ].isin(
        [
            "anomalous",
            "anomalous_bridged"
        ]
    )
]


uncertain = local[
    ~local[
        "class_final"
    ].isin(
        [
            "normal",
            "anomalous",
            "anomalous_bridged"
        ]
    )
]


if len(
    normal
) > 0:

    ax.scatter(

        normal[
            "T"
        ],

        normal[
            "P"
        ],

        s=12,

        alpha=0.25,

        label="normal"
    )


if len(
    uncertain
) > 0:

    ax.scatter(

        uncertain[
            "T"
        ],

        uncertain[
            "P"
        ],

        s=14,

        marker="x",

        alpha=0.40,

        label="uncertain"
    )


if len(
    anom
) > 0:

    ax.scatter(

        anom[
            "T"
        ],

        anom[
            "P"
        ],

        s=20,

        label="diffusion anomaly"
    )


# ======================================================================
# 34. LOWER BOUNDARY TYPES
# ======================================================================

if len(
    main
) > 0:

    resolved_low = main[
        main[
            "low_status"
        ]
        ==
        "resolved"
    ].sort_values(
        "T"
    )


    if len(
        resolved_low
    ) > 0:

        ax.plot(

            resolved_low[
                "T"
            ],

            resolved_low[
                "P_low"
            ],

            "--",

            lw=1.6,

            label=(
                "resolved lower boundary"
            )
        )


    open_low = main[
        main[
            "low_status"
        ]
        ==
        "open_low"
    ]


    if len(
        open_low
    ) > 0:

        ax.scatter(

            open_low[
                "T"
            ],

            open_low[
                "P_first_anomalous"
            ],

            marker="v",

            s=42,

            label=(
                "open lower boundary"
            )
        )


    unresolved_low = main[
        main[
            "low_status"
        ]
        ==
        "unresolved"
    ]


    if len(
        unresolved_low
    ) > 0:

        ax.scatter(

            unresolved_low[
                "T"
            ],

            unresolved_low[
                "P_first_anomalous"
            ],

            marker="x",

            s=42,

            label=(
                "unresolved lower boundary"
            )
        )


# ======================================================================
# 35. UPPER BOUNDARY TYPES
# ======================================================================

if len(
    main
) > 0:

    resolved_high = main[
        main[
            "high_status"
        ]
        ==
        "resolved"
    ].sort_values(
        "T"
    )


    if len(
        resolved_high
    ) > 0:

        ax.plot(

            resolved_high[
                "T"
            ],

            resolved_high[
                "P_high"
            ],

            ":",

            lw=1.6,

            label=(
                "resolved upper boundary"
            )
        )


    open_high = main[
        main[
            "high_status"
        ]
        ==
        "open_high"
    ]


    if len(
        open_high
    ) > 0:

        ax.scatter(

            open_high[
                "T"
            ],

            open_high[
                "P_last_anomalous"
            ],

            marker="^",

            s=30,

            alpha=0.60,

            label=(
                "open upper boundary"
            )
        )


# ======================================================================
# 36. FILL ONLY CLOSED, FULLY RESOLVED REGIONS
# ======================================================================

if len(
    main
) > 0:

    closed = main[
        (
            main[
                "low_status"
            ]
            ==
            "resolved"
        )
        &
        (
            main[
                "high_status"
            ]
            ==
            "resolved"
        )
        &
        np.isfinite(
            main[
                "P_low"
            ]
        )
        &
        np.isfinite(
            main[
                "P_high"
            ]
        )
    ].sort_values(
        "T"
    )


    if len(
        closed
    ) >= 2:

        ax.fill_between(

            closed[
                "T"
            ],

            closed[
                "P_low"
            ],

            closed[
                "P_high"
            ],

            alpha=0.07
        )


ax.set_xlabel(
    r"$T^*$"
)


ax.set_ylabel(
    r"$P^*$"
)


ax.set_xlim(
    T_MIN-0.01,
    T_MAX+0.01
)


ax.set_ylim(
    P_MIN-0.05,
    P_MAX+0.05
)


ax.grid(
    alpha=0.12
)


ax.legend(
    frameon=False,
    fontsize=8
)


fig.tight_layout()


fig.savefig(

    OUTDIR
    /
    "diffusion_anomaly_PT.pdf",

    bbox_inches="tight"
)


fig.savefig(

    OUTDIR
    /
    "diffusion_anomaly_PT.png",

    dpi=500,

    bbox_inches="tight"
)


plt.show()


# ======================================================================
# 37. FINAL REPORT
# ======================================================================

print()
print("=" * 88)
print("OUTPUT")
print("=" * 88)


print(
    f"Local derivative classification:\n"
    f"  {OUT_LOCAL}"
)


print(
    f"\nAll anomalous intervals:\n"
    f"  {OUT_INTERVALS}"
)


print(
    f"\nMain anomaly region:\n"
    f"  {OUT_MAIN}"
)


print(
    f"\nBoundary table:\n"
    f"  {OUT_BOUNDARIES}"
)


print(
    f"\nRepresentative isotherms:\n"
    f"  "
    f"{OUTDIR/'representative_isotherms.pdf'}"
)


print(
    f"\nP-T diagnostic:\n"
    f"  "
    f"{OUTDIR/'diffusion_anomaly_PT.pdf'}"
)


print()

print("=" * 88)
print("INTERPRETATION OF BOUNDARY STATUS")
print("=" * 88)


print(
    "resolved   : derivative zero is bracketed "
    "and interpolated."
)


print(
    "open_low   : anomaly already exists at the "
    "lowest simulated pressure."
)


print(
    "open_high  : anomaly persists up to the "
    "highest simulated pressure."
)


print(
    "unresolved : anomaly is observed, but its "
    "boundary cannot be located robustly."
)


print()

print("=" * 88)
print("IMPORTANT FOR THIS RUN")
print("=" * 88)


print(
    f"The investigated pressure interval is now:"
)

print(
    f"  {P_MIN:.2f} <= P* <= {P_MAX:.2f}"
)


print()

print(
    "If high_status = open_high and P_high = 6.0, "
    "this does NOT mean that the diffusion-anomaly "
    "boundary occurs at P*=6."
)


print(
    "It means that the anomalous behavior persists "
    "up to at least P*=6, with no upper boundary "
    "resolved inside the investigated phase-diagram domain."
)
