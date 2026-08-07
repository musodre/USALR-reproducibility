#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermodynamic response analysis for the SALR system
===================================================

Expected directory structure:

../allpress/
    P_0.100/
        thermo/
            outvars_P_0.10_T_0.02.profile
            outvars_P_0.10_T_0.04.profile
            ...
    P_0.200/
        thermo/
            ...

Input columns:

0  TimeStep
1  c_thermo_temp
2  c_thermo_press
3  v_myvol
4  v_dens
5  c_thermo_pe
6  c_kinetic
7  v_myenthalpy

The last column is treated as reduced enthalpy per particle h*.

Response functions:

    cp*      = (dh*/dT*)_P

    kappa_T* = (1/rho*) (drho*/dP*)_T

    alpha_P* = -(1/rho*) (drho*/dT*)_P

Important:
----------
These response functions are obtained from derivatives of
equilibrium averages, NOT from fluctuation formulas.

The .profile files contain time-averaged quantities and should
not be treated as instantaneous ensemble samples for variance-
based estimators.
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

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, PchipInterpolator
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ============================================================
# 1. PATHS
# ============================================================

BASE = RAW_DATA_ROOT / "allpress"

OUTDIR = DERIVED_DATA_ROOT / "thermo_response"
OUTDIR.mkdir(parents=True, exist_ok=True)

PLOTDIR = OUTDIR / "plots"
PLOTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EQUILIBRIUM / PLATEAU SETTINGS
# ============================================================

# Fraction of the end of each profile used to characterize
# the converged plateau.
TAIL_FRACTION = 0.20

# Require at least this many rows in the selected tail.
MIN_TAIL_POINTS = 50

# The final thermodynamic estimate will be the MEDIAN of the
# tail. This is robust against a small residual drift.
USE_MEDIAN = True

# Relative drift criterion:
#
# linear change across the selected tail divided by |mean|
#
# This is primarily a diagnostic flag; states are NOT
# discarded automatically unless REJECT_BAD_STATES=True.
MAX_RELATIVE_DRIFT = 5.0e-3

REJECT_BAD_STATES = False


# ============================================================
# 3. STATE-SPACE RANGE
# ============================================================

P_MIN = 0.10
P_MAX = 6.00

T_MIN = 0.02
T_MAX = 0.60


# ============================================================
# 4. LOCAL DERIVATIVE SETTINGS
# ============================================================

# Number of neighboring state points used in each local fit.
#
# Should normally be odd.
LOCAL_POINTS = 7

# Quadratic fit gives both first and second derivatives.
LOCAL_DEGREE = 2

# Minimum points accepted
MIN_LOCAL_POINTS = 5

# Numerical condition warning
MAX_CONDITION_NUMBER = 1.0e10


# ============================================================
# 5. LOCUS DETECTION
# ============================================================

# Minimum number of points along an isobar before looking for
# extrema.
MIN_LOCUS_POINTS = 8

# Gaussian smoothing applied ONLY for identification of broad
# maxima in response-function curves.
#
# The actual reported extremum is subsequently refined using
# a local quadratic fit to the UNSMOOTHED response data.
LOCUS_SMOOTH_SIGMA = 1.0

# Relative prominence threshold used by find_peaks.
PROMINENCE_FRACTION = 0.08

# Local points used to refine each extremum
EXTREMUM_LOCAL_POINTS = 5

# alpha=0 root filtering
MAX_ALPHA_ROOT_JUMP = 0.08


# ============================================================
# 6. REPRESENTATIVE PRESSURES FOR CURVE PLOTS
# ============================================================

REPRESENTATIVE_PRESSURES = [
    0.50,
    0.80,
    1.00,
    1.20,
    1.50,
    1.70,
    2.00,
    2.50,
]


# ============================================================
# 7. PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 13,

    "axes.labelsize": 17,
    "axes.titlesize": 15,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 10,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,
})


# ============================================================
# 8. FILE-NAME PARSING
# ============================================================

def parse_PT(filename):
    """
    Extract nominal P and T from:

        outvars_P_1.00_T_0.26.profile
    """

    name = Path(filename).name

    match = re.search(
        r"outvars_P_([0-9.]+)_T_([0-9.]+)\.profile",
        name
    )

    if match is None:
        return None

    P = float(match.group(1))
    T = float(match.group(2))

    return P, T


# ============================================================
# 9. LINEAR DRIFT
# ============================================================

def relative_drift(x, y):
    """
    Estimate residual drift across a tail.

    Returns:
        absolute fractional change across the tail.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    x0 = x - np.mean(x)

    coeff = np.polyfit(
        x0,
        y,
        1
    )

    slope = coeff[0]

    delta = (
        slope
        *
        (x0.max() - x0.min())
    )

    scale = abs(
        np.mean(y)
    )

    if scale == 0:
        return np.nan

    return abs(delta) / scale


# ============================================================
# 10. EQUILIBRIUM VALUE FROM ONE PROFILE
# ============================================================

def analyze_profile(filename):

    data = np.loadtxt(
        filename,
        comments="#"
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 8:
        raise ValueError(
            f"{filename}: expected >=8 columns."
        )

    n = len(data)

    n_tail = max(
        MIN_TAIL_POINTS,
        int(TAIL_FRACTION*n)
    )

    n_tail = min(
        n_tail,
        n
    )

    tail = data[-n_tail:]

    step = tail[:, 0]

    temp = tail[:, 1]
    press = tail[:, 2]
    vol = tail[:, 3]
    rho = tail[:, 4]
    pe = tail[:, 5]
    ke = tail[:, 6]
    h = tail[:, 7]

    def central(x):

        x = x[np.isfinite(x)]

        if len(x) == 0:
            return np.nan

        if USE_MEDIAN:
            return np.median(x)

        return np.mean(x)

    # --------------------------------------------------------
    # equilibrium estimates
    # --------------------------------------------------------

    T_eq = central(temp)
    P_eq = central(press)

    V_eq = central(vol)
    rho_eq = central(rho)

    PE_eq = central(pe)
    KE_eq = central(ke)

    h_eq = central(h)

    # --------------------------------------------------------
    # residual plateau scatter
    #
    # This is NOT a statistical error bar, because the values
    # are themselves time-averaged.
    # --------------------------------------------------------

    rho_scatter = np.std(rho, ddof=1)
    V_scatter = np.std(vol, ddof=1)
    h_scatter = np.std(h, ddof=1)

    # --------------------------------------------------------
    # drift diagnostics
    # --------------------------------------------------------

    drift_rho = relative_drift(
        step,
        rho
    )

    drift_V = relative_drift(
        step,
        vol
    )

    drift_h = relative_drift(
        step,
        h
    )

    max_drift = np.nanmax(
        [
            drift_rho,
            drift_V,
            drift_h
        ]
    )

    converged = (
        np.isfinite(max_drift)
        and
        max_drift <= MAX_RELATIVE_DRIFT
    )

    # --------------------------------------------------------
    # Infer approximate particle number from
    #
    # H_total = PE + KE + P V
    #
    # and h = H/N
    #
    # This is diagnostic only.
    # --------------------------------------------------------

    if (
        np.isfinite(h_eq)
        and
        abs(h_eq) > 1.0e-12
    ):

        Htot_est = (
            PE_eq
            +
            KE_eq
            +
            P_eq*V_eq
        )

        N_est = Htot_est / h_eq

    else:

        N_est = np.nan

    return {
        "T_eq": T_eq,
        "P_eq": P_eq,

        "V": V_eq,
        "rho": rho_eq,

        "PE": PE_eq,
        "KE": KE_eq,
        "h": h_eq,

        "rho_scatter": rho_scatter,
        "V_scatter": V_scatter,
        "h_scatter": h_scatter,

        "drift_rho": drift_rho,
        "drift_V": drift_V,
        "drift_h": drift_h,

        "max_drift": max_drift,
        "converged": converged,

        "N_est": N_est,

        "n_total": n,
        "n_tail": n_tail,
    }


# ============================================================
# 11. SCAN ALL FILES
# ============================================================

files = sorted(
    BASE.glob(
        "P_*/thermo/outvars_P_*_T_*.profile"
    )
)

if len(files) == 0:

    raise FileNotFoundError(
        f"No thermo profiles found under {BASE.resolve()}"
    )


rows = []

print()
print("="*80)
print("READING THERMODYNAMIC STATES")
print("="*80)

for i, filename in enumerate(files, start=1):

    PT = parse_PT(
        filename
    )

    if PT is None:
        continue

    P_nom, T_nom = PT

    if not (
        P_MIN <= P_nom <= P_MAX
        and
        T_MIN <= T_nom <= T_MAX
    ):
        continue

    try:

        result = analyze_profile(
            filename
        )

    except Exception as exc:

        warnings.warn(
            f"Could not read {filename}: {exc}"
        )

        continue

    if (
        REJECT_BAD_STATES
        and
        not result["converged"]
    ):
        continue

    row = {
        "P": P_nom,
        "T": T_nom,
        "file": str(filename),
    }

    row.update(
        result
    )

    rows.append(
        row
    )

    if i % 200 == 0:

        print(
            f"processed {i}/{len(files)} files..."
        )


df = pd.DataFrame(
    rows
)


if len(df) == 0:

    raise RuntimeError(
        "No valid thermodynamic states were loaded."
    )


# ============================================================
# 12. HANDLE DUPLICATE P,T STATES
# ============================================================

numeric_cols = [
    "T_eq",
    "P_eq",

    "V",
    "rho",

    "PE",
    "KE",
    "h",

    "rho_scatter",
    "V_scatter",
    "h_scatter",

    "drift_rho",
    "drift_V",
    "drift_h",

    "max_drift",

    "N_est",

    "n_total",
    "n_tail",
]


agg_dict = {
    col: "mean"
    for col in numeric_cols
}

agg_dict["converged"] = "all"


df = (
    df
    .groupby(
        ["P", "T"],
        as_index=False
    )
    .agg(agg_dict)
)


df = df.sort_values(
    ["P", "T"]
).reset_index(
    drop=True
)


print()
print(
    f"Unique states loaded: {len(df)}"
)

print(
    f"Pressure range: "
    f"{df.P.min():.3f} -- {df.P.max():.3f}"
)

print(
    f"Temperature range: "
    f"{df['T'].min():.3f} -- {df['T'].max():.3f}"
)

# ============================================================
# 13. PARTICLE-NUMBER DIAGNOSTIC
# ============================================================

N_valid = df[
    np.isfinite(df["N_est"])
]["N_est"].values


if len(N_valid) > 0:

    print()
    print("="*80)
    print("ENTHALPY CONSISTENCY DIAGNOSTIC")
    print("="*80)

    print(
        f"median inferred N = "
        f"{np.median(N_valid):.3f}"
    )

    print(
        f"mean inferred N   = "
        f"{np.mean(N_valid):.3f}"
    )

    print(
        f"std inferred N    = "
        f"{np.std(N_valid):.3f}"
    )


# ============================================================
# 14. SAVE EQUILIBRIUM TABLE
# ============================================================

equilibrium_file = (
    OUTDIR
    /
    "equilibrium_thermodynamics.dat"
)


df.to_csv(
    equilibrium_file,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 15. LOCAL POLYNOMIAL DERIVATIVE
# ============================================================

def local_derivative(
    x,
    y,
    x0,
    npoints=LOCAL_POINTS,
    degree=LOCAL_DEGREE
):
    """
    Local polynomial derivative dy/dx at x0.

    The nearest npoints are used.

    Returns:
        derivative,
        second derivative,
        local R2,
        condition number
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    good = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    if len(x) < MIN_LOCAL_POINTS:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    order = np.argsort(
        np.abs(x-x0)
    )

    use = order[
        :min(npoints, len(order))
    ]

    xx = x[use]
    yy = y[use]

    order2 = np.argsort(
        xx
    )

    xx = xx[order2]
    yy = yy[order2]

    # --------------------------------------------------------
    # Center coordinate around x0.
    #
    # Improves numerical conditioning and makes the first
    # derivative simply coefficient of x^1.
    # --------------------------------------------------------

    z = xx - x0

    deg = min(
        degree,
        len(z)-1
    )

    # Vandermonde diagnostic
    A = np.vander(
        z,
        deg+1
    )

    cond = np.linalg.cond(
        A
    )

    coeff = np.polyfit(
        z,
        yy,
        deg
    )

    fit = np.polyval(
        coeff,
        z
    )

    ss_res = np.sum(
        (yy-fit)**2
    )

    ss_tot = np.sum(
        (yy-np.mean(yy))**2
    )

    if ss_tot > 0:

        R2 = (
            1.0
            -
            ss_res/ss_tot
        )

    else:

        R2 = np.nan

    p = np.poly1d(
        coeff
    )

    dp = np.polyder(
        p,
        1
    )

    d2p = np.polyder(
        p,
        2
    )

    d1 = float(
        dp(0.0)
    )

    if deg >= 2:

        d2 = float(
            d2p(0.0)
        )

    else:

        d2 = np.nan

    return (
        d1,
        d2,
        R2,
        cond
    )


# ============================================================
# 16. RESPONSE FUNCTIONS
# ============================================================

response_rows = []


print()
print("="*80)
print("CALCULATING RESPONSE FUNCTIONS")
print("="*80)


# ------------------------------------------------------------
# Isobaric derivatives:
#
# dh/dT    -> cp
# drho/dT  -> alpha
# ------------------------------------------------------------

for P, group in df.groupby("P"):

    group = group.sort_values(
        "T"
    )

    Tvals = group["T"].values
    hvals = group["h"].values
    rvals = group["rho"].values

    for _, row in group.iterrows():

        T0 = row["T"]

        dhdT, d2hdT2, r2_h, cond_h = (
            local_derivative(
                Tvals,
                hvals,
                T0
            )
        )

        drhodT, d2rhodT2, r2_rhoT, cond_rhoT = (
            local_derivative(
                Tvals,
                rvals,
                T0
            )
        )

        rho0 = row["rho"]

        cp = dhdT

        if (
            np.isfinite(rho0)
            and
            rho0 != 0
        ):

            alpha = (
                -drhodT
                /
                rho0
            )

        else:

            alpha = np.nan

        response_rows.append({
            "P": P,
            "T": T0,

            "cp": cp,
            "alpha": alpha,

            "dhdT": dhdT,
            "d2hdT2": d2hdT2,

            "drhodT": drhodT,
            "d2rhodT2": d2rhodT2,

            "R2_h_T": r2_h,
            "R2_rho_T": r2_rhoT,

            "cond_h_T": cond_h,
            "cond_rho_T": cond_rhoT,
        })


resp = pd.DataFrame(
    response_rows
)


# ------------------------------------------------------------
# Isothermal derivative:
#
# drho/dP -> kappa_T
# ------------------------------------------------------------

kappa_rows = []


for T, group in df.groupby("T"):

    group = group.sort_values(
        "P"
    )

    Pvals = group["P"].values
    rvals = group["rho"].values

    for _, row in group.iterrows():

        P0 = row["P"]

        drhodP, d2rhodP2, r2_rhoP, cond_rhoP = (
            local_derivative(
                Pvals,
                rvals,
                P0
            )
        )

        rho0 = row["rho"]

        if (
            np.isfinite(rho0)
            and
            rho0 != 0
        ):

            kappa = (
                drhodP
                /
                rho0
            )

        else:

            kappa = np.nan

        kappa_rows.append({
            "P": P0,
            "T": T,

            "kappa": kappa,

            "drhodP": drhodP,
            "d2rhodP2": d2rhodP2,

            "R2_rho_P": r2_rhoP,
            "cond_rho_P": cond_rhoP,
        })


kappa_df = pd.DataFrame(
    kappa_rows
)


# ============================================================
# 17. MERGE EVERYTHING
# ============================================================

resp = resp.merge(
    kappa_df,
    on=["P", "T"],
    how="left"
)


resp = resp.merge(
    df[
        [
            "P",
            "T",
            "rho",
            "V",
            "h",
            "max_drift",
            "converged",
        ]
    ],
    on=["P", "T"],
    how="left"
)


response_file = (
    OUTDIR
    /
    "thermodynamic_response_functions.dat"
)


resp.to_csv(
    response_file,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 18. BASIC PHYSICAL CHECKS
# ============================================================

print()
print("="*80)
print("RESPONSE-FUNCTION SUMMARY")
print("="*80)


for col in [
    "cp",
    "kappa",
    "alpha"
]:

    arr = resp[col].values

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        continue

    print(
        f"{col:8s}: "
        f"min={np.min(arr): .6g}   "
        f"max={np.max(arr): .6g}   "
        f"median={np.median(arr): .6g}"
    )


# ============================================================
# 19. ALPHA=0 ROOTS = TMD CANDIDATES
# ============================================================

alpha_roots = []


for P, group in resp.groupby("P"):

    group = group.sort_values(
        "T"
    )

    T = group["T"].values
    alpha = group["alpha"].values

    good = (
        np.isfinite(T)
        &
        np.isfinite(alpha)
    )

    T = T[good]
    alpha = alpha[good]

    if len(T) < 4:
        continue

    for i in range(
        len(T)-1
    ):

        a1 = alpha[i]
        a2 = alpha[i+1]

        if a1 == 0:

            root = T[i]

        elif (
            a1*a2 < 0
        ):

            # linear interpolation of the zero
            root = (
                T[i]
                -
                a1
                *
                (T[i+1]-T[i])
                /
                (a2-a1)
            )

        else:

            continue

        # ----------------------------------------------------
        # Check that the density curvature is negative.
        #
        # TMD must be a density MAXIMUM.
        # ----------------------------------------------------

        local = group.iloc[
            max(0, i-2):
            min(len(group), i+4)
        ]

        if len(local) >= 4:

            _, d2rho, _, _ = (
                local_derivative(
                    local["T"].values,
                    local["rho"].values,
                    root,
                    npoints=min(
                        5,
                        len(local)
                    )
                )
            )

        else:

            d2rho = np.nan

        is_maximum = (
            np.isfinite(d2rho)
            and
            d2rho < 0
        )

        alpha_roots.append({
            "P": P,
            "T_root": root,
            "d2rho_dT2": d2rho,
            "density_maximum": is_maximum,
        })


alpha_roots = pd.DataFrame(
    alpha_roots
)


alpha_root_file = (
    OUTDIR
    /
    "alpha_zero_TMD_candidates.dat"
)


alpha_roots.to_csv(
    alpha_root_file,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 20. RESPONSE MAXIMUM REFINEMENT
# ============================================================

def refine_local_maximum(
    x,
    y,
    idx
):
    """
    Refine a candidate maximum with a local quadratic fit to
    the UNSMOOTHED response data.
    """

    n = len(x)

    half = (
        EXTREMUM_LOCAL_POINTS
        //
        2
    )

    i0 = max(
        0,
        idx-half
    )

    i1 = min(
        n,
        idx+half+1
    )

    xx = np.asarray(
        x[i0:i1],
        dtype=float
    )

    yy = np.asarray(
        y[i0:i1],
        dtype=float
    )

    good = (
        np.isfinite(xx)
        &
        np.isfinite(yy)
    )

    xx = xx[good]
    yy = yy[good]

    if len(xx) < 4:

        return None

    coeff = np.polyfit(
        xx,
        yy,
        2
    )

    a, b, c = coeff

    if a >= 0:

        return None

    xmax = (
        -b
        /
        (2*a)
    )

    if not (
        xx.min()
        <=
        xmax
        <=
        xx.max()
    ):

        return None

    ymax = np.polyval(
        coeff,
        xmax
    )

    fit = np.polyval(
        coeff,
        xx
    )

    ss_res = np.sum(
        (yy-fit)**2
    )

    ss_tot = np.sum(
        (yy-np.mean(yy))**2
    )

    if ss_tot > 0:

        R2 = (
            1.0
            -
            ss_res/ss_tot
        )

    else:

        R2 = np.nan

    return (
        xmax,
        ymax,
        R2
    )


# ============================================================
# 21. LOCI OF CP AND KAPPA MAXIMA ALONG ISOBARS
# ============================================================

def response_locus(
    dataframe,
    column
):

    results = []

    for P, group in dataframe.groupby("P"):

        group = group.sort_values(
            "T"
        )

        T = group["T"].values
        Y = group[column].values

        good = (
            np.isfinite(T)
            &
            np.isfinite(Y)
        )

        T = T[good]
        Y = Y[good]

        if len(T) < MIN_LOCUS_POINTS:
            continue

        smooth = gaussian_filter1d(
            Y,
            sigma=LOCUS_SMOOTH_SIGMA,
            mode="nearest"
        )

        yrange = (
            np.nanmax(smooth)
            -
            np.nanmin(smooth)
        )

        prominence = (
            PROMINENCE_FRACTION
            *
            yrange
        )

        if prominence <= 0:
            continue

        peaks, properties = find_peaks(
            smooth,
            prominence=prominence
        )

        for idx in peaks:

            refined = (
                refine_local_maximum(
                    T,
                    Y,
                    idx
                )
            )

            if refined is None:
                continue

            Tmax, Ymax, R2 = refined

            results.append({
                "P": P,
                "T_extremum": Tmax,
                "response_extremum": Ymax,
                "R2_local": R2,
            })

    return pd.DataFrame(
        results
    )


cp_locus = response_locus(
    resp,
    "cp"
)


kappa_locus = response_locus(
    resp,
    "kappa"
)


cp_locus_file = (
    OUTDIR
    /
    "cp_maxima_candidates.dat"
)


kappa_locus_file = (
    OUTDIR
    /
    "kappa_maxima_candidates.dat"
)


cp_locus.to_csv(
    cp_locus_file,
    sep=" ",
    index=False,
    float_format="%.10g"
)


kappa_locus.to_csv(
    kappa_locus_file,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 22. REPRESENTATIVE RESPONSE CURVES
# ============================================================

def nearest_pressure(
    target
):

    Pvalues = np.asarray(
        sorted(
            resp["P"].unique()
        )
    )

    return Pvalues[
        np.argmin(
            abs(Pvalues-target)
        )
    ]


# ------------------------------------------------------------
# cp(T)|P
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.4, 5.8)
)


for target in REPRESENTATIVE_PRESSURES:

    P = nearest_pressure(
        target
    )

    g = resp[
        np.isclose(
            resp["P"],
            P
        )
    ].sort_values(
        "T"
    )

    ax.plot(
        g["T"],
        g["cp"],
        "o-",
        ms=4,
        lw=1.1,
        label=rf"$P^*={P:.2f}$"
    )


ax.set_xlabel(
    r"$T^*$"
)

ax.set_ylabel(
    r"$c_P^*$"
)

ax.grid(
    alpha=0.15
)

ax.legend(
    frameon=False,
    ncol=2
)

fig.tight_layout()

fig.savefig(
    PLOTDIR / "cp_isobars.pdf",
    bbox_inches="tight"
)

fig.savefig(
    PLOTDIR / "cp_isobars.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# kappa(T)|P
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.4, 5.8)
)


for target in REPRESENTATIVE_PRESSURES:

    P = nearest_pressure(
        target
    )

    g = resp[
        np.isclose(
            resp["P"],
            P
        )
    ].sort_values(
        "T"
    )

    ax.plot(
        g["T"],
        g["kappa"],
        "o-",
        ms=4,
        lw=1.1,
        label=rf"$P^*={P:.2f}$"
    )


ax.set_xlabel(
    r"$T^*$"
)

ax.set_ylabel(
    r"$\kappa_T^*$"
)

ax.grid(
    alpha=0.15
)

ax.legend(
    frameon=False,
    ncol=2
)

fig.tight_layout()

fig.savefig(
    PLOTDIR / "kappa_isobars.pdf",
    bbox_inches="tight"
)

fig.savefig(
    PLOTDIR / "kappa_isobars.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# alpha(T)|P
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.4, 5.8)
)


for target in REPRESENTATIVE_PRESSURES:

    P = nearest_pressure(
        target
    )

    g = resp[
        np.isclose(
            resp["P"],
            P
        )
    ].sort_values(
        "T"
    )

    ax.plot(
        g["T"],
        g["alpha"],
        "o-",
        ms=4,
        lw=1.1,
        label=rf"$P^*={P:.2f}$"
    )


ax.axhline(
    0,
    color="black",
    ls="--",
    lw=1.2
)


ax.set_xlabel(
    r"$T^*$"
)

ax.set_ylabel(
    r"$\alpha_P^*$"
)

ax.grid(
    alpha=0.15
)

ax.legend(
    frameon=False,
    ncol=2
)

fig.tight_layout()

fig.savefig(
    PLOTDIR / "alpha_isobars.pdf",
    bbox_inches="tight"
)

fig.savefig(
    PLOTDIR / "alpha_isobars.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 23. 2D RESPONSE MAPS
# ============================================================

def response_map(
    dataframe,
    column,
    ylabel
):

    g = dataframe[
        np.isfinite(
            dataframe[column]
        )
    ]

    if len(g) < 10:
        return

    T = g["T"].values
    P = g["P"].values
    Z = g[column].values

    Ti = np.linspace(
        T.min(),
        T.max(),
        250
    )

    Pi = np.linspace(
        P.min(),
        P.max(),
        250
    )

    TT, PP = np.meshgrid(
        Ti,
        Pi
    )

    ZZ = griddata(
        (T, P),
        Z,
        (TT, PP),
        method="linear"
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 5.8)
    )

    cs = ax.contourf(
        TT,
        PP,
        ZZ,
        levels=25,
        cmap="viridis"
    )

    cbar = fig.colorbar(
        cs,
        ax=ax
    )

    cbar.set_label(
        ylabel
    )

    ax.scatter(
        T,
        P,
        s=5,
        color="black",
        alpha=0.10
    )

    ax.set_xlabel(
        r"$T^*$"
    )

    ax.set_ylabel(
        r"$P^*$"
    )

    fig.tight_layout()

    fig.savefig(
        PLOTDIR
        /
        f"{column}_map.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        PLOTDIR
        /
        f"{column}_map.png",
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fig)


response_map(
    resp,
    "cp",
    r"$c_P^*$"
)


response_map(
    resp,
    "kappa",
    r"$\kappa_T^*$"
)


response_map(
    resp,
    "alpha",
    r"$\alpha_P^*$"
)


# ============================================================
# 24. CONVERGENCE DIAGNOSTIC
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.4, 5.8)
)


sc = ax.scatter(
    df["T"],
    df["P"],
    c=np.log10(
        np.maximum(
            df["max_drift"],
            1.0e-12
        )
    ),
    s=18,
    cmap="viridis"
)


cbar = fig.colorbar(
    sc,
    ax=ax
)

cbar.set_label(
    r"$\log_{10}$(relative tail drift)"
)


ax.set_xlabel(
    r"$T^*$"
)

ax.set_ylabel(
    r"$P^*$"
)


fig.tight_layout()


fig.savefig(
    PLOTDIR
    /
    "equilibrium_drift_map.pdf",
    bbox_inches="tight"
)


fig.savefig(
    PLOTDIR
    /
    "equilibrium_drift_map.png",
    dpi=400,
    bbox_inches="tight"
)


plt.close(fig)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print()
print("="*80)
print("FILES WRITTEN")
print("="*80)

print(
    f"Equilibrium table:"
    f"\n  {equilibrium_file}"
)

print(
    f"\nResponse functions:"
    f"\n  {response_file}"
)

print(
    f"\nalpha=0 / TMD candidates:"
    f"\n  {alpha_root_file}"
)

print(
    f"\ncp maxima:"
    f"\n  {cp_locus_file}"
)

print(
    f"\nkappa maxima:"
    f"\n  {kappa_locus_file}"
)

print(
    f"\nPlots:"
    f"\n  {PLOTDIR}"
)

print()
print("="*80)
print("IMPORTANT INTERPRETATION")
print("="*80)

print(
    "1. alpha=0 candidates are mathematically equivalent to "
    "density extrema obtained from rho(T)|P."
)

print(
    "2. Only roots with negative d2rho/dT2 correspond to "
    "density maxima (TMD)."
)

print(
    "3. cp and kappa extrema are only CANDIDATES at this "
    "stage and must be visually/statistically validated."
)

print(
    "4. Smooth curves/maps are diagnostic tools; they should "
    "not automatically define thermodynamic loci."
)
