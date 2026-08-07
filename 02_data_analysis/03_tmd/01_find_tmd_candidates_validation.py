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

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE = RAW_DATA_ROOT / "allpress"

OUTDIR = DERIVED_DATA_ROOT / "thermo_response" / "TMD_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUTDIR / "TMD_all_pressures_validation.dat"
OUT_VALID = OUTDIR / "TMD_validated_full_locus.dat"
OUT_PDF = OUTDIR / "TMD_validated_full_locus.pdf"
OUT_PNG = OUTDIR / "TMD_validated_full_locus.png"


# ============================================================
# 2. PRESSURE / TEMPERATURE DOMAIN
# ============================================================

P_MIN = 0.10
P_MAX = 3.00

T_MIN = 0.02
T_MAX = 0.60


# ============================================================
# 3. EQUILIBRIUM EXTRACTION
# ============================================================

TAIL_FRACTION = 0.20
MIN_TAIL_POINTS = 50

USE_MEDIAN = True


# ============================================================
# 4. TMD SEARCH DOMAIN
#
# We do NOT search over the whole 0.02--0.60 interval.
# The previously validated TMD lies in this region.
# ============================================================

TMD_SEARCH_MIN = 0.12
TMD_SEARCH_MAX = 0.36


# ============================================================
# 5. ROBUSTNESS TEST
# ============================================================

# Number of nearest temperature points tested around
# the candidate maximum
WINDOW_POINTS = [
    5,
    7,
    9,
]

# Quadratic is the physically preferred local representation.
# Cubic is included only as a robustness check.
POLY_DEGREES = [
    2,
    3,
]

MIN_SUCCESSFUL_FITS = 4

# Maximum variation in T_TMD among successful fits
MAX_T_SPREAD = 0.040

# Standard deviation among fitted maxima
MAX_T_STD = 0.018

# Local fit quality
MIN_INDIVIDUAL_R2 = 0.60
MIN_MEDIAN_R2 = 0.80

# Candidate must not sit at edge of available data
EDGE_MARGIN_POINTS = 2


# ============================================================
# 6. ADDITIONAL PHYSICAL CRITERIA
# ============================================================

# Require a visible density rise before and fall after
# the candidate maximum.
MIN_SIDE_POINTS = 2

# Minimum density contrast relative to local maximum.
# Kept deliberately weak.
MIN_RELATIVE_CONTRAST = 2.0e-4

# We know from previous analysis that the robust branch
# disappears before high pressures. Do not hard-code its end,
# but require continuity between neighboring validated points.
MAX_BRANCH_JUMP = 0.050


# ============================================================
# 7. PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 13,
    "axes.labelsize": 17,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# ============================================================
# 8. FILE PARSING
# ============================================================

def parse_PT(filename):

    name = Path(filename).name

    match = re.search(
        r"outvars_P_([0-9.]+)_T_([0-9.]+)\.profile",
        name
    )

    if match is None:
        return None

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


# ============================================================
# 9. READ EQUILIBRIUM DENSITY
# ============================================================

def read_equilibrium_density(filename):

    data = np.loadtxt(
        filename,
        comments="#"
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 5:
        return np.nan

    n = len(data)

    n_tail = max(
        MIN_TAIL_POINTS,
        int(TAIL_FRACTION*n)
    )

    n_tail = min(
        n_tail,
        n
    )

    rho = data[-n_tail:, 4]

    rho = rho[
        np.isfinite(rho)
    ]

    if len(rho) == 0:
        return np.nan

    if USE_MEDIAN:
        return float(np.median(rho))

    return float(np.mean(rho))


# ============================================================
# 10. BUILD GLOBAL rho(P,T) TABLE
# ============================================================

files = sorted(
    BASE.glob(
        "P_*/thermo/outvars_P_*_T_*.profile"
    )
)

if len(files) == 0:

    raise FileNotFoundError(
        f"No thermo files found under {BASE.resolve()}"
    )


rows = []


print()
print("=" * 80)
print("READING DENSITY ISOBARS")
print("=" * 80)


for i, filename in enumerate(
    files,
    start=1
):

    PT = parse_PT(
        filename
    )

    if PT is None:
        continue

    P, T = PT

    if not (
        P_MIN <= P <= P_MAX
        and
        T_MIN <= T <= T_MAX
    ):
        continue

    try:

        rho = read_equilibrium_density(
            filename
        )

    except Exception:

        continue

    if not np.isfinite(rho):
        continue

    rows.append({
        "P": P,
        "T": T,
        "rho": rho,
    })

    if i % 250 == 0:

        print(
            f"processed {i}/{len(files)} files..."
        )


df = pd.DataFrame(
    rows
)


if len(df) == 0:

    raise RuntimeError(
        "No valid density states loaded."
    )


# Average duplicates
df = (
    df
    .groupby(
        ["P", "T"],
        as_index=False
    )
    .agg({
        "rho": "mean"
    })
)


df = df.sort_values(
    ["P", "T"]
).reset_index(
    drop=True
)


print()
print(
    f"Unique states = {len(df)}"
)

print(
    f"P range = "
    f"{df['P'].min():.3f} -- {df['P'].max():.3f}"
)

print(
    f"T range = "
    f"{df['T'].min():.3f} -- {df['T'].max():.3f}"
)


# ============================================================
# 11. R2
# ============================================================

def calc_r2(y, yfit):

    ss_res = np.sum(
        (y-yfit)**2
    )

    ss_tot = np.sum(
        (y-np.mean(y))**2
    )

    if ss_tot <= 0:
        return np.nan

    return (
        1.0
        -
        ss_res/ss_tot
    )


# ============================================================
# 12. ONE LOCAL POLYNOMIAL MAXIMUM
# ============================================================

def local_maximum_fit(
    T,
    rho,
    candidate_index,
    npoints,
    degree
):

    T = np.asarray(
        T,
        dtype=float
    )

    rho = np.asarray(
        rho,
        dtype=float
    )


    T0 = T[
        candidate_index
    ]


    order = np.argsort(
        np.abs(T-T0)
    )


    use = order[
        :min(npoints, len(T))
    ]


    Tf = T[use]
    rf = rho[use]


    order2 = np.argsort(
        Tf
    )


    Tf = Tf[order2]
    rf = rf[order2]


    if len(Tf) <= degree:
        return None


    # Center coordinates for numerical stability
    Tc = np.mean(
        Tf
    )

    x = Tf - Tc


    try:

        coeff = np.polyfit(
            x,
            rf,
            degree
        )

    except Exception:

        return None


    poly = np.poly1d(
        coeff
    )


    fit = poly(
        x
    )


    R2 = calc_r2(
        rf,
        fit
    )


    if (
        np.isfinite(R2)
        and
        R2 < MIN_INDIVIDUAL_R2
    ):
        return None


    d1 = np.polyder(
        poly,
        1
    )


    d2 = np.polyder(
        poly,
        2
    )


    roots = d1.r


    candidates = []


    for root in roots:

        if abs(np.imag(root)) > 1.0e-10:
            continue


        xr = float(
            np.real(root)
        )


        Tr = xr + Tc


        # no extrapolation
        if not (
            Tf.min()
            <
            Tr
            <
            Tf.max()
        ):
            continue


        curvature = float(
            d2(
                xr
            )
        )


        # must be density maximum
        if curvature >= 0:
            continue


        rr = float(
            poly(
                xr
            )
        )


        candidates.append({
            "Tmax": Tr,
            "rho_max": rr,
            "curvature": curvature,
            "R2": R2,
            "window": npoints,
            "degree": degree,
        })


    if len(candidates) == 0:
        return None


    # choose stationary point closest to raw candidate
    best = min(
        candidates,
        key=lambda x: abs(
            x["Tmax"]-T0
        )
    )


    return best


# ============================================================
# 13. RAW CANDIDATE MAXIMUM
# ============================================================

def raw_candidate(T, rho):

    T = np.asarray(T)
    rho = np.asarray(rho)


    mask = (
        (T >= TMD_SEARCH_MIN)
        &
        (T <= TMD_SEARCH_MAX)
    )


    Ts = T[mask]
    rs = rho[mask]


    if len(Ts) < 5:
        return None


    idx_local = int(
        np.argmax(
            rs
        )
    )


    if (
        idx_local < EDGE_MARGIN_POINTS
        or
        idx_local >
        len(Ts)-1-EDGE_MARGIN_POINTS
    ):

        return None


    T_candidate = Ts[
        idx_local
    ]


    # locate index in full arrays
    idx_full = int(
        np.argmin(
            abs(T-T_candidate)
        )
    )


    return idx_full


# ============================================================
# 14. SIDE-CONTRAST TEST
# ============================================================

def density_contrast_test(
    T,
    rho,
    Tm
):

    T = np.asarray(T)
    rho = np.asarray(rho)


    idx = np.argmin(
        abs(T-Tm)
    )


    left = rho[
        max(0, idx-MIN_SIDE_POINTS):
        idx
    ]


    right = rho[
        idx+1:
        min(
            len(rho),
            idx+1+MIN_SIDE_POINTS
        )
    ]


    if (
        len(left) < MIN_SIDE_POINTS
        or
        len(right) < MIN_SIDE_POINTS
    ):
        return False


    rho0 = rho[idx]


    left_mean = np.mean(
        left
    )


    right_mean = np.mean(
        right
    )


    contrast_left = (
        rho0-left_mean
    ) / abs(rho0)


    contrast_right = (
        rho0-right_mean
    ) / abs(rho0)


    return (
        contrast_left > MIN_RELATIVE_CONTRAST
        and
        contrast_right > MIN_RELATIVE_CONTRAST
    )


# ============================================================
# 15. VALIDATE ONE ISOBAR
# ============================================================

def validate_isobar(
    T,
    rho
):

    T = np.asarray(
        T,
        dtype=float
    )

    rho = np.asarray(
        rho,
        dtype=float
    )


    good = (
        np.isfinite(T)
        &
        np.isfinite(rho)
    )


    T = T[good]
    rho = rho[good]


    order = np.argsort(
        T
    )


    T = T[order]
    rho = rho[order]


    if len(T) < 9:

        return {
            "status": "rejected",
            "reason": "too_few_points",
        }


    candidate_index = raw_candidate(
        T,
        rho
    )


    if candidate_index is None:

        return {
            "status": "rejected",
            "reason": "no_internal_raw_candidate",
        }


    fits = []


    for window in WINDOW_POINTS:

        for degree in POLY_DEGREES:

            result = local_maximum_fit(
                T,
                rho,
                candidate_index,
                window,
                degree
            )


            if result is not None:

                fits.append(
                    result
                )


    if len(fits) == 0:

        return {
            "status": "rejected",
            "reason": "all_fits_failed",
        }


    Tmax_values = np.array(
        [
            x["Tmax"]
            for x in fits
        ]
    )


    rho_values = np.array(
        [
            x["rho_max"]
            for x in fits
        ]
    )


    R2_values = np.array(
        [
            x["R2"]
            for x in fits
        ]
    )


    T_median = np.median(
        Tmax_values
    )


    T_std = (
        np.std(
            Tmax_values,
            ddof=1
        )
        if len(Tmax_values) > 1
        else 0.0
    )


    T_spread = (
        np.max(Tmax_values)
        -
        np.min(Tmax_values)
    )


    rho_median = np.median(
        rho_values
    )


    R2_median = np.nanmedian(
        R2_values
    )


    contrast_ok = density_contrast_test(
        T,
        rho,
        T_median
    )


    criteria = {
        "enough_fits":
            len(fits)
            >=
            MIN_SUCCESSFUL_FITS,

        "small_spread":
            T_spread
            <=
            MAX_T_SPREAD,

        "small_std":
            T_std
            <=
            MAX_T_STD,

        "good_R2":
            R2_median
            >=
            MIN_MEDIAN_R2,

        "density_contrast":
            contrast_ok,
    }


    accepted = all(
        criteria.values()
    )


    return {
        "status":
            "validated"
            if accepted
            else "candidate",

        "reason":
            "OK"
            if accepted
            else "robustness_failed",

        "T_TMD": T_median,

        "rho_TMD": rho_median,

        "T_std": T_std,

        "T_spread": T_spread,

        "R2_median": R2_median,

        "n_success": len(fits),

        "contrast_ok": contrast_ok,
    }


# ============================================================
# 16. RUN ALL PRESSURES
# ============================================================

rows = []


print()
print("=" * 80)
print("VALIDATING TMD PRESSURE BY PRESSURE")
print("=" * 80)


for P, group in df.groupby(
    "P"
):

    g = group.sort_values(
        "T"
    )


    result = validate_isobar(
        g["T"].values,
        g["rho"].values
    )


    row = {
        "P": P,
        "status": result["status"],
        "reason": result["reason"],
        "T_TMD": result.get(
            "T_TMD",
            np.nan
        ),
        "rho_TMD": result.get(
            "rho_TMD",
            np.nan
        ),
        "T_std": result.get(
            "T_std",
            np.nan
        ),
        "T_spread": result.get(
            "T_spread",
            np.nan
        ),
        "R2_median": result.get(
            "R2_median",
            np.nan
        ),
        "n_success": result.get(
            "n_success",
            np.nan
        ),
        "contrast_ok": result.get(
            "contrast_ok",
            False
        ),
    }


    rows.append(
        row
    )


summary = pd.DataFrame(
    rows
)


# ============================================================
# 17. CONTINUITY FILTER
#
# This is intentionally applied only after local validation.
# It prevents isolated high-P extrema from being mistaken for
# continuation of the actual TMD branch.
# ============================================================

summary = summary.sort_values(
    "P"
).reset_index(
    drop=True
)


summary["branch_validated"] = False


validated_local = summary[
    summary["status"]
    ==
    "validated"
].copy()


if len(validated_local) > 0:

    validated_local = validated_local.sort_values(
        "P"
    )


    previous_T = None


    for idx, row in validated_local.iterrows():

        Tm = row["T_TMD"]


        if previous_T is None:

            summary.loc[
                idx,
                "branch_validated"
            ] = True

            previous_T = Tm

            continue


        if abs(
            Tm
            -
            previous_T
        ) <= MAX_BRANCH_JUMP:

            summary.loc[
                idx,
                "branch_validated"
            ] = True

            previous_T = Tm


# ============================================================
# 18. SAVE COMPLETE VALIDATION
# ============================================================

summary.to_csv(
    OUT_ALL,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 19. SAVE FINAL VALIDATED LOCUS
# ============================================================

final_tmd = summary[
    summary["branch_validated"]
].copy()


final_tmd = final_tmd[
    [
        "P",
        "T_TMD",
        "rho_TMD",
        "T_std",
        "T_spread",
        "R2_median",
        "n_success",
    ]
]


final_tmd.to_csv(
    OUT_VALID,
    sep=" ",
    index=False,
    float_format="%.10g"
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print()
print("=" * 100)
print("TMD VALIDATION SUMMARY")
print("=" * 100)


print(
    summary[
        [
            "P",
            "status",
            "branch_validated",
            "T_TMD",
            "T_std",
            "T_spread",
            "R2_median",
            "n_success",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print(
    f"Locally validated maxima = "
    f"{np.sum(summary['status']=='validated')}"
)


print(
    f"Final continuous TMD points = "
    f"{len(final_tmd)}"
)


# ============================================================
# 21. PLOT LOCUS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.4)
)


candidate = summary[
    (
        summary["status"]
        ==
        "candidate"
    )
    |
    (
        (
            summary["status"]
            ==
            "validated"
        )
        &
        (
            ~summary["branch_validated"]
        )
    )
]


if len(candidate) > 0:

    ax.scatter(
        candidate["T_TMD"],
        candidate["P"],
        marker="s",
        s=45,
        facecolors="none",
        edgecolors="0.5",
        label="candidate / disconnected"
    )


if len(final_tmd) > 0:

    ax.errorbar(
        final_tmd["T_TMD"],
        final_tmd["P"],
        xerr=final_tmd["T_std"],
        fmt="o-",
        ms=5.5,
        lw=1.5,
        capsize=3,
        label="validated TMD"
    )


ax.set_xlabel(
    r"$T^*_{\mathrm{TMD}}$"
)


ax.set_ylabel(
    r"$P^*$"
)


ax.grid(
    alpha=0.15
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
# 22. REPORT
# ============================================================

print()
print("=" * 80)
print("OUTPUT")
print("=" * 80)


print(
    f"Complete validation:\n  {OUT_ALL}"
)


print(
    f"\nValidated full TMD locus:\n  {OUT_VALID}"
)


print(
    f"\nFigure:\n  {OUT_PDF}"
)


print()
print(
    "Use TMD_validated_full_locus.dat for the phase diagram."
)
