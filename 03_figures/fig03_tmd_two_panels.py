#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator, UnivariateSpline



from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT, FIGURE_OUTPUT_ROOT

# ============================================================
# PATHS
# ============================================================

BASE = RAW_DATA_ROOT / "allpress"

OUTDIR = FIGURE_OUTPUT_ROOT / "fig03_tmd"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTDIR / "TMD_final_publication_joined.pdf"
OUT_PNG = OUTDIR / "TMD_final_publication_joined.png"
OUT_DATA = OUTDIR / "TMD_final_values.dat"


# ============================================================
# PRESSURES
# ============================================================

P_LEFT = [
    0.50,
    0.80,
    1.00,
    1.20,
    1.50,
    1.70,
]

P_RIGHT = [
    2.20,
    2.40,
    2.50,
    2.60,
    2.80,
    3.00,
]


# ============================================================
# TEMPERATURE RANGE
# ============================================================

TMIN = 0.02
TMAX = 0.60


# ============================================================
# LOCAL TMD FIT WINDOWS
# ============================================================

FIT_WINDOWS = {
    0.50: (0.13, 0.30),
    0.80: (0.14, 0.32),
    1.00: (0.15, 0.35),
    1.20: (0.16, 0.36),
    1.50: (0.17, 0.39),
    1.70: (0.17, 0.40),
}

MIN_POINTS_FIT = 5


# ============================================================
# PANEL (a): SMOOTHING
# ============================================================

GAUSSIAN_SIGMA_LEFT = 1.35

LEFT_SIGMA_OVERRIDE = {
    1.70: 1.80,
}


# ============================================================
# PANEL (b): ADAPTIVE SPLINE
# ============================================================

RIGHT_SPLINE_FACTOR = {
    2.20: 0.55,
    2.40: 0.55,
    2.50: 0.55,
    2.60: 0.50,
    2.80: 0.45,
    3.00: 0.40,
}

RIGHT_NOISE_SIGMA = 1.0

N_SMOOTH = 700


# ============================================================
# FIGURE SETTINGS
# ============================================================

FIGSIZE = (10.8, 4.8)

# Larger MD points
MARKER_SIZE = 44

SMOOTH_LW_LEFT = 1.85
SMOOTH_LW_RIGHT = 1.80

FIT_LW = 1.20

RAW_ALPHA_LEFT = 0.72
RAW_ALPHA_RIGHT = 0.72

# Small separation between panels
WSPACE = 0.015


# ============================================================
# LEGEND POSITION — PANEL (a)
# ============================================================

LEGEND_A_X = 0.225
LEGEND_A_Y = 0.018


# ============================================================
# MATPLOTLIB STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 13,

    "axes.labelsize": 20,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "legend.fontsize": 10.2,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 5,
    "ytick.major.size": 5,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})


# ============================================================
# FIND PRESSURE DIRECTORY
# ============================================================

def find_pressure_directory(P):

    candidates = [
        BASE / f"P_{P:.3f}",
        BASE / f"P_{P:.2f}",
        BASE / f"P_{P:.1f}",
    ]

    for path in candidates:

        if path.exists():
            return path

    matches = sorted(
        BASE.glob(f"P_{P}*")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"\nCould not find directory for P*={P:.3f}\n"
        f"inside:\n{BASE.resolve()}"
    )


# ============================================================
# EXTRACT TEMPERATURE
# ============================================================

def extract_temperature(filename):

    match = re.search(
        r"_T_([0-9]+(?:\.[0-9]+)?)",
        filename
    )

    if match is None:
        return None

    return float(
        match.group(1)
    )


# ============================================================
# READ DENSITY
# ============================================================

def read_density(filename):

    try:

        data = np.loadtxt(
            filename,
            comments="#"
        )

    except Exception:

        return np.nan

    if data.ndim == 1:

        data = data.reshape(
            1,
            -1
        )

    if data.shape[1] < 5:
        return np.nan

    rho = data[:, 4]

    rho = rho[
        np.isfinite(rho)
    ]

    if len(rho) == 0:
        return np.nan

    return np.mean(rho)


# ============================================================
# LOAD ISOBAR
# ============================================================

def load_isobar(P):

    pressure_dir = find_pressure_directory(P)

    thermo_dir = pressure_dir / "thermo"

    if not thermo_dir.exists():

        raise FileNotFoundError(
            f"thermo directory not found:\n{thermo_dir}"
        )

    files = sorted(
        thermo_dir.glob("outvars*.profile")
    )

    T_values = []
    rho_values = []

    for filename in files:

        T = extract_temperature(
            filename.name
        )

        if T is None:
            continue

        if not (
            TMIN <= T <= TMAX
        ):
            continue

        rho = read_density(
            filename
        )

        if not np.isfinite(rho):
            continue

        T_values.append(T)
        rho_values.append(rho)

    if len(T_values) == 0:

        raise RuntimeError(
            f"No valid states found for P*={P}"
        )

    T = np.asarray(
        T_values,
        dtype=float
    )

    rho = np.asarray(
        rho_values,
        dtype=float
    )

    order = np.argsort(T)

    T = T[order]
    rho = rho[order]

    # --------------------------------------------------------
    # Average duplicate temperatures
    # --------------------------------------------------------

    unique_T = np.unique(T)

    unique_rho = np.empty_like(
        unique_T
    )

    for i, temp in enumerate(unique_T):

        mask = np.isclose(
            T,
            temp
        )

        unique_rho[i] = np.mean(
            rho[mask]
        )

    return unique_T, unique_rho


# ============================================================
# QUADRATIC
# ============================================================

def quadratic(T, a, b, c):

    return (
        a*T*T
        +
        b*T
        +
        c
    )


# ============================================================
# LOCAL TMD FIT
# ============================================================

def local_tmd(P, T, rho):

    if P not in FIT_WINDOWS:
        return None

    Tmin_fit, Tmax_fit = FIT_WINDOWS[P]

    mask = (
        (T >= Tmin_fit)
        &
        (T <= Tmax_fit)
    )

    Tf = T[mask]
    rf = rho[mask]

    if len(Tf) < MIN_POINTS_FIT:
        return None

    try:

        parameters, covariance = curve_fit(
            quadratic,
            Tf,
            rf
        )

    except Exception:

        return None

    a, b, c = parameters

    # Maximum requires downward concavity
    if a >= 0:
        return None

    Tm = -b / (2.0*a)

    # No extrapolation
    if not (
        Tmin_fit < Tm < Tmax_fit
    ):
        return None

    rhom = quadratic(
        Tm,
        a,
        b,
        c
    )

    predicted = quadratic(
        Tf,
        a,
        b,
        c
    )

    ss_res = np.sum(
        (rf - predicted)**2
    )

    ss_tot = np.sum(
        (rf - np.mean(rf))**2
    )

    if ss_tot > 0:

        R2 = 1.0 - ss_res/ss_tot

    else:

        R2 = np.nan

    return {
        "T": Tm,
        "rho": rhom,
        "R2": R2,

        "a": a,
        "b": b,
        "c": c,

        "Tf": Tf,
        "rf": rf,
    }


# ============================================================
# PANEL (a): GAUSSIAN + PCHIP
# ============================================================

def smooth_left_isobar(P, T, rho):

    sigma = LEFT_SIGMA_OVERRIDE.get(
        P,
        GAUSSIAN_SIGMA_LEFT
    )

    rho_filtered = gaussian_filter1d(
        rho,
        sigma=sigma,
        mode="nearest"
    )

    interpolator = PchipInterpolator(
        T,
        rho_filtered,
        extrapolate=False
    )

    T_dense = np.linspace(
        T.min(),
        T.max(),
        N_SMOOTH
    )

    rho_dense = interpolator(
        T_dense
    )

    return T_dense, rho_dense


# ============================================================
# PANEL (b): ADAPTIVE SPLINE
# ============================================================

def smooth_right_isobar(P, T, rho):

    local_background = gaussian_filter1d(
        rho,
        sigma=RIGHT_NOISE_SIGMA,
        mode="nearest"
    )

    residual_noise = (
        rho
        -
        local_background
    )

    noise_sigma = np.std(
        residual_noise
    )

    if not np.isfinite(noise_sigma):
        noise_sigma = 0.0

    if noise_sigma <= 0.0:

        noise_sigma = (
            1.0e-6
            *
            max(
                abs(np.mean(rho)),
                1.0
            )
        )

    factor = RIGHT_SPLINE_FACTOR.get(
        P,
        0.50
    )

    smoothing_parameter = (
        factor
        *
        len(T)
        *
        noise_sigma**2
    )

    spline = UnivariateSpline(
        T,
        rho,
        k=3,
        s=smoothing_parameter
    )

    T_dense = np.linspace(
        T.min(),
        T.max(),
        N_SMOOTH
    )

    rho_dense = spline(
        T_dense
    )

    return T_dense, rho_dense


# ============================================================
# LOAD DATA
# ============================================================

pressures = (
    P_LEFT
    +
    P_RIGHT
)

data = {}

print()
print("=" * 76)
print("READING ISOBARS")
print("=" * 76)

for P in pressures:

    T, rho = load_isobar(P)

    data[P] = {
        "T": T,
        "rho": rho
    }

    print(
        f"P*={P:5.2f}: "
        f"{len(T):3d} states, "
        f"T=[{T.min():.3f}, {T.max():.3f}]"
    )


# ============================================================
# TMD VALUES
# ============================================================

TMD = {}

print()
print("=" * 76)
print("LOCAL TMD FITS")
print("=" * 76)

for P in P_LEFT:

    T = data[P]["T"]
    rho = data[P]["rho"]

    result = local_tmd(
        P,
        T,
        rho
    )

    TMD[P] = result

    if result is None:

        print(
            f"P*={P:.2f}: "
            "no valid local maximum"
        )

    else:

        print(
            f"P*={P:.2f}: "
            f"T_TMD*={result['T']:.5f}   "
            f"R²={result['R2']:.4f}"
        )


# ============================================================
# CREATE JOINED FIGURE
# ============================================================

fig, (
    ax1,
    ax2
) = plt.subplots(
    1,
    2,
    figsize=FIGSIZE,
    sharex=True,
    sharey=True,
    gridspec_kw={
        "wspace": WSPACE
    }
)


# ============================================================
# COLORS
# ============================================================

cmap = plt.get_cmap(
    "viridis"
)

colors_left = cmap(
    np.linspace(
        0.06,
        0.72,
        len(P_LEFT)
    )
)

colors_right = cmap(
    np.linspace(
        0.34,
        0.96,
        len(P_RIGHT)
    )
)


# ============================================================
# PANEL (a)
# ============================================================

for color, P in zip(
    colors_left,
    P_LEFT
):

    T = data[P]["T"]
    rho = data[P]["rho"]

    rho_ref = np.max(rho)

    rho_norm = (
        rho
        /
        rho_ref
    )

    T_smooth, rho_smooth = smooth_left_isobar(
        P,
        T,
        rho
    )

    rho_smooth_norm = (
        rho_smooth
        /
        rho_ref
    )

    # Raw MD points
    ax1.scatter(
        T,
        rho_norm,

        s=MARKER_SIZE,

        color=color,

        alpha=RAW_ALPHA_LEFT,

        edgecolors="none",

        zorder=5,

        label=rf"$P^*={P:.2f}$"
    )

    # Smooth guide
    ax1.plot(
        T_smooth,
        rho_smooth_norm,

        color=color,

        lw=SMOOTH_LW_LEFT,

        alpha=0.97,

        zorder=3
    )

    # TMD result
    result = TMD[P]

    if result is None:
        continue

    Tm = result["T"]

    rhom_norm = (
        result["rho"]
        /
        rho_ref
    )

    # Local quadratic segment
    Tfit_dense = np.linspace(
        result["Tf"].min(),
        result["Tf"].max(),
        250
    )

    rho_fit_dense = quadratic(
        Tfit_dense,
        result["a"],
        result["b"],
        result["c"]
    )

    rho_fit_dense /= rho_ref

    ax1.plot(
        Tfit_dense,
        rho_fit_dense,

        color=color,

        lw=FIT_LW,

        linestyle="--",

        alpha=0.78,

        zorder=4
    )

    # TMD marker
    ax1.scatter(
        Tm,
        rhom_norm,

        s=55,

        marker="o",

        facecolors="white",

        edgecolors=color,

        linewidths=1.7,

        zorder=8
    )


# ============================================================
# PANEL (b)
# ============================================================

for color, P in zip(
    colors_right,
    P_RIGHT
):

    T = data[P]["T"]
    rho = data[P]["rho"]

    rho_ref = np.max(rho)

    rho_norm = (
        rho
        /
        rho_ref
    )

    T_smooth, rho_smooth = smooth_right_isobar(
        P,
        T,
        rho
    )

    rho_smooth_norm = (
        rho_smooth
        /
        rho_ref
    )

    ax2.scatter(
        T,
        rho_norm,

        s=MARKER_SIZE,

        color=color,

        alpha=RAW_ALPHA_RIGHT,

        edgecolors="none",

        zorder=5,

        label=rf"$P^*={P:.2f}$"
    )

    ax2.plot(
        T_smooth,
        rho_smooth_norm,

        color=color,

        lw=SMOOTH_LW_RIGHT,

        alpha=0.90,

        zorder=3
    )


# ============================================================
# PANEL LABELS
# ============================================================

ax1.text(
    0.025,
    0.965,

    r"(a)",

    transform=ax1.transAxes,

    ha="left",
    va="top",

    fontsize=17
)

ax2.text(
    0.025,
    0.965,

    r"(b)",

    transform=ax2.transAxes,

    ha="left",
    va="top",

    fontsize=17
)


# ============================================================
# AXES
# ============================================================

for ax in (
    ax1,
    ax2
):

    ax.set_xlim(
        TMIN,
        TMAX
    )

    ax.grid(
        alpha=0.09,
        linewidth=0.55
    )


# ============================================================
# COMMON Y RANGE
# ============================================================

all_normalized = []

for P in pressures:

    rho = data[P]["rho"]

    all_normalized.extend(
        rho
        /
        np.max(rho)
    )

all_normalized = np.asarray(
    all_normalized
)

ymin = max(
    0.90,
    np.nanmin(all_normalized)
    -
    0.004
)

ymax = 1.006

ax1.set_ylim(
    ymin,
    ymax
)


# ============================================================
# JOINED-PANEL AXIS STYLE
# ============================================================

# Panel (b) uses the same y scale, so labels are unnecessary
ax2.tick_params(
    axis="y",
    labelleft=False
)


# ============================================================
# AXIS LABELS
# ============================================================

# Larger vertical label
ax1.set_ylabel(
    r"$\rho^*/\rho_{\max}^*$",
    fontsize=22,
    labelpad=10
)

# Single horizontal label for the whole figure
ax1.set_xlabel("")
ax2.set_xlabel("")

fig.supxlabel(
    r"$T^*$",
    fontsize=24,
    y=0.020
)


# ============================================================
# LEGENDS
# ============================================================

ax1.legend(
    frameon=False,

    loc="lower left",

    bbox_to_anchor=(
        LEGEND_A_X,
        LEGEND_A_Y
    ),

    bbox_transform=ax1.transAxes,

    ncol=2,

    handlelength=1.0,

    handletextpad=0.45,

    columnspacing=0.85,

    labelspacing=0.30,

    borderaxespad=0.0,

    fontsize=10.2
)


ax2.legend(
    frameon=False,

    loc="lower left",

    ncol=2,

    handlelength=1.0,

    handletextpad=0.45,

    columnspacing=0.90,

    labelspacing=0.30,

    borderaxespad=0.45,

    fontsize=10.2
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.090,
    right=0.985,
    bottom=0.19,
    top=0.97,
    wspace=WSPACE
)


# ============================================================
# SAVE
# ============================================================

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


# ============================================================
# SAVE TMD VALUES
# ============================================================

with open(
    OUT_DATA,
    "w"
) as file:

    file.write(
        "# P T_TMD rho_TMD R2\n"
    )

    for P in P_LEFT:

        result = TMD[P]

        if result is None:
            continue

        file.write(
            f"{P:.8f} "
            f"{result['T']:.10f} "
            f"{result['rho']:.10f} "
            f"{result['R2']:.8f}\n"
        )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 76)
print("OUTPUT")
print("=" * 76)

print(f"PDF : {OUT_PDF}")
print(f"PNG : {OUT_PNG}")
print(f"TMD : {OUT_DATA}")

print()
print(
    f"Marker size = {MARKER_SIZE}"
)

print(
    f"Panel spacing WSPACE = {WSPACE}"
)

print()
print(
    "Axis font sizes:"
)

print(
    "  x label = 24"
)

print(
    "  y label = 22"
)

print(
    "  tick labels = 14"
)
