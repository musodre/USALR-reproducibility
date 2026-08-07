#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intermediate-range SALR correlations

Panel (a):
    Representative S(k*) curves at T*=0.20 for selected pressures.

Panel (b):
    Original A_SALR(P*) along representative isotherms.

Definition:
    A_SALR = integral_{1.30}^{3.00} S(k*) dk*

IMPORTANT:
    A_SALR is NOT recalculated.
    Values are read from the original SALR_area_summary.dat.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



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

ASALR_FILE = DERIVED_DATA_ROOT / "static_structure_factor" / "Asalr_analysis" / "SALR_area_summary.dat"

SK_DIR = DERIVED_DATA_ROOT / "static_structure_factor" / "Sk_files"

OUTDIR = FIGURE_OUTPUT_ROOT / "fig10_structure_factor_ASALR"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTDIR / "intermediate_salr_structure.pdf"
OUT_PNG = OUTDIR / "intermediate_salr_structure.png"


# =============================================================================
# 2. PHYSICAL SETTINGS
# =============================================================================

KMIN_SALR = 1.30
KMAX_SALR = 3.00

T_SK = 0.20

P_SK_SELECTED = [
    0.10,
    0.60,
    1.00,
    2.00,
    4.00,
    6.00,
]

T_SELECTED = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]

P_MIN = 0.10
P_MAX = 6.00

K_PLOT_MIN = 0.20
K_PLOT_MAX = 6.00

T_TOL = 1.0e-6


# =============================================================================
# 3. GRAPHICAL SETTINGS
# =============================================================================

FIGSIZE = (10.9, 5.3)

CMAP = "plasma"

LINEWIDTH = 1.80
LINE_ALPHA = 0.92

MARKER_SIZE = 39
MARKER_EDGEWIDTH = 0.50

LABEL_FS = 18
TICK_FS = 11.5
PANEL_FS = 16
LEGEND_FS = 12

CBAR_LABEL_FS = 18
CBAR_TICK_FS = 11.5


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "axes.linewidth": 1.15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


# =============================================================================
# 4. CHECK INPUTS
# =============================================================================

if not ASALR_FILE.exists():
    raise FileNotFoundError(
        f"A_SALR file not found:\n{ASALR_FILE.resolve()}"
    )

if not SK_DIR.exists():
    raise FileNotFoundError(
        f"S(k) directory not found:\n{SK_DIR.resolve()}"
    )


# =============================================================================
# 5. READ ORIGINAL A_SALR DATA
# =============================================================================

asalr = pd.read_csv(
    ASALR_FILE,
    sep=r"\s+",
    comment="#",
    header=None,
    names=["P", "T", "A_SALR", "D", "lnD"],
)

for col in asalr.columns:
    asalr[col] = pd.to_numeric(asalr[col], errors="coerce")

asalr = asalr.replace([np.inf, -np.inf], np.nan)

asalr = asalr.dropna(
    subset=["P", "T", "A_SALR"]
).copy()

asalr = asalr[
    (asalr["P"] >= P_MIN)
    & (asalr["P"] <= P_MAX)
].copy()

asalr = asalr.sort_values(
    ["T", "P"]
).reset_index(drop=True)


print()
print("=" * 92)
print("ORIGINAL INTERMEDIATE-RANGE SALR DATA")
print("=" * 92)

print(f"Input:\n  {ASALR_FILE.resolve()}")
print()
print(f"States = {len(asalr)}")
print(f"P range = {asalr['P'].min():.3f} -- {asalr['P'].max():.3f}")
print(f"T range = {asalr['T'].min():.3f} -- {asalr['T'].max():.3f}")
print(
    f"A_SALR range = "
    f"{asalr['A_SALR'].min():.6f} -- "
    f"{asalr['A_SALR'].max():.6f}"
)
print(
    f"SALR integration interval: "
    f"{KMIN_SALR:.2f} <= k* <= {KMAX_SALR:.2f}"
)


# =============================================================================
# 6. SELECT A_SALR ISOTHERMS
# =============================================================================

frames = []

print()
print("=" * 92)
print("SELECTED A_SALR ISOTHERMS")
print("=" * 92)

for T0 in T_SELECTED:

    iso = asalr[
        np.isclose(
            asalr["T"].to_numpy(dtype=float),
            T0,
            atol=T_TOL
        )
    ].copy()

    if len(iso) == 0:
        print(f"WARNING: T*={T0:.2f} not found.")
        continue

    iso = iso.sort_values("P")
    iso["T_selected"] = T0

    frames.append(iso)

    print(
        f"T*={T0:.2f}: "
        f"{len(iso)} states"
    )


if len(frames) == 0:
    raise RuntimeError(
        "No requested A_SALR isotherms were found."
    )

asalr_plot = pd.concat(
    frames,
    ignore_index=True
)


# =============================================================================
# 7. S(k) FILE NAME
# =============================================================================

def sk_filename(P_value, T_value):
    return SK_DIR / f"Sk_P_{P_value:.3f}_T_{T_value:.2f}.dat"


# =============================================================================
# 8. ROBUST S(k) READER
# =============================================================================

def read_sk(filename):
    """
    Read first two numerical columns as:
        column 0 -> k*
        column 1 -> S(k*)
    """

    arr = np.loadtxt(
        filename,
        comments="#",
        dtype=float,
        ndmin=2,
    )

    if arr.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns in {filename}"
        )

    x = np.asarray(
        arr[:, 0],
        dtype=float
    )

    y = np.asarray(
        arr[:, 1],
        dtype=float
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if x.size < 3:
        raise ValueError(
            f"Too few valid points in {filename}"
        )

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    return x, y


# =============================================================================
# 9. LOAD REPRESENTATIVE S(k) CURVES
# =============================================================================

sk_curves = []

print()
print("=" * 92)
print("REPRESENTATIVE S(k) STATES")
print("=" * 92)


for P0 in P_SK_SELECTED:

    filename = sk_filename(
        P0,
        T_SK
    )

    if not filename.exists():

        print(
            f"WARNING: S(k) file not found for "
            f"P*={P0:.2f}, T*={T_SK:.2f}"
        )

        continue


    try:
        xdata, ydata = read_sk(filename)

    except Exception as exc:

        print(
            f"WARNING: could not read {filename}:\n"
            f"  {exc}"
        )

        continue


    sk_curves.append({
        "P": P0,
        "T": T_SK,
        "x": xdata,
        "y": ydata,
        "file": filename,
    })


    print(
        f"P*={P0:.2f}, T*={T_SK:.2f}\n"
        f"  {filename}\n"
        f"  points     = {xdata.size}\n"
        f"  k range    = {xdata.min():.4f} -- {xdata.max():.4f}\n"
        f"  S(k) range = {ydata.min():.4f} -- {ydata.max():.4f}"
    )


if len(sk_curves) == 0:
    raise RuntimeError(
        "No representative S(k) curves could be loaded."
    )


# =============================================================================
# 10. COLOR MAPS
# =============================================================================

cmap = plt.get_cmap(CMAP)

pressure_norm = Normalize(
    vmin=min(P_SK_SELECTED),
    vmax=max(P_SK_SELECTED),
)

temperature_norm = Normalize(
    vmin=min(T_SELECTED),
    vmax=max(T_SELECTED),
)


# =============================================================================
# 11. FIGURE
# =============================================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=FIGSIZE
)


# =============================================================================
# 12. PANEL (a): S(k*)
# =============================================================================

ax1.axvspan(
    KMIN_SALR,
    KMAX_SALR,
    color="0.82",
    alpha=0.22,
    zorder=0,)

ax1.axvline(
    KMIN_SALR,
    linestyle="--",
    linewidth=1.0,
    color="0.40",
    zorder=1,
)

ax1.axvline(
    KMAX_SALR,
    linestyle="--",
    linewidth=1.0,
    color="0.40",
    zorder=1,
)


for item in sk_curves:

    pressure = item["P"]
    xdata = item["x"]
    ydata = item["y"]

    mask = (
        (xdata >= K_PLOT_MIN)
        & (xdata <= K_PLOT_MAX)
    )

    color = cmap(
        pressure_norm(
            pressure
        )
    )

    ax1.plot(
        xdata[mask],
        ydata[mask],
        linewidth=LINEWIDTH,
        alpha=LINE_ALPHA,
        color=color,
        label=rf"$P^*={pressure:.2f}$",
        zorder=3,
    )


ax1.text(0.6 * (KMIN_SALR+ KMAX_SALR),
    0.785,
    r"$A_{\mathrm{SALR}}$ window",
    transform=ax1.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=18, rotation=90, zorder=10)


ax1.set_xlim(
    K_PLOT_MIN,
    K_PLOT_MAX
)

ax1.set_xlabel(
    r"$k^*$"
)

ax1.set_ylabel(
    r"$S(k^*)$"
)

ax1.text(
    0.035,
    0.955,
    r"\textbf{(a)}",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=PANEL_FS,
)


ax1.legend(
    loc="best",
    fontsize=LEGEND_FS,
    frameon=True,
    framealpha=0.94,
    facecolor="white",
    edgecolor="0.70",
    ncol=1,labelspacing=0.32,
    borderpad=0.40,
    handletextpad=0.50
)


# =============================================================================
# 13. PANEL (b): A_SALR(P*)
# =============================================================================

for T0 in T_SELECTED:

    iso = asalr_plot[
        np.isclose(
            asalr_plot["T_selected"],
            T0,
            atol=T_TOL
        )
    ].copy()

    if len(iso) == 0:
        continue

    iso = iso.sort_values(
        "P"
    )

    color = cmap(
        temperature_norm(
            T0
        )
    )

    ax2.plot(
        iso["P"],
        iso["A_SALR"],
        color=color,
        linewidth=LINEWIDTH,
        alpha=LINE_ALPHA,
        zorder=2,
    )

    ax2.scatter(
        iso["P"],
        iso["A_SALR"],
        s=MARKER_SIZE,
        facecolor=color,
        edgecolor="black",
        linewidth=MARKER_EDGEWIDTH,
        alpha=0.97,
        zorder=3,
    )


ax2.set_xlim(
    P_MIN,
    P_MAX
)

ax2.set_xlabel(
    r"$P^*$"
)

ax2.set_ylabel(
    r"$A_{\mathrm{SALR}}$"
)

ax2.text(
    0.035,
    0.955,
    r"\textbf{(b)}",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=PANEL_FS,
)


# =============================================================================
# 14. COMMON FORMATTING
# =============================================================================

for ax in (ax1, ax2):

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1.0,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        alpha=0.23,
        zorder=0,
    )


# =============================================================================
# 15. LAYOUT
# =============================================================================

FIG_LEFT = 0.085
FIG_RIGHT = 0.850
FIG_BOTTOM = 0.145
FIG_TOP = 0.975

fig.subplots_adjust(
    left=FIG_LEFT,
    right=FIG_RIGHT,
    bottom=FIG_BOTTOM,
    top=FIG_TOP,
    wspace=0.27,
)


# =============================================================================
# 16. TEMPERATURE COLORBAR FOR PANEL (b)
# =============================================================================

sm = ScalarMappable(
    norm=temperature_norm,
    cmap=cmap
)

sm.set_array([])


CBAR_LEFT = 0.878
CBAR_WIDTH = 0.038


cax = fig.add_axes([
    CBAR_LEFT,
    FIG_BOTTOM,
    CBAR_WIDTH,
    FIG_TOP - FIG_BOTTOM,
])


cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation="vertical"
)


cbar.set_label(
    r"$T^*$",
    fontsize=CBAR_LABEL_FS,
    labelpad=11,
)


cbar.set_ticks(
    T_SELECTED
)


cbar.ax.tick_params(
    direction="in",
    length=4.5,
    width=0.9,
    labelsize=CBAR_TICK_FS,
    pad=5,
)


cbar.outline.set_linewidth(
    1.0
)


# =============================================================================
# 17. SAVE
# =============================================================================

fig.savefig(
    OUT_PDF,
    dpi=600
)

fig.savefig(
    OUT_PNG,
    dpi=600
)

plt.show()


# =============================================================================
# 18. FINAL REPORT
# =============================================================================

print()
print("=" * 92)
print("INTERMEDIATE-RANGE SALR FIGURE")
print("=" * 92)

print(
    f"Original A_SALR states = {len(asalr)}"
)

print(
    f"S(k) curves plotted    = {len(sk_curves)}"
)

print(
    f"S(k) temperature       = {T_SK:.2f}"
)

print(
    f"SALR k-window          = "
    f"{KMIN_SALR:.2f} -- {KMAX_SALR:.2f}"
)

print()

print("Representative pressures:")

for item in sk_curves:
    print(
        f"  P* = {item['P']:.2f}"
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

print()

print("✓ Original A_SALR values used directly.")
print("✓ A_SALR was not recalculated.")
print("✓ Exact original S(k) files were used.")
print("✓ S(k) files read with numpy.loadtxt.")
print("✓ No recursive file search.")
print("✓ No smoothing, fitting, interpolation, or extrapolation.")
print("✓ Panel (a): pressure varies at fixed T*=0.20.")
print("✓ Panel (b): temperature distinguishes A_SALR isotherms.")

print("=" * 92)
