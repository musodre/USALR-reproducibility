#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global pair-order descriptors versus dynamics
==============================================

Final main-text version.

Panels:
    (a) ln D* versus -s2*
    (b) ln D* versus tau

Main-text restriction:
    T* >= 0.20

Rationale:
    Focus on the mobile-fluid regime in which the diffusion anomaly
    is clearly developed, excluding the extremely sluggish
    low-temperature states from the visualization.

IMPORTANT:
    - The T* >= 0.20 restriction applies to the MAIN FIGURE.
    - Full statistical information for the selected regime is printed.
    - Linear and quadratic fits are calculated but NOT drawn.
    - Same temperature normalization is used in both panels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score



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
# 1. USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# Input files
# -------------------------------------------------------------------------

S2_FILE = DERIVED_DATA_ROOT / "s2" / "s2_global.dat"
TAU_FILE = DERIVED_DATA_ROOT / "tau" / "tau_global.dat"

# Diffusion file
#
# Expected columns:
#     P  T  D  ...
#
# Adjust ONLY this path if your diffusion file has another location.
#
D_FILE = DERIVED_DATA_ROOT / "asymptotic_diffusion" / "D_asymptotic_global.dat"


# -------------------------------------------------------------------------
# Main-text temperature window
# -------------------------------------------------------------------------

T_MIN = 0.20
T_MAX = 0.60


# -------------------------------------------------------------------------
# Numerical matching tolerance
# -------------------------------------------------------------------------

ROUND_DECIMALS = 8


# -------------------------------------------------------------------------
# Figure output
# -------------------------------------------------------------------------

OUTDIR = FIGURE_OUTPUT_ROOT / "fig07_diffusion_vs_global_order"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTDIR / "global_pair_order_vs_dynamics_Tmin020.pdf"
OUT_PNG = OUTDIR / "global_pair_order_vs_dynamics_Tmin020.png"


# -------------------------------------------------------------------------
# Figure aesthetics
# -------------------------------------------------------------------------

FIGSIZE = (10.2, 5.35)

POINT_SIZE = 29
POINT_ALPHA = 0.88
POINT_EDGEWIDTH = 0.35

CMAP = "cividis"

# Fixed temperature range so that the color scale has direct physical meaning
COLOR_TMIN = T_MIN
COLOR_TMAX = T_MAX

# Colorbar geometry
CBAR_PAD = 0.022
CBAR_FRACTION = 0.040
CBAR_ASPECT = 20

# Font sizes
LABEL_FS = 18
TICK_FS = 12
PANEL_FS = 16
CBAR_LABEL_FS = 16
CBAR_TICK_FS = 11
STATS_FS = 14.0


# =============================================================================
# 2. MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",

    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "axes.linewidth": 1.1,

    "legend.frameon": True,
    "legend.framealpha": 0.94,

    "savefig.bbox": "tight",
})


# =============================================================================
# 3. READING FUNCTIONS
# =============================================================================

def read_s2(filename):
    """
    Expected:
        P  T  rho  s2

    IMPORTANT:
        s2 is the FOURTH column.
    """

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["P", "T", "rho", "s2"],
    )

    for col in ["P", "T", "rho", "s2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    return df


def read_tau(filename):
    """
    Expected:
        P  T  rho  tau

    IMPORTANT:
        tau is the FOURTH column.

    Duplicate (P,T) states are averaged before matching with diffusion.
    """

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["P", "T", "rho", "tau"],
    )

    for col in ["P", "T", "rho", "tau"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Average duplicate states
    df = (
        df.groupby(["P", "T"], as_index=False)
          .agg({
              "rho": "mean",
              "tau": "mean",
          })
    )

    return df


def read_diffusion(filename):
    """
    Expected minimum:
        P  T  D

    Additional columns are ignored.
    """

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None,
    )

    if df.shape[1] < 3:
        raise ValueError(
            f"Diffusion file {filename} must contain at least 3 columns: P T D"
        )

    df = df.iloc[:, :3].copy()
    df.columns = ["P", "T", "D"]

    for col in ["P", "T", "D"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Physical requirement for ln(D)
    df = df[df["D"] > 0].copy()

    # Average duplicates, if any
    df = (
        df.groupby(["P", "T"], as_index=False)
          .agg({"D": "mean"})
    )

    df["lnD"] = np.log(df["D"])

    return df


# =============================================================================
# 4. STATE MATCHING
# =============================================================================

def prepare_keys(df):
    """
    Create rounded thermodynamic-state keys to avoid floating-point
    mismatches during merge.
    """

    out = df.copy()

    out["P_key"] = out["P"].round(ROUND_DECIMALS)
    out["T_key"] = out["T"].round(ROUND_DECIMALS)

    return out


def merge_descriptor_with_diffusion(desc, diffusion, descriptor_name):

    d1 = prepare_keys(desc)
    d2 = prepare_keys(diffusion)

    keep_desc = ["P_key", "T_key", "P", "T", descriptor_name]

    merged = pd.merge(
        d1[keep_desc],
        d2[["P_key", "T_key", "D", "lnD"]],
        on=["P_key", "T_key"],
        how="inner",
    )

    merged = merged.rename(
        columns={
            "P": "P",
            "T": "T",
        }
    )

    return merged


# =============================================================================
# 5. STATISTICS
# =============================================================================

def calculate_statistics(df, xcol):

    work = df[[xcol, "lnD"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    x = work[xcol].to_numpy(dtype=float)
    y = work["lnD"].to_numpy(dtype=float)

    N = len(x)

    if N < 3:
        raise RuntimeError(
            f"Not enough points for statistical analysis of {xcol}"
        )

    # ---------------------------------------------------------------------
    # Pearson
    # ---------------------------------------------------------------------

    rP, pP = pearsonr(x, y)

    # ---------------------------------------------------------------------
    # Spearman
    # ---------------------------------------------------------------------

    rS, pS = spearmanr(x, y)

    # ---------------------------------------------------------------------
    # Linear regression
    # ---------------------------------------------------------------------

    coef_lin = np.polyfit(x, y, 1)
    y_lin = np.polyval(coef_lin, x)

    R2_lin = r2_score(y, y_lin)

    # ---------------------------------------------------------------------
    # Quadratic regression
    # ---------------------------------------------------------------------

    coef_quad = np.polyfit(x, y, 2)
    y_quad = np.polyval(coef_quad, x)

    R2_quad = r2_score(y, y_quad)

    return {
        "N": N,

        "rP": rP,
        "pP": pP,

        "rS": rS,
        "pS": pS,

        "coef_lin": coef_lin,
        "coef_quad": coef_quad,

        "R2_lin": R2_lin,
        "R2_quad": R2_quad,
    }


# =============================================================================
# 6. LOAD DATA
# =============================================================================

print()
print("=" * 92)
print("INPUT")
print("=" * 92)

s2 = read_s2(S2_FILE)
tau = read_tau(TAU_FILE)
diff = read_diffusion(D_FILE)

print(f"Diffusion states available = {len(diff)}")
print(f"s2 states available        = {len(s2)}")
print(f"tau states available       = {len(tau)}")


# =============================================================================
# 7. MERGE STRUCTURE AND DYNAMICS
# =============================================================================

s2D = merge_descriptor_with_diffusion(
    s2,
    diff,
    "s2"
)

tauD = merge_descriptor_with_diffusion(
    tau,
    diff,
    "tau"
)

# Use -s2 as structural descriptor
s2D["minus_s2"] = -s2D["s2"]


print()
print("=" * 92)
print("MERGED STRUCTURE-DYNAMICS DATA")
print("=" * 92)

print(f"s2-D matched states  = {len(s2D)}")
print(f"tau-D matched states = {len(tauD)}")


# =============================================================================
# 8. TEMPERATURE RESTRICTION FOR MAIN FIGURE
# =============================================================================

s2_plot = s2D[
    (s2D["T"] >= T_MIN) &
    (s2D["T"] <= T_MAX)
].copy()

tau_plot = tauD[
    (tauD["T"] >= T_MIN) &
    (tauD["T"] <= T_MAX)
].copy()


print()
print("=" * 92)
print("MAIN-TEXT TEMPERATURE WINDOW")
print("=" * 92)

print(
    f"Selected temperature interval: "
    f"{T_MIN:.2f} <= T* <= {T_MAX:.2f}"
)

print(f"s2-D states retained  = {len(s2_plot)}")
print(f"tau-D states retained = {len(tau_plot)}")

print(
    f"s2-D actual T range   = "
    f"{s2_plot['T'].min():.3f} -- {s2_plot['T'].max():.3f}"
)

print(
    f"tau-D actual T range  = "
    f"{tau_plot['T'].min():.3f} -- {tau_plot['T'].max():.3f}"
)


# =============================================================================
# 9. STATISTICS FOR SELECTED REGIME
# =============================================================================

stats_s2 = calculate_statistics(
    s2_plot,
    "minus_s2"
)

stats_tau = calculate_statistics(
    tau_plot,
    "tau"
)


def print_statistics(name, stats):

    print()
    print(f"Descriptor: {name}")
    print("-" * 50)

    print(f"N = {stats['N']}")

    print(
        f"Pearson  = {stats['rP']: .6f}"
        f"   (p = {stats['pP']:.3e})"
    )

    print(
        f"Spearman = {stats['rS']: .6f}"
        f"   (p = {stats['pS']:.3e})"
    )

    print(
        f"R2 linear    = {stats['R2_lin']: .6f}"
    )

    print(
        f"R2 quadratic = {stats['R2_quad']: .6f}"
    )


print()
print("=" * 92)
print("STRUCTURE-DYNAMICS STATISTICS")
print("=" * 92)

print_statistics("-s2", stats_s2)
print_statistics("tau", stats_tau)


# =============================================================================
# 10. OPTIONAL FULL-DATA ROBUSTNESS STATISTICS
# =============================================================================
#
# These are NOT plotted.
# They are retained as an internal robustness check.
#

stats_s2_all = calculate_statistics(
    s2D,
    "minus_s2"
)

stats_tau_all = calculate_statistics(
    tauD,
    "tau"
)


print()
print("=" * 92)
print("FULL-DATA ROBUSTNESS CHECK")
print("=" * 92)

print_statistics("-s2, all temperatures", stats_s2_all)
print_statistics("tau, all temperatures", stats_tau_all)


# =============================================================================
# 11. FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=FIGSIZE,
    sharey=True,
)

ax1, ax2 = axes


# -------------------------------------------------------------------------
# Common temperature normalization
# -------------------------------------------------------------------------

norm = Normalize(
    vmin=COLOR_TMIN,
    vmax=COLOR_TMAX
)

cmap = plt.get_cmap(CMAP)


# =============================================================================
# 12. PANEL (a): -s2 versus lnD
# =============================================================================

ax1.scatter(
    s2_plot["minus_s2"],
    s2_plot["lnD"],
    c=s2_plot["T"],
    cmap=cmap,
    norm=norm,

    s=POINT_SIZE,
    alpha=POINT_ALPHA,

    edgecolors="black",
    linewidths=POINT_EDGEWIDTH,

    rasterized=True,
    zorder=3,
)

ax1.set_xlabel(r"$-s_2^*$", fontsize=22)

ax1.set_ylabel(r"$\ln D^*$", fontsize=22)

#ax1.text(0.045,0.945, r"\textbf{(a)}", transform=ax1.transAxes, ha="left", va="top", fontsize=PANEL_FS,)


# -------------------------------------------------------------------------
# Statistics box
# -------------------------------------------------------------------------

text_s2 = (
    rf"$N={stats_s2['N']}$" "\n"
    rf"$r_{{\rm P}}={stats_s2['rP']:.3f}$" "\n"
    rf"$r_{{\rm S}}={stats_s2['rS']:.3f}$" "\n"
    rf"$R^2_{{\rm quad}}={stats_s2['R2_quad']:.3f}$"
)

ax1.text(0.905, 0.060, text_s2, transform=ax1.transAxes,
ha="right", va="bottom", fontsize=STATS_FS,
    bbox=dict(
        boxstyle="round,pad=0.42",
        facecolor="white",
        edgecolor="0.60",
        alpha=0.94,), zorder=10,)


# =============================================================================
# 13. PANEL (b): tau versus lnD
# =============================================================================

ax2.scatter(
    tau_plot["tau"],
    tau_plot["lnD"],
    c=tau_plot["T"],
    cmap=cmap,
    norm=norm,

    s=POINT_SIZE,
    alpha=POINT_ALPHA,

    edgecolors="black",
    linewidths=POINT_EDGEWIDTH,

    rasterized=True,
    zorder=3,
)

ax2.set_xlabel(r"$\tau^*$", fontsize=22)

#ax2.text(0.045, 0.945, r"\textbf{(b)}",transform=ax2.transAxes, ha="left", va="top", fontsize=PANEL_FS,)

# -------------------------------------------------------------------------
# Statistics box
# -------------------------------------------------------------------------

text_tau = (rf"$N={stats_tau['N']}$" "\n"
    rf"$r_{{\rm P}}={stats_tau['rP']:.3f}$" "\n"
    rf"$r_{{\rm S}}={stats_tau['rS']:.3f}$" "\n"
    rf"$R^2_{{\rm quad}}={stats_tau['R2_quad']:.3f}$")

ax2.text(0.350, 0.055, text_tau, transform=ax2.transAxes, ha="right", va="bottom", fontsize=STATS_FS, bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.65", alpha=0.92,), zorder=10,)


# =============================================================================
# 14. AXIS FORMATTING
# =============================================================================

for ax in axes:

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
        which="major",
        linestyle=":",
        linewidth=0.55,
        alpha=0.42,
        zorder=0,
    )

    ax.margins(
        x=0.02,
        y=0.04,
    )




# =============================================================================
# 15. COLORBAR — explicit external axis
# =============================================================================

# Reserve space on the right for the colorbar
fig.subplots_adjust(
    left=0.085,
    right=0.865,
    bottom=0.145,
    top=0.975,
    wspace=0.075,
)

# Explicit colorbar axis:
# [left, bottom, width, height]
cax = fig.add_axes([
    0.91,   # move further right
    0.155,   # bottom
    0.024,   # width
    0.825     # height
])

sm = ScalarMappable(
    norm=norm,
    cmap=cmap
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    cax=cax,
    orientation="vertical",
)

cbar.set_label(
    r"$T^*$",
    fontsize=22,
    labelpad=10,
)

cbar.ax.tick_params(
    labelsize=12,
    direction="in",
    length=4.5,
    width=0.9,
)

cbar.set_ticks(
    np.arange(
        COLOR_TMIN,
        COLOR_TMAX + 0.001,
        0.05,
    )
)

cbar.outline.set_linewidth(1.0)


# =============================================================================
# 16. LAYOUT
# =============================================================================

fig.subplots_adjust(
    left=0.085,
    right=0.905,
    bottom=0.145,
    top=0.975,
    wspace=0.075,
)


# =============================================================================
# 17. SAVE
# =============================================================================

fig.savefig(
    OUT_PDF,
    dpi=600,
)

fig.savefig(
    OUT_PNG,
    dpi=600,
)

plt.show()


# =============================================================================
# 18. FINAL REPORT
# =============================================================================

print()
print("=" * 92)
print("FINAL GLOBAL PAIR-ORDER vs DYNAMICS FIGURE")
print("=" * 92)

print(
    f"Main-text regime: T* >= {T_MIN:.2f}"
)

print()

print("Panel (a): ln(D*) versus -s2*")
print(
    f"  N        = {stats_s2['N']}"
)
print(
    f"  Pearson  = {stats_s2['rP']:.6f}"
)
print(
    f"  Spearman = {stats_s2['rS']:.6f}"
)
print(
    f"  R2 lin   = {stats_s2['R2_lin']:.6f}"
)
print(
    f"  R2 quad  = {stats_s2['R2_quad']:.6f}"
)

print()

print("Panel (b): ln(D*) versus tau")
print(
    f"  N        = {stats_tau['N']}"
)
print(
    f"  Pearson  = {stats_tau['rP']:.6f}"
)
print(
    f"  Spearman = {stats_tau['rS']:.6f}"
)
print(
    f"  R2 lin   = {stats_tau['R2_lin']:.6f}"
)
print(
    f"  R2 quad  = {stats_tau['R2_quad']:.6f}"
)

print()

print("Figure:")
print(f"  {OUT_PDF}")
print(f"  {OUT_PNG}")

print()

print("✓ Only T* >= 0.20 shown in the main figure.")
print("✓ Low-temperature sluggish states excluded from visualization.")
print("✓ Full-temperature statistics retained as robustness check.")
print("✓ Linear and quadratic regressions calculated but not drawn.")
print("✓ Pearson and Spearman correlations retained.")
print("✓ Quadratic R2 shown in the figure.")
print("✓ Common T* normalization used in both panels.")
print("✓ s2 taken from the fourth column.")
print("✓ tau taken from the fourth column.")
print("✓ tau duplicate states averaged before matching.")
print("✓ Only common (P,T) structure-dynamics states analyzed.")

print("=" * 92)
