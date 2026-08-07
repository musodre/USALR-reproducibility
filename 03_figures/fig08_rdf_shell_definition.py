#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shell-resolved structural descriptors
======================================

First figure of subsection:
    Shell-resolved structural reorganization

Representative state:
    P* = 1.00
    T* = 0.20

Panel (a)
---------
Radial distribution function g(r*) showing:

    g1 : peak height in W1 = [1.10, 1.35]
    g2 : peak height in W2 = [1.85, 2.15]

and

    Rg = g2/g1

Panel (b)
---------
Coordination-population density:

    4*pi*rho* r*^2 g(r*)

with shaded contributions

    n1 = 4*pi*rho* int_W1 r*^2 g(r*) dr*
    n2 = 4*pi*rho* int_W2 r*^2 g(r*) dr*

and

    Rn = n2/n1

IMPORTANT
---------
W1 and W2 are operational characteristic radial windows and should not
automatically be interpreted as complete coordination shells bounded by
successive minima of g(r).

Input files
-----------
../allpress/P_X.XXX/rdfs/P_X.XX_T_X.XX.rdf

../allpress/P_X.XXX/thermo/
    outvars_P_X.XX_T_X.XX.profile

Outputs
-------
shell_resolved_structure/
    rdf_shell_definition.pdf
    rdf_shell_definition.png
    shell_descriptor_values.dat
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import trapezoid



from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, FIGURE_OUTPUT_ROOT, DERIVED_DATA_ROOT

# =============================================================================
# 1. REPRESENTATIVE STATE
# =============================================================================

P_TARGET = 1.00
T_TARGET = 0.20


# =============================================================================
# 2. OPERATIONAL RADIAL WINDOWS
# =============================================================================

R1_MIN = 1.10
R1_MAX = 1.35

R2_MIN = 1.85
R2_MAX = 2.15


# =============================================================================
# 3. PLOTTED RADIAL DOMAIN
# =============================================================================

R_PLOT_MIN = 0.80
R_PLOT_MAX = 2.75


# =============================================================================
# 4. INPUT ROOT
# =============================================================================

ROOT = RAW_DATA_ROOT / "allpress"


# =============================================================================
# 5. OUTPUT
# =============================================================================

OUTDIR = Path(
    "shell_resolved_structure"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


OUT_PDF = (
    OUTDIR
    /
    "rdf_shell_definition.pdf"
)

OUT_PNG = (
    OUTDIR
    /
    "rdf_shell_definition.png"
)

OUT_DATA = (
    OUTDIR
    /
    "shell_descriptor_values.dat"
)


# =============================================================================
# 6. GRAPHICAL PARAMETERS
# =============================================================================

FIGSIZE = (
    8.2,
    7.1
)


RDF_LW = 2.15

POP_LW = 1.95


# -------------------------------------------------------------------------
# Window shading
# -------------------------------------------------------------------------

WINDOW_ALPHA = 0.22

POPULATION_ALPHA = 0.38


# -------------------------------------------------------------------------
# Font sizes
# -------------------------------------------------------------------------

LABEL_FS = 18

TICK_FS = 12

ANNOT_FS = 14

WINDOW_FS = 15

PANEL_FS = 16

BOX_FS = 13.5


# -------------------------------------------------------------------------
# Peak symbols
# -------------------------------------------------------------------------

PEAK_MARKER_SIZE = 78

PEAK_EDGE_WIDTH = 1.25


# =============================================================================
# 7. MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

    "text.usetex": True,

    "font.family": "serif",

    "font.size": 12,

    "axes.labelsize": LABEL_FS,

    "xtick.labelsize": TICK_FS,

    "ytick.labelsize": TICK_FS,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "xtick.top": True,

    "ytick.right": True,

    "xtick.major.size": 5,

    "ytick.major.size": 5,

    "xtick.major.width": 1.0,

    "ytick.major.width": 1.0,

    "axes.linewidth": 1.15,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "savefig.bbox": "tight",
})


# =============================================================================
# 8. INPUT FILE NAMES
# =============================================================================

P_DIR = (
    ROOT
    /
    f"P_{P_TARGET:.3f}"
)


RDF_FILE = (
    P_DIR
    /
    "rdfs"
    /
    f"P_{P_TARGET:.2f}_T_{T_TARGET:.2f}.rdf"
)


THERMO_FILE = (
    P_DIR
    /
    "thermo"
    /
    f"outvars_P_{P_TARGET:.2f}_T_{T_TARGET:.2f}.profile"
)


# =============================================================================
# 9. INPUT REPORT
# =============================================================================

print()

print(
    "=" * 88
)

print(
    "REPRESENTATIVE SHELL-RESOLVED STRUCTURE"
)

print(
    "=" * 88
)

print(
    f"P* = {P_TARGET:.3f}"
)

print(
    f"T* = {T_TARGET:.3f}"
)

print()

print(
    "RDF file:"
)

print(
    f"  {RDF_FILE}"
)

print()

print(
    "Thermo file:"
)

print(
    f"  {THERMO_FILE}"
)


# =============================================================================
# 10. CHECK FILES
# =============================================================================

if not RDF_FILE.exists():

    raise FileNotFoundError(

        "\nRDF file not found:\n"
        f"{RDF_FILE.resolve()}"
    )


if not THERMO_FILE.exists():

    raise FileNotFoundError(

        "\nThermodynamic file not found:\n"
        f"{THERMO_FILE.resolve()}"
    )


# =============================================================================
# 11. READ THERMODYNAMIC DENSITY
# =============================================================================

def read_density(filename):

    """
    Expected columns:

        timestep
        T
        P
        V
        rho
        U
        K
        H

    rho = column index 4.

    All valid entries are averaged.
    """

    rho_values = []


    with open(
        filename,
        "r"
    ) as f:

        for line in f:

            line = line.strip()


            if not line:

                continue


            if line.startswith("#"):

                continue


            parts = line.split()


            if len(parts) < 5:

                continue


            try:

                rho_value = float(
                    parts[4]
                )

            except ValueError:

                continue


            if np.isfinite(
                rho_value
            ):

                rho_values.append(
                    rho_value
                )


    if len(
        rho_values
    ) == 0:

        raise RuntimeError(

            "No valid density values found in:\n"
            f"{filename}"
        )


    return float(

        np.mean(
            rho_values
        )
    )


rho = read_density(
    THERMO_FILE
)


print()

print(
    f"Mean density rho* = {rho:.8f}"
)


# =============================================================================
# 12. READ FINAL LAMMPS RDF BLOCK
# =============================================================================

def read_lammps_rdf_last_block(filename):

    """
    Read the final complete RDF block.

    Assumed block format:

        timestep number_of_bins
        bin r g(r) coord
        ...

    Columns used:

        column 1 -> r
        column 2 -> g(r)
    """

    with open(
        filename,
        "r"
    ) as f:

        lines = f.readlines()


    blocks = []

    i = 0


    while i < len(
        lines
    ):

        s = lines[
            i
        ].strip()


        if (
            not s
            or
            s.startswith("#")
        ):

            i += 1

            continue


        parts = s.split()


        if len(
            parts
        ) != 2:

            i += 1

            continue


        try:

            timestep = int(
                float(
                    parts[0]
                )
            )

            nrows = int(
                float(
                    parts[1]
                )
            )

        except ValueError:

            i += 1

            continue


        data = []


        for j in range(

            i + 1,

            min(
                i + 1 + nrows,
                len(lines)
            )
        ):

            row = lines[
                j
            ].strip()


            if (
                not row
                or
                row.startswith("#")
            ):

                continue


            values = row.split()


            if len(
                values
            ) < 3:

                continue


            try:

                numeric = [

                    float(x)

                    for x in values
                ]

            except ValueError:

                continue


            data.append(
                numeric
            )


        if len(
            data
        ) == nrows:

            blocks.append(

                (

                    timestep,

                    np.asarray(
                        data,
                        dtype=float
                    )
                )
            )


        i += (
            nrows
            +
            1
        )


    if len(
        blocks
    ) == 0:

        raise RuntimeError(

            "No complete RDF block found in:\n"
            f"{filename}"
        )


    timestep, array = blocks[-1]


    r = array[
        :,
        1
    ]


    g = array[
        :,
        2
    ]


    good = (

        np.isfinite(r)

        &

        np.isfinite(g)
    )


    r = r[
        good
    ]


    g = g[
        good
    ]


    order = np.argsort(
        r
    )


    r = r[
        order
    ]


    g = g[
        order
    ]


    return (
        timestep,
        r,
        g
    )


rdf_timestep, r, g = read_lammps_rdf_last_block(
    RDF_FILE
)


print(
    f"RDF block timestep = {rdf_timestep}"
)

print(
    f"RDF points         = {len(r)}"
)

print(
    f"R range            = "
    f"{r.min():.4f} -- {r.max():.4f}"
)


# =============================================================================
# 13. DEFINE CHARACTERISTIC RADIAL REGIONS
# =============================================================================

mask1 = (

    (r >= R1_MIN)

    &

    (r <= R1_MAX)
)


mask2 = (

    (r >= R2_MIN)

    &

    (r <= R2_MAX)
)


if np.count_nonzero(
    mask1
) < 3:

    raise RuntimeError(

        "Too few RDF points inside W1."
    )


if np.count_nonzero(
    mask2
) < 3:

    raise RuntimeError(

        "Too few RDF points inside W2."
    )


r1 = r[
    mask1
]

g1_region = g[
    mask1
]


r2 = r[
    mask2
]

g2_region = g[
    mask2
]


# =============================================================================
# 14. PEAK HEIGHTS
# =============================================================================

i1 = np.argmax(
    g1_region
)


i2 = np.argmax(
    g2_region
)


r_peak1 = float(
    r1[i1]
)


g1 = float(
    g1_region[i1]
)


r_peak2 = float(
    r2[i2]
)


g2 = float(
    g2_region[i2]
)


Rg = (
    g2/g1
)


# =============================================================================
# 15. COORDINATION-POPULATION DENSITY
# =============================================================================

coord_integrand = (

    4.0

    *

    np.pi

    *

    rho

    *

    r**2

    *

    g
)


coord_integrand1 = (

    4.0

    *

    np.pi

    *

    rho

    *

    r1**2

    *

    g1_region
)


coord_integrand2 = (

    4.0

    *

    np.pi

    *

    rho

    *

    r2**2

    *

    g2_region
)


# =============================================================================
# 16. POPULATIONS
# =============================================================================

n1 = float(

    trapezoid(

        coord_integrand1,

        r1
    )
)


n2 = float(

    trapezoid(

        coord_integrand2,

        r2
    )
)


Rn = (
    n2/n1
)


# =============================================================================
# 17. NUMERICAL REPORT
# =============================================================================

print()

print(
    "=" * 88
)

print(
    "SHELL-RESOLVED DESCRIPTORS"
)

print(
    "=" * 88
)


print(
    f"First radial window:"
    f" {R1_MIN:.2f} <= r* <= {R1_MAX:.2f}"
)


print(
    f"Second radial window:"
    f" {R2_MIN:.2f} <= r* <= {R2_MAX:.2f}"
)


print()


print(
    f"r1_peak = {r_peak1:.6f}"
)


print(
    f"g1      = {g1:.6f}"
)


print()


print(
    f"r2_peak = {r_peak2:.6f}"
)


print(
    f"g2      = {g2:.6f}"
)


print()


print(
    f"g2/g1   = {Rg:.6f}"
)


print()


print(
    f"n1      = {n1:.6f}"
)


print(
    f"n2      = {n2:.6f}"
)


print(
    f"n2/n1   = {Rn:.6f}"
)


# =============================================================================
# 18. SAVE NUMERICAL VALUES
# =============================================================================

with open(
    OUT_DATA,
    "w"
) as f:

    f.write(

        "# P T rho "
        "r_peak1 g1 "
        "r_peak2 g2 "
        "g2_over_g1 "
        "n1 n2 "
        "n2_over_n1\n"
    )


    f.write(

        f"{P_TARGET:.8f} "

        f"{T_TARGET:.8f} "

        f"{rho:.10e} "

        f"{r_peak1:.10e} "

        f"{g1:.10e} "

        f"{r_peak2:.10e} "

        f"{g2:.10e} "

        f"{Rg:.10e} "

        f"{n1:.10e} "

        f"{n2:.10e} "

        f"{Rn:.10e}\n"
    )


# =============================================================================
# 19. CREATE FIGURE
# =============================================================================

fig = plt.figure(
    figsize=FIGSIZE
)


gs = fig.add_gridspec(

    2,
    1,

    height_ratios=[
        1.58,
        1.00
    ],

    hspace=0.055
)


ax1 = fig.add_subplot(
    gs[0]
)


ax2 = fig.add_subplot(
    gs[1],
    sharex=ax1
)


# =============================================================================
# 20. PANEL (a): RDF
# =============================================================================

ax1.plot(

    r,

    g,

    color="black",

    linewidth=RDF_LW,

    zorder=5
)


# -------------------------------------------------------------------------
# Operational radial windows
# -------------------------------------------------------------------------

ax1.axvspan(

    R1_MIN,
    R1_MAX,

    alpha=WINDOW_ALPHA,

    zorder=0
)


ax1.axvspan(

    R2_MIN,
    R2_MAX,

    alpha=WINDOW_ALPHA,

    zorder=0
)


# -------------------------------------------------------------------------
# Peak markers
# -------------------------------------------------------------------------

ax1.scatter(

    [r_peak1],

    [g1],

    s=PEAK_MARKER_SIZE,

    marker="o",

    facecolor="white",

    edgecolor="black",

    linewidth=PEAK_EDGE_WIDTH,

    zorder=10
)


ax1.scatter(

    [r_peak2],

    [g2],

    s=PEAK_MARKER_SIZE,

    marker="o",

    facecolor="white",

    edgecolor="black",

    linewidth=PEAK_EDGE_WIDTH,

    zorder=10
)


# =============================================================================
# 21. PEAK ANNOTATIONS
# =============================================================================

ax1.annotate(

    rf"$g_1={g1:.2f}$",

    xy=(
        r_peak1,
        g1
    ),

    xytext=(
        18,
        22
    ),

    textcoords="offset points",

    ha="left",

    va="bottom",

    fontsize=ANNOT_FS,

    arrowprops=dict(

        arrowstyle="->",

        linewidth=1.05,

        color="black",

        shrinkA=2,

        shrinkB=3
    ),

    zorder=12
)


ax1.annotate(

    rf"$g_2={g2:.2f}$",

    xy=(
        r_peak2,
        g2
    ),

    xytext=(
        18,
        20
    ),

    textcoords="offset points",

    ha="left",

    va="bottom",

    fontsize=ANNOT_FS,

    arrowprops=dict(

        arrowstyle="->",

        linewidth=1.05,

        color="black",

        shrinkA=2,

        shrinkB=3
    ),

    zorder=12
)


# =============================================================================
# 22. WINDOW LABELS
# =============================================================================

ax1.text(

    0.5
    *
    (
        R1_MIN
        +
        R1_MAX
    ),

    0.13,

    r"$\mathcal{W}_1$",

    transform=ax1.get_xaxis_transform(),

    ha="center",

    va="bottom",

    fontsize=WINDOW_FS,

    zorder=8
)


ax1.text(

    0.5
    *
    (
        R2_MIN
        +
        R2_MAX
    ),

    0.13,

    r"$\mathcal{W}_2$",

    transform=ax1.get_xaxis_transform(),

    ha="center",

    va="bottom",

    fontsize=WINDOW_FS,

    zorder=8
)


# =============================================================================
# 23. THERMODYNAMIC-STATE BOX
# =============================================================================

state_text = (

    rf"$P^*={P_TARGET:.2f}$"

    "\n"

    rf"$T^*={T_TARGET:.2f}$"
)


ax1.text(

    0.965,

    0.945,

    state_text,

    transform=ax1.transAxes,

    ha="right",

    va="top",

    fontsize=BOX_FS,

    bbox=dict(

        boxstyle="round,pad=0.34",

        facecolor="white",

        edgecolor="0.62",

        linewidth=0.9,

        alpha=0.95
    ),

    zorder=15
)


ax1.set_ylabel(
    r"$g(r^*)$"
)


# =============================================================================
# 24. PANEL (b): COORDINATION-POPULATION DENSITY
# =============================================================================

ax2.plot(

    r,

    coord_integrand,

    color="black",

    linewidth=POP_LW,

    zorder=5
)


# -------------------------------------------------------------------------
# Population associated with W1
# -------------------------------------------------------------------------

ax2.fill_between(

    r1,

    0.0,

    coord_integrand1,

    alpha=POPULATION_ALPHA,

    zorder=3
)


# -------------------------------------------------------------------------
# Population associated with W2
# -------------------------------------------------------------------------

ax2.fill_between(

    r2,

    0.0,

    coord_integrand2,

    alpha=POPULATION_ALPHA,

    zorder=3
)


# =============================================================================
# 25. RADIAL-WINDOW LIMITS
# =============================================================================

for x in [

    R1_MIN,
    R1_MAX,
    R2_MIN,
    R2_MAX

]:

    ax2.axvline(

        x,

        color="0.45",

        linestyle="--",

        linewidth=0.85,

        alpha=0.60,

        zorder=1
    )


# =============================================================================
# 26. POPULATION LABELS
# =============================================================================

x1_center = (
    0.5
    *
    (
        R1_MIN
        +
        R1_MAX
    )
)


x2_center = (
    0.5
    *
    (
        R2_MIN
        +
        R2_MAX
    )
)


y1_text = (
    0.50
    *
    np.max(
        coord_integrand1
    )
)


y2_text = (
    0.49
    *
    np.max(
        coord_integrand2
    )
)


ax2.text(

    x1_center,

    y1_text,

    rf"$n_1={n1:.2f}$",

    ha="center",

    va="center",

    fontsize=ANNOT_FS,

    zorder=10
)


ax2.text(

    x2_center,

    y2_text,

    rf"$n_2={n2:.2f}$",

    ha="center",

    va="center",

    fontsize=ANNOT_FS,

    zorder=10
)


# =============================================================================
# 27. RATIO BOX
# =============================================================================

ratio_text = (

    rf"$R_g=g_2/g_1={Rg:.3f}$"

    "\n"

    rf"$R_n=n_2/n_1={Rn:.3f}$"
)


ax2.text(

    0.965,

    0.92,

    ratio_text,

    transform=ax2.transAxes,

    ha="right",

    va="top",

    fontsize=BOX_FS,

    bbox=dict(

        boxstyle="round,pad=0.37",

        facecolor="white",

        edgecolor="0.62",

        linewidth=0.9,

        alpha=0.95
    ),

    zorder=15
)


# =============================================================================
# 28. AXIS LABELS
# =============================================================================

ax2.set_xlabel(
    r"$r^*$"
)


ax2.set_ylabel(

    r"$4\pi\rho^*r^{*2}g(r^*)$"
)


# =============================================================================
# 29. PANEL LABELS
# =============================================================================

ax1.text(

    0.025,

    0.955,

    r"\textbf{(a)}",

    transform=ax1.transAxes,

    ha="left",

    va="top",

    fontsize=PANEL_FS,

    zorder=20
)


ax2.text(

    0.025,

    0.945,

    r"\textbf{(b)}",

    transform=ax2.transAxes,

    ha="left",

    va="top",

    fontsize=PANEL_FS,

    zorder=20
)


# =============================================================================
# 30. COMMON AXIS FORMATTING
# =============================================================================

ax1.set_xlim(

    R_PLOT_MIN,

    R_PLOT_MAX
)


for ax in [
    ax1,
    ax2
]:

    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

        length=5,

        width=1.0
    )


    ax.grid(

        True,

        which="major",

        linestyle=":",

        linewidth=0.55,

        alpha=0.25,

        zorder=0
    )


# No x labels on upper panel
plt.setp(

    ax1.get_xticklabels(),

    visible=False
)


# =============================================================================
# 31. SMALL Y MARGINS
# =============================================================================

ymin1, ymax1 = ax1.get_ylim()

dy1 = (
    ymax1-ymin1
)

ax1.set_ylim(

    ymin1,

    ymax1
    +
    0.035*dy1
)


ymin2, ymax2 = ax2.get_ylim()

dy2 = (
    ymax2-ymin2
)

ax2.set_ylim(

    0.0,

    ymax2
    +
    0.045*dy2
)


# =============================================================================
# 32. LAYOUT
# =============================================================================

fig.subplots_adjust(

    left=0.115,

    right=0.975,

    bottom=0.105,

    top=0.965,

    hspace=0.055
)


# =============================================================================
# 33. SAVE
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
# 34. FINAL REPORT
# =============================================================================

print()

print(
    "=" * 88
)

print(
    "OUTPUT"
)

print(
    "=" * 88
)


print(
    f"Figure PDF:\n"
    f"  {OUT_PDF.resolve()}"
)

print()


print(
    f"Figure PNG:\n"
    f"  {OUT_PNG.resolve()}"
)

print()


print(
    f"Descriptor values:\n"
    f"  {OUT_DATA.resolve()}"
)

print()


print(
    "✓ Redundant g(r*) legend removed."
)

print(
    "✓ Panel label (a) no longer competes with legend."
)

print(
    "✓ W1 and W2 labels enlarged and repositioned."
)

print(
    "✓ g1 and g2 annotations enlarged."
)

print(
    "✓ n1 and n2 annotations enlarged."
)

print(
    "✓ g1 and g2 extracted independently within their radial windows."
)

print(
    "✓ n1 and n2 obtained from the proper coordination integral."
)

print(
    "✓ g2/g1 and n2/n1 reported separately."
)

print(
    "✓ W1 and W2 treated explicitly as operational radial regions."
)

print()

print(
    "=" * 88
)
