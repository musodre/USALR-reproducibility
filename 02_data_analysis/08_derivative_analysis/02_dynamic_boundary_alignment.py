#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
derivative_boundary_alignment.py

Compara, isoterma por isoterma, a fronteira da anomalia difusional

    (d ln D*/dP*)_T = 0

com a evolução das respostas estruturais

    dRg/dP*
    dRn/dP*
    dA_SALR/dP*

Input
-----
plots/derivative_structure_dynamics/derivative_master.dat

Outputs
-------
plots/derivative_boundary_alignment/

    dynamic_boundaries.dat
    structural_response_at_boundaries.dat

    derivative_alignment_selected_isotherms.pdf
    derivative_alignment_selected_isotherms.png

    structural_derivatives_raw.pdf
    structural_derivatives_raw.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

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



# =============================================================================
# 1. INPUT / OUTPUT
# =============================================================================

INPUT = (
    DERIVED_DATA_ROOT
    / "derivative_structure_dynamics"
    / "derivative_master.dat"
)

OUTDIR = (
    DERIVED_DATA_ROOT
    / "derivative_boundary_alignment"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

T_SELECTED = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]

# Intervalo em P usado para comparar a resposta estrutural
# imediatamente antes/depois da fronteira dinâmica.
LOCAL_DP = 0.30

# mínimo de pontos para interpolação
MIN_POINTS = 5

# grid apenas para localização gráfica/interpolação
NFINE = 1500


# =============================================================================
# 3. STYLE
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
# 4. LOAD
# =============================================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"\nArquivo não encontrado:\n"
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

    "dlnD_dP",
    "dlnD_dP_sign",
    "dlnD_dP_class",

    "dRg_dP",
    "dRn_dP",
    "dA_SALR_dP",

]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise RuntimeError(
        "Colunas ausentes:\n"
        +
        "\n".join(missing)
    )


numeric_cols = [

    "T",
    "P",

    "dlnD_dP",
    "dlnD_dP_sign",

    "dRg_dP",
    "dRn_dP",
    "dA_SALR_dP",

]


for c in numeric_cols:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


df = df.sort_values(
    ["T", "P"]
).reset_index(
    drop=True
)


print()
print("=" * 90)
print("INPUT")
print("=" * 90)

print(
    f"N states    = {len(df)}"
)

print(
    f"N isotherms = {df['T'].nunique()}"
)

print(
    f"T range     = "
    f"{df['T'].min():.3f} -- "
    f"{df['T'].max():.3f}"
)

print(
    f"P range     = "
    f"{df['P'].min():.3f} -- "
    f"{df['P'].max():.3f}"
)


# =============================================================================
# 5. ROBUST NORMALIZATION
# =============================================================================

def robust_normalize(y):

    """
    Normalização exclusiva para visualização.

    Preserva zero e sinal.

    A escala é o percentil 90 de |y|, mais robusto contra outliers
    que max(|y|).
    """

    y = np.asarray(
        y,
        dtype=float
    )

    good = np.isfinite(y)

    result = np.full(
        len(y),
        np.nan
    )

    if np.sum(good) == 0:

        return result


    scale = np.nanpercentile(
        np.abs(y[good]),
        90
    )


    if (
        not np.isfinite(scale)
        or
        scale <= 0
    ):

        scale = np.nanmax(
            np.abs(y[good])
        )


    if (
        not np.isfinite(scale)
        or
        scale <= 0
    ):

        return result


    result[good] = (
        y[good]
        /
        scale
    )


    return result


# =============================================================================
# 6. ZERO CROSSINGS OF dlnD/dP
# =============================================================================

def find_zero_crossings(
    P,
    Y
):

    """
    Localiza cruzamentos de zero usando os valores discretos da derivada.

    O zero é interpolado linearmente entre dois estados adjacentes
    com sinais opostos.

    Não cria zeros fora do domínio amostrado.
    """

    P = np.asarray(
        P,
        dtype=float
    )

    Y = np.asarray(
        Y,
        dtype=float
    )


    order = np.argsort(
        P
    )

    P = P[order]
    Y = Y[order]


    zeros = []


    for i in range(
        len(P) - 1
    ):

        p1 = P[i]
        p2 = P[i + 1]

        y1 = Y[i]
        y2 = Y[i + 1]


        if not (
            np.isfinite(p1)
            and
            np.isfinite(p2)
            and
            np.isfinite(y1)
            and
            np.isfinite(y2)
        ):

            continue


        if y1 == 0:

            zeros.append(
                p1
            )

            continue


        if y1 * y2 < 0:

            pzero = (

                p1

                -

                y1
                *
                (p2 - p1)
                /
                (y2 - y1)

            )


            if (
                p1 <= pzero <= p2
            ):

                zeros.append(
                    float(pzero)
                )


    if len(zeros) == 0:

        return []


    # remove zeros duplicados numericamente

    zeros = np.array(
        sorted(zeros)
    )


    keep = [zeros[0]]


    for z in zeros[1:]:

        if abs(
            z - keep[-1]
        ) > 1e-6:

            keep.append(
                z
            )


    return keep


# =============================================================================
# 7. INTERPOLATED VALUE AT BOUNDARY
# =============================================================================

def interpolate_at(
    P,
    Y,
    P0
):

    good = (

        np.isfinite(P)
        &
        np.isfinite(Y)

    )


    P = np.asarray(
        P[good],
        dtype=float
    )

    Y = np.asarray(
        Y[good],
        dtype=float
    )


    if len(P) < MIN_POINTS:

        return np.nan


    order = np.argsort(
        P
    )

    P = P[order]
    Y = Y[order]


    Punique, idx = np.unique(
        P,
        return_index=True
    )

    Yunique = Y[idx]


    if len(Punique) < MIN_POINTS:

        return np.nan


    if not (
        Punique.min()
        <=
        P0
        <=
        Punique.max()
    ):

        return np.nan


    try:

        interpolator = PchipInterpolator(
            Punique,
            Yunique,
            extrapolate=False
        )

        return float(
            interpolator(P0)
        )

    except Exception:

        return np.nan


# =============================================================================
# 8. LOCAL BEFORE/AFTER RESPONSE
# =============================================================================

def local_side_mean(
    P,
    Y,
    P0,
    side,
    width=LOCAL_DP
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

            (P >= P0 - width)
            &
            (P < P0)

        )


    elif side == "right":

        mask = (

            (P > P0)
            &
            (P <= P0 + width)

        )


    else:

        raise ValueError(
            "side must be left or right"
        )


    mask &= np.isfinite(
        Y
    )


    if np.sum(mask) == 0:

        return np.nan


    return float(
        np.nanmean(
            Y[mask]
        )
    )


# =============================================================================
# 9. FIND ALL DYNAMIC BOUNDARIES
# =============================================================================

boundary_rows = []
response_rows = []


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


    dD = group[
        "dlnD_dP"
    ].to_numpy(
        dtype=float
    )


    zeros = find_zero_crossings(
        P,
        dD
    )


    for izero, Pzero in enumerate(
        zeros,
        start=1
    ):

        boundary_rows.append([

            T,
            izero,
            Pzero,

        ])


        row = {

            "T": float(T),

            "boundary_index": int(
                izero
            ),

            "P_boundary": float(
                Pzero
            ),

        }


        for name, column in [

            (
                "Rg",
                "dRg_dP"
            ),

            (
                "Rn",
                "dRn_dP"
            ),

            (
                "A_SALR",
                "dA_SALR_dP"
            ),

        ]:

            Y = group[
                column
            ].to_numpy(
                dtype=float
            )


            value = interpolate_at(
                P,
                Y,
                Pzero
            )


            left = local_side_mean(
                P,
                Y,
                Pzero,
                "left"
            )


            right = local_side_mean(
                P,
                Y,
                Pzero,
                "right"
            )


            delta = (
                right - left
                if
                np.isfinite(left)
                and
                np.isfinite(right)
                else
                np.nan
            )


            ratio = (

                right / left

                if
                np.isfinite(left)
                and
                np.isfinite(right)
                and
                abs(left) > 1e-14

                else
                np.nan

            )


            row[
                f"d{name}_dP_at_boundary"
            ] = value


            row[
                f"d{name}_dP_left"
            ] = left


            row[
                f"d{name}_dP_right"
            ] = right


            row[
                f"delta_d{name}_dP"
            ] = delta


            row[
                f"ratio_right_left_{name}"
            ] = ratio


        response_rows.append(
            row
        )


boundaries = pd.DataFrame(

    boundary_rows,

    columns=[

        "T",
        "boundary_index",
        "P_boundary",

    ]

)


responses = pd.DataFrame(
    response_rows
)


# =============================================================================
# 10. PRINT BOUNDARIES
# =============================================================================

print()
print("=" * 90)
print("DYNAMIC BOUNDARIES FROM dlnD/dP = 0")
print("=" * 90)


if len(boundaries) == 0:

    print(
        "No zero crossings found."
    )

else:

    print(
        boundaries.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# =============================================================================
# 11. PRINT LOCAL STRUCTURAL RESPONSE
# =============================================================================

print()
print("=" * 90)
print("STRUCTURAL RESPONSE AT DYNAMIC BOUNDARIES")
print("=" * 90)


if len(responses) > 0:

    cols_print = [

        "T",
        "P_boundary",

        "dRg_dP_left",
        "dRg_dP_at_boundary",
        "dRg_dP_right",

        "dRn_dP_left",
        "dRn_dP_at_boundary",
        "dRn_dP_right",

        "dA_SALR_dP_left",
        "dA_SALR_dP_at_boundary",
        "dA_SALR_dP_right",

    ]


    print(

        responses[
            cols_print
        ].to_string(

            index=False,

            float_format=lambda x:
            f"{x: .6e}"

        )

    )


# =============================================================================
# 12. SAVE TABLES
# =============================================================================

boundaries.to_csv(

    OUTDIR /
    "dynamic_boundaries.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


responses.to_csv(

    OUTDIR /
    "structural_response_at_boundaries.dat",

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


# =============================================================================
# 13. FIGURE 1 — NORMALIZED OVERLAY
# =============================================================================

fig, axes = plt.subplots(

    len(T_SELECTED),
    1,

    figsize=(
        8.2,
        2.35 * len(T_SELECTED)
    ),

    sharex=True

)


if len(T_SELECTED) == 1:

    axes = [
        axes
    ]


curve_specs = [

    (
        "dlnD_dP",
        r"$\partial_{P^*}\ln D^*$"
    ),

    (
        "dRg_dP",
        r"$\partial_{P^*}R_g$"
    ),

    (
        "dRn_dP",
        r"$\partial_{P^*}R_n$"
    ),

    (
        "dA_SALR_dP",
        r"$\partial_{P^*}A_{\rm SALR}$"
    ),

]


for ax, Tsel in zip(
    axes,
    T_SELECTED
):

    group = df[
        np.isclose(
            df["T"],
            Tsel
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


    for column, label in curve_specs:

        Y = group[
            column
        ].to_numpy(
            dtype=float
        )


        Yn = robust_normalize(
            Y
        )


        ax.plot(

            P,
            Yn,

            marker="o",

            markersize=2.7,

            linewidth=1.35,

            label=label

        )


    ax.axhline(

        0.0,

        linestyle="--",

        linewidth=0.9

    )


    subbounds = boundaries[
        np.isclose(
            boundaries["T"],
            Tsel
        )
    ]


    for Pzero in subbounds[
        "P_boundary"
    ]:

        ax.axvline(

            Pzero,

            linestyle=":",

            linewidth=1.5

        )


    ax.text(

        0.985,
        0.88,

        rf"$T^*={Tsel:.2f}$",

        transform=ax.transAxes,

        ha="right",
        va="top",

        fontsize=12

    )


    ax.set_ylabel(
        "normalized\nresponse"
    )


axes[-1].set_xlabel(
    r"$P^*$"
)


axes[0].legend(

    ncol=2,

    frameon=False,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        1.34
    )

)


fig.subplots_adjust(

    left=0.13,

    right=0.98,

    bottom=0.075,

    top=0.94,

    hspace=0.08

)


fig.savefig(

    OUTDIR /
    "derivative_alignment_selected_isotherms.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "derivative_alignment_selected_isotherms.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 14. FIGURE 2 — RAW STRUCTURAL DERIVATIVES
# =============================================================================

fig, axes = plt.subplots(

    3,
    1,

    figsize=(8.2, 8.3),

    sharex=True

)


raw_specs = [

    (
        "dRg_dP",
        r"$"
        r"\left("
        r"\partial R_g/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$"
    ),

    (
        "dRn_dP",
        r"$"
        r"\left("
        r"\partial R_n/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$"
    ),

    (
        "dA_SALR_dP",
        r"$"
        r"\left("
        r"\partial A_{\rm SALR}/"
        r"\partial P^*"
        r"\right)_{T^*}"
        r"$"
    ),

]


for ax, (
    column,
    ylabel
) in zip(
    axes,
    raw_specs
):

    for Tsel in T_SELECTED:

        group = df[
            np.isclose(
                df["T"],
                Tsel
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

            markersize=3,

            linewidth=1.25,

            label=(
                rf"$T^*={Tsel:.2f}$"
            )

        )


    ax.axhline(

        0.0,

        linestyle="--",

        linewidth=0.9

    )


    ax.set_ylabel(
        ylabel
    )


axes[-1].set_xlabel(
    r"$P^*$"
)


axes[0].legend(

    ncol=3,

    frameon=False,

    loc="best"

)


fig.tight_layout()


fig.savefig(

    OUTDIR /
    "structural_derivatives_raw.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR /
    "structural_derivatives_raw.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 15. SUMMARY OF BEFORE/AFTER CHANGES
# =============================================================================

print()
print("=" * 90)
print("GLOBAL BEFORE/AFTER SUMMARY")
print("=" * 90)


if len(responses) > 0:

    for name in [
        "Rg",
        "Rn",
        "A_SALR",
    ]:

        left = responses[
            f"d{name}_dP_left"
        ].to_numpy(
            dtype=float
        )


        right = responses[
            f"d{name}_dP_right"
        ].to_numpy(
            dtype=float
        )


        mask = (
            np.isfinite(left)
            &
            np.isfinite(right)
        )


        if np.sum(mask) == 0:

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
            f"fraction decrease  = "
            f"{np.mean(delta < 0):.3f}"
        )

        print(
            f"fraction increase  = "
            f"{np.mean(delta > 0):.3f}"
        )


# =============================================================================
# 16. OUTPUT
# =============================================================================

print()
print("=" * 90)
print("OUTPUT FILES")
print("=" * 90)


for f in sorted(
    OUTDIR.iterdir()
):

    print(
        f
    )


print()
print("Done.")
