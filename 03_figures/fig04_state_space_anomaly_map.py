#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle



from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT, FIGURE_OUTPUT_ROOT

# ======================================================================
# 1. ESTILO GERAL
# ======================================================================

plt.rcParams.update({

    "font.family": "serif",
    "font.size": 12,

    "axes.labelsize": 18,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 7.4,
    "legend.title_fontsize": 8.0,

    "text.usetex": True,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.linewidth": 1.15,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,
})


# ======================================================================
# 2. ARQUIVOS
# ======================================================================

arquivo_csv = (
    "../analysis/"
    "mapeamento_fases_salr_final.csv"
)

arquivo_tmd = (
    "thermo_response/"
    "TMD_branch_tracking/"
    "TMD_physical_branch.dat"
)

arquivo_alpha = (
    "thermo_response/"
    "thermodynamic_response_functions.dat"
)

arquivo_cp = (
    "thermo_response/"
    "extrema_validation/"
    "cp_validated_locus.dat"
)

arquivo_diff_local = (
    "diffusion_anomaly_validation/"
    "diffusion_local_derivatives.dat"
)

arquivo_diff_region = (
    "diffusion_anomaly_validation/"
    "diffusion_anomaly_main_region.dat"
)


# ======================================================================
# 3. DOMÍNIO DO DIAGRAMA
# ======================================================================

P_DIAGRAM_MIN = 0.10
P_DIAGRAM_MAX = 6.00

T_DIAGRAM_MIN = 0.02
T_DIAGRAM_MAX = 0.60


# ======================================================================
# 4. DOMÍNIO DA ANOMALIA DINÂMICA
# ======================================================================

P_DYNAMIC_MIN = 0.10
P_DYNAMIC_MAX = 6.00

T_DYNAMIC_MIN = 0.02
T_DYNAMIC_MAX = 0.60


# ======================================================================
# 5. PARÂMETROS DA ANOMALIA DE DENSIDADE
# ======================================================================

ALPHA_ROOT_TMIN = 0.12
ALPHA_ROOT_TMAX = 0.34

ALPHA_MIN_SIDE_POINTS = 2

MAX_ALPHA_TMD_DISTANCE = 0.030

# ligeiramente mais visível
ALPHA_REGION_ALPHA = 0.115


# ======================================================================
# 6. PARÂMETROS DA ANOMALIA DINÂMICA
# ======================================================================

# menos densa que antes
DIFFUSION_HATCH = "//"

plt.rcParams[
    "hatch.linewidth"
] = 0.20

DIFFUSION_HATCH_COLOR = (
    0.20,
    0.20,
    0.20,
    0.17
)

DRAW_DIFFUSION_RESOLVED_BOUNDARY = True

DIFFUSION_BOUNDARY_LW = 0.85


# ======================================================================
# 7. LOCALIZADOR DE ARQUIVOS
# ======================================================================

def localizar_arquivo(caminho):

    candidatos = [

        Path(caminho),

        Path("../") / caminho,

        Path("../analysis") / caminho,

        Path(__file__).resolve().parent / caminho,

        Path(__file__).resolve().parent
        / "../"
        / caminho,
    ]


    for candidato in candidatos:

        candidato = candidato.resolve()

        if candidato.exists():

            return candidato


    return Path(caminho)


arquivo_tmd = localizar_arquivo(
    arquivo_tmd
)

arquivo_alpha = localizar_arquivo(
    arquivo_alpha
)

arquivo_cp = localizar_arquivo(
    arquivo_cp
)

arquivo_diff_local = localizar_arquivo(
    arquivo_diff_local
)

arquivo_diff_region = localizar_arquivo(
    arquivo_diff_region
)


# ======================================================================
# 8. ARQUIVO PRINCIPAL
# ======================================================================

if not os.path.exists(
    arquivo_csv
):

    raise FileNotFoundError(
        f"Arquivo não encontrado: "
        f"{arquivo_csv}"
    )


df_completo = pd.read_csv(
    arquivo_csv
)


df = df_completo[
    df_completo[
        "Pressao"
    ]
    <=
    P_DIAGRAM_MAX
].copy()


# ======================================================================
# 9. FILTRO VERTICAL DAS FASES
# ======================================================================

df_lista_suave = []


fases_globais = sorted(
    df[
        "Fase_Provavel"
    ].unique()
)


str_to_int = {
    fase: i
    for i, fase
    in enumerate(
        fases_globais
    )
}


int_to_str = {
    i: fase
    for i, fase
    in enumerate(
        fases_globais
    )
}


for temp, grupo in df.groupby(
    "Temperatura"
):

    grupo = grupo.sort_values(
        "Pressao"
    ).copy()


    pressoes = grupo[
        "Pressao"
    ].to_numpy()


    fases_originais = grupo[
        "Fase_Provavel"
    ].to_numpy()


    fases_num = np.array(
        [
            str_to_int[f]
            for f
            in fases_originais
        ]
    )


    fases_num_suave = np.copy(
        fases_num
    )


    tamanho_janela = 7

    half = (
        tamanho_janela
        //
        2
    )


    for i in range(
        len(fases_num)
    ):

        if (
            temp > 0.45
            and
            pressoes[i] < 0.350
        ):

            continue


        idx_inicio = max(
            0,
            i-half
        )


        idx_fim = min(
            len(fases_num),
            i+half+1
        )


        sub_secao = fases_num[
            idx_inicio:
            idx_fim
        ]


        valores, contagens = np.unique(
            sub_secao,
            return_counts=True
        )


        fases_num_suave[i] = valores[
            np.argmax(
                contagens
            )
        ]


    fases_finais = np.copy(
        fases_num_suave
    )


    for i in range(
        1,
        len(fases_finais)-1
    ):

        if (
            fases_finais[i]
            !=
            fases_finais[i-1]
            and
            fases_finais[i]
            !=
            fases_finais[i+1]
        ):

            fases_finais[i] = (
                fases_finais[
                    i-1
                ]
            )


    grupo[
        "Fase_Provavel"
    ] = [

        int_to_str[n]

        for n
        in fases_finais
    ]


    df_lista_suave.append(
        grupo
    )


df_suave = pd.concat(
    df_lista_suave,
    ignore_index=True
)


# ======================================================================
# 10. PALETA DAS FASES
# ======================================================================

paleta_cores_en = {

    "Close-Packed Solid (FCC+HCP)":
        "#00994c",

    "BCC Solid":
        "#ffd700",

    "Amorphous Solid / Structural Gel":
        "#ff8000",

    "Constrained Amorphous Solid":
        "#0000ff",

    "Gel / Amorphous Solid":
        "#e65100",

    "Structured Fluid":
        "#1b5e20",

    "Correlated Liquid / Dense Fluid":
        "#cc0000",

    "Cluster Fluid":
        "#808080",

    "Homogeneous Fluid / Gas":
        "#cc0066",
}


# ======================================================================
# 11. FIGURA
# ======================================================================

fig, ax = plt.subplots(

    figsize=(
        7.0,
        5.0
    ),

    dpi=300
)


# ======================================================================
# 12. MAPA DE FASES
# ======================================================================

sns.scatterplot(

    data=df_suave,

    x="Temperatura",
    y="Pressao",

    hue="Fase_Provavel",

    palette=paleta_cores_en,

    s=13,

    alpha=0.16,

    legend=False,

    edgecolor="none",

    ax=ax,

    zorder=1
)


# ======================================================================
# 13. TAMANHOS LOCAIS DAS CÉLULAS
# ======================================================================

def local_half_widths(values):

    values = np.asarray(

        sorted(
            np.unique(
                values
            )
        ),

        dtype=float
    )


    result = {}


    if len(values) == 1:

        result[
            values[0]
        ] = 0.01

        return result


    for i, value in enumerate(
        values
    ):

        if i == 0:

            spacing = (
                values[1]
                -
                values[0]
            )


        elif i == len(values)-1:

            spacing = (
                values[-1]
                -
                values[-2]
            )


        else:

            spacing = 0.5 * (

                values[i+1]
                -
                values[i-1]
            )


        result[
            value
        ] = 0.5 * spacing


    return result


# ======================================================================
# 14. ANOMALIA DE DIFUSÃO
# ======================================================================

diffusion_handle = None


if arquivo_diff_local.exists():

    try:

        df_diff = pd.read_csv(

            arquivo_diff_local,

            sep=r"\s+"
        )


        # coerção explícita
        df_diff[
            "P"
        ] = pd.to_numeric(

            df_diff[
                "P"
            ],

            errors="coerce"
        )


        df_diff[
            "T"
        ] = pd.to_numeric(

            df_diff[
                "T"
            ],

            errors="coerce"
        )


        df_diff = df_diff[

            np.isfinite(
                df_diff[
                    "P"
                ]
            )

            &

            np.isfinite(
                df_diff[
                    "T"
                ]
            )

            &

            (
                df_diff[
                    "P"
                ]
                >=
                P_DYNAMIC_MIN
            )

            &

            (
                df_diff[
                    "P"
                ]
                <=
                P_DYNAMIC_MAX
            )

            &

            (
                df_diff[
                    "T"
                ]
                >=
                T_DYNAMIC_MIN
            )

            &

            (
                df_diff[
                    "T"
                ]
                <=
                T_DYNAMIC_MAX
            )

        ].copy()


        anomalous = df_diff[

            df_diff[
                "class_final"
            ].isin(
                [
                    "anomalous",
                    "anomalous_bridged"
                ]
            )

        ].copy()


        T_half = local_half_widths(
            df_diff[
                "T"
            ].unique()
        )


        P_half = local_half_widths(
            df_diff[
                "P"
            ].unique()
        )


        for row in anomalous.itertuples():

            T0 = float(
                row.T
            )


            P0 = float(
                row.P
            )


            dT = T_half.get(
                T0,
                0.01
            )


            dP = P_half.get(
                P0,
                0.025
            )


            T_cell_left = max(
                T0-dT,
                T_DIAGRAM_MIN
            )


            T_cell_right = min(
                T0+dT,
                T_DIAGRAM_MAX
            )


            P_cell_bottom = max(
                P0-dP,
                P_DYNAMIC_MIN
            )


            P_cell_top = min(
                P0+dP,
                P_DYNAMIC_MAX
            )


            width = (
                T_cell_right
                -
                T_cell_left
            )


            height = (
                P_cell_top
                -
                P_cell_bottom
            )


            if (
                width <= 0.0
                or
                height <= 0.0
            ):

                continue


            rectangle = Rectangle(

                (
                    T_cell_left,
                    P_cell_bottom
                ),

                width,

                height,

                facecolor=(
                    1.0,
                    1.0,
                    1.0,
                    0.0
                ),

                edgecolor=(
                    DIFFUSION_HATCH_COLOR
                ),

                linewidth=0.0,

                hatch=(
                    DIFFUSION_HATCH
                ),

                clip_on=True,

                zorder=2.5
            )


            ax.add_patch(
                rectangle
            )


        diffusion_handle = Patch(

            facecolor="white",

            edgecolor=(
                0.30,
                0.30,
                0.30,
                0.40
            ),

            hatch=(
                DIFFUSION_HATCH
            ),

            linewidth=0.0,

            label=(

                r"$(\partial D^{\ast}/"
                r"\partial P^{\ast})_{T}>0$"
            )
        )


        print()
        print("=" * 74)
        print("DIFFUSION ANOMALY")
        print("=" * 74)


        print(
            f"Dynamic states loaded = "
            f"{len(df_diff)}"
        )


        print(
            f"Anomalous states      = "
            f"{len(anomalous)}"
        )


        print(
            f"Dynamic P range       = "
            f"{df_diff['P'].min():.3f}"
            f" -- "
            f"{df_diff['P'].max():.3f}"
        )


        print(
            f"Anomalous P range     = "
            f"{anomalous['P'].min():.3f}"
            f" -- "
            f"{anomalous['P'].max():.3f}"
        )


    except Exception as e:

        print(
            f"✕ Erro na anomalia dinâmica: "
            f"{e}"
        )


# ======================================================================
# 15. FRONTEIRA INFERIOR RESOLVIDA DA DIFUSÃO
#
# Corrige o erro de dtype/object.
# ======================================================================

if (
    DRAW_DIFFUSION_RESOLVED_BOUNDARY
    and
    arquivo_diff_region.exists()
):

    try:

        df_diff_region = pd.read_csv(

            arquivo_diff_region,

            sep=r"\s+"
        )


        # ==============================================================
        # CORREÇÃO PRINCIPAL
        # ==============================================================

        for col in [
            "T",
            "P_low",
            "P_high",
            "P_first_anomalous",
            "P_last_anomalous"
        ]:

            if col in df_diff_region.columns:

                df_diff_region[
                    col
                ] = pd.to_numeric(

                    df_diff_region[
                        col
                    ],

                    errors="coerce"
                )


        df_diff_region = df_diff_region[

            np.isfinite(
                df_diff_region[
                    "T"
                ]
            )

        ].copy()


        mask_resolved = (

            df_diff_region[
                "low_status"
            ].astype(str)
            ==
            "resolved"
        )


        mask_finite = np.isfinite(

            df_diff_region[
                "P_low"
            ].to_numpy(
                dtype=float
            )
        )


        resolved = df_diff_region[

            mask_resolved
            &
            mask_finite

        ].copy()


        resolved = resolved[

            (
                resolved[
                    "T"
                ]
                >=
                T_DYNAMIC_MIN
            )

            &

            (
                resolved[
                    "T"
                ]
                <=
                T_DYNAMIC_MAX
            )

            &

            (
                resolved[
                    "P_low"
                ]
                >=
                P_DYNAMIC_MIN
            )

            &

            (
                resolved[
                    "P_low"
                ]
                <=
                P_DYNAMIC_MAX
            )

        ].sort_values(
            "T"
        )


        if len(
            resolved
        ) > 0:

            T_available = np.sort(

                pd.to_numeric(

                    df_diff_region[
                        "T"
                    ],

                    errors="coerce"
                )
                .dropna()
                .unique()
            )


            if len(
                T_available
            ) > 1:

                dT_typical = np.median(

                    np.diff(
                        T_available
                    )
                )


            else:

                dT_typical = 0.02


            segment_T = []
            segment_P = []

            previous_T = None


            for _, row in resolved.iterrows():

                Tnow = float(
                    row[
                        "T"
                    ]
                )


                Pnow = float(
                    row[
                        "P_low"
                    ]
                )


                # ------------------------------------------------------
                # quebra a curva onde houver temperaturas não resolvidas
                # ------------------------------------------------------

                if (
                    previous_T is not None
                    and
                    Tnow-previous_T
                    >
                    1.55*dT_typical
                ):

                    if len(
                        segment_T
                    ) >= 2:

                        ax.plot(

                            segment_T,

                            segment_P,

                            linestyle="--",

                            color="0.20",

                            lw=(
                                DIFFUSION_BOUNDARY_LW
                            ),

                            alpha=0.78,

                            zorder=5
                        )


                    elif len(
                        segment_T
                    ) == 1:

                        ax.scatter(

                            segment_T,

                            segment_P,

                            s=13,

                            marker="o",

                            facecolor="0.20",

                            edgecolor="none",

                            zorder=5
                        )


                    segment_T = []
                    segment_P = []


                segment_T.append(
                    Tnow
                )


                segment_P.append(
                    Pnow
                )


                previous_T = Tnow


            # ----------------------------------------------------------
            # último segmento
            # ----------------------------------------------------------

            if len(
                segment_T
            ) >= 2:

                ax.plot(

                    segment_T,

                    segment_P,

                    linestyle="--",

                    color="0.20",

                    lw=(
                        DIFFUSION_BOUNDARY_LW
                    ),

                    alpha=0.78,

                    zorder=5
                )


            elif len(
                segment_T
            ) == 1:

                ax.scatter(

                    segment_T,

                    segment_P,

                    s=13,

                    marker="o",

                    facecolor="0.20",

                    edgecolor="none",

                    zorder=5
                )


            print(
                f"✓ Fronteira dinâmica inferior: "
                f"{len(resolved)} pontos resolved."
            )


        else:

            print(
                "⚠ Nenhum ponto resolved encontrado "
                "para a fronteira dinâmica inferior."
            )


    except Exception as e:

        print(
            f"✕ Erro ao desenhar fronteira dinâmica: "
            f"{e}"
        )


# ======================================================================
# 16. CARREGA TMD
# ======================================================================

df_tmd_ref = None


if arquivo_tmd.exists():

    try:

        df_tmd_ref = pd.read_csv(

            arquivo_tmd,

            sep=r"\s+"
        )


        for col in [
            "P",
            "T_TMD",
            "T_std"
        ]:

            if col in df_tmd_ref.columns:

                df_tmd_ref[
                    col
                ] = pd.to_numeric(

                    df_tmd_ref[
                        col
                    ],

                    errors="coerce"
                )


        df_tmd_ref = df_tmd_ref[

            np.isfinite(
                df_tmd_ref[
                    "P"
                ]
            )

            &

            np.isfinite(
                df_tmd_ref[
                    "T_TMD"
                ]
            )

        ].sort_values(
            "P"
        ).reset_index(
            drop=True
        )


    except Exception as e:

        print(
            f"✕ Erro ao carregar TMD: "
            f"{e}"
        )

        df_tmd_ref = None


# ======================================================================
# 17. INTERPOLAÇÃO DA TMD
# ======================================================================

def tmd_reference_at_pressure(P):

    if (
        df_tmd_ref is None
        or
        len(df_tmd_ref) < 2
    ):

        return np.nan


    Pvals = df_tmd_ref[
        "P"
    ].to_numpy(
        dtype=float
    )


    Tvals = df_tmd_ref[
        "T_TMD"
    ].to_numpy(
        dtype=float
    )


    if (
        P < Pvals.min()
        or
        P > Pvals.max()
    ):

        return np.nan


    return float(

        np.interp(

            P,

            Pvals,

            Tvals
        )
    )


# ======================================================================
# 18. RAÍZES alpha = 0
# ======================================================================

def encontrar_raizes_alpha(
    T,
    alpha
):

    T = np.asarray(
        T,
        dtype=float
    )


    alpha = np.asarray(
        alpha,
        dtype=float
    )


    good = (
        np.isfinite(T)
        &
        np.isfinite(alpha)
    )


    T = T[
        good
    ]

    alpha = alpha[
        good
    ]


    order = np.argsort(
        T
    )


    T = T[
        order
    ]

    alpha = alpha[
        order
    ]


    mask = (

        (T >= ALPHA_ROOT_TMIN)

        &

        (T <= ALPHA_ROOT_TMAX)
    )


    T = T[
        mask
    ]

    alpha = alpha[
        mask
    ]


    if len(T) < 5:

        return []


    roots = []


    for i in range(
        len(T)-1
    ):

        a1 = alpha[i]
        a2 = alpha[i+1]


        if not (
            a1 <= 0.0
            and
            a2 > 0.0
        ):

            continue


        i_left = max(

            0,

            i
            -
            ALPHA_MIN_SIDE_POINTS
            +
            1
        )


        i_right = min(

            len(alpha),

            i
            +
            1
            +
            ALPHA_MIN_SIDE_POINTS
        )


        left = alpha[
            i_left:
            i+1
        ]


        right = alpha[
            i+1:
            i_right
        ]


        if (
            len(left)
            <
            ALPHA_MIN_SIDE_POINTS
            or
            len(right)
            <
            ALPHA_MIN_SIDE_POINTS
        ):

            continue


        if not np.all(
            left <= 0.0
        ):

            continue


        if not np.all(
            right > 0.0
        ):

            continue


        denominator = (
            a2-a1
        )


        if abs(
            denominator
        ) < 1.0e-14:

            continue


        T1 = T[i]
        T2 = T[i+1]


        root = (

            T1

            -

            a1
            *
            (T2-T1)
            /
            denominator
        )


        roots.append(
            float(root)
        )


    return roots


# ======================================================================
# 19. FRONTEIRA FÍSICA alpha = 0
# ======================================================================

def construir_fronteira_alpha(
    dataframe
):

    if (
        df_tmd_ref is None
        or
        len(df_tmd_ref) < 2
    ):

        return pd.DataFrame()


    P_TMD_MIN = float(
        df_tmd_ref[
            "P"
        ].min()
    )


    P_TMD_MAX = float(
        df_tmd_ref[
            "P"
        ].max()
    )


    rows = []


    for P, group in dataframe.groupby(
        "P"
    ):

        P = float(P)


        if (
            P < P_TMD_MIN
            or
            P > P_TMD_MAX
        ):

            continue


        Tref = (
            tmd_reference_at_pressure(
                P
            )
        )


        if not np.isfinite(
            Tref
        ):

            continue


        roots = encontrar_raizes_alpha(

            group[
                "T"
            ].to_numpy(),

            group[
                "alpha"
            ].to_numpy()
        )


        if len(
            roots
        ) == 0:

            continue


        selected = min(

            roots,

            key=lambda r:
                abs(
                    r-Tref
                )
        )


        delta = (
            selected
            -
            Tref
        )


        if abs(
            delta
        ) > MAX_ALPHA_TMD_DISTANCE:

            continue


        rows.append({

            "P":
                P,

            "T_alpha_zero":
                float(
                    selected
                ),

            "T_TMD_reference":
                float(
                    Tref
                ),

            "delta_T":
                float(
                    delta
                ),

            "abs_delta_T":
                float(
                    abs(
                        delta
                    )
                ),
        })


    boundary = pd.DataFrame(
        rows
    )


    if len(
        boundary
    ) == 0:

        return boundary


    return (

        boundary
        .sort_values(
            "P"
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# 20. REGIÃO alpha_P < 0
# ======================================================================

alpha_handle = None


if arquivo_alpha.exists():

    try:

        df_alpha = pd.read_csv(

            arquivo_alpha,

            sep=r"\s+"
        )


        for col in [
            "P",
            "T",
            "alpha"
        ]:

            df_alpha[
                col
            ] = pd.to_numeric(

                df_alpha[
                    col
                ],

                errors="coerce"
            )


        df_alpha = df_alpha[

            np.isfinite(
                df_alpha[
                    "P"
                ]
            )

            &

            np.isfinite(
                df_alpha[
                    "T"
                ]
            )

            &

            np.isfinite(
                df_alpha[
                    "alpha"
                ]
            )

        ].copy()


        alpha_boundary = (
            construir_fronteira_alpha(
                df_alpha
            )
        )


        if len(
            alpha_boundary
        ) >= 2:

            P_boundary = (
                alpha_boundary[
                    "P"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            T_boundary = (
                alpha_boundary[
                    "T_alpha_zero"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            T_left = float(
                df_alpha[
                    "T"
                ].min()
            )


            ax.fill_betweenx(

                P_boundary,

                T_left,

                T_boundary,

                facecolor="#8064a2",

                alpha=(
                    ALPHA_REGION_ALPHA
                ),

                edgecolor="none",

                zorder=3
            )


            ax.plot(

                T_boundary,

                P_boundary,

                color="#8064a2",

                lw=0.75,

                alpha=0.42,

                zorder=4
            )


            alpha_handle = Patch(

                facecolor="#8064a2",

                edgecolor="none",

                alpha=0.18,

                label=(
                    r"$\alpha_P^{\ast}<0$"
                )
            )


            print(
                "✓ Região alpha_P < 0 adicionada."
            )


    except Exception as e:

        print(
            f"✕ Erro ao processar alpha: "
            f"{e}"
        )


# ======================================================================
# 21. TMD FÍSICA
# ======================================================================

tmd_handle = None


if (
    df_tmd_ref is not None
    and
    len(
        df_tmd_ref
    ) > 0
):

    df_tmd = (
        df_tmd_ref.copy()
    )


    ax.plot(

        df_tmd[
            "T_TMD"
        ],

        df_tmd[
            "P"
        ],

        color="#006666",

        lw=1.85,

        alpha=0.98,

        zorder=8
    )


    if (
        "final_class"
        in df_tmd.columns
    ):

        strict = df_tmd[

            df_tmd[
                "final_class"
            ]
            ==
            "validated"
        ]


        supported = df_tmd[

            df_tmd[
                "final_class"
            ]
            ==
            "branch_supported"
        ]


        if len(
            strict
        ) > 0:

            ax.errorbar(

                strict[
                    "T_TMD"
                ],

                strict[
                    "P"
                ],

                xerr=(
                    strict[
                        "T_std"
                    ]
                    if
                    "T_std"
                    in strict.columns
                    else None
                ),

                fmt="o",

                ms=5.6,

                mfc="#00cccc",

                mec="#004c4c",

                mew=1.0,

                ecolor="#006666",

                elinewidth=0.75,

                capsize=1.8,

                linestyle="none",

                zorder=10
            )


        if len(
            supported
        ) > 0:

            ax.errorbar(

                supported[
                    "T_TMD"
                ],

                supported[
                    "P"
                ],

                xerr=(
                    supported[
                        "T_std"
                    ]
                    if
                    "T_std"
                    in supported.columns
                    else None
                ),

                fmt="o",

                ms=5.6,

                mfc="white",

                mec="#006666",

                mew=1.15,

                ecolor="#006666",

                elinewidth=0.70,

                capsize=1.8,

                linestyle="none",

                zorder=10
            )


    tmd_handle = Line2D(

        [0],
        [0],

        marker="o",

        linestyle="-",

        color="#006666",

        markerfacecolor="#00cccc",

        markeredgecolor="#004c4c",

        lw=1.5,

        markersize=5.2,

        label="TMD"
    )


    print(
        f"✓ TMD física adicionada: "
        f"{len(df_tmd)} pontos."
    )


# ======================================================================
# 22. LOCUS cP MAX
# ======================================================================

cp_handle = None


SHOW_CP_LOCUS = False

if SHOW_CP_LOCUS and arquivo_cp.exists():

    try:

        df_cp = pd.read_csv(

            arquivo_cp,

            sep=r"\s+"
        )


        for col in [
            "P",
            "T_ext",
            "T_ext_std"
        ]:

            if col in df_cp.columns:

                df_cp[
                    col
                ] = pd.to_numeric(

                    df_cp[
                        col
                    ],

                    errors="coerce"
                )


        df_cp = df_cp[

            np.isfinite(
                df_cp[
                    "P"
                ]
            )

            &

            np.isfinite(
                df_cp[
                    "T_ext"
                ]
            )

        ].sort_values(
            "P"
        )


        ax.plot(

            df_cp[
                "T_ext"
            ],

            df_cp[
                "P"
            ],

            color="#8b008b",

            linestyle="--",

            lw=1.4,

            alpha=0.90,

            zorder=8
        )


        ax.errorbar(

            df_cp[
                "T_ext"
            ],

            df_cp[
                "P"
            ],

            xerr=(
                df_cp[
                    "T_ext_std"
                ]
                if
                "T_ext_std"
                in df_cp.columns
                else None
            ),

            fmt="D",

            ms=3.8,

            mfc="#da70d6",

            mec="#6a006a",

            mew=0.80,

            ecolor="#8b008b",

            elinewidth=0.65,

            capsize=1.6,

            linestyle="none",

            zorder=9
        )


        cp_handle = Line2D(

            [0],
            [0],

            marker="D",

            linestyle="--",

            color="#8b008b",

            markerfacecolor="#da70d6",

            markeredgecolor="#6a006a",

            lw=1.3,

            markersize=4.3,

            label=(
                r"$c_P^{\ast,\mathrm{max}}$"
            )
        )


        print(
            f"✓ Locus cP^max adicionado: "
            f"{len(df_cp)} pontos."
        )


    except Exception as e:

        print(
            f"✕ Erro ao processar cP: "
            f"{e}"
        )


# ======================================================================
# 23. EIXOS
# ======================================================================

ax.set_xlabel(
    r"$T^{\ast}$"
)


ax.set_ylabel(
    r"$P^{\ast}$"
)


ax.set_xlim(

    T_DIAGRAM_MIN
    -
    0.015,

    T_DIAGRAM_MAX
    +
    0.015
)


ax.set_ylim(

    P_DIAGRAM_MIN
    -
    0.15,

    P_DIAGRAM_MAX
    +
    0.15
)


ax.grid(False)


ax.tick_params(

    direction="in",

    top=True,

    right=True,

    length=4
)


# ======================================================================
# 24. STRUCTURAL-REGIME LABELS — CONTEXT ONLY
# ======================================================================
#
# The background classification is retained only to provide state-space
# context.  Labels are deliberately neutral and visually subordinate to
# the anomaly loci.
#

REGIME_LABEL_COLOR = "0.38"
REGIME_LABEL_ALPHA = 0.55

ax.text(
    0.03, 0.15,
    r'\textbf{I}',
    fontsize=8.5,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    rotation=30,
    ha='center',
    va='center',
    zorder=6
)

ax.text(
    0.06, 2.8,
    r'\textbf{CAS}',
    fontsize=9.5,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    rotation=30,
    ha='center',
    va='center',
    zorder=6
)

ax.text(
    0.03, 5.2,
    r'\textbf{CPS}',
    fontsize=8.5,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    rotation=40,
    ha='center',
    va='center',
    zorder=6
)

ax.text(
    0.10, 5.75,
    r'\textbf{II}',
    fontsize=8.0,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    rotation=30,
    ha='center',
    va='center',
    zorder=6
)

ax.text(
    0.55, 0.10,
    r'\textbf{III}',
    fontsize=8.0,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    rotation=30,
    ha='center',
    va='center',
    zorder=6
)

ax.text(
    0.405, 2.30,
    r'\textbf{CLUSTERED FLUID}',
    fontsize=9.5,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    rotation=25,
    ha='center',
    va='center',
    zorder=6
)

ax.text(
    0.375, 5.20,
    r'\textbf{DENSE FLUID}',
    fontsize=9.5,
    color=REGIME_LABEL_COLOR,
    alpha=REGIME_LABEL_ALPHA,
    ha='center',
    va='center',
    zorder=6
)


# ======================================================================
# 25. LEGENDA COMPACTA
# ======================================================================

handles = []


if diffusion_handle is not None:

    handles.append(
        diffusion_handle
    )


if alpha_handle is not None:

    handles.append(
        alpha_handle
    )


if tmd_handle is not None:

    handles.append(
        tmd_handle
    )


if cp_handle is not None:

    handles.append(
        cp_handle
    )


if len(handles) > 0:

    ax.legend(

        handles=handles,

        loc="upper right",

        fontsize=8.0,

        frameon=True,

        framealpha=0.88,

        facecolor="#ffffff",

        edgecolor="#e6e6e6",

        labelspacing=0.18,

        handletextpad=0.28,

        handlelength=1.65,

        handleheight=0.75,

        borderpad=0.28,

        borderaxespad=0.35,

        columnspacing=0.50
    )


# ======================================================================
# 26. LAYOUT
# ======================================================================

fig.tight_layout()


# ======================================================================
# 27. EXPORTAÇÃO
# ======================================================================

saida_dir = Path(
    "../analysis/"
    "PTphase_diagram"
)


saida_dir.mkdir(
    parents=True,
    exist_ok=True
)


arquivo_pdf = (

    saida_dir

    /

    "fig4_PTdiagram_USALR_contextual_final_"
    "thermo_dynamic_anomalies_P6_refined.pdf"
)


arquivo_png = (

    saida_dir

    /

    "fig4_PTdiagram_USALR_contextual_final_"
    "thermo_dynamic_anomalies_P6_refined.png"
)


fig.savefig(

    arquivo_pdf,

    bbox_inches="tight"
)


fig.savefig(

    arquivo_png,

    dpi=600,

    bbox_inches="tight"
)


plt.show()


# ======================================================================
# 28. RELATÓRIO FINAL
# ======================================================================

print()
print("=" * 82)

print(
    " FINAL P*-T* STATE-SPACE CONTEXT WITH ANOMALIES"
)

print("=" * 82)


print(
    "✓ Structural/morphological background retained as contextual information."
)

print(
    "✓ Phase colors and labels deliberately de-emphasized."
)

print(
    "✓ Diffusion anomaly plotted up to P*=6.00."
)

print(
    "✓ Diffusion hatching softened."
)

print(
    "✓ Dynamic lower-boundary dtype bug corrected."
)

print(
    "✓ Only resolved dynamic-boundary segments plotted."
)

print(
    "✓ No artificial upper dynamic boundary at P*=6.00."
)

print(
    "✓ alpha_P < 0 region slightly enhanced."
)

print(
    "✓ Physical TMD preserved."
)

print(
    "✓ cP-max locus disabled by default for the main-text figure."
)

print(
    "✓ Legend compacted."
)


print()
print(
    f"PDF:\n"
    f"  {arquivo_pdf.resolve()}"
)


print()
print(
    f"PNG:\n"
    f"  {arquivo_png.resolve()}"
)


print("=" * 82)
