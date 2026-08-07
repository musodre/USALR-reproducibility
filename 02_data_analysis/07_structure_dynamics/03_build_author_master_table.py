#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_final_author_table.py

Constrói a tabela canônica de dados utilizada para comunicação entre
os autores e para análises posteriores do manuscrito USALR.

IMPORTANTE
----------
Este script deve ser executado a partir de:

    USALR/python-inputs

Fontes:
    D       -> D_asymptotic_global.dat
    rho,s2  -> s2_global.dat
    rho,tau -> tau_global.dat
    Rg      -> rdf_peak_ratio_allT.dat
    Rn      -> shell_ratio_allT.dat
    A_SALR  -> SALR_area_summary.dat

Saídas:
    plots/final_author_table/author_master_table_all.dat
    plots/final_author_table/author_master_table_common.dat
    plots/final_author_table/author_master_table_all.csv
    plots/final_author_table/author_master_table_common.csv
    plots/final_author_table/author_master_table_diagnostics.dat

Definições:
    Rg = g2/g1
    Rn = n2/n1

A_SALR é EXCLUSIVAMENTE o descritor em espaço recíproco:

    A_SALR = integral_{1.3}^{3.0} S(k*) dk*

Não confundir com antigas definições exploratórias baseadas em áreas
das RDFs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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
# 1. PATHS
# =============================================================================

D_FILE = (
    DERIVED_DATA_ROOT
    / "asymptotic_diffusion"
    / "D_asymptotic_global.dat"
)

S2_FILE = (
    DERIVED_DATA_ROOT
    / "s2"
    / "s2_global.dat"
)

TAU_FILE = (
    DERIVED_DATA_ROOT
    / "tau"
    / "tau_global.dat"
)

RG_FILE = (
    DERIVED_DATA_ROOT
    / "rdf_peak_ratio_allT"
    / "rdf_peak_ratio_allT.dat"
)

RN_FILE = (
    DERIVED_DATA_ROOT
    / "shell_ratio_allT"
    / "shell_ratio_allT.dat"
)

ASALR_FILE = (
    DERIVED_DATA_ROOT
    / "static_structure_factor"
    / "Asalr_analysis"
    / "SALR_area_summary.dat"
)


# =============================================================================
# 2. OUTPUT
# =============================================================================

OUTDIR = (
    DERIVED_DATA_ROOT
    / "final_author_table"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_ALL_DAT = OUTDIR / "author_master_table_all.dat"
OUT_COMMON_DAT = OUTDIR / "author_master_table_common.dat"

OUT_ALL_CSV = OUTDIR / "author_master_table_all.csv"
OUT_COMMON_CSV = OUTDIR / "author_master_table_common.csv"

OUT_DIAG = OUTDIR / "author_master_table_diagnostics.dat"


# =============================================================================
# 3. SETTINGS
# =============================================================================

# precisão usada para casar P,T provenientes de arquivos diferentes
KEY_DECIMALS = 6

# domínio completo da tabela
P_MIN = 0.10
P_MAX = 6.00

T_MIN = 0.01
T_MAX = 0.60

# Também gerar diagnóstico específico para o domínio usado
# nas correlações estrutura--dinâmica do paper
MOBILE_T_MIN = 0.20


# =============================================================================
# 4. BASIC UTILITIES
# =============================================================================

def check_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n"
            f"    {path.resolve()}\n"
        )


def numeric_dataframe(path):

    """
    Lê arquivo numérico ignorando linhas iniciadas por #.
    """

    check_file(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python"
    )

    return df


def add_keys(df):

    df = df.copy()

    df["P"] = pd.to_numeric(
        df["P"],
        errors="coerce"
    )

    df["T"] = pd.to_numeric(
        df["T"],
        errors="coerce"
    )

    df["Pkey"] = np.round(
        df["P"],
        KEY_DECIMALS
    )

    df["Tkey"] = np.round(
        df["T"],
        KEY_DECIMALS
    )

    return df


def filter_domain(df):

    df = df.copy()

    mask = (
        np.isfinite(df["P"])
        &
        np.isfinite(df["T"])
        &
        (df["P"] >= P_MIN)
        &
        (df["P"] <= P_MAX)
        &
        (df["T"] >= T_MIN)
        &
        (df["T"] <= T_MAX)
    )

    return df.loc[mask].copy()


def deduplicate(df, value_columns, label):

    """
    Estados duplicados são promediados por (P,T).

    Isso é particularmente importante para tau_global.dat,
    que sabemos conter alguns estados duplicados.
    """

    before = len(df)

    agg = {
        col: "mean"
        for col in value_columns
        if col in df.columns
    }

    out = (
        df
        .groupby(
            ["Pkey", "Tkey"],
            as_index=False
        )
        .agg(agg)
    )

    after = len(out)

    print(
        f"{label:12s}: "
        f"{before:5d} linhas -> "
        f"{after:5d} estados únicos"
    )

    return out


# =============================================================================
# 5. DIFFUSION
# =============================================================================

def load_diffusion():

    raw = numeric_dataframe(
        D_FILE
    )

    if raw.shape[1] < 3:

        raise RuntimeError(
            "Formato inesperado em "
            f"{D_FILE}"
        )

    df = pd.DataFrame({

        "P": raw.iloc[:, 0],

        "T": raw.iloc[:, 1],

        "D": raw.iloc[:, 2],

    })

    # colunas auxiliares, se existirem

    if raw.shape[1] >= 4:

        df["D_residual_slope"] = (
            raw.iloc[:, 3]
        )

    if raw.shape[1] >= 5:

        df["D_regime"] = (
            raw.iloc[:, 4]
        )

    df = add_keys(df)
    df = filter_domain(df)

    df["D"] = pd.to_numeric(
        df["D"],
        errors="coerce"
    )

    df.loc[
        df["D"] <= 0,
        "D"
    ] = np.nan

    df["lnD"] = np.log(
        df["D"]
    )

    value_columns = [
        "D",
        "lnD",
        "D_residual_slope",
        "D_regime",
    ]

    return deduplicate(
        df,
        value_columns,
        "D"
    )


# =============================================================================
# 6. s2 + rho
# =============================================================================

def load_s2():

    raw = numeric_dataframe(
        S2_FILE
    )

    if raw.shape[1] < 4:

        raise RuntimeError(
            "s2_global.dat deveria conter:\n"
            "P T rho s2"
        )

    df = pd.DataFrame({

        "P": raw.iloc[:, 0],

        "T": raw.iloc[:, 1],

        "rho_s2": raw.iloc[:, 2],

        "s2": raw.iloc[:, 3],

    })

    df = add_keys(df)
    df = filter_domain(df)

    df["minus_s2"] = -pd.to_numeric(
        df["s2"],
        errors="coerce"
    )

    return deduplicate(

        df,

        [
            "rho_s2",
            "s2",
            "minus_s2",
        ],

        "s2"
    )


# =============================================================================
# 7. tau + rho
# =============================================================================

def load_tau():

    raw = numeric_dataframe(
        TAU_FILE
    )

    if raw.shape[1] < 4:

        raise RuntimeError(
            "tau_global.dat deveria conter:\n"
            "P T rho tau"
        )

    df = pd.DataFrame({

        "P": raw.iloc[:, 0],

        "T": raw.iloc[:, 1],

        "rho_tau": raw.iloc[:, 2],

        "tau": raw.iloc[:, 3],

    })

    df = add_keys(df)
    df = filter_domain(df)

    return deduplicate(

        df,

        [
            "rho_tau",
            "tau",
        ],

        "tau"
    )


# =============================================================================
# 8. Rg = g2/g1
# =============================================================================

def load_Rg():

    raw = numeric_dataframe(
        RG_FILE
    )

    # Formato final conhecido:
    #
    # P T r1 g1 r2 g2 Rg D lnD
    #
    # portanto Rg = coluna 7 (índice Python 6)

    if raw.shape[1] < 7:

        raise RuntimeError(
            "\nFormato inesperado em "
            "rdf_peak_ratio_allT.dat\n"
            f"Número de colunas = {raw.shape[1]}\n"
        )

    df = pd.DataFrame({

        "P": raw.iloc[:, 0],

        "T": raw.iloc[:, 1],

        "Rg": raw.iloc[:, 6],

    })

    df = add_keys(df)
    df = filter_domain(df)

    return deduplicate(
        df,
        ["Rg"],
        "Rg"
    )


# =============================================================================
# 9. Rn = n2/n1
# =============================================================================

def load_Rn():

    raw = numeric_dataframe(
        RN_FILE
    )

    ncols = raw.shape[1]

    print(
        f"Rn file     : {ncols} colunas detectadas"
    )

    # -------------------------------------------------------------------------
    # Os scripts históricos que utilizam shell_ratio_allT.dat consideram
    # P,T nas duas primeiras colunas.
    #
    # Tentamos identificar automaticamente Rn usando os formatos empregados
    # no projeto.
    # -------------------------------------------------------------------------

    if ncols >= 9:

        # Formatos completos usados em alguns inputs:
        #
        # P T rho n1 n2 ntot ratio_n2_n1 D lnD
        #
        # -> ratio = coluna 7, índice 6

        P = raw.iloc[:, 0]
        T = raw.iloc[:, 1]

        candidate = raw.iloc[:, 6]

    elif ncols >= 6:

        # Formato reduzido típico:
        #
        # P T rho n1 n2 ratio
        #
        # ou estrutura equivalente onde a última coluna
        # estrutural corresponde a n2/n1.
        #
        # Primeiro tentamos razão explícita n2/n1 a partir
        # das colunas 4 e 5.

        P = raw.iloc[:, 0]
        T = raw.iloc[:, 1]

        n1 = pd.to_numeric(
            raw.iloc[:, 3],
            errors="coerce"
        )

        n2 = pd.to_numeric(
            raw.iloc[:, 4],
            errors="coerce"
        )

        candidate = n2 / n1

    elif ncols >= 3:

        # formato mínimo:
        #
        # P T Rn

        P = raw.iloc[:, 0]
        T = raw.iloc[:, 1]
        candidate = raw.iloc[:, 2]

    else:

        raise RuntimeError(
            "\nFormato inesperado em "
            "shell_ratio_allT.dat\n"
            f"Número de colunas = {ncols}\n"
        )

    df = pd.DataFrame({

        "P": P,

        "T": T,

        "Rn": candidate,

    })

    df = add_keys(df)
    df = filter_domain(df)

    df["Rn"] = pd.to_numeric(
        df["Rn"],
        errors="coerce"
    )

    return deduplicate(
        df,
        ["Rn"],
        "Rn"
    )


# =============================================================================
# 10. A_SALR
# =============================================================================

def load_ASALR():

    raw = numeric_dataframe(
        ASALR_FILE
    )

    # Formato confirmado:
    #
    # P T A_salr D lnD
    #
    # A_SALR = integral S(k*) dk*
    #           de 1.3 a 3.0

    if raw.shape[1] < 3:

        raise RuntimeError(
            "Formato inesperado em "
            "SALR_area_summary.dat"
        )

    df = pd.DataFrame({

        "P": raw.iloc[:, 0],

        "T": raw.iloc[:, 1],

        "A_SALR": raw.iloc[:, 2],

    })

    df = add_keys(df)
    df = filter_domain(df)

    return deduplicate(
        df,
        ["A_SALR"],
        "A_SALR"
    )


# =============================================================================
# 11. MERGE
# =============================================================================

def merge_outer(datasets):

    master = None

    for label, df in datasets:

        cols = [
            c for c in df.columns
            if c not in ["P", "T"]
        ]

        df2 = df[cols].copy()

        if master is None:

            master = df2

        else:

            before = len(master)

            master = pd.merge(
                master,
                df2,
                on=["Pkey", "Tkey"],
                how="outer"
            )

            print(
                f"merge {label:8s}: "
                f"{before:5d} -> "
                f"{len(master):5d}"
            )

    master["P"] = master["Pkey"]
    master["T"] = master["Tkey"]

    return master


# =============================================================================
# 12. DENSITY CONSISTENCY
# =============================================================================

def construct_density(master):

    master = master.copy()

    # rho_s2 e rho_tau deveriam coincidir
    # para os mesmos estados.

    if (
        "rho_s2" in master.columns
        and
        "rho_tau" in master.columns
    ):

        both = (
            np.isfinite(master["rho_s2"])
            &
            np.isfinite(master["rho_tau"])
        )

        delta = (
            master.loc[both, "rho_s2"]
            -
            master.loc[both, "rho_tau"]
        )

        if len(delta) > 0:

            print()
            print(
                "Checagem rho(s2) vs rho(tau)"
            )
            print(
                "N comum             =",
                len(delta)
            )
            print(
                "max |Delta rho|     =",
                np.nanmax(
                    np.abs(delta)
                )
            )
            print(
                "mean |Delta rho|    =",
                np.nanmean(
                    np.abs(delta)
                )
            )

    # Fonte prioritária:
    # rho presente no arquivo s2.
    #
    # Se ausente, usa rho_tau.

    master["rho"] = np.nan

    if "rho_s2" in master.columns:

        master["rho"] = (
            master["rho_s2"]
        )

    if "rho_tau" in master.columns:

        mask = ~np.isfinite(
            master["rho"]
        )

        master.loc[
            mask,
            "rho"
        ] = master.loc[
            mask,
            "rho_tau"
        ]

    return master


# =============================================================================
# 13. DIAGNOSTICS
# =============================================================================

def diagnostics(master):

    columns = [

        "rho",
        "D",
        "lnD",
        "Rg",
        "Rn",
        "A_SALR",
        "s2",
        "minus_s2",
        "tau",

    ]

    rows = []

    print()
    print("=" * 78)
    print("COLUMN COVERAGE")
    print("=" * 78)

    for c in columns:

        if c not in master.columns:

            n = 0
            xmin = np.nan
            xmax = np.nan

        else:

            x = pd.to_numeric(
                master[c],
                errors="coerce"
            )

            good = np.isfinite(x)

            n = int(
                good.sum()
            )

            if n > 0:

                xmin = float(
                    np.nanmin(x)
                )

                xmax = float(
                    np.nanmax(x)
                )

            else:

                xmin = np.nan
                xmax = np.nan

        print(
            f"{c:12s} "
            f"N = {n:5d}   "
            f"min = {xmin: .6e}   "
            f"max = {xmax: .6e}"
        )

        rows.append([
            c,
            n,
            xmin,
            xmax,
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "N_finite",
            "min",
            "max",
        ]
    )


# =============================================================================
# 14. MAIN
# =============================================================================

def main():

    print()
    print("=" * 78)
    print("FINAL USALR AUTHOR MASTER TABLE")
    print("=" * 78)
    print()

    print("Input files:")
    print(f"D       : {D_FILE}")
    print(f"s2      : {S2_FILE}")
    print(f"tau     : {TAU_FILE}")
    print(f"Rg      : {RG_FILE}")
    print(f"Rn      : {RN_FILE}")
    print(f"A_SALR  : {ASALR_FILE}")
    print()

    # -------------------------------------------------------------------------
    # leitura
    # -------------------------------------------------------------------------

    D = load_diffusion()
    s2 = load_s2()
    tau = load_tau()
    Rg = load_Rg()
    Rn = load_Rn()
    asalr = load_ASALR()

    datasets = [

        ("D", D),
        ("s2", s2),
        ("tau", tau),
        ("Rg", Rg),
        ("Rn", Rn),
        ("ASALR", asalr),

    ]

    # -------------------------------------------------------------------------
    # união
    # -------------------------------------------------------------------------

    master = merge_outer(
        datasets
    )

    master = construct_density(
        master
    )

    # -------------------------------------------------------------------------
    # organização
    # -------------------------------------------------------------------------

    final_columns = [

        "T",
        "P",
        "rho",
        "D",
        "lnD",
        "Rg",
        "Rn",
        "A_SALR",
        "s2",
        "minus_s2",
        "tau",

    ]

    for c in final_columns:

        if c not in master.columns:

            master[c] = np.nan

    master = master[
        final_columns
    ]

    master = master.sort_values(
        ["T", "P"]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # tabela COMMON
    # -------------------------------------------------------------------------

    required = [

        "T",
        "P",
        "rho",
        "D",
        "lnD",
        "Rg",
        "Rn",
        "A_SALR",
        "s2",
        "minus_s2",
        "tau",

    ]

    mask_common = np.ones(
        len(master),
        dtype=bool
    )

    for c in required:

        mask_common &= np.isfinite(
            pd.to_numeric(
                master[c],
                errors="coerce"
            )
        )

    common = master.loc[
        mask_common
    ].copy()

    common = common.sort_values(
        ["T", "P"]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # domínio móvel do paper
    # -------------------------------------------------------------------------

    mobile_common = common[
        common["T"] >= MOBILE_T_MIN
    ]

    # -------------------------------------------------------------------------
    # diagnóstico
    # -------------------------------------------------------------------------

    diag = diagnostics(
        master
    )

    print()
    print("=" * 78)
    print("STATE COUNTS")
    print("=" * 78)

    print(
        f"Union of all states          = {len(master)}"
    )

    print(
        f"Complete common states       = {len(common)}"
    )

    print(
        f"Common states with T >= 0.20 = {len(mobile_common)}"
    )

    print()

    if len(common) > 0:

        print(
            "Common domain:"
        )

        print(
            f"P = "
            f"{common['P'].min():.3f}"
            f" -- "
            f"{common['P'].max():.3f}"
        )

        print(
            f"T = "
            f"{common['T'].min():.3f}"
            f" -- "
            f"{common['T'].max():.3f}"
        )

    if len(mobile_common) > 0:

        print()

        print(
            "Mobile common domain:"
        )

        print(
            f"P = "
            f"{mobile_common['P'].min():.3f}"
            f" -- "
            f"{mobile_common['P'].max():.3f}"
        )

        print(
            f"T = "
            f"{mobile_common['T'].min():.3f}"
            f" -- "
            f"{mobile_common['T'].max():.3f}"
        )

    # -------------------------------------------------------------------------
    # save DAT
    # -------------------------------------------------------------------------

    master.to_csv(

        OUT_ALL_DAT,

        sep=" ",

        index=False,

        na_rep="nan",

        float_format="%.10e"

    )

    common.to_csv(

        OUT_COMMON_DAT,

        sep=" ",

        index=False,

        na_rep="nan",

        float_format="%.10e"

    )

    # -------------------------------------------------------------------------
    # save CSV
    # -------------------------------------------------------------------------

    master.to_csv(

        OUT_ALL_CSV,

        index=False,

        na_rep="NaN",

        float_format="%.10e"

    )

    common.to_csv(

        OUT_COMMON_CSV,

        index=False,

        na_rep="NaN",

        float_format="%.10e"

    )

    diag.to_csv(

        OUT_DIAG,

        sep=" ",

        index=False,

        float_format="%.10e"

    )

    # -------------------------------------------------------------------------
    # finished
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FILES WRITTEN")
    print("=" * 78)

    print(
        OUT_ALL_DAT
    )

    print(
        OUT_COMMON_DAT
    )

    print(
        OUT_ALL_CSV
    )

    print(
        OUT_COMMON_CSV
    )

    print(
        OUT_DIAG
    )

    print()
    print(
        "Final table columns:"
    )

    print(
        "T P rho D lnD Rg Rn "
        "A_SALR s2 minus_s2 tau"
    )

    print()
    print(
        "Done."
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
