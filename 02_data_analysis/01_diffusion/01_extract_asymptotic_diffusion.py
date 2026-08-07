from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import (
    RAW_DATA_ROOT,
    DERIVED_DATA_ROOT,
    FIGURE_OUTPUT_ROOT,
)
import matplotlib
matplotlib.use('Agg')

import numpy as np
import glob
import re
import os

import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

# =========================================================
# PARÂMETROS
# =========================================================

pattern = str(
    RAW_DATA_ROOT
    / "allpress"
    / "P_*"
    / "msds"
    / "*.msd"
)

# ---------------------------------------------------------
# fração final da trajetória
# ---------------------------------------------------------

FINAL_FRAC = 0.30

# ---------------------------------------------------------
# suavização
# ---------------------------------------------------------

WINDOW = 11
POLY = 2

# ---------------------------------------------------------
# thresholds da inclinação residual
# ---------------------------------------------------------

PLATEAU_THR = 0.05
SLOW_THR = 0.20

# =========================================================
# PASTAS
# =========================================================

analysis_dir = (
    DERIVED_DATA_ROOT
    / "asymptotic_diffusion"
)

plot_dir = (
    FIGURE_OUTPUT_ROOT
    / "asymptotic_diffusion"
)

analysis_dir.mkdir(
    parents=True,
    exist_ok=True,
)

plot_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nPastas verificadas.\n")
print(f"Dados derivados: {analysis_dir}")
print(f"Figuras:        {plot_dir}\n")

# =========================================================
# FUNÇÕES
# =========================================================

def parse_PT(filename):

    name = os.path.basename(filename)
    match = re.search(r'P_(\d+\.\d+)_T_(\d+\.\d+)', name)
    if not match:
        raise ValueError(
            f"Nome inválido: {name}")
    return (float(match.group(1)), float(match.group(2))    )

# =========================================================

def smooth(y):
    if len(y) < WINDOW:
        return y
    return savgol_filter(y, WINDOW, POLY )

# =========================================================

def classify_regime(slope):
    s = abs(slope)
    if s < PLATEAU_THR:
        return 0, "diffusive plateau"
    elif s < SLOW_THR:
        return 1, "slow crossover"
    else:
        return 2, "non-converged"
# =========================================================

def make_plot(t, Dt, start, Dmean, slope,
    regime_label, P, T):
    fig, ax = plt.subplots(figsize=(8,6))

    # -----------------------------------------------------
    # curva
    # -----------------------------------------------------

    ax.semilogx(

        t,
        Dt,

        linewidth=2,

        label=r'$D(t)$'

    )

    # -----------------------------------------------------
    # região final
    # -----------------------------------------------------

    ax.semilogx(

        t[start:],
        Dt[start:],

        linewidth=4,

        label='asymptotic region'

    )

    # -----------------------------------------------------
    # média
    # -----------------------------------------------------

    ax.axhline(

        Dmean,

        linestyle='--',

        linewidth=2,

        label=(
            rf'$D_{{eff}}={Dmean:.3e}$'
        )

    )

    # -----------------------------------------------------
    # labels
    # -----------------------------------------------------

    ax.set_xlabel("t")
    ax.set_ylabel("D(t)")

    ax.set_title(

        f"P={P:.3f} "
        f"T={T:.2f}\n"
        f"{regime_label}\n"
        f"residual slope={slope:.3e}"

    )

    ax.legend()

    fig.tight_layout()

    outfile = (

        plot_dir
        / f"Dt_P_{P:.3f}_T_{T:.2f}.png"

    )

    fig.savefig(

        outfile,

        dpi=120,
        bbox_inches='tight'

    )

    plt.close(fig)
    plt.close('all')

# =========================================================
# ARQUIVOS
# =========================================================

files = sorted(
    glob.glob(pattern)
)

print("\nArquivos encontrados:\n")

for f in files:
    print(f)

if len(files) == 0:

    raise RuntimeError(
        "Nenhum MSD encontrado."
    )

# =========================================================
# GLOBAL
# =========================================================

global_data = []

# =========================================================
# LOOP PRINCIPAL
# =========================================================

for nfile, f in enumerate(files, start=1):

    print("\n===================================")
    print(f"[{nfile}/{len(files)}]")
    print(f"Processando {f}")
    print("===================================\n")

    try:

        # -------------------------------------------------
        # PT
        # -------------------------------------------------

        P, T = parse_PT(f)

        # -------------------------------------------------
        # leitura
        # -------------------------------------------------

        data = np.loadtxt(f)

        t = data[:,0]
        msd = data[:,1]

        # -------------------------------------------------
        # remover zeros
        # -------------------------------------------------

        mask = (

            (t > 0)
            &
            (msd > 0)

        )

        t = t[mask]
        msd = msd[mask]

        # -------------------------------------------------
        # D(t)
        # -------------------------------------------------

        Dt = msd / (6.0*t)

        Dt = smooth(Dt)

        # -------------------------------------------------
        # região final
        # -------------------------------------------------

        start = int(

            (1.0 - FINAL_FRAC)
            * len(t)

        )

        tfit = t[start:]
        Dfit = Dt[start:]

        # -------------------------------------------------
        # fit residual
        # -------------------------------------------------

        coeffs = np.polyfit(

            tfit,
            Dfit,

            1

        )

        slope = coeffs[0]

        # -------------------------------------------------
        # D efetivo
        # -------------------------------------------------

        Dmean = np.mean(Dfit)

        # -------------------------------------------------
        # classificação
        # -------------------------------------------------

        regime, regime_label = classify_regime(
            slope
        )

        print(
            f"D_eff = {Dmean:.8e}"
        )

        print(
            f"Residual slope = {slope:.8e}"
        )

        print(
            f"Regime = {regime_label}"
        )

        # -------------------------------------------------
        # salvar dados
        # -------------------------------------------------

        outfile = (

            analysis_dir
            / f"Dt_P_{P:.3f}_T_{T:.2f}.dat"

        )

        np.savetxt(

            outfile,

            np.column_stack([

                t,
                Dt

            ]),

            header="t D(t)"

        )

        # -------------------------------------------------
        # plot
        # -------------------------------------------------

        make_plot(

            t,
            Dt,

            start,

            Dmean,
            slope,

            regime_label,

            P,
            T

        )

        print(
            "Plot salvo."
        )

        # -------------------------------------------------
        # global
        # -------------------------------------------------

        global_data.append([

            P,
            T,

            Dmean,

            slope,

            regime

        ])

        print("\nEstado concluído.\n")

    except Exception as e:

        print(
            f"Erro: {e}"
        )

# =========================================================
# GLOBAL
# =========================================================

if len(global_data) == 0:
    raise RuntimeError(
        "Nenhum estado de difusão foi processado com sucesso."
    )

global_data = np.asarray(
    global_data,
    dtype=float,
)

global_data = np.atleast_2d(
    global_data
)

# ordenação determinística por P e T
idx = np.lexsort(
    (
        global_data[:, 1],
        global_data[:, 0],
    )
)

global_data = global_data[idx]

outfile = (

    analysis_dir
    / "D_asymptotic_global.dat"

)

np.savetxt(

    outfile,

    global_data,

    header=(
        "P T D_eff residual_slope regime"
    )

)

print(
    f"\n{outfile} salvo."
)

# =========================================================
# HISTOGRAMA
# =========================================================

mask = np.isfinite(global_data[:,2])

Dvals = global_data[:,2][mask]

if len(Dvals) > 0:

    fig, ax = plt.subplots(
        figsize=(7,5)
    )

    ax.hist(

        np.log10(Dvals),

        bins=30

    )

    ax.set_xlabel("log10(D_eff)")
    ax.set_ylabel("Frequency")

    ax.set_title(
        "Distribution of asymptotic diffusion"
    )

    fig.tight_layout()

    outfile_hist = (

        plot_dir
        / "D_eff_histogram.png"

    )

    fig.savefig(

        outfile_hist,

        dpi=120,
        bbox_inches='tight'

    )

    plt.close(fig)
    plt.close('all')

    print(
        f"{outfile_hist} salvo."
    )

print("\n===================================")
print("ETAPA 1H FINALIZADA")
print("===================================")
print(f"MSDs encontrados:          {len(files)}")
print(f"Estados processados:       {len(global_data)}")
print(f"Tabela global:             {analysis_dir / 'D_asymptotic_global.dat'}")
print(f"Diretório de figuras:      {plot_dir}")
print("===================================\n")
