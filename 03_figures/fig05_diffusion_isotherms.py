import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import Akima1DInterpolator
import os


from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import DERIVED_DATA_ROOT, FIGURE_OUTPUT_ROOT

# =========================================================
# INPUT
# =========================================================

infile = DERIVED_DATA_ROOT / "asymptotic_diffusion" / "D_asymptotic_global.dat"

# =========================================================
# ISOTERMAS
# =========================================================
#T_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
#T_list = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60]

T_list = [0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56]#, 0.60]

# =========================================================
# PARÂMETROS
# =========================================================

P_max_plot = 6.0
SMOOTH_WINDOW = 7
SMOOTH_POLY = 4
N_INTERP = 400

# ---------------------------------------------------------
# ajuste polinomial das isotermas
# ---------------------------------------------------------

POLY_DEGREE_ISOTHERM = 6

# ---------------------------------------------------------
# ajuste das linhas de mínimos/máximos
# ---------------------------------------------------------

POLY_DEGREE_EXTREMA = 4

# =========================================================
# OUTPUT
# =========================================================

outdir = FIGURE_OUTPUT_ROOT / "fig05_diffusion_isotherms"
os.makedirs(outdir, exist_ok=True)

# =========================================================
# LEITURA
# =========================================================

data = np.loadtxt(infile)
P_all = data[:,0]
T_all = data[:,1]
D_all = data[:,2]
regime_all = data[:,4]

# =========================================================
# SOMENTE ESTADOS DIFUSIVOS
# =========================================================

mask = ((regime_all == 0)
    &
    np.isfinite(D_all))
P_all = P_all[mask]
T_all = T_all[mask]
D_all = D_all[mask]

# =========================================================
# ESTILO
# =========================================================

plt.rcParams.update({

    'text.usetex': True,

    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],

    'mathtext.fontset': 'cm',

    'font.size': 16,

    'axes.linewidth': 1.6,

    'xtick.direction': 'in',
    'ytick.direction': 'in',

    'xtick.top': True,
    'ytick.right': True,

    'xtick.major.size': 7,
    'ytick.major.size': 7,

    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,

    'axes.grid': True,

    'grid.linestyle': '--',
    'grid.alpha': 0.35,

    'figure.dpi': 160,

    'savefig.dpi': 300

})

# =========================================================
# FIGURA
# =========================================================

fig, ax = plt.subplots(figsize=(10.5,7.2))

# =========================================================
# INSET: REGIÃO DOS MÍNIMOS
# =========================================================

axins = inset_axes(ax, width="42%", height="42%", loc='lower right', bbox_to_anchor=(-0.025, 0.05, 1, 1),
    bbox_transform=ax.transAxes, borderpad=1.0)

axins.set_yscale('log')

axins.set_xlim(0.05, 0.5)
axins.set_ylim(1.35e-6, 3.6e-5)

axins.grid(True, linestyle='--', alpha=0.35)

axins.tick_params(
    direction='in',
    top=True,
    right=True,
    labelsize=11)

axins.set_xlabel(r'$P^{\ast}$', fontsize=16, labelpad=1)
axins.set_ylabel(r'$D^{\ast}$', fontsize=16, labelpad=1)

# =========================================================
# CORES
# =========================================================
cmap = plt.get_cmap('magma', len(T_list))

# =========================================================
# ARRAYS DOS EXTREMOS
# =========================================================
Tmins = []
Pmins = []
Tmaxs = []
Pmaxs = []

# =========================================================
# LEGENDA
# =========================================================
legend_handles = []
legend_labels = []

# =========================================================
# LOOP ISOTERMAS
# =========================================================

for i, T in enumerate(T_list):

    # -----------------------------------------------------
    # seleção
    # -----------------------------------------------------

    maskT = np.isclose(T_all, T, atol=1e-6)
    P = P_all[maskT]
    D = D_all[maskT]

    if len(P) < 6:
        continue

    # -----------------------------------------------------
    # ordenação
    # -----------------------------------------------------

    order = np.argsort(P)
    P = P[order]
    D = D[order]

    # -----------------------------------------------------
    # remover pressões repetidas
    #   -----------------------------------------------------

    P_unique = []
    D_unique = []

    for pval in np.unique(P):
        mask_dup = np.isclose(P, pval, atol=1e-10)

        P_unique.append(pval)

        # se houver mais de um D para a mesma pressão,
        # usamos a média em log10(D), mais adequada para escala log
        D_unique.append(10**np.mean(np.log10(D[mask_dup])))

    P = np.array(P_unique)
    D = np.array(D_unique)

    # -----------------------------------------------------
    # limitar pressão
    # -----------------------------------------------------
    maskP = P <= P_max_plot
    P = P[maskP]
    D = D[maskP]
    if len(P) < 6:
        continue

    # =====================================================
    # INTERPOLAÇÃO PRESERVANDO A FORMA DA ISOTERMA
    # =====================================================

    Pfine = np.linspace(P.min(), P.max(), N_INTERP)

    logD = np.log10(D)
    interp = Akima1DInterpolator(P, np.log10(D))
    #interp = PchipInterpolator(P, logD)

    logDfine = interp(Pfine)

    Dfine = 10**logDfine
    # -----------------------------------------------------
    # cor
    # -----------------------------------------------------

    color = cmap(i)

    # =====================================================
    # CURVA
    # =====================================================

    line, = ax.plot(Pfine, Dfine, color=color, lw=2)
    # curva no inset
    axins.plot(Pfine, Dfine, color=color, lw=1.6)

    # pontos reais no inset
    axins.plot(P, D, 'o',  color=color, mec='white', mew=0.5, ms=3.5)

    # -----------------------------------------------------
    # pontos reais
    # -----------------------------------------------------

    ax.plot(P, D, 'o', color=color, mec='white', mew=0.75, ms=5)
    legend_handles.append(line)
    legend_labels.append(rf'$T^*={T:.2f}$')

    # =====================================================
    # MÍNIMO DA PARÁBOLA/POLIFIT
    # =====================================================

    imin = np.argmin(D)
    Pmin = P[imin]
    Dmin = D[imin]

    Pmins.append(Pmin)
    Tmins.append(T)
    ax.plot(Pmin, Dmin, marker='*', ms=15, color='#00994c', mec='white', mew=1.2, zorder=8)
    axins.plot(Pmin, Dmin, marker='*', ms=11, color='#00994c', mec='white', mew=0.8, zorder=8)
    # =====================================================
    # MÁXIMO LOCAL (dados reais)
    # =====================================================

    mask_max = P > 2.0
    if np.any(mask_max):
        P_aux = P[mask_max]
        D_aux = D[mask_max]
        imax = np.argmax(D_aux)
        # Se o máximo estiver na borda direita,
        # não o consideramos um máximo físico.
        if imax == len(D_aux) - 1:
            pass
        else:
            Pmax = P_aux[imax]
            Dmax = D_aux[imax]

            Pmaxs.append(Pmax)
            Tmaxs.append(T)

            ax.plot(Pmax, Dmax, marker='d',
            ms=8, color='white', mec='k',
            mew=1.2, linestyle='None', zorder=8)

# =========================================================
# EIXOS
# =========================================================
ax.set_xlim(0,  6.1)
ax.set_ylim(10**-6.15, 10**-3.15)
ax.set_yscale('log')
ax.set_xlabel(r'$P^{\ast}$', fontsize=30, labelpad=10)
ax.set_ylabel(r'$D^{\ast}$', fontsize=36, labelpad=10)

# =========================================================
# CAIXA INDICANDO A REGIÃO AMPLIADA
# =========================================================

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.25", lw=1.0,linestyle='--')
# =========================================================
# LEGENDA ISOTERMAS
# =========================================================

leg = ax.legend(

    legend_handles,
    legend_labels,

    loc='upper center',

    bbox_to_anchor=(0.73,0.99),

    ncol=3,

    fontsize=13,

    frameon=True,

    fancybox=True,

    framealpha=0.95

)

for lh in leg.legend_handles:
    lh.set_linewidth(2.7)

# =========================================================
# LEGENDA EXTRA
# =========================================================

from matplotlib.lines import Line2D
extra_handles = [Line2D([0],[0], marker='*', color='#00994c', markersize=18, linestyle='None',
    markeredgecolor='white', markeredgewidth=1.2, label=r'$D^{\ast}_{min}$' )]#,

    #Line2D([0],[0], marker='d', color='white', markersize=8, linestyle='None',
    #markeredgecolor='k', markeredgewidth=1.2, label=r'$D^{\ast}_{max}$'),

#    Line2D(

#        [0],[0],

#        linestyle='--',

#        color='k',

#        lw=1.5,

#        label='Polyfit (minima)'

#    ),

#    Line2D(

#        [0],[0],

#        linestyle='-.',

#        color='k',

#        lw=1.5,

#        label='Polyfit (maxima)'

#    )


leg2 = ax.legend(handles=extra_handles, loc='upper left', fontsize=16, frameon=True, fancybox=True, framealpha=0.95)

ax.add_artist(leg)

# =========================================================
# LAYOUT
# =========================================================

fig.tight_layout()

# =========================================================
# SALVAR
# =========================================================

outfile_png = (f'{outdir}/DP_USALR.png')
outfile_pdf = (f'{outdir}/DP_USALR.pdf')
fig.savefig(outfile_png, dpi=600, bbox_inches='tight')
fig.savefig(outfile_pdf, bbox_inches='tight')

print(f'\\n{outfile_png} salvo.')
print(f'{outfile_pdf} salvo.')

# =========================================================
# MOSTRAR
# =========================================================

plt.show()

# =========================================================
# LIMPEZA
# =========================================================

plt.close(fig)
plt.close('all')
