import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.stats import pearsonr

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


# ==========================================================
# INPUT
# ==========================================================

SK_FOLDER = str(DERIVED_DATA_ROOT / 'static_structure_factor' / 'Sk_files')
D_FILE = str(DERIVED_DATA_ROOT / 'asymptotic_diffusion' / 'D_asymptotic_global.dat')

OUTDIR = str(DERIVED_DATA_ROOT / 'static_structure_factor' / 'Asalr_analysis')
os.makedirs(OUTDIR, exist_ok=True)

# Janela do pico SALR em k
KMIN_SALR = 1.3
KMAX_SALR = 3.0

SELECTED_T = [
    0.20, 0.22, 0.24, 0.26, 0.28,
    0.30, 0.32, 0.34, 0.36, 0.38,
    0.40, 0.42, 0.44, 0.46, 0.48,
    0.50, 0.52, 0.54, 0.56, 0.58,
    0.60
]

# ==========================================================
# DIFUSÃO
# ==========================================================

Ddata = np.loadtxt(D_FILE)

PD = Ddata[:, 0]
TD = Ddata[:, 1]
D  = Ddata[:, 2]

maskD = np.isfinite(D) & (D > 0.0)

PD = PD[maskD]
TD = TD[maskD]
D  = D[maskD]

diff_dict = {}

for p, t, d in zip(PD, TD, D):
    key = (round(p, 6), round(t, 6))
    diff_dict[key] = d

# ==========================================================
# LEITURA DOS S(k) E CÁLCULO DE A_SALR
# ==========================================================

rows = []

sk_files = sorted(
    glob.glob(
        os.path.join(
            SK_FOLDER,
            'Sk_P_*_T_*.dat'
        )
    )
)

if len(sk_files) == 0:
    raise RuntimeError('Nenhum arquivo S(k) encontrado em Sk_files.')

for f in sk_files:
    base = os.path.basename(f)

    try:
        txt = base.replace('Sk_P_', '').replace('.dat', '')
        partP, partT = txt.split('_T_')

        P = float(partP)
        T = float(partT)

    except Exception:
        continue

    if not any(np.isclose(T, tt, atol=1e-6) for tt in SELECTED_T):
        continue

    key = (round(P, 6), round(T, 6))

    if key not in diff_dict:
        continue

    data = np.loadtxt(f)

    k = data[:, 0]
    Sk = data[:, 1]

    maskK = (
        (k >= KMIN_SALR)
        &
        (k <= KMAX_SALR)
    )

    if np.sum(maskK) < 2:
        continue

    A_salr = np.trapezoid(
        Sk[maskK],
        k[maskK]
    )

    Dval = diff_dict[key]
    lnD = np.log(Dval)

    rows.append([
        P,
        T,
        A_salr,
        Dval,
        lnD
    ])

rows = np.array(rows)

if rows.size == 0:
    raise RuntimeError('Nenhum ponto válido foi cruzado entre S(k) e D.')

order = np.lexsort((rows[:, 0], rows[:, 1]))
rows = rows[order]

P = rows[:, 0]
T = rows[:, 1]
A = rows[:, 2]
D = rows[:, 3]
lnD = rows[:, 4]

# ==========================================================
# SALVAR TABELA RESUMO
# ==========================================================

summary_file = os.path.join(
    OUTDIR,
    'SALR_area_summary.dat'
)

np.savetxt(
    summary_file,
    rows,
    header=(
        'P T A_salr D lnD '
        f'KMIN_SALR={KMIN_SALR} KMAX_SALR={KMAX_SALR}'
    ),
    fmt='%.10e'
)

print(f'\nTabela salva: {summary_file}')
print(f'Pontos válidos: {len(rows)}')

# ==========================================================
# CORRELAÇÃO GLOBAL
# ==========================================================

corr, pvalue = pearsonr(A, lnD)

coef = np.polyfit(A, lnD, 1)
slope = coef[0]
intercept = coef[1]

print('\n====================================')
print(f'Pearson r = {corr:.5f}')
print(f'p-value   = {pvalue:.5e}')
print(f'lnD = {slope:.6f} * A_SALR + {intercept:.6f}')
print('====================================\n')

# ==========================================================
# ESTILO
# ==========================================================

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 14,
    'axes.linewidth': 1.3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'figure.dpi': 140
})

# ==========================================================
# FIGURA 1: MULTIPAINEL
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 10)
)

ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

temps = sorted(np.unique(T))

# ----------------------------------------------------------
# (a) A_SALR vs P
# ----------------------------------------------------------

for Temp in temps:
    maskT = np.isclose(T, Temp, atol=1e-6)

    pp = P[maskT]
    aa = A[maskT]

    order = np.argsort(pp)

    ax1.plot(
        pp[order],
        aa[order],
        '-o',
        ms=3,
        lw=1.2,
        label=rf'$T={Temp:.2f}$'
    )

ax1.set_xlabel(r'$P^*$')
ax1.set_ylabel(r'$A_{\mathrm{SALR}}$')
ax1.set_title(r'(a) SALR peak area')
ax1.grid(alpha=0.18)

# ----------------------------------------------------------
# (b) lnD vs A_SALR por isotermas
# ----------------------------------------------------------

for Temp in temps:
    maskT = np.isclose(T, Temp, atol=1e-6)

    aa = A[maskT]
    dd = lnD[maskT]

    order = np.argsort(aa)

    ax2.plot(
        aa[order],
        dd[order],
        '-o',
        ms=3,
        lw=1.2
    )

ax2.set_xlabel(r'$A_{\mathrm{SALR}}$')
ax2.set_ylabel(r'$\ln D$')
ax2.set_title(r'(b) Diffusion vs SALR area')
ax2.grid(alpha=0.18)

# ----------------------------------------------------------
# (c) mapa A_SALR
# ----------------------------------------------------------

tri = Triangulation(P, T)

c3 = ax3.tricontourf(
    tri,
    A,
    levels=30
)

fig.colorbar(
    c3,
    ax=ax3,
    label=r'$A_{\mathrm{SALR}}$'
)

ax3.set_xlabel(r'$P^*$')
ax3.set_ylabel(r'$T^*$')
ax3.set_title(r'(c) SALR-area map')

# ----------------------------------------------------------
# (d) mapa lnD
# ----------------------------------------------------------

c4 = ax4.tricontourf(
    tri,
    lnD,
    levels=30
)

fig.colorbar(
    c4,
    ax=ax4,
    label=r'$\ln D$'
)

ax4.set_xlabel(r'$P^*$')
ax4.set_ylabel(r'$T^*$')
ax4.set_title(r'(d) Diffusion map')

ax1.legend(
    fontsize=7,
    ncol=2,
    frameon=False,
    loc='best'
)

fig.tight_layout()

outfile_png = os.path.join(
    OUTDIR,
    'Asalr_diffusion_multipanel.png'
)

outfile_pdf = os.path.join(
    OUTDIR,
    'Asalr_diffusion_multipanel.pdf'
)

fig.savefig(
    outfile_png,
    dpi=300,
    bbox_inches='tight'
)

fig.savefig(
    outfile_pdf,
    bbox_inches='tight'
)

print(f'Figura salva: {outfile_png}')
print(f'Figura salva: {outfile_pdf}')

plt.show()
plt.close(fig)

# ==========================================================
# FIGURA 2: COLAPSO COLORIDO POR T
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

sc = ax.scatter(
    A,
    lnD,
    c=T,
    cmap='viridis',
    s=42,
    edgecolor='none'
)

xfit = np.linspace(A.min(), A.max(), 500)
yfit = slope * xfit + intercept

ax.plot(
    xfit,
    yfit,
    'k--',
    lw=2.2,
    label=rf'$r={corr:.3f}$'
)

cbar = fig.colorbar(
    sc,
    ax=ax
)

cbar.set_label(r'$T^*$')

ax.set_xlabel(r'$A_{\mathrm{SALR}}$')
ax.set_ylabel(r'$\ln D$')

ax.set_title(
    r'$\ln D$ versus SALR-order area'
)

ax.grid(alpha=0.20)

ax.legend(
    frameon=False
)

fig.tight_layout()

outfile_png = os.path.join(
    OUTDIR,
    'LnD_vs_Asalr_collapse.png'
)

outfile_pdf = os.path.join(
    OUTDIR,
    'LnD_vs_Asalr_collapse.pdf'
)

fig.savefig(
    outfile_png,
    dpi=300,
    bbox_inches='tight'
)

fig.savefig(
    outfile_pdf,
    bbox_inches='tight'
)

print(f'Figura salva: {outfile_png}')
print(f'Figura salva: {outfile_pdf}')

plt.show()
plt.close(fig)
