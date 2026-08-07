import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pathlib import Path
from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT


# ==========================================================
# INPUT
# ==========================================================

rdf_pattern = str(RAW_DATA_ROOT / 'allpress' / 'P_*' / 'rdfs' / 'P_*_T_*.rdf')

Dfile = str(DERIVED_DATA_ROOT / 'asymptotic_diffusion' / 'D_asymptotic_global.dat')

outdir = str(DERIVED_DATA_ROOT / 'rdf_peak_ratio_allT')
os.makedirs(outdir, exist_ok=True)

SELECTED_T = [
    0.20, 0.22, 0.24, 0.26, 0.28,
    0.30, 0.32, 0.34, 0.36, 0.38,
    0.40, 0.42, 0.44, 0.46, 0.48,
    0.50, 0.52, 0.54, 0.56, 0.58,
    0.60
]

# ==========================================================
# JANELAS PARA BUSCA DOS PICOS
# ==========================================================

peak1_min = 1.10
peak1_max = 1.35

peak2_min = 1.85
peak2_max = 2.15

# ==========================================================
# FUNÇÕES
# ==========================================================

def get_PT_from_rdf_filename(filename):
    base = os.path.basename(filename)
    txt = base.replace('.rdf', '')
    partP, partT = txt.split('_T_')

    P = float(partP.replace('P_', ''))
    T = float(partT)

    return P, T


def load_rdf(filename):
    data = np.loadtxt(
        filename,
        skiprows=4,
        usecols=(1, 3)
    )

    r = data[:, 0]
    g = data[:, 1]

    return r, g


def load_diffusion_table(filename):
    data = np.loadtxt(filename)

    P = data[:, 0]
    T = data[:, 1]
    D = data[:, 2]

    diffusion = {}

    mask = (
        np.isfinite(D)
        &
        (D > 0.0)
    )

    P = P[mask]
    T = T[mask]
    D = D[mask]

    for p, t, d in zip(P, T, D):
        key = (round(p, 6), round(t, 6))
        diffusion[key] = d

    return diffusion


def peak_height(r, g, rmin, rmax):
    mask = (
        (r >= rmin)
        &
        (r <= rmax)
    )

    rr = r[mask]
    gg = g[mask]

    if len(rr) == 0:
        return np.nan, np.nan

    idx = np.argmax(gg)

    r_peak = rr[idx]
    g_peak = gg[idx]

    return r_peak, g_peak


# ==========================================================
# LEITURA
# ==========================================================

diffusion = load_diffusion_table(Dfile)

rdf_files = sorted(glob.glob(rdf_pattern))

if len(rdf_files) == 0:
    raise RuntimeError('Nenhum RDF encontrado. Verifique rdf_pattern.')

# ==========================================================
# LOOP
# ==========================================================

rows = []

for rdf_file in rdf_files:
    P, T = get_PT_from_rdf_filename(rdf_file)

    if not any(np.isclose(T, tt, atol=1e-6) for tt in SELECTED_T):
        continue

    key = (round(P, 6), round(T, 6))

    if key not in diffusion:
        continue

    D = diffusion[key]

    r, g = load_rdf(rdf_file)

    r1, g1 = peak_height(
        r,
        g,
        peak1_min,
        peak1_max
    )

    r2, g2 = peak_height(
        r,
        g,
        peak2_min,
        peak2_max
    )

    if not np.isfinite(g1):
        continue

    if not np.isfinite(g2):
        continue

    if g1 <= 0.0:
        continue

    Rg = g2 / g1

    rows.append([
        P,
        T,
        r1,
        g1,
        r2,
        g2,
        Rg,
        D,
        np.log(D)
    ])

rows = np.array(rows)

if rows.size == 0:
    raise RuntimeError('Nenhum ponto válido foi cruzado.')

order = np.lexsort((rows[:, 0], rows[:, 1]))
rows = rows[order]

# ==========================================================
# SALVAR TABELA
# ==========================================================

outfile_dat = os.path.join(
    outdir,
    'rdf_peak_ratio_allT.dat'
)

header = (
    'P T r_peak1 g_peak1 r_peak2 g_peak2 Rg_g2_over_g1 D lnD '
    f'peak1=[{peak1_min},{peak1_max}] '
    f'peak2=[{peak2_min},{peak2_max}]'
)

np.savetxt(
    outfile_dat,
    rows,
    header=header,
    fmt='%.10f'
)

print(f'\nTabela salva: {outfile_dat}')
print(f'Número de pontos válidos: {len(rows)}')

# ==========================================================
# COLUNAS
# ==========================================================

P = rows[:, 0]
T = rows[:, 1]
r1 = rows[:, 2]
g1 = rows[:, 3]
r2 = rows[:, 4]
g2 = rows[:, 5]
Rg = rows[:, 6]
lnD = rows[:, 8]

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
    'figure.dpi': 140
})

# ==========================================================
# FIGURA MULTIPAINEL
# ==========================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

# ==========================================================
# (a) g1 e g2 vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    yy1 = g1[maskT]
    yy2 = g2[maskT]

    order = np.argsort(pp)

    ax1.plot(
        pp[order],
        yy1[order],
        '-o',
        ms=2.8,
        lw=1.0,
        alpha=0.75
    )

    ax1.plot(
        pp[order],
        yy2[order],
        '--s',
        ms=2.8,
        lw=1.0,
        alpha=0.75
    )

ax1.set_xlabel(r'$P^*$')
ax1.set_ylabel(r'$g_1,\ g_2$')
ax1.set_title(r'(a) RDF peak heights')
ax1.grid(alpha=0.18)

# ==========================================================
# (b) Rg vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    yy = Rg[maskT]

    order = np.argsort(pp)

    ax2.plot(
        pp[order],
        yy[order],
        '-o',
        ms=3,
        lw=1.1,
        label=rf'$T={Temp:.2f}$'
    )

ax2.set_xlabel(r'$P^*$')
ax2.set_ylabel(r'$g_2/g_1$')
ax2.set_title(r'(b) RDF peak-height ratio')
ax2.grid(alpha=0.18)

# ==========================================================
# (c) lnD vs Rg
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    xx = Rg[maskT]
    yy = lnD[maskT]

    order = np.argsort(xx)

    ax3.plot(
        xx[order],
        yy[order],
        '-o',
        ms=3,
        lw=1.1
    )

ax3.set_xlabel(r'$g_2/g_1$')
ax3.set_ylabel(r'$\ln D$')
ax3.set_title(r'(c) Diffusion vs RDF peak ratio')
ax3.grid(alpha=0.18)

# ==========================================================
# (d) r_peak1 e r_peak2 vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    rr1 = r1[maskT]
    rr2 = r2[maskT]

    order = np.argsort(pp)

    ax4.plot(
        pp[order],
        rr1[order],
        '-o',
        ms=2.8,
        lw=1.0,
        alpha=0.75
    )

    ax4.plot(
        pp[order],
        rr2[order],
        '--s',
        ms=2.8,
        lw=1.0,
        alpha=0.75
    )

ax4.set_xlabel(r'$P^*$')
ax4.set_ylabel(r'$r_{p1},\ r_{p2}$')
ax4.set_title(r'(d) RDF peak positions')
ax4.grid(alpha=0.18)

# ==========================================================
# (e) mapa Rg
# ==========================================================

tri = Triangulation(P, T)

c5 = ax5.tricontourf(
    tri,
    Rg,
    levels=30
)

fig.colorbar(
    c5,
    ax=ax5,
    label=r'$g_2/g_1$'
)

ax5.set_xlabel(r'$P^*$')
ax5.set_ylabel(r'$T^*$')
ax5.set_title(r'(e) RDF peak-ratio map')

# ==========================================================
# (f) mapa lnD
# ==========================================================

c6 = ax6.tricontourf(
    tri,
    lnD,
    levels=30
)

fig.colorbar(
    c6,
    ax=ax6,
    label=r'$\ln D$'
)

ax6.set_xlabel(r'$P^*$')
ax6.set_ylabel(r'$T^*$')
ax6.set_title(r'(f) Diffusion map')

# ==========================================================
# LEGENDA
# ==========================================================

ax2.legend(
    fontsize=7,
    ncol=2,
    frameon=False,
    loc='best'
)

# ==========================================================
# LAYOUT E SALVAR
# ==========================================================

fig.tight_layout()

outfile_png = os.path.join(
    outdir,
    'rdf_peak_ratio_multipanel.png'
)

outfile_pdf = os.path.join(
    outdir,
    'rdf_peak_ratio_multipanel.pdf'
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

print(f'\nFigura salva: {outfile_png}')
print(f'Figura salva: {outfile_pdf}')

plt.show()
plt.close(fig)
