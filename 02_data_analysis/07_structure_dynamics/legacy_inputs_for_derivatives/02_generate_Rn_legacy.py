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

rho_file = str(DERIVED_DATA_ROOT / 's2' / 's2_global.dat')
Dfile = str(DERIVED_DATA_ROOT / 'asymptotic_diffusion' / 'D_asymptotic_global.dat')

outdir = str(DERIVED_DATA_ROOT / 'shell_ratio_allT')
os.makedirs(outdir, exist_ok=True)

# ==========================================================
# TEMPERATURAS
# ==========================================================

SELECTED_T = [
    0.20, 0.22, 0.24, 0.26, 0.28,
    0.30, 0.32, 0.34, 0.36, 0.38,
    0.40, 0.42, 0.44, 0.46, 0.48,
    0.50, 0.52, 0.54, 0.56, 0.58,
    0.60
]

# ==========================================================
# LIMITES DOS SHELLS
# ==========================================================

r1_min = 1.55
r2_max = 2.30

# ==========================================================
# FUNÇÕES
# ==========================================================

def get_PT_from_rdf_filename(filename):
    base = os.path.basename(filename)

    # Exemplo: P_0.600_T_0.20.rdf
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


def load_density_table(filename):
    data = np.loadtxt(filename)

    P = data[:, 0]
    T = data[:, 1]
    rho = data[:, 2]

    density = {}

    for p, t, rh in zip(P, T, rho):
        key = (round(p, 6), round(t, 6))
        density[key] = rh

    return density


def load_diffusion_table(filename):
    data = np.loadtxt(filename)

    P = data[:, 0]
    T = data[:, 1]
    D = data[:, 2]
    regime = data[:, 4]

    diffusion = {}

    mask = (
        np.isfinite(D)
        &
        (D > 0.0)
    )

    P = P[mask]
    T = T[mask]
    D = D[mask]
    regime = regime[mask]

    for p, t, d, r in zip(P, T, D, regime):
        key = (round(p, 6), round(t, 6))

        diffusion[key] = d

    return diffusion


def coordination_number(r, g, rho, rmin, rmax):
    mask = (r >= rmin) & (r <= rmax)

    rr = r[mask]
    gg = g[mask]

    if len(rr) < 2:
        return np.nan

    integrand = gg * rr**2

    n = 4.0 * np.pi * rho * np.trapezoid(integrand, rr)

    return n


# ==========================================================
# LEITURA DAS TABELAS
# ==========================================================

density = load_density_table(rho_file)
diffusion = load_diffusion_table(Dfile)

rdf_files = sorted(glob.glob(rdf_pattern))

if len(rdf_files) == 0:
    raise RuntimeError('Nenhum RDF encontrado. Verifique rdf_pattern.')

# ==========================================================
# LOOP SOBRE TODOS OS RDFS
# ==========================================================

rows = []

for rdf_file in rdf_files:
    P, T = get_PT_from_rdf_filename(rdf_file)

    if not any(np.isclose(T, tt, atol=1e-6) for tt in SELECTED_T):
        continue

    key = (round(P, 6), round(T, 6))

    if key not in density:
        print(f'Densidade não encontrada: P={P:.3f}, T={T:.2f}')
        continue

    if key not in diffusion:
        print(f'Difusão não encontrada: P={P:.3f}, T={T:.2f}')
        continue

    rho = density[key]
    D = diffusion[key]

    r, g = load_rdf(rdf_file)

    n1 = coordination_number(r, g, rho, 0.0, r1_min)
    n2 = coordination_number(r, g, rho, r1_min, r2_max)
    ntot = coordination_number(r, g, rho, 0.0, r2_max)

    if not np.isfinite(n1) or not np.isfinite(n2) or n1 <= 0:
        continue

    ratio = n2 / n1
    f1 = n1 / ntot
    f2 = n2 / ntot
    lnD = np.log(D)

    rows.append([
        P,
        T,
        rho,
        n1,
        n2,
        ntot,
        ratio,
        f1,
        f2,
        D,
        lnD
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
    'shell_ratio_allT.dat'
)

header = (
    'P T rho n1 n2 ntot ratio_n2_n1 '
    'f1_n1_ntot f2_n2_ntot D lnD'
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
# EXTRAÇÃO DAS COLUNAS
# ==========================================================

P = rows[:, 0]
T = rows[:, 1]
ratio = rows[:, 6]
lnD = rows[:, 10]

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
# FIGURA MULTIPAINEL
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# ==========================================================
# PAINEL A: ratio vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    rr = ratio[maskT]

    order = np.argsort(pp)

    ax1.plot(
        pp[order],
        rr[order],
        '-o',
        ms=3,
        lw=1.2,
        label=rf'$T={Temp:.2f}$'
    )

ax1.set_xlabel(r'$P^*$')
ax1.set_ylabel(r'$n_2/n_1$')
ax1.set_title(r'(a) Shell ratio along isotherms')
ax1.grid(alpha=0.18)

# ==========================================================
# PAINEL B: lnD vs ratio
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    rr = ratio[maskT]
    dd = lnD[maskT]

    order = np.argsort(rr)

    ax2.plot(
        rr[order],
        dd[order],
        '-o',
        ms=3,
        lw=1.2
    )

ax2.set_xlabel(r'$n_2/n_1$')
ax2.set_ylabel(r'$\ln D$')
ax2.set_title(r'(b) Diffusion vs shell ratio')
ax2.grid(alpha=0.18)

# ==========================================================
# PAINEL C: mapa ratio
# ==========================================================

tri_ratio = Triangulation(P, T)

c3 = ax3.tricontourf(
    tri_ratio,
    ratio,
    levels=30
)

fig.colorbar(
    c3,
    ax=ax3,
    label=r'$n_2/n_1$'
)

ax3.set_xlabel(r'$P^*$')
ax3.set_ylabel(r'$T^*$')
ax3.set_title(r'(c) Shell-ratio map')

# ==========================================================
# PAINEL D: mapa lnD
# ==========================================================

tri_lnD = Triangulation(P, T)

c4 = ax4.tricontourf(
    tri_lnD,
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

# ==========================================================
# LEGENDA
# ==========================================================

ax1.legend(
    fontsize=7,
    ncol=2,
    frameon=False,
    loc='best'
)

# ==========================================================
# LAYOUT
# ==========================================================

fig.tight_layout()

# ==========================================================
# SALVAR
# ==========================================================

outfile_png = os.path.join(
    outdir,
    'shell_ratio_diffusion_multipanel.png'
)

outfile_pdf = os.path.join(
    outdir,
    'shell_ratio_diffusion_multipanel.pdf'
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
