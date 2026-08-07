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

traj_pattern = str(RAW_DATA_ROOT / 'allpress' / 'trajs' / '*.lammpstrj')

Dfile = str(DERIVED_DATA_ROOT / 'asymptotic_diffusion' / 'D_asymptotic_global.dat')

outdir = str(DERIVED_DATA_ROOT / 'static_structure_factor')
os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join(outdir, 'Sk_files'), exist_ok=True)

SELECTED_T = [
    0.20, 0.22, 0.24, 0.26, 0.28,
    0.30, 0.32, 0.34, 0.36, 0.38,
    0.40, 0.42, 0.44, 0.46, 0.48,
    0.50, 0.52, 0.54, 0.56, 0.58,
    0.60
]

# ==========================================================
# PARÂMETROS S(k)
# ==========================================================

kmax = 12.0
dk = 0.05

# janelas para identificar pré-pico e pico principal
# ajuste depois de ver alguns S(k)
k_pre_min = 0.2
k_pre_max = 2.0

k_main_min = 2.0
k_main_max = 8.0

# usa apenas últimos frames? Para economizar, sim.
# se quiser média temporal, podemos adaptar depois.
USE_2D = False

# ==========================================================
# FUNÇÕES
# ==========================================================

def get_PT_from_traj_filename(filename):
    base = os.path.basename(filename)
    txt = base.replace('.lammpstrj', '')
    partP, partT = txt.split('_T_')

    P = float(partP.replace('P_', ''))
    T = float(partT)

    return P, T


def load_diffusion_table(filename):
    data = np.loadtxt(filename)

    P = data[:, 0]
    T = data[:, 1]
    D = data[:, 2]

    diffusion = {}

    mask = np.isfinite(D) & (D > 0.0)

    P = P[mask]
    T = T[mask]
    D = D[mask]

    for p, t, d in zip(P, T, D):
        diffusion[(round(p, 6), round(t, 6))] = d

    return diffusion


def read_last_frame(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    starts = [
        i for i, line in enumerate(lines)
        if line.startswith('ITEM: TIMESTEP')
    ]

    if len(starts) == 0:
        raise RuntimeError(f'Nenhum timestep em {filename}')

    start = starts[-1]

    natoms = int(lines[start + 3].strip())

    bounds = []

    for i in range(3):
        vals = lines[start + 5 + i].split()
        bounds.append([float(vals[0]), float(vals[1])])

    bounds = np.array(bounds)

    header = lines[start + 8].split()[2:]

    atom_start = start + 9
    atom_end = atom_start + natoms

    data = np.array([
        list(map(float, line.split()))
        for line in lines[atom_start:atom_end]
    ])

    col = {name: i for i, name in enumerate(header)}

    if 'x' in col and 'y' in col and 'z' in col:
        pos = np.column_stack((
            data[:, col['x']],
            data[:, col['y']],
            data[:, col['z']]
        ))

    elif 'xs' in col and 'ys' in col and 'zs' in col:
        L = bounds[:, 1] - bounds[:, 0]

        pos = np.column_stack((
            bounds[0, 0] + data[:, col['xs']] * L[0],
            bounds[1, 0] + data[:, col['ys']] * L[1],
            bounds[2, 0] + data[:, col['zs']] * L[2]
        ))

    elif 'xu' in col and 'yu' in col and 'zu' in col:
        pos = np.column_stack((
            data[:, col['xu']],
            data[:, col['yu']],
            data[:, col['zu']]
        ))

    else:
        raise RuntimeError(
            f'Não encontrei x/y/z, xs/ys/zs ou xu/yu/zu em {filename}'
        )

    return pos, bounds


def generate_k_vectors(bounds, kmax, use_2d=False):
    L = bounds[:, 1] - bounds[:, 0]

    nmax = np.floor(kmax * L / (2.0 * np.pi)).astype(int)

    kvecs = []

    if use_2d:
        for nx in range(-nmax[0], nmax[0] + 1):
            for ny in range(-nmax[1], nmax[1] + 1):
                if nx == 0 and ny == 0:
                    continue

                k = np.array([
                    2.0 * np.pi * nx / L[0],
                    2.0 * np.pi * ny / L[1],
                    0.0
                ])

                kmag = np.linalg.norm(k)

                if kmag <= kmax:
                    kvecs.append(k)

    else:
        for nx in range(-nmax[0], nmax[0] + 1):
            for ny in range(-nmax[1], nmax[1] + 1):
                for nz in range(-nmax[2], nmax[2] + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue

                    k = np.array([
                        2.0 * np.pi * nx / L[0],
                        2.0 * np.pi * ny / L[1],
                        2.0 * np.pi * nz / L[2]
                    ])

                    kmag = np.linalg.norm(k)

                    if kmag <= kmax:
                        kvecs.append(k)

    return np.array(kvecs)


def static_structure_factor(pos, bounds, kmax, dk, use_2d=False):
    n = len(pos)

    kvecs = generate_k_vectors(
        bounds,
        kmax,
        use_2d=use_2d
    )

    if len(kvecs) == 0:
        raise RuntimeError('Nenhum vetor k gerado.')

    kmags = np.linalg.norm(kvecs, axis=1)

    Sk_values = np.zeros(len(kvecs))

    for i, k in enumerate(kvecs):
        phase = pos @ k

        rho_k_real = np.sum(np.cos(phase))
        rho_k_imag = np.sum(np.sin(phase))

        Sk_values[i] = (
            rho_k_real**2
            +
            rho_k_imag**2
        ) / n

    bins = np.arange(0.0, kmax + dk, dk)

    centers = 0.5 * (bins[:-1] + bins[1:])

    Sk_binned = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)

    for i in range(len(centers)):
        mask = (
            (kmags >= bins[i])
            &
            (kmags < bins[i + 1])
        )

        if np.sum(mask) > 0:
            Sk_binned[i] = np.mean(Sk_values[mask])
            counts[i] = np.sum(mask)

    valid = np.isfinite(Sk_binned)

    return centers[valid], Sk_binned[valid], counts[valid]


def find_peak_in_window(k, Sk, kmin, kmax):
    mask = (k >= kmin) & (k <= kmax)

    kk = k[mask]
    ss = Sk[mask]

    if len(kk) == 0:
        return np.nan, np.nan

    idx = np.argmax(ss)

    return kk[idx], ss[idx]


# ==========================================================
# LEITURA
# ==========================================================

diffusion = load_diffusion_table(Dfile)

traj_files = sorted(glob.glob(traj_pattern))

if len(traj_files) == 0:
    raise RuntimeError('Nenhuma trajetória encontrada.')

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

rows = []

for traj in traj_files:
    try:
        P, T = get_PT_from_traj_filename(traj)
    except Exception:
        continue

    if not any(np.isclose(T, tt, atol=1e-6) for tt in SELECTED_T):
        continue

    key = (round(P, 6), round(T, 6))

    if key not in diffusion:
        continue

    print(f'Calculando S(k): P={P:.3f}, T={T:.2f}')

    pos, bounds = read_last_frame(traj)

    k, Sk, nk = static_structure_factor(
        pos,
        bounds,
        kmax,
        dk,
        use_2d=USE_2D
    )

    kpre, Spre = find_peak_in_window(
        k,
        Sk,
        k_pre_min,
        k_pre_max
    )

    kmain, Smain = find_peak_in_window(
        k,
        Sk,
        k_main_min,
        k_main_max
    )

    ratio = Spre / Smain if Smain > 0 else np.nan

    D = diffusion[key]
    lnD = np.log(D)

    outfile_sk = os.path.join(
        outdir,
        'Sk_files',
        f'Sk_P_{P:.3f}_T_{T:.2f}.dat'
    )

    np.savetxt(
        outfile_sk,
        np.column_stack([k, Sk]),
        header='k S(k)'
    )

    rows.append([
        P,
        T,
        kpre,
        Spre,
        kmain,
        Smain,
        ratio,
        D,
        lnD
    ])

rows = np.array(rows)

if rows.size == 0:
    raise RuntimeError('Nenhum S(k) calculado.')

order = np.lexsort((rows[:, 0], rows[:, 1]))
rows = rows[order]

# ==========================================================
# SALVAR RESUMO
# ==========================================================

summary_file = os.path.join(
    outdir,
    'Sk_summary.dat'
)

np.savetxt(
    summary_file,
    rows,
    header='P T kpre Spre kmain Smain Spre_over_Smain D lnD',
    fmt='%.10f'
)

print(f'\nResumo salvo: {summary_file}')

# ==========================================================
# COLUNAS
# ==========================================================

P = rows[:, 0]
T = rows[:, 1]
kpre = rows[:, 2]
Spre = rows[:, 3]
kmain = rows[:, 4]
Smain = rows[:, 5]
ratio = rows[:, 6]
lnD = rows[:, 8]

# ==========================================================
# PLOT
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

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

# ==========================================================
# (a) S_pre vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    yy = Spre[maskT]

    order = np.argsort(pp)

    ax1.plot(
        pp[order],
        yy[order],
        '-o',
        ms=3,
        lw=1.1,
        label=rf'$T={Temp:.2f}$'
    )

ax1.set_xlabel(r'$P^*$')
ax1.set_ylabel(r'$S(k_{\mathrm{pre}})$')
ax1.set_title(r'(a) Pre-peak height')
ax1.grid(alpha=0.18)

# ==========================================================
# (b) k_pre vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    yy = kpre[maskT]

    order = np.argsort(pp)

    ax2.plot(
        pp[order],
        yy[order],
        '-o',
        ms=3,
        lw=1.1
    )

ax2.set_xlabel(r'$P^*$')
ax2.set_ylabel(r'$k_{\mathrm{pre}}$')
ax2.set_title(r'(b) Pre-peak position')
ax2.grid(alpha=0.18)

# ==========================================================
# (c) ratio vs P
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    pp = P[maskT]
    yy = ratio[maskT]

    order = np.argsort(pp)

    ax3.plot(
        pp[order],
        yy[order],
        '-o',
        ms=3,
        lw=1.1
    )

ax3.set_xlabel(r'$P^*$')
ax3.set_ylabel(r'$S(k_{\mathrm{pre}})/S(k_{\mathrm{main}})$')
ax3.set_title(r'(c) Pre/main peak ratio')
ax3.grid(alpha=0.18)

# ==========================================================
# (d) lnD vs ratio
# ==========================================================

for Temp in SELECTED_T:
    maskT = np.isclose(T, Temp, atol=1e-6)

    if np.sum(maskT) == 0:
        continue

    xx = ratio[maskT]
    yy = lnD[maskT]

    order = np.argsort(xx)

    ax4.plot(
        xx[order],
        yy[order],
        '-o',
        ms=3,
        lw=1.1
    )

ax4.set_xlabel(r'$S(k_{\mathrm{pre}})/S(k_{\mathrm{main}})$')
ax4.set_ylabel(r'$\ln D$')
ax4.set_title(r'(d) Diffusion vs pre/main ratio')
ax4.grid(alpha=0.18)

# ==========================================================
# (e) mapa ratio
# ==========================================================

tri = Triangulation(P, T)

c5 = ax5.tricontourf(
    tri,
    ratio,
    levels=30
)

fig.colorbar(
    c5,
    ax=ax5,
    label=r'$S(k_{\mathrm{pre}})/S(k_{\mathrm{main}})$'
)

ax5.set_xlabel(r'$P^*$')
ax5.set_ylabel(r'$T^*$')
ax5.set_title(r'(e) Pre/main ratio map')

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

ax1.legend(
    fontsize=7,
    ncol=2,
    frameon=False,
    loc='best'
)

# ==========================================================
# SALVAR
# ==========================================================

fig.tight_layout()

outfile_png = os.path.join(
    outdir,
    'Sk_multipanel.png'
)

outfile_pdf = os.path.join(
    outdir,
    'Sk_multipanel.pdf'
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
