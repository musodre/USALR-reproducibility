import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 35
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{mathptmx}
\usepackage[T1]{fontenc}
'''

plt.rcParams['axes.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25


from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import REPO_ROOT, FIGURE_OUTPUT_ROOT

# =======================
# Carregar dados
# =======================
dados2  = np.loadtxt("pot_Uyes04pol.dat", skiprows=3, usecols=(1, 2))
forcas2 = np.loadtxt("pot_Uyes04pol.dat", skiprows=3, usecols=(1, 3))
x2, u2 = dados2[:, 0], dados2[:, 1]
_, f2 = forcas2[:, 0], forcas2[:, 1]


# ======================================================
#               Figure of U_SALR
# ======================================================
fig2, ax2 = plt.subplots(figsize=(7, 6), dpi = 150)

ax2.plot(x2, u2, color='#99004c', lw=3)
ax2.set_xlabel(r"$r_{ij}^{\ast}$")
ax2.set_ylabel(r"$U_{SALR}(r_{ij}^{\ast})$")
ax2.set_xlim(0, 5)
ax2.set_ylim(-2, 4)

# Inset — Força SALR
inset2 = inset_axes(ax2, width="40%", height="40%", loc="upper right")
inset2.plot(x2, f2, color='#00994c', lw=2)
inset2.set_xlim(0, 3.5)
inset2.set_ylim(-10, 7)
inset2.set_xlabel(r"$r_{ij}^{\ast}$", fontsize=12)
inset2.set_ylabel(r"$F_{SALR}(r_{ij}^{\ast})$", fontsize=12)
inset2.tick_params(labelsize=10)

plt.savefig("fig2_USALR.pdf", bbox_inches="tight")
plt.savefig("fig2_USALR.png", dpi=600, bbox_inches="tight")

plt.show()
