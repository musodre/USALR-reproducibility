from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT
#!/usr/bin/env python3
# =========================================================
# CALCULA O PARÂMETRO DE ORDEM TRANSLACIONAL τ
#
# τ = ∫ |g(ξ)-1| dξ
#
# COM:
#
# ξ = r ρ^(1/3)
#
# =========================================================
#
# VERSÃO ROBUSTA:
#
# ✔ lê RDFs mistos
# ✔ ignora headers LAMMPS
# ✔ aceita:
#
#   r g(r)
#
# OU
#
#   index r g(r) coord
#
# ✔ usa densidades reais
#
# =========================================================

import numpy as np
import glob
import os
import re
import sys
from scipy.integrate import trapezoid
# =========================================================
# INPUTS
# =========================================================

rdf_pattern = str(RAW_DATA_ROOT / "allpress" / "P_*" / "rdfs" / "*.rdf")

thermo_pattern = str(RAW_DATA_ROOT / "allpress" / "P_*" / "thermo" / "outvars_P_*_T_*.profile")

# =========================================================
# OUTPUT
# =========================================================

outdir = str(DERIVED_DATA_ROOT / "tau")

os.makedirs(outdir, exist_ok=True)

# =========================================================
# PARAMETERS
# =========================================================

xi_cut = 5.0

rho_min = 1e-10

# =========================================================
# τ
# =========================================================

def compute_tau(r, g, rho):

    if rho <= rho_min:
        return np.nan

    mask = (

        np.isfinite(r)
        &
        np.isfinite(g)
        &
        (r > 0)

    )

    r = r[mask]
    g = g[mask]

    if len(r) < 10:
        return np.nan

    # -----------------------------------------------------
    # coordenada reduzida
    # -----------------------------------------------------

    xi = r * rho**(1.0/3.0)

    mask = xi <= xi_cut

    if np.sum(mask) < 10:
        return np.nan

    xi = xi[mask]
    g = g[mask]

    tau = trapezoid(np.abs(g - 1.0), xi)

    return tau

# =========================================================
# LEITOR RDF ROBUSTO
# =========================================================

def read_rdf_file(rdf_file):

    rows = []

    with open(rdf_file,'r') as f:

        for line in f:

            line = line.strip()

            # ---------------------------------------------
            # ignora comentários/vazios
            # ---------------------------------------------

            if len(line) == 0:
                continue

            if line.startswith('#'):
                continue

            cols = line.split()

            # ---------------------------------------------
            # tenta converter
            # ---------------------------------------------

            try:

                vals = [float(x) for x in cols]

            except:
                continue

            rows.append(vals)

    # -----------------------------------------------------
    # nenhum dado
    # -----------------------------------------------------

    if len(rows) == 0:
        return None, None

    # -----------------------------------------------------
    # detecta formato dominante
    # -----------------------------------------------------

    ncols = max(len(r) for r in rows)

    rvals = []
    gvals = []

    for vals in rows:

        # ---------------------------------------------
        # FORMATO:
        # r g(r)
        # ---------------------------------------------

        if len(vals) == 2:

            rvals.append(vals[0])
            gvals.append(vals[1])

        # ---------------------------------------------
        # FORMATO:
        # index r g(r) ...
        # ---------------------------------------------

        elif len(vals) >= 3:

            rvals.append(vals[1])
            gvals.append(vals[2])

    if len(rvals) == 0:
        return None, None

    r = np.array(rvals)
    g = np.array(gvals)

    return r, g


def get_PT_from_filename(path):
    fname = os.path.basename(path)
    tmp = fname.replace(".rdf", "")
    parts = tmp.split("_")
    return float(parts[1]), float(parts[3])


def canonicalize_raw_state_files(files):
    """
    Resolve duplicate raw-state directories that map to the same physical (P,T).

    Current provenance rule:
      - if both P_0.40 and P_0.400 exist, prefer P_0.400
      - preserve legitimate additional states such as P*=0.1, T*=0.01
      - abort on any other unresolved duplicate state
    """
    chosen = {}
    excluded = []

    for path in sorted(files):
        P, T = get_PT_from_filename(path)
        key = (round(P, 10), round(T, 10))

        parent_pressure_dir = Path(path).parents[1].name

        if key not in chosen:
            chosen[key] = path
            continue

        old = chosen[key]
        old_dir = Path(old).parents[1].name

        pair = {old_dir, parent_pressure_dir}

        if pair == {"P_0.40", "P_0.400"}:
            if parent_pressure_dir == "P_0.400":
                excluded.append(old)
                chosen[key] = path
            else:
                excluded.append(path)
            continue

        raise RuntimeError(
            "Unexpected duplicate physical state "
            f"(P={P:.10g}, T={T:.10g}) from:\n"
            f"  {old}\n"
            f"  {path}\n"
            "No automatic resolution rule is defined."
        )

    return [chosen[k] for k in sorted(chosen)], excluded

# =========================================================
# RDF FILES + CANONICAL STATE SELECTION
# =========================================================

raw_rdf_files = sorted(
    glob.glob(rdf_pattern)
)

rdf_files, excluded_rdfs = canonicalize_raw_state_files(
    raw_rdf_files
)

print("\n===================================")
print(f"{len(raw_rdf_files)} RDFs brutos encontrados")
print(f"{len(rdf_files)} estados canônicos selecionados")
print(f"{len(excluded_rdfs)} RDFs redundantes excluídos")
print("===================================\n")

if len(rdf_files) == 0:
    print("\nNenhum RDF encontrado.\n")
    sys.exit()

# =========================================================
# CARREGA DENSIDADES CANÔNICAS
# =========================================================

print("\n===================================")
print("CARREGANDO DENSIDADES CANÔNICAS")
print("===================================\n")

rho_dict = {}

for rdf_file in rdf_files:

    P, T = get_PT_from_filename(rdf_file)
    key = (round(P, 10), round(T, 10))

    pressure_dir = Path(rdf_file).parents[1]
    thermo_dir = pressure_dir / "thermo"

    candidates = sorted(
        thermo_dir.glob(
            f"outvars_P_*_T_{T:.2f}.profile"
        )
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one thermo file for {rdf_file}; "
            f"found {len(candidates)}"
        )

    thermo = np.loadtxt(candidates[0])
    thermo = np.atleast_2d(thermo)

    rho = float(np.mean(thermo[:, 4]))

    if not np.isfinite(rho):
        raise RuntimeError(
            f"Invalid density for {rdf_file}"
        )

    rho_dict[key] = rho

print(f"{len(rho_dict)} densidades carregadas.\n")

# =========================================================
# GLOBAL
# =========================================================

global_data = []

# =========================================================
# LOOP RDFs
# =========================================================

for rdf_file in rdf_files:

    try:

        P, T = get_PT_from_filename(rdf_file)
        key = (round(P, 10), round(T, 10))

        # =================================================
        # densidade
        # =================================================

        if key not in rho_dict:

            print(
                f"\nDensidade ausente:"
                f" P={P:.3f} T={T:.2f}"
            )

            continue

        rho = rho_dict[key]

        # =================================================
        # RDF
        # =================================================

        r, g = read_rdf_file(rdf_file)

        if r is None:

            print(f"\nRDF inválido:")
            print(rdf_file)

            continue

        # =================================================
        # τ
        # =================================================

        tau = compute_tau(

            r,
            g,
            rho

        )

        if not np.isfinite(tau):
            continue

        # =================================================
        # salva
        # =================================================

        global_data.append([

            P,
            T,
            rho,
            tau

        ])

        print(

            f"P={P:.3f} "
            f"T={T:.2f} "
            f"rho={rho:.4f} "
            f"tau={tau:.6f}"

        )

    except Exception as e:

        print(f"\nErro RDF:")
        print(rdf_file)
        print(e)

# =========================================================
# CHECA
# =========================================================

if len(global_data) == 0:

    print("\nNenhum tau válido calculado.\n")
    sys.exit()

# =========================================================
# ARRAY
# =========================================================

global_data = np.array(global_data)

global_data = np.atleast_2d(global_data)

mask = np.isfinite(global_data[:,3])

global_data = global_data[mask]

if len(global_data) == 0:

    print("\nTodos os taus são NaN.\n")
    sys.exit()

# =========================================================
# ORDENA
# =========================================================

idx = np.lexsort(

    (

        global_data[:,1],
        global_data[:,0]

    )

)

global_data = global_data[idx]

# =========================================================
# SAVE GLOBAL
# =========================================================

outfile_global = (
    f"{outdir}/tau_global.dat"
)

np.savetxt(

    outfile_global,

    global_data,

    fmt='%.8e',

    header='P T rho tau'

)

print("\n===================================")
print(f"{outfile_global} salvo")
print("===================================\n")

# =========================================================
# ISOTERMAS
# =========================================================

temperatures = np.unique(

    np.round(global_data[:,1],2)

)

for T in temperatures:

    mask = np.abs(

        global_data[:,1] - T

    ) < 1e-5

    block = global_data[mask]

    block = block[np.argsort(block[:,0])]

    outfile = (

        f"{outdir}/"
        f"tauP_T_{T:.2f}.dat"

    )

    np.savetxt(

        outfile,

        block,

        fmt='%.8e',

        header='P T rho tau'

    )

# =========================================================
# ISÓBARAS
# =========================================================

pressures = np.unique(

    np.round(global_data[:,0],3)

)

for P in pressures:

    mask = np.abs(

        global_data[:,0] - P

    ) < 1e-5

    block = global_data[mask]

    block = block[np.argsort(block[:,1])]

    outfile = (

        f"{outdir}/"
        f"tauT_P_{P:.3f}.dat"

    )

    np.savetxt(

        outfile,

        block,

        fmt='%.8e',

        header='P T rho tau'

    )

# =========================================================
# FINAL
# =========================================================

print("\n===================================")
print("ANÁLISE τ FINALIZADA")
print(f"Estados canônicos gravados: {len(global_data)}")
print(f"RDFs redundantes excluídos: {len(excluded_rdfs)}")
print("===================================\n")
