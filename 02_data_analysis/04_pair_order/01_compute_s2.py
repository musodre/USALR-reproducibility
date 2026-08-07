from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
import glob
import os

# =========================================================
# INPUT
# =========================================================

rdf_pattern = str(RAW_DATA_ROOT / "allpress" / "P_*" / "rdfs" / "*.rdf")

thermo_pattern = str(RAW_DATA_ROOT / "allpress" / "P_{PVAL}" / "thermo" / "outvars_P_{PSTR}_T_{TSTR}.profile")

# =========================================================
# OUTPUT
# =========================================================

outdir_s2 = str(DERIVED_DATA_ROOT / "s2")
outdir_cs2 = str(DERIVED_DATA_ROOT / "Cs2")

os.makedirs(outdir_s2, exist_ok=True)
os.makedirs(outdir_cs2, exist_ok=True)

# Caminhos dos arquivos globais
global_s2_file = f"{outdir_s2}/s2_global.dat"
global_cs2_file = f"{outdir_cs2}/Cs2_global.dat"

# =========================================================
# INITIALIZE HEADERS
# =========================================================

with open(global_s2_file, 'w') as f:
    f.write("# P T rho s2\n")

with open(global_cs2_file, 'w') as f:
    f.write("# P T rho s2_final Cs2_r1 Cs2_r2 rmin1 rmin2 Nshells\n")

# Dicionários para agrupar isotermas e isóbaras na memória
isobaras = {}  # Chave: P_fixa -> Lista de [T, rho, s2]
isotermas = {} # Chave: T_fixa -> Lista de [P, rho, s2]

# =========================================================
# FUNCTIONS
# =========================================================

def load_rdf(path):
    data = np.loadtxt(path, comments='#', skiprows=4)
    # col1 -> r, col2 -> g(r)
    r = data[:, 1]
    g = data[:, 2]
    return r, g

def get_PT_from_filename(path):
    fname = os.path.basename(path)
    tmp = fname.replace(".rdf", "")
    parts = tmp.split("_")
    P = float(parts[1])
    T = float(parts[3])
    return P, T


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

def build_density_cache(rdf_files):
    print("\n===================================")
    print("Construindo cache de densidades")
    print("===================================\n")

    cache = {}

    for rdf_file in rdf_files:
        try:
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

            cache[key] = rho

        except Exception as exc:
            raise RuntimeError(
                f"Failed to build density cache for {rdf_file}: {exc}"
            ) from exc

    print(f"{len(cache)} densidades carregadas.\n")
    return cache

def integrand_s2(r, g, rho):
    g_safe = np.copy(g)
    g_safe[g_safe <= 1e-12] = 1e-12
    term = (g_safe * np.log(g_safe) - g_safe + 1.0)
    integrand = (-2.0 * np.pi * rho * term * r**2)
    return integrand

def cumulative_s2(r, integrand):
    Cs2 = cumulative_trapezoid(integrand, r, initial=0)
    return Cs2

def find_shell_information(r, g):
    """
    Algoritmo robusto adaptado para sistemas de nanopartículas funcionalizadas.
    Evita NaNs relaxando critérios de proeminência e buscando mínimos locais diretamente.
    """
    # Encontra picos sem exigência estrita de proeminência inicial
    peaks, _ = find_peaks(g, distance=5)
    Nshells = len(peaks)

    # Fallback: Se g(r) for muito amortecida e find_peaks falhar, pega o máximo global
    if Nshells == 0:
        p1 = np.argmax(g)
        peaks = [p1]
        Nshells = 1

    p1 = peaks[0]

    # --- Determinação do rmin1 ---
    if len(peaks) >= 2:
        p2 = peaks[1]
        g_between_1 = g[p1:p2]
        rmin1 = r[p1 + np.argmin(g_between_1)]
    else:
        # Se não houver segundo pico, busca o primeiro mínimo local após o primeiro pico
        idx_search = g[p1:]
        if len(idx_search) > 1:
            # Primeiro ponto onde a g(r) volta a subir ou estabiliza
            diffs = np.diff(idx_search)
            pos_subida = np.where(diffs > 0)[0]
            if len(pos_subida) > 0:
                min1 = p1 + pos_subida[0]
            else:
                min1 = p1 + np.argmin(idx_search)
            rmin1 = r[min1]
        else:
            rmin1 = np.nan

    # --- Determinação do rmin2 ---
    if len(peaks) >= 3:
        p3 = peaks[2]
        g_between_2 = g[peaks[1]:p3]
        rmin2 = r[peaks[1] + np.argmin(g_between_2)]
    elif len(peaks) == 2:
        # Busca o mínimo após o segundo pico
        p2 = peaks[1]
        idx_search2 = g[p2:]
        diffs2 = np.diff(idx_search2)
        pos_subida2 = np.where(diffs2 > 0)[0]
        if len(pos_subida2) > 0:
            min2 = p2 + pos_subida2[0]
            rmin2 = r[min2]
        else:
            rmin2 = np.nan
    else:
        rmin2 = np.nan

    return rmin1, rmin2, Nshells

def cumulative_value_at_r(r, Cs2, rval):
    if rval is None or not np.isfinite(rval):
        return np.nan
    idx = np.argmin(np.abs(r - rval))
    return Cs2[idx]

# =========================================================
# RUN ANALYSIS
# =========================================================

raw_rdf_files = sorted(glob.glob(rdf_pattern))
rdf_files, excluded_rdfs = canonicalize_raw_state_files(raw_rdf_files)
rho_cache = build_density_cache(rdf_files)

print("===================================")
print(f"{len(raw_rdf_files)} RDFs brutos encontrados")
print(f"{len(rdf_files)} estados canônicos selecionados")
print(f"{len(excluded_rdfs)} RDFs redundantes excluídos")
print("===================================\n")

for i, rdf_file in enumerate(rdf_files):
    if (i + 1) % 25 == 0 or (i + 1) == len(rdf_files):
        print(f"Processando: [{i + 1}/{len(rdf_files)}]")

    try:
        P, T = get_PT_from_filename(rdf_file)
        key = (P, T)

        if key not in rho_cache:
            continue

        rho = rho_cache[key]
        r, g = load_rdf(rdf_file)
        integrand = integrand_s2(r, g, rho)
        Cs2 = cumulative_s2(r, integrand)
        s2_final = Cs2[-1]

        # Processamento robusto das camadas
        rmin1, rmin2, Nshells = find_shell_information(r, g)
        Cs2_r1 = cumulative_value_at_r(r, Cs2, rmin1)
        Cs2_r2 = cumulative_value_at_r(r, Cs2, rmin2)

        # -------------------------------------------------
        # Salva nos arquivos Globais (Append)
        # -------------------------------------------------
        with open(global_s2_file, 'a') as f:
            f.write(f"{P:.3f} {T:.2f} {rho:.8f} {s2_final:.8f}\n")

        with open(global_cs2_file, 'a') as f:
            f.write(
                f"{P:.3f} {T:.2f} {rho:.8f} {s2_final:.8f} "
                f"{Cs2_r1:.8f} {Cs2_r2:.8f} "
                f"{rmin1:.8f} {rmin2:.8f} {Nshells:d}\n"
            )

        # -------------------------------------------------
        # Coleta os dados para Isóbaras e Isotermas
        # -------------------------------------------------
        # Agrupa por Pressão (Isóbaras: fixa P, varia T)
        P_key = f"{P:.3f}"
        if P_key not in isobaras:
            isobaras[P_key] = []
        isobaras[P_key].append([T, rho, s2_final])

        # Agrupa por Temperatura (Isotermas: fixa T, varia P)
        T_key = f"{T:.2f}"
        if T_key not in isotermas:
            isotermas[T_key] = []
        isotermas[T_key].append([P, rho, s2_final])

    except Exception as e:
        print(f"\nErro no arquivo: {rdf_file}")
        print(f"Detalhes do erro: {e}\n")

# =========================================================
# GERANDO ARQUIVOS DE ISÓBARAS E ISOTERMAS
# =========================================================
print("\n===================================")
print("Gravando arquivos de Isóbaras e Isotermas")
print("===================================\n")

# Escrevendo Isóbaras (sexT_P_X.XXX.dat) -> Mantém P fixo, varia T
for P_str, dados in isobaras.items():
    # Ordena os dados pelo valor de Temperatura (dados[0])
    dados_ordenados = sorted(dados, key=lambda x: x[0])

    filename_isobara = f"{outdir_s2}/sexT_P_{P_str}.dat"
    with open(filename_isobara, 'w') as f:
        f.write(f"# Isóbara sob Pressão P = {P_str}\n")
        f.write("# T rho s2\n")
        for item in dados_ordenados:
            f.write(f"{item[0]:.2f} {item[1]:.8f} {item[2]:.8f}\n")

# Escrevendo Isotermas (sexP_T_X.XX.dat) -> Mantém T fixo, varia P
for T_str, dados in isotermas.items():
    # Ordena os dados pelo valor de Pressão (dados[0])
    dados_ordenados = sorted(dados, key=lambda x: x[0])

    filename_isotermal = f"{outdir_s2}/sexP_T_{T_str}.dat"
    with open(filename_isotermal, 'w') as f:
        f.write(f"# Isoterma sob Temperatura T = {T_str}\n")
        f.write("# P rho s2\n")
        for item in dados_ordenados:
            f.write(f"{item[0]:.3f} {item[1]:.8f} {item[2]:.8f}\n")

print("===================================")
print("Etapa 3B CONCLUÍDA COM SUCESSO")
print("===================================\n")
print(f"Globais: {global_s2_file} e {global_cs2_file} atualizados.")
print(f"Isóbaras (sexT_P_X.XXX.dat) salvas em '{outdir_s2}'.")
print(f"Isotermas (sexP_T_X.XX.dat) salvas em '{outdir_s2}'.")
print(f"Estados canônicos gravados: {len(rdf_files)}")
print(f"RDFs redundantes excluídos: {len(excluded_rdfs)}\n")
