#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_P040_branches.py

Identify how the two complete P*=0.4 datasets

    data/raw/allpress/P_0.40/
    data/raw/allpress/P_0.400/

map onto the duplicated historical s2_global.dat and tau_global.dat rows.

Run from the USALR_reproducibility repository root.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid


ROOT = Path("data/raw/allpress")
OLD_S2 = Path("../python-inputs/somepress/analysis/s2/s2_global.dat")
OLD_TAU = Path("../python-inputs/somepress/analysis/tau/tau_global.dat")

BRANCHES = {
    "P_0.40": ROOT / "P_0.40",
    "P_0.400": ROOT / "P_0.400",
}


def load_rdf(path):
    rows = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(x) for x in s.split()]
            except Exception:
                continue
            if len(vals) == 2:
                rows.append((vals[0], vals[1]))
            elif len(vals) >= 3:
                rows.append((vals[1], vals[2]))

    if not rows:
        raise RuntimeError(f"Unreadable RDF: {path}")

    a = np.asarray(rows, dtype=float)
    return a[:, 0], a[:, 1]


def density_from_profile(path):
    a = np.loadtxt(path)
    a = np.atleast_2d(a)
    return float(np.mean(a[:, 4]))


def calc_s2(r, g, rho):
    gsafe = np.array(g, copy=True)
    gsafe[gsafe <= 1e-12] = 1e-12
    term = gsafe * np.log(gsafe) - gsafe + 1.0
    integrand = -2.0 * np.pi * rho * term * r**2
    cs2 = cumulative_trapezoid(integrand, r, initial=0.0)
    return float(cs2[-1])


def calc_tau(r, g, rho, xi_cut=5.0):
    mask = np.isfinite(r) & np.isfinite(g) & (r > 0)
    r = r[mask]
    g = g[mask]
    xi = r * rho**(1.0 / 3.0)
    m = xi <= xi_cut
    return float(trapezoid(np.abs(g[m] - 1.0), xi[m]))


def read_old(path, value):
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["P", "T", "rho", value],
        header=None,
    )


old_s2 = read_old(OLD_S2, "s2")
old_tau = read_old(OLD_TAU, "tau")

old_s2 = old_s2[np.isclose(old_s2["P"], 0.4)].copy()
old_tau = old_tau[np.isclose(old_tau["P"], 0.4)].copy()

records = []

for branch_name, d in BRANCHES.items():

    for rdf in sorted((d / "rdfs").glob("*.rdf")):
        m = re.search(r"_T_([0-9.]+)\.rdf$", rdf.name)
        if not m:
            continue

        T = float(m.group(1))

        thermo_candidates = sorted(
            (d / "thermo").glob(f"outvars_P_*_T_{T:.2f}.profile")
        )

        if len(thermo_candidates) != 1:
            raise RuntimeError(
                f"{branch_name} T={T:.2f}: "
                f"expected one thermo file, found {len(thermo_candidates)}"
            )

        rho = density_from_profile(thermo_candidates[0])
        r, g = load_rdf(rdf)

        records.append({
            "branch": branch_name,
            "T": T,
            "rho": rho,
            "s2": calc_s2(r, g, rho),
            "tau": calc_tau(r, g, rho),
        })

new = pd.DataFrame(records)

print("=" * 92)
print("P*=0.4 BRANCH PROVENANCE")
print("=" * 92)
print(new.groupby("branch").size())
print()

for obs, old in [("s2", old_s2), ("tau", old_tau)]:

    print("=" * 92)
    print(obs.upper())
    print("=" * 92)

    # Historical first / last occurrence at each T.
    hist = (
        old.groupby("T", sort=True)
           .agg(
               rho_first=("rho", "first"),
               rho_last=("rho", "last"),
               **{
                   f"{obs}_first": (obs, "first"),
                   f"{obs}_last": (obs, "last"),
                   f"{obs}_mean": (obs, "mean"),
               },
           )
           .reset_index()
    )

    for branch in BRANCHES:
        b = new[new["branch"] == branch][["T", "rho", obs]]
        m = hist.merge(b, on="T", how="inner")

        print(f"\nBranch {branch}")

        for target in ("first", "last", "mean"):
            if target == "mean":
                rho_ref = (
                    old.groupby("T")["rho"].mean()
                    .reindex(m["T"]).to_numpy()
                )
            else:
                rho_ref = m[f"rho_{target}"].to_numpy()

            val_ref = m[f"{obs}_{target}"].to_numpy()

            drho = np.abs(m["rho"].to_numpy() - rho_ref)
            dval = np.abs(m[obs].to_numpy() - val_ref)

            print(
                f"  vs HIST {target.upper():5s}: "
                f"max|drho|={drho.max():.12e}   "
                f"max|d{obs}|={dval.max():.12e}   "
                f"mean|d{obs}|={dval.mean():.12e}"
            )

    print()

# Detailed winner table.
summary = []

for obs, old in [("s2", old_s2), ("tau", old_tau)]:
    for T, grp in old.groupby("T"):
        if len(grp) < 2:
            continue

        first = grp.iloc[0]
        last = grp.iloc[-1]

        for branch in BRANCHES:
            row = new[
                (new["branch"] == branch)
                & np.isclose(new["T"], T)
            ].iloc[0]

            summary.append({
                "obs": obs,
                "T": T,
                "branch": branch,
                "d_first": abs(row[obs] - first[obs]),
                "d_last": abs(row[obs] - last[obs]),
            })

s = pd.DataFrame(summary)

print("=" * 92)
print("WINNER COUNTS")
print("=" * 92)

for obs in ("s2", "tau"):
    x = s[s["obs"] == obs].copy()
    x["winner"] = np.where(x["d_first"] < x["d_last"], "FIRST", "LAST")
    print(f"\n{obs}:")
    print(
        x.groupby(["branch", "winner"])
         .size()
         .to_string()
    )

print("\nDone.")
