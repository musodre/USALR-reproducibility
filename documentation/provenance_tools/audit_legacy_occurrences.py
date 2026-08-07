#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_legacy_occurrences.py

Compare historical and regenerated aggregate tables occurrence-by-occurrence
without recalculating s2 or tau.

Run from the root of USALR_reproducibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd

FILES = {
    "s2": (
        Path("data/derived/s2/s2_global.dat"),
        Path("../python-inputs/somepress/analysis/s2/s2_global.dat"),
        ["P", "T", "rho", "s2"],
    ),
    "tau": (
        Path("data/derived/tau/tau_global.dat"),
        Path("../python-inputs/somepress/analysis/tau/tau_global.dat"),
        ["P", "T", "rho", "tau"],
    ),
}

PRESSURES = [0.1, 0.2, 0.4]


def read(path, cols):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=cols,
        header=None,
    )
    df["P"] = df["P"].round(10)
    df["T"] = df["T"].round(10)
    return df


for obs, (newfile, oldfile, cols) in FILES.items():

    new = read(newfile, cols)
    old = read(oldfile, cols)

    print()
    print("=" * 96)
    print(obs.upper())
    print("=" * 96)

    for P in PRESSURES:

        nP = new[np.isclose(new["P"], P)].copy()
        oP = old[np.isclose(old["P"], P)].copy()

        print()
        print(f"P* = {P}")
        print(
            f"new rows={len(nP)}  old rows={len(oP)}  "
            f"new unique T={nP['T'].nunique()}  old unique T={oP['T'].nunique()}"
        )

        common_T = sorted(
            set(nP["T"].unique()).intersection(oP["T"].unique())
        )

        stats = []

        for T in common_T:
            nn = nP[np.isclose(nP["T"], T)].reset_index(drop=True)
            oo = oP[np.isclose(oP["T"], T)].reset_index(drop=True)

            for i in range(len(nn)):
                for j in range(len(oo)):
                    stats.append({
                        "T": T,
                        "new_occ": i + 1,
                        "old_occ": j + 1,
                        "drho": abs(float(nn.loc[i, "rho"]) - float(oo.loc[j, "rho"])),
                        "dval": abs(float(nn.loc[i, obs]) - float(oo.loc[j, obs])),
                    })

        s = pd.DataFrame(stats)

        if len(s) == 0:
            print("No comparable rows.")
            continue

        summary = (
            s.groupby(["new_occ", "old_occ"])
             .agg(
                 N=("T", "size"),
                 max_drho=("drho", "max"),
                 mean_drho=("drho", "mean"),
                 max_dval=("dval", "max"),
                 mean_dval=("dval", "mean"),
             )
             .reset_index()
             .sort_values(["mean_dval", "max_dval"])
        )

        print()
        print("Occurrence mapping:")
        print(summary.to_string(index=False))

        best = summary.iloc[0]

        print()
        print(
            "BEST: "
            f"new occurrence {int(best['new_occ'])} "
            f"<-> old occurrence {int(best['old_occ'])} | "
            f"mean |delta {obs}| = {best['mean_dval']:.12e}, "
            f"max = {best['max_dval']:.12e}"
        )

print()
print("Done.")
