#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

REF = Path("../python-inputs/somepress/analysis/tau/tau_global.dat")
NEW = Path("data/derived/tau/tau_global.dat")

def read(path):
    return pd.read_csv(
        path, sep=r"\s+", comment="#",
        names=["P","T","rho","tau"], header=None
    )

ref = read(REF)
new = read(NEW)

for df in (ref, new):
    df["Pkey"] = df["P"].round(10)
    df["Tkey"] = df["T"].round(10)

dup = (
    ref.groupby(["Pkey","Tkey"])
       .size()
       .reset_index(name="multiplicity")
)
dup = dup[dup["multiplicity"] > 1].copy()

print("="*80)
print("REFERENCE DUPLICATES")
print("="*80)
print("duplicate (P,T) states =", len(dup))
print("extra rows             =", int((dup["multiplicity"]-1).sum()))
if len(dup):
    print("P values               =", sorted(dup["Pkey"].unique()))
    print("T range                =", dup["Tkey"].min(), "--", dup["Tkey"].max())

rows = []
for _, d in dup.iterrows():
    P, T = d["Pkey"], d["Tkey"]
    rr = ref[(ref["Pkey"]==P) & (ref["Tkey"]==T)]
    nn = new[(new["Pkey"]==P) & (new["Tkey"]==T)]
    if len(nn) == 0:
        continue
    n = nn.iloc[-1]
    rows.append({
        "P":P, "T":T,
        "rho_new":n["rho"], "tau_new":n["tau"],
        "rho_ref_first":rr.iloc[0]["rho"],
        "rho_ref_last":rr.iloc[-1]["rho"],
        "rho_ref_mean":rr["rho"].mean(),
        "tau_ref_first":rr.iloc[0]["tau"],
        "tau_ref_last":rr.iloc[-1]["tau"],
        "tau_ref_mean":rr["tau"].mean(),
    })

a = pd.DataFrame(rows)

if len(a):
    for mode in ("first","last","mean"):
        dr = np.abs(a["rho_new"] - a[f"rho_ref_{mode}"])
        dt = np.abs(a["tau_new"] - a[f"tau_ref_{mode}"])
        print()
        print(f"NEW vs REF {mode.upper()}")
        print("max |delta rho| =", dr.max())
        print("max |delta tau| =", dt.max())
        print("mean|delta tau| =", dt.mean())

    print()
    print("First 15 duplicated states:")
    print(a.head(15).to_string(index=False))
