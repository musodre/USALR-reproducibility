#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_analysis.py

Static cluster analysis for the USALR system.

Run from:
    USALR/python-inputs

Trajectories:
    ../allpress/trajs/P_XXX_T_YY.lammpstrj

Cluster criterion:
    r_ij* <= R_CLUSTER

The published cluster analysis uses the fixed operational
connectivity cutoff

    R_CLUSTER = 1.20

with a sensitivity check over nearby cutoff values (1.10--1.25).
An earlier exploratory RDF-first-minimum analysis suggested a substantially
larger value (~1.59), but that criterion is not the one used for the published
cluster results and is therefore not part of the main reproduction workflow.

Main outputs:
    plots/cluster_analysis/cluster_state_summary.dat
    plots/cluster_analysis/cluster_state_summary.csv
    plots/cluster_analysis/distributions/*.dat
    plots/cluster_analysis/cluster_mean_vs_pressure.pdf
    plots/cluster_analysis/cluster_weighted_vs_pressure.pdf
    plots/cluster_analysis/largest_cluster_fraction.pdf
    plots/cluster_analysis/clustered_fraction.pdf
    plots/cluster_analysis/cluster_vs_diffusion.pdf
"""

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree



from pathlib import Path
import sys

for _repo_parent in Path(__file__).resolve().parents:
    if (_repo_parent / "usalr_paths.py").exists():
        sys.path.insert(0, str(_repo_parent)) if str(_repo_parent) not in sys.path else None
        break
else:
    raise RuntimeError("Could not locate USALR repository root")

from usalr_paths import RAW_DATA_ROOT, DERIVED_DATA_ROOT

# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# Fixed cluster connectivity cutoff
# -------------------------------------------------------------------------

R_CLUSTER = 1.20


# -------------------------------------------------------------------------
# Thermodynamic domain
# -------------------------------------------------------------------------

T_MIN = 0.20
T_MAX = 0.60

P_MIN = 0.10
P_MAX = 6.00


# -------------------------------------------------------------------------
# Trajectories
# -------------------------------------------------------------------------

TRAJECTORY_ROOT = RAW_DATA_ROOT / "allpress" / "trajs"

TRAJECTORY_GLOB = "*.lammpstrj"


# -------------------------------------------------------------------------
# Frame sampling
# -------------------------------------------------------------------------

# Fraction of the beginning of each trajectory to discard.

DISCARD_FRACTION = 0.20

# Analyze one frame every FRAME_STRIDE production frames.

FRAME_STRIDE = 5

# None = no explicit upper limit after stride selection.

MAX_FRAMES_PER_STATE = None


# -------------------------------------------------------------------------
# Particle type
# -------------------------------------------------------------------------
#
# None = use all particles in the dump.
#
# IMPORTANT:
# If the trajectory contains additional particle species that should not
# enter the SALR cluster analysis, set the appropriate LAMMPS atom type.

PARTICLE_TYPE = None


# -------------------------------------------------------------------------
# Large-cluster diagnostic
# -------------------------------------------------------------------------
#
# This is NOT a rigorous wrapping/percolation criterion.
# It only identifies states where the largest cluster contains a
# macroscopic fraction of the particles.

LARGEST_CLUSTER_THRESHOLD = 0.50


# -------------------------------------------------------------------------
# Isotherms highlighted in the figures
# -------------------------------------------------------------------------

T_SELECTED = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
]


# =============================================================================
# 2. DIFFUSION FILE
# =============================================================================

D_FILE = DERIVED_DATA_ROOT / "asymptotic_diffusion" / "D_asymptotic_global.dat"


# =============================================================================
# 3. OUTPUT
# =============================================================================

OUTDIR = DERIVED_DATA_ROOT / "cluster_analysis"

DISTDIR = OUTDIR / "distributions"

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

DISTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 4. MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 11,

    "axes.labelsize": 15,

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "legend.fontsize": 9.5,

    "axes.linewidth": 1.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,

    "legend.frameon": False,
})


# =============================================================================
# 5. COLOR PALETTE
# =============================================================================

COLORS = {

    0.20: "#3B5B92",
    0.30: "#397A6D",
    0.40: "#B38A3E",
    0.50: "#B85C5C",
    0.60: "#76558F",

}


# =============================================================================
# 6. VALIDATION
# =============================================================================

if R_CLUSTER <= 0:

    raise ValueError(
        "R_CLUSTER must be positive."
    )


if not TRAJECTORY_ROOT.exists():

    raise FileNotFoundError(
        "\nTrajectory directory not found:\n"
        f"{TRAJECTORY_ROOT.resolve()}\n"
    )


# =============================================================================
# 7. PARSE P,T
# =============================================================================

def parse_PT(path):

    name = path.name

    match = re.search(
        r"P_([0-9]+(?:\.[0-9]+)?)_T_([0-9]+(?:\.[0-9]+)?)",
        name
    )

    if match is None:

        raise ValueError(
            f"Could not extract P,T from:\n{path}"
        )

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


# =============================================================================
# 8. UNION-FIND
# =============================================================================

class UnionFind:

    def __init__(self, n):

        self.parent = np.arange(
            n,
            dtype=int
        )

        self.size = np.ones(
            n,
            dtype=int
        )


    def find(self, x):

        root = x

        while self.parent[root] != root:

            root = self.parent[root]


        while self.parent[x] != x:

            nxt = self.parent[x]

            self.parent[x] = root

            x = nxt


        return root


    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)


        if ra == rb:
            return


        if self.size[ra] < self.size[rb]:

            ra, rb = rb, ra


        self.parent[rb] = ra

        self.size[ra] += self.size[rb]


    def cluster_sizes(self):

        roots = np.asarray(
            [
                self.find(i)
                for i in range(
                    len(self.parent)
                )
            ],
            dtype=int
        )


        _, counts = np.unique(
            roots,
            return_counts=True
        )


        return counts.astype(
            int
        )


# =============================================================================
# 9. COUNT FRAMES WITHOUT LOADING TRAJECTORY
# =============================================================================

def count_frames(path):

    n = 0

    with open(
        path,
        "r"
    ) as fh:

        for line in fh:

            if line.startswith(
                "ITEM: TIMESTEP"
            ):

                n += 1

    return n


# =============================================================================
# 10. STREAM LAMMPS TRAJECTORY
# =============================================================================

def iter_lammpstrj(path):

    """
    Stream a LAMMPS dump trajectory.

    Supported coordinate sets:
        x y z
        xu yu zu
        xs ys zs

    Orthorhombic periodic simulation boxes are assumed.

    Returns:
        frame_index
        timestep
        box_lengths
        positions
        atom_types
    """

    frame_index = 0


    with open(
        path,
        "r"
    ) as fh:

        while True:

            line = fh.readline()


            if not line:
                break


            if not line.startswith(
                "ITEM: TIMESTEP"
            ):
                continue


            timestep_line = fh.readline()

            if not timestep_line:
                break


            timestep = int(
                timestep_line.strip()
            )


            # -------------------------------------------------------------
            # Number of atoms
            # -------------------------------------------------------------

            line = fh.readline()


            if not line.startswith(
                "ITEM: NUMBER OF ATOMS"
            ):

                raise RuntimeError(
                    f"Unexpected trajectory format:\n{path}"
                )


            natoms = int(
                fh.readline().strip()
            )


            # -------------------------------------------------------------
            # Box
            # -------------------------------------------------------------

            box_header = fh.readline()


            if not box_header.startswith(
                "ITEM: BOX BOUNDS"
            ):

                raise RuntimeError(
                    f"BOX BOUNDS not found:\n{path}"
                )


            # Triclinic boxes are deliberately rejected here.
            #
            # For the present simulations we expect orthorhombic boxes.

            if (
                "xy" in box_header
                or
                "xz" in box_header
                or
                "yz" in box_header
            ):

                raise RuntimeError(
                    "\nTriclinic box detected.\n"
                    "The present cluster script assumes an "
                    "orthorhombic box.\n"
                    f"File: {path}\n"
                )


            bounds = []


            for _ in range(3):

                vals = fh.readline().split()


                if len(vals) < 2:

                    raise RuntimeError(
                        f"Invalid BOX BOUNDS in:\n{path}"
                    )


                lo = float(
                    vals[0]
                )

                hi = float(
                    vals[1]
                )


                bounds.append(
                    [lo, hi]
                )


            bounds = np.asarray(
                bounds,
                dtype=float
            )


            lo = bounds[:, 0]

            L = (
                bounds[:, 1]
                -
                bounds[:, 0]
            )


            # -------------------------------------------------------------
            # Atom columns
            # -------------------------------------------------------------

            atom_header = (
                fh.readline()
                .strip()
            )


            if not atom_header.startswith(
                "ITEM: ATOMS"
            ):

                raise RuntimeError(
                    f"ATOMS section not found:\n{path}"
                )


            columns = (
                atom_header
                .split()[2:]
            )


            col_index = {

                name: i

                for i, name
                in enumerate(columns)

            }


            # -------------------------------------------------------------
            # Read current frame only
            # -------------------------------------------------------------

            data = np.empty(
                (
                    natoms,
                    len(columns)
                ),
                dtype=float
            )


            for iatom in range(
                natoms
            ):

                vals = (
                    fh.readline()
                    .split()
                )


                if len(vals) != len(columns):

                    raise RuntimeError(
                        "\nUnexpected number of atom columns.\n"
                        f"File: {path}\n"
                        f"Expected: {len(columns)}\n"
                        f"Found: {len(vals)}\n"
                    )


                data[
                    iatom,
                    :
                ] = vals


            # -------------------------------------------------------------
            # Atom types
            # -------------------------------------------------------------

            if "type" in col_index:

                atom_types = data[
                    :,
                    col_index["type"]
                ].astype(
                    int
                )

            else:

                atom_types = np.ones(
                    natoms,
                    dtype=int
                )


            # -------------------------------------------------------------
            # Wrapped Cartesian coordinates
            # -------------------------------------------------------------

            if all(
                x in col_index
                for x in [
                    "x",
                    "y",
                    "z"
                ]
            ):

                pos = np.column_stack([

                    data[
                        :,
                        col_index["x"]
                    ],

                    data[
                        :,
                        col_index["y"]
                    ],

                    data[
                        :,
                        col_index["z"]
                    ],

                ])


                pos = (
                    pos
                    -
                    lo
                )


            # -------------------------------------------------------------
            # Unwrapped Cartesian coordinates
            # -------------------------------------------------------------

            elif all(
                x in col_index
                for x in [
                    "xu",
                    "yu",
                    "zu"
                ]
            ):

                pos = np.column_stack([

                    data[
                        :,
                        col_index["xu"]
                    ],

                    data[
                        :,
                        col_index["yu"]
                    ],

                    data[
                        :,
                        col_index["zu"]
                    ],

                ])


                pos = (
                    pos
                    -
                    lo
                )


            # -------------------------------------------------------------
            # Scaled coordinates
            # -------------------------------------------------------------

            elif all(
                x in col_index
                for x in [
                    "xs",
                    "ys",
                    "zs"
                ]
            ):

                scaled = np.column_stack([

                    data[
                        :,
                        col_index["xs"]
                    ],

                    data[
                        :,
                        col_index["ys"]
                    ],

                    data[
                        :,
                        col_index["zs"]
                    ],

                ])


                pos = (
                    scaled
                    *
                    L
                )


            else:

                raise RuntimeError(
                    "\nCould not identify coordinates.\n"
                    f"File: {path}\n"
                    f"Columns: {columns}\n"
                )


            # Wrap into [0,L)

            pos = np.mod(
                pos,
                L
            )


            yield (

                frame_index,
                timestep,
                L,
                pos,
                atom_types

            )


            frame_index += 1


# =============================================================================
# 11. CLUSTERS IN ONE FRAME
# =============================================================================

def clusters_from_positions(
    positions,
    box_lengths
):

    n = len(
        positions
    )


    if n == 0:

        return np.asarray(
            [],
            dtype=int
        )


    # Periodic KD-tree

    tree = cKDTree(

        positions,

        boxsize=box_lengths

    )


    pairs = tree.query_pairs(

        r=R_CLUSTER,

        output_type="ndarray"

    )


    uf = UnionFind(
        n
    )


    for pair in pairs:

        i = int(
            pair[0]
        )

        j = int(
            pair[1]
        )


        uf.union(
            i,
            j
        )


    return uf.cluster_sizes()


# =============================================================================
# 12. FRAME STATISTICS
# =============================================================================

def frame_statistics(
    sizes,
    N
):

    sizes = np.asarray(
        sizes,
        dtype=float
    )


    if len(sizes) == 0:

        return None


    # -------------------------------------------------------------------------
    # Number-average cluster size
    #
    # <s>_cl = sum_s s N_s / sum_s N_s
    # -------------------------------------------------------------------------

    mean_cluster = np.mean(
        sizes
    )


    # -------------------------------------------------------------------------
    # Particle-weighted cluster size
    #
    # <s>_w = sum_s s^2 N_s / sum_s s N_s
    #
    # This is the mean cluster size experienced by a randomly chosen particle.
    # -------------------------------------------------------------------------

    weighted_cluster = (

        np.sum(
            sizes**2
        )

        /

        np.sum(
            sizes
        )

    )


    largest_cluster = np.max(
        sizes
    )


    largest_fraction = (

        largest_cluster
        /
        float(N)

    )


    # -------------------------------------------------------------------------
    # Fraction of particles belonging to non-monomer clusters
    # -------------------------------------------------------------------------

    clustered_particles = np.sum(
        sizes[
            sizes >= 2
        ]
    )


    clustered_fraction = (

        clustered_particles
        /
        float(N)

    )


    # -------------------------------------------------------------------------
    # Monomer fraction
    # -------------------------------------------------------------------------

    n_monomers = np.sum(
        sizes == 1
    )


    monomer_fraction = (

        n_monomers
        /
        float(N)

    )


    return {

        "Nclusters":
        float(
            len(sizes)
        ),

        "mean_cluster":
        float(
            mean_cluster
        ),

        "weighted_cluster":
        float(
            weighted_cluster
        ),

        "largest_cluster":
        float(
            largest_cluster
        ),

        "largest_fraction":
        float(
            largest_fraction
        ),

        "clustered_fraction":
        float(
            clustered_fraction
        ),

        "monomer_fraction":
        float(
            monomer_fraction
        ),

    }


# =============================================================================
# 13. FIND AND FILTER TRAJECTORIES
# =============================================================================

all_files = sorted(
    TRAJECTORY_ROOT.glob(
        TRAJECTORY_GLOB
    )
)


print()
print("=" * 90)
print("USALR STATIC CLUSTER ANALYSIS")
print("=" * 90)

print(
    f"Trajectory directory = "
    f"{TRAJECTORY_ROOT.resolve()}"
)

print(
    f"All trajectories found = "
    f"{len(all_files)}"
)

print(
    f"R_CLUSTER = "
    f"{R_CLUSTER:.4f}"
)

print(
    f"T domain = "
    f"{T_MIN:.2f} -- {T_MAX:.2f}"
)

print(
    f"P domain = "
    f"{P_MIN:.2f} -- {P_MAX:.2f}"
)


if len(all_files) == 0:

    raise RuntimeError(
        "\nNo trajectories were found.\n"
    )


# -------------------------------------------------------------------------
# Build explicit list of trajectories inside the target domain
# -------------------------------------------------------------------------

files = []


for path in all_files:

    try:

        P, T = parse_PT(
            path
        )

    except Exception:

        continue


    if not (
        P_MIN <= P <= P_MAX
        and
        T_MIN <= T <= T_MAX
    ):

        continue


    files.append(
        (
            path,
            P,
            T
        )
    )


print(
    f"Trajectories inside target domain = "
    f"{len(files)}"
)


if len(files) == 0:

    raise RuntimeError(
        "\nNo trajectories remain after P,T filtering.\n"
    )


# =============================================================================
# 14. MAIN LOOP
# =============================================================================

state_rows = []

failed_states = []


for ifile, (
    path,
    P,
    T
) in enumerate(
    files,
    start=1
):


    print()
    print("-" * 90)

    print(
        f"[{ifile}/{len(files)}] "
        f"P*={P:.3f}  "
        f"T*={T:.2f}"
    )

    print(
        path
    )


    try:

        # ---------------------------------------------------------------------
        # Count frames
        # ---------------------------------------------------------------------

        n_total = count_frames(
            path
        )


        if n_total == 0:

            raise RuntimeError(
                "Trajectory contains no frames."
            )


        first_frame = int(
            np.floor(
                DISCARD_FRACTION
                *
                n_total
            )
        )


        print(
            f"total frames = {n_total}"
        )

        print(
            f"first production frame = {first_frame}"
        )


        frame_stats = []

        size_counter = Counter()

        N_particles = None

        n_selected = 0


        # ---------------------------------------------------------------------
        # Stream trajectory
        # ---------------------------------------------------------------------

        for (
            frame_index,
            timestep,
            box_lengths,
            positions,
            atom_types
        ) in iter_lammpstrj(
            path
        ):


            # -------------------------------------------------------------
            # Discard initial fraction
            # -------------------------------------------------------------

            if frame_index < first_frame:

                continue


            # -------------------------------------------------------------
            # Stride
            # -------------------------------------------------------------

            production_index = (
                frame_index
                -
                first_frame
            )


            if (
                production_index
                %
                FRAME_STRIDE
                != 0
            ):

                continue


            # -------------------------------------------------------------
            # Maximum selected frames
            # -------------------------------------------------------------

            if (
                MAX_FRAMES_PER_STATE
                is not None
                and
                n_selected
                >=
                MAX_FRAMES_PER_STATE
            ):

                break


            # -------------------------------------------------------------
            # Particle selection
            # -------------------------------------------------------------

            if PARTICLE_TYPE is None:

                positions_use = positions

            else:

                mask_type = (
                    atom_types
                    ==
                    PARTICLE_TYPE
                )


                positions_use = positions[
                    mask_type
                ]


            N = len(
                positions_use
            )


            if N == 0:

                continue


            if N_particles is None:

                N_particles = N


            elif N != N_particles:

                raise RuntimeError(
                    "\nParticle number changed between frames.\n"
                    f"Initial N = {N_particles}\n"
                    f"Current N = {N}\n"
                )


            # -------------------------------------------------------------
            # Cluster connectivity
            # -------------------------------------------------------------

            sizes = clusters_from_positions(

                positions_use,

                box_lengths

            )


            stat = frame_statistics(

                sizes,
                N

            )


            if stat is None:

                continue


            frame_stats.append(
                stat
            )


            for s in sizes:

                size_counter[
                    int(s)
                ] += 1


            n_selected += 1


        # ---------------------------------------------------------------------
        # Validate state
        # ---------------------------------------------------------------------

        if len(frame_stats) == 0:

            raise RuntimeError(
                "No frames were selected/analyzed."
            )


        # ---------------------------------------------------------------------
        # State averages + SEM
        # ---------------------------------------------------------------------

        keys = list(
            frame_stats[0].keys()
        )


        means = {}
        errors = {}


        for key in keys:

            values = np.asarray(
                [
                    item[key]
                    for item in frame_stats
                ],
                dtype=float
            )


            means[key] = np.mean(
                values
            )


            if len(values) > 1:

                errors[key] = (

                    np.std(
                        values,
                        ddof=1
                    )

                    /

                    np.sqrt(
                        len(values)
                    )

                )

            else:

                errors[key] = np.nan


        # ---------------------------------------------------------------------
        # Large-cluster diagnostic
        # ---------------------------------------------------------------------

        macrocluster_flag = int(

            means[
                "largest_fraction"
            ]

            >=

            LARGEST_CLUSTER_THRESHOLD

        )


        # ---------------------------------------------------------------------
        # Save state row
        # ---------------------------------------------------------------------

        state_rows.append({

            "P":
            P,

            "T":
            T,

            "N":
            N_particles,

            "Nframes_total":
            n_total,

            "Nframes_analyzed":
            len(frame_stats),

            "Nclusters":
            means["Nclusters"],

            "Nclusters_sem":
            errors["Nclusters"],

            "mean_cluster":
            means["mean_cluster"],

            "mean_cluster_sem":
            errors["mean_cluster"],

            "weighted_cluster":
            means["weighted_cluster"],

            "weighted_cluster_sem":
            errors["weighted_cluster"],

            "largest_cluster":
            means["largest_cluster"],

            "largest_cluster_sem":
            errors["largest_cluster"],

            "largest_fraction":
            means["largest_fraction"],

            "largest_fraction_sem":
            errors["largest_fraction"],

            "clustered_fraction":
            means["clustered_fraction"],

            "clustered_fraction_sem":
            errors["clustered_fraction"],

            "monomer_fraction":
            means["monomer_fraction"],

            "monomer_fraction_sem":
            errors["monomer_fraction"],

            "macrocluster_flag":
            macrocluster_flag,

        })


        # ---------------------------------------------------------------------
        # Cluster-size distribution
        # ---------------------------------------------------------------------

        sizes_sorted = np.asarray(
            sorted(
                size_counter.keys()
            ),
            dtype=int
        )


        counts = np.asarray(
            [
                size_counter[
                    int(s)
                ]
                for s in sizes_sorted
            ],
            dtype=float
        )


        # Cluster-number probability

        P_cluster = (

            counts
            /
            np.sum(
                counts
            )

        )


        # Particle-weighted probability

        particle_weight = (

            sizes_sorted
            *
            counts

        )


        P_particle = (

            particle_weight
            /
            np.sum(
                particle_weight
            )

        )


        dist_file = (

            DISTDIR
            /
            (
                f"cluster_distribution_"
                f"P_{P:.3f}_T_{T:.2f}.dat"
            )

        )


        np.savetxt(

            dist_file,

            np.column_stack([

                sizes_sorted,
                counts,
                P_cluster,
                P_particle,

            ]),

            header=(
                "s N_s "
                "P_cluster_s "
                "P_particle_s"
            ),

            fmt=[
                "%d",
                "%.10e",
                "%.10e",
                "%.10e",
            ]

        )


        # ---------------------------------------------------------------------
        # Screen summary
        # ---------------------------------------------------------------------

        print(
            f"analyzed frames       = "
            f"{len(frame_stats)}"
        )

        print(
            f"<s>_cl                = "
            f"{means['mean_cluster']:.6f}"
        )

        print(
            f"<s>_w                 = "
            f"{means['weighted_cluster']:.6f}"
        )

        print(
            f"<s_max/N>             = "
            f"{means['largest_fraction']:.6f}"
        )

        print(
            f"clustered fraction    = "
            f"{means['clustered_fraction']:.6f}"
        )

        print(
            f"monomer fraction      = "
            f"{means['monomer_fraction']:.6f}"
        )

        print(
            f"macrocluster flag     = "
            f"{macrocluster_flag}"
        )


    except Exception as exc:

        print(
            "FAILED:"
        )

        print(
            exc
        )


        failed_states.append({

            "P":
            P,

            "T":
            T,

            "file":
            str(path),

            "error":
            str(exc).replace(
                "\n",
                " "
            ),

        })


# =============================================================================
# 15. BUILD STATE DATABASE
# =============================================================================

cluster = pd.DataFrame(
    state_rows
)


if len(cluster) == 0:

    raise RuntimeError(
        "\nNo states were successfully analyzed.\n"
    )


cluster = (

    cluster
    .sort_values(
        [
            "T",
            "P"
        ]
    )
    .reset_index(
        drop=True
    )

)


# =============================================================================
# 16. MERGE WITH DIFFUSION
# =============================================================================

cluster["D"] = np.nan
cluster["lnD"] = np.nan


if D_FILE.exists():

    D_raw = pd.read_csv(

        D_FILE,

        sep=r"\s+",

        comment="#",

        header=None,

        engine="python"

    )


    if D_raw.shape[1] < 3:

        raise RuntimeError(
            "\nDiffusion file must contain at least:\n"
            "P T D\n"
        )


    D = pd.DataFrame({

        "P":
        pd.to_numeric(
            D_raw.iloc[:, 0],
            errors="coerce"
        ),

        "T":
        pd.to_numeric(
            D_raw.iloc[:, 1],
            errors="coerce"
        ),

        "D":
        pd.to_numeric(
            D_raw.iloc[:, 2],
            errors="coerce"
        ),

    })


    D = D[
        np.isfinite(
            D["P"]
        )
        &
        np.isfinite(
            D["T"]
        )
        &
        np.isfinite(
            D["D"]
        )
        &
        (
            D["D"] > 0
        )
    ].copy()


    D = (

        D
        .groupby(
            [
                "P",
                "T"
            ],
            as_index=False
        )
        .mean(
            numeric_only=True
        )

    )


    D["lnD"] = np.log(
        D["D"]
    )


    # ---------------------------------------------------------------------
    # Rounded merge keys
    # ---------------------------------------------------------------------

    cluster["Pkey"] = np.round(
        cluster["P"],
        6
    )

    cluster["Tkey"] = np.round(
        cluster["T"],
        6
    )


    D["Pkey"] = np.round(
        D["P"],
        6
    )

    D["Tkey"] = np.round(
        D["T"],
        6
    )


    # Remove empty D columns before merge

    cluster = cluster.drop(
        columns=[
            "D",
            "lnD"
        ]
    )


    cluster = pd.merge(

        cluster,

        D[
            [
                "Pkey",
                "Tkey",
                "D",
                "lnD"
            ]
        ],

        on=[
            "Pkey",
            "Tkey"
        ],

        how="left"

    )


    cluster = cluster.drop(
        columns=[
            "Pkey",
            "Tkey"
        ]
    )


else:

    print()
    print(
        "WARNING: diffusion file not found:"
    )

    print(
        D_FILE
    )


# =============================================================================
# 17. SAVE STATE DATABASE
# =============================================================================

OUT_DAT = (
    OUTDIR
    /
    "cluster_state_summary.dat"
)

OUT_CSV = (
    OUTDIR
    /
    "cluster_state_summary.csv"
)


cluster.to_csv(

    OUT_DAT,

    sep=" ",

    index=False,

    na_rep="nan",

    float_format="%.10e"

)


cluster.to_csv(

    OUT_CSV,

    index=False,

    na_rep="NaN",

    float_format="%.10e"

)


# =============================================================================
# 18. SAVE FAILURES
# =============================================================================

if len(
    failed_states
) > 0:

    pd.DataFrame(
        failed_states
    ).to_csv(

        OUTDIR
        /
        "cluster_failed_states.csv",

        index=False

    )


# =============================================================================
# 19. GLOBAL SUMMARY
# =============================================================================

print()
print("=" * 90)
print("GLOBAL SUMMARY")
print("=" * 90)

print(
    f"States successfully analyzed = "
    f"{len(cluster)}"
)

print(
    f"Failed states                = "
    f"{len(failed_states)}"
)

print(
    f"P range = "
    f"{cluster['P'].min():.3f} -- "
    f"{cluster['P'].max():.3f}"
)

print(
    f"T range = "
    f"{cluster['T'].min():.3f} -- "
    f"{cluster['T'].max():.3f}"
)

print(
    f"Mean <s>_cl = "
    f"{cluster['mean_cluster'].mean():.6f}"
)

print(
    f"Mean <s>_w  = "
    f"{cluster['weighted_cluster'].mean():.6f}"
)

print(
    f"Maximum <s>_w = "
    f"{cluster['weighted_cluster'].max():.6f}"
)

print(
    f"Maximum <s_max/N> = "
    f"{cluster['largest_fraction'].max():.6f}"
)

print(
    "Macrocluster-flagged states = "
    f"{int(cluster['macrocluster_flag'].sum())}"
)

print(
    "States with matched diffusion = "
    f"{int(np.isfinite(cluster['D']).sum())}"
)


# =============================================================================
# 20. PLOT HELPER
# =============================================================================

def plot_isotherms(
    ycol,
    ylabel,
    stem,
    sem_col=None
):

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            5.3
        )
    )


    for T in T_SELECTED:

        sub = cluster[
            np.isclose(
                cluster["T"],
                T,
                atol=1e-8
            )
        ].sort_values(
            "P"
        )


        if len(sub) == 0:

            continue


        color = COLORS.get(
            T,
            None
        )


        if (
            sem_col is not None
            and
            sem_col in sub.columns
        ):

            ax.errorbar(

                sub["P"],
                sub[ycol],

                yerr=sub[sem_col],

                marker="o",

                markersize=4.8,

                linewidth=1.6,

                elinewidth=0.7,

                capsize=1.5,

                color=color,

                label=(
                    rf"$T^*={T:.2f}$"
                )

            )


        else:

            ax.plot(

                sub["P"],
                sub[ycol],

                marker="o",

                markersize=4.8,

                linewidth=1.7,

                color=color,

                label=(
                    rf"$T^*={T:.2f}$"
                )

            )


    ax.set_xlabel(
        r"$P^*$"
    )

    ax.set_ylabel(
        ylabel
    )


    ax.legend(
        ncol=2
    )


    fig.tight_layout()


    fig.savefig(

        OUTDIR
        /
        f"{stem}.pdf",

        bbox_inches="tight"

    )


    fig.savefig(

        OUTDIR
        /
        f"{stem}.png",

        dpi=350,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )


# =============================================================================
# 21. CLUSTER PLOTS
# =============================================================================

plot_isotherms(

    "mean_cluster",

    r"$\langle s\rangle_{\rm cl}$",

    "cluster_mean_vs_pressure",

    "mean_cluster_sem"

)


plot_isotherms(

    "weighted_cluster",

    r"$\langle s\rangle_{\rm w}$",

    "cluster_weighted_vs_pressure",

    "weighted_cluster_sem"

)


plot_isotherms(

    "largest_fraction",

    r"$\langle s_{\max}/N\rangle$",

    "largest_cluster_fraction",

    "largest_fraction_sem"

)


plot_isotherms(

    "clustered_fraction",

    r"$f_{\rm cl}$",

    "clustered_fraction",

    "clustered_fraction_sem"

)


# =============================================================================
# 22. DIFFUSION vs CLUSTER SIZE
# =============================================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        10.8,
        4.5
    ),

    sharex=False

)


ax1 = axes[0]
ax2 = axes[1]


for T in T_SELECTED:

    sub = cluster[
        np.isclose(
            cluster["T"],
            T,
            atol=1e-8
        )
    ].sort_values(
        "P"
    )


    if len(sub) == 0:

        continue


    color = COLORS.get(
        T,
        None
    )


    mask_D = np.isfinite(
        sub["lnD"]
    )


    if np.any(
        mask_D
    ):

        ax1.plot(

            sub.loc[
                mask_D,
                "P"
            ],

            sub.loc[
                mask_D,
                "lnD"
            ],

            marker="o",

            markersize=4.5,

            linewidth=1.6,

            color=color,

            label=(
                rf"$T^*={T:.2f}$"
            )

        )


    ax2.errorbar(

        sub["P"],
        sub["weighted_cluster"],

        yerr=sub[
            "weighted_cluster_sem"
        ],

        marker="o",

        markersize=4.5,

        linewidth=1.6,

        elinewidth=0.7,

        capsize=1.5,

        color=color

    )


ax1.set_xlabel(
    r"$P^*$"
)

ax2.set_xlabel(
    r"$P^*$"
)


ax1.set_ylabel(
    r"$\ln D^*$"
)

ax2.set_ylabel(
    r"$\langle s\rangle_{\rm w}$"
)


ax1.text(

    0.04,
    0.95,

    r"(a)",

    transform=ax1.transAxes,

    ha="left",
    va="top",

    fontsize=13

)


ax2.text(

    0.04,
    0.95,

    r"(b)",

    transform=ax2.transAxes,

    ha="left",
    va="top",

    fontsize=13

)


ax1.legend(
    ncol=2
)


fig.tight_layout()


fig.savefig(

    OUTDIR
    /
    "cluster_vs_diffusion.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR
    /
    "cluster_vs_diffusion.png",

    dpi=350,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 23. SCATTER lnD vs WEIGHTED CLUSTER SIZE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        6.8,
        5.2
    )
)


for T in T_SELECTED:

    sub = cluster[
        np.isclose(
            cluster["T"],
            T,
            atol=1e-8
        )
    ].copy()


    mask = (
        np.isfinite(
            sub["lnD"]
        )
        &
        np.isfinite(
            sub["weighted_cluster"]
        )
    )


    sub = sub[
        mask
    ]


    if len(sub) == 0:

        continue


    ax.scatter(

        sub[
            "weighted_cluster"
        ],

        sub[
            "lnD"
        ],

        s=32,

        color=COLORS.get(
            T,
            None
        ),

        label=(
            rf"$T^*={T:.2f}$"
        )

    )


ax.set_xlabel(
    r"$\langle s\rangle_{\rm w}$"
)

ax.set_ylabel(
    r"$\ln D^*$"
)


ax.legend(
    ncol=2
)


fig.tight_layout()


fig.savefig(

    OUTDIR
    /
    "lnD_vs_weighted_cluster.pdf",

    bbox_inches="tight"

)


fig.savefig(

    OUTDIR
    /
    "lnD_vs_weighted_cluster.png",

    dpi=350,

    bbox_inches="tight"

)


plt.close(
    fig
)


# =============================================================================
# 24. FINAL OUTPUT
# =============================================================================

print()
print("=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(
    OUT_DAT
)

print(
    OUT_CSV
)

print(
    OUTDIR /
    "cluster_mean_vs_pressure.pdf"
)

print(
    OUTDIR /
    "cluster_weighted_vs_pressure.pdf"
)

print(
    OUTDIR /
    "largest_cluster_fraction.pdf"
)

print(
    OUTDIR /
    "clustered_fraction.pdf"
)

print(
    OUTDIR /
    "cluster_vs_diffusion.pdf"
)

print(
    OUTDIR /
    "lnD_vs_weighted_cluster.pdf"
)


if len(
    failed_states
) > 0:

    print(
        OUTDIR /
        "cluster_failed_states.csv"
    )


print()
print("Done.")
