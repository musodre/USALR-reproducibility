# USALR fluid — reproducibility package
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21844211.svg)](https://doi.org/10.5281/zenodo.21844211)


This repository contains the simulation inputs, numerical-analysis scripts, figure renderers, provenance records, and numerical-regression tools supporting the manuscript **“Compression-enhanced mobility during finite-cluster growth in a soft colloidal SALR fluid.”**

This is the consolidated release tree. Development snapshots (“stages”) are not part of the public workflow.

## Reproducibility status

The consolidated numerical regression currently reports:

| Status | Products |
|---|---:|
| `PASS` | 10 |
| `PROVENANCE_VERIFIED` | 4 |
| `DIFFERENT` | 0 |
| `ERROR` | 0 |
| `REGENERATED_MISSING` | 0 |

The four `PROVENANCE_VERIFIED` products are `s2_global`, `tau_global`,
`author_master_common`, and `derivative_master`. These are not unresolved
numerical failures: their differences from archived historical tables have been
traced to documented historical data lineage and duplicate-state conventions.
See the provenance documentation and the numerical-regression report under
`outputs/tables/`.

## Repository layout

```text
01_data_generation/   LAMMPS potential/template and representative production inputs
02_data_analysis/     numerical post-processing in scientific workflow order
03_figures/           manuscript figure renderers
data/raw/             raw simulation data or symlink to an external archive
data/derived/         regenerated intermediate/final numerical products
outputs/figures/      generated manuscript/diagnostic figures
outputs/tables/       validation and summary tables
documentation/        workflow, provenance, audits, and release records
```

Large raw trajectories are intentionally not tracked by Git.

## Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate usalr-reproducibility
```

or a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data locations

By default, the repository expects the primary simulation archive at

```text
data/raw/allpress/
```

This may be a symbolic link to the external raw-data archive. Alternatively,
configure locations through environment variables:

```bash
export USALR_RAW_DATA=/path/to/directory/containing/allpress
export USALR_DERIVED_DATA=/path/to/derived/data
export USALR_FIGURES=/path/to/figure/output
export USALR_TABLES=/path/to/table/output
```

Repository paths are centralized by `usalr_paths.py`. Run

```bash
python check_paths.py
```

before a full reproduction when using non-default locations.

## Quick start

All commands should be run from the repository root. No manual `PYTHONPATH`
editing is required.

Examples:

```bash
python 02_data_analysis/04_pair_order/01_compute_s2.py
python 02_data_analysis/04_pair_order/02_compute_tau.py
python 02_data_analysis/05_shell_structure/01_shell_resolved_descriptors.py
```

For the complete execution order, see `RUNNING.md` and
`documentation/REPRODUCTION_WORKFLOW.md`.

## Scientific workflow

The high-level dependency chain is

```text
LAMMPS production
  -> diffusion / thermodynamic response
  -> s2 and tau
  -> TMD analysis
  -> RDF shell-resolved descriptors
  -> S(k*) and A_SALR
  -> common structure-dynamics database
  -> local pressure derivatives / anomaly boundary
  -> cluster analysis
  -> manuscript figures
```

The numbered directories under `02_data_analysis/` encode this organization.

## Important provenance notes

### TMD

Two distinct TMD workflows are retained and must not be conflated.

**Figure 3** directly analyzes selected density isobars and displays their
density maxima.

The **global physical TMD branch** is produced by the dedicated branch-tracking
workflow under `02_data_analysis/03_tmd/`. The repository retains the provenance
record that the historical intermediate TMD candidate table and the later
candidate-validation reconstruction are not numerically identical. They must
not be silently substituted for one another.

### Pair-order tables (`s2` and `tau`)

Historical tables contain duplicate-state/data-lineage effects associated with
low-pressure datasets and alternate pressure-directory representations. The
canonical regenerated tables use the audited current raw-data provenance.
Consequently, `s2_global` and `tau_global` are classified
`PROVENANCE_VERIFIED`, not `DIFFERENT`, by the release validator.

### Fig. 12 `Rn`

The derivative/Fig. 12 chain deliberately retains the historical broad-shell
population ratio `Rn = n2/n1`. This is distinct from the later fixed-window
shell descriptor used in the shell-resolved structural analysis. The historical
definition was reconstructed numerically and is retained for manuscript
reproducibility; it must not be silently replaced by the fixed-window quantity.

Because the author common table and derivative master table inherit audited
historical provenance choices, `author_master_common` and `derivative_master`
are also classified `PROVENANCE_VERIFIED`.

### Cluster connectivity

The manuscript cluster workflow uses

```text
R_CLUSTER = 1.20
```

An older historical documentation value near `1.5942` was identified as a stale
documentation artifact and is not the production cutoff used by the
reproducibility workflow.

## Numerical regression

After regenerating the required products, run the canonical validator from the
repository root:

```bash
python validate_numerical_outputs.py \
  --reference-root /path/to/original/USALR \
  --regenerated-root data/derived
```

For the consolidated release, the expected summary is:

```text
PASS                     10
PROVENANCE_VERIFIED       4
```

with zero `DIFFERENT`, `ERROR`, and `REGENERATED_MISSING` products.

The machine-readable report is written to:

```text
outputs/tables/numerical_regression_report.csv
```

`PROVENANCE_VERIFIED` means that a non-bitwise historical difference has been
explicitly audited and attributed to documented provenance rather than accepted
through relaxed numerical tolerances.

## Reproduction and provenance documentation

Start with:

- `RUNNING.md` — practical execution instructions;
- `documentation/REPRODUCTION_WORKFLOW.md` — scientific dependency chain;
- provenance/audit records under `documentation/` — historical lineage and
  migration decisions;
- `documentation/RELEASE_CHECKLIST.md` — release checks.

The public workflow is the consolidated tree. Historical stage documents are
retained only as an audit trail and are not separate execution workflows.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The repository is distributed
under the license in `LICENSE`.

Checksums for release files are recorded in `SHA256SUMS`.
