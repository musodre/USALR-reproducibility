# Reproduction workflow

The repository is intentionally separated into simulation generation, numerical
analysis, manuscript rendering, and reproducibility validation. This document
describes the consolidated public workflow; historical stage records under
`documentation/` are an audit trail, not alternative execution pipelines.

## 0. Configure data locations

Default locations are:

```text
data/raw/
data/derived/
outputs/figures/
outputs/tables/
```

The complete raw trajectory ensemble is intentionally not tracked by Git. Large
raw data may remain outside the repository:

```bash
export USALR_RAW_DATA=/absolute/path/to/raw/data
export USALR_DERIVED_DATA=/absolute/path/to/derived/data
export USALR_FIGURES=/absolute/path/to/figures
export USALR_TABLES=/absolute/path/to/tables
```

Alternatively, `data/raw/allpress` may be a symbolic link to the external
`allpress` archive.

Verify the resolved configuration:

```bash
python check_paths.py
```

## 1. Simulation generation

Use the LAMMPS material under `01_data_generation/lammps/`. The repository
contains the USALR potential/template and representative production inputs.

The raw production archive supplies, as applicable, thermodynamic profiles,
RDFs, MSDs, and trajectories used by the downstream analyses.

## 2. Base observables

Run the numbered scripts under:

```text
02_data_analysis/01_diffusion/
02_data_analysis/02_thermodynamics/
02_data_analysis/04_pair_order/
```

These generate the base dynamic, thermodynamic, and pair-order quantities used
by later stages.

### Pair-order provenance

The canonical regenerated `s2` and `tau` products are based on the audited
current raw-data tree. Historical archived tables contain duplicate-state and
low-pressure lineage effects. These differences were investigated explicitly;
they are therefore represented by `PROVENANCE_VERIFIED` in the final numerical
regression rather than being hidden by loose tolerances.

## 3. Temperature of maximum density (TMD)

Use the scripts under:

```text
02_data_analysis/03_tmd/
```

Two TMD uses must remain distinct:

- the selected-isobar density-maxima analysis used for Figure 3;
- the global physical TMD branch used in the phase-diagram workflow.

The provenance audit established that a historical intermediate TMD candidate
table and the later validation reconstruction are not numerically identical.
The branch-tracking workflow therefore preserves the documented physical branch
rather than silently substituting a different candidate generator.

## 4. Shell-resolved structure

Run:

```text
02_data_analysis/05_shell_structure/
```

This produces the fixed-window shell descriptors used by the manuscript's
shell-resolved structural analysis.

The fixed-window population ratio is not interchangeable with the historical
broad-shell `Rn` retained in the Fig. 12 derivative chain.

## 5. Static structure factor and SALR-area descriptor

Run the numbered scripts under:

```text
02_data_analysis/06_structure_factor/
```

The first part computes the static structure factor from the available
trajectories; the subsequent analysis constructs the associated SALR-area
descriptor used downstream.

## 6. Structure–dynamics hierarchy

Run:

```text
02_data_analysis/07_structure_dynamics/
```

in numerical filename order.

This stage constructs the common structure–dynamics products used in the
manuscript hierarchy analysis.

### Historical `Rn` provenance

The Fig. 12 chain deliberately retains the historical broad-shell population
ratio `Rn = n2/n1`. Its provenance was numerically reconstructed and verified.
It must not be silently replaced by the later fixed-window shell descriptor.

This provenance choice propagates into the common author table and the
derivative master table. Their archived-vs-regenerated differences are
classified `PROVENANCE_VERIFIED` by the release validator.

## 7. Derivative analysis and dynamic boundary

Run:

```text
02_data_analysis/08_derivative_analysis/
```

in numerical filename order.

This stage evaluates local pressure derivatives, classifies dynamic behavior,
and constructs the dynamic-anomaly boundary used by downstream cluster
alignment and manuscript analysis.

## 8. Cluster analysis

Run:

```text
02_data_analysis/09_clusters/
```

in numerical filename order.

The production connectivity cutoff is:

```text
R_CLUSTER = 1.20
```

An older value near `1.5942` found in historical documentation was identified
as a stale documentation artifact and is not the cutoff used by the reproduced
production workflow.

The cluster chain includes the state summary, alignment with the dynamic
boundary, dynamic master products, and selected cluster-size distributions.

## 9. Manuscript figures

Run the relevant scripts under:

```text
03_figures/
```

after their numerical prerequisites have been generated.

Figure 11 is generated directly by the corresponding structure–dynamics
analysis script rather than by a duplicate renderer.

## 10. Numerical regression

Run the canonical validator from the repository root:

```bash
python validate_numerical_outputs.py \
  --reference-root /path/to/original/USALR \
  --regenerated-root data/derived
```

The consolidated release currently validates 14 principal products:

```text
diffusion_global                 PASS
s2_global                        PROVENANCE_VERIFIED
tau_global                       PROVENANCE_VERIFIED
thermodynamic_response           PASS
tmd_physical_branch              PASS
shell_descriptors_fixed_window   PASS
structure_factor_area            PASS
structure_dynamics_hierarchy     PASS
author_master_common             PROVENANCE_VERIFIED
derivative_master                PROVENANCE_VERIFIED
dynamic_boundaries               PASS
cluster_state_summary            PASS
cluster_dynamic_master           PASS
selected_cluster_states          PASS
```

Summary:

```text
PASS                     10
PROVENANCE_VERIFIED       4
DIFFERENT                  0
ERROR                      0
REGENERATED_MISSING        0
```

The detailed machine-readable report is written to:

```text
outputs/tables/numerical_regression_report.csv
```

`PROVENANCE_VERIFIED` is deliberately distinct from `PASS`: it records a
historical numerical difference whose origin has been explicitly reconstructed
and documented. It is not a relaxed-tolerance numerical match.

## Reproducibility terminology

- `PASS`: regenerated and archived numerical products agree under the declared
  comparison rules and tolerances.
- `PROVENANCE_VERIFIED`: a remaining historical difference has an explicitly
  audited data-lineage/provenance explanation.
- `DIFFERENT`: an unexplained numerical mismatch remains.
- `ERROR`: validation could not be completed.
- `REGENERATED_MISSING`: an expected regenerated product is absent.

For the consolidated release there are no products in the last three
categories.

## Release discipline

Before tagging a release:

1. verify data paths with `check_paths.py`;
2. regenerate the required derived products from the intended raw-data archive;
3. run the canonical numerical validator;
4. require the documented `10 PASS + 4 PROVENANCE_VERIFIED` result;
5. inspect `outputs/tables/numerical_regression_report.csv`;
6. complete `documentation/RELEASE_CHECKLIST.md`;
7. update release checksums as appropriate.

Historical stage notes should be retained as provenance records, but the
consolidated files in the repository root and this document define the public
workflow.
