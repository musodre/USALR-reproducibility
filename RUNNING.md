# Running the USALR reproducibility repository

All commands below are run from the repository root.

## 1. Prepare the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate usalr-reproducibility
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure the data layout

Primary simulation data are expected by default under:

```text
data/raw/allpress/
```

Derived products are written under:

```text
data/derived/
```

Figures and validation tables are written under:

```text
outputs/figures/
outputs/tables/
```

For a large external raw-data archive, either create a symbolic link:

```bash
ln -s /absolute/path/to/allpress data/raw/allpress
```

or set:

```bash
export USALR_RAW_DATA=/absolute/path/to/directory/containing/allpress
```

Optional output overrides are:

```bash
export USALR_DERIVED_DATA=/absolute/path/to/derived/data
export USALR_FIGURES=/absolute/path/to/figures
export USALR_TABLES=/absolute/path/to/tables
```

Check the resolved paths with:

```bash
python check_paths.py
```

No manual `PYTHONPATH` modification is required. Analysis and figure scripts
locate the repository root automatically when executed directly.

## 3. Run the analysis workflow

The numbered directories under `02_data_analysis/` encode the intended
scientific order. Follow `documentation/REPRODUCTION_WORKFLOW.md` for the full
dependency chain.

Typical commands include:

```bash
python 02_data_analysis/04_pair_order/01_compute_s2.py
python 02_data_analysis/04_pair_order/02_compute_tau.py
python 02_data_analysis/05_shell_structure/01_shell_resolved_descriptors.py
```

Run multi-script analysis directories in numerical filename order unless the
workflow documentation states otherwise.

## 4. Render figures

After their numerical prerequisites have been generated, run the appropriate
scripts under `03_figures/`.

Figure 11 is generated directly by the corresponding structure–dynamics
analysis rather than by a duplicate renderer.

## 5. Run numerical regression

After regeneration, validate the principal published-data products:

```bash
python validate_numerical_outputs.py \
  --reference-root /path/to/original/USALR \
  --regenerated-root data/derived
```

The consolidated release target is:

```text
PASS                     10
PROVENANCE_VERIFIED       4
```

with no `DIFFERENT`, `ERROR`, or `REGENERATED_MISSING` products.

The detailed report is:

```text
outputs/tables/numerical_regression_report.csv
```

## Provenance constraints

Three distinctions are essential when reproducing the manuscript.

1. The selected-isobar TMD analysis used for Figure 3 and the global physical
   TMD branch are related but distinct workflows. Do not silently substitute
   candidate tables between them.

2. The derivative/Fig. 12 workflow intentionally preserves the historical
   broad-shell `Rn = n2/n1` definition. The fixed-window shell database is a
   distinct structural product.

3. The production cluster connectivity cutoff is `R_CLUSTER = 1.20`.

The historical `s2` and `tau` reference tables also contain audited
duplicate/data-lineage effects. Their final regression status is therefore
`PROVENANCE_VERIFIED`. The inherited author-master and derivative-master
differences are treated the same way after explicit provenance verification.

See the records under `documentation/` before changing any of these scientific
or provenance conventions.
