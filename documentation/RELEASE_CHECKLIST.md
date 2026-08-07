# Public-release checklist

- [x] Code separated into generation / analysis / figure layers
- [x] Central path configuration
- [x] Python dependency files
- [x] Git ignore rules for large data/output
- [x] TMD provenance caveat documented
- [x] Fig. 12 Rn provenance reconstructed numerically
- [x] Cluster cutoff provenance corrected
- [x] Principal figure renderers path-normalized
- [x] Full Python compile audit
- [x] LICENSE added
- [x] CITATION.cff template added
- [x] Reproduction workflow added
- [ ] Resolve remaining active-path candidates individually
- [x] Numerical regression framework created
- [ ] Numerical regression executed against full archived outputs
- [ ] Add final manuscript citation
- [ ] Add GitHub repository URL
- [ ] Create Zenodo deposition and add DOI
- [ ] Tag immutable release (recommended: v1.0.0)
### Whitespace exceptions

The following archived scientific-source/audit files contain trailing
whitespace inherited from their original formatting and are intentionally
preserved without normalization:

- `01_data_generation/lammps/pot_Uyes04pol.dat`
- `SI/multireference/S7_raw_audit.txt`

These files are excluded from the whitespace-only release check. This
exception does not apply to executable Python source code.
