# Confirmation-cohort eligibility audit — 11 October 2026

No currently verified local cohort supports an untouched confirmation of the
same sepsis-onset endpoint. This is a limitation of the requested third addition,
not a reason to stop the independent baseline and pretraining-control jobs.

| Candidate | Known prior use / endpoint issue | Current decision |
| --- | --- | --- |
| CinC set B test | Repeatedly scored and interpreted in transfer and measurement studies | Exploratory evaluation only |
| CinC set A historical mirror | Used in earlier development; BIDMC overlap with MIMIC source data | Not independent confirmation |
| Additional set A records | Some records outside mirror, but patient identity cannot exclude overlap with source MIMIC; original provenance census already inspected inputs | Cannot certify as untouched across all arms |
| MIMIC held-out records | Earlier supervised experiments and audit test evaluations used MIMIC holdouts | Not a new confirmation cohort |
| SICdb preserved people | Original 30% person holdout excluded from source pretraining; however no culture/microbiology table, so the same infection-pairing onset label cannot be reconstructed | Preserve holdout; do not substitute antibiotic-only labels for the primary endpoint |
| SDY1662 | COVID prognosis cohort; not the requested sepsis-onset task | Ineligible |

Evidence: `scripts/prepare_transfer_sources.py` preserves the SICdb people;
`DATA_ACCESS_SPEC.md` section 12 and `sepsentinel/data/sicdb.py` document absent
cultures and a different antibiotic/SOFA proxy. The prior transfer manifests
and experiment records establish repeated use of the target test data.

A label-free SICdb alert-rate analysis could measure how often the system fires,
but cannot call alarms false or estimate sepsis precision/recall without validated
outcomes. That is not a substitute confirmation of this experiment.

Next eligible option requires another authorized cohort with reliable onset
labels, or independently adjudicated labels in the preserved SICdb holdout.
Before any such evaluation: audit patient overlap, freeze preprocessing and one
selection rule using existing development/calibration data, then evaluate once.
An eICU/HiRID extraction would require verified access and a validated endpoint;
neither is currently provisioned in this local workspace inventory.
