# Biopharma and drug-discovery scope

**Status:** experimental planning and screening utilities; not a drug-development
or clinical decision system.

## What is implemented

| Component | Current behavior | Evidence tier |
|---|---|---|
| Biomarker analyzer | Compares a small set of supplied values with hard-coded reference ranges. Values without a supported range are returned as not assessed. Its risk score is an illustrative threshold count, not a calibrated clinical risk estimate. | `rule_screen` |
| Target list | Returns a static list and an illustrative tractability-based ordering. It does not analyze omics; passing `omics_data` raises `NotImplementedError`. No source-level provenance is attached to each example target. | `static_example` |
| ADMET | Applies Lipinski and Veber descriptor rules only. It does not predict absorption, distribution, metabolism, excretion, toxicity, clearance, permeability, hERG liability, or hepatotoxicity. | `rule_screen` |
| Repurposing and combinations | Uses small static example tables and deterministic planning sketches. Scores and assumed response rates are not learned from datasets or patient outcomes. | `static_example` / `planning_template` |
| Trial design | Emits a trial-design template and fixed defaults. It does not calculate a powered sample size from a specified endpoint, treatment effect, variance, error rates, or dropout assumptions. | `planning_template` |
| Molecular relaxation | The separate SE(3)-equivariant module implements a simple classical distance-based energy and gradient-flow demonstration. It is not a validated molecular mechanics force field and does not model electronic structure, protein folding, or binding. | `experimental` |
| Multiplicity correction | `apedp_dyn.epistemic.multiple_testing.adjust_p_values` adjusts an explicit p-value family with BH, BY, or Holm and reports adjusted values, decisions, and the largest rejected raw p-value. It does not calculate or validate the input p-values. Caller must supply the preregistered alpha. | Statistical utility |

## Capabilities not implemented

This checkout does not provide molecular docking, SMILES parsing or generation,
conformer generation, retrosynthesis, validated binding-affinity prediction,
or a validated predictive ADMET model. The multiple-testing utility is not a
biomarker discovery pipeline: suitable study design, model fitting, covariate
handling, batch correction, and p-value generation remain external.

BH's usual FDR guarantee assumes independent or suitably positively dependent
tests; BY is more conservative and accommodates arbitrary dependence. Holm
controls family-wise error. None of these methods repairs biased measurements,
post-selection p-values, confounding, or a misspecified study.

## Evidence required for stronger claims

Before calling a component predictive, validate it on independent, appropriately
licensed datasets with source and license provenance, locked data splits,
predeclared endpoints and metrics, strong baselines, calibration and uncertainty
analysis, and external replication. Molecular claims additionally require
validated structures and appropriate physics/chemistry baselines; clinical
claims require relevant clinical evidence and qualified review. Preserve
negative results and known failure domains. No output from these utilities
should be used alone to select a therapy, dose, trial, or patient.
