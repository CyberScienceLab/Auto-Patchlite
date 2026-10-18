# TDSC experiments

This folder adds reviewer-facing stress tests and tradeoff artifacts using existing AutoPatch-Lite logs.

## What this package covers

Experiment-backed requirements addressed here:
- Oracle trust stress-check (component firing distribution)
- Non-deny patch-family evidence (retrieval sanitization/schema-validation row)
- Operating-point sweep (refusal vs ASRpolicy vs benign correctness)
- Failure intake poisoning (harvester abuse simulation with admission + PatchEval gating)
- Patch interference/compositionality (incremental enablement + rollback gate trigger checks)
- Rollback sanity table (rollback restores baseline mode within one interval)
- Deny-only baseline matched on refusal
- Sanitize/validate-only baseline (no synthesis loop)
- Metrics + denominator definitions box

## Run

From `Autopatchlite`:

```bash
python "TDSC experiments/generate_tdsc_results.py"
```

## Outputs

Generated into `TDSC experiments/results/`:

- `operating_point_sweep.csv`
- `operating_point_tradeoff.png`
- `deny_only_matched_baseline.csv`
- `sanitize_validate_baseline.csv`
- `non_deny_patch_family_result.csv`
- `oracle_trust_stress_check.csv`
- `failure_intake_poisoning.csv`
- `patch_interference_test.csv`
- `rollback_sanity_table.csv`
- `tdsc_experiment_summary.md`

Additional main-body artifact:
- `metrics_denominators_box.md`

## Repro mapping (section/table/figure)

- Figure (operating-point sweep): `generate_tdsc_results.py` -> `results/operating_point_tradeoff.png`
- Deny-only matched-refusal baseline table row: `generate_tdsc_results.py` -> `results/deny_only_matched_baseline.csv`
- Sanitize/validate-only baseline table row: `generate_tdsc_results.py` -> `results/sanitize_validate_baseline.csv`
- Oracle trust stress-check table: `generate_tdsc_results.py` -> `results/oracle_trust_stress_check.csv`
- Failure intake poisoning table: `generate_tdsc_results.py` -> `results/failure_intake_poisoning.csv`
- Patch interference table: `generate_tdsc_results.py` -> `results/patch_interference_test.csv`
- Rollback sanity table: `generate_tdsc_results.py` -> `results/rollback_sanity_table.csv`
