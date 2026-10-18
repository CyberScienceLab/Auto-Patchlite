# TDSC Additional Experiment Summary

This package generates oracle-defined stress tests and tradeoff analyses requested for dependable-systems framing.

## Key outcomes
- Operating-point sweep generated in `results/operating_point_sweep.csv` and `results/operating_point_tradeoff.png`.
- AutoPatch-Lite operating point (oracle-defined): refusal=0.305, ASRpolicy=0.100, benign_correctness=0.630.
- Deny-only matched-refusal baseline: refusal=0.220, ASRpolicy=0.410, benign_correctness=0.190.
- Oracle trust stress-check added with per-component firing frequencies (`results/oracle_trust_stress_check.csv`).
- Failure-intake poisoning simulation: blocked_pct=1.000, deployed_after_gates=0.
- Patch interference simulation produced superlinear penalty and rollback-gate triggers (`results/patch_interference_test.csv`).
- Rollback sanity table confirms rollback behavior from canary-stage failures (`results/rollback_sanity_table.csv`).

## Audit limitation language
Observed independent-labeling κ≈0.064. κ near 0 implies substantial label divergence; therefore safety outcomes should be treated as oracle-defined rather than audit-validated ground truth.

## Oracle component firing
- classifier: fires=33 (0.165)
- regex: fires=149 (0.745)
- schema: fires=0 (0.000)
- sandbox: fires=17 (0.085)

## Notes
All metrics in this folder are oracle-defined and derived from existing run logs; they are intended for transparent tradeoff characterization and stress-checking, not claims of external ground truth.
