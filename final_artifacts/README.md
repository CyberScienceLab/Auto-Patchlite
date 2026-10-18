# Final Artifacts for AutoPatch-Lite Publication

## Files & Paper Mappings

- **j1_j2_baselines_extended.csv** → Paper Table 1 & 2: Baseline defense comparison. Raw counts columns show: harmful_total/violations/blocked/other + benign_total/passed/blocked. ASR, Safety_Pass_Rate, and overhead metrics are fully populated for all baselines including AutoPatch-Lite.

- **j3_targeted_mitigation.csv** → Paper Table 3: Targeted mitigation effectiveness (Tool-argument constraint & RAG context sanitization). Shows ASR, Safety_Pass_Rate, Benign_Pass_Rate, mitigation type, and placement point. All raw counts are complete for both mitigation rows evaluated on their respective slices.

- **j3_trigger_coverage.json** → Paper Section J3 (Targeted Mitigation Analysis): Trigger coverage statistics for tool-misuse and RAG-injection slices. Reports apply_rate per slice and categorizes non-application mismatches (SCOPE_MISMATCH, MATCH_FALSE, TOOL_NAME_MISMATCH, ARG_SCHEMA_NOT_PRESENT, RETRIEVAL_TEXT_EMPTY, etc.).

- **j4_sanity_plot.png** → Paper Figure 4: Visual sanity check of targeted mitigation effectiveness. Shows No Defense vs. mitigation ASR and benign pass rates for Tool and RAG slices. Includes exact raw count labels on all bars for reproducibility.

- **j4_writing_tables.json** → Paper Section J4 (Dependability/Deployability): Quantitative metrics for rollback success, poisoning admission blocking, time-to-safe-state, and operational overhead (avg/p95 in ms).

- **metrics_sanity_note.txt** → Appendix: Comprehensive metrics documentation. Defines formulas for ASR, Safety_Pass_Rate, benign_pass_rate, and harmful_other. Lists exact per-row raw counts for all baselines and targeted mitigations. Includes reviewer checklist for metric validation.

- **utility_decomposition_rule.txt** → Appendix: Utility preservation rule decomposition for AutoPatch-Lite. Specifies tool-argument constraint scope and RAG-sanitization blocking thresholds to maintain benign utility while maximizing harm mitigation.

---

**Status**: All metrics are frozen, raw counts are verified for consistency, and J4 plot regenerated with exact values. Ready for publication.
