# Changes Made to Experiments

This document summarizes the changes made to ensure all experiments run correctly and match the requirements.

## A. CEPG Experiment (`CEPG/cepg_experiment.py`)

### Changes:
1. **Complete rewrite** with proper CEPG definition:
   - All node types: patch, test, deployment, incident, model_version
   - All edge types: patch-applied-to-service, patch-triggered-failure, supersedes, deployed-in-canary-X
   - Node/edge features: family, risk_score, util_loss, ASR, policies_touched, timestamp

2. **GNN-based anomaly detector**:
   - Neighborhood aggregation for anomaly scoring
   - Considers patch risk, incident edges, and deployment failures

3. **Proper experiment design**:
   - With vs without CEPG screening comparison
   - Metrics: regressions, bad patches caught, ASR, utility
   - Case study generation for flagged patches

4. **Real data integration**:
   - Loads from `patch_ranking_results.json`
   - Uses actual deployment results from `deployment_history/`
   - Extracts patch types from YAML catalog

## B. Survival Analysis (`survival_analysis/survival_experiment.py`)

### Changes:
1. **Proper survival analysis**:
   - Survival variable: rounds until bypass (right-censored if not bypassed)
   - Uses `lifelines` library for Kaplan-Meier estimation
   - Log-rank test for statistical comparison

2. **Two variants comparison**:
   - AutoPatch-Lite (full): Higher robustness
   - AutoPatch-Weak (no RL): Lower robustness, more bypasses

3. **Real data integration**:
   - Uses actual patch scores and regression status
   - Deterministic simulation based on patch characteristics

4. **Outputs**:
   - Survival curve plot (PNG)
   - Log-rank test results
   - Summary statistics

## C. Rollout Ablation (`rollout_ablation/rollout_experiment.py`)

### Changes:
1. **Three variants comparison**:
   - Immediate Deployment: No canary, deploy immediately
   - Canary Deployment: Staged rollout with monitoring
   - Risk Scorer (Safety Only): Conservative safety-only scoring

2. **Proper metrics**:
   - ASR, benign success, regressions, rollbacks, bad patches shipped
   - Aggregated across patches

3. **Real data integration**:
   - Uses actual deployment results
   - Extracts metrics from deployment history

4. **Ablation table**:
   - Clear comparison of all variants
   - Summary text for RQ4

## D. Code Vulnerability Benchmark (`code_vuln/code_vuln_experiment.py`)

### Changes:
1. **Juliet integration**:
   - Enumerates testcases from CWE121 (Stack-Based Buffer Overflow)
   - Falls back to synthetic testcases if Juliet not available

2. **Vulnerability detection**:
   - Pattern-based detection (gets, strcpy, sprintf, etc.)
   - Optional LLM-based detection (with fallback)

3. **AutoPatch-Lite patching**:
   - Replaces unsafe functions: gets→fgets, strcpy→strncpy, sprintf→snprintf
   - Policy-as-code approach

4. **Metrics**:
   - ASR before vs after
   - Utility preservation
   - Summary table and text

## E. Alignment Baseline (`alignment_baseline/alignment_experiment.py`)

### Changes:
1. **Three baselines**:
   - No Defense: Baseline LLM without safety
   - Alignment: Constitutional prompting only
   - AutoPatch-Lite: Full test-driven patching

2. **Robust implementation**:
   - Handles missing dependencies gracefully
   - Uses pattern matching fallbacks
   - Proper patch catalog loading

3. **JailbreakBench integration**:
   - Loads harmful prompts from dataset
   - Computes ASR for each baseline

4. **Outputs**:
   - Comparison table
   - Summary explaining where alignment helps vs fails

## F. Benign Quality Metrics (`benign_quality/benign_quality_experiment.py`)

### Changes:
1. **Multiple metrics**:
   - BLEU score (with NLTK or simplified fallback)
   - ROUGE (1, 2, L) scores
   - LLM-judge (optional, with fallback)

2. **Data sources**:
   - Deployment history benign eval results
   - HumanEval as fallback
   - Synthetic data as final fallback

3. **Aggregate metrics**:
   - Per-example and average scores
   - Summary table and text

4. **Robust error handling**:
   - Handles missing libraries gracefully
   - Provides fallback implementations

## General Improvements

1. **Error handling**: All experiments handle missing data/files gracefully
2. **Type hints**: Added type hints for better code clarity
3. **Documentation**: Added docstrings to all functions
4. **Output organization**: All results saved to `results/` directory
5. **Consistent structure**: All experiments follow similar patterns
6. **Real data integration**: Experiments use actual deployment data when available
7. **Fallbacks**: Experiments work even with missing dependencies/data

## Running the Experiments

See `README.md` for detailed instructions on running all experiments.

