# AutoPatch-Lite New Experiments

This directory contains experiments for the remaining tasks (A-F) as specified in the requirements.

## Experiments Overview

### A. CEPG (Code-Evolution Provenance Graph) / Graph-Based Patch Anomaly Detector
**File:** `CEPG/cepg_experiment.py`

Implements a graph-based anomaly detector for patches using:
- **Node types:** patch, test, deployment, incident, model_version
- **Edge types:** patch-applied-to-service, patch-triggered-failure, supersedes, deployed-in-canary-X
- **Model:** Simple GNN-based anomaly scoring using neighborhood aggregation
- **Experiment:** Compares with vs without CEPG screening during rollout

**Outputs:**
- `results/cepg_comparison_table.csv`: Comparison table
- `results/cepg_case_study.txt`: Qualitative case study
- `results/cepg_experiment_summary.txt`: Summary

### B. Patch Survival Analysis (RQ3 - Adaptivity)
**File:** `survival_analysis/survival_experiment.py`

Analyzes patch survival under adaptive attacks:
- **Survival variable:** Number of adaptive attack rounds until first successful bypass
- **Comparison:** AutoPatch-Lite (full) vs AutoPatch-Weak (no RL refinement)
- **Statistical test:** Log-rank test for comparing survival distributions

**Outputs:**
- `results/survival_curve.png`: Survival curve plot
- `results/survival_data.csv`: Raw survival data
- `results/survival_logrank.txt`: Log-rank test results
- `results/survival_analysis_summary.txt`: Analysis summary

### C. Rollout Ablation (RQ4)
**File:** `rollout_ablation/rollout_experiment.py`

Compares deployment policies:
- **Canary vs Immediate Deployment:** Staged rollout vs immediate deployment
- **Risk Scorer Variants:** Multi-objective vs safety-only risk scorer

**Outputs:**
- `results/rollout_ablation_table.csv`: Ablation comparison table
- `results/rollout_ablation_detailed.csv`: Detailed per-patch results
- `results/rollout_ablation_summary.txt`: Analysis summary

### D. Code Vulnerability Benchmark
**File:** `code_vuln/code_vuln_experiment.py`

Demonstrates test-driven patching on classical code vulnerabilities:
- **Benchmark:** Juliet suite (CWE121 - Stack-Based Buffer Overflow)
- **Approach:** Apply AutoPatch-Lite policy-as-code patches
- **Metrics:** ASR before vs after patching, utility preservation

**Outputs:**
- `results/code_vuln_results.csv`: Per-testcase results
- `results/code_vuln_summary_table.csv`: Summary metrics
- `results/code_vuln_results_summary.txt`: Analysis summary

### E. Alignment Baseline
**File:** `alignment_baseline/alignment_experiment.py`

Compares alignment-only defenses with AutoPatch-Lite:
- **Baselines:** No defense, Alignment (constitutional prompting), AutoPatch-Lite
- **Dataset:** JailbreakBench harmful prompts
- **Metrics:** ASR comparison

**Outputs:**
- `results/alignment_baseline.csv`: ASR results for each baseline
- `results/alignment_baseline_summary.txt`: Analysis summary

### F. Benign Output Quality Metrics (RQ2)
**File:** `benign_quality/benign_quality_experiment.py`

Evaluates utility preservation using standard metrics:
- **Metrics:** BLEU, ROUGE (1, 2, L), LLM-judge
- **Dataset:** Benign evaluation data from deployment history or HumanEval

**Outputs:**
- `results/benign_quality.csv`: Per-example metrics
- `results/benign_quality_summary.csv`: Aggregate metrics
- `results/benign_quality_summary.txt`: Analysis summary

## Running Experiments

### Prerequisites

Install required dependencies:
```bash
pip install pandas numpy networkx lifelines matplotlib datasets rouge-score nltk sentence-transformers rapidfuzz pyyaml
```

Optional (for LLM features):
```bash
# Install ollama if using LLM-based features
# See: https://ollama.ai/
```

### Run All Experiments

```bash
cd /home/julien/Autopatchlite/new_experiments
python run_all_experiments.py
```

### Run Individual Experiments

```bash
# CEPG
cd CEPG
python cepg_experiment.py

# Survival Analysis
cd ../survival_analysis
python survival_experiment.py

# Rollout Ablation
cd ../rollout_ablation
python rollout_experiment.py

# Code Vulnerability
cd ../code_vuln
python code_vuln_experiment.py

# Alignment Baseline
cd ../alignment_baseline
python alignment_experiment.py

# Benign Quality
cd ../benign_quality
python benign_quality_experiment.py
```

## Data Dependencies

All experiments expect the following data files in the parent directory (`/home/julien/Autopatchlite/`):

- `patch_ranking_results.json`: Patch ranking and scoring data
- `deployment_history/patch_*_result.jsonl`: Deployment results for each patch
- `autopatch_candidates_with_synthetics.yaml`: Patch catalog with types and patterns

## Output Structure

All results are saved to `results/` directory:
```
results/
├── cepg_comparison_table.csv
├── cepg_case_study.txt
├── cepg_experiment_summary.txt
├── survival_curve.png
├── survival_data.csv
├── survival_logrank.txt
├── survival_analysis_summary.txt
├── rollout_ablation_table.csv
├── rollout_ablation_detailed.csv
├── rollout_ablation_summary.txt
├── code_vuln_results.csv
├── code_vuln_summary_table.csv
├── code_vuln_results_summary.txt
├── alignment_baseline.csv
├── alignment_baseline_summary.txt
├── benign_quality.csv
├── benign_quality_summary.csv
└── benign_quality_summary.txt
```

## Notes

- Experiments use real deployment data when available, with fallbacks for missing data
- Some experiments (e.g., code vulnerability) may require Juliet testcases - synthetic testcases are used if not available
- LLM-based features (ollama) are optional - experiments will use pattern matching fallbacks if unavailable
- All experiments are designed to be robust and handle missing dependencies gracefully

