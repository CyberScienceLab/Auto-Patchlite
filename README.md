# AutoPatch-Lite

Research code and retained experimental records for the manuscript:

**AutoPatch-Lite: Closed-Loop Boundary Hardening for LLMs via Test-Driven Patch Synthesis and Staged Rollouts**

Julien Serbanescu, Samaneh Shamshiri, Fattane Zarrinkalam, and Ali Dehghantanha

Corresponding author: Ali Dehghantanha ([adehghan@uoguelph.ca](mailto:adehghan@uoguelph.ca))

The manuscript has been prepared for *IEEE Transactions on Dependable and Secure Computing*. It has not been accepted or published.

> **Read this first.**
> - The commands below were taken from the scripts and checked by reading the code. **None of them were run while this repository was prepared for release.** We have not verified that they run end to end.
> - Almost every script **overwrites** the retained catalogs, rankings, deployment logs, or result tables, or appends to them. To rerun anything, use a **separate working copy** of this repository so the saved evidence stays intact.
> - All reported policy-violation and attack-success rates are **oracle-defined**: they come from the project's own detection and validation harness, not from audit-validated ground truth. Benign pass-through (a benign prompt that is not blocked) is **not** semantic task success.

## Overview

AutoPatch-Lite is an operational control plane for mitigating LLM boundary failures **without modifying model weights**. Mitigations are external, policy-as-code input patches (regex, substring, and embedding-similarity rules with `deny`-style actions). The closed loop has five stages:

1. **Failure harvesting.** Prompts that the triage policy lets through (the `ALLOW` action on known-malicious prompts) become failure exemplars and patch candidates.
2. **Patch synthesis.** Regex, keyword, and semantic patch candidates are generated with an LLM through DSPy and Ollama, then filtered by the triage policy.
3. **PatchEval gating.** Candidates are scored on failure-derived safety tests, benign utility and false-positive checks, and specificity, then ranked overall and per patch type.
4. **Staged rollout with rollback.** Top patches go through a canary fraction and a progressive traffic schedule. Metrics are checked against SLO budgets at each interval, and a patch is rolled back on the first violation.
5. **Provenance.** Deployment decisions, interval metrics, and per-patch outcomes are appended to `deployment_history/`.

## Repository map

### Core pipeline (repository root)

| File | Role |
|---|---|
| `inference.py` | Failure harvesting. Loads `COMPLETE_DATAMOD/`, the PPO checkpoint `DEMO.zip`, the ONNX policy `DEMO.onnx`, and `garak*.hitlog.jsonl`. Embeds prompts with `all-MiniLM-L6-v2` and picks one of ALLOW/BLOCK/REWRITE/ASK/ESCALATE. On ALLOW it adds an embedding patch candidate to `autopatch_candidates.yaml`. Writes `inference_resultsADV.json` and `inference_run_stats.json`. |
| `envtest.py` | Gymnasium environment `JailbreakDefenseEnv` (observation and reward definition), imported by `inference.py`. |
| `add_synthetic_patches.py` | Reads `autopatch_candidates.yaml` and garak prompts, synthesizes candidates with DSPy using `ollama/mistral`, filters them with `DEMO.onnx`, and writes `autopatch_candidates_with_synthetics.yaml` and `stats.txt`. |
| `rankcands.py` | PatchEval-style scoring and ranking of `autopatch_candidates_with_synthetics.yaml`. Writes `patch_ranking_results.json` and `patch_ranking_top5_by_type.json`. |
| `runpatches.py` | Staged deployment (canary, then progressive rollout, with rollback), live metric collection through Ollama, and logging to `deployment_history/` and `live_metrics.json`. Imported by many evaluation scripts. |
| `llm_inference.py` | Ollama chat wrapper (default model `mistral`). |
| `load_phi3_jsonl.py` | Loader for a garak report file (`phi3_allprompts.report.jsonl`). |
| `strongreject_asr.py` | Wrapper around StrongREJECT `string_matching`. |
| `req.txt` | Pinned Python dependency list. |

### Retained pipeline artifacts (repository root)

- **Checkpoints:** `DEMO.zip` (Stable-Baselines3 PPO; `PPO.load("DEMO")` adds the `.zip` extension), `DEMO.onnx` (policy with a 393-dimensional input), and `DEMO_GARAK.zip` (a second checkpoint that no retained script references).
- **Patch catalogs and rankings:** `autopatch_candidates.yaml`, `autopatch_candidates_with_synthetics.yaml`, `patch_ranking_results.json`, `patch_ranking_top5_by_type.json`.
- **Inference records:** `inference_resultsADV.json`, `inference_run_stats.json`, `synthetic_patch_results.json`, `stats.txt`, `matched_prompts.json`, `live_metrics.json`.
- **Attack prompts:** `garak_falcon.hitlog.jsonl`, `garak_jailbreak_subset.hitlog.jsonl`.
- **Garak collection helpers:** `hitlogcmd.txt`, `garakrun.sh`, `garakstuff.yml`, `garak_all_probes.txt`.
- **Other result files:** latency benchmark outputs (`patch_latency_benchmark*.csv/json`), J1–J4 artifacts (`j1_j2_baselines_extended.csv`, `j3_*.csv/json/txt`, `j4_*`), `metrics_sanity_note.txt`, `utility_decomposition_rule.txt`, `table2_d1_filled.tex`, and `otherdatasets.md` (a list of external dataset links).

**Unresolved alternative versions.** These files are kept as found. No script references them, and this release does not decide which version is authoritative:

`autopatch_candidates copy.yaml`, `autopatch_candidatesMark.yaml`, `autopatch_candidatesold.yaml`, `autopatch_candidatesSMALL.yaml`, `autopatch_candidatesTHING.yaml`, `autopatch_candidates_gold.yaml`, `autopatch_candidates_nonsynth.yaml`, `patch_ranking_resultsold.json`, `patch_ranking_resultsoldstuff.json`, `patch_ranking_resultsoriginal.json`, `patch_ranking_top5_by_typeoriginal.json`, `inference_resultsADV1.json`, `live_metricsold.json`, `deployment_history/old_results/`, `deployment_history/autopatchesold.yaml` (read by `run_component_ablation.py`), and `new_experiments/results_old/`.

### Evaluation and table scripts (repository root)

| Script | Writes |
|---|---|
| `run_baseline_no_defense.py`, `run_ablation_patch_types.py`, `run_campaign_full_defense.py` | `deployment_history/*_metrics.json` and `matched_prompts_*.json` |
| `run_save_matched_prompts_per_run.py <run_name>` | Copies `deployment_history/matched_prompts.json` to `matched_prompts_<run_name>.json` |
| `run_repeat_experiment.py <script.py> <num_runs> [run_name_prefix]` | Repeated runs with copied metrics |
| `eval_external_datasets.py` | `deployment_history/per_prompt_results_*.jsonl`, `external_*_metrics*.json`, `table_results/table3_asr_results.csv` |
| `eval_benign_patch_effect.py` | `deployment_history/benign_patch_eval_*` |
| `eval_table6_patchwise.py` | `table6/result_*.json`, `table6/table6_patch_eval_summary.csv` |
| `eval_table7_coverage.py`, `eval_table7_coverage_new.py` | `table7/table7_coverage_across_families.csv` |
| `run_component_ablation.py` | `table_results/component_ablation_results.csv` |
| `generate_patch_survival_table.py`, `generate_patch_survival_metrics.py` | `table_results/patch_survival_*.csv` |
| `benchmark_patch_latency.py` | `patch_latency_benchmark.json`, `patch_latency_benchmark.csv` |
| `aggregate_eval_tables.py` | `table_results/table3_asr_final.csv`, `table4_utility_final.csv`, `table5_latency_final.csv` |
| `fill_publication_tables.py` | `table_results/table4_utility_retention.csv`, `table6_latency_overhead.csv`, `table7_patch_performance.csv`, `table14_canary_catch.csv` |
| `generate_table2_d1_attempt_counts.py` | `table_results/table2_d1_attempt_counts.csv` |
| `generate_table8_canary_catch_rollback.py` | `table_results/table8_canary_catch_rollback.csv` |
| `generate_table6_attack_patch_sequence.py`, `generate_table7_coverage_across_families.py`, `table6/generate_canary_rollback_table.py` | Template tables. The per-patch and per-family metric files they look for are not in this snapshot, so they emit placeholder values such as `XX.X`, `R`, and `Yes/No`. |
| `generate_j4_plot.py`, `extract_j4_exact_counts.py`, `plot_j4_sanity.py`, `regenerate_j4_exact_plot.py`, `check_baseline_hygiene.py` | J4 figure data and plots, and baseline CSV checks |
| `run_full_publication_pipeline.sh`, `run_full_publication_pipeline_new_asr.sh`, `run_generate_all_tables.sh` | Historical shell drivers that chain the scripts above |

The table numbers in file and script names (for example `table3_*` and `table7_*`) are historical working labels. **They are not guaranteed to match the table numbering of the current manuscript.**

### Directories

| Directory | Contents |
|---|---|
| `COMPLETE_DATAMOD/` | Main prompt dataset in Hugging Face `save_to_disk` format (`data-00000-of-00001.arrow`, `dataset_info.json`, `state.json`). Loaded by `inference.py`, `rankcands.py`, and `runpatches.py`. |
| `deployment_history/` | Rollout provenance (`deployment_history.jsonl`, `interval_metrics.jsonl`, `live_metrics_history.jsonl`, `per_type_run_summaries.jsonl`, `patch_*_result.jsonl`), per-prompt evaluation logs, and metric JSON files read by the aggregation and analysis scripts. |
| `table_results/`, `table6/`, `table7/` | Saved table CSVs and per-patch result JSON files. |
| `new_experiments/` | Follow-up experiments, one per subfolder (`CEPG`, `survival_analysis`, `rollout_ablation`, `code_vuln`, `alignment_baseline`, `benign_quality`, `safety_labeling_audit`, `utility_decomposition`, `strong_baseline`), the J1–J4 script `j1_j2_j3_j4_unified_experiment.py`, runners, and saved outputs in `results/`. See `new_experiments/README.md`, `CHANGES.md`, and `J1_J2_J3_J4_IMPLEMENTATION_SUMMARY.md`. |
| `TDSC experiments/` | `generate_tdsc_results.py` and `guardrail_threshold_sweep_experiment.py`, which derive stress-test and trade-off artifacts from the saved logs. Outputs are in `results/`. See its `README.md`. |
| `final_artifacts/`, `send/` | Bundles of J1–J4 artifacts. `final_artifacts/README.md` contains the authors' own file-to-paper mapping, which was not re-verified against the current manuscript. The `send/j4_sanity_plot.png` and `send/j4_sanity_plot.pdf` files **differ** from the root `j4_sanity_plot.*` files. Both versions are kept. |
| `frontend/`, `api.py` | Optional demo dashboard: a Vite/React app generated with Lovable, plus a FastAPI backend. It reads `patch_ranking_results.json`, `autopatch_candidates.yaml`, and `live_metrics.json`. Parts of the UI use mock data (`frontend/src/lib/mockData.ts`). It is not used by any experiment, table, or figure script. |

## Environment

- **Python packages.** `req.txt` is a pinned, `pip freeze`-style list. It includes `git+https://github.com/dsbowen/strong_reject.git@main` (an unpinned branch), CUDA 12 `nvidia-*` wheels, `triton`, and `uvloop`, which suggests the original environment was Linux x86-64 with an NVIDIA GPU. The original virtual environment (not included) was created with Python 3.12.3, and bytecode caches for CPython 3.10 and 3.12 were also present. **Python version compatibility has not been confirmed.**
- **Packages imported but missing from `req.txt`:** `statsmodels` (`aggregate_eval_tables.py`), `lifelines` (`new_experiments/survival_analysis`), `seaborn` (`new_experiments/results/visualize_confusion_matrix.py`), and `rouge-score` (`new_experiments/benign_quality`). Install them separately if you need those scripts.
- **Ollama.** `req.txt` pins only the Python client (`ollama==0.5.1`). The Ollama server/runtime is a separate installation, and its version was not recorded. Models referenced by name: `mistral` (generation, DSPy synthesis, and the independent judge), `llama-guard3:8b` (in `new_experiments/alignment_baseline` and `rollout_ablation`), and `phi3:mini` (the garak target in `hitlogcmd.txt` and `garakrun.sh`). Model digests, quantizations, context lengths, and decoding settings were not recorded.

An untested setup sketch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r req.txt
# Only needed for the scripts listed above:
pip install statsmodels lifelines seaborn rouge-score
# Install and start the Ollama server separately, then pull the models you need, e.g.:
ollama pull mistral
```

## Required data, checkpoints, and external services

- **Included:** `COMPLETE_DATAMOD/`, `DEMO.zip`, `DEMO.onnx`, both garak hitlogs, the patch catalogs, and the deployment logs.
- **`COMPLETE_DATAMOD/` provenance.** No preparation script or source description is included. Its features include JailbreakBench-style metadata, precomputed embeddings and heuristic scores, and fields named `simulated_latency_ms`, `simulated_rewrite_output`, `simulated_ask_output`, and `simulated_escalate_output`. The `license` field in `dataset_info.json` is empty, and redistribution terms have not been confirmed.
- **No training or export code for the checkpoints.** The repository contains no script that trains the PPO policy or exports `DEMO.onnx`.
- **Not included, needed for some workflows:**
  - `phi3_allprompts.report.jsonl`, a garak report that was excluded by the original `.gitignore`. `inference.py` prints a warning and continues without it, so the original inference prompt set cannot be reconstructed exactly.
  - The StrongREJECT phrase list. `runpatches.py` hardcodes `/home/julien/Volunteer/strong_reject/strong_reject/eval_files/string_matching.txt`, and every script that collects live metrics through `runpatches.collectLiveMetrics` needs this file at that path.
  - `benchmark_prompts.jsonl`. Without it, `benchmark_patch_latency.py` falls back to dummy prompts (`"Test prompt i"`).
  - The NIST Juliet C/C++ test suite. `new_experiments/code_vuln/code_vuln_experiment.py` reads `C/testcases/CWE121_Stack_Based_Buffer_Overflow/s01` relative to `new_experiments/code_vuln/`, and `unzip.py` expects `Juliet_Test_Suite_v1.3_for_C_Cpp.zip`. Obtain the suite from its official source.
- **Downloaded at runtime from Hugging Face:**
  - Datasets: `JailbreakBench/JBB-Behaviors`, `TrustAIRLab/in-the-wild-jailbreak-prompts`, `microsoft/llmail-inject-challenge`, `aurora-m/redteam` (written to output files under the name "CyberLLMInstruct"), `guychuk/benign-malicious-prompt-classification`, `VH1213141516/benign_data_v1`, `JailbreakV-28K/JailBreakV-28k`, and `CyberNative/Code_Vulnerability_Security_DPO`.
  - Models: `sentence-transformers` `all-MiniLM-L6-v2`, and `meta-llama/Prompt-Guard-86M`, which is gated and requires accepting its license on Hugging Face. Dataset and model revisions were not pinned.

## Main workflow (historical commands)

Run these from the repository root unless noted, **in a separate working copy**.

1. **Optional: collect garak attack prompts.** The commands in `hitlogcmd.txt` require garak and Ollama with `phi3:mini`. `garakrun.sh` writes `phi3_all_prompts.report.jsonl`, while `inference.py` looks for `phi3_allprompts.report.jsonl`. `garakrun.sh` also has `OUTDIR=""`, which must be set before it can find per-probe reports.
2. **Harvest failures:** `python inference.py`. This rewrites `autopatch_candidates.yaml`, `inference_resultsADV.json`, `inference_run_stats.json`, and `synthetic_patch_results.json`.
3. **Synthesize patches:** `python add_synthetic_patches.py`. Requires Ollama with `mistral`. Overwrites `autopatch_candidates_with_synthetics.yaml` and `stats.txt`.
4. **Rank patches (PatchEval):** `python rankcands.py`. Overwrites `patch_ranking_results.json` and `patch_ranking_top5_by_type.json`.
5. **Staged rollout:** `python runpatches.py`. Requires Ollama with `mistral` and the StrongREJECT phrase file. Rewrites `autopatch_candidates_with_synthetics.yaml` when a patch fully deploys, overwrites `live_metrics.json` and `deployment_history/matched_prompts.json`, and appends to the other `deployment_history/` logs. The rollout parameters (canary fraction, step schedule, SLO budget, number of intervals, `use_per_type`) are set in its `__main__` block.

This order follows each script's file inputs and outputs. The original run order and dates are not documented beyond the timestamps inside the saved files.

**Evaluation drivers:**

- `bash run_full_publication_pipeline.sh` (or `run_full_publication_pipeline_new_asr.sh`) runs the baseline, ablation, campaign, external-dataset evaluation, and aggregation. It overwrites `deployment_history/` metrics and `table_results/`.
- `new_experiments/`: run each experiment from inside its own folder, because the scripts use `../../` and `../results` relative paths. For example: `cd new_experiments/strong_baseline && python strong_baseline_experiment.py`. The runners `python run_all_experiments.py` and `python run_part_b_experiments.py` (run from `new_experiments/`) launch subsets of the experiments.
- `python new_experiments/j1_j2_j3_j4_unified_experiment.py [--n-harmful 300] [--n-benign 150] [--j3-dataset main|tool|strict]` resolves its paths from its own location and writes to `new_experiments/results/`.
- `python "TDSC experiments/generate_tdsc_results.py"` and `python "TDSC experiments/guardrail_threshold_sweep_experiment.py"` write to `TDSC experiments/results/`.

## Examining saved results versus rerunning

**Examining the evidence needs no execution.** The saved outputs are in `deployment_history/`, `table_results/`, `table6/`, `table7/`, `new_experiments/results/`, `TDSC experiments/results/`, and `final_artifacts/`, and they can be inspected directly.

**Offline post-processing scripts** read only saved logs and CSVs. Static inspection found no LLM calls in them. They still overwrite their output files.

- `aggregate_eval_tables.py` (needs `statsmodels`)
- `fill_publication_tables.py`
- `generate_table2_d1_attempt_counts.py [--source audit|table7] [--audit-csv PATH] [--table7-csv PATH] [--variants N] [--rounds N] [-o OUTPUT]`
- `generate_table8_canary_catch_rollback.py`
- `TDSC experiments/generate_tdsc_results.py`
- `new_experiments/CEPG/cepg_experiment.py` (run from its folder)
- `new_experiments/j1_j2_j3_j4_unified_experiment.py`
- `new_experiments/results/create_grouped_benign_quality.py` (run from `new_experiments/results/`) and `new_experiments/results/visualize_confusion_matrix.py` (needs `seaborn`)

`TDSC experiments/guardrail_threshold_sweep_experiment.py` also works from saved logs, but it downloads the embedding model.

`extract_j4_exact_counts.py` and `check_baseline_hygiene.py` hardcode `/home/julien/Autopatchlite` paths, so they will not run elsewhere unchanged.

**Rerunning inference or experiments** additionally needs Ollama, Hugging Face downloads, the missing inputs listed above, and in some cases a GPU. LLM outputs are not deterministic across runtimes or model versions, so reruns should not be expected to reproduce the saved numbers exactly.

A complete mapping from scripts to every final manuscript table and figure is **not** provided here, and no single command reproduces all results.

## Known limitations and caveats

- **Oracle-defined metrics.** Attack success and policy violations in `runpatches.py` are decided by StrongREJECT string matching plus response-length and gibberish heuristics. Other scripts use refusal-keyword heuristics or an Ollama judge. The retained independent-labeling audit (`new_experiments/results/table_s1_*`, `safety_labeling_audit_summary.txt`) reports low agreement with these oracle labels.
- **Simulated or heuristic components. Do not read these as measured production results:**
  - The REWRITE and ESCALATE actions in `inference.py` return placeholder strings (`[SIMULATED REGENERATION]`, `[Simulated human moderator review]`).
  - `rankcands.py` uses a simulated benign diff (`"ok"`/`"BLOCKED"`), constant latency values in `timedRun`, and `FAST_MODE = True`. It also replaces the first patch's `prompt` with the first safety test prompt in memory before scoring ("Force a match for the first patch for demonstration").
  - `envtest.py` and `COMPLETE_DATAMOD` use simulated latency and output fields.
  - Rollout intervals in `runpatches.py` are accelerated (`time.sleep(5)`), not real-time traffic windows.
  - `new_experiments/survival_analysis` simulates adaptive-attack rounds. Its seed is `hash(patch_id)`, which changes between Python processes, so its outputs are not reproducible run to run.
  - `new_experiments/rollout_ablation` uses `simulate_*` deployment functions.
  - The TDSC failure-intake poisoning and patch-interference results are simulations.
  - `new_experiments/code_vuln` uses a keyword "unsafe pattern" oracle and a fixed utility of 1.0.
  - `plot_j4_sanity.py` hardcodes estimated "No Defense" bar values, and `regenerate_j4_exact_plot.py` hardcodes its plotted values.
  - Several `new_experiments` scripts fall back to heuristics or synthetic responses when Ollama or a dataset is unavailable, and the saved outputs do not always record whether a fallback was used.
- **Synthetic patch generation inside `inference.py` produces no patches.** Its generator has no `generate_synthetic_patches` method; the retained `synthetic_patch_results.json` reports 0 generated. Synthetic candidates come from `add_synthetic_patches.py`.
- **Script and result versions do not always match.**
  - `eval_table6_patchwise.py` currently has its JailBreakV-28k and CodeVuln datasets commented out, while `table6/` contains results for them.
  - Several alternative catalogs, rankings, and plots are kept unresolved (listed above).
- **Non-determinism and missing pins:** UUID-based patch IDs, timestamps, LLM sampling, unpinned Hugging Face revisions, the `strong_reject@main` dependency, and unrecorded Ollama model digests.
- **Content warning.** Prompt and response logs contain jailbreak prompts and model outputs that may be harmful or offensive. Some third-party prompt records contain email addresses.

## Items needing maintainer confirmation before public release

- Choice of a software license. **No license file is included, and none has been selected.**
- Redistribution terms for `COMPLETE_DATAMOD/`, the garak hitlogs, and the third-party prompts and responses stored in `deployment_history/` and the other result files.
- Whether the Juliet CWE-121 slice, `phi3_allprompts.report.jsonl`, the StrongREJECT phrase file, and `benchmark_prompts.jsonl` should be documented further or added.
- Which of the retained alternative result versions correspond to the manuscript.
- Whether to keep the optional `frontend/` dashboard. Its `frontend/README.md` links to a Lovable project.

## Citation

If you use this code, please cite the manuscript by its title and author list above. Venue details will be added if and when the paper is published.
