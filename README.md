# AutoPatch-Lite

**AutoPatch-Lite: Closed-Loop Boundary Hardening for LLMs via Test-Driven Patch Synthesis and Staged Rollouts**

Julien Serbanescu, Samaneh Shamshiri, Fattane Zarrinkalam, and Ali Dehghantanha

## Overview

AutoPatch-Lite is a test-driven framework for mitigating policy violations at the boundaries of LLM applications without changing model weights. It turns observed failures into policy-as-code patch candidates, evaluates them with PatchEval-style safety and benign-behavior checks, and deploys selected patches through staged rollout with SLO-based rollback. Deployment outcomes and patch histories are recorded for inspection.

The evaluation reports policy-violation rates under a fixed operational oracle, alongside benign pass/block behavior and deployment-control measurements.

## Main components

| File or directory | Purpose |
|---|---|
| `inference.py` | Collect failure exemplars and initial patch candidates. |
| `add_synthetic_patches.py` | Generate additional patch candidates using DSPy and Ollama. |
| `rankcands.py` | Score and rank candidate patches. |
| `runpatches.py` | Apply selected patches using staged rollout, monitoring, and rollback. |
| `envtest.py`, `llm_inference.py` | RL environment and LLM inference support. |
| `eval_external_datasets.py`, `aggregate_eval_tables.py` | Benchmark evaluation and result aggregation. |
| `TDSC experiments/`, `new_experiments/` | Additional evaluation and analysis scripts. |
| `deployment_history/`, `table_results/`, `table6/`, `table7/` | Retained evaluation and deployment records. |
| `final_artifacts/` | Selected experiment artifacts and related documentation. |
| `COMPLETE_DATAMOD/`, `DEMO.zip`, `DEMO.onnx` | Dataset and saved policy checkpoints used by the pipeline. |

## Setup

The project uses Python, the dependencies listed in `req.txt`, and [Ollama](https://ollama.com/) for model inference. Some dependency packages are specific to the original Linux/GPU environment.

```bash
python -m venv .venv
# Activate the environment, then install dependencies:
pip install -r req.txt
ollama pull mistral
```

Make sure the Ollama server is running before using scripts that call the model. Some evaluation scripts use additional packages or external benchmark datasets; consult their imports and data paths before running them.

## Core workflow

Run the stages in order from the repository root, after configuring the inputs and local resources expected by the scripts:

```bash
python inference.py
python add_synthetic_patches.py
python rankcands.py
python runpatches.py
```

The pipeline uses the included saved policy checkpoints and patch catalogs. The rollout code also requires the StrongREJECT phrase resource referenced in `runpatches.py`.

**Run new experiments in a separate working copy:** pipeline and evaluation scripts may replace saved catalogs, rankings, and output logs.

## Evaluation material

Saved execution records, patch rankings, and evaluation outputs are organized under `deployment_history/`, `table_results/`, `final_artifacts/`, and the experiment directories. See the corresponding scripts and directory READMEs for the analyses associated with each output. Additional benchmark inputs may be required for particular evaluation workflows.

## Citation

If you use this repository, please cite the associated *AutoPatch-Lite* manuscript listed above.
