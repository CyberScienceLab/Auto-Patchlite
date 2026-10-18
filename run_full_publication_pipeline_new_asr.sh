#!/bin/bash
# Full pipeline for publication-ready tables (Tables 3, 4, 5) with new ASR criteria and 1000 samples per evaluation
set -e

# 1. Baseline (no defense)
echo "[1/7] Running baseline (no defense)"
python run_baseline_no_defense.py

# 2. Ablation (regex, substring, semantic)
echo "[2/7] Running ablation patch types"
python run_ablation_patch_types.py

# 3. Campaign (full defense)
echo "[3/7] Running campaign full defense"
python run_campaign_full_defense.py

# 4. (Optional) Repeat for error bars (uncomment to use)
# echo "[4/7] Repeating experiments for error bars"
# python run_repeat_experiment.py run_baseline_no_defense.py 5 baseline_no_defense
# python run_repeat_experiment.py run_ablation_patch_types.py 5 ablation
# python run_repeat_experiment.py run_campaign_full_defense.py 5 top_5_per_type

# 5. Evaluate on external datasets
echo "[5/7] Evaluating on external datasets (1000 samples per dataset)"
python eval_external_datasets.py

# 6. Aggregate results for publication tables
echo "[6/7] Aggregating results for Tables 3, 4, 5"
python aggregate_eval_tables.py

# 7. Done
echo "[7/7] All tables written to table_results/"
echo "Check table_results/table3_asr_final.csv, table4_utility_final.csv, table5_latency_final.csv"
