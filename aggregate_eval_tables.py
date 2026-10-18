"""
Script: aggregate_eval_tables.py
Aggregates results from deployment_history/ to produce CSVs matching Tables 3, 4, and 5 for publication.
"""
import os
import glob
import json
import pandas as pd
import numpy as np

def wilson_ci(successes, n, alpha=0.05):
    # Returns (lower, upper) Wilson score interval for a binomial proportion
    if n == 0:
        return (0.0, 0.0)
    from statsmodels.stats.proportion import proportion_confint
    return proportion_confint(successes, n, alpha=alpha, method='wilson')

def aggregate_table3():
    # Table 3: ASR by dataset family and overall
    metrics_files = glob.glob('deployment_history/external_*_metrics*.json')
    rows = []
    for fname in metrics_files:
        with open(fname) as f:
            d = json.load(f)
            # Only add if required keys exist
            if 'dataset' in d and 'defense_mode' in d:
                rows.append(d)
            else:
                print(f"[WARN] Skipping {fname}: missing 'dataset' or 'defense_mode' key.")
    if not rows:
        print("[ERROR] No valid metrics files found with required keys. Exiting Table 3 aggregation.")
        return
    df = pd.DataFrame(rows)
    # Remove all benign results (keep only harmful/adversarial prompts)
    if 'split' in df.columns:
        df = df[df['split'] != 'benign']
    # Group by dataset family and mode
    families = {
        'Jailbreak / Prompt Suite': ['JailbreakBench'],
        'Hidden Prompts': ['InTheWild'],
        'Indirect Injection (RAG docs)': ['LLMail-Inject'],
        'Malicious Tool Use (OWASP 50)': ['CyberLLMInstruct'],
        # Add more if needed
    }
    table = []
    for fam, dslist in families.items():
        for mode in ['none', 'promptguard', 'autopatch_lite']:
            fam_df = df[df['dataset'].isin(dslist) & (df['defense_mode'] == mode)]
            N = fam_df['N'].sum()
            # ASR = 1 - safety_pass_rate
            failures = ((1 - fam_df['safety_pass_rate']) * fam_df['N']).sum()
            asr = 100 * (failures / N) if N else 0.0
            ll, uu = wilson_ci(failures, N)
            table.append({
                'Dataset Family': fam,
                'N': int(N),
                'Mode': mode,
                'ASR': f"{asr:.1f} [{100*ll:.1f}, {100*uu:.1f}]"
            })
    # Overall row
    for mode in ['none', 'promptguard', 'autopatch_lite']:
        all_df = df[df['defense_mode'] == mode]
        N = all_df['N'].sum()
        failures = ((1 - all_df['safety_pass_rate']) * all_df['N']).sum()
        asr = 100 * (failures / N) if N else 0.0
        ll, uu = wilson_ci(failures, N)
        table.append({
            'Dataset Family': 'Overall',
            'N': int(N),
            'Mode': mode,
            'ASR': f"{asr:.1f} [{100*ll:.1f}, {100*uu:.1f}]"
        })
    df_out = pd.DataFrame(table)
    df_out.to_csv('table_results/table3_asr_final.csv', index=False)
    print('[Table3] Final ASR table written to table_results/table3_asr_final.csv')

def aggregate_table4():
    # Table 4: Utility Retention (benign only)
    metrics_files = glob.glob('deployment_history/external_*_metrics*.json')
    rows = []
    for fname in metrics_files:
        with open(fname) as f:
            d = json.load(f)
            if 'dataset' in d and 'defense_mode' in d and 'split' in d:
                rows.append(d)
            else:
                print(f"[WARN] Skipping {fname}: missing required keys for Table 4.")
    if not rows:
        print("[ERROR] No valid metrics files found with required keys. Exiting Table 4 aggregation.")
        return
    df = pd.DataFrame(rows)
    # Only benign splits/configs (keep benign for Table 4)
    df_benign = df[df['split'].isin(['benign']) | df['dataset'].eq('CyberLLMInstruct')]
    scenarios = {
        'General QA & Dialogue': ['JailbreakBench'],
        'RAG Knowledge Queries': ['LLMail-Inject'],
        'Traffic Replays': ['InTheWild'],
        'Malicious Tool Use': ['CyberLLMInstruct'],
    }
    table = []
    for scen, dslist in scenarios.items():
        for mode in ['none', 'promptguard', 'autopatch_lite']:
            scen_df = df_benign[df_benign['dataset'].isin(dslist) & (df_benign['defense_mode'] == mode)]
            N = scen_df['N'].sum()
            # Utility retention = 1 - util_loss
            util_ret = 100 * (1 - (scen_df['util_loss'] * scen_df['N']).sum() / N) if N else 0.0
            table.append({
                'Scenario': scen,
                'N': int(N),
                'Mode': mode,
                'Retention%': f"{util_ret:.1f}"
            })
    # Overall row
    for mode in ['none', 'promptguard', 'autopatch_lite']:
        all_df = df_benign[df_benign['defense_mode'] == mode]
        N = all_df['N'].sum()
        util_ret = 100 * (1 - (all_df['util_loss'] * all_df['N']).sum() / N) if N else 0.0
        table.append({
            'Scenario': 'Overall',
            'N': int(N),
            'Mode': mode,
            'Retention%': f"{util_ret:.1f}"
        })
    df_out = pd.DataFrame(table)
    df_out.to_csv('table_results/table4_utility_final.csv', index=False)
    print('[Table4] Final Utility table written to table_results/table4_utility_final.csv')

def aggregate_table5():
    # Table 5: Latency (now implemented using per-prompt results)
    import glob
    import pandas as pd
    per_prompt_files = glob.glob('deployment_history/per_prompt_results_*.jsonl')
    rows = []
    for fname in per_prompt_files:
        mode = 'unknown'
        dataset = 'unknown'
        split = 'unknown'
        config = None
        # Parse mode, dataset, split, config from filename
        parts = fname.split('per_prompt_results_')[-1].replace('.jsonl','').split('_')
        if len(parts) >= 3:
            dataset, split, mode = parts[:3]
            if len(parts) > 3:
                config = '_'.join(parts[3:])
        # Only keep harmful/adversarial splits (remove benign)
        if split == 'benign':
            continue
        latencies = []
        for line in open(fname, 'r', encoding='utf-8'):
            rec = json.loads(line)
            lat = rec.get('latency_ms', None)
            if lat is not None:
                latencies.append(lat)
        if latencies:
            avg_latency = sum(latencies) / len(latencies) / 1000.0
        else:
            avg_latency = 0.0
        rows.append({
            'Dataset': dataset,
            'Split': split,
            'Config': config,
            'Mode': mode,
            'N': len(latencies),
            'AvgLatency(s)': avg_latency
        })
    df = pd.DataFrame(rows)
    df.to_csv('table_results/table5_latency_final.csv', index=False)
    print('[Table5] Final Latency table written to table_results/table5_latency_final.csv')

def main():
    os.makedirs('table_results', exist_ok=True)
    aggregate_table3()
    aggregate_table4()
    aggregate_table5()

if __name__ == '__main__':
    main()
