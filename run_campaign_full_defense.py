"""
Run full campaign/production defense: deploy all patches (or top-K/top-N-per-type) and log metrics for publication tables.
Saves all outputs to deployment_history/.
"""
import os
import json
from runpatches import (
    deploy_top_k_patches,
    deploy_top_n_per_type,
    collectLiveMetrics,
    save_live_metrics
)

def main():
    # Choose deployment strategy
    use_per_type = True  # Set to False for overall top-K
    k = 5
    n_per_type = 5
    canary_frac = 0.1
    step_schedule = [0.1, 0.3, 0.6, 1.0]
    metric_budget = {
        'safety_pass_rate': 0.95,
        'util_loss': 0.02,
        'fp_rate': 0.01,
        'latency': 0.15
    }
    N = 10
    os.makedirs('deployment_history', exist_ok=True)
    if use_per_type:
        print("[Campaign] Deploying top N per type...")
        deploy_top_n_per_type(n_per_type, canary_frac, step_schedule, metric_budget, N)
        mode = f'top_{n_per_type}_per_type'
    else:
        print("[Campaign] Deploying top K overall...")
        deploy_top_k_patches(k, canary_frac, step_schedule, metric_budget, N)
        mode = f'top_{k}_overall'
    # Collect and save final live metrics
    metrics = collectLiveMetrics(interval=1)
    metrics['defense_mode'] = mode
    os.makedirs('deployment_history', exist_ok=True)
    fname = os.path.join('deployment_history', f'campaign_{mode}_metrics.json')
    with open(fname, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[Campaign] ASR-based metrics written to {fname}:")
    print(json.dumps(metrics, indent=2))
    # Save matched prompts for this run
    from shutil import copyfile
    src = os.path.join('deployment_history', 'matched_prompts.json')
    dst = os.path.join('deployment_history', f'matched_prompts_{mode}.json')
    if os.path.exists(src):
        copyfile(src, dst)
        print(f"[Campaign] Matched prompts saved to {dst}")

if __name__ == '__main__':
    main()
