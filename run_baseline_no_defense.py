"""
Run baseline (no defense) evaluation: logs ASR, utility, latency, etc. for all prompts with no patches or defenses enabled.
"""
import os
import json
from runpatches import collectLiveMetrics, save_live_metrics

def main():
    # No patches active, no defense
    print("[Baseline] Running with NO defense (no patches, no guardrails)")
    metrics = collectLiveMetrics(interval=1)
    metrics['defense_mode'] = 'none'
    os.makedirs('deployment_history', exist_ok=True)
    fname = os.path.join('deployment_history', 'baseline_no_defense_metrics.json')
    with open(fname, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[Baseline] ASR-based metrics written to {fname}:")
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
