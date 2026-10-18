"""
Run ablation study: evaluate each patch type separately (e.g., only regex, only embedding, etc.).
"""
import os
import json
from runpatches import collectLiveMetrics, save_live_metrics, patch_catalog, PatchCatalog

from runpatches import activatePatch, deactivatePatch, active_patches
def run_ablation_for_patch_type(patch_type):
    # Deactivate all patches first (and update active_patches set)
    for patch in patch_catalog:
        deactivatePatch(patch)
    # Activate only patches of the given type (and update active_patches set)
    for patch in patch_catalog:
        if patch.get('type') == patch_type:
            activatePatch(patch, traffic_fraction=1.0)
    print(f"[Ablation] Running with only {patch_type} patches active.")
    metrics = collectLiveMetrics(interval=1)
    metrics['defense_mode'] = 'autopatch_lite'
    metrics['ablation'] = patch_type
    os.makedirs('deployment_history', exist_ok=True)
    fname = os.path.join('deployment_history', f'ablation_{patch_type}_metrics.json')
    with open(fname, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[Ablation] ASR-based metrics written to {fname}:")
    print(json.dumps(metrics, indent=2))

def main():
    patch_types = set(p.get('type') for p in patch_catalog if p.get('type'))
    for patch_type in patch_types:
        run_ablation_for_patch_type(patch_type)

if __name__ == '__main__':
    main()
