import os
import json
import time
import torch
from tqdm import tqdm
from collections import Counter
import numpy as np
from datasets import load_dataset
from runpatches import collectLiveMetrics, save_live_metrics, PatchCatalog, activatePatch, deactivatePatch
from runpatches import patch_catalog as all_patches
from sentence_transformers import SentenceTransformer, util
from llm_inference import call_llm_ollama
from strongreject_asr import is_harmful_response_strongreject
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load sentence transformer model for semantic matching (global, so only loaded once)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Patch IDs and types from the pasted image
PATCHES_TO_TEST = [
    {"id": "e797d27a", "type": "semantic"},
    {"id": "e61ba3b3", "type": "semantic"},
    {"id": "0cd819e5", "type": "substring"},
    {"id": "3801858c", "type": "regex"},
]

# Datasets in the requested order
DATASETS = [
    {
        'name': 'JailbreakBench',
        'hf_path': 'JailbreakBench/JBB-Behaviors',
        'prompt_field': 'Goal',  # Use correct field for adversarial prompts
        'label_field': 'label',
        'split': 'harmful',
        'config': 'behaviors'
    },
    # {
    #     'name': 'JailBreakV-28k-hidden',
    #     'hf_path': 'JailbreakV-28K/JailBreakV-28k',
    #     'prompt_field': 'jailbreak_query',
    #     'label_field': None,
    #     'split': 'JailBreakV_28K',
    #     'config': "JailBreakV_28K"
    # },
    # {
    #     'name': 'CodeVuln',
    #     'hf_path': 'CyberNative/Code_Vulnerability_Security_DPO',
    #     'prompt_field': 'question',
    #     'label_field': None,
    #     'split': 'train',
    #     'config': None
    # },
    # {
    #     'name': 'JailBreakV-28k-redteam',
    #     'hf_path': 'JailbreakV-28K/JailBreakV-28k',
    #     'prompt_field': 'redteam_query',
    #     'label_field': None,
    #     'split': 'JailBreakV_28K',
    #     'config': "JailBreakV_28K"
    # },
]

os.makedirs('table6', exist_ok=True)

# Helper to get patch object by id
patch_dict = {p.get('id'): p for p in all_patches}
def get_patch_by_id(pid):
    return patch_dict.get(pid)

def evaluate_patch_on_dataset(patch, ds_info):
    # Load dataset
    if ds_info['config']:
        ds = load_dataset(ds_info['hf_path'], ds_info['config'], split=ds_info['split'])
    else:
        ds = load_dataset(ds_info['hf_path'], split=ds_info['split'])
    prompts = []
    for ex in ds:
        prompt = ex.get(ds_info['prompt_field'])
        if prompt:
            prompts.append({'prompt': prompt, 'label': 1})
    prompts = prompts[:100]  # Limit for speed
    # ASR before (no defense)
    asr_before, ttm_before = run_asr(prompts, patch=None)
    # ASR after (with patch)
    asr_after, ttm_after = run_asr(prompts, patch=patch)
    # Save results
    result = {
        'dataset': ds_info['name'],
        'patch_id': patch.get('id'),
        'patch_type': patch.get('type'),
        'asr_before': asr_before,
        'asr_after': asr_after,
        'ttm_before': ttm_before,
        'ttm_after': ttm_after,
    }
    fname = f"table6/result_{ds_info['name']}_{patch.get('id')}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Table6] Saved: {fname}")
    return result

def run_asr(prompts, patch=None):
    harmful = 0
    total = 0
    total_latency = 0.0
    for r in tqdm(prompts, desc=f"{'Patched' if patch else 'No defense'}", ncols=80):
        prompt = r['prompt']
        t0 = time.time()
        blocked = False
        if patch:
            # Simulate patch block
            if patch['type'] == 'regex' and patch.get('pattern'):
                import re
                try:
                    if re.search(patch['pattern'], prompt, re.IGNORECASE):
                        blocked = True
                except Exception:
                    pass
            elif patch['type'] == 'substring' and patch.get('pattern'):
                if patch['pattern'].lower() in prompt.lower():
                    blocked = True
            elif patch['type'] == 'semantic' and patch.get('embedding'):
                try:
                    prompt_emb = semantic_model.encode(prompt, convert_to_tensor=True)
                    import torch
                    patch_emb_tensor = torch.tensor(patch['embedding'])
                    sim = util.pytorch_cos_sim(prompt_emb, patch_emb_tensor).item()
                    threshold = patch.get('semantic_threshold', 0.7)
                    if sim >= threshold:
                        blocked = True
                except Exception:
                    pass
        if blocked:
            is_harmful = False
        else:
            llm_response = call_llm_ollama(prompt, model='mistral')
            is_harmful = is_harmful_response_strongreject(prompt, llm_response)
        latency = time.time() - t0
        total_latency += latency
        if is_harmful:
            harmful += 1
        total += 1
    asr = harmful / total if total else 0.0
    ttm = total_latency / total if total else 0.0
    return asr, ttm

def main():
    results = []
    ds_info = DATASETS[0]  # Only JailbreakBench
    for patch_info in PATCHES_TO_TEST:
        patch = get_patch_by_id(patch_info['id'])
        if not patch:
            print(f"[WARN] Patch {patch_info['id']} not found in patch_catalog.")
            continue
        # Add type if missing
        if 'type' not in patch:
            patch['type'] = patch_info['type']
        result = evaluate_patch_on_dataset(patch, ds_info)
        results.append(result)
    # Save summary CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('table6/table6_patch_eval_summary.csv', index=False)
    print('[Table6] Summary CSV written to table6/table6_patch_eval_summary.csv')

if __name__ == '__main__':
    main()
