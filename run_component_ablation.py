import os
import json
import pandas as pd
from runpatches import patch_catalog
import random
import yaml
from llm_inference import call_llm_ollama
from strongreject_asr import is_harmful_response_strongreject
from sentence_transformers import SentenceTransformer, util

# Ablation configurations
ABLATIONS = [
    {"name": "Baseline (No Defense)", "patches": []},
    {"name": "No RL Refinement", "patches": "no_rl"},
    {"name": "No Pattern Mining", "patches": "no_mining"},
    {"name": "No PatchEval", "patches": "no_patcheval"},
    {"name": "Full AutoPatch-Lite", "patches": "full"},
]



# Load prompts using the same logic as eval_external_datasets.py
EXTERNAL_DATASETS = [
    {
        'name': 'JailbreakBench',
        'hf_path': 'JailbreakBench/JBB-Behaviors',
        'configs': ['behaviors'],
        'splits': ['harmful', 'benign'],
        'prompt_field': 'prompt',
        'label_field': 'label',
    },
    {
        'name': 'CyberLLMInstruct',
        'hf_path': 'aurora-m/redteam',
        'splits': ['train'],
        'prompt_field': 'text',
        'label_field': 'label',
    },
    {
        'name': 'LLMail-Inject',
        'hf_path': 'microsoft/llmail-inject-challenge',
        'splits': ['Phase1', 'Phase2'],
        'prompt_field': 'body',
        'label_field': 'label',
    },
    {
        'name': 'InTheWild',
        'hf_path': 'TrustAIRLab/in-the-wild-jailbreak-prompts',
        'configs': ['jailbreak_2023_05_07', 'jailbreak_2023_12_25', 'regular_2023_05_07', 'regular_2023_12_25'],
        'splits': ['train'],
        'prompt_field': 'prompt',
        'label_field': 'label',
    },
]




# JailbreakBench uses 'Goal' as the prompt field for harmful split
def load_harmful_prompts():
    from datasets import load_dataset
    prompts = []
    ds_info = EXTERNAL_DATASETS[0]  # Only use the first dataset (JailbreakBench)
    for config in ds_info.get('configs', []):
        try:
            ds = load_dataset(ds_info['hf_path'], config, split='harmful')
            for ex in ds:
                prompt = ex.get('Goal')
                if prompt:
                    prompts.append({'prompt': prompt, 'label': 1})
        except Exception as e:
            print(f"[WARN] Could not load {ds_info['name']} config {config} split harmful: {e}")
    print(f"[INFO] Loaded {len(prompts)} harmful prompts from first dataset: {ds_info['name']}.")
    return prompts[:1000]  # Limit to 1000 for speed

PROMPT_LABELS = load_harmful_prompts()

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


# Load top 50 patches by score from patch_ranking_results.json
def get_top_50_patches():
    try:
        with open('patch_ranking_results.json') as f:
            patch_ranking = json.load(f)
        top_patch_ids = [p['id'] for p in sorted(patch_ranking, key=lambda x: -x['score'])[:50]]
        top_patches = [p for p in patch_catalog if p.get('id') in top_patch_ids]
        return top_patches
    except Exception as e:
        print(f"[WARN] Could not load top 50 patches: {e}")
        return []

# Load random patches from autopatchesold.yaml for 'no_rl' ablation
def get_random_old_patches(n=50):
    try:
        with open('deployment_history/autopatchesold.yaml') as f:
            old_patches = yaml.safe_load(f)
        if isinstance(old_patches, dict):
            old_patches = list(old_patches.values())
        elif not isinstance(old_patches, list):
            old_patches = []
        sampled = random.sample(old_patches, min(n, len(old_patches)))
        return sampled
    except Exception as e:
        print(f"[WARN] Could not load or sample old patches: {e}")
        return []

def get_patch_set(ablation):
    if ablation == "no_rl":
        # Use random patches from autopatchesold.yaml
        return get_random_old_patches(50)
    else:
        # Use top 50 patches for all other ablations
        top_patches = get_top_50_patches()
        if ablation == "no_mining":
            return [p for p in top_patches if p.get('type') == 'substring']
        elif ablation == "no_patcheval":
            return [p for p in top_patches if not p.get('patcheval', False)]
        elif ablation == "full":
            return [p for p in top_patches if p.get('active', True)]
        else:
            return []

def patch_blocks(patch, prompt):
    # Matching logic (same as eval_external_datasets)
    if patch.get('type') == 'regex' and patch.get('pattern'):
        import re
        try:
            if re.search(patch['pattern'], prompt, re.IGNORECASE):
                return True
        except Exception:
            pass
    elif patch.get('type') == 'substring' and patch.get('pattern'):
        if patch['pattern'].lower() in prompt.lower():
            return True
    elif patch.get('type') == 'semantic' and patch.get('pattern'):
        try:
            prompt_emb = semantic_model.encode(prompt, convert_to_tensor=True)
            patch_emb = patch.get('embedding')
            if patch_emb is not None:
                import torch
                patch_emb_tensor = torch.tensor(patch_emb)
                sim = util.pytorch_cos_sim(prompt_emb, patch_emb_tensor).item()
                threshold = patch.get('semantic_threshold', 0.7)
                if sim >= threshold:
                    return True
        except Exception:
            pass
    return False

def evaluate_ablation(patches, prompt_labels):
    asr_total = 0
    coverage_total = 0
    for rec in prompt_labels:
        prompt = rec['prompt']
        label = rec.get('label', 1)
        blocked = False
        for patch in patches:
            if patch_blocks(patch, prompt):
                blocked = True
                break
        if blocked:
            coverage_total += 1
        else:
            llm_response = call_llm_ollama(prompt, model='mistral')
            if label == 1 and is_harmful_response_strongreject(prompt, llm_response):
                asr_total += 1
    N = len(prompt_labels)
    asr = asr_total / N if N else 0.0
    coverage = coverage_total / N if N else 0.0
    return asr, coverage


def main():
    results = []
    for ablation in ABLATIONS:
        print(f"[INFO] Running ablation: {ablation['name']}")
        patches = get_patch_set(ablation['patches'])
        asr, coverage = evaluate_ablation(patches, PROMPT_LABELS)
        results.append({
            "Ablation": ablation['name'],
            "ASR": asr,
            "Coverage": coverage,
            "Num Patches": len(patches)
        })
        print(f"[RESULT] {ablation['name']}: ASR={asr}, Coverage={coverage}, NumPatches={len(patches)}")
    df = pd.DataFrame(results)
    os.makedirs("table_results", exist_ok=True)
    df.to_csv("table_results/component_ablation_results.csv", index=False)
    print("Component ablation results written to table_results/component_ablation_results.csv")

if __name__ == "__main__":
    main()