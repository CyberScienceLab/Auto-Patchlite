import os
import json
import pandas as pd
from datasets import load_dataset
from runpatches import collectLiveMetrics, save_live_metrics, PatchCatalog, activatePatch, deactivatePatch
from runpatches import patch_catalog as all_patches
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Table 7: Coverage Across Attack Families
os.makedirs('table7', exist_ok=True)

ATTACK_FAMILIES = [
    {
        'name': 'Jailbreak / Prompt Suite',
        'hf_path': 'JailbreakBench/JBB-Behaviors',
        'prompt_field': 'Goal',
        'split': 'harmful',
        'config': 'behaviors'
    },
    {
        'name': 'Indirect Injection (RAG)',
        'hf_path': 'microsoft/llmail-inject-challenge',
        'prompt_field': 'body',
        'split': 'Phase1',
        'config': None
    },
    {
        'name': 'Malicious Tool Use (OWASP-inspired)',
        'hf_path': 'CyberNative/Code_Vulnerability_Security_DPO',
        'prompt_field': 'question',
        'split': 'train',
        'config': None
    },
    {
        'name': 'Retrieval Attack Data (hidden prompts)',
        'hf_path': 'JailbreakV-28K/JailBreakV-28k',
        'prompt_field': 'jailbreak_query',
        'split': 'JailBreakV_28K',
        'config': 'JailBreakV_28K'
    },
    {
        'name': 'Code Vulnerability Scenarios (ManyBugs/Juliet)',
        'hf_path': 'CyberNative/Code_Vulnerability_Security_DPO',
        'prompt_field': 'question',
        'split': 'train',
        'config': None
    },
]

def get_top_patches(n=10):
    with open('patch_ranking_results.json') as f:
        patch_ranking = json.load(f)
    top_patch_ids = [p['id'] for p in sorted(patch_ranking, key=lambda x: -x['score'])[:n]]
    return [p for p in all_patches if p.get('id') in top_patch_ids]

def autopatch_block(prompt, patch_catalog, semantic_model):
    for patch in patch_catalog:
        if not patch.get('active', True):
            continue
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
        elif patch.get('type') == 'fuzzy' and patch.get('pattern'):
            try:
                from rapidfuzz import fuzz
                score = fuzz.partial_ratio(patch['pattern'].lower(), prompt.lower())
                threshold = patch.get('fuzzy_threshold', 80)
                if score >= threshold:
                    return True
            except ImportError:
                pass
        elif patch.get('type') == 'semantic' and patch.get('embedding'):
            try:
                prompt_emb = semantic_model.encode(prompt, convert_to_tensor=True)
                import torch
                patch_emb_tensor = torch.tensor(patch['embedding'])
                sim = util.pytorch_cos_sim(prompt_emb, patch_emb_tensor).item()
                threshold = patch.get('semantic_threshold', 0.7)
                if sim >= threshold:
                    return True
            except Exception:
                pass
    return False

def evaluate_family(fam, patch_catalog, semantic_model):
    # Load dataset
    if fam['config']:
        ds = load_dataset(fam['hf_path'], fam['config'], split=fam['split'])
    else:
        ds = load_dataset(fam['hf_path'], split=fam['split'])
    prompts = []
    for ex in ds:
        prompt = ex.get(fam['prompt_field'])
        if prompt:
            prompts.append({'prompt': prompt, 'label': 1})
    n_attempts = len(prompts)
    n_mitigated = 0
    # Deactivate all patches first
    for patch in all_patches:
        deactivatePatch(patch)
    # Activate only top N patches
    for patch in patch_catalog:
        activatePatch(patch, traffic_fraction=1.0)
    for r in tqdm(prompts, desc=fam['name'], ncols=80):
        prompt = r['prompt']
        if autopatch_block(prompt, patch_catalog, semantic_model):
            n_mitigated += 1
    coverage = 100 * n_mitigated / n_attempts if n_attempts else 0
    residual_asr = 100 - coverage
    return {
        'Attack Family': fam['name'],
        'N (#Attempts)': n_attempts,
        '# Mitigated': n_mitigated,
        'Coverage (%)': coverage,
        'Residual ASR (%)': residual_asr
    }

def main():
    patch_catalog = get_top_patches(50)
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    results = []
    for fam in ATTACK_FAMILIES:
        res = evaluate_family(fam, patch_catalog, semantic_model)
        results.append(res)
    df = pd.DataFrame(results)
    os.makedirs('table7', exist_ok=True)
    df.to_csv('table7/table7_coverage_across_families.csv', index=False)
    print('[Table7] Coverage table written to table7/table7_coverage_across_families.csv')

if __name__ == '__main__':
    main()
