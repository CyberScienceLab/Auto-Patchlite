# PromptGuard static filter (copied from eval_external_datasets.py)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
promptguard_model_id = "meta-llama/Prompt-Guard-86M"
promptguard_tokenizer = AutoTokenizer.from_pretrained(promptguard_model_id)
promptguard_model = AutoModelForSequenceClassification.from_pretrained(promptguard_model_id)
promptguard_action_names = ["BENIGN", "INJECTION", "JAILBREAK"]
def promptguard_filter(text_to_check):
    # Normalize any weird input
    if not isinstance(text_to_check, str):
        try:
            text_to_check = str(text_to_check)
        except Exception:
            return False  # skip entirely if it's too broken

    try:
        # Tokenize safely
        inputs = promptguard_tokenizer(
            [text_to_check],
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Model inference
        with torch.no_grad():
            outputs = promptguard_model(**inputs)
            preds = torch.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(preds, dim=-1).item()

        # 1 = harmful, 0 = benign (PromptGuard convention)
        return bool(pred_label == 1)

    except Exception as e:
        print(f"[PromptGuard Error] Skipping bad input: {e}")
        return False




import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch_device = "cpu"
import json
from tqdm import tqdm
from datasets import load_dataset
from runpatches import activatePatch, deactivatePatch
from runpatches import patch_catalog as all_patches
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model for semantic matching (force CPU)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=torch_device)


# Load top N patches
with open('patch_ranking_results.json') as f:
    patch_ranking = json.load(f)
top_patch_ids = [p['id'] for p in sorted(patch_ranking, key=lambda x: -x['score'])[:50]]
patch_catalog = [p for p in all_patches if p.get('id') in top_patch_ids]

# Activate all selected patches
for patch in patch_catalog:
    activatePatch(patch, traffic_fraction=1.0)

# Patch matching logic (copied from eval_external_datasets.py)
def autopatch_block(prompt):
    for patch in patch_catalog:
        if not patch.get('active', False):
            continue
        # Regex match
        if patch.get('type') == 'regex' and patch.get('pattern'):
            import re
            try:
                if re.search(patch['pattern'], prompt, re.IGNORECASE):
                    return True
            except Exception:
                pass
        # Substring match
        elif patch.get('type') == 'substring' and patch.get('pattern'):
            if patch['pattern'].lower() in prompt.lower():
                return True
        # Fuzzy match
        elif patch.get('type') == 'fuzzy' and patch.get('pattern'):
            try:
                from rapidfuzz import fuzz
                score = fuzz.partial_ratio(patch['pattern'].lower(), prompt.lower())
                threshold = patch.get('fuzzy_threshold', 80)
                if score >= threshold:
                    return True
            except ImportError:
                pass
        # Semantic match
        elif patch.get('type') == 'semantic' and patch.get('pattern'):
            try:
                prompt_emb = semantic_model.encode(prompt, convert_to_tensor=True, device=torch_device)
                patch_emb = patch.get('embedding')
                if patch_emb is not None:
                    patch_emb_tensor = torch.tensor(patch_emb)
                    sim = util.pytorch_cos_sim(prompt_emb, patch_emb_tensor).item()
                    threshold = patch.get('semantic_threshold', 0.7)
                    if sim >= threshold:
                        return True
            except Exception:
                pass
    return False

def get_guychuk_benign(n=500):
    print("[Eval] Loading guychuk/benign-malicious-prompt-classification (benign only)...")
    ds = load_dataset('guychuk/benign-malicious-prompt-classification', split='train')
    ds_list = [dict(ex) for ex in ds]
    benign_prompts = [ex['prompt'] for ex in ds_list if ex.get('label', 1) == 0 and ex.get('prompt')]
    print(f"[Eval] Loaded {len(benign_prompts)} benign prompts from guychuk/benign-malicious-prompt-classification.")
    # Sample about 500
    import random
    random.seed(42)
    if len(benign_prompts) > n:
        benign_prompts = random.sample(benign_prompts, n)
    return benign_prompts

def get_jailbreakbench_benign():
    DATASET = {
        'name': 'JailbreakBench',
        'hf_path': 'JailbreakBench/JBB-Behaviors',
        'config': 'behaviors',
        'split': 'benign',
        'prompt_field': 'prompt',
        'label_field': 'label',
    }
    print(f"[Eval] Loading {DATASET['name']} (benign only)...")
    ds = load_dataset(DATASET['hf_path'], DATASET['config'], split=DATASET['split'])
    ds_list = [dict(ex) for ex in ds]
    benign_prompts = [ex.get(DATASET['prompt_field']) for ex in ds_list]
    print(f"[Eval] Loaded {len(benign_prompts)} benign prompts.")
    return benign_prompts

def get_vh_benign(n=500):
    print("[Eval] Loading VH1213141516/benign_data_v1 (benign only)...")
    ds = load_dataset('VH1213141516/benign_data_v1', split='train')
    ds_list = [dict(ex) for ex in ds]
    # Try to find a reasonable prompt field
    prompt_field = None
    for k in ds_list[0].keys():
        if 'prompt' in k or 'text' in k:
            prompt_field = k
            break
    if not prompt_field:
        prompt_field = list(ds_list[0].keys())[0]
    benign_prompts = [ex.get(prompt_field) for ex in ds_list if ex.get(prompt_field)]
    print(f"[Eval] Loaded {len(benign_prompts)} benign prompts from VH1213141516/benign_data_v1.")
    # Sample about 500
    import random
    random.seed(42)
    if len(benign_prompts) > n:
        benign_prompts = random.sample(benign_prompts, n)
    return benign_prompts


# Main evaluation entry point
if __name__ == "__main__":
    from tqdm import tqdm
    # JailbreakBench
    benign_prompts = get_jailbreakbench_benign()
    # VH1213141516/benign_data_v1
    benign_prompts_vh = get_vh_benign(2000)
    # guychuk/benign-malicious-prompt-classification
    benign_prompts_guychuk = get_guychuk_benign(2000)
    # Evaluate all three
    summary_results = []
    for dataset_name, prompts in [
        ("JailbreakBench", benign_prompts),
        ("VH1213141516/benign_data_v1", benign_prompts_vh),
        ("guychuk/benign-malicious-prompt-classification", benign_prompts_guychuk)
    ]:
        print(f"\n[Eval] Running patch evaluation on {dataset_name} ({len(prompts)} prompts)...")
        # Autopatch evaluation
        # Autopatch evaluation (cache results)
        autopatch_blocked_list = []
        allowed_autopatch = 0
        blocked_autopatch = 0
        for prompt in tqdm(prompts, desc=f"Benign patch eval [Autopatch: {dataset_name}]"):
            blocked = autopatch_block(prompt)
            autopatch_blocked_list.append(blocked)
            if blocked:
                blocked_autopatch += 1
            else:
                allowed_autopatch += 1
        print(f"[RESULT] {dataset_name} - Benign prompts allowed (autopatch): {allowed_autopatch} / {len(prompts)}")
        print(f"[RESULT] {dataset_name} - Benign prompts blocked (autopatch): {blocked_autopatch} / {len(prompts)}")
        # Save per-prompt results for autopatch
        os.makedirs('deployment_history', exist_ok=True)
        outname = f"deployment_history/benign_patch_eval_results_{dataset_name.replace('/', '_')}_autopatch.jsonl"
        with open(outname, 'w') as f:
            for prompt, blocked in tqdm(zip(prompts, autopatch_blocked_list), total=len(prompts), desc=f"Saving results [Autopatch: {dataset_name}]"):
                result = {
                    'prompt': prompt,
                    'blocked': blocked
                }
                f.write(json.dumps(result) + '\n')

        # PromptGuard evaluation (cache results)
        promptguard_blocked_list = []
        allowed_promptguard = 0
        blocked_promptguard = 0
        for prompt in tqdm(prompts, desc=f"Benign patch eval [PromptGuard: {dataset_name}]"):
            blocked = promptguard_filter(prompt)
            promptguard_blocked_list.append(blocked)
            if blocked:
                blocked_promptguard += 1
            else:
                allowed_promptguard += 1
        print(f"[RESULT] {dataset_name} - Benign prompts allowed (PromptGuard): {allowed_promptguard} / {len(prompts)}")
        print(f"[RESULT] {dataset_name} - Benign prompts blocked (PromptGuard): {blocked_promptguard} / {len(prompts)}")
        # Save per-prompt results for PromptGuard
        outname_pg = f"deployment_history/benign_patch_eval_results_{dataset_name.replace('/', '_')}_promptguard.jsonl"
        with open(outname_pg, 'w') as f:
            for prompt, blocked in tqdm(zip(prompts, promptguard_blocked_list), total=len(prompts), desc=f"Saving results [PromptGuard: {dataset_name}]"):
                result = {
                    'prompt': prompt,
                    'blocked': blocked
                }
                f.write(json.dumps(result) + '\n')

        # Save summary result for this dataset (both methods)
        summary_results.append({
            'dataset': dataset_name,
            'total': len(prompts),
            'autopatch_allowed': allowed_autopatch,
            'autopatch_blocked': blocked_autopatch,
            'promptguard_allowed': allowed_promptguard,
            'promptguard_blocked': blocked_promptguard
        })
    # Write all summary results to a JSON file
    summary_outname = 'deployment_history/benign_patch_eval_summary.json'
    with open(summary_outname, 'w') as f:
        json.dump(summary_results, f, indent=2)




