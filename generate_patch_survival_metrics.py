import os
import csv
import random
import string
from runpatches import patch_catalog
from sentence_transformers import SentenceTransformer, util

# Top patch IDs
PATCH_IDS = [
    "e797d27a",
    "3801858c",
    "0cd819e5",
    "e61ba3b3",
]

# Number of variants to generate per patch
N_VARIANTS = 50

# Load semantic model for semantic patches
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def mutate_substring(s):
    # Simple mutations: case, insert, delete, swap
    variants = set()
    variants.add(s)
    variants.add(s.upper())
    variants.add(s.lower())
    # Insert random char
    for _ in range(5):
        idx = random.randint(0, len(s))
        c = random.choice(string.ascii_letters)
        variants.add(s[:idx] + c + s[idx:])
    # Delete random char
    for _ in range(5):
        if len(s) > 1:
            idx = random.randint(0, len(s)-1)
            variants.add(s[:idx] + s[idx+1:])
    # Swap two chars
    for _ in range(5):
        if len(s) > 1:
            idx = random.randint(0, len(s)-2)
            swapped = list(s)
            swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
            variants.add(''.join(swapped))
    return list(variants)[:N_VARIANTS]

def mutate_regex(pattern):
    # Generate variants by inserting, deleting, or swapping chars in the pattern string
    # (Not true regex fuzzing, but a simple proxy)
    return mutate_substring(pattern)

def mutate_semantic(text):
    # For semantic, generate paraphrases by simple word swaps and case changes
    words = text.split()
    variants = set()
    variants.add(text)
    variants.add(text.upper())
    variants.add(text.lower())
    # Swap two words
    for _ in range(5):
        if len(words) > 1:
            idx = random.randint(0, len(words)-2)
            swapped = words[:]
            swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
            variants.add(' '.join(swapped))
    # Insert random word
    for _ in range(5):
        idx = random.randint(0, len(words))
        w = random.choice(["foo", "bar", "baz", "qux"])
        new = words[:idx] + [w] + words[idx:]
        variants.add(' '.join(new))
    # Delete random word
    for _ in range(5):
        if len(words) > 1:
            idx = random.randint(0, len(words)-1)
            new = words[:idx] + words[idx+1:]
            variants.add(' '.join(new))
    return list(variants)[:N_VARIANTS]

def patch_blocks(patch, prompt):
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

def generate_variants(patch):
    if patch.get('type') == 'regex':
        return mutate_regex(patch.get('pattern', ''))
    elif patch.get('type') == 'substring':
        return mutate_substring(patch.get('pattern', ''))
    elif patch.get('type') == 'semantic':
        # Use pattern or prompt field for semantic
        return mutate_semantic(patch.get('pattern', patch.get('prompt', '')))
    else:
        return []

def main():
    rows = []
    patch_dict = {p.get('id'): p for p in patch_catalog}
    survival_counts = []
    for patch_id in PATCH_IDS:
        patch = patch_dict.get(patch_id)
        if not patch:
            print(f"[WARN] Patch {patch_id} not found.")
            continue
        variants = generate_variants(patch)
        blocked = 0
        for v in variants:
            if patch_blocks(patch, v):
                blocked += 1
            else:
                break  # Patch is bypassed
        rows.append({
            "Patch ID": patch_id,
            "Patch Type": patch.get('type', 'unknown'),
            "Survival Count": blocked,
            "Bypassed": "Yes" if blocked < len(variants) else "No"
        })
        survival_counts.append(blocked)
        print(f"[INFO] Patch {patch_id}: Survival={blocked}, Bypassed={'Yes' if blocked < len(variants) else 'No'}")
    # Summary statistics
    mean_survival = sum(survival_counts)/len(survival_counts) if survival_counts else 0
    min_survival = min(survival_counts) if survival_counts else 0
    max_survival = max(survival_counts) if survival_counts else 0
    print(f"[SUMMARY] Mean={mean_survival}, Min={min_survival}, Max={max_survival}")
    # Save table
    os.makedirs("table_results", exist_ok=True)
    with open("table_results/patch_survival_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Patch ID", "Patch Type", "Survival Count", "Bypassed"])
        writer.writeheader()
        writer.writerows(rows)
    print("Patch survival metrics table written to table_results/patch_survival_metrics.csv")

if __name__ == "__main__":
    main()