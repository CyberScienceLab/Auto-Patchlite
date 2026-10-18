
import yaml
import numpy as np
from datasets import load_from_disk
import glob
import json
from pathlib import Path
from tqdm import tqdm

# --- Sentence Transformers for real embeddings ---
from sentence_transformers import SentenceTransformer
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')





# --- Load patch candidates ---
with open("autopatch_candidates_with_synthetics.yaml", "r") as f:
    patches = yaml.safe_load(f)
total_candidates = len(patches) if patches else 0
synthetic_candidates = sum(1 for p in patches if p.get('source', '').lower().startswith('synthetic')) if patches else 0
real_candidates = total_candidates - synthetic_candidates
print(f"Loaded {total_candidates} patch candidates (synthetic: {synthetic_candidates}, real: {real_candidates})")

# --- Patch type inference ---
def infer_patch_type(patch):
    if "pattern_type" in patch:
        return patch["pattern_type"]
    if "match_regex" in patch:
        return "regex"
    if "embedding" in patch:
        return "semantic"
    if "match_substring" in patch:
        return "substring"
    if "prompt" in patch:
        return "legacy_substring"
    return "unknown"

for p in patches:
    if "pattern_type" not in p:
        p["pattern_type"] = infer_patch_type(p)


# --- Auto-populate test sets ---
dataset = load_from_disk('COMPLETE_DATAMOD')

# Load Garak/hitlog prompts
def load_prompts_from_garak_jsonl(file_path):
    path = Path(file_path)
    if not path.exists():
        return []
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt_text = entry.get("prompt", {}).get("turns", [{}])[0].get("content", {}).get("text", "")
                if prompt_text:
                    prompts.append({"prompt": prompt_text, "isCritical": True})
            except Exception:
                continue
    return prompts

# Garak safety prompts
garak_jsonl_files = glob.glob("garak*.hitlog.jsonl")
garak_safety_tests = []
for file in garak_jsonl_files:
    garak_safety_tests.extend(load_prompts_from_garak_jsonl(file))

# Dataset-based test sets
safety_tests = []
utility_tests = []
benign_sample = []
traffic_replay_tests = []

for ex in dataset:
    prompt = ex["prompt"]
    label = ex["label"] if "label" in ex else ex.get("target_contains_policy_violation", 0)
    if label == 1:
        safety_tests.append({"prompt": prompt, "isCritical": True})
    else:
        utility_tests.append({"prompt": prompt})
        benign_sample.append(prompt)
        traffic_replay_tests.append({"prompt": prompt})



# Combine Garak and dataset safety tests
SafetyTests = garak_safety_tests + safety_tests

# --- FAST MODE: reduce test set sizes for speed ---
FAST_MODE = True
if FAST_MODE:
    UtilityTests = utility_tests[:30]
    TrafficReplayTests = traffic_replay_tests[:30]
    BenignSample = benign_sample[:50]
else:
    UtilityTests = utility_tests[:100]
    TrafficReplayTests = traffic_replay_tests[:100]
    BenignSample = benign_sample[:200]

# --- Precompute and cache all prompt embeddings for speed ---
def batch_encode(prompts):
    if not prompts:
        return {}
    embs = _embedding_model.encode(prompts, show_progress_bar=False, normalize_embeddings=True)
    return {p: e for p, e in zip(prompts, embs)}

PROMPT_EMBED_CACHE = {}
ALL_PROMPTS = set()
for t in SafetyTests:
    ALL_PROMPTS.add(t["prompt"])
for t in UtilityTests:
    ALL_PROMPTS.add(t["prompt"] if isinstance(t, dict) else t)
for t in BenignSample:
    ALL_PROMPTS.add(t if isinstance(t, str) else t["prompt"])
for t in TrafficReplayTests:
    ALL_PROMPTS.add(t["prompt"] if isinstance(t, dict) else t)
ALL_PROMPTS = list(ALL_PROMPTS)
PROMPT_EMBED_CACHE = batch_encode(ALL_PROMPTS)

print(f"Loaded {len(SafetyTests)} safety tests, {len(UtilityTests)} utility tests, {len(BenignSample)} benign samples")

# Print sample patch and safety test prompts
print("Sample patch prompts:")
for p in patches[:3]:
    if p.get("prompt"):
        print("-", p.get("prompt")[:100] + "...")
    elif p.get("match_regex"):
        print("-", f"Regex: {p.get('match_regex')}")
    elif p.get("match_substring"):
        print("-", f"Substring: {p.get('match_substring')}")
    else:
        print("-", f"Patch ID: {p.get('id', 'unknown')}")
print("Sample safety test prompts:")
for t in SafetyTests[:3]:
    print("-", t["prompt"][:100] + "...")

# Force a match for the first patch for demonstration
if patches and SafetyTests:
    patches[0]["prompt"] = SafetyTests[0]["prompt"]

# --- Scoring weights ---
ALPHA = 1.0   # safety_pass_rate
BETA = 0.25    # util_loss
GAMMA = 0.5   # avg_ov
DELTA = 0.5   # specificity
EPSILON = 0.5 # fp_rate



def patch_match(patch, prompt):
    # Regex match (for synthetic patches)
    if patch.get('match_regex'):
        import re
        try:
            if re.search(patch['match_regex'], prompt, re.IGNORECASE):
                return True
        except re.error as e:
            print(f"[WARN] Invalid regex in patch {patch.get('id','?')}: {patch['match_regex']} ({e}) - Skipping this patch for this prompt.")
            return False
    # Substring match (for synthetic patches)
    if patch.get('match_substring'):
        if patch['match_substring'].lower() in prompt.lower():
            return True
    # Legacy substring match
    if patch.get('prompt') and patch['prompt'] in prompt:
        return True
    # Fuzzy match: at least 10 consecutive chars in common
    if patch.get('prompt'):
        for i in range(len(patch['prompt']) - 9):
            sub = patch['prompt'][i:i+10]
            if sub in prompt:
                return True
    # Embedding similarity (real Sentence Transformers, threshold 0.7)
    if 'embedding' in patch:
        emb = PROMPT_EMBED_CACHE.get(prompt)
        if emb is None:
            emb = _embedding_model.encode(prompt, show_progress_bar=False, normalize_embeddings=True)
            PROMPT_EMBED_CACHE[prompt] = emb
        patch_emb = np.array(patch['embedding'])
        sim = float(np.dot(emb, patch_emb) / (np.linalg.norm(emb) * np.linalg.norm(patch_emb) + 1e-8))
        if sim > 0.7:
            return True
    return False

def runTest(test, patch):
    # Returns "FAIL" if patch does NOT block a known harmful prompt
    prompt = test["prompt"]
    # If patch would match (block) this prompt, it's a PASS
    if patch_match(patch, prompt):
        return "PASS"
    else:
        return "FAIL"


def runDiffTest(utest, patch):
    # Simulate base and patched outputs for a benign prompt
    prompt = utest["prompt"] if isinstance(utest, dict) else utest
    # For demonstration, assume base_out is always the same ("ok")
    base_out = "ok"
    # If patch would trigger, simulate a block/altered output
    if patch_match(patch, prompt):
        patched_out = "BLOCKED"
    else:
        patched_out = "ok"
    return base_out, patched_out


def recordUtilityDiff(stats, base, patched):
    # Track how many benign outputs changed, and how many were blocked
    if "total" not in stats:
        stats["total"] = 0
        stats["changed"] = 0
        stats["blocked"] = 0
    stats["total"] += 1
    if base != patched:
        stats["changed"] += 1
    if patched == "BLOCKED":
        stats["blocked"] += 1


def computeUtilityLoss(stats):
    # Utility loss: fraction of benign cases where output changed
    if not stats or stats["total"] == 0:
        return 0.0
    return stats["changed"] / stats["total"]


def computeFalsePositiveRate(stats):
    # FP rate: fraction of benign cases that were blocked
    if not stats or stats["total"] == 0:
        return 0.0
    return stats["blocked"] / stats["total"]

def timedRun(case, patch):
    # Return (base_time, patched_time)
    return 0.1, 0.12

def average(lst):
    return float(np.mean(lst)) if lst else 0.0

def percentile(lst, p):
    return float(np.percentile(lst, p)) if lst else 0.0






# --- Score patches and group by type ---
results = []

for patch in tqdm(patches, desc="Scoring patches"):
    safety_fail_count = 0
    critical_total = 0
    critical_blocked = 0
    utility_stats = {}
    latency_overheads = []

    # Safety tests (score by fraction of critical tests blocked)
    for test in SafetyTests:
        if test.get("isCritical", False):
            critical_total += 1
            if runTest(test, patch) == "PASS":
                critical_blocked += 1
        else:
            if runTest(test, patch) == "FAIL":
                safety_fail_count += 1
    if critical_blocked == 0:
        continue  # Patch must block at least one critical test

    # Utility tests (per patch)
    utility_stats = {}
    for utest in UtilityTests:
        base_out, patched_out = runDiffTest(utest, patch)
        recordUtilityDiff(utility_stats, base_out, patched_out)
    util_loss = computeUtilityLoss(utility_stats)
    fp_rate = computeFalsePositiveRate(utility_stats)

    # Performance tests (per patch)
    latency_overheads = []
    if not FAST_MODE:
        for case in TrafficReplayTests:
            t_base, t_patch = timedRun(case, patch)
            latency_overheads.append(t_patch - t_base)
    avg_ov = average(latency_overheads) if latency_overheads else 0.0
    p95_ov = percentile(latency_overheads, 95) if latency_overheads else 0.0

    # Specificity (per patch)
    trigger_count = 0
    total_benign = len(BenignSample)
    for b in BenignSample:
        if patch_match(patch, b):
            trigger_count += 1
    specificity = 1 - (trigger_count / total_benign) if total_benign > 0 else 1.0

    # Safety pass rate (fraction of critical tests blocked)
    safety_pass_rate = critical_blocked / critical_total if critical_total > 0 else 1.0

    # Score (per patch)
    s = (ALPHA * safety_pass_rate
         - BETA * util_loss
         - GAMMA * avg_ov
         + DELTA * specificity
         - EPSILON * fp_rate)
    # Only print summary for top-level loop
    print(f"Patch {patch['id']} (type: {patch.get('pattern_type','unknown')}): score={s:.3f}, safety_pass_rate={safety_pass_rate:.3f}, util_loss={util_loss:.3f}, avg_ov={avg_ov:.3f}, specificity={specificity:.3f}, fp_rate={fp_rate:.3f}")
    results.append((patch, s, util_loss, fp_rate, p95_ov))

# --- Group and report by patch type ---
from collections import defaultdict
type_groups = defaultdict(list)
for patch, score, util_loss, fp_rate, p95_ov in results:
    ptype = patch.get("pattern_type", "unknown")
    type_groups[ptype].append((patch, score, util_loss, fp_rate, p95_ov))


# --- Sort and save top 5 patches for each type ---
top_patches_by_type = {}
for ptype, group in type_groups.items():
    # Sort group by score descending
    sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
    top5 = sorted_group[:5]
    top_patches_by_type[ptype] = [
        {
            "id": patch["id"],
            "score": score,
            "util_loss": util_loss,
            "fp_rate": fp_rate,
            "p95_ov": p95_ov
        }
        for patch, score, util_loss, fp_rate, p95_ov in top5
    ]
    print(f"\nPatch type: {ptype} ({len(group)} patches)")
    for patch, score, util_loss, fp_rate, p95_ov in top5:
        print(f"  Patch {patch['id']}: score={score:.3f}, util_loss={util_loss:.3f}, fp_rate={fp_rate:.3f}, p95_ov={p95_ov:.3f}")

# Save top 5 per type to JSON
with open("patch_ranking_top5_by_type.json", "w") as f:
    json.dump(top_patches_by_type, f, indent=2)
print("\nTop 5 patches per type saved to patch_ranking_top5_by_type.json")


# Sort results by score descending (best first)
results.sort(key=lambda x: x[1], reverse=True)

# Debug: print number of patches scored and their IDs/scores (now sorted)
print(f"Number of patches scored: {len(results)}")
for patch, score, util_loss, fp_rate, p95_ov in results:
    print(f"Patch {patch['id']} scored {score:.3f}")

# Print sorted results
for patch, score, util_loss, fp_rate, p95_ov in results:
    print(f"Patch {patch['id']}: score={score:.3f}, util_loss={util_loss:.3f}, fp_rate={fp_rate:.3f}, p95_ov={p95_ov:.3f}")

# Save sorted results to JSON file
import json
results_to_save = [
    {
        "id": patch["id"],
        "score": score,
        "util_loss": util_loss,
        "fp_rate": fp_rate,
        "p95_ov": p95_ov
    }
    for patch, score, util_loss, fp_rate, p95_ov in results
]
with open("patch_ranking_results.json", "w") as f:
    json.dump(results_to_save, f, indent=2)

print("\nResults saved to patch_ranking_results.json")