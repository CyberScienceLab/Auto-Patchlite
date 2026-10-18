import os
import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from runpatches import patch_catalog as all_patches, activatePatch, deactivatePatch

# Load sentence transformer model for semantic matching (global, so only loaded once)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load top prompts for benchmarking (use a sample from JailbreakBench or a local file)
PROMPT_SAMPLE_FILE = 'benchmark_prompts.jsonl'  # Each line: {"prompt": ...}
if os.path.exists(PROMPT_SAMPLE_FILE):
    with open(PROMPT_SAMPLE_FILE) as f:
        prompts = [json.loads(line)['prompt'] for line in f][:100]
else:
    # Fallback: use dummy prompts
    prompts = [f"Test prompt {i}" for i in range(100)]

# Patch sizes to benchmark
PATCH_COUNTS = [10, 25, 50, 100, 200]

# Load patch ranking
with open('patch_ranking_results.json') as f:
    patch_ranking = json.load(f)

# Patch matching logic (same as autopatch_block)
def autopatch_block(prompt, patch_catalog):
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

results = []
for N in PATCH_COUNTS:
    top_patch_ids = [p['id'] for p in sorted(patch_ranking, key=lambda x: -x['score'])[:N]]
    patch_catalog = [p for p in all_patches if p.get('id') in top_patch_ids]
    # Activate only these patches
    for patch in all_patches:
        deactivatePatch(patch)
    for patch in patch_catalog:
        activatePatch(patch, traffic_fraction=1.0)
    # Measure latency
    latencies = []
    for prompt in tqdm(prompts, desc=f"Patches={N}"):
        t0 = time.time()
        autopatch_block(prompt, patch_catalog)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
    latencies_np = latencies
    result = {
        'num_patches': N,
        'mean_latency_ms': sum(latencies_np)/len(latencies_np),
        'median_latency_ms': sorted(latencies_np)[len(latencies_np)//2],
        'max_latency_ms': max(latencies_np),
        'min_latency_ms': min(latencies_np),
        'std_latency_ms': (sum((x - sum(latencies_np)/len(latencies_np))**2 for x in latencies_np)/len(latencies_np))**0.5,
    }
    print(f"[Latency] Patches={N}: Mean={result['mean_latency_ms']:.2f} ms, Median={result['median_latency_ms']:.2f} ms, Max={result['max_latency_ms']:.2f} ms")
    results.append(result)

# Save results
with open('patch_latency_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print("[Done] Latency benchmark results saved to patch_latency_benchmark.json")


import csv

# Compute baseline values
baseline = results[0]
baseline_median = baseline["median_latency_ms"]
baseline_p95 = baseline["max_latency_ms"]

# Save results as CSV with overhead columns
csv_path = 'patch_latency_benchmark.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "num_patches",
        "mean_latency_ms",
        "median_latency_ms",
        "max_latency_ms",
        "min_latency_ms",
        "std_latency_ms",
        "median_overhead_percent",
        "p95_overhead_percent"
    ])
    for r in results:
        N = r["num_patches"]
        median = r["median_latency_ms"]
        p95 = r["max_latency_ms"]
        median_overhead = ((median - baseline_median) / baseline_median) * 100 if baseline_median > 0 else 0
        p95_overhead = ((p95 - baseline_p95) / baseline_p95) * 100 if baseline_p95 > 0 else 0
        writer.writerow([
            N,
            r["mean_latency_ms"],
            median,
            p95,
            r["min_latency_ms"],
            r["std_latency_ms"],
            median_overhead,
            p95_overhead
        ])
print(f"[Done] CSV results saved to {csv_path}")
