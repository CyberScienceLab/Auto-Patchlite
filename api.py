from fastapi import FastAPI
import yaml, json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend dev server to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/top-patches")
def get_top_patches():
    with open("patch_ranking_results.json") as f:
        ranking = json.load(f)
    with open("autopatch_candidates.yaml") as f:
        candidates = yaml.safe_load(f)
    candidate_map = {p["id"]: p for p in candidates}
    top_patches = []
    for patch in ranking[:6]:
        details = candidate_map.get(patch["id"], {})
        merged = {**details, **patch}
        top_patches.append(merged)
    print("Top 6 patches returned by /api/top-patches:")
    for p in top_patches:
        print(f"ID: {p.get('id')}, Score: {p.get('score')}, Prompt: {p.get('prompt', '')[:40]}")
    return top_patches

@app.get("/api/live-metrics")
def get_live_metrics():
    try:
        with open("live_metrics.json") as f:
            metrics = json.load(f)
        return metrics
    except Exception:
        # Fallback: return zeros
        return {
            "safety_pass_rate": 0.0,
            "util_loss": 0.0,
            "fp_rate": 0.0,
            "latency": 0.0
        }
