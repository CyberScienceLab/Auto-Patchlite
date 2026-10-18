import yaml
import uuid
from datetime import datetime
import json
import re
from typing import List, Dict, Any
import math

def estimate_confidence(pattern: str, pattern_type: str) -> float:
    """Estimate confidence score locally based on pattern features."""
    if not pattern:
        return 0.0

    base = 0.6

    # Regex: longer & more specific patterns → higher confidence
    if pattern_type == "regex":
        length_factor = min(len(pattern) / 40, 1.0)
        complexity = len(re.findall(r'[\[\]\(\)\?\*\+\|]', pattern)) / 5
        return round(min(1.0, base + 0.2 * length_factor + 0.2 * complexity), 3)

    # Keyword: rare or multiword phrases → higher confidence
    if pattern_type == "substring":
        word_count = len(pattern.split())
        rarity_bonus = 0.15 if re.search(r'ignore|override|jailbreak|system|bypass', pattern, re.I) else 0.0
        return round(min(1.0, base + 0.1 * word_count + rarity_bonus), 3)

    # Semantic: use embedding norm as a weak proxy for information density
    if pattern_type == "semantic":
        emb = compute_embedding(pattern)
        if emb is None:
            return 0.7
        norm = math.sqrt(sum(e * e for e in emb)) / len(emb)
        return round(min(1.0, base + 0.3 * norm), 3)

    return base

# =============================
# --- Embedding utility ---
try:
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _embedding_model = None


def compute_embedding(text):
    if _embedding_model is None:
        return None
    emb = _embedding_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return emb.tolist() if hasattr(emb, 'tolist') else emb


# =============================
# --- FailureExemplar ---
class FailureExemplar:
    """Represents a failure case that needs patching"""
    def __init__(self, prompt: str, context: dict, violation_type: str, source: str):
        self.prompt = prompt
        self.context = context
        self.violation_type = violation_type
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        self.source = source

    def description(self) -> str:
        return f"Prompt: {self.prompt[:100]}... | Type: {self.violation_type} | Source: {self.source}"


# =============================
# --- Synthetic Patch Generator ---
class SyntheticPatchGenerator:
    def __init__(self, model_name: str = 'mistral'):
        self.model_name = model_name

    def _build_pattern_mining_prompt(self, failures: List[FailureExemplar]) -> str:
        """Constructs a consistent mining prompt for jailbreak analysis."""
        prompt = (
            "You are a security expert analyzing prompt injection attacks. "
            "The following are examples of malicious prompts that successfully bypassed safety measures:\n\n"
        )
        for i, failure in enumerate(failures[:5]):
            prompt += f"{i+1}. {failure.description()}\n"

        prompt += (
            "\nAnalyze these examples and identify common patterns that could be used to detect similar attacks.\n"
            "Focus on:\n"
            "1. Common keywords or phrases\n"
            "2. Structural patterns (e.g., 'ignore X and do Y')\n"
            "3. Semantic themes (e.g., attempts to override instructions)\n\n"
            "Respond in **valid JSON** with these fields:\n"
            "{\n"
            "  \"regex_patterns\": [\"list of regex patterns that would catch these\"],\n"
            "  \"keyword_patterns\": [\"list of key phrases to look for\"],\n"
            "  \"structural_patterns\": [\"patterns in how the attacks are structured\"],\n"
            "  \"semantic_description\": \"one-sentence summary of this jailbreak style\"\n"
            "}\n\n"
            "Be specific and actionable — these patterns will be used to generate security patches."
        )
        return prompt

    def _extract_simple_patterns(self, failures: List[FailureExemplar]) -> List[Dict[str, Any]]:
        """Fallback basic pattern extractor if DSPy or model inference fails."""
        patterns = []
        injection_patterns = [
            r'(?i)(ignore|forget|disregard).*(instruction|rule|filter)',
            r'(?i)(override|bypass).*(system|safety|guard)',
            r'(?i)(new\s+)?(instruction|prompt|command)',
            r'(?i)(pretend|act\s+as|roleplay)',
            r'(?i)(jailbreak|hack|exploit)',
        ]
        for pat in injection_patterns:
            patterns.append({'type': 'regex', 'pattern': pat, 'confidence': estimate_confidence(pat, 'regex')})
        return patterns


PATCH_YAML_PATH = "autopatch_candidates.yaml"


def load_patches():
    try:
        with open(PATCH_YAML_PATH, 'r') as f:
            return yaml.safe_load(f) or []
    except Exception:
        return []


# =============================
# --- Synthetic Patch Generation ---
def add_synthetic_cases():
    import numpy as np
    import os
    import onnxruntime as ort
    from tqdm import tqdm
    import dspy

    patches = load_patches()
    print(f"[DEBUG] Current working directory: {os.getcwd()}")

    # --- ONNX RL filtering ---
    ONNX_MODEL_PATH = "DEMO.onnx"
    OBS_LENGTH = 393
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH)
    except Exception:
        print(f"[ERROR] Could not load ONNX model at {ONNX_MODEL_PATH}")
        return

    def get_onnx_action(prompt):
        emb = compute_embedding(prompt)
        if emb is None:
            return 0
        if len(emb) < OBS_LENGTH:
            emb = np.concatenate([emb, np.zeros(OBS_LENGTH - len(emb))])
        elif len(emb) > OBS_LENGTH:
            emb = emb[:OBS_LENGTH]
        emb = np.array(emb, dtype=np.float32).reshape(1, -1)
        ort_inputs = {session.get_inputs()[0].name: emb}
        ort_outs = session.run(None, ort_inputs)
        return int(np.argmax(ort_outs[0]))

    # --- Load Garak prompts ---
    garak_jsonl_files = [f for f in os.listdir('.') if f.startswith('garak') and f.endswith('.hitlog.jsonl')]
    garak_prompts = set()
    for file in garak_jsonl_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        prompt_text = entry.get("prompt", {}).get("turns", [{}])[0].get("content", {}).get("text", "")
                        if prompt_text:
                            garak_prompts.add(prompt_text)
                    except Exception:
                        continue
        except Exception:
            continue

    eligible_prompts = list(garak_prompts)
    print(f"[INFO] Number of eligible prompts: {len(eligible_prompts)}")

    # --- Initialize DSPy properly ---
    print("[INFO] Initializing DSPy for structured patch generation...")

    class PatchPatternSignature(dspy.Signature):
        """Extracts structured JSON patch data for a single jailbreak prompt."""
        prompt = dspy.InputField(desc="Prompt to analyze for jailbreak patterns")
        regex_pattern = dspy.OutputField(desc="Regex pattern for this jailbreak")
        keyword_pattern = dspy.OutputField(desc="Keyword/substring pattern for this jailbreak")
        semantic_comment = dspy.OutputField(desc="One-sentence semantic summary")


    llm = dspy.LM('ollama/mistral')
    dspy.configure(lm=llm)
    extractor = dspy.Predict(PatchPatternSignature)

    print("[INFO] DSPy extractor ready. Generating structured patches...")

    autopatch_copy = patches[:]
    added = 0
    synth_gen = SyntheticPatchGenerator()

    for prompt_text in tqdm(eligible_prompts):
        failure = FailureExemplar(
            prompt=prompt_text,
            context={'source': 'garak'},
            violation_type='jailbreak',
            source='garak'
        )

        # Use unified mining prompt
        mining_prompt = synth_gen._build_pattern_mining_prompt([failure])

        try:
            result = extractor(prompt=mining_prompt)
            #result_data = result.dict()
        except Exception as e:
            print(f"[WARN] DSPy extraction failed: {e}")
            continue

        regex_pattern = getattr(result, "regex_pattern", "").strip()
        keyword_pattern = getattr(result, "keyword_pattern", "").strip()
        semantic_comment = getattr(result, "semantic_comment", "").strip()

        patch_candidates = []

        if regex_pattern:
            patch_candidates.append({
                'id': str(uuid.uuid4())[:8],
                'created': datetime.utcnow().isoformat() + 'Z',
                'source': 'synthetic',
                'pattern_type': 'regex',
                'match_regex': regex_pattern,
                'action': 'deny',
                'type': 'regex',
                'pattern': regex_pattern,
                'confidence': estimate_confidence(regex_pattern, 'regex')
            })

        if keyword_pattern:
            patch_candidates.append({
                'id': str(uuid.uuid4())[:8],
                'created': datetime.utcnow().isoformat() + 'Z',
                'source': 'synthetic',
                'pattern_type': 'substring',
                'match_substring': keyword_pattern,
                'action': 'deny',
                'type': 'substring',
                'pattern': keyword_pattern,
                'confidence': estimate_confidence(keyword_pattern, 'substring')
            })

        if semantic_comment:
            emb = compute_embedding(semantic_comment)
            if emb is not None:
                patch_candidates.append({
                    'id': str(uuid.uuid4())[:8],
                    'created': datetime.utcnow().isoformat() + 'Z',
                    'source': 'synthetic',
                    'pattern_type': 'semantic',
                    'action': 'deny',
                    'type': 'semantic',
                    'pattern': semantic_comment,
                    'embedding': emb,
                    'confidence': estimate_confidence(semantic_comment, 'semantic')
                })

        # --- RL filtering ---
        for patch in patch_candidates:
            test_prompt = patch.get('pattern')
            if not test_prompt:
                continue
            action = get_onnx_action(test_prompt)
            if action == 1:
                print(f"[SKIP] RL filtered patch: {test_prompt[:60]}...")
                continue

            autopatch_copy.append(patch)
            added += 1

    newfile = "autopatch_candidates_with_synthetics.yaml"
    with open(newfile, 'w') as f:
        yaml.safe_dump(autopatch_copy, f, sort_keys=False)

    num_synth = sum(1 for p in autopatch_copy if p.get('source') == 'synthetic')
    num_orig = sum(1 for p in autopatch_copy if p.get('source') != 'synthetic')

    stats = (
        f"Total patches: {len(autopatch_copy)}\n"
        f"Synthetic patches: {num_synth}\n"
        f"Original patches: {num_orig}\n"
    )
    with open("stats.txt", "w") as statsf:
        statsf.write(stats)

    print(f"[DONE] Added {added} new DSPy-based synthetic patches.")
    print(f"[INFO] Patch stats written to stats.txt")


# =============================
if __name__ == "__main__":
    add_synthetic_cases()
